"""Main window cho HALCON Vision Studio (VisionPro-style sidebar)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.operators import (
    HALCON_AVAILABLE,
    HALCON_VERSION,
    Grabber,
    adaptive_threshold,
    apply_filter,
    apply_mask,
    color_stats,
    contour_analysis,
    decode_codes,
    edges_sub_pix,
    histogram,
    image_diff,
    mask_from_gray_range,
    mask_from_roi,
    measure_pairs,
    morphology,
    read_image,
    shape_match,
    threshold_blob,
)
from app.widgets import ImageCanvas, OperatorSidebar, ResultsView

SUPPORTED_EXTS = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HALCON Vision Studio")
        self.resize(1480, 920)
        self.setMinimumSize(QSize(1180, 720))

        self._image_path: Optional[Path] = None
        self._original_image: Optional[np.ndarray] = None
        self._template_path: Optional[Path] = None
        self._template_image: Optional[np.ndarray] = None
        self._reference_image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

        self._grabber = Grabber()
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._on_live_tick)

        # ROI mode state — phân biệt mục đích lấy ROI: "template" / "color" / "mask"
        self._roi_purpose: Optional[str] = None

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        h_split = QSplitter(Qt.Horizontal, central)

        # ---- LEFT: sidebar (collapsible tools) ----
        self.sidebar = OperatorSidebar()
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(440)
        h_split.addWidget(self.sidebar)

        # ---- CENTER: canvas (large) + collapsible results panel ----
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(6)

        # Header row above viewer
        title_row = QHBoxLayout()
        title = QLabel("Image Viewer")
        title.setProperty("heading", True)
        self.viewer_badge = QLabel("idle")
        self.viewer_badge.setProperty("badge", True)
        self.mask_badge = QLabel("no mask")
        self.mask_badge.setProperty("badge", True)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.mask_badge)
        title_row.addWidget(self.viewer_badge)
        center_lay.addLayout(title_row)

        # Vertical splitter — canvas chiếm phần lớn, results nhỏ ở dưới
        v_split = QSplitter(Qt.Vertical)
        v_split.setChildrenCollapsible(True)
        self._v_split = v_split

        self.canvas = ImageCanvas()
        v_split.addWidget(self.canvas)

        results_holder = QWidget()
        r_lay = QVBoxLayout(results_holder)
        r_lay.setContentsMargins(0, 4, 0, 0)
        r_lay.setSpacing(6)
        r_header = QHBoxLayout()
        r_title = QLabel("Results")
        r_title.setProperty("heading", True)
        self.results_toggle_btn = QPushButton("▾  Hide")
        self.results_toggle_btn.setProperty("ghost", True)
        self.results_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.results_toggle_btn.clicked.connect(self._toggle_results_panel)
        r_header.addWidget(r_title)
        r_header.addStretch()
        r_header.addWidget(self.results_toggle_btn)
        r_lay.addLayout(r_header)
        self.results_view = ResultsView()
        r_lay.addWidget(self.results_view, 1)
        v_split.addWidget(results_holder)

        # Tỷ lệ ban đầu: canvas ≈ 78%, results ≈ 22%
        v_split.setStretchFactor(0, 5)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([800, 220])
        center_lay.addWidget(v_split, 1)

        h_split.addWidget(center)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)

        root.addWidget(h_split)
        self.setCentralWidget(central)

    def _toggle_results_panel(self):
        sizes = self._v_split.sizes()
        if sizes[1] > 8:
            self._results_prev_size = sizes[1]
            self._v_split.setSizes([sizes[0] + sizes[1], 0])
            self.results_toggle_btn.setText("▴  Show")
        else:
            prev = getattr(self, "_results_prev_size", 220)
            total = sizes[0]
            self._v_split.setSizes([max(100, total - prev), prev])
            self.results_toggle_btn.setText("▾  Hide")

    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("&File")
        act_open = QAction("Open Image…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open_image)
        m_file.addAction(act_open)
        act_template = QAction("Open Template…", self)
        act_template.triggered.connect(self._on_open_template)
        m_file.addAction(act_template)
        m_file.addSeparator()
        act_save = QAction("Save Result Image…", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save_result)
        m_file.addSeparator()
        m_file.addAction(act_save)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_view = bar.addMenu("&View")
        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self.canvas.fit_to_view)
        m_view.addAction(act_fit)
        act_reset = QAction("Reset Zoom", self)
        act_reset.setShortcut("Ctrl+1")
        act_reset.triggered.connect(self.canvas.reset_zoom)
        m_view.addAction(act_reset)
        m_view.addSeparator()
        act_expand = QAction("Expand all sections", self)
        act_expand.triggered.connect(self.sidebar.expand_all)
        m_view.addAction(act_expand)
        act_collapse = QAction("Collapse all sections", self)
        act_collapse.triggered.connect(self.sidebar.collapse_all)
        m_view.addAction(act_collapse)

        m_help = bar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        m_help.addAction(act_about)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(QAction("📂  Open", self, triggered=self._on_open_image))
        tb.addAction(QAction("🧩  Template", self, triggered=self._on_open_template))
        tb.addSeparator()
        tb.addAction(QAction("↺  Reset Image", self, triggered=self._on_reset_image))
        tb.addAction(QAction("🔍  Fit", self, triggered=self.canvas.fit_to_view))
        tb.addAction(QAction("1:1  Reset Zoom", self, triggered=self.canvas.reset_zoom))
        tb.addSeparator()
        tb.addAction(QAction("◧  Expand sidebar", self, triggered=self.sidebar.expand_all))
        tb.addAction(QAction("◨  Collapse sidebar", self, triggered=self.sidebar.collapse_all))
        tb.addSeparator()
        tb.addAction(QAction("💾  Save Result", self, triggered=self._on_save_result))

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        engine = (
            f"HALCON {HALCON_VERSION}  ●" if HALCON_AVAILABLE
            else "Engine: OpenCV fallback  ●"
        )
        self._engine_label = QLabel(engine)
        self._image_label = QLabel("No image loaded")
        self._cursor_label = QLabel("")
        sb.addPermanentWidget(self._image_label, 2)
        sb.addPermanentWidget(self._cursor_label, 2)
        sb.addPermanentWidget(self._engine_label, 1)

    def _wire_signals(self):
        s = self.sidebar
        # Tools
        s.filter_run.connect(self._run_filter)
        s.morphology_run.connect(self._run_morphology)
        s.threshold_run.connect(self._run_threshold)
        s.adaptive_run.connect(self._run_adaptive)
        s.edges_run.connect(self._run_edges)
        s.contour_run.connect(self._run_contours)
        s.shape_match_run.connect(self._run_shape_match)
        s.shape_template_load.connect(self._on_open_template)
        s.shape_template_save.connect(self._on_save_template)
        s.shape_template_clear.connect(self._on_clear_template)
        s.shape_pick_roi_toggled.connect(self._on_pick_template_roi)
        s.measure_run.connect(self._run_measure)
        s.measure_mode_toggled.connect(self.canvas.set_measure_mode)
        s.histogram_run.connect(self._run_histogram)
        s.idread_run.connect(self._run_idread)
        s.color_pick_roi_toggled.connect(self._on_pick_color_roi)
        s.color_run.connect(self._run_color)
        s.diff_run.connect(self._run_diff)
        s.diff_load_reference.connect(self._on_load_reference)
        s.diff_clear_reference.connect(self._on_clear_reference)

        # Mask
        s.mask_gen_gray.connect(self._on_mask_from_gray)
        s.mask_gen_hsv.connect(self._on_mask_from_hsv)
        s.mask_pick_roi_toggled.connect(self._on_pick_mask_roi)
        s.mask_invert.connect(self._on_mask_invert)
        s.mask_clear.connect(self._on_mask_clear)
        s.mask_show_toggled.connect(self.canvas.set_show_mask)
        s.mask_save.connect(self._on_mask_save)
        s.mask_load.connect(self._on_mask_load)

        # Acquisition
        s.acq_connect.connect(self._on_acq_connect)
        s.acq_disconnect.connect(self._on_acq_disconnect)
        s.acq_live.connect(self._on_live_toggled)
        s.acq_snapshot.connect(self._on_snapshot)
        s.acq_fps.connect(self._on_fps_changed)

        # Canvas
        self.canvas.measure_segment_drawn.connect(self.sidebar.set_measure_segment)
        self.canvas.roi_drawn.connect(self._on_roi_drawn)
        self.canvas.mouse_moved.connect(self._on_cursor_moved)

    # ==================================================================
    # File / template
    # ==================================================================
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", str(Path.home()), SUPPORTED_EXTS)
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được ảnh:\n{exc}")
            return
        self._image_path = Path(path)
        self._original_image = img
        self.canvas.set_image(img)
        self._image_label.setText(f"{self._image_path.name}  •  {img.shape[1]}×{img.shape[0]}")
        self.viewer_badge.setText("file")
        self.results_view.append_log(f"Loaded image: {self._image_path.name}")

    def _on_open_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Template", str(Path.home()), SUPPORTED_EXTS)
        if not path:
            return
        try:
            tpl = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được template:\n{exc}")
            return
        self._set_template(tpl, name=Path(path).name, source="file")

    def _set_template(self, img: np.ndarray, name: str, source: str):
        self._template_image = img
        self._template_path = Path(name) if source == "file" else None
        self.sidebar.set_template_name(f"{name}  •  {img.shape[1]}×{img.shape[0]}")
        self.sidebar.set_template_preview(img)
        self.results_view.append_log(f"Template ← {source}: {name} ({img.shape[1]}×{img.shape[0]})")

    def _on_save_template(self):
        if self._template_image is None:
            QMessageBox.information(self, "Template", "Chưa có template để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", str(Path.home() / "template.png"), SUPPORTED_EXTS)
        if path and cv2.imwrite(path, self._template_image):
            self.results_view.append_log(f"Saved template → {path}")

    def _on_clear_template(self):
        self._template_image = None
        self._template_path = None
        self.sidebar.set_template_name(None)
        self.sidebar.set_template_preview(None)
        self.results_view.append_log("Template cleared.")

    def _on_reset_image(self):
        if self._original_image is not None:
            self.canvas.set_image(self._original_image)

    def _on_save_result(self):
        img = self.canvas.current_image()
        if img is None:
            QMessageBox.information(self, "Save", "Chưa có ảnh để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", str(Path.home() / "result.png"), SUPPORTED_EXTS)
        if path and cv2.imwrite(path, img):
            self.results_view.append_log(f"Saved → {path}")

    def _on_about(self):
        msg = (
            "<b>HALCON Vision Studio</b><br>"
            "PySide6 GUI inspired by Cognex VisionPro, wrapping MVTec HALCON.<br><br>"
            f"Engine: <b>{'HALCON ' + HALCON_VERSION if HALCON_AVAILABLE else 'OpenCV fallback'}</b>"
        )
        QMessageBox.about(self, "About", msg)

    def _on_cursor_moved(self, row: int, col: int, value):
        self._cursor_label.setText(f"({col}, {row})  px={value}")

    # ==================================================================
    # ROI handling — distinguish purpose
    # ==================================================================
    def _on_pick_template_roi(self, on: bool):
        self._roi_purpose = "template" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: kéo chuột để chọn ROI làm template…")
            self.sidebar.focus_match()
        else:
            self.canvas.clear_roi()

    def _on_pick_color_roi(self, on: bool):
        self._roi_purpose = "color" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: kéo chuột để chọn ROI cho color stats…")
        else:
            self.canvas.clear_roi()

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int):
        if self._original_image is None or w < 5 or h < 5:
            return
        if self._roi_purpose == "template":
            crop = self._original_image[y : y + h, x : x + w].copy()
            self._set_template(crop, name=f"ROI({x},{y},{w}×{h})", source="ROI")
            self.sidebar.reset_pick_button()
        elif self._roi_purpose == "color":
            self.sidebar.set_color_roi(x, y, w, h)
        elif self._roi_purpose == "mask":
            mask = mask_from_roi(self._original_image, x, y, w, h)
            self._set_mask(mask, source=f"ROI({x},{y},{w}×{h})")
            self.sidebar.reset_mask_pick()
        self._roi_purpose = None
        self.canvas.set_roi_mode(False)
        self.canvas.clear_roi()

    def _on_pick_mask_roi(self, on: bool):
        self._roi_purpose = "mask" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: kéo chuột chọn vùng làm mask…")
            self.sidebar.mask_section.set_expanded(True)
        else:
            self.canvas.clear_roi()

    # ==================================================================
    # Operator runs
    # ==================================================================
    def _ensure_image(self) -> bool:
        if self._original_image is None:
            QMessageBox.information(self, "No Image", "Hãy mở ảnh / acquire frame trước.")
            return False
        return True

    def _input_image(self) -> np.ndarray:
        """Ảnh đầu vào cho operator: áp mask nếu có."""
        if self._mask is not None:
            return apply_mask(self._original_image, self._mask)
        return self._original_image

    def _run_filter(self, params: dict):
        if not self._ensure_image():
            return
        result = apply_filter(self._input_image(), **params)
        self._original_image = result.image  # filter ghi đè ảnh nguồn để xếp chuỗi tool
        self._show_result(result, title="Filter")

    def _run_morphology(self, params: dict):
        if not self._ensure_image():
            return
        result = morphology(self._input_image(), **params)
        self._original_image = result.image
        self._show_result(result, title="Morphology")

    def _run_threshold(self, params: dict):
        if not self._ensure_image():
            return
        self._show_result(threshold_blob(self._input_image(), **params), title="Blob")

    def _run_adaptive(self, params: dict):
        if not self._ensure_image():
            return
        self._show_result(adaptive_threshold(self._input_image(), **params), title="Adaptive Threshold")

    def _run_edges(self, params: dict):
        if not self._ensure_image():
            return
        self._show_result(edges_sub_pix(self._input_image(), **params), title="Edges")

    def _run_contours(self, params: dict):
        if not self._ensure_image():
            return
        self._show_result(contour_analysis(self._input_image(), **params), title="Contours")

    def _run_shape_match(self, params: dict):
        if not self._ensure_image():
            return
        if self._template_image is None:
            QMessageBox.information(self, "Template", "Hãy chọn template trước (Pick ROI hoặc Load file).")
            return
        self._show_result(shape_match(self._input_image(), self._template_image, **params), title="Pattern Match")

    def _run_measure(self, params: dict):
        if not self._ensure_image():
            return
        self._show_result(measure_pairs(self._input_image(), **params), title="Caliper")

    def _run_histogram(self):
        if not self._ensure_image():
            return
        self._show_result(histogram(self._input_image()), title="Histogram")

    def _run_idread(self):
        if not self._ensure_image():
            return
        self._show_result(decode_codes(self._input_image()), title="ID Read")

    def _run_color(self):
        if not self._ensure_image():
            return
        x, y, w, h = self.sidebar.color_content.roi
        self._show_result(color_stats(self._original_image, x, y, w, h), title="Color")

    def _run_diff(self, params: dict):
        if not self._ensure_image():
            return
        if self._reference_image is None:
            QMessageBox.information(self, "Reference", "Hãy load reference image (golden) trước.")
            return
        self._show_result(image_diff(self._original_image, self._reference_image, **params), title="Image Diff")

    # ------------------------------------------------------------------
    # Reference (golden) image
    # ------------------------------------------------------------------
    def _on_load_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference", str(Path.home()), SUPPORTED_EXTS)
        if not path:
            return
        try:
            ref = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Reference", f"Không đọc được ảnh:\n{exc}")
            return
        self._reference_image = ref
        self.sidebar.set_diff_reference_name(Path(path).name)
        self.results_view.append_log(f"Reference loaded: {Path(path).name}")

    def _on_clear_reference(self):
        self._reference_image = None
        self.sidebar.set_diff_reference_name(None)
        self.results_view.append_log("Reference cleared.")

    # ------------------------------------------------------------------
    # Mask handlers
    # ------------------------------------------------------------------
    def _set_mask(self, mask: Optional[np.ndarray], source: str = ""):
        self._mask = mask
        self.canvas.set_mask(mask)
        if mask is None:
            self.mask_badge.setText("no mask")
            self.sidebar.mask_content.set_status("(none)")
            return
        nz = int(np.count_nonzero(mask))
        ratio = nz / mask.size
        self.mask_badge.setText(f"mask {ratio*100:.1f}%")
        info = f"{mask.shape[1]}×{mask.shape[0]} • {nz} px ({ratio*100:.1f}%)"
        if source:
            info += f" • {source}"
        self.sidebar.mask_content.set_status(info)

    def _on_mask_from_gray(self, mn: int, mx: int):
        if not self._ensure_image():
            return
        mask = mask_from_gray_range(self._original_image, mn, mx)
        self._set_mask(mask, source=f"gray[{mn},{mx}]")
        self.results_view.append_log(f"[Mask] gray[{mn},{mx}] -> {int(np.count_nonzero(mask))} px")

    def _on_mask_from_hsv(self, h1, h2, s1, s2, v1, v2):
        if not self._ensure_image():
            return
        img = self._original_image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([h1, s1, v1], np.uint8),
                           np.array([h2, s2, v2], np.uint8))
        self._set_mask(mask, source=f"HSV[{h1}-{h2},{s1}-{s2},{v1}-{v2}]")
        self.results_view.append_log(
            f"[Mask] HSV H[{h1},{h2}] S[{s1},{s2}] V[{v1},{v2}] -> {int(np.count_nonzero(mask))} px"
        )

    def _on_mask_invert(self):
        if self._mask is None:
            return
        self._set_mask(cv2.bitwise_not(self._mask), source="inverted")

    def _on_mask_clear(self):
        self._set_mask(None)
        self.results_view.append_log("Mask cleared.")

    def _on_mask_save(self):
        if self._mask is None:
            QMessageBox.information(self, "Mask", "Chưa có mask để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", str(Path.home() / "mask.png"), "PNG (*.png)"
        )
        if path and cv2.imwrite(path, self._mask):
            self.results_view.append_log(f"Mask saved → {path}")

    def _on_mask_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Mask", str(Path.home()), "Images (*.png *.bmp *.tif *.tiff)")
        if not path:
            return
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            QMessageBox.critical(self, "Mask", "Không đọc được file.")
            return
        # binarize
        _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
        self._set_mask(m, source=Path(path).name)

    def _show_result(self, result, title: str):
        self.canvas.set_image(result.image)
        self.results_view.set_metrics(result.metrics)
        self.results_view.append_log(f"=== {title} ===")
        self.results_view.append_log(result.log)

    # ==================================================================
    # Acquisition
    # ==================================================================
    def _on_acq_connect(self, interface: str, device: str):
        try:
            self._grabber.open(interface, device)
        except Exception as exc:
            QMessageBox.critical(self, "Acquisition", f"Connect lỗi:\n{exc}")
            self.sidebar.set_acq_connected(False)
            return
        self.sidebar.set_acq_connected(True, f"{self._grabber.backend} / {self._grabber.device_name}")
        self.results_view.append_log(f"Connected: {self._grabber.backend} / {self._grabber.device_name}")
        try:
            self._set_acquired_image(self._grabber.grab(), source="connect")
        except Exception as exc:
            self.results_view.append_log(f"Grab khởi tạo lỗi: {exc}")

    def _on_acq_disconnect(self):
        self._stop_live()
        self._grabber.close()
        self.sidebar.set_acq_connected(False)
        self.results_view.append_log("Disconnected.")
        self.viewer_badge.setText("idle")

    def _on_live_toggled(self, on: bool):
        if on:
            if not self._grabber.is_open:
                QMessageBox.information(self, "Live", "Hãy connect thiết bị trước.")
                self.sidebar.set_acq_live(False)
                return
            interval = max(33, int(1000 / max(1, self.sidebar.acq_fps_value)))
            self._live_timer.start(interval)
            self.sidebar.set_acq_live(True)
            self.results_view.append_log(f"Live started @ {self.sidebar.acq_fps_value} fps")
            self.viewer_badge.setText("live")
        else:
            self._stop_live()

    def _stop_live(self):
        if self._live_timer.isActive():
            self._live_timer.stop()
            self.sidebar.set_acq_live(False)
            self.results_view.append_log("Live stopped.")
            self.viewer_badge.setText("connected")

    def _on_fps_changed(self, fps: int):
        if self._live_timer.isActive():
            self._live_timer.start(max(33, int(1000 / max(1, fps))))

    def _on_live_tick(self):
        if not self._grabber.is_open:
            self._stop_live()
            return
        try:
            self._set_acquired_image(self._grabber.grab(), source="live", silent=True)
        except Exception as exc:
            self.results_view.append_log(f"Live grab lỗi: {exc}")
            self._stop_live()

    def _on_snapshot(self):
        if not self._grabber.is_open:
            return
        try:
            self._set_acquired_image(self._grabber.grab(), source="snapshot")
        except Exception as exc:
            QMessageBox.critical(self, "Snapshot", f"Grab lỗi:\n{exc}")

    def _set_acquired_image(self, img: np.ndarray, source: str, silent: bool = False):
        self._image_path = None
        self._original_image = img
        self.canvas.set_image(img)
        self._image_label.setText(f"[{source}]  {img.shape[1]}×{img.shape[0]}")
        self.viewer_badge.setText(source)
        if not silent:
            self.results_view.append_log(f"Acquired ({source}): {img.shape[1]}×{img.shape[0]}")

    def closeEvent(self, event):  # type: ignore[override]
        self._stop_live()
        self._grabber.close()
        super().closeEvent(event)
