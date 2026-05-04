"""Main window cho HALCON Vision Studio."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
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
    edges_sub_pix,
    measure_pairs,
    read_image,
    shape_match,
    threshold_blob,
)
from app.widgets import (
    AcquisitionPanel,
    ImageCanvas,
    OperatorPanel,
    ResultsView,
)


SUPPORTED_EXTS = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HALCON Vision Studio")
        self.resize(1400, 880)
        self.setMinimumSize(QSize(1100, 700))

        self._image_path: Optional[Path] = None
        self._original_image: Optional[np.ndarray] = None
        self._template_path: Optional[Path] = None
        self._template_image: Optional[np.ndarray] = None

        # Acquisition
        self._grabber = Grabber()
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._on_live_tick)

        # ROI picking flag — khi True, ROI vẽ sẽ được dùng làm template
        self._picking_template = False

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

        splitter = QSplitter(Qt.Horizontal, central)

        # Left: image canvas
        canvas_holder = QWidget()
        cv_lay = QVBoxLayout(canvas_holder)
        cv_lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Image Viewer")
        title.setProperty("heading", True)
        cv_lay.addWidget(title)
        self.canvas = ImageCanvas()
        cv_lay.addWidget(self.canvas, 1)

        # Right: operator panel + results
        right = QSplitter(Qt.Vertical)
        self.operator_panel = OperatorPanel()
        self.results_view = ResultsView()
        right.addWidget(self.operator_panel)
        right.addWidget(self.results_view)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)

        splitter.addWidget(canvas_holder)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)
        self.setCentralWidget(central)

        # Acquisition dock (left)
        self.acq_panel = AcquisitionPanel()
        dock = QDockWidget("Acquisition", self)
        dock.setObjectName("AcquisitionDock")
        dock.setWidget(self.acq_panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._acq_dock = dock

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
        m_file.addAction(act_save)

        m_file.addSeparator()
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
        m_view.addAction(self._acq_dock.toggleViewAction())

        m_help = bar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        m_help.addAction(act_about)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("📂  Open Image", self)
        act_open.triggered.connect(self._on_open_image)
        tb.addAction(act_open)

        act_template = QAction("🧩  Open Template", self)
        act_template.triggered.connect(self._on_open_template)
        tb.addAction(act_template)

        tb.addSeparator()

        act_reset_img = QAction("↺  Reset Image", self)
        act_reset_img.triggered.connect(self._on_reset_image)
        tb.addAction(act_reset_img)

        act_fit = QAction("🔍  Fit", self)
        act_fit.triggered.connect(self.canvas.fit_to_view)
        tb.addAction(act_fit)

        tb.addSeparator()
        act_save = QAction("💾  Save Result", self)
        act_save.triggered.connect(self._on_save_result)
        tb.addAction(act_save)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        engine = (
            f"HALCON {HALCON_VERSION} ✓" if HALCON_AVAILABLE else "HALCON unavailable — using OpenCV fallback"
        )
        self._engine_label = QLabel(engine)
        self._image_label = QLabel("No image")
        self._cursor_label = QLabel("")
        sb.addPermanentWidget(self._image_label, 2)
        sb.addPermanentWidget(self._cursor_label, 2)
        sb.addPermanentWidget(self._engine_label, 1)

    def _wire_signals(self):
        self.operator_panel.threshold_run.connect(self._run_threshold)
        self.operator_panel.edges_run.connect(self._run_edges)
        self.operator_panel.shape_match_run.connect(self._run_shape_match)
        self.operator_panel.shape_template_load.connect(self._on_open_template)
        self.operator_panel.shape_template_save.connect(self._on_save_template)
        self.operator_panel.shape_template_clear.connect(self._on_clear_template)
        self.operator_panel.shape_pick_roi_toggled.connect(self._on_pick_roi_toggled)
        self.operator_panel.measure_run.connect(self._run_measure)
        self.operator_panel.measure_mode_toggled.connect(self.canvas.set_measure_mode)

        self.canvas.measure_segment_drawn.connect(
            self.operator_panel.set_measure_segment
        )
        self.canvas.roi_drawn.connect(self._on_roi_drawn)
        self.canvas.mouse_moved.connect(self._on_cursor_moved)

        # Acquisition
        self.acq_panel.connect_requested.connect(self._on_acq_connect)
        self.acq_panel.disconnect_requested.connect(self._on_acq_disconnect)
        self.acq_panel.live_toggled.connect(self._on_live_toggled)
        self.acq_panel.snapshot_requested.connect(self._on_snapshot)
        self.acq_panel.fps_changed.connect(self._on_fps_changed)

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), SUPPORTED_EXTS
        )
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
        self._image_label.setText(
            f"{self._image_path.name}  •  {img.shape[1]}×{img.shape[0]}"
        )
        self.results_view.append_log(f"Loaded image: {self._image_path.name}")

    def _on_open_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Template", str(Path.home()), SUPPORTED_EXTS
        )
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
        self.operator_panel.set_template_name(f"{name}  •  {img.shape[1]}×{img.shape[0]}")
        self.operator_panel.set_template_preview(img)
        self.results_view.append_log(
            f"Template ← {source}: {name} ({img.shape[1]}×{img.shape[0]})"
        )

    def _on_save_template(self):
        if self._template_image is None:
            QMessageBox.information(self, "Template", "Chưa có template để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", str(Path.home() / "template.png"), SUPPORTED_EXTS
        )
        if not path:
            return
        if cv2.imwrite(path, self._template_image):
            self.results_view.append_log(f"Saved template → {path}")
        else:
            QMessageBox.warning(self, "Save", "Lưu thất bại.")

    def _on_clear_template(self):
        self._template_image = None
        self._template_path = None
        self.operator_panel.set_template_name(None)
        self.operator_panel.set_template_preview(None)
        self.results_view.append_log("Template cleared.")

    def _on_reset_image(self):
        if self._original_image is not None:
            self.canvas.set_image(self._original_image)

    def _on_save_result(self):
        img = self.canvas.current_image()
        if img is None:
            QMessageBox.information(self, "Save", "Chưa có ảnh để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", str(Path.home() / "result.png"), SUPPORTED_EXTS
        )
        if not path:
            return
        if cv2.imwrite(path, img):
            self.results_view.append_log(f"Saved → {path}")
        else:
            QMessageBox.warning(self, "Save", "Lưu thất bại.")

    def _on_about(self):
        msg = (
            "<b>HALCON Vision Studio</b><br>"
            "PySide6 GUI wrapping MVTec HALCON operators.<br><br>"
            f"Engine: <b>{'HALCON ' + HALCON_VERSION if HALCON_AVAILABLE else 'OpenCV fallback'}</b>"
        )
        QMessageBox.about(self, "About", msg)

    def _on_cursor_moved(self, row: int, col: int, value):
        self._cursor_label.setText(f"({col}, {row})  px={value}")

    # ------------------------------------------------------------------
    # Operator runs
    # ------------------------------------------------------------------
    def _ensure_image(self) -> bool:
        if self._original_image is None:
            QMessageBox.information(self, "No Image", "Hãy mở một ảnh trước.")
            return False
        return True

    def _run_threshold(self, params: dict):
        if not self._ensure_image():
            return
        result = threshold_blob(self._original_image, **params)
        self._show_result(result, title="Threshold + Blob")

    def _run_edges(self, params: dict):
        if not self._ensure_image():
            return
        result = edges_sub_pix(self._original_image, **params)
        self._show_result(result, title="Edges")

    def _run_shape_match(self, params: dict):
        if not self._ensure_image():
            return
        if self._template_image is None:
            QMessageBox.information(
                self, "Template", "Hãy load template trước khi match."
            )
            return
        result = shape_match(self._original_image, self._template_image, **params)
        self._show_result(result, title="Shape Match")

    def _run_measure(self, params: dict):
        if not self._ensure_image():
            return
        result = measure_pairs(self._original_image, **params)
        self._show_result(result, title="Measure 1D")

    def _show_result(self, result, title: str):
        self.canvas.set_image(result.image)
        self.results_view.set_metrics(result.metrics)
        self.results_view.append_log(f"=== {title} ===")
        self.results_view.append_log(result.log)

    # ------------------------------------------------------------------
    # ROI / template picking
    # ------------------------------------------------------------------
    def _on_pick_roi_toggled(self, enabled: bool):
        self._picking_template = enabled
        self.canvas.set_roi_mode(enabled)
        if enabled:
            self.results_view.append_log(
                "Pick mode: kéo chuột để chọn ROI làm template…"
            )
            self.operator_panel.focus_match_tab()
        else:
            self.canvas.clear_roi()

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int):
        if not self._picking_template or self._original_image is None:
            return
        if w < 5 or h < 5:
            self.results_view.append_log("ROI quá nhỏ — bỏ qua.")
            return
        crop = self._original_image[y : y + h, x : x + w].copy()
        self._set_template(
            crop,
            name=f"ROI({x},{y},{w}×{h})",
            source="ROI",
        )
        # tắt pick mode sau khi lấy mẫu xong
        self._picking_template = False
        self.canvas.set_roi_mode(False)
        self.canvas.clear_roi()
        self.operator_panel.reset_pick_button()

    # ------------------------------------------------------------------
    # Acquisition handlers
    # ------------------------------------------------------------------
    def _on_acq_connect(self, interface: str, device: str):
        try:
            self._grabber.open(interface, device)
        except Exception as exc:
            QMessageBox.critical(self, "Acquisition", f"Connect lỗi:\n{exc}")
            self.acq_panel.set_connected(False)
            return
        self.acq_panel.set_connected(
            True, f"{self._grabber.backend} / {self._grabber.device_name}"
        )
        self.results_view.append_log(
            f"Connected: {self._grabber.backend} / {self._grabber.device_name}"
        )
        # grab 1 frame ngay để có ảnh khởi tạo
        try:
            frame = self._grabber.grab()
            self._set_acquired_image(frame, source="connect")
        except Exception as exc:
            self.results_view.append_log(f"Grab khởi tạo lỗi: {exc}")

    def _on_acq_disconnect(self):
        self._stop_live()
        self._grabber.close()
        self.acq_panel.set_connected(False)
        self.results_view.append_log("Disconnected.")

    def _on_live_toggled(self, on: bool):
        if on:
            if not self._grabber.is_open:
                QMessageBox.information(self, "Live", "Hãy connect thiết bị trước.")
                self.acq_panel.set_live(False)
                return
            interval_ms = max(33, int(1000 / max(1, self.acq_panel.fps_spin.value())))
            self._live_timer.start(interval_ms)
            self.acq_panel.set_live(True)
            self.results_view.append_log(
                f"Live started @ {self.acq_panel.fps_spin.value()} fps"
            )
        else:
            self._stop_live()

    def _stop_live(self):
        if self._live_timer.isActive():
            self._live_timer.stop()
            self.acq_panel.set_live(False)
            self.results_view.append_log("Live stopped.")

    def _on_fps_changed(self, fps: int):
        if self._live_timer.isActive():
            self._live_timer.start(max(33, int(1000 / max(1, fps))))

    def _on_live_tick(self):
        if not self._grabber.is_open:
            self._stop_live()
            return
        try:
            frame = self._grabber.grab()
        except Exception as exc:
            self.results_view.append_log(f"Live grab lỗi: {exc}")
            self._stop_live()
            return
        self._set_acquired_image(frame, source="live", silent=True)

    def _on_snapshot(self):
        if not self._grabber.is_open:
            return
        try:
            frame = self._grabber.grab()
        except Exception as exc:
            QMessageBox.critical(self, "Snapshot", f"Grab lỗi:\n{exc}")
            return
        self._set_acquired_image(frame, source="snapshot")

    def _set_acquired_image(self, img: np.ndarray, source: str, silent: bool = False):
        self._image_path = None
        self._original_image = img
        self.canvas.set_image(img)
        self._image_label.setText(
            f"[{source}]  {img.shape[1]}×{img.shape[0]}"
        )
        if not silent:
            self.results_view.append_log(
                f"Acquired ({source}): {img.shape[1]}×{img.shape[0]}"
            )

    # ------------------------------------------------------------------
    def closeEvent(self, event):  # type: ignore[override]
        self._stop_live()
        self._grabber.close()
        super().closeEvent(event)
