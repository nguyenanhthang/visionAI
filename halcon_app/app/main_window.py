"""Main window — pipeline-first layout (no left sidebar).

Layout:
  • Toolbar (top)
  • Center: large image viewer (splitter với Results panel collapsible bên dưới)
  • Right dock: QTabWidget với hai tab
      - Pipeline   (drag-drop tool chain)
      - Resources  (Acquisition, Mask, Template, Reference, Canvas inputs)
"""
from __future__ import annotations

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
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.operators import (
    HALCON_AVAILABLE,
    HALCON_VERSION,
    TOOLS,
    Grabber,
    Pipeline,
    PipelineContext,
    apply_mask,
    mask_from_gray_range,
    mask_from_roi,
    read_image,
)
from app.widgets import (
    ImageCanvas,
    ParamDialog,
    PipelinePanel,
    ResourcesPanel,
    ResultsView,
)

SUPPORTED_EXTS = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HALCON Vision Studio")
        self.resize(1480, 920)
        self.setMinimumSize(QSize(1180, 720))

        # State
        self._image_path: Optional[Path] = None
        self._original_image: Optional[np.ndarray] = None
        self._template_path: Optional[Path] = None
        self._template_image: Optional[np.ndarray] = None
        self._reference_image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None

        self._grabber = Grabber()
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._on_live_tick)

        self._roi_purpose: Optional[str] = None  # template / mask / color

        # Pipeline
        self._pipeline = Pipeline()
        self._segment: Optional[tuple[int, int, int, int]] = None
        self._color_roi: Optional[tuple[int, int, int, int]] = None
        self._selected_node_idx: Optional[int] = None

        self._build_ui()
        self._build_dock()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # Header row
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
        v.addLayout(title_row)

        # Vertical splitter — canvas (large) + results (collapsible)
        self._v_split = QSplitter(Qt.Vertical)
        self._v_split.setChildrenCollapsible(True)

        self.canvas = ImageCanvas()
        self._v_split.addWidget(self.canvas)

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
        self._v_split.addWidget(results_holder)

        self._v_split.setStretchFactor(0, 5)
        self._v_split.setStretchFactor(1, 1)
        self._v_split.setSizes([800, 220])
        v.addWidget(self._v_split, 1)

        self.setCentralWidget(central)

    def _build_dock(self):
        self.pipeline_panel = PipelinePanel()
        self.resources = ResourcesPanel()

        self._dock_tabs = QTabWidget()
        self._dock_tabs.addTab(self.pipeline_panel, "▶  Pipeline")
        self._dock_tabs.addTab(self.resources, "🧰  Resources")

        dock = QDockWidget("Pipeline & Resources", self)
        dock.setObjectName("PipelineDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setWidget(self._dock_tabs)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._dock = dock

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

    # ==================================================================
    # Menu / toolbar / statusbar
    # ==================================================================
    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("&File")
        a_open = QAction("Open Image…", self); a_open.setShortcut(QKeySequence.Open)
        a_open.triggered.connect(self._on_open_image); m_file.addAction(a_open)
        a_tpl = QAction("Open Template…", self); a_tpl.triggered.connect(self._on_open_template)
        m_file.addAction(a_tpl)
        a_ref = QAction("Open Reference…", self); a_ref.triggered.connect(self._on_load_reference)
        m_file.addAction(a_ref)
        m_file.addSeparator()
        a_save = QAction("Save Result Image…", self); a_save.setShortcut(QKeySequence.Save)
        a_save.triggered.connect(self._on_save_result); m_file.addAction(a_save)
        m_file.addSeparator()
        a_quit = QAction("Quit", self); a_quit.setShortcut(QKeySequence.Quit)
        a_quit.triggered.connect(self.close); m_file.addAction(a_quit)

        m_view = bar.addMenu("&View")
        m_view.addAction(QAction("Fit to Window", self,
                                 shortcut="Ctrl+0",
                                 triggered=self.canvas.fit_to_view))
        m_view.addAction(QAction("Reset Zoom", self,
                                 shortcut="Ctrl+1",
                                 triggered=self.canvas.reset_zoom))
        m_view.addSeparator()
        m_view.addAction(QAction("Expand resources", self, triggered=self.resources.expand_all))
        m_view.addAction(QAction("Collapse resources", self, triggered=self.resources.collapse_all))
        m_view.addSeparator()
        m_view.addAction(self._dock.toggleViewAction())

        m_help = bar.addMenu("&Help")
        m_help.addAction(QAction("About", self, triggered=self._on_about))

    def _build_toolbar(self):
        tb = QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        tb.addAction(QAction("📂  Open", self, triggered=self._on_open_image))
        tb.addAction(QAction("🧩  Template", self, triggered=self._on_open_template))
        tb.addAction(QAction("📋  Reference", self, triggered=self._on_load_reference))
        tb.addSeparator()
        tb.addAction(QAction("↺  Reset Image", self, triggered=self._on_reset_image))
        tb.addAction(QAction("🔍  Fit", self, triggered=self.canvas.fit_to_view))
        tb.addAction(QAction("1:1  Reset Zoom", self, triggered=self.canvas.reset_zoom))
        tb.addSeparator()
        tb.addAction(QAction("▶▶  Run Pipeline", self, triggered=self._on_pipeline_run))
        tb.addSeparator()
        tb.addAction(QAction("💾  Save Result", self, triggered=self._on_save_result))

    def _build_statusbar(self):
        sb = QStatusBar(); self.setStatusBar(sb)
        engine = (f"HALCON {HALCON_VERSION}  ●" if HALCON_AVAILABLE
                  else "Engine: OpenCV fallback  ●")
        self._engine_label = QLabel(engine)
        self._image_label = QLabel("No image loaded")
        self._cursor_label = QLabel("")
        sb.addPermanentWidget(self._image_label, 2)
        sb.addPermanentWidget(self._cursor_label, 2)
        sb.addPermanentWidget(self._engine_label, 1)

    def _wire_signals(self):
        # ---- Resources ----
        r = self.resources
        r.acq_connect.connect(self._on_acq_connect)
        r.acq_disconnect.connect(self._on_acq_disconnect)
        r.acq_live.connect(self._on_live_toggled)
        r.acq_snapshot.connect(self._on_snapshot)
        r.acq_fps.connect(self._on_fps_changed)

        r.mask_gen_gray.connect(self._on_mask_from_gray)
        r.mask_gen_hsv.connect(self._on_mask_from_hsv)
        r.mask_pick_roi_toggled.connect(self._on_pick_mask_roi)
        r.mask_invert.connect(self._on_mask_invert)
        r.mask_clear.connect(self._on_mask_clear)
        r.mask_show_toggled.connect(self.canvas.set_show_mask)
        r.mask_save.connect(self._on_mask_save)
        r.mask_load.connect(self._on_mask_load)

        r.template_load.connect(self._on_open_template)
        r.template_save.connect(self._on_save_template)
        r.template_clear.connect(self._on_clear_template)
        r.template_pick_roi_toggled.connect(self._on_pick_template_roi)

        r.reference_load.connect(self._on_load_reference)
        r.reference_clear.connect(self._on_clear_reference)

        r.measure_mode_toggled.connect(self.canvas.set_measure_mode)
        r.color_pick_roi_toggled.connect(self._on_pick_color_roi)

        # ---- Canvas ----
        self.canvas.measure_segment_drawn.connect(self._on_segment_drawn)
        self.canvas.roi_drawn.connect(self._on_roi_drawn)
        self.canvas.mouse_moved.connect(self._on_cursor_moved)

        # ---- Pipeline panel ----
        p = self.pipeline_panel
        p.add_node_requested.connect(self._on_pipeline_add)
        p.edit_node_requested.connect(self._on_pipeline_edit)
        p.delete_node_requested.connect(self._on_pipeline_delete)
        p.enable_node_changed.connect(self._on_pipeline_enable)
        p.select_node.connect(self._on_pipeline_select)
        p.run_requested.connect(self._on_pipeline_run)
        p.clear_requested.connect(self._on_pipeline_clear)
        p.reorder_changed.connect(self._on_pipeline_reorder)
        p.params_changed.connect(self._on_pipeline_params_changed)
        p.live_apply_requested.connect(self._on_pipeline_run)

    # ==================================================================
    # File / image
    # ==================================================================
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", str(Path.home()), SUPPORTED_EXTS)
        if not path: return
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
        if not path: return
        try:
            tpl = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được template:\n{exc}"); return
        self._set_template(tpl, name=Path(path).name, source="file")

    def _set_template(self, img: np.ndarray, name: str, source: str):
        self._template_image = img
        self._template_path = Path(name) if source == "file" else None
        self.resources.set_template_name(f"{name}  •  {img.shape[1]}×{img.shape[0]}")
        self.resources.set_template_preview(img)
        self.resources.focus_template()
        self.results_view.append_log(f"Template ← {source}: {name} ({img.shape[1]}×{img.shape[0]})")

    def _on_save_template(self):
        if self._template_image is None:
            QMessageBox.information(self, "Template", "Chưa có template để lưu."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Template", str(Path.home() / "template.png"), SUPPORTED_EXTS)
        if path and cv2.imwrite(path, self._template_image):
            self.results_view.append_log(f"Saved template → {path}")

    def _on_clear_template(self):
        self._template_image = None; self._template_path = None
        self.resources.set_template_name(None)
        self.resources.set_template_preview(None)
        self.results_view.append_log("Template cleared.")

    def _on_load_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Reference", str(Path.home()), SUPPORTED_EXTS)
        if not path: return
        try:
            ref = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Reference", f"Không đọc được ảnh:\n{exc}"); return
        self._reference_image = ref
        self.resources.set_reference_name(Path(path).name)
        self.results_view.append_log(f"Reference loaded: {Path(path).name}")

    def _on_clear_reference(self):
        self._reference_image = None
        self.resources.set_reference_name(None)
        self.results_view.append_log("Reference cleared.")

    def _on_reset_image(self):
        if self._original_image is not None:
            self.canvas.set_image(self._original_image)

    def _on_save_result(self):
        img = self.canvas.current_image()
        if img is None:
            QMessageBox.information(self, "Save", "Chưa có ảnh để lưu."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", str(Path.home() / "result.png"), SUPPORTED_EXTS)
        if path and cv2.imwrite(path, img):
            self.results_view.append_log(f"Saved → {path}")

    def _on_about(self):
        QMessageBox.about(
            self, "About",
            "<b>HALCON Vision Studio</b><br>"
            "PySide6 GUI inspired by Cognex VisionPro, wrapping MVTec HALCON.<br><br>"
            f"Engine: <b>{'HALCON ' + HALCON_VERSION if HALCON_AVAILABLE else 'OpenCV fallback'}</b>"
        )

    def _on_cursor_moved(self, row: int, col: int, value):
        self._cursor_label.setText(f"({col}, {row})  px={value}")

    # ==================================================================
    # ROI dispatch
    # ==================================================================
    def _on_pick_template_roi(self, on: bool):
        self._roi_purpose = "template" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: chọn ROI cho template…")
            self.resources.focus_template()
        else:
            self.canvas.clear_roi()

    def _on_pick_mask_roi(self, on: bool):
        self._roi_purpose = "mask" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: chọn ROI làm mask…")
        else:
            self.canvas.clear_roi()

    def _on_pick_color_roi(self, on: bool):
        self._roi_purpose = "color" if on else None
        self.canvas.set_roi_mode(on)
        if on:
            self.results_view.append_log("Pick mode: chọn ROI cho color stats…")
        else:
            self.canvas.clear_roi()

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int):
        if self._original_image is None or w < 5 or h < 5:
            return
        if self._roi_purpose == "template":
            crop = self._original_image[y : y + h, x : x + w].copy()
            self._set_template(crop, name=f"ROI({x},{y},{w}×{h})", source="ROI")
            self.resources.reset_template_pick()
        elif self._roi_purpose == "mask":
            mask = mask_from_roi(self._original_image, x, y, w, h)
            self._set_mask(mask, source=f"ROI({x},{y},{w}×{h})")
            self.resources.reset_mask_pick()
        elif self._roi_purpose == "color":
            self._color_roi = (x, y, w, h)
            self.resources.set_color_roi(x, y, w, h)
        self._roi_purpose = None
        self.canvas.set_roi_mode(False); self.canvas.clear_roi()

    def _on_segment_drawn(self, r1: int, c1: int, r2: int, c2: int):
        self._segment = (r1, c1, r2, c2)
        self.resources.set_segment(r1, c1, r2, c2)

    # ==================================================================
    # Mask
    # ==================================================================
    def _set_mask(self, mask: Optional[np.ndarray], source: str = ""):
        self._mask = mask
        self.canvas.set_mask(mask)
        if mask is None:
            self.mask_badge.setText("no mask")
            self.resources.set_mask_status("(none)")
            return
        nz = int(np.count_nonzero(mask))
        ratio = nz / mask.size
        self.mask_badge.setText(f"mask {ratio*100:.1f}%")
        info = f"{mask.shape[1]}×{mask.shape[0]} • {nz} px ({ratio*100:.1f}%)"
        if source: info += f" • {source}"
        self.resources.set_mask_status(info)

    def _on_mask_from_gray(self, mn: int, mx: int):
        if self._original_image is None: return
        mask = mask_from_gray_range(self._original_image, mn, mx)
        self._set_mask(mask, source=f"gray[{mn},{mx}]")
        self.results_view.append_log(f"[Mask] gray[{mn},{mx}] -> {int(np.count_nonzero(mask))} px")

    def _on_mask_from_hsv(self, h1, h2, s1, s2, v1, v2):
        if self._original_image is None: return
        img = self._original_image
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([h1, s1, v1], np.uint8),
                           np.array([h2, s2, v2], np.uint8))
        self._set_mask(mask, source=f"HSV[{h1}-{h2},{s1}-{s2},{v1}-{v2}]")
        self.results_view.append_log(
            f"[Mask] HSV H[{h1},{h2}] S[{s1},{s2}] V[{v1},{v2}] -> {int(np.count_nonzero(mask))} px"
        )

    def _on_mask_invert(self):
        if self._mask is None: return
        self._set_mask(cv2.bitwise_not(self._mask), source="inverted")

    def _on_mask_clear(self):
        self._set_mask(None)
        self.results_view.append_log("Mask cleared.")

    def _on_mask_save(self):
        if self._mask is None:
            QMessageBox.information(self, "Mask", "Chưa có mask để lưu."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Mask", str(Path.home() / "mask.png"), "PNG (*.png)")
        if path and cv2.imwrite(path, self._mask):
            self.results_view.append_log(f"Mask saved → {path}")

    def _on_mask_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Mask", str(Path.home()), "Images (*.png *.bmp *.tif *.tiff)")
        if not path: return
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            QMessageBox.critical(self, "Mask", "Không đọc được file."); return
        _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
        self._set_mask(m, source=Path(path).name)

    # ==================================================================
    # Pipeline
    # ==================================================================
    def _build_ctx(self) -> PipelineContext:
        return PipelineContext(
            template=self._template_image,
            reference=self._reference_image,
            mask=self._mask,
            segment=self._segment,
            color_roi=self._color_roi,
        )

    def _on_pipeline_add(self, tool_id: str):
        self._pipeline.add(tool_id)
        idx = len(self._pipeline.nodes) - 1
        self.pipeline_panel.rebuild(self._pipeline.nodes, select_idx=idx)
        self.pipeline_panel.set_status(
            f"Added '{TOOLS[tool_id].display}' (#{idx+1}). Click ▶ Run All để chạy."
        )

    def _on_pipeline_edit(self, idx: int):
        if not (0 <= idx < len(self._pipeline.nodes)): return
        node = self._pipeline.nodes[idx]
        spec = TOOLS[node.tool_id]
        dlg = ParamDialog(spec, node.params, self)
        if dlg.exec() == ParamDialog.Accepted:
            node.params = dlg.values()
            node.last_image = None; node.last_metrics = None
            node.last_thumbnail = None; node.error = None
            self.pipeline_panel.refresh_node(idx, node)
            self.pipeline_panel.set_status(f"Updated #{idx+1} '{spec.display}'")

    def _on_pipeline_delete(self, idx: int):
        if not (0 <= idx < len(self._pipeline.nodes)): return
        spec = TOOLS[self._pipeline.nodes[idx].tool_id]
        self._pipeline.remove(idx)
        self.pipeline_panel.rebuild(self._pipeline.nodes)
        self.pipeline_panel.set_status(f"Removed '{spec.display}'")

    def _on_pipeline_enable(self, idx: int, on: bool):
        if 0 <= idx < len(self._pipeline.nodes):
            self._pipeline.nodes[idx].enabled = on

    def _on_pipeline_reorder(self, src: int, dst: int):
        self._pipeline.move(src, dst)
        self.pipeline_panel.rebuild(self._pipeline.nodes, select_idx=dst)
        self.pipeline_panel.set_status(f"Moved #{src+1} → #{dst+1}")

    def _on_pipeline_clear(self):
        if not self._pipeline.nodes: return
        if QMessageBox.question(self, "Clear Pipeline", "Xoá toàn bộ pipeline?") != QMessageBox.Yes:
            return
        self._pipeline.clear()
        self._selected_node_idx = None
        self.pipeline_panel.rebuild(self._pipeline.nodes)
        self.pipeline_panel.set_status("Pipeline cleared.")

    def _on_pipeline_select(self, idx: int):
        if not (0 <= idx < len(self._pipeline.nodes)): return
        self._selected_node_idx = idx
        node = self._pipeline.nodes[idx]
        spec = TOOLS[node.tool_id]
        if node.last_image is not None:
            self.canvas.set_image(node.last_image)
            self.results_view.set_metrics(node.last_metrics or {})
            self.results_view.append_log(f"--- Preview #{idx+1} {spec.icon} {spec.display} ---")
            if node.last_log: self.results_view.append_log(node.last_log)
            self.viewer_badge.setText(f"#{idx+1} {spec.display}")
        elif node.error:
            self.results_view.append_log(f"⚠ Node #{idx+1} error: {node.error}")
        else:
            self.results_view.append_log(f"Node #{idx+1} chưa chạy. Bấm '▶ Run All' trước.")

    def _on_pipeline_run(self, focus_idx: Optional[int] = None):
        if not self._pipeline.nodes:
            QMessageBox.information(self, "Pipeline", "Chưa có node nào."); return
        if self._original_image is None:
            QMessageBox.information(self, "Pipeline", "Hãy mở ảnh / acquire frame trước."); return
        ctx = self._build_ctx()
        self._pipeline.run(self._original_image, ctx)
        # Refresh tất cả thumbnail mà KHÔNG rebuild form (giữ widget user đang chỉnh)
        for i, node in enumerate(self._pipeline.nodes):
            self.pipeline_panel.refresh_node(i, node)
        ok = sum(1 for n in self._pipeline.nodes if n.error is None and n.enabled)
        err = sum(1 for n in self._pipeline.nodes if n.error)
        skipped = sum(1 for n in self._pipeline.nodes if not n.enabled)
        self.pipeline_panel.set_status(f"Run — {ok} ok, {err} error, {skipped} skipped")

        # Chọn node để preview: ưu tiên focus_idx (node vừa edit), fallback last
        if focus_idx is not None and 0 <= focus_idx < len(self._pipeline.nodes):
            target = focus_idx
        elif self._selected_node_idx is not None:
            target = self._selected_node_idx
        else:
            target = next(
                (i for i in range(len(self._pipeline.nodes) - 1, -1, -1)
                 if self._pipeline.nodes[i].last_image is not None),
                None,
            )
        if target is not None:
            # Sync list selection → trigger props panel update + canvas preview
            self.pipeline_panel.select(target)
            self._on_pipeline_select(target)

    def _on_pipeline_params_changed(self, idx: int, new_params: dict):
        """Live edit: cập nhật params, chạy pipeline, preview node vừa chỉnh."""
        if not (0 <= idx < len(self._pipeline.nodes)):
            return
        node = self._pipeline.nodes[idx]
        node.params = new_params
        # Invalidate cached output
        node.last_image = None; node.last_metrics = None
        node.last_thumbnail = None; node.error = None
        if not self.pipeline_panel.props_panel.live:
            # Live tắt → chỉ cập nhật subtitle/badge, chờ Apply
            self.pipeline_panel.refresh_node(idx, node)
            self.pipeline_panel.set_status(
                f"Edited #{idx+1} (live preview off — bấm Apply để chạy)"
            )
            return
        # Live on → chạy pipeline, preview tại node vừa edit
        if self._original_image is None:
            return
        self._on_pipeline_run(focus_idx=idx)

    # ==================================================================
    # Acquisition
    # ==================================================================
    def _on_acq_connect(self, interface: str, device: str):
        try:
            self._grabber.open(interface, device)
        except Exception as exc:
            QMessageBox.critical(self, "Acquisition", f"Connect lỗi:\n{exc}")
            self.resources.set_acq_connected(False); return
        self.resources.set_acq_connected(True, f"{self._grabber.backend} / {self._grabber.device_name}")
        self.results_view.append_log(f"Connected: {self._grabber.backend} / {self._grabber.device_name}")
        try:
            self._set_acquired_image(self._grabber.grab(), source="connect")
        except Exception as exc:
            self.results_view.append_log(f"Grab khởi tạo lỗi: {exc}")

    def _on_acq_disconnect(self):
        self._stop_live()
        self._grabber.close()
        self.resources.set_acq_connected(False)
        self.results_view.append_log("Disconnected.")
        self.viewer_badge.setText("idle")

    def _on_live_toggled(self, on: bool):
        if on:
            if not self._grabber.is_open:
                QMessageBox.information(self, "Live", "Hãy connect thiết bị trước.")
                self.resources.set_acq_live(False); return
            interval = max(33, int(1000 / max(1, self.resources.acq_fps_value)))
            self._live_timer.start(interval)
            self.resources.set_acq_live(True)
            self.results_view.append_log(f"Live started @ {self.resources.acq_fps_value} fps")
            self.viewer_badge.setText("live")
        else:
            self._stop_live()

    def _stop_live(self):
        if self._live_timer.isActive():
            self._live_timer.stop()
            self.resources.set_acq_live(False)
            self.results_view.append_log("Live stopped.")
            self.viewer_badge.setText("connected")

    def _on_fps_changed(self, fps: int):
        if self._live_timer.isActive():
            self._live_timer.start(max(33, int(1000 / max(1, fps))))

    def _on_live_tick(self):
        if not self._grabber.is_open:
            self._stop_live(); return
        try:
            self._set_acquired_image(self._grabber.grab(), source="live", silent=True)
        except Exception as exc:
            self.results_view.append_log(f"Live grab lỗi: {exc}")
            self._stop_live()

    def _on_snapshot(self):
        if not self._grabber.is_open: return
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
