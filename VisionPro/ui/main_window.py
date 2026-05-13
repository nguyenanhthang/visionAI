"""
ui/main_window.py — v3
- Fix: double-click node → NodeDetailDialog (cửa sổ mới)
- Fix: ImageViewerPanel là tab xem ảnh chính
- Fix: nối port working
"""
from __future__ import annotations
import os, time
from typing import Optional

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QSplitter, QLabel, QPushButton, QStatusBar,
                                QFileDialog, QMessageBox, QProgressBar,
                                QFrame, QTabWidget, QApplication)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence, QFont

from core.flow_graph import FlowGraph
from ui.canvas_view import AOICanvas
from ui.tool_library import ToolLibraryPanel
from ui.properties_panel import PropertiesPanel
from ui.results_panel import ResultsPanel
from ui.image_viewer import ImageViewerPanel
from ui.node_detail_dialog import NodeDetailDialog
from core.plc import PLCManager


# ── Worker ────────────────────────────────────────────────────────
class PipelineWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict, float)
    error    = Signal(str)

    def __init__(self, graph: FlowGraph):
        super().__init__()
        self.graph = graph

    def run(self):
        try:
            self.graph.reset_status()
            t0 = time.perf_counter()
            results = self.graph.execute(progress_cb=self.progress.emit)
            self.finished.emit(results, (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.error.emit(str(e))


# ── Toolbar button styles ─────────────────────────────────────────
_TB_STYLE = """
    QPushButton{background:#111827;border:1px solid #1e2d45;border-radius:5px;
                color:#94a3b8;font-size:12px;font-weight:600;padding:0 12px;}
    QPushButton:hover{background:#1a2236;border-color:#00d4ff;color:#e2e8f0;}
    QPushButton:pressed{background:#0d1a2a;}
"""
_RUN_STYLE = """
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #006688,stop:1 #009bbb);
                border:none;border-radius:5px;color:#000;
                font-size:13px;font-weight:700;letter-spacing:1px;}
    QPushButton:hover{background:#00d4ff;}
"""
_STOP_STYLE = """
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #8b1a2a,stop:1 #cc1a3a);
                border:none;border-radius:5px;color:#fff;
                font-size:13px;font-weight:700;letter-spacing:1px;}
    QPushButton:hover{background:#ff3860;}
"""


def _tb_btn(icon: str, label: str, tip: str = "") -> QPushButton:
    b = QPushButton(f"{icon}  {label}")
    b.setToolTip(tip)
    b.setFixedHeight(34)
    b.setStyleSheet(_TB_STYLE)
    return b


def _sep() -> QFrame:
    s = QFrame()
    s.setFrameShape(QFrame.VLine)
    s.setStyleSheet("color:#1e2d45;")
    return s


# ── Main Window ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEO Vision Pro  v1.0")
        self.resize(1560, 940)
        self.setMinimumSize(1100, 700)

        self._graph         = FlowGraph()
        self._current_file: Optional[str] = None
        self._worker_thread: Optional[QThread] = None
        self._is_running    = False
        self._detail_dialogs: dict = {}   # node_id → NodeDetailDialog

        # PLC integration — persistent manager, shared with PLCDialog
        self._plc_manager = PLCManager()
        self._plc_dialog = None

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._restore_state()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._tick)
        self._status_timer.start(800)

    # ── UI BUILD ─────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        # ── Outer horizontal split: [library | center | props] ───
        outer = QSplitter(Qt.Horizontal)
        outer.setHandleWidth(1)
        outer.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(outer, 1)

        # Left — tool library
        self._tool_lib = ToolLibraryPanel()
        outer.addWidget(self._tool_lib)

        # Center — vertical split: [canvas+viewer | results]
        center_split = QSplitter(Qt.Vertical)
        center_split.setHandleWidth(1)
        center_split.setStyleSheet("QSplitter::handle{background:#1e2d45;}")

        # Top center — tabs: Canvas | Image Viewer
        self._center_tabs = QTabWidget()
        self._center_tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0a0e1a;}
            QTabBar::tab{background:#060a14;color:#64748b;
                         padding:7px 16px;border:none;
                         font-size:12px;font-weight:600;border-right:1px solid #1e2d45;}
            QTabBar::tab:selected{color:#00d4ff;background:#0a0e1a;
                                  border-bottom:2px solid #00d4ff;}
            QTabBar::tab:hover{color:#e2e8f0;}
        """)

        self._canvas = AOICanvas(self._graph)
        self._center_tabs.addTab(self._canvas, "🔧  Pipeline Canvas")

        self._img_viewer = ImageViewerPanel()
        self._img_viewer.set_graph(self._graph)
        self._center_tabs.addTab(self._img_viewer, "👁  Image Viewer")

        center_split.addWidget(self._center_tabs)

        # Bottom — results
        self._results = ResultsPanel()
        center_split.addWidget(self._results)
        center_split.setSizes([680, 200])
        outer.addWidget(center_split)

        # Right — properties
        self._props = PropertiesPanel()
        self._props.set_graph(self._graph)
        outer.addWidget(self._props)

        outer.setSizes([240, 1020, 280])
        outer.setCollapsible(0, False)
        outer.setCollapsible(2, False)

    def _build_toolbar(self) -> QWidget:
        tb = QWidget()
        tb.setFixedHeight(50)
        tb.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(tb)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.setSpacing(6)

        logo = QLabel("⬡ AOI Vision Pro")
        logo.setStyleSheet("color:#00d4ff;font-size:15px;font-weight:700;letter-spacing:2px;")
        hl.addWidget(logo)
        hl.addWidget(_sep())

        self._btn_new   = _tb_btn("📄", "New",   "Ctrl+N")
        self._btn_open  = _tb_btn("📂", "Open",  "Ctrl+O")
        self._btn_save  = _tb_btn("💾", "Save",  "Ctrl+S")
        for b in (self._btn_new, self._btn_open, self._btn_save):
            hl.addWidget(b)

        hl.addWidget(_sep())
        self._btn_fit   = _tb_btn("⊡", "Fit",   "Fit canvas")
        self._btn_zoom1 = _tb_btn("1:1", "Zoom", "Reset zoom")
        self._btn_clear = _tb_btn("🗑", "Clear", "Clear all")
        for b in (self._btn_fit, self._btn_zoom1, self._btn_clear):
            hl.addWidget(b)

        hl.addStretch()

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedSize(160, 20)
        self._progress.hide()
        self._progress.setStyleSheet("""
            QProgressBar{background:#0a0e1a;border:1px solid #1e2d45;
                         border-radius:3px;color:#00d4ff;font-size:10px;text-align:center;}
            QProgressBar::chunk{background:#00d4ff;border-radius:3px;}
        """)
        hl.addWidget(self._progress)

        self._node_count = QLabel("0 nodes")
        self._node_count.setStyleSheet("color:#1e2d45;font-size:11px;font-family:'Courier New';")
        hl.addWidget(self._node_count)
        hl.addWidget(_sep())

        self._run_btn = QPushButton("▶   RUN")
        self._run_btn.setFixedSize(110, 34)
        self._run_btn.setStyleSheet(_RUN_STYLE)
        hl.addWidget(self._run_btn)

        sep_y = QFrame(); sep_y.setFrameShape(QFrame.VLine)
        sep_y.setStyleSheet("color:#1e2d45;")
        hl.addWidget(sep_y)

        self._btn_yolo = QPushButton("🤖  YOLO")
        self._btn_yolo.setFixedHeight(34)
        self._btn_yolo.setStyleSheet(
            "QPushButton{background:#1a0a3a;border:1px solid #9b59b6;"
            "border-radius:5px;color:#9b59b6;"
            "font-size:12px;font-weight:700;padding:0 12px;}"
            "QPushButton:hover{background:#9b59b6;color:#fff;}")
        self._btn_yolo.clicked.connect(self._open_yolo_studio)
        hl.addWidget(self._btn_yolo)
        return tb

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("File")
        for label, shortcut, slot in [
            ("New Pipeline",   "Ctrl+N", self._new_pipeline),
            ("Open Pipeline…", "Ctrl+O", self._open_pipeline),
            ("Save Pipeline",  "Ctrl+S", self._save_pipeline),
            ("Save As…", "Ctrl+Shift+S", self._save_as),
        ]:
            a = file_m.addAction(label)
            a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
        file_m.addSeparator()
        q = file_m.addAction("Quit")
        q.setShortcut(QKeySequence("Ctrl+Q"))
        q.triggered.connect(self.close)

        run_m = mb.addMenu("Run")
        a5 = run_m.addAction("Run Pipeline"); a5.setShortcut("F5")
        a5.triggered.connect(self._toggle_run)
        a6 = run_m.addAction("Stop");         a6.setShortcut("F6")
        a6.triggered.connect(self._stop_run)

        view_m = mb.addMenu("View")
        af = view_m.addAction("Fit Canvas"); af.setShortcut("F")
        af.triggered.connect(lambda: self._canvas.zoom_fit())
        ar = view_m.addAction("Reset Zoom"); ar.setShortcut("R")
        ar.triggered.connect(lambda: self._canvas.zoom_reset())
        view_m.addSeparator()
        av = view_m.addAction("Switch to Image Viewer"); av.setShortcut("Tab")
        av.triggered.connect(lambda: self._center_tabs.setCurrentIndex(
            1 if self._center_tabs.currentIndex() == 0 else 0))

        yolo_m = mb.addMenu("YOLO")
        act_yolo = yolo_m.addAction("🤖 Open YOLO Studio")
        act_yolo.setShortcut("Ctrl+Y")
        act_yolo.triggered.connect(self._open_yolo_studio)
        yolo_m.addSeparator()
        act_yolo_label = yolo_m.addAction("✏ Label Images...")
        act_yolo_label.triggered.connect(self._open_yolo_studio)

        tools_m = mb.addMenu("Tools")
        act_plc = tools_m.addAction("🔌  PLC Connection…")
        act_plc.setShortcut("Ctrl+P")
        act_plc.triggered.connect(self._open_plc_dialog)

        help_m = mb.addMenu("Help")
        help_m.addAction("About").triggered.connect(self._about)
        help_m.addAction("Shortcuts").triggered.connect(self._shortcuts)

    # ── Connect signals ───────────────────────────────────────────
    def _connect_signals(self):
        self._btn_new.clicked.connect(self._new_pipeline)
        self._btn_open.clicked.connect(self._open_pipeline)
        self._btn_save.clicked.connect(self._save_pipeline)
        self._btn_fit.clicked.connect(self._canvas.zoom_fit)
        self._btn_zoom1.clicked.connect(self._canvas.zoom_reset)
        self._btn_clear.clicked.connect(self._clear_canvas)
        self._run_btn.clicked.connect(self._toggle_run)

        scene = self._canvas.aoi_scene
        scene.node_selected.connect(self._on_node_selected)
        scene.node_deselected.connect(self._on_node_deselected)
        scene.graph_changed.connect(self._on_graph_changed)
        scene.run_single.connect(self._run_single_node)
        # Double-click → open detail dialog (open_props signal = node_selected when double-clicked)
        # node_item emits open_props via signals.open_props → connected to node_selected
        # We intercept: scene.node_selected when triggered by double-click vs single-click
        # Solution: AOIScene.node_selected is emitted for BOTH; use separate signal
        # node_item.signals.open_props → open detail dialog
        scene._signals.open_props.connect(self._open_node_detail)
        # single select only updates props panel
        scene._signals.selected.connect(self._props.show_node)

        self._props.params_changed.connect(self._on_graph_changed)

    # ── Node selection ────────────────────────────────────────────
    def _on_node_selected(self, node_id: str):
        self._props.show_node(node_id)

    def _on_node_deselected(self):
        self._props.clear()

    def _on_graph_changed(self, *_):
        n = len(self._graph.nodes)
        c = len(self._graph.connections)
        self._node_count.setText(f"{n} nodes")
        self.statusBar().clearMessage()
        # Refresh image viewer node list
        self._img_viewer.refresh_node_list()

    # ── Open node detail dialog ───────────────────────────────────
    def _open_node_detail(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return

        # Nếu dialog đã mở, bring to front
        dlg = self._detail_dialogs.get(node_id)
        if dlg and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return

        node = self._graph.nodes[node_id]

        # PatMax / PatFind → mở PatMaxDialog chuyên dụng
        # YOLO Detect → open YOLO Studio
        if node.tool.tool_id == "yolo_detect":
            initial_img = node.outputs.get("image") if node.outputs else None
            self._open_yolo_studio(initial_img)
            return

        if node.tool.tool_id in ("patmax", "patmax_align", "patfind"):
            from ui.patmax_dialog import PatMaxDialog
            dlg = PatMaxDialog(node, self._graph, self)
            dlg.run_requested.connect(self._on_detail_run)
            dlg.model_trained.connect(lambda: self._canvas.aoi_scene.refresh_node(node_id))
            dlg.finished.connect(lambda _, nid=node_id: self._detail_dialogs.pop(nid, None))
            self._detail_dialogs[node_id] = dlg
            dlg.show()
            return

        # Các tool khác → NodeDetailDialog
        dlg = NodeDetailDialog(node, self._graph, self)
        dlg.run_requested.connect(self._on_detail_run)
        dlg.finished.connect(lambda _, nid=node_id: self._detail_dialogs.pop(nid, None))
        self._detail_dialogs[node_id] = dlg
        dlg.show()

    def _on_detail_run(self, node_id: str):
        """Node chạy từ detail dialog → refresh canvas + viewer."""
        self._canvas.aoi_scene.refresh_node(node_id)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.show_node(node_id)
        self._center_tabs.setCurrentIndex(1)   # Switch to image viewer

    # ── Run pipeline ──────────────────────────────────────────────
    def _toggle_run(self):
        if self._is_running:
            self._stop_run()
        else:
            self._start_run()

    def _start_run(self):
        if not self._graph.nodes:
            self.statusBar().showMessage("No nodes to run.", 3000)
            return
        self._is_running = True
        self._run_btn.setText("■   STOP")
        self._run_btn.setStyleSheet(_STOP_STYLE)
        self._progress.show()
        self._progress.setValue(0)
        self._set_status("RUNNING", "#ffd700")

        self._worker_thread = QThread()
        self._worker = PipelineWorker(self._graph)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_run_done)
        self._worker.error.connect(self._on_run_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _stop_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._finalize_run()

    def _on_run_done(self, results: dict, dur_ms: float):
        scene = self._canvas.aoi_scene
        scene.refresh_all_nodes()
        scene.refresh_connections()
        self._results.report_run(self._graph, results, dur_ms)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.refresh_current()

        # Refresh any open detail dialogs (PatMaxDialog không có refresh_outputs)
        for nid, dlg in self._detail_dialogs.items():
            if dlg.isVisible() and hasattr(dlg, "refresh_outputs"):
                dlg.refresh_outputs()

        self._finalize_run()
        self._set_status("PASS", "#39ff14")
        self.statusBar().showMessage(
            f"Pipeline done in {dur_ms:.1f} ms  —  {len(results)} nodes", 5000)
        QTimer.singleShot(5000, lambda: self._set_status("IDLE", "#64748b"))

        # Đẩy kết quả về PLC (nếu đang connected)
        self._send_result_to_plc(results)

    def _on_run_error(self, msg: str):
        self._finalize_run()
        self._set_status("ERROR", "#ff3860")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _finalize_run(self):
        self._is_running = False
        self._run_btn.setText("▶   RUN")
        self._run_btn.setStyleSheet(_RUN_STYLE)
        self._progress.hide()

    def _run_single_node(self, node_id: str):
        node = self._graph.nodes.get(node_id)
        if not node:
            return
        inputs = {p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id == node_id:
                src = self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port] = src.outputs[conn.src_port]
        try:
            out = node.tool.process_fn(inputs, node.params)
            node.outputs = out or {}
            node.status = "pass"
            if "pass" in node.outputs:
                node.status = "pass" if node.outputs["pass"] else "fail"
            node.error_msg = ""
        except Exception as e:
            node.outputs = {}
            node.status = "error"
            node.error_msg = str(e)

        self._canvas.aoi_scene.refresh_node(node_id)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.show_node(node_id)
        self._center_tabs.setCurrentIndex(1)

    # ── File ops ──────────────────────────────────────────────────
    def _new_pipeline(self):
        if self._graph.nodes:
            r = QMessageBox.question(self, "New", "Discard current pipeline?",
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._graph = FlowGraph()
        self._current_file = None
        self._rebuild_canvas()

    def _rebuild_canvas(self):
        old = self._canvas
        idx = self._center_tabs.indexOf(old)
        self._canvas = AOICanvas(self._graph)
        scene = self._canvas.aoi_scene
        scene.node_selected.connect(self._on_node_selected)
        scene.node_deselected.connect(self._on_node_deselected)
        scene.graph_changed.connect(self._on_graph_changed)
        scene.run_single.connect(self._run_single_node)
        scene._signals.open_props.connect(self._open_node_detail)
        scene._signals.selected.connect(self._props.show_node)
        self._center_tabs.removeTab(idx)
        self._center_tabs.insertTab(idx, self._canvas, "🔧  Pipeline Canvas")
        self._center_tabs.setCurrentIndex(idx)
        old.deleteLater()
        self._props.set_graph(self._graph)
        self._props.clear()
        self._img_viewer.set_graph(self._graph)
        self._img_viewer.refresh_node_list()
        self._on_graph_changed()
        self._update_title()

    def _open_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline", "",
            "AOI Pipeline (*.aoi *.json);;All Files (*)")
        if not path:
            return
        try:
            self._graph = FlowGraph.load(path)
            self._current_file = path
            self._rebuild_canvas()
            self.statusBar().showMessage(f"Loaded: {path}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load:\n{e}")

    def _save_pipeline(self):
        if not self._current_file:
            self._save_as()
            return
        try:
            self._graph.save(self._current_file)
            self.statusBar().showMessage(f"Saved: {self._current_file}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "pipeline.aoi",
            "AOI Pipeline (*.aoi);;JSON (*.json)")
        if path:
            self._current_file = path
            self._save_pipeline()
            self._update_title()

    def _clear_canvas(self):
        r = QMessageBox.question(self, "Clear", "Remove all nodes?",
                                 QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            for nid in list(self._graph.nodes.keys()):
                self._canvas.aoi_scene._delete_node(nid)
            self._on_graph_changed()

    # ── Status helpers ────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        sb = self.statusBar()
        sb.showMessage(f"  ● {text}", 0)

    def _tick(self):
        zoom = int(getattr(self._canvas, '_zoom', 1.0) * 100)
        n = len(self._graph.nodes)
        c = len(self._graph.connections)
        self.statusBar().showMessage(
            f"  Nodes: {n}   Connections: {c}   Zoom: {zoom}%", 0)

    def _update_title(self):
        fname = os.path.basename(self._current_file) if self._current_file else "Untitled"
        self.setWindowTitle(f"AOI Vision Pro — {fname}")

    # ── About ─────────────────────────────────────────────────────
    def _open_yolo_studio(self, initial_image=None):
        """Mở YOLO Studio dialog."""
        from ui.yolo_studio import YoloStudioDialog
        # Lấy ảnh hiện tại từ node đang chọn (nếu có)
        if initial_image is None and self._current_node_id_for_yolo():
            node = self._graph.nodes.get(self._current_node_id_for_yolo())
            if node:
                initial_image = node.outputs.get("image")
        dlg = YoloStudioDialog(self, initial_image)
        dlg.model_trained.connect(self._on_yolo_model_trained)
        dlg.show()

    def _current_node_id_for_yolo(self):
        """Trả về node_id đang được chọn (nếu có)."""
        for nid, node in self._graph.nodes.items():
            if node.status in ("pass","fail") and "image" in node.outputs:
                return nid
        return None

    def _on_yolo_model_trained(self, model_path: str):
        """Khi YOLO Studio train xong — tự tạo node YOLO Detect với model đó."""
        from core.tool_registry import TOOL_BY_ID
        if "yolo_detect" not in TOOL_BY_ID:
            return
        # Thêm node YOLO Detect vào canvas ở giữa
        scene = self._canvas.aoi_scene
        import random
        pos_x = random.randint(400, 800)
        pos_y = random.randint(100, 400)
        from PySide6.QtCore import QPointF
        node_item = scene.add_node("yolo_detect", QPointF(pos_x, pos_y))
        node_item.node.params["model_path"] = model_path
        self._on_graph_changed()
        self.statusBar().showMessage(
            f"✅ YOLO model added: {model_path}", 5000)

    # ── PLC ───────────────────────────────────────────────────────
    def _open_plc_dialog(self):
        from ui.plc_dialog import PLCDialog
        if self._plc_dialog and self._plc_dialog.isVisible():
            self._plc_dialog.raise_()
            self._plc_dialog.activateWindow()
            return
        self._plc_dialog = PLCDialog(self._plc_manager, self)
        self._plc_dialog.trigger_fired.connect(self._on_plc_trigger)
        self._plc_dialog.show()

    def _on_plc_trigger(self):
        """PLC kích hoạt → chạy pipeline (chỉ khi đang idle)."""
        if self._is_running:
            self.statusBar().showMessage("PLC trigger ignored — pipeline already running", 3000)
            return
        self._start_run()

    def _send_result_to_plc(self, results: dict):
        """Gửi PASS/FAIL + giá trị số về PLC sau khi pipeline xong."""
        if not self._plc_manager.is_connected:
            return
        # PASS nếu toàn bộ node có status không phải 'fail'/'error'
        passed = all(n.status not in ("fail", "error")
                     for n in self._graph.nodes.values())
        # Thu thập số liệu từ outputs (int/float scalar)
        values = []
        for nid, out in results.items():
            if not isinstance(out, dict):
                continue
            for v in out.values():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    values.append(v)
        try:
            self._plc_manager.write_result(passed=passed, values=values[:16])
            self.statusBar().showMessage(
                f"→ PLC: {'PASS' if passed else 'FAIL'}  ({len(values)} values)", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"PLC write error: {e}", 5000)

    def _about(self):
        QMessageBox.about(self, "AOI Vision Pro",
            "<h2 style='color:#00d4ff;'>AOI Vision Pro v1.0</h2>"
            "<p>Automated Optical Inspection<br>PySide6 + OpenCV</p>"
            "<ul><li>Drag-drop pipeline</li>"
            "<li>30+ inspection tools</li>"
            "<li>Real-time image viewer</li>"
            "<li>Pass/Fail judgment</li></ul>")

    def _shortcuts(self):
        QMessageBox.information(self, "Shortcuts",
            "F5 — Run pipeline\n"
            "F6 — Stop\n"
            "Delete — Delete selected\n"
            "Tab — Toggle Canvas/Viewer\n"
            "F — Fit canvas\n"
            "R — Reset zoom\n"
            "Scroll — Zoom\n"
            "Middle-drag — Pan canvas\n"
            "Double-click node — Open detail window\n"
            "Drag port→port — Connect nodes\n"
            "Ctrl+S/O/N — Save/Open/New")

    # ── State ─────────────────────────────────────────────────────
    def _restore_state(self):
        s = QSettings()
        g = s.value("geometry")
        if g:
            self.restoreGeometry(g)

    def closeEvent(self, event):
        QSettings().setValue("geometry", self.saveGeometry())
        if self._is_running:
            self._stop_run()
        for dlg in list(self._detail_dialogs.values()):
            dlg.close()
        try:
            self._plc_manager.disconnect()
        except Exception:
            pass
        super().closeEvent(event)
