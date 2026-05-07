"""
ui/properties_panel.py  — FIXED
Fix: QScrollArea.setWidget() xóa widget cũ → không tái dùng placeholder.
     Thay bằng _make_placeholder() tạo mới mỗi lần cần.
"""
from __future__ import annotations
from typing import Optional, Any

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QLineEdit, QSpinBox, QDoubleSpinBox,
                                QComboBox, QCheckBox, QPushButton,
                                QScrollArea, QFrame, QTabWidget, QFileDialog)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage

from core.flow_graph import FlowGraph, NodeInstance
from core.tool_registry import ToolDef, ParamDef


def _make_placeholder(text: str) -> QLabel:
    """Tạo QLabel placeholder MỚI mỗi lần — tránh C++ deleted object crash."""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color:#1e2d45; font-size:13px; padding:20px;")
    return lbl


class ParamRow(QWidget):
    value_changed = Signal(str, object)

    def __init__(self, param: ParamDef, current_value: Any, parent=None):
        super().__init__(parent)
        self.param = param
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        lbl = QLabel(param.label)
        lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(110)
        lbl.setMaximumWidth(130)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self._editor = self._build_editor(param, current_value)
        lay.addWidget(self._editor, 1)

    def _build_editor(self, p: ParamDef, val: Any) -> QWidget:
        if p.ptype == "bool":
            w = QCheckBox()
            w.setChecked(bool(val))
            w.stateChanged.connect(lambda s: self.value_changed.emit(p.name, bool(s)))
            return w

        if p.ptype == "enum":
            w = QComboBox()
            w.addItems(p.choices)
            if str(val) in p.choices:
                w.setCurrentText(str(val))
            w.currentTextChanged.connect(lambda t: self.value_changed.emit(p.name, t))
            return w

        if p.ptype == "int":
            w = QSpinBox()
            w.setMinimum(int(p.min_val) if p.min_val is not None else -999999)
            w.setMaximum(int(p.max_val) if p.max_val is not None else  999999)
            w.setSingleStep(int(p.step) if p.step else 1)
            w.setValue(int(val) if val is not None else 0)
            w.valueChanged.connect(lambda v: self.value_changed.emit(p.name, v))
            return w

        if p.ptype == "float":
            w = QDoubleSpinBox()
            w.setMinimum(float(p.min_val) if p.min_val is not None else -1e9)
            w.setMaximum(float(p.max_val) if p.max_val is not None else  1e9)
            w.setSingleStep(float(p.step) if p.step else 0.1)
            w.setDecimals(4)
            w.setValue(float(val) if val is not None else 0.0)
            w.valueChanged.connect(lambda v: self.value_changed.emit(p.name, v))
            return w

        # str
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        le = QLineEdit(str(val) if val else "")
        le.setPlaceholderText(p.label)
        le.textChanged.connect(lambda t: self.value_changed.emit(p.name, t))
        hl.addWidget(le)
        if "path" in p.name.lower():
            btn = QPushButton("…")
            btn.setFixedWidth(24)
            btn.setStyleSheet(
                "QPushButton{background:#1e2d45;border:none;border-radius:3px;color:#e2e8f0;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            btn.clicked.connect(lambda: self._browse_file(le))
            hl.addWidget(btn)
        return w

    def _browse_file(self, le: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)")
        if path:
            le.setText(path)


class NodeInfoWidget(QWidget):
    def __init__(self, node: NodeInstance, parent=None):
        super().__init__(parent)
        tool: ToolDef = node.tool
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        icon_lbl = QLabel(tool.icon)
        icon_lbl.setStyleSheet(
            f"font-size:24px; background:{tool.color};"
            f"border-radius:6px; padding:4px 8px;")
        icon_lbl.setFixedSize(44, 44)
        icon_lbl.setAlignment(Qt.AlignCenter)
        top.addWidget(icon_lbl)

        info_lay = QVBoxLayout()
        name_lbl = QLabel(tool.name)
        name_lbl.setStyleSheet("color:#e2e8f0; font-size:14px; font-weight:700;")
        cat_lbl = QLabel(f"Category: {tool.category}")
        cat_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        id_lbl = QLabel(f"Node ID: {node.node_id}")
        id_lbl.setStyleSheet("color:#1e2d45; font-size:10px; font-family:'Courier New';")
        info_lay.addWidget(name_lbl)
        info_lay.addWidget(cat_lbl)
        info_lay.addWidget(id_lbl)
        top.addLayout(info_lay, 1)
        lay.addLayout(top)

        desc = QLabel(tool.description)
        desc.setStyleSheet("color:#94a3b8; font-size:11px; padding:6px 0;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)

        ports_lbl = QLabel("PORTS")
        ports_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:1.5px;")
        lay.addWidget(ports_lbl)

        for p in tool.inputs:
            r = QLabel(f"  ⬤  IN  •  {p.name}  [{p.data_type}]"
                       f"{'  (opt)' if not p.required else ''}")
            r.setStyleSheet("color:#00b4d8; font-size:11px; font-family:'Courier New';")
            lay.addWidget(r)
        for p in tool.outputs:
            r = QLabel(f"  ⬤  OUT  •  {p.name}  [{p.data_type}]")
            r.setStyleSheet("color:#ff8c42; font-size:11px; font-family:'Courier New';")
            lay.addWidget(r)

        lay.addStretch()


class OutputsWidget(QWidget):
    def __init__(self, node: NodeInstance, parent=None):
        super().__init__(parent)
        self._node = node
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 10, 12, 10)
        self._lay.setSpacing(6)
        self._build()

    def _build(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        node = self._node
        status_colors = {
            "idle": "#64748b", "running": "#ffd700",
            "pass": "#39ff14", "fail": "#ff3860", "error": "#ff3860"
        }
        sc = status_colors.get(node.status, "#64748b")
        st = QLabel(f"Status: {node.status.upper()}")
        st.setStyleSheet(f"color:{sc}; font-size:13px; font-weight:700;")
        self._lay.addWidget(st)

        if node.error_msg:
            err = QLabel(f"Error: {node.error_msg}")
            err.setStyleSheet("color:#ff3860; font-size:11px;")
            err.setWordWrap(True)
            self._lay.addWidget(err)

        if not node.outputs:
            lbl = QLabel("No output yet.\nRun the pipeline first.")
            lbl.setStyleSheet("color:#1e2d45; font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter)
            self._lay.addWidget(lbl)
            self._lay.addStretch()
            return

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        self._lay.addWidget(sep)

        for key, val in node.outputs.items():
            if key == "image":
                continue
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)

            k = QLabel(key)
            k.setStyleSheet("color:#64748b; font-size:11px; font-family:'Courier New';")
            k.setMinimumWidth(100)
            rl.addWidget(k)

            if isinstance(val, bool):
                v_txt = "✔ TRUE" if val else "✖ FALSE"
                v_col = "#39ff14" if val else "#ff3860"
            elif isinstance(val, float):
                v_txt = f"{val:.5f}"; v_col = "#00d4ff"
            elif isinstance(val, int):
                v_txt = str(val); v_col = "#00d4ff"
            elif isinstance(val, str):
                v_txt = val[:60]; v_col = "#e2e8f0"
            elif isinstance(val, list):
                v_txt = f"[list: {len(val)} items]"; v_col = "#ff8c42"
            else:
                v_txt = type(val).__name__; v_col = "#64748b"

            v = QLabel(v_txt)
            v.setStyleSheet(f"color:{v_col}; font-size:11px; font-weight:600;")
            v.setWordWrap(True)
            rl.addWidget(v, 1)
            self._lay.addWidget(row)

        self._lay.addStretch()


class ImagePreviewWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background:#000; border:1px solid #1e2d45; border-radius:6px;"
            "color:#1e2d45; font-size:12px;")
        self.setMinimumHeight(180)
        self.setText("No Image")
        self._current_image = None

    def set_image(self, img_array):
        if img_array is None:
            self.setText("No Image")
            self._current_image = None
            return
        import cv2
        arr = img_array.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

        h, w, ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pw = max(1, self.width() - 4)
        ph = max(1, self.height() - 4)
        scaled = pix.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self._current_image = img_array

    def resizeEvent(self, event):
        if self._current_image is not None:
            self.set_image(self._current_image)
        super().resizeEvent(event)


class PropertiesPanel(QWidget):
    params_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)
        self._current_node_id: Optional[str] = None
        self._graph: Optional[FlowGraph] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._title = QLabel("⚙  PROPERTIES")
        self._title.setStyleSheet("""
            background:#060a14; color:#00d4ff;
            font-size:11px; font-weight:700; letter-spacing:2px;
            padding:10px 12px; border-bottom:1px solid #1e2d45;
        """)
        lay.addWidget(self._title)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border:none; background:#0d1220; }
            QTabBar::tab {
                background:#0a0e1a; color:#64748b;
                padding:6px 10px; border:none;
                font-size:11px; font-weight:600;
            }
            QTabBar::tab:selected { color:#00d4ff; border-bottom:2px solid #00d4ff; }
            QTabBar::tab:hover { color:#e2e8f0; }
        """)
        lay.addWidget(self._tabs)

        # ── Tab: Info ──────────────────────────────────────────────
        self._info_scroll = QScrollArea()
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setFrameShape(QFrame.NoFrame)
        self._info_scroll.setWidget(_make_placeholder("Select a node\nto view properties."))
        self._tabs.addTab(self._info_scroll, "Info")

        # ── Tab: Params ────────────────────────────────────────────
        self._params_scroll = QScrollArea()
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(_make_placeholder("Select a node\nto edit parameters."))
        self._tabs.addTab(self._params_scroll, "Params")

        # ── Tab: Output ────────────────────────────────────────────
        self._out_scroll = QScrollArea()
        self._out_scroll.setWidgetResizable(True)
        self._out_scroll.setFrameShape(QFrame.NoFrame)
        self._out_scroll.setWidget(_make_placeholder("Run pipeline to\nsee outputs."))
        self._tabs.addTab(self._out_scroll, "Output")

        # ── Tab: Preview ───────────────────────────────────────────
        preview_tab = QWidget()
        pl = QVBoxLayout(preview_tab)
        pl.setContentsMargins(8, 8, 8, 8)
        self._img_preview = ImagePreviewWidget()
        pl.addWidget(self._img_preview)
        self._prev_info = QLabel("")
        self._prev_info.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New';")
        self._prev_info.setAlignment(Qt.AlignCenter)
        self._prev_info.setWordWrap(True)
        pl.addWidget(self._prev_info)
        pl.addStretch()
        self._tabs.addTab(preview_tab, "Preview")

        # ── Bottom hint ────────────────────────────────────────────
        self._empty_lbl = QLabel("Click a node to inspect it")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color:#1e2d45; font-size:11px; padding:8px;")
        lay.addWidget(self._empty_lbl)

    # ── Public API ────────────────────────────────────────────────
    def set_graph(self, graph: FlowGraph):
        self._graph = graph

    def show_node(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return
        self._current_node_id = node_id
        node = self._graph.nodes[node_id]
        self._title.setText(f"⚙  {node.tool.name}")
        self._empty_lbl.hide()

        self._info_scroll.setWidget(NodeInfoWidget(node))
        self._build_params_tab(node)
        self._refresh_outputs_tab(node)
        self._refresh_preview(node)

    def refresh_outputs(self):
        if not self._current_node_id or not self._graph:
            return
        if self._current_node_id not in self._graph.nodes:
            return
        node = self._graph.nodes[self._current_node_id]
        self._refresh_outputs_tab(node)
        self._refresh_preview(node)

    def clear(self):
        """Reset về trạng thái rỗng — luôn tạo placeholder MỚI."""
        self._current_node_id = None
        self._title.setText("⚙  PROPERTIES")
        self._empty_lbl.show()
        # Tạo widget mới mỗi lần — KHÔNG tái dùng object cũ đã bị Qt xóa
        self._info_scroll.setWidget(
            _make_placeholder("Select a node\nto view properties."))
        self._params_scroll.setWidget(
            _make_placeholder("Select a node\nto edit parameters."))
        self._out_scroll.setWidget(
            _make_placeholder("Run pipeline to\nsee outputs."))
        self._img_preview.set_image(None)
        self._prev_info.setText("")

    # ── Internal ──────────────────────────────────────────────────
    def _build_params_tab(self, node: NodeInstance):
        tool = node.tool
        if not tool.params:
            self._params_scroll.setWidget(
                _make_placeholder("No parameters\nfor this tool."))
            return

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        for param in tool.params:
            pr = ParamRow(param, node.params.get(param.name, param.default))
            pr.value_changed.connect(
                lambda name, val, nid=node.node_id:
                    self._on_param_changed(nid, name, val))
            cl.addWidget(pr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        cl.addWidget(sep)
        note = QLabel("Changes apply on next run  ▶")
        note.setStyleSheet("color:#1e2d45; font-size:10px;")
        note.setAlignment(Qt.AlignCenter)
        cl.addWidget(note)
        cl.addStretch()
        self._params_scroll.setWidget(container)

    def _on_param_changed(self, node_id: str, name: str, value: Any):
        if self._graph and node_id in self._graph.nodes:
            self._graph.nodes[node_id].params[name] = value
            self.params_changed.emit(node_id)

    def _refresh_outputs_tab(self, node: NodeInstance):
        self._out_scroll.setWidget(OutputsWidget(node))

    def _refresh_preview(self, node: NodeInstance):
        img = node.outputs.get("image")
        if img is not None:
            import numpy as np
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                self._prev_info.setText(f"{w} × {h} px  |  {img.dtype}")
                self._img_preview.set_image(img)
                self._tabs.setCurrentIndex(3)
                return
        self._img_preview.set_image(None)
        self._prev_info.setText("")
