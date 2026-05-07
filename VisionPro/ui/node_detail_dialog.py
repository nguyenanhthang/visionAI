"""
ui/node_detail_dialog.py — v5
Fix crop_roi:
  - Không có port kết nối → kéo chuột vẽ thủ công → lưu _drawn_roi
  - Có port kết nối → hiển thị readonly, vẽ rect từ port value
  - Mode hint thông minh hiển thị đang dùng mode nào
"""
from __future__ import annotations
from typing import Optional, Any, Tuple
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QTabWidget, QWidget, QScrollArea, QFrame,
                                QSplitter, QPushButton, QGroupBox,
                                QSizePolicy, QApplication, QFileDialog,
                                QMessageBox)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
                            QPen, QBrush, QCursor, QMouseEvent)

from core.flow_graph import NodeInstance, FlowGraph
# PatMaxDialog imported lazily to avoid circular import
from core.tool_registry import ToolDef, ParamDef
from ui.properties_panel import ParamRow


# ════════════════════════════════════════════════════════════════════
#  Interactive image label
# ════════════════════════════════════════════════════════════════════
class InteractiveImageLabel(QLabel):
    """
    Hiển thị ảnh + overlay tương tác.
    mode="roi"      → kéo chuột chọn vùng (màu cyan)
    mode="template" → kéo chuột chọn template (màu cam)
    mode="pick"     → click lấy pixel color
    mode="view"     → chỉ xem, không tương tác
    mode="readonly" → xem + hiển thị rect cố định (port connected)
    """
    roi_changed    = Signal(int, int, int, int)
    pixel_picked   = Signal(int, int)
    template_drawn = Signal(int, int, int, int)

    def __init__(self, mode="view", parent=None):
        super().__init__(parent)
        self.mode        = mode
        self._arr        = None
        self._scale      = 1.0
        self._off_x      = 0
        self._off_y      = 0
        self._rect: Optional[QRect] = None
        self._drag_start: Optional[QPoint] = None
        self._dragging   = False
        self._pick_pos: Optional[Tuple[int,int]] = None
        self._readonly_rect: Optional[Tuple[int,int,int,int]] = None

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:6px;")

        cur_map = {
            "roi": Qt.CrossCursor, "template": Qt.CrossCursor,
            "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
            "readonly": Qt.ArrowCursor,
        }
        self.setCursor(QCursor(cur_map.get(mode, Qt.ArrowCursor)))

    # ── Image ──────────────────────────────────────────────────────
    def set_image(self, arr: Optional[np.ndarray]):
        self._arr = arr
        self._render()

    def set_rect_from_params(self, x, y, w, h):
        """Hiển thị rect khởi tạo từ params/port (image coords)."""
        if self._arr is None:
            return
        ih, iw = self._arr.shape[:2]
        x = max(0, min(int(x), iw-1))
        y = max(0, min(int(y), ih-1))
        w = max(1, min(int(w), iw-x))
        h = max(1, min(int(h), ih-y))
        wx, wy = self._img_to_widget(x, y)
        ww = int(w * self._scale)
        wh = int(h * self._scale)
        self._rect = QRect(wx, wy, ww, wh)
        self._render()

    def set_readonly_rect(self, x, y, w, h):
        """Hiển thị rect từ port (không cho phép kéo thay đổi)."""
        self._readonly_rect = (x, y, w, h)
        self._render()

    def _img_to_widget(self, ix, iy):
        return int(ix * self._scale + self._off_x), int(iy * self._scale + self._off_y)

    def _widget_to_img(self, wx, wy):
        if self._scale == 0:
            return 0, 0
        return int((wx - self._off_x) / self._scale), int((wy - self._off_y) / self._scale)

    def _render(self):
        if self._arr is None:
            self.setText("No Image\nRun pipeline first or load image source.")
            return
        import cv2
        arr = self._arr.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        h, w, ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        pw = max(1, self.width() - 4)
        ph = max(1, self.height() - 4)
        sx = pw / w; sy = ph / h
        self._scale = min(sx, sy)
        dw = int(w * self._scale); dh = int(h * self._scale)
        self._off_x = (self.width() - dw) // 2
        self._off_y = (self.height() - dh) // 2

        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(5, 8, 16))
        p = QPainter(canvas)
        p.drawPixmap(self._off_x, self._off_y, dw, dh, pix)

        # Readonly rect (port connected) — màu xanh lá
        if self._readonly_rect:
            rx, ry, rw, rh = self._readonly_rect
            wx, wy = self._img_to_widget(rx, ry)
            ww = int(rw * self._scale); wh = int(rh * self._scale)
            col = QColor(57, 255, 20)  # bright green
            p.setPen(QPen(col, 2, Qt.SolidLine))
            p.setBrush(QBrush(QColor(57, 255, 20, 25)))
            p.drawRect(wx, wy, ww, wh)
            p.setPen(QPen(col))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(wx + 4, wy - 6, f"[PORT] ({rx},{ry}) {rw}×{rh}")

        # Interactive / drawn rect — màu cyan (roi) hoặc cam (template)
        draw_rect = self._rect
        if draw_rect and not draw_rect.isNull():
            if self.mode == "template":
                col = QColor(255, 140, 50)
            else:
                col = QColor(0, 212, 255)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 28)))
            p.drawRect(draw_rect)
            # Corner squares
            p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
            for cx, cy in [(draw_rect.left(), draw_rect.top()),
                           (draw_rect.right(), draw_rect.top()),
                           (draw_rect.left(), draw_rect.bottom()),
                           (draw_rect.right(), draw_rect.bottom())]:
                p.drawRect(cx - 4, cy - 4, 8, 8)
            ix, iy = self._widget_to_img(draw_rect.left(), draw_rect.top())
            iw2 = int(draw_rect.width() / self._scale)
            ih2 = int(draw_rect.height() / self._scale)
            p.setPen(QPen(col)); p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(draw_rect.left() + 4, draw_rect.top() - 6,
                       f"({ix},{iy})  {iw2}×{ih2}")

        # Pixel pick marker
        if self._pick_pos and self.mode == "pick":
            px2, py2 = self._img_to_widget(*self._pick_pos)
            p.setPen(QPen(QColor(255, 255, 0), 2))
            p.drawLine(px2 - 10, py2, px2 + 10, py2)
            p.drawLine(px2, py2 - 10, px2, py2 + 10)
            p.drawEllipse(px2 - 6, py2 - 6, 12, 12)

        p.end()
        self.setPixmap(canvas)

    # ── Mouse ──────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        if self.mode == "pick":
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            if self._arr is not None:
                h2, w2 = self._arr.shape[:2]
                ix = max(0, min(ix, w2-1)); iy = max(0, min(iy, h2-1))
                self._pick_pos = (ix, iy)
                self._render()
                self.pixel_picked.emit(ix, iy)
        elif self.mode in ("roi", "template"):
            self._drag_start = pos
            self._dragging   = True
            self._rect = QRect(pos, QSize(0, 0))

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._dragging or self._drag_start is None:
            return
        self._rect = QRect(self._drag_start, event.position().toPoint()).normalized()
        self._render()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._dragging:
            return
        self._dragging = False
        if self._rect and self._rect.width() > 4 and self._rect.height() > 4:
            ix, iy = self._widget_to_img(self._rect.left(), self._rect.top())
            iw2 = max(1, int(self._rect.width() / self._scale))
            ih2 = max(1, int(self._rect.height() / self._scale))
            if self._arr is not None:
                H2, W2 = self._arr.shape[:2]
                ix  = max(0, min(ix, W2-1)); iy  = max(0, min(iy, H2-1))
                iw2 = max(1, min(iw2, W2-ix)); ih2 = max(1, min(ih2, H2-iy))
            self.roi_changed.emit(ix, iy, iw2, ih2)
            if self.mode == "template":
                self.template_drawn.emit(ix, iy, iw2, ih2)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ════════════════════════════════════════════════════════════════════
#  Main Dialog
# ════════════════════════════════════════════════════════════════════
class NodeDetailDialog(QDialog):
    run_requested = Signal(str)

    def __init__(self, node: NodeInstance, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._node  = node
        self._graph = graph
        tool: ToolDef = node.tool

        self.setWindowTitle(f"{tool.icon}  {tool.name}  —  Detail")
        self.setMinimumSize(1000, 650)
        self.resize(1120, 740)
        self.setModal(False)
        self.setStyleSheet("""
            QDialog { background:#0a0e1a; color:#e2e8f0; }
            QGroupBox { border:1px solid #1e2d45; border-radius:6px;
                        margin-top:8px; padding-top:8px;
                        color:#64748b; font-size:11px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QScrollArea { border:none; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(54)
        hdr.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {tool.color},stop:1 #0a0e1a);"
            f"border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16, 0, 16, 0)

        icon_l = QLabel(tool.icon)
        icon_l.setStyleSheet("font-size:28px; background:transparent;")
        hl.addWidget(icon_l)

        tc = QVBoxLayout()
        t1 = QLabel(tool.name)
        t1.setStyleSheet("color:#fff; font-size:16px; font-weight:700; background:transparent;")
        cog = f"  {tool.cognex_equiv}" if tool.cognex_equiv else ""
        t2 = QLabel(f"{tool.category}{cog}  •  {tool.description}")
        t2.setStyleSheet("color:#ffffff88; font-size:11px; background:transparent;")
        tc.addWidget(t1); tc.addWidget(t2)
        hl.addLayout(tc, 1)

        self._run_btn = QPushButton("▶  Run Node")
        self._run_btn.setFixedSize(120, 34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#00d4ff;border:none;border-radius:5px;"
            "color:#000;font-weight:700;font-size:13px;}"
            "QPushButton:hover{background:#33ddff;}"
            "QPushButton:pressed{background:#0099bb;}")
        self._run_btn.clicked.connect(self._on_run)
        hl.addWidget(self._run_btn)
        root.addWidget(hdr)

        # ── Mode hint bar ─────────────────────────────────────────
        self._mode_hint = QLabel("")
        self._mode_hint.setStyleSheet(
            "background:#0d1a2a; color:#ffd700; font-size:11px;"
            "padding:5px 16px; border-bottom:1px solid #1e2d45;")
        self._mode_hint.hide()
        root.addWidget(self._mode_hint)

        # ── Splitter ──────────────────────────────────────────────
        spl = QSplitter(Qt.Horizontal)
        spl.setHandleWidth(1)
        spl.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(spl, 1)

        # LEFT — params
        left = QWidget(); left.setMaximumWidth(300); left.setMinimumWidth(240)
        ll   = QVBoxLayout(left); ll.setContentsMargins(10, 10, 10, 10); ll.setSpacing(8)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#0a0e1a;color:#64748b;padding:6px 10px;
                         border:none;font-size:11px;font-weight:600;}
            QTabBar::tab:selected{color:#00d4ff;border-bottom:2px solid #00d4ff;}
        """)
        params_scroll = QScrollArea(); params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.NoFrame)
        params_scroll.setWidget(self._build_params_widget())
        tabs.addTab(params_scroll, "⚙ Params")
        tabs.addTab(self._build_ports_widget(), "🔌 Ports")
        ll.addWidget(tabs)

        self._out_group = QGroupBox("Output Values")
        og = QVBoxLayout(self._out_group)
        og.setContentsMargins(8, 12, 8, 8); og.setSpacing(4)
        self._out_labels = {}
        self._build_output_labels(og)
        ll.addWidget(self._out_group)

        self._status_lbl = QLabel("Status: IDLE")
        self._status_lbl.setStyleSheet("color:#64748b; font-size:11px; padding:2px;")
        ll.addWidget(self._status_lbl)
        spl.addWidget(left)

        # RIGHT — image
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6, 6, 6, 6); rl.setSpacing(4)

        img_hdr = QHBoxLayout()
        img_lbl = QLabel("OUTPUT IMAGE")
        img_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        img_hdr.addWidget(img_lbl); img_hdr.addStretch()

        # ── Chọn mode interactive theo tool ──────────────────────
        self._roi_port_connected = False

        if tool.tool_id == "crop_roi":
            self._roi_port_connected = self._check_roi_ports_connected()
            if self._roi_port_connected:
                # Port đang kết nối → readonly, hiển thị rect từ port
                mode_str = "readonly"
                self._mode_hint.setText(
                    "🔗  Port x/y/w/h đang được kết nối — ROI tự động theo upstream tool.")
            else:
                # Không kết nối → vẽ thủ công
                mode_str = "roi"
                self._mode_hint.setText(
                    "✏  Kéo chuột trên ảnh để vẽ vùng ROI thủ công  "
                    "—  kết nối port x/y/w/h để tracking tự động.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode=mode_str)
            if mode_str == "roi":
                self._img_label.roi_changed.connect(self._on_roi_changed)
                # Init rect từ _drawn_roi hoặc params
                drawn = node.params.get("_drawn_roi")
                if drawn:
                    x2, y2, w2, h2 = drawn
                else:
                    x2 = node.params.get("x", 0); y2 = node.params.get("y", 0)
                    w2 = node.params.get("crop_w", 320); h2 = node.params.get("crop_h", 240)
                QTimer.singleShot(120, lambda: self._img_label.set_rect_from_params(x2, y2, w2, h2))

        elif tool.tool_id in ("patmax", "patfind"):
            # PatMax/PatFind → mở PatMaxDialog chuyên dụng
            self._mode_hint.setText(
                "🎯  PatMax/PatFind — Cửa sổ Train & Search chuyên dụng đang mở...")
            self._mode_hint.setStyleSheet(
                "background:#0d1a2a; color:#00d4ff; font-size:11px;"
                "padding:5px 16px; border-bottom:1px solid #1e2d45;")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="view")
            # Mở PatMaxDialog sau khi dialog này xuất hiện
            QTimer.singleShot(150, self._open_patmax_dialog)

        elif tool.tool_id == "color_picker":
            self._mode_hint.setText(
                "🎨  Click chuột vào ảnh để lấy màu tại điểm đó.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="pick")
            self._img_label.pixel_picked.connect(self._on_pixel_picked)

        else:
            self._img_label = InteractiveImageLabel(mode="view")

        img_hdr.addWidget(QWidget())   # spacer placeholder
        rl.addLayout(img_hdr)
        rl.addWidget(self._img_label, 1)

        self._img_info = QLabel("")
        self._img_info.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New'; padding:2px;")
        self._img_info.setAlignment(Qt.AlignCenter)
        rl.addWidget(self._img_info)

        self._pixel_bar = QLabel("")
        self._pixel_bar.setStyleSheet(
            "color:#ffd700; font-size:11px; font-family:'Courier New';"
            "background:#111827; border-radius:4px; padding:3px 10px;")
        self._pixel_bar.hide()
        rl.addWidget(self._pixel_bar)

        spl.addWidget(right)
        spl.setSizes([270, 830])

        cb = QPushButton("Close")
        cb.setFixedHeight(30)
        cb.setStyleSheet(
            "QPushButton{background:#1e2d45;border:none;border-radius:4px;"
            "color:#94a3b8;font-size:12px;margin:4px 12px;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        cb.clicked.connect(self.close)
        root.addWidget(cb)

        self.refresh_outputs()

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════
    def _check_roi_ports_connected(self) -> bool:
        """Trả về True nếu ít nhất một port x/y/w/h được kết nối."""
        node = self._node
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id and conn.dst_port in ("x","y","w","h"):
                return True
        return False

    def _get_input_image(self) -> Optional[np.ndarray]:
        for conn in self._graph.connections:
            if conn.dst_id == self._node.node_id and conn.dst_port == "image":
                src = self._graph.nodes.get(conn.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    # ════════════════════════════════════════════════════════════════
    #  Build sub-widgets
    # ════════════════════════════════════════════════════════════════
    def _build_params_widget(self) -> QWidget:
        node = self._node; tool = node.tool
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(5)
        if not tool.params:
            lbl = QLabel("No parameters.")
            lbl.setStyleSheet("color:#1e2d45; font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter); lay.addWidget(lbl)
        else:
            self._param_rows = {}
            for param in tool.params:
                pr = ParamRow(param, node.params.get(param.name, param.default))
                if param.tooltip:
                    pr.setToolTip(param.tooltip)
                pr.value_changed.connect(
                    lambda name, val, nid=node.node_id: self._on_param(nid, name, val))
                lay.addWidget(pr)
                self._param_rows[param.name] = pr
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)
        note = QLabel("▶ Run Node để áp dụng thay đổi")
        note.setStyleSheet("color:#1e2d45; font-size:10px;")
        note.setAlignment(Qt.AlignCenter)
        lay.addWidget(note); lay.addStretch()
        return w

    def _build_ports_widget(self) -> QWidget:
        tool = self._node.tool; w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)

        def section(label, color):
            h = QLabel(label)
            h.setStyleSheet(f"color:{color}; font-size:10px; font-weight:700; "
                            f"letter-spacing:1.5px; margin-top:4px;")
            lay.addWidget(h)

        if tool.inputs:
            section("INPUTS", "#00d4ff")
            for p in tool.inputs:
                connected = any(c.dst_id == self._node.node_id and c.dst_port == p.name
                                for c in self._graph.connections)
                status = "🔗" if connected else ("○" if not p.required else "●")
                r = QLabel(f"  {status}  {p.name}  [{p.data_type}]"
                           f"{'  (opt)' if not p.required else ''}")
                col = "#39ff14" if connected else "#00b4d8"
                r.setStyleSheet(
                    f"color:{col}; font-size:11px; font-family:'Courier New';"
                    f"background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        if tool.outputs:
            section("OUTPUTS", "#ff8c42")
            for p in tool.outputs:
                r = QLabel(f"  ⬤  {p.name}  [{p.data_type}]")
                r.setStyleSheet(
                    "color:#ff8c42; font-size:11px; font-family:'Courier New';"
                    "background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        # Crop ROI special: show connection status
        if tool.tool_id == "crop_roi":
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#1e2d45;"); lay.addWidget(sep)
            ports_status = []
            for pname in ("x","y","w","h"):
                conn = any(c.dst_id == self._node.node_id and c.dst_port == pname
                           for c in self._graph.connections)
                ports_status.append(f"{pname}:{'🔗' if conn else '✏'}")
            note = QLabel("  ".join(ports_status))
            note.setStyleSheet(
                "color:#ffd700; font-size:11px; font-family:'Courier New';"
                "padding:4px 6px; background:#0d1a2a; border-radius:4px;")
            lay.addWidget(note)

        lay.addStretch()
        return w

    def _build_output_labels(self, layout):
        self._out_labels = {}
        for port in self._node.tool.outputs:
            if port.name == "image":
                continue
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
            k = QLabel(port.name)
            k.setStyleSheet("color:#64748b; font-size:11px; font-family:'Courier New';")
            k.setMinimumWidth(80)
            v = QLabel("—")
            v.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            rl.addWidget(k); rl.addWidget(v, 1)
            layout.addWidget(row)
            self._out_labels[port.name] = v

    # ════════════════════════════════════════════════════════════════
    #  Interactive callbacks
    # ════════════════════════════════════════════════════════════════
    def _on_roi_changed(self, x, y, w, h):
        """
        Kéo chuột vẽ ROI thủ công → lưu vào _drawn_roi.
        Không ghi đè x/y/crop_w/crop_h (đó là params spinbox).
        proc_crop sẽ đọc _drawn_roi khi không có port kết nối.
        """
        node = self._node
        node.params["_drawn_roi"] = (x, y, w, h)

        # Cập nhật spinbox params để hiển thị (nhưng _drawn_roi là nguồn truth)
        for name, val in [("x", x), ("y", y), ("crop_w", w), ("crop_h", h)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)

        # Cập nhật info label
        self._img_info.setText(
            f"Manual ROI: ({x},{y})  {w}×{h} px  —  ▶ Run Node để crop")
        self._img_info.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _open_patmax_dialog(self):
        """Mở cửa sổ PatMax chuyên dụng."""
        from ui.patmax_dialog import PatMaxDialog
        # Nếu đã mở rồi, bring to front
        existing = getattr(self, '_patmax_dlg', None)
        if existing and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        dlg = PatMaxDialog(self._node, self._graph, self)
        dlg.run_requested.connect(self.run_requested)
        dlg.model_trained.connect(self._on_patmax_model_trained)
        self._patmax_dlg = dlg
        dlg.show()
        # Update mode hint
        self._mode_hint.setText("🎯  PatMax dialog mở — Train & Search tại đó.")

    def _on_patmax_model_trained(self):
        """Callback khi PatMax train xong — refresh image preview."""
        self.refresh_outputs()

    def _on_template_drawn(self, x, y, w, h):
        """Vẽ ROI → cắt template → lưu vào params."""
        node = self._node
        img = node.outputs.get("image") or self._get_input_image()
        if img is None:
            QMessageBox.warning(self, "Template",
                                "Cần ảnh để cắt template.\n"
                                "Kết nối node Image Source và Run Node trước.")
            return
        H2, W2 = img.shape[:2]
        x = max(0, min(x, W2-1)); y = max(0, min(y, H2-1))
        w = max(1, min(w, W2-x)); h = max(1, min(h, H2-y))
        templ = img[y:y+h, x:x+w]
        node.params["_template_array"] = templ
        node.params["_template_rect"]  = (x, y, w, h)
        self._img_info.setText(
            f"✔ Template saved: ({x},{y})  {w}×{h} px  —  ▶ Run Node")
        self._img_info.setStyleSheet(
            "color:#ffd700; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _on_pixel_picked(self, x, y):
        """Click → lấy màu pixel."""
        node = self._node
        node.params["pick_x"] = x; node.params["pick_y"] = y
        for name, val in [("pick_x", x), ("pick_y", y)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        self._pixel_bar.show()
        self._on_run()

    def _on_param(self, node_id, name, value):
        if self._graph and node_id in self._graph.nodes:
            self._graph.nodes[node_id].params[name] = value

    # ════════════════════════════════════════════════════════════════
    #  Run
    # ════════════════════════════════════════════════════════════════
    def _on_run(self):
        node = self._node
        # Build inputs: defaults + upstream outputs
        inputs = {p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id:
                src = self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port] = src.outputs[conn.src_port]

        try:
            out = node.tool.process_fn(inputs, node.params)
            node.outputs  = out or {}
            node.status   = "pass"
            if "pass" in node.outputs:
                node.status = "pass" if node.outputs["pass"] else "fail"
            node.error_msg = ""
        except Exception as e:
            node.outputs  = {}
            node.status   = "error"
            node.error_msg = str(e)

        self.refresh_outputs()
        self.run_requested.emit(node.node_id)

    # ════════════════════════════════════════════════════════════════
    #  Refresh
    # ════════════════════════════════════════════════════════════════
    def refresh_outputs(self):
        node = self._node
        sc = {"pass":"#39ff14","fail":"#ff3860","error":"#ff3860",
              "idle":"#64748b","running":"#ffd700"}.get(node.status, "#64748b")
        self._status_lbl.setText(f"Status: {node.status.upper()}")
        self._status_lbl.setStyleSheet(
            f"color:{sc}; font-size:12px; font-weight:700; padding:2px;")

        if node.error_msg:
            self._img_info.setText(f"Error: {node.error_msg}")
            self._img_info.setStyleSheet(
                "color:#ff3860; font-size:10px; font-family:'Courier New'; padding:2px;")

        # Scalar outputs
        for name, lbl in self._out_labels.items():
            val = node.outputs.get(name)
            if val is None:
                lbl.setText("—"); lbl.setStyleSheet("color:#1e2d45; font-size:11px;")
            elif isinstance(val, bool):
                lbl.setText("✔ TRUE" if val else "✖ FALSE")
                lbl.setStyleSheet(
                    f"color:{'#39ff14' if val else '#ff3860'}; font-size:11px; font-weight:700;")
            elif isinstance(val, float):
                lbl.setText(f"{val:.5f}")
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            elif isinstance(val, int):
                lbl.setText(str(val))
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            else:
                lbl.setText(str(val)[:50])
                lbl.setStyleSheet("color:#e2e8f0; font-size:11px;")

        # Color picker
        if node.tool.tool_id == "color_picker" and node.outputs:
            r2 = node.outputs.get("r", 0); g2 = node.outputs.get("g", 0)
            b2 = node.outputs.get("b", 0); H2 = node.outputs.get("h", 0)
            S2 = node.outputs.get("s", 0); V2 = node.outputs.get("v", 0)
            self._pixel_bar.setText(
                f"  ({node.params.get('pick_x',0)}, {node.params.get('pick_y',0)})  "
                f"  RGB {r2},{g2},{b2}  #{r2:02X}{g2:02X}{b2:02X}  "
                f"  HSV {H2},{S2},{V2}")
            self._pixel_bar.setStyleSheet(
                f"color:rgb({r2},{g2},{b2}); font-size:11px; font-family:'Courier New';"
                f"background:#111827; border-radius:4px; padding:3px 10px;"
                f"border:1px solid rgb({r2},{g2},{b2});")
            self._pixel_bar.show()

        # Image
        img = node.outputs.get("image")
        if img is not None and isinstance(img, np.ndarray):
            h2, w2 = img.shape[:2]
            if not node.error_msg:
                self._img_info.setText(
                    f"{w2}×{h2} px  |  {img.dtype}  |  {node.status.upper()}")
                self._img_info.setStyleSheet(
                    f"color:{sc}; font-size:10px; font-family:'Courier New'; padding:2px;")
            self._img_label.set_image(img)

            # crop_roi: hiển thị rect
            if node.tool.tool_id == "crop_roi":
                if self._roi_port_connected:
                    # Lấy giá trị thực từ output (sau khi proc_crop chạy)
                    ox = node.outputs.get("x", 0); oy = node.outputs.get("y", 0)
                    ow = node.outputs.get("w", 0); oh = node.outputs.get("h", 0)
                    self._img_label.set_readonly_rect(ox, oy, ow, oh)
                else:
                    drawn = node.params.get("_drawn_roi")
                    if drawn:
                        self._img_label.set_rect_from_params(*drawn)

            # patmax/patfind: hiển thị output image từ engine
            elif node.tool.tool_id in ("patmax", "patfind"):
                pass   # PatMaxDialog tự quản lý display

        elif not node.error_msg:
            self._img_label.set_image(None)
