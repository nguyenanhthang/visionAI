"""
ui/node_detail_dialog.py — v4
- ROI Crop: kéo vùng bằng chuột trên ảnh → cập nhật x/y/w/h
- Template Matching: vẽ ROI trên ảnh → tự lưu làm template
- Color Picker: click chuột lấy màu trực tiếp
- Params đầy đủ với tooltip
"""
from __future__ import annotations
from typing import Optional, Any, List, Tuple
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QTabWidget, QWidget, QScrollArea, QFrame,
                                QSplitter, QPushButton, QGroupBox,
                                QSizePolicy, QSizePolicy, QApplication,
                                QSlider, QLineEdit, QFileDialog, QSpinBox,
                                QDoubleSpinBox, QComboBox, QCheckBox,
                                QGridLayout, QToolButton, QMessageBox)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
                            QPen, QBrush, QCursor, QMouseEvent,
                            QKeySequence)

from core.flow_graph import NodeInstance, FlowGraph
from core.tool_registry import ToolDef, ParamDef
from ui.properties_panel import ParamRow


# ── Interactive image widget ──────────────────────────────────────
class InteractiveImageLabel(QLabel):
    """
    QLabel hiển thị ảnh + overlay.
    Hỗ trợ:
      mode="roi"      → kéo chọn vùng ROI
      mode="template" → kéo chọn vùng template
      mode="pick"     → click lấy tọa độ pixel
    """
    roi_changed    = Signal(int, int, int, int)   # x,y,w,h
    pixel_picked   = Signal(int, int)              # x,y
    template_drawn = Signal(int, int, int, int)    # x,y,w,h

    def __init__(self, mode="roi", parent=None):
        super().__init__(parent)
        self.mode   = mode
        self._arr   = None
        self._scale = 1.0
        self._off_x = 0
        self._off_y = 0
        self._rect: Optional[QRect] = None   # in widget coords
        self._drag_start: Optional[QPoint] = None
        self._dragging  = False
        self._pick_pos: Optional[Tuple[int,int]] = None

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#050810; border:1px solid #1e2d45; border-radius:6px;")

        if mode in ("roi","template"):
            self.setCursor(QCursor(Qt.CrossCursor))
        elif mode == "pick":
            self.setCursor(QCursor(Qt.PointingHandCursor))

    def set_image(self, arr: Optional[np.ndarray], init_rect: Optional[Tuple]=None):
        self._arr = arr
        if init_rect:
            # Convert img coords → widget coords later in paintEvent
            ix,iy,iw,ih = init_rect
            self._init_img_rect = (ix,iy,iw,ih)
        else:
            self._init_img_rect = None
        self._render()

    def set_rect_from_params(self, x,y,w,h):
        """Set initial rect from params (image coordinates)."""
        self._init_img_rect = (x,y,w,h)
        self._render()

    def _img_to_widget(self, ix, iy):
        return int(ix*self._scale + self._off_x), int(iy*self._scale + self._off_y)

    def _widget_to_img(self, wx, wy):
        if self._scale == 0: return 0,0
        return int((wx - self._off_x)/self._scale), int((wy - self._off_y)/self._scale)

    def _render(self):
        if self._arr is None:
            self.setText("No Image\nRun pipeline first")
            return
        import cv2
        arr = self._arr.copy()
        if len(arr.shape)==2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2]==3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h,w,ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        pw = max(1, self.width()-4)
        ph = max(1, self.height()-4)
        sx = pw/w; sy = ph/h
        self._scale = min(sx, sy)
        dw = int(w*self._scale); dh = int(h*self._scale)
        self._off_x = (self.width()-dw)//2
        self._off_y = (self.height()-dh)//2

        # Draw pixmap + overlay on a canvas
        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(5,8,16))
        p = QPainter(canvas)
        p.drawPixmap(self._off_x, self._off_y, dw, dh, pix)

        # Draw existing rect
        draw_rect = self._rect
        if draw_rect is None and self._init_img_rect:
            ix,iy,iw,ih = self._init_img_rect
            wx,wy = self._img_to_widget(ix,iy)
            ww = int(iw*self._scale); wh = int(ih*self._scale)
            draw_rect = QRect(wx,wy,ww,wh)

        if draw_rect and not draw_rect.isNull():
            col = QColor(0,212,255) if self.mode!="template" else QColor(255,140,50)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),30)))
            p.drawRect(draw_rect)
            # Corner handles
            p.setPen(QPen(col,2))
            p.setBrush(QBrush(col))
            for cx,cy in [(draw_rect.left(),draw_rect.top()),
                          (draw_rect.right(),draw_rect.top()),
                          (draw_rect.left(),draw_rect.bottom()),
                          (draw_rect.right(),draw_rect.bottom())]:
                p.drawRect(cx-4,cy-4,8,8)
            # Label
            ix,iy = self._widget_to_img(draw_rect.left(), draw_rect.top())
            iw = int(draw_rect.width()/self._scale)
            ih = int(draw_rect.height()/self._scale)
            p.setPen(QPen(col))
            p.setFont(QFont("Courier New",10,QFont.Bold))
            p.drawText(draw_rect.left()+4, draw_rect.top()-6,
                       f"({ix},{iy})  {iw}×{ih}")

        # Pick marker
        if self._pick_pos and self.mode=="pick":
            px,py = self._img_to_widget(*self._pick_pos)
            p.setPen(QPen(QColor(255,255,0),2))
            p.drawLine(px-10,py,px+10,py)
            p.drawLine(px,py-10,px,py+10)
            p.drawEllipse(px-6,py-6,12,12)

        p.end()
        self.setPixmap(canvas)

    # ── Mouse events ─────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton: return
        pos = event.position().toPoint()
        if self.mode == "pick":
            ix,iy = self._widget_to_img(pos.x(), pos.y())
            if self._arr is not None:
                h,w = self._arr.shape[:2]
                ix = max(0,min(ix,w-1)); iy = max(0,min(iy,h-1))
                self._pick_pos = (ix,iy)
                self._render()
                self.pixel_picked.emit(ix,iy)
        else:
            self._drag_start = pos
            self._dragging   = True
            self._rect = QRect(pos, QSize(0,0))

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._dragging or self._drag_start is None: return
        pos = event.position().toPoint()
        self._rect = QRect(self._drag_start, pos).normalized()
        self._render()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._dragging: return
        self._dragging = False
        if self._rect and self._rect.width()>4 and self._rect.height()>4:
            ix,iy = self._widget_to_img(self._rect.left(), self._rect.top())
            iw    = int(self._rect.width()/self._scale)
            ih    = int(self._rect.height()/self._scale)
            if self._arr is not None:
                H,W = self._arr.shape[:2]
                ix=max(0,min(ix,W-1)); iy=max(0,min(iy,H-1))
                iw=max(1,min(iw,W-ix)); ih=max(1,min(ih,H-iy))
            if self.mode in ("roi","template"):
                self.roi_changed.emit(ix,iy,iw,ih)
                if self.mode=="template":
                    self.template_drawn.emit(ix,iy,iw,ih)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ── Main dialog ───────────────────────────────────────────────────
class NodeDetailDialog(QDialog):
    run_requested = Signal(str)

    def __init__(self, node: NodeInstance, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._node  = node
        self._graph = graph
        tool: ToolDef = node.tool

        self.setWindowTitle(f"{tool.icon}  {tool.name}  —  Detail")
        self.setMinimumSize(1000, 650)
        self.resize(1100, 720)
        self.setModal(False)
        self.setStyleSheet("""
            QDialog{background:#0a0e1a;color:#e2e8f0;}
            QGroupBox{border:1px solid #1e2d45;border-radius:6px;
                      margin-top:8px;padding-top:8px;color:#64748b;font-size:11px;font-weight:700;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}
            QScrollArea{border:none;}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(54)
        hdr.setStyleSheet(f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                          f"stop:0 {tool.color},stop:1 #0a0e1a);"
                          f"border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(16,0,16,0)

        icon_l = QLabel(tool.icon)
        icon_l.setStyleSheet("font-size:28px;background:transparent;")
        hl.addWidget(icon_l)

        tc = QVBoxLayout()
        t1 = QLabel(tool.name)
        t1.setStyleSheet("color:#fff;font-size:16px;font-weight:700;background:transparent;")
        t2 = QLabel(f"{tool.category}  •  {tool.description}")
        t2.setStyleSheet("color:#ffffff88;font-size:11px;background:transparent;")
        tc.addWidget(t1); tc.addWidget(t2)
        hl.addLayout(tc,1)

        self._run_btn = QPushButton("▶  Run Node")
        self._run_btn.setFixedSize(120,34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#00d4ff;border:none;border-radius:5px;"
            "color:#000;font-weight:700;font-size:13px;}"
            "QPushButton:hover{background:#33ddff;}"
            "QPushButton:pressed{background:#0099bb;}")
        self._run_btn.clicked.connect(self._on_run)
        hl.addWidget(self._run_btn)
        root.addWidget(hdr)

        # ── Mode hint ─────────────────────────────────────────────
        self._mode_hint = QLabel("")
        self._mode_hint.setStyleSheet(
            "background:#0d1a2a;color:#ffd700;font-size:11px;"
            "padding:5px 16px;border-bottom:1px solid #1e2d45;")
        self._mode_hint.hide()
        root.addWidget(self._mode_hint)

        # ── Splitter: left params | right image ───────────────────
        spl = QSplitter(Qt.Horizontal)
        spl.setHandleWidth(1)
        spl.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(spl,1)

        # LEFT ────────────────────────────────────────────────────
        left = QWidget(); left.setMaximumWidth(290); left.setMinimumWidth(230)
        ll   = QVBoxLayout(left); ll.setContentsMargins(10,10,10,10); ll.setSpacing(8)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#0a0e1a;color:#64748b;padding:6px 10px;
                         border:none;font-size:11px;font-weight:600;}
            QTabBar::tab:selected{color:#00d4ff;border-bottom:2px solid #00d4ff;}
        """)

        # Params tab
        params_scroll = QScrollArea(); params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.NoFrame)
        params_scroll.setWidget(self._build_params_widget())
        tabs.addTab(params_scroll,"⚙ Params")

        # Ports tab
        tabs.addTab(self._build_ports_widget(),"🔌 Ports")
        ll.addWidget(tabs)

        # Output values
        self._out_group = QGroupBox("Output Values")
        og = QVBoxLayout(self._out_group)
        og.setContentsMargins(8,12,8,8); og.setSpacing(4)
        self._out_labels = {}
        self._build_output_labels(og)
        ll.addWidget(self._out_group)

        # Status label
        self._status_lbl = QLabel("Status: IDLE")
        self._status_lbl.setStyleSheet("color:#64748b;font-size:11px;padding:2px;")
        ll.addWidget(self._status_lbl)
        spl.addWidget(left)

        # RIGHT ───────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6,6,6,6); rl.setSpacing(4)

        img_hdr = QHBoxLayout()
        img_lbl = QLabel("OUTPUT IMAGE")
        img_lbl.setStyleSheet("color:#00d4ff;font-size:10px;font-weight:700;letter-spacing:2px;")
        img_hdr.addWidget(img_lbl)
        img_hdr.addStretch()

        # Mode-specific toolbar
        self._draw_mode_bar = QWidget()
        dm_lay = QHBoxLayout(self._draw_mode_bar)
        dm_lay.setContentsMargins(0,0,0,0); dm_lay.setSpacing(4)

        if tool.tool_id in ("crop",):
            self._mode_hint.setText("✏  Kéo chuột trên ảnh để chọn vùng ROI (x/y/w/h sẽ tự cập nhật)")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="roi")
            self._img_label.roi_changed.connect(self._on_roi_changed)
            # Init rect from params
            x=node.params.get("x",0); y=node.params.get("y",0)
            w=node.params.get("crop_w",320); h=node.params.get("crop_h",240)
            QTimer.singleShot(100, lambda: self._img_label.set_rect_from_params(x,y,w,h))
        elif tool.tool_id == "template_match":
            self._mode_hint.setText("✏  Kéo chuột vẽ vùng template trên ảnh — sẽ tự lưu làm template")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="template")
            self._img_label.template_drawn.connect(self._on_template_drawn)
        elif tool.tool_id == "color_picker":
            self._mode_hint.setText("🎨  Click chuột vào ảnh để lấy màu tại điểm đó")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="pick")
            self._img_label.pixel_picked.connect(self._on_pixel_picked)
        else:
            self._img_label = InteractiveImageLabel(mode="view")

        img_hdr.addWidget(self._draw_mode_bar)
        rl.addLayout(img_hdr)
        rl.addWidget(self._img_label,1)

        self._img_info = QLabel("")
        self._img_info.setStyleSheet(
            "color:#64748b;font-size:10px;font-family:'Courier New';padding:2px;")
        self._img_info.setAlignment(Qt.AlignCenter)
        rl.addWidget(self._img_info)

        # Pixel info bar (for color picker)
        self._pixel_bar = QLabel("")
        self._pixel_bar.setStyleSheet(
            "color:#ffd700;font-size:11px;font-family:'Courier New';"
            "background:#111827;border-radius:4px;padding:3px 10px;")
        self._pixel_bar.hide()
        rl.addWidget(self._pixel_bar)

        spl.addWidget(right)
        spl.setSizes([260, 820])

        # Close
        cb = QPushButton("Close")
        cb.setFixedHeight(30)
        cb.setStyleSheet("""QPushButton{background:#1e2d45;border:none;border-radius:4px;
                           color:#94a3b8;font-size:12px;margin:4px 12px;}
                           QPushButton:hover{background:#00d4ff;color:#000;}""")
        cb.clicked.connect(self.close)
        root.addWidget(cb)

        self.refresh_outputs()

    # ── Build widgets ──────────────────────────────────────────────
    def _build_params_widget(self) -> QWidget:
        node = self._node; tool = node.tool
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4,4,4,4); lay.setSpacing(5)
        if not tool.params:
            lbl = QLabel("No parameters.")
            lbl.setStyleSheet("color:#1e2d45;font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter); lay.addWidget(lbl)
        else:
            self._param_rows = {}
            for param in tool.params:
                pr = ParamRow(param, node.params.get(param.name, param.default))
                pr.value_changed.connect(
                    lambda name,val,nid=node.node_id: self._on_param(nid,name,val))
                lay.addWidget(pr)
                self._param_rows[param.name] = pr
        sep=QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)
        note=QLabel("▶ Run Node để áp dụng")
        note.setStyleSheet("color:#1e2d45;font-size:10px;"); note.setAlignment(Qt.AlignCenter)
        lay.addWidget(note); lay.addStretch()
        return w

    def _build_ports_widget(self) -> QWidget:
        tool=self._node.tool; w=QWidget(); lay=QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8); lay.setSpacing(4)
        if tool.inputs:
            h=QLabel("INPUTS"); h.setStyleSheet("color:#00d4ff;font-size:10px;font-weight:700;letter-spacing:1.5px;")
            lay.addWidget(h)
            for p in tool.inputs:
                r=QLabel(f"  ⬤  {p.name}  [{p.data_type}]{'  (opt)' if not p.required else ''}")
                r.setStyleSheet("color:#00b4d8;font-size:11px;font-family:'Courier New';"
                                "background:#0a0e1a;border-radius:3px;padding:3px 6px;")
                lay.addWidget(r)
        if tool.outputs:
            h=QLabel("OUTPUTS"); h.setStyleSheet("color:#ff8c42;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-top:6px;")
            lay.addWidget(h)
            for p in tool.outputs:
                r=QLabel(f"  ⬤  {p.name}  [{p.data_type}]")
                r.setStyleSheet("color:#ff8c42;font-size:11px;font-family:'Courier New';"
                                "background:#0a0e1a;border-radius:3px;padding:3px 6px;")
                lay.addWidget(r)
        lay.addStretch()
        return w

    def _build_output_labels(self, layout):
        self._out_labels={}
        tool=self._node.tool
        has=False
        for port in tool.outputs:
            if port.name=="image": continue
            has=True
            row=QWidget(); rl=QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
            k=QLabel(port.name)
            k.setStyleSheet("color:#64748b;font-size:11px;font-family:'Courier New';"); k.setMinimumWidth(80)
            v=QLabel("—"); v.setStyleSheet("color:#00d4ff;font-size:11px;font-weight:700;")
            rl.addWidget(k); rl.addWidget(v,1); layout.addWidget(row)
            self._out_labels[port.name]=v
        if not has:
            layout.addWidget(QLabel("—"))

    # ── Interactive callbacks ──────────────────────────────────────
    def _on_roi_changed(self, x,y,w,h):
        """Cập nhật params khi kéo ROI."""
        node=self._node
        node.params["x"]=x; node.params["y"]=y
        node.params["crop_w"]=w; node.params["crop_h"]=h
        # Refresh param spinboxes nếu có
        for name,val in [("x",x),("y",y),("crop_w",w),("crop_h",h)]:
            pr=getattr(self,"_param_rows",{}).get(name)
            if pr:
                editor=pr._editor
                if hasattr(editor,"setValue"):
                    editor.blockSignals(True); editor.setValue(val); editor.blockSignals(False)

    def _on_template_drawn(self, x,y,w,h):
        """Cắt vùng template và lưu vào params."""
        node=self._node
        img = node.outputs.get("image") or self._get_input_image()
        if img is None:
            QMessageBox.warning(self,"Template","Chạy node trước để có ảnh.")
            return
        H,W = img.shape[:2]
        x=max(0,min(x,W-1)); y=max(0,min(y,H-1))
        w=max(1,min(w,W-x)); h=max(1,min(h,H-y))
        templ = img[y:y+h, x:x+w]
        node.params["_template_array"] = templ
        node.params["_template_rect"]  = (x,y,w,h)
        self._img_info.setText(
            f"Template saved: ({x},{y}) {w}×{h} px  — Run node to apply")
        self._img_info.setStyleSheet(
            "color:#ffd700;font-size:10px;font-family:'Courier New';padding:2px;")

    def _get_input_image(self):
        """Lấy ảnh input từ upstream node."""
        for conn in self._graph.connections:
            if conn.dst_id==self._node.node_id and conn.dst_port=="image":
                src=self._graph.nodes.get(conn.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _on_pixel_picked(self, x, y):
        """Cập nhật pick_x/pick_y và chạy lại để lấy màu."""
        node=self._node
        node.params["pick_x"]=x; node.params["pick_y"]=y
        # Refresh spinbox
        for name,val in [("pick_x",x),("pick_y",y)]:
            pr=getattr(self,"_param_rows",{}).get(name)
            if pr:
                ed=pr._editor
                if hasattr(ed,"setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        self._pixel_bar.show()
        self._on_run()   # Auto-run to get color

    def _on_param(self, node_id, name, value):
        if self._graph and node_id in self._graph.nodes:
            self._graph.nodes[node_id].params[name]=value

    # ── Run ────────────────────────────────────────────────────────
    def _on_run(self):
        node=self._node
        inputs={p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id==node.node_id:
                src=self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port]=src.outputs[conn.src_port]
        try:
            out=node.tool.process_fn(inputs, node.params)
            node.outputs=out or {}
            node.status="pass"
            if "pass" in node.outputs:
                node.status="pass" if node.outputs["pass"] else "fail"
            node.error_msg=""
        except Exception as e:
            node.outputs={}; node.status="error"; node.error_msg=str(e)

        self.refresh_outputs()
        self.run_requested.emit(node.node_id)

    # ── Refresh ────────────────────────────────────────────────────
    def refresh_outputs(self):
        node=self._node
        # Status
        sc={"pass":"#39ff14","fail":"#ff3860","error":"#ff3860",
            "idle":"#64748b","running":"#ffd700"}.get(node.status,"#64748b")
        self._status_lbl.setText(f"Status: {node.status.upper()}")
        self._status_lbl.setStyleSheet(f"color:{sc};font-size:12px;font-weight:700;padding:2px;")

        if node.error_msg:
            self._img_info.setText(f"Error: {node.error_msg}")
            self._img_info.setStyleSheet("color:#ff3860;font-size:10px;padding:2px;")

        # Scalar outputs
        for name,lbl in self._out_labels.items():
            val=node.outputs.get(name)
            if val is None:
                lbl.setText("—"); lbl.setStyleSheet("color:#1e2d45;font-size:11px;")
            elif isinstance(val,bool):
                lbl.setText("✔ TRUE" if val else "✖ FALSE")
                lbl.setStyleSheet(f"color:{'#39ff14' if val else '#ff3860'};font-size:11px;font-weight:700;")
            elif isinstance(val,float):
                lbl.setText(f"{val:.5f}"); lbl.setStyleSheet("color:#00d4ff;font-size:11px;font-weight:700;")
            elif isinstance(val,int):
                lbl.setText(str(val)); lbl.setStyleSheet("color:#00d4ff;font-size:11px;font-weight:700;")
            else:
                lbl.setText(str(val)[:50]); lbl.setStyleSheet("color:#e2e8f0;font-size:11px;")

        # Color picker specific
        if self._node.tool.tool_id=="color_picker" and node.outputs:
            r=node.outputs.get("r",0); g=node.outputs.get("g",0); b=node.outputs.get("b",0)
            H=node.outputs.get("h",0); S=node.outputs.get("s",0); V=node.outputs.get("v",0)
            self._pixel_bar.setText(
                f"  Pixel ({node.params.get('pick_x',0)}, {node.params.get('pick_y',0)})  "
                f"  RGB: {r},{g},{b}  #{r:02X}{g:02X}{b:02X}  "
                f"  HSV: {H},{S},{V}")
            self._pixel_bar.setStyleSheet(
                f"color:rgb({r},{g},{b});font-size:11px;font-family:'Courier New';"
                f"background:#111827;border-radius:4px;padding:3px 10px;"
                f"border:1px solid rgb({r},{g},{b});")
            self._pixel_bar.show()

        # Image
        img=node.outputs.get("image")
        if img is not None and isinstance(img, np.ndarray):
            h,w=img.shape[:2]
            self._img_info.setText(f"{w}×{h} px  |  {img.dtype}  |  {node.status.upper()}")
            self._img_label.set_image(img)

            # For crop: restore rect
            if self._node.tool.tool_id=="crop":
                x=node.params.get("x",0); y=node.params.get("y",0)
                cw=node.params.get("crop_w",320); ch=node.params.get("crop_h",240)
                self._img_label.set_rect_from_params(x,y,cw,ch)

            # For template: show template rect if saved
            elif self._node.tool.tool_id=="template_match":
                tr=node.params.get("_template_rect")
                if tr:
                    self._img_label.set_rect_from_params(*tr)
                    self._img_label.set_image(img)

        elif not node.error_msg:
            self._img_label.set_image(None)
