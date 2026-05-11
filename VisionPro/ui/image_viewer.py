"""
ui/image_viewer.py
Panel xem ảnh chính — hiển thị ảnh kết quả với zoom/pan, overlay info,
chọn node để xem output image.
"""
from __future__ import annotations
from typing import Optional, Dict
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QPushButton, QScrollArea,
                                QSizePolicy, QFrame, QSlider, QCheckBox,
                                QToolBar, QSplitter, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (QPixmap, QImage, QColor, QPainter, QPen, QBrush,
                            QFont, QWheelEvent, QMouseEvent, QTransform,
                            QPainterPath)

from core.flow_graph import FlowGraph


# ── Zoomable image widget ─────────────────────────────────────────
class ZoomableImageWidget(QWidget):
    """Widget hiển thị ảnh có zoom/pan bằng mouse."""
    pixel_info = Signal(int, int, tuple)   # x, y, (r,g,b)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#050810;")

        self._pixmap:   Optional[QPixmap] = None
        self._arr:      Optional[np.ndarray] = None
        self._scale     = 1.0
        self._offset    = QPointF(0, 0)
        self._panning   = False
        self._pan_start = QPointF(0, 0)
        self._show_grid = False
        self._zoom_text = ""

    # ── Image ───────────────────────────────────────────────────
    def set_image(self, arr: Optional[np.ndarray]):
        self._arr = arr
        if arr is None:
            self._pixmap = None
            self.update()
            return
        import cv2
        a = arr.copy()
        if len(a.shape) == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        elif a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        elif a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        h, w, ch = a.shape
        qimg = QImage(a.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        if not self._pixmap:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if ww < 1 or wh < 1:
            return
        self._scale  = min(ww / pw, wh / ph) * 0.95
        self._offset = QPointF(
            (ww - pw * self._scale) / 2,
            (wh - ph * self._scale) / 2)

    def fit(self):
        self._fit_to_window()
        self.update()

    def set_zoom(self, factor: float):
        if not self._pixmap:
            return
        cx = self.width() / 2
        cy = self.height() / 2
        img_cx = (cx - self._offset.x()) / self._scale
        img_cy = (cy - self._offset.y()) / self._scale
        self._scale = max(0.05, min(20.0, factor))
        self._offset = QPointF(cx - img_cx * self._scale,
                               cy - img_cy * self._scale)
        self._zoom_text = f"{self._scale * 100:.0f}%"
        self.update()

    # ── Paint ───────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(5, 8, 16))

        if not self._pixmap:
            painter.setPen(QPen(QColor(30, 45, 69)))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No Image\n\nRun pipeline to see results")
            return

        # Draw image
        dx = self._offset.x()
        dy = self._offset.y()
        pw = self._pixmap.width()  * self._scale
        ph = self._pixmap.height() * self._scale
        painter.drawPixmap(int(dx), int(dy), int(pw), int(ph), self._pixmap)

        # Grid overlay at high zoom
        if self._scale > 8 and self._arr is not None:
            painter.setPen(QPen(QColor(0, 212, 255, 40), 0.5))
            # Vertical lines
            x0 = int(dx % self._scale)
            while x0 < self.width():
                painter.drawLine(x0, 0, x0, self.height())
                x0 += int(self._scale)
            # Horizontal lines
            y0 = int(dy % self._scale)
            while y0 < self.height():
                painter.drawLine(0, y0, self.width(), y0)
                y0 += int(self._scale)

        # Zoom indicator
        if self._zoom_text:
            painter.setPen(QPen(QColor(0, 212, 255, 180)))
            painter.setFont(QFont("Courier New", 11, QFont.Bold))
            painter.drawText(self.rect().adjusted(10, 8, -10, -8),
                             Qt.AlignTop | Qt.AlignRight, self._zoom_text)

    # ── Mouse ───────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap:
            return
        pos = event.position()
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_scale = max(0.05, min(20.0, self._scale * factor))

        # Zoom toward cursor
        img_x = (pos.x() - self._offset.x()) / self._scale
        img_y = (pos.y() - self._offset.y()) / self._scale
        self._scale  = new_scale
        self._offset = QPointF(pos.x() - img_x * self._scale,
                               pos.y() - img_y * self._scale)
        self._zoom_text = f"{self._scale * 100:.0f}%"
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning   = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._panning:
            d = pos - self._pan_start
            self._pan_start = pos
            self._offset += QPointF(d.x(), d.y())
            self.update()

        # Pixel info
        if self._arr is not None and self._pixmap:
            ix = int((pos.x() - self._offset.x()) / self._scale)
            iy = int((pos.y() - self._offset.y()) / self._scale)
            h, w = self._arr.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                px = self._arr[iy, ix]
                if len(self._arr.shape) == 2:
                    rgb = (int(px), int(px), int(px))
                else:
                    rgb = (int(px[2]), int(px[1]), int(px[0]))  # BGR→RGB
                self.pixel_info.emit(ix, iy, rgb)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    def resizeEvent(self, event):
        if self._pixmap and self._scale < 0.1:
            self._fit_to_window()
        super().resizeEvent(event)


# ── Main ImageViewer panel ────────────────────────────────────────
class ImageViewerPanel(QWidget):
    """
    Panel xem ảnh chính — hiển thị output image của node được chọn.
    Có thể chọn node từ dropdown, zoom/pan, xem pixel info.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: Optional[FlowGraph] = None
        self._current_node_id: Optional[str] = None
        self._node_map: Dict[str, str] = {}   # display_name → node_id

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Toolbar ────────────────────────────────────────────────
        tb = QWidget()
        tb.setFixedHeight(40)
        tb.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(8)

        view_lbl = QLabel("👁  IMAGE VIEWER")
        view_lbl.setStyleSheet(
            "color:#00d4ff; font-size:11px; font-weight:700; letter-spacing:2px;")
        tl.addWidget(view_lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#1e2d45;")
        tl.addWidget(sep)

        node_lbl = QLabel("Node:")
        node_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        tl.addWidget(node_lbl)

        self._node_combo = QComboBox()
        self._node_combo.setMinimumWidth(180)
        self._node_combo.setStyleSheet("""
            QComboBox{background:#0a0e1a;border:1px solid #1e2d45;
                      color:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:11px;}
            QComboBox::drop-down{border:none;}
            QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;
                                         border:1px solid #1e2d45;
                                         selection-background-color:#1a2236;}
        """)
        self._node_combo.currentIndexChanged.connect(self._on_node_selected)
        tl.addWidget(self._node_combo)

        tl.addStretch()

        # Zoom controls
        def tb_btn(txt, tip):
            b = QPushButton(txt)
            b.setFixedSize(32, 28)
            b.setToolTip(tip)
            b.setStyleSheet("""
                QPushButton{background:#111827;border:1px solid #1e2d45;
                            border-radius:4px;color:#94a3b8;font-size:12px;}
                QPushButton:hover{background:#00d4ff;color:#000;}
            """)
            return b

        btn_fit  = tb_btn("⊡", "Fit to window (F)")
        btn_1to1 = tb_btn("1:1", "Actual pixels")
        btn_in   = tb_btn("+", "Zoom in")
        btn_out  = tb_btn("−", "Zoom out")
        btn_fit.clicked.connect(self._fit)
        btn_1to1.clicked.connect(lambda: self._img_view.set_zoom(1.0))
        btn_in.clicked.connect(lambda: self._img_view.set_zoom(self._img_view._scale * 1.5))
        btn_out.clicked.connect(lambda: self._img_view.set_zoom(self._img_view._scale / 1.5))
        for b in (btn_out, btn_in, btn_1to1, btn_fit):
            tl.addWidget(b)

        # Show Result toggle — switch giữa "Result" (output đã annotate) và
        # "Source" (ảnh gốc input của node, không overlay).
        self._show_result = True
        self._btn_show_result = QPushButton("👁 Result")
        self._btn_show_result.setCheckable(True)
        self._btn_show_result.setChecked(True)
        self._btn_show_result.setFixedHeight(28)
        self._btn_show_result.setToolTip(
            "Toggle Result/Source — Result: output đã vẽ marker; "
            "Source: ảnh gốc input của node.")
        self._btn_show_result.setStyleSheet("""
            QPushButton{background:#111827;border:1px solid #1e2d45;
                        border-radius:4px;color:#94a3b8;font-size:11px;
                        padding:0 10px;font-weight:600;}
            QPushButton:hover{background:#1a2236;color:#00d4ff;}
            QPushButton:checked{background:#0d2a1a;border-color:#39ff14;
                                color:#39ff14;}
        """)
        self._btn_show_result.toggled.connect(self._on_show_result_toggled)
        tl.addWidget(self._btn_show_result)

        lay.addWidget(tb)

        # ── Image view ─────────────────────────────────────────────
        self._img_view = ZoomableImageWidget()
        self._img_view.pixel_info.connect(self._on_pixel_info)
        lay.addWidget(self._img_view, 1)

        # ── Status bar ─────────────────────────────────────────────
        status = QWidget()
        status.setFixedHeight(24)
        status.setStyleSheet("background:#060a14; border-top:1px solid #1e2d45;")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(10, 0, 10, 0)
        sl.setSpacing(16)

        self._lbl_size   = QLabel("—")
        self._lbl_pixel  = QLabel("Hover over image for pixel info")
        self._lbl_status = QLabel("IDLE")
        for lbl in (self._lbl_size, self._lbl_pixel, self._lbl_status):
            lbl.setStyleSheet(
                "color:#1e2d45; font-size:10px; font-family:'Courier New';")
        sl.addWidget(self._lbl_size)
        sl.addWidget(QLabel("|"))
        sl.addWidget(self._lbl_pixel, 1)
        sl.addWidget(QLabel("|"))
        sl.addWidget(self._lbl_status)
        lay.addWidget(status)

    # ── Public API ────────────────────────────────────────────────
    def set_graph(self, graph: FlowGraph):
        self._graph = graph

    def refresh_node_list(self):
        """Cập nhật dropdown list các node có output image."""
        if not self._graph:
            return
        self._node_combo.blockSignals(True)
        prev = self._current_node_id
        self._node_combo.clear()
        self._node_map = {}
        self._node_combo.addItem("— Select node —", None)

        for nid, node in self._graph.nodes.items():
            has_img_output = any(p.name == "image" for p in node.tool.outputs)
            if has_img_output:
                label = f"{node.tool.icon} {node.tool.name}  [{nid}]"
                self._node_combo.addItem(label, nid)
                self._node_map[label] = nid

        # Restore selection
        if prev:
            for i in range(self._node_combo.count()):
                if self._node_combo.itemData(i) == prev:
                    self._node_combo.setCurrentIndex(i)
                    break

        self._node_combo.blockSignals(False)

    def show_node(self, node_id: str):
        """Hiển thị output image của node_id."""
        for i in range(self._node_combo.count()):
            if self._node_combo.itemData(i) == node_id:
                self._node_combo.setCurrentIndex(i)
                break
        self._display_node(node_id)

    def refresh_current(self):
        """Refresh ảnh của node đang xem."""
        if self._current_node_id:
            self._display_node(self._current_node_id)

    # ── Internal ─────────────────────────────────────────────────
    def _on_node_selected(self, idx: int):
        node_id = self._node_combo.itemData(idx)
        if node_id:
            self._display_node(node_id)
        else:
            self._img_view.set_image(None)
            self._lbl_size.setText("—")
            self._lbl_status.setText("IDLE")
            self._current_node_id = None

    def _on_show_result_toggled(self, on: bool):
        self._show_result = on
        self._btn_show_result.setText("👁 Result" if on else "🖼 Source")
        self.refresh_current()

    def _get_source_image(self, node):
        """Tìm ảnh input của node — lấy output 'image' của upstream gần nhất."""
        if not self._graph:
            return None
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port == "image":
                src = self._graph.nodes.get(c.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _display_node(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return
        self._current_node_id = node_id
        node = self._graph.nodes[node_id]

        if self._show_result:
            img = node.outputs.get("image")
        else:
            img = self._get_source_image(node)
            if img is None:
                # Node nguồn (Acquire Image): output chính là source → hiển thị output
                img = node.outputs.get("image")

        if img is not None and isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            ch = img.shape[2] if len(img.shape) == 3 else 1
            self._lbl_size.setText(f"{w}×{h}  ch:{ch}  dtype:{img.dtype}")
            self._img_view.set_image(img)
        else:
            self._img_view.set_image(None)
            self._lbl_size.setText("No image output yet")

        status_colors = {
            "pass": "#39ff14", "fail": "#ff3860",
            "error": "#ff3860", "idle": "#64748b", "running": "#ffd700"
        }
        sc = status_colors.get(node.status, "#64748b")
        self._lbl_status.setText(node.status.upper())
        self._lbl_status.setStyleSheet(
            f"color:{sc}; font-size:10px; font-family:'Courier New'; font-weight:700;")

    def _fit(self):
        self._img_view.fit()

    def _on_pixel_info(self, x: int, y: int, rgb: tuple):
        r, g, b = rgb
        self._lbl_pixel.setText(
            f"X:{x:4d}  Y:{y:4d}    R:{r:3d}  G:{g:3d}  B:{b:3d}"
            f"    #{r:02X}{g:02X}{b:02X}")
        self._lbl_pixel.setStyleSheet(
            f"color:rgb({r},{g},{b}); font-size:10px; font-family:'Courier New';"
            f"background:#111827; border-radius:3px; padding:0 6px;")
