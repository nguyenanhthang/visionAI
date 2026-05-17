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
        if arr is None:
            self._arr = None
            self._pixmap = None
            self.update()
            return
        # Đảm bảo contiguous để Qt dùng buffer trực tiếp (không copy).
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        self._arr = arr   # giữ alive cho QImage buffer reference
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_Grayscale8)
        elif arr.shape[2] == 3:
            h, w, _ = arr.shape
            # Format_BGR888 dùng trực tiếp BGR của OpenCV → skip cvtColor.
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_BGR888)
        elif arr.shape[2] == 4:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            self._arr = arr
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_RGBA8888)
        else:
            self._pixmap = None
            self.update()
            return
        # QPixmap.fromImage copy data sang pixmap format → arr có thể GC sau
        # call này. Nhưng giữ self._arr để pixel-pick / hover còn truy cập.
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

        # Results dropdown — chọn tool nào để overlay annotation lên ảnh gốc.
        # Menu rebuilt động khi mở: list tất cả node có image output, mỗi
        # node 1 checkbox. Ít nhất 1 cái tick → composite mode, base = ảnh gốc
        # (Acquire Image), overlay = diff(input, output) của các node ticked.
        from PySide6.QtWidgets import QToolButton, QMenu, QWidgetAction, QCheckBox
        self._selected_overlays: Dict[str, bool] = {}   # node_id → bật/tắt
        self._btn_results = QToolButton()
        self._btn_results.setText("📊 Results ▾")
        self._btn_results.setPopupMode(QToolButton.InstantPopup)
        self._btn_results.setFixedHeight(28)
        self._btn_results.setToolTip(
            "Composite Results — pick tools để overlay annotation lên ảnh gốc. "
            "Khi không tick gì → chỉ hiện output của node đang chọn.")
        self._btn_results.setStyleSheet("""
            QToolButton{background:#111827;border:1px solid #1e2d45;
                        border-radius:4px;color:#94a3b8;font-size:11px;
                        padding:0 10px;font-weight:600;}
            QToolButton:hover{background:#1a2236;color:#00d4ff;}
            QToolButton::menu-indicator{image:none;}
        """)

        self._results_menu = QMenu(self._btn_results)
        self._results_menu.setStyleSheet(
            "QMenu{background:#0d1220;border:1px solid #1e2d45;"
            "padding:4px;color:#e2e8f0;}"
            "QMenu::separator{height:1px;background:#1e2d45;margin:4px 6px;}")
        self._results_menu.aboutToShow.connect(self._rebuild_results_menu)
        self._btn_results.setMenu(self._results_menu)
        tl.addWidget(self._btn_results)

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

    def _rebuild_results_menu(self):
        """Rebuild menu mỗi khi mở → reflect graph hiện tại."""
        from PySide6.QtWidgets import QWidgetAction, QCheckBox, QLabel
        menu = self._results_menu
        menu.clear()
        if not self._graph:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (No pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
            return

        # Header — base image
        wa_hdr = QWidgetAction(menu)
        hdr = QLabel("  Base: ảnh gốc (Acquire Image)  ")
        hdr.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; "
            "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        # List tất cả node có image output (loại Acquire — đó là base)
        nodes = [(nid, n) for nid, n in self._graph.nodes.items()
                 if "image" in n.outputs
                 and getattr(n.tool, "category", "") != "Acquire Image"]
        # Sắp theo thứ tự topo (đơn giản: theo node_id để stable)
        nodes.sort(key=lambda x: x[0])

        if not nodes:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (Chưa có tool nào trong pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
        else:
            for nid, node in nodes:
                wa = QWidgetAction(menu)
                cb = QCheckBox(f"  {node.tool.icon}  {node.tool.name}  "
                                f"({node.tool.tool_id})")
                cb.setChecked(self._selected_overlays.get(nid, False))
                cb.setStyleSheet(
                    "QCheckBox{color:#e2e8f0; font-size:11px; padding:4px 8px;}"
                    "QCheckBox::indicator{width:14px; height:14px;}")
                cb.toggled.connect(
                    lambda on, _nid=nid: self._on_overlay_toggled(_nid, on))
                wa.setDefaultWidget(cb)
                menu.addAction(wa)

        menu.addSeparator()
        # Quick actions
        from PySide6.QtGui import QAction
        act_all = QAction("✓  Select All", menu)
        act_none = QAction("✗  Clear All", menu)
        act_all.triggered.connect(lambda: self._set_all_overlays(True))
        act_none.triggered.connect(lambda: self._set_all_overlays(False))
        menu.addAction(act_all)
        menu.addAction(act_none)

    def _on_overlay_toggled(self, node_id: str, on: bool):
        self._selected_overlays[node_id] = on
        self._update_results_btn_text()
        self.refresh_current()

    def _set_all_overlays(self, on: bool):
        if not self._graph:
            return
        for nid, n in self._graph.nodes.items():
            if "image" in n.outputs \
                    and getattr(n.tool, "category", "") != "Acquire Image":
                self._selected_overlays[nid] = on
        self._update_results_btn_text()
        self.refresh_current()

    def _update_results_btn_text(self):
        n = sum(1 for v in self._selected_overlays.values() if v)
        if n == 0:
            self._btn_results.setText("📊 Results ▾")
        else:
            self._btn_results.setText(f"📊 Results ({n}) ▾")

    def _find_acquire_root_image(self):
        """Trả output 'image' của node đầu chuỗi (Acquire Image)."""
        if not self._graph:
            return None
        for nid, n in self._graph.nodes.items():
            if getattr(n.tool, "category", "") == "Acquire Image" \
                    and "image" in n.outputs:
                return n.outputs["image"]
        return None

    def _overlay_diff(self, base: np.ndarray, before: np.ndarray,
                      after: np.ndarray) -> np.ndarray:
        """Compose pixel khác biệt (before→after) lên base. Dùng cho Shared
        Graphics: lấy annotation upstream-tool đã vẽ rồi áp lên ảnh hiển thị."""
        if (before is None or after is None
                or before.shape != after.shape
                or before.shape[:2] != base.shape[:2]):
            return base
        import cv2
        b = before if before.ndim == 3 else cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
        a = after  if after.ndim  == 3 else cv2.cvtColor(after,  cv2.COLOR_GRAY2BGR)
        diff = cv2.absdiff(a, b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Pixel coi như annotation nếu lệch ≥ 20 (loại noise nhỏ)
        mask = (gray > 20)
        if not mask.any():
            return base
        out = base.copy()
        out[mask] = a[mask]
        return out

    def _node_input_image(self, node):
        """Output 'image' của upstream gần nhất của node (input của node)."""
        if not self._graph:
            return None
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port == "image":
                src = self._graph.nodes.get(c.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _get_source_image(self, node):
        """Tìm ảnh gốc (raw) của pipeline — traverse ngược về node category
        'Acquire Image' đầu chuỗi. Cho phép Show Result OFF hiện ảnh thô,
        không phải output đã annotate của upstream gần nhất."""
        if not self._graph:
            return None
        visited = set()
        cur = node
        # Walk upstream qua port "image" để tìm root source
        for _ in range(64):    # an toàn: pipeline khó dài hơn 64 node
            if cur is None or cur.node_id in visited:
                break
            visited.add(cur.node_id)
            cat = getattr(cur.tool, "category", "")
            if cat == "Acquire Image" and "image" in cur.outputs:
                return cur.outputs["image"]
            # Tìm upstream nối vào port "image"
            upstream = None
            for c in self._graph.connections:
                if c.dst_id == cur.node_id and c.dst_port == "image":
                    upstream = self._graph.nodes.get(c.src_id)
                    break
            cur = upstream
        # Fallback: upstream gần nhất nếu không tìm thấy Acquire root
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

        active_overlays = [nid for nid, on in self._selected_overlays.items()
                            if on and nid in self._graph.nodes]

        def _vis_of(n):
            """Pick the annotated frame to render for node `n`:
            `_display_image` (private overlay) first, fall back to `image`
            (clean port). `or` is unsafe on numpy arrays — use is-None."""
            v = n.outputs.get("_display_image")
            if v is None:
                v = n.outputs.get("image")
            return v

        if active_overlays:
            # Composite mode: base = ảnh gốc Acquire, overlay = các tool đã tick.
            # Tool annotation lấy từ `_display_image` (ảnh + overlay) thay vì
            # port `image` (đã đổi sang clean pass-through).
            base = self._find_acquire_root_image()
            if base is None:
                img = _vis_of(node)
            else:
                import cv2
                comp = base.copy()
                if comp.ndim == 2:
                    comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
                for nid in active_overlays:
                    n = self._graph.nodes[nid]
                    before = self._node_input_image(n)
                    after  = _vis_of(n)
                    if before is not None and after is not None:
                        comp = self._overlay_diff(comp, before, after)
                img = comp
        else:
            # Mode bình thường: hiển thị output của node đang chọn — ưu tiên
            # `_display_image` (có overlay) rồi fall back `image` clean.
            img = _vis_of(node)

        if img is not None and isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            ch = img.shape[2] if len(img.shape) == 3 else 1
            tag = f"  •  Composite ({len(active_overlays)} overlays)" \
                if active_overlays else ""
            self._lbl_size.setText(
                f"{w}×{h}  ch:{ch}  dtype:{img.dtype}{tag}")
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
