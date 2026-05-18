"""
ui/image_viewer.py
Panel xem ảnh chính — hiển thị ảnh kết quả với zoom/pan, overlay info,
chọn node để xem output image.
"""
from __future__ import annotations
from typing import Optional, Dict, List
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
        # Luôn refit khi widget thay đổi kích thước (vd Full Image View
        # toggle, window resize) — tránh ảnh tràn ra ngoài viewport.
        if self._pixmap is not None:
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
        btn_1to1 = tb_btn("1:1", "Actual pixels (cap ở fit nếu ảnh > viewport)")
        btn_in   = tb_btn("+", "Zoom in")
        btn_out  = tb_btn("−", "Zoom out")
        btn_fit.clicked.connect(self._fit)
        btn_1to1.clicked.connect(self._zoom_actual)
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
            "Results — pick tools để hiển thị. Single view: composite "
            "annotation lên ảnh gốc Acquire. Multi-view (⊞): mỗi item ticked "
            "= 1 ô riêng. Không tick gì → auto theo Acquire/Camera pipeline.")
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

        # Multi-view toggle — auto split khung khi có nhiều input branches.
        # • 2+ Acquire/Camera roots → 1 ô per root (mỗi pipeline 1 view).
        # • 1 root → auto-expand thành nhiều ô (mỗi node trong branch 1 ô).
        # • Mỗi ô có combo độc lập chọn node từ bất kỳ branch nào (Acquire
        #   hoặc Camera), header hiển thị pipeline gốc của node đang xem.
        self._btn_multi = tb_btn(
            "⊞",
            "Toggle multi-view — tick item nào trong Results menu thì item đó "
            "thành 1 ô. Không tick gì → auto 1 ô per Acquire/Camera pipeline.")
        self._btn_multi.setCheckable(True)
        self._btn_multi.toggled.connect(self._on_multi_toggled)
        tl.addWidget(self._btn_multi)

        lay.addWidget(tb)

        # ── Image view ─────────────────────────────────────────────
        # Stack: index 0 = single, index 1 = multi-grid
        from PySide6.QtWidgets import QStackedWidget, QGridLayout
        self._view_stack = QStackedWidget()
        self._img_view = ZoomableImageWidget()
        self._img_view.pixel_info.connect(self._on_pixel_info)
        self._view_stack.addWidget(self._img_view)

        # Multi-view grid (built lazily — refresh khi toggle / graph changes)
        self._multi_container = QWidget()
        self._multi_grid = QGridLayout(self._multi_container)
        self._multi_grid.setContentsMargins(0, 0, 0, 0)
        self._multi_grid.setSpacing(2)
        # Mỗi entry là dict {root, cell_widget, view, combo, status, root_lbl}.
        # `root` = default Acquire/Camera root cho ô (dùng để fallback combo
        # khi node đang chọn bị xóa); user có thể switch combo sang node thuộc
        # bất kỳ pipeline nào, header sẽ cập nhật theo.
        self._multi_views: List[dict] = []
        self._view_stack.addWidget(self._multi_container)

        lay.addWidget(self._view_stack, 1)

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
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()

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
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()

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
        if self._btn_multi.isChecked():
            self._refresh_multi_views()

    # ── Multi-view ────────────────────────────────────────────────
    def _on_multi_toggled(self, checked: bool):
        """Toggle giữa single view và multi-view grid."""
        if checked:
            self._rebuild_multi_grid()
            self._view_stack.setCurrentIndex(1)
        else:
            self._view_stack.setCurrentIndex(0)

    # Tool IDs nhận diện 2 loại "Acquire Image" pipeline:
    #   acquire_image  → file-based (folder/file load) → header "Acquire Image"
    #   camera_acquire → camera-based (OpenCV/HikRobot) → header "Camera Image"
    _FILE_ACQUIRE_ID = "acquire_image"
    _CAMERA_ACQUIRE_ID = "camera_acquire"

    def _enumerate_branch_roots(self) -> List[str]:
        """Pipeline roots = node category 'Acquire Image' (tool_id
        `acquire_image` cho file, `camera_acquire` cho camera). Sort: file
        trước, camera sau, stable theo node_id trong mỗi nhóm.

        Lọc category thay vì 'has image-out & no image-in' để tránh leak các
        tool có image-input optional (vd Area Measure nối qua `mask`/
        `contours` thay vì port `image`) thành false-positive root.
        """
        if self._graph is None:
            return []
        roots = [nid for nid, node in self._graph.nodes.items()
                 if getattr(node.tool, "category", "") == "Acquire Image"]

        def _order(nid: str) -> int:
            tid = self._graph.nodes[nid].tool.tool_id
            if tid == self._FILE_ACQUIRE_ID:
                return 0
            if tid == self._CAMERA_ACQUIRE_ID:
                return 1
            return 2

        roots.sort(key=lambda nid: (_order(nid), nid))
        return roots

    def _root_pipeline_label(self, root_id: str) -> str:
        """Header label cho pipeline gốc — 'Acquire Image' (file) hoặc
        'Camera Image' (camera). Dùng cho cell header trong multi-view và
        section trong combo dropdown."""
        node = self._graph.nodes.get(root_id) if self._graph else None
        if node is None:
            return "?"
        if node.tool.tool_id == self._CAMERA_ACQUIRE_ID:
            return "Camera Image"
        if node.tool.tool_id == self._FILE_ACQUIRE_ID:
            return "Acquire Image"
        return node.tool.name

    def _node_pipeline_root(self, node_id: str) -> Optional[str]:
        """Tìm Acquire/Camera root mà node thuộc về (đi ngược upstream theo
        port `image`). Trả None nếu node không thuộc pipeline Acquire/Camera
        nào (vd dangling tool)."""
        if self._graph is None or node_id not in self._graph.nodes:
            return None
        roots = set(self._enumerate_branch_roots())
        if node_id in roots:
            return node_id
        visited = set()
        cur = node_id
        for _ in range(64):
            if cur in visited:
                break
            visited.add(cur)
            if cur in roots:
                return cur
            upstream = None
            for c in self._graph.connections:
                if c.dst_id == cur and c.dst_port == "image":
                    upstream = c.src_id
                    break
            if upstream is None:
                return None
            cur = upstream
        return None

    def _branch_terminal(self, root_id: str) -> str:
        """BFS xuôi dòng từ root theo image connections → trả về node cuối
        cùng (xa nhất từ root) có image output. Khi pipeline rẽ nhánh
        → chọn nhánh dài nhất.
        """
        if self._graph is None:
            return root_id
        img_outs: Dict[str, List[str]] = {}
        for c in self._graph.connections:
            if c.src_port == "image" and c.dst_port == "image":
                img_outs.setdefault(c.src_id, []).append(c.dst_id)
        # BFS với depth tracking → terminal = node depth lớn nhất có image out
        best = (0, root_id)
        visited = {root_id}
        queue = [(root_id, 0)]
        while queue:
            cur, depth = queue.pop(0)
            for dst in img_outs.get(cur, []):
                if dst in visited:
                    continue
                visited.add(dst)
                node = self._graph.nodes.get(dst)
                if node and "image" in {p.name for p in node.tool.outputs}:
                    if depth + 1 > best[0]:
                        best = (depth + 1, dst)
                queue.append((dst, depth + 1))
        return best[1]

    def _branch_image_nodes(self, root_id: str) -> List[str]:
        """BFS xuôi dòng từ root → list mọi node có image output trong branch.
        Dùng để populate dropdown trong mỗi ô của multi-view.
        """
        if self._graph is None:
            return []
        img_outs: Dict[str, List[str]] = {}
        for c in self._graph.connections:
            if c.src_port == "image" and c.dst_port == "image":
                img_outs.setdefault(c.src_id, []).append(c.dst_id)
        result = []
        seen = set()
        queue = [root_id]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            node = self._graph.nodes.get(cur)
            if node and "image" in {p.name for p in node.tool.outputs}:
                result.append(cur)
            for dst in img_outs.get(cur, []):
                if dst not in seen:
                    queue.append(dst)
        return result

    def _plan_multi_cells(self, roots: List[str]) -> List[tuple]:
        """Quyết định cells = list (root_default, node_default). Rules:
          • 2+ Acquire/Camera roots → 1 ô per root, default = terminal.
          • 1 root → auto-expand: 1 ô per image node trong branch
            (root + downstream). Multi-view stays useful kể cả khi pipeline
            chỉ có 1 source.
        Cap ở 9 cells để tránh grid quá đông.
        """
        if not roots:
            return []
        if len(roots) >= 2:
            return [(r, self._branch_terminal(r)) for r in roots][:9]
        root = roots[0]
        nodes = self._branch_image_nodes(root) or [root]
        return [(root, nid) for nid in nodes][:9]

    def _multi_cells_plan(self) -> List[tuple]:
        """Cells trong multi-view = list (root_id, node_id).
        Priority:
          1. User tick item trong Results menu → mỗi item ticked = 1 ô. Cho
             phép user explicit chọn "tách result này thành từng view".
          2. Không tick gì → fallback auto theo Acquire/Camera roots (giữ
             multi-view useful khi user chưa pick result nào).
        Sort theo BFS order trong pipeline (Acquire/Camera roots upstream
        first). Cap ở 9 ô.
        """
        if self._graph is None:
            return []
        selected = [nid for nid, on in self._selected_overlays.items()
                    if on and nid in self._graph.nodes]
        if not selected:
            return self._plan_multi_cells(self._enumerate_branch_roots())

        # Topological order: BFS từ mọi Acquire/Camera root, nodes ngoài
        # pipeline (vd Image Convert đứng độc lập) xếp cuối theo node_id.
        master: List[str] = []
        for root in self._enumerate_branch_roots():
            for nid in self._branch_image_nodes(root):
                if nid not in master:
                    master.append(nid)
        order = {nid: i for i, nid in enumerate(master)}
        selected.sort(key=lambda nid: (order.get(nid, 10**9), nid))

        cells = []
        for nid in selected:
            root = self._node_pipeline_root(nid) or nid
            cells.append((root, nid))
        return cells[:9]

    def _rebuild_multi_grid(self):
        """Detect Acquire/Camera branches và build grid ZoomableImageWidget.
        Mỗi ô có combo chọn node từ BẤT KỲ Acquire/Camera branch nào — user
        có thể tự chọn 'view nào' (Acquire Image hoặc Camera Image) cho từng
        ô độc lập."""
        # Clear old widgets
        for cell in self._multi_views:
            cell["cell_widget"].setParent(None)
            cell["cell_widget"].deleteLater()
        self._multi_views = []

        cells_plan = self._multi_cells_plan()
        n = len(cells_plan)
        if n == 0:
            return

        # Grid layout: 1 → 1×1; 2 → 1×2; 3-4 → 2×2; 5-6 → 2×3; 7-9 → 3×3
        if n <= 1:    cols = 1
        elif n <= 2:  cols = 2
        elif n <= 6:  cols = (n + 1) // 2
        else:         cols = 3

        from PySide6.QtWidgets import (QVBoxLayout as _QV, QHBoxLayout as _QH,
                                       QToolButton, QMenu)
        for i, (root_id, default_nid) in enumerate(cells_plan):
            cell = QWidget()
            cell_lay = _QV(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(0)

            # Header: pipeline label + node combo + per-cell Results button + status
            hdr = QWidget()
            hdr.setStyleSheet(
                "background:#060a14;border-bottom:1px solid #1e2d45;")
            hl = _QH(hdr)
            hl.setContentsMargins(6, 3, 6, 3); hl.setSpacing(6)
            root_lbl = QLabel("")
            root_lbl.setTextFormat(Qt.RichText)
            hl.addWidget(root_lbl)

            cb = QComboBox()
            cb.setStyleSheet("""
                QComboBox{background:#0a0e1a;border:1px solid #1e2d45;
                          color:#e2e8f0;padding:1px 6px;border-radius:3px;
                          font-size:10px;}
                QComboBox::drop-down{border:none;}
                QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;
                                             border:1px solid #1e2d45;
                                             selection-background-color:#1a2236;}
            """)
            cb.setToolTip(
                "Chọn base node hiển thị trong ô — list gom cả Acquire Image "
                "và Camera Image branches.")
            self._populate_cell_combo(cb, default_nid)
            cb.currentIndexChanged.connect(
                lambda _idx, idx=i: self._on_multi_cell_changed(idx))
            hl.addWidget(cb, 1)

            # Per-cell Results button: pick overlay results để composite lên
            # base của ô này. Menu group theo pipeline (Acquire/Camera) như
            # global Results, nhưng selection độc lập cho từng ô.
            cell_results_btn = QToolButton()
            cell_results_btn.setText("📊")
            cell_results_btn.setPopupMode(QToolButton.InstantPopup)
            cell_results_btn.setFixedHeight(22)
            cell_results_btn.setToolTip(
                "Pick result(s) để composite lên ô này. Tick 1+ item → "
                "overlay annotation lên base image của pipeline. Không tick "
                "gì → chỉ hiện base node (combo bên trái).")
            cell_results_btn.setStyleSheet("""
                QToolButton{background:#111827;border:1px solid #1e2d45;
                            border-radius:3px;color:#94a3b8;font-size:11px;
                            padding:0 6px;font-weight:600;}
                QToolButton:hover{background:#1a2236;color:#00d4ff;}
                QToolButton::menu-indicator{image:none;}
            """)
            cell_menu = QMenu(cell_results_btn)
            cell_menu.setStyleSheet(
                "QMenu{background:#0d1220;border:1px solid #1e2d45;"
                "padding:4px;color:#e2e8f0;}"
                "QMenu::separator{height:1px;background:#1e2d45;margin:4px 6px;}")
            cell_results_btn.setMenu(cell_menu)
            hl.addWidget(cell_results_btn)

            status_lbl = QLabel("●")
            status_lbl.setStyleSheet("color:#64748b;font-size:11px;")
            hl.addWidget(status_lbl)

            cell_lay.addWidget(hdr)

            view = ZoomableImageWidget()
            cell_lay.addWidget(view, 1)

            r, c = divmod(i, cols)
            self._multi_grid.addWidget(cell, r, c)
            entry = {
                "root": root_id, "cell_widget": cell,
                "view": view, "combo": cb, "status": status_lbl,
                "root_lbl": root_lbl,
                "cell_overlays": {},
                "cell_results_btn": cell_results_btn,
                "cell_menu": cell_menu,
            }
            self._multi_views.append(entry)
            # Rebuild menu khi mở → reflect graph hiện tại + checked state
            cell_menu.aboutToShow.connect(
                lambda _entry=entry: self._rebuild_cell_results_menu(_entry))
        self._refresh_multi_views()

    def _populate_cell_combo(self, cb: QComboBox, default_nid: Optional[str]):
        """Fill combo với tất cả image nodes từ MỌI Acquire/Camera branch,
        group theo pipeline. Đặt mặc định ở `default_nid` nếu có."""
        cb.blockSignals(True)
        cb.clear()
        roots = self._enumerate_branch_roots()
        for root in roots:
            label = self._root_pipeline_label(root)
            # Section separator để user phân biệt pipelines trong dropdown
            if cb.count() > 0:
                cb.insertSeparator(cb.count())
            head_idx = cb.count()
            cb.addItem(f"── {label} ──", None)
            # Disable head row (visual section header only)
            model = cb.model()
            from PySide6.QtCore import Qt as _Qt
            item = model.item(head_idx)
            if item is not None:
                item.setFlags(item.flags() & ~_Qt.ItemIsEnabled
                              & ~_Qt.ItemIsSelectable)
                item.setData("color:#64748b;font-style:italic;",
                             _Qt.ToolTipRole)
            for nid in self._branch_image_nodes(root):
                node = self._graph.nodes.get(nid)
                if not node:
                    continue
                cb.addItem(f"  {node.tool.icon} {node.tool.name}", nid)
        # Default select
        if default_nid is not None:
            for j in range(cb.count()):
                if cb.itemData(j) == default_nid:
                    cb.setCurrentIndex(j); break
        cb.blockSignals(False)

    def _on_multi_cell_changed(self, cell_idx: int):
        """User pick node khác cho ô `cell_idx` → load image + sync header."""
        if 0 <= cell_idx < len(self._multi_views):
            self._push_multi_cell(self._multi_views[cell_idx])

    def _cell_pipeline_root(self, entry) -> Optional[str]:
        """Pipeline gốc của ô = pipeline của base node đang chọn ở combo
        (nếu node thuộc Acquire/Camera branch), fallback về entry['root']."""
        nid = entry["combo"].currentData() if "combo" in entry else None
        if nid:
            r = self._node_pipeline_root(nid)
            if r is not None:
                return r
        return entry.get("root")

    def _cell_branch_tools(self, entry) -> List[tuple]:
        """List (nid, node) các tool thuộc pipeline của ô — exclude Acquire
        root (đó là base, không phải tool overlay)."""
        if self._graph is None:
            return []
        root = self._cell_pipeline_root(entry)
        if root is None:
            return []
        tools = []
        for nid in self._branch_image_nodes(root):
            node = self._graph.nodes.get(nid)
            if (node is None
                    or getattr(node.tool, "category", "") == "Acquire Image"):
                continue
            tools.append((nid, node))
        return tools

    def _rebuild_cell_results_menu(self, entry):
        """Menu Results của 1 cell — CHỈ list tool thuộc pipeline của ô đó
        (Acquire Image branch HOẶC Camera Image branch, tùy combo base
        node). Tránh user phải scroll tìm tool giữa nhiều pipeline."""
        from PySide6.QtWidgets import QWidgetAction, QCheckBox, QLabel
        from PySide6.QtGui import QAction
        menu = entry['cell_menu']
        menu.clear()
        if not self._graph:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (No pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
            return

        root = self._cell_pipeline_root(entry)
        section_label = (self._root_pipeline_label(root) if root
                         else "—")

        # Header — ghi rõ ô đang là pipeline nào
        wa_hdr = QWidgetAction(menu)
        hdr = QLabel(f"  Overlay cho {section_label}  ")
        hdr.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; "
            "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        tools = self._cell_branch_tools(entry)
        if not tools:
            wa = QWidgetAction(menu)
            msg = ("(Pipeline chưa có tool nào)" if root
                   else "(Pick base node ở combo trước)")
            lbl = QLabel(f"  {msg}  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
        else:
            for nid, node in tools:
                wa = QWidgetAction(menu)
                cb = QCheckBox(f"  {node.tool.icon}  {node.tool.name}  "
                                f"({node.tool.tool_id})")
                cb.setChecked(entry['cell_overlays'].get(nid, False))
                cb.setStyleSheet(
                    "QCheckBox{color:#e2e8f0; font-size:11px; padding:4px 8px;}"
                    "QCheckBox::indicator{width:14px; height:14px;}")
                cb.toggled.connect(
                    lambda on, _nid=nid, _entry=entry:
                        self._on_cell_overlay_toggled(_entry, _nid, on))
                wa.setDefaultWidget(cb)
                menu.addAction(wa)

        menu.addSeparator()
        act_clear = QAction("✗  Clear All", menu)
        act_clear.triggered.connect(
            lambda _checked=False, _entry=entry: self._clear_cell_overlays(_entry))
        menu.addAction(act_clear)

    def _active_cell_overlays(self, entry) -> List[str]:
        """Overlay node_ids đang ACTIVE cho ô — chỉ những item thuộc pipeline
        hiện tại của ô (lọc stale items từ pipeline khác). Cho phép user giữ
        selection per-pipeline: switch combo qua-lại không mất tick."""
        if not self._graph:
            return []
        tool_ids = {nid for nid, _ in self._cell_branch_tools(entry)}
        return [nid for nid, on in entry.get('cell_overlays', {}).items()
                if on and nid in tool_ids]

    def _on_cell_overlay_toggled(self, entry, node_id: str, on: bool):
        entry['cell_overlays'][node_id] = on
        self._update_cell_results_btn(entry)
        self._push_multi_cell(entry)

    def _clear_cell_overlays(self, entry):
        """Clear chỉ overlay của pipeline hiện tại — selection các pipeline
        khác (nếu user đã switch qua-lại) được giữ."""
        active = self._active_cell_overlays(entry)
        for nid in active:
            entry['cell_overlays'][nid] = False
        self._update_cell_results_btn(entry)
        self._push_multi_cell(entry)

    def _update_cell_results_btn(self, entry):
        n = len(self._active_cell_overlays(entry))
        btn = entry['cell_results_btn']
        btn.setText("📊" if n == 0 else f"📊 ({n})")

    def _push_multi_cell(self, entry):
        """Load ảnh + status của node đang chọn trong ô vào view; cập nhật
        header để reflect pipeline gốc. Nếu cell có overlay items ticked
        (📊 button), composite chúng lên base image của pipeline."""
        nid = entry["combo"].currentData()
        node = self._graph.nodes.get(nid) if self._graph and nid else None

        pipeline_root = self._node_pipeline_root(nid) if nid else None
        if pipeline_root:
            label = self._root_pipeline_label(pipeline_root)
        else:
            label = self._root_pipeline_label(entry["root"])
        entry["root_lbl"].setText(f"<b>{label}</b>  →")
        entry["root_lbl"].setStyleSheet("color:#64748b;font-size:10px;")

        if node is None:
            return

        def _vis_of(n):
            v = n.outputs.get("_display_image")
            if v is None:
                v = n.outputs.get("image")
            return v

        # Chỉ apply overlays thuộc pipeline hiện tại của ô — bỏ stale items
        # nếu user đã switch combo qua pipeline khác.
        active = self._active_cell_overlays(entry)

        img = None
        if active:
            # Base = ảnh Acquire của pipeline ô này (file hoặc camera root).
            base_root = pipeline_root or entry["root"]
            base_node = self._graph.nodes.get(base_root)
            base = base_node.outputs.get("image") if base_node else None
            if base is not None and isinstance(base, np.ndarray):
                import cv2
                comp = base.copy()
                if comp.ndim == 2:
                    comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
                for oid in active:
                    on_node = self._graph.nodes[oid]
                    before = self._node_input_image(on_node)
                    after = _vis_of(on_node)
                    if before is not None and after is not None:
                        comp = self._overlay_diff(comp, before, after)
                img = comp
        if img is None:
            img = _vis_of(node)
        if img is not None:
            entry["view"].set_image(img)

        status = getattr(node, "status", "—") or "—"
        color = {"pass": "#39ff14", "fail": "#ff3860",
                 "error": "#ff3860", "running": "#ffd700"}.get(status, "#64748b")
        entry["status"].setStyleSheet(
            f"color:{color};font-size:13px;font-weight:bold;")
        entry["status"].setToolTip(status.upper())
        self._update_cell_results_btn(entry)

    def _refresh_multi_views(self):
        """Push ảnh mới nhất lên từng ô. Cũng resync combo items khi graph
        thay đổi (node mới được thêm/xóa)."""
        if self._graph is None:
            return
        # Snapshot tất cả image nodes hiện tại từ mọi Acquire/Camera root
        roots = self._enumerate_branch_roots()
        current_ids = []
        for r in roots:
            for nid in self._branch_image_nodes(r):
                if nid not in current_ids:
                    current_ids.append(nid)

        for entry in self._multi_views:
            existing_ids = [entry["combo"].itemData(j)
                             for j in range(entry["combo"].count())
                             if entry["combo"].itemData(j) is not None]
            if list(existing_ids) != list(current_ids):
                prev = entry["combo"].currentData()
                self._populate_cell_combo(entry["combo"], prev)
                # Nếu prev không còn tồn tại, fallback về terminal của root
                if entry["combo"].currentData() != prev:
                    term = self._branch_terminal(entry["root"])
                    for j in range(entry["combo"].count()):
                        if entry["combo"].itemData(j) == term:
                            entry["combo"].setCurrentIndex(j); break
            self._push_multi_cell(entry)

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

        # Header — mode-dependent caption: single view = composite mode,
        # multi-view = mỗi item ticked thành 1 ô riêng.
        wa_hdr = QWidgetAction(menu)
        if self._btn_multi.isChecked():
            hdr_text = "  Multi-view: mỗi item ticked = 1 ô  "
        else:
            hdr_text = "  Base: ảnh gốc (Acquire Image) + overlay  "
        hdr = QLabel(hdr_text)
        hdr.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; "
            "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        # Group nodes theo pipeline (Acquire Image / Camera Image) — dễ tìm
        # tool nào thuộc flow nào. Trong mỗi group sort theo BFS order
        # (xuôi dòng từ root), phản ánh đúng flow execution.
        roots = self._enumerate_branch_roots()
        groups: List[tuple] = []   # [(section_label, [(nid, node), ...])]
        accounted: set = set()
        for root in roots:
            section_label = self._root_pipeline_label(root)
            tools = []
            for nid in self._branch_image_nodes(root):
                if nid in accounted:
                    continue
                accounted.add(nid)
                node = self._graph.nodes.get(nid)
                if (node is None
                        or getattr(node.tool, "category", "") == "Acquire Image"):
                    # Skip root acquire/camera node (đó là base, không phải tool)
                    continue
                tools.append((nid, node))
            if tools:
                groups.append((section_label, tools))

        # Nodes không thuộc Acquire/Camera flow nào (vd dangling tool) → group
        # "Other" để vẫn cho user pick được. Check tool output port (static)
        # thay vì n.outputs (chỉ có sau khi pipeline run).
        others = [(nid, n) for nid, n in self._graph.nodes.items()
                  if any(p.name == "image" for p in n.tool.outputs)
                  and nid not in accounted
                  and getattr(n.tool, "category", "") != "Acquire Image"]
        others.sort(key=lambda x: x[0])
        if others:
            groups.append(("Other", others))

        total_tools = sum(len(items) for _, items in groups)
        if total_tools == 0:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (Chưa có tool nào trong pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
        else:
            for gi, (section_label, items) in enumerate(groups):
                if gi > 0:
                    menu.addSeparator()
                # Section header — pipeline gốc (Acquire Image / Camera Image)
                wa_sec = QWidgetAction(menu)
                sec_lbl = QLabel(f"  ── {section_label} ──  ")
                sec_lbl.setStyleSheet(
                    "color:#94a3b8; font-size:10px; font-weight:600; "
                    "padding:4px 8px; background:#0d1220;")
                wa_sec.setDefaultWidget(sec_lbl)
                menu.addAction(wa_sec)
                for nid, node in items:
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
        if self._btn_multi.isChecked():
            # Multi-view: số ô = số item ticked → cần rebuild grid để add/
            # remove cell, không chỉ refresh nội dung.
            self._rebuild_multi_grid()
        else:
            self.refresh_current()

    def _set_all_overlays(self, on: bool):
        if not self._graph:
            return
        for nid, n in self._graph.nodes.items():
            if "image" in n.outputs \
                    and getattr(n.tool, "category", "") != "Acquire Image":
                self._selected_overlays[nid] = on
        self._update_results_btn_text()
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()
        else:
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

    def _zoom_actual(self):
        """1:1 button — cap zoom ở fit_scale để ảnh không tràn viewport.
        Ảnh nhỏ hơn viewport: scale = 1.0 (actual pixels).
        Ảnh lớn hơn viewport: scale = fit_scale (vẫn hiển thị đầy đủ,
        không cần pan). User muốn zoom thật > fit có thể dùng wheel hoặc +.
        """
        v = self._img_view
        if v._pixmap is None:
            return
        pw, ph = v._pixmap.width(), v._pixmap.height()
        ww, wh = v.width(), v.height()
        if ww < 1 or wh < 1 or pw < 1 or ph < 1:
            return
        fit_scale = min(ww / pw, wh / ph) * 0.95
        v.set_zoom(min(1.0, fit_scale))

    def _on_pixel_info(self, x: int, y: int, rgb: tuple):
        r, g, b = rgb
        self._lbl_pixel.setText(
            f"X:{x:4d}  Y:{y:4d}    R:{r:3d}  G:{g:3d}  B:{b:3d}"
            f"    #{r:02X}{g:02X}{b:02X}")
        self._lbl_pixel.setStyleSheet(
            f"color:rgb({r},{g},{b}); font-size:10px; font-family:'Courier New';"
            f"background:#111827; border-radius:3px; padding:0 6px;")
