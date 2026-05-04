"""
Widget hiển thị ảnh với khả năng zoom và pan
Hỗ trợ cả file path và numpy array
"""
from typing import Optional

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QPixmap, QImage, QWheelEvent, QMouseEvent, QPainter
)


class ImageViewer(QGraphicsView):
    """
    Widget xem ảnh với zoom (chuột lăn) và pan (kéo chuột trái).

    Thao tác:
        - Lăn chuột: zoom in/out (tại vị trí chuột)
        - Giữ chuột trái + kéo: pan ảnh
        - Double-click: fit ảnh vừa khung
        - Ctrl+0: reset zoom về 100%
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_factor = 1.0
        self._panning = False
        self._last_pan_pos = None
        # Flag ngăn fit_to_window ghi đè zoom thủ công
        self._user_has_zoomed = False

        # ── Cấu hình view ──
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Bật anti-aliasing để ảnh zoom đẹp hơn
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setStyleSheet(
            "background-color: #11111b; border: 1px solid #45475a;"
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def load_image(self, source) -> bool:
        """
        Tải ảnh từ file path (str/Path) hoặc numpy array.
        Trả về True nếu thành công.
        """
        try:
            pixmap = self._to_pixmap(source)
            if pixmap is None or pixmap.isNull():
                return False
            self._set_pixmap(pixmap)
            return True
        except Exception as e:
            print(f"Lỗi tải ảnh: {e}")
            return False

    def clear_image(self):
        """Xóa ảnh hiện tại."""
        self._scene.clear()
        self._pixmap_item = None
        self._zoom_factor = 1.0
        self._user_has_zoomed = False

    def fit_to_window(self):
        """Zoom vừa khung nhìn."""
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self._zoom_factor = self.transform().m11()
        self._user_has_zoomed = False

    def reset_zoom(self):
        """Reset zoom về 100% (1 pixel ảnh = 1 pixel màn hình)."""
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self._zoom_factor = 1.0
        self._user_has_zoomed = True

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _to_pixmap(self, source) -> Optional[QPixmap]:
        """Chuyển đổi nguồn ảnh sang QPixmap."""
        if source is None:
            return None

        # Nếu là numpy array
        if _is_numpy(source):
            return _numpy_to_pixmap(source)

        # Nếu là PIL Image
        if _is_pil(source):
            return _pil_to_pixmap(source)

        # Nếu là string hoặc Path → đọc từ file
        path = str(source)
        pixmap = QPixmap(path)
        return pixmap if not pixmap.isNull() else None

    def _set_pixmap(self, pixmap: QPixmap):
        """Hiển thị pixmap lên scene."""
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        # Chỉ fit khi lần đầu load hoặc chưa zoom thủ công
        if not self._user_has_zoomed:
            self.fit_to_window()

    # ==================================================================
    # Events
    # ==================================================================

    def wheelEvent(self, event: QWheelEvent):
        """Zoom bằng chuột lăn — zoom tại vị trí chuột."""
        if self._pixmap_item is None:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        # Zoom nhanh hơn khi đang ở mức zoom lớn
        factor = 1.20 if delta > 0 else 1 / 1.20
        new_zoom = self._zoom_factor * factor

        # Giới hạn zoom: 0.01x → 100x
        if 0.01 <= new_zoom <= 100:
            self.scale(factor, factor)
            self._zoom_factor = new_zoom
            self._user_has_zoomed = True

    def mousePressEvent(self, event: QMouseEvent):
        """Bắt đầu pan khi nhấn chuột trái."""
        if (event.button() == Qt.MouseButton.LeftButton
                and self._pixmap_item is not None):
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan ảnh khi kéo chuột."""
        if self._panning and self._last_pan_pos is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_pan_pos
            self._last_pan_pos = current_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Kết thúc pan."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click → fit ảnh vừa khung."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_window()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        """Chỉ auto-fit khi user CHƯA zoom thủ công."""
        super().resizeEvent(event)
        if self._pixmap_item is not None and not self._user_has_zoomed:
            self.fit_to_window()


# ==================================================================
# Helper functions (module-level)
# ==================================================================

def _is_numpy(obj) -> bool:
    try:
        import numpy as np
        return isinstance(obj, np.ndarray)
    except ImportError:
        return False


def _is_pil(obj) -> bool:
    try:
        from PIL import Image
        return isinstance(obj, Image.Image)
    except ImportError:
        return False


def _numpy_to_pixmap(arr) -> QPixmap:
    """Chuyển numpy array (H,W,3 BGR hoặc RGB) sang QPixmap."""
    import numpy as np

    arr = np.ascontiguousarray(arr, dtype=np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        bytes_per_line = w
        qimg = QImage(
            arr.data, w, h, bytes_per_line,
            QImage.Format.Format_Grayscale8
        )
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        # OpenCV dùng BGR → chuyển sang RGB cho Qt
        import cv2
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr_rgb = np.ascontiguousarray(arr_rgb)
        bytes_per_line = w * 3
        qimg = QImage(
            arr_rgb.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
    elif arr.ndim == 3 and arr.shape[2] == 4:
        h, w, _ = arr.shape
        bytes_per_line = w * 4
        qimg = QImage(
            arr.data, w, h, bytes_per_line,
            QImage.Format.Format_RGBA8888
        )
    else:
        return QPixmap()

    # Tạo deep copy để tránh dangling pointer khi numpy GC
    return QPixmap.fromImage(qimg.copy())


def _pil_to_pixmap(img) -> QPixmap:
    """Chuyển PIL Image sang QPixmap."""
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    bytes_per_line = img.width * 3
    qimg = QImage(
        data, img.width, img.height, bytes_per_line,
        QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimg.copy())