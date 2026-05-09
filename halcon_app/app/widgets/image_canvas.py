"""Image canvas: hiển thị numpy image, hỗ trợ zoom (wheel), pan (drag),
và vẽ segment cho thao tác Measure 1D."""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _MovableRoiItem(QGraphicsRectItem):
    """Rect item kéo được; gọi callback khi thay đổi vị trí."""

    def __init__(self, rect: QRectF, on_change):
        super().__init__(rect)
        self._on_change = on_change
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.SizeAllCursor)
        self.setAcceptHoverEvents(True)

    def itemChange(self, change, value):
        if (change == QGraphicsItem.ItemPositionHasChanged
                and self._on_change is not None):
            self._on_change()
        return super().itemChange(change, value)


def _blend_mask(img: np.ndarray, mask: np.ndarray,
                color: tuple[int, int, int] = (54, 197, 214),
                alpha: float = 0.45) -> np.ndarray:
    """Trộn mask (uint8) lên ảnh BGR/GRAY -> ảnh BGR có vùng mask tô màu."""
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
    if mask is None:
        return base
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(mask, (base.shape[1], base.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    overlay = base.copy()
    overlay[mask > 0] = (
        np.array(color, dtype=np.uint8) * alpha
        + base[mask > 0].astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    return overlay


class ImageCanvas(QGraphicsView):
    """Canvas hiển thị ảnh với zoom + pan + vẽ segment để measure."""

    measure_segment_drawn = Signal(int, int, int, int)  # row1, col1, row2, col2
    roi_drawn = Signal(int, int, int, int)  # x, y, w, h (image coords)
    persistent_roi_changed = Signal(int, int, int, int)  # khi user kéo overlay
    mouse_moved = Signal(int, int, object)  # row, col, pixel value (int|tuple|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
            | QPainter.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#0e1117"))
        self.setMouseTracking(True)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._show_mask: bool = True

        # Measure-segment tools
        self._measure_mode = False
        self._segment_start: Optional[QPointF] = None
        self._segment_item: Optional[QGraphicsLineItem] = None

        # ROI rectangle tools
        self._roi_mode = False
        self._roi_start: Optional[QPointF] = None
        self._roi_item: Optional[QGraphicsRectItem] = None

        # Persistent ROI overlay (movable rect — dùng cho ROI tool)
        self._roi_persistent: Optional[_MovableRoiItem] = None
        self._roi_persistent_size: tuple[int, int] = (0, 0)
        self._roi_persistent_suppress: bool = False

    # -- public API ---------------------------------------------------------
    def set_image(self, img: np.ndarray) -> None:
        self._image = img
        self._render()
        self.fit_to_view()

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask
        self._render()

    def set_show_mask(self, show: bool) -> None:
        self._show_mask = show
        self._render()

    def get_mask(self) -> Optional[np.ndarray]:
        return self._mask

    def _render(self) -> None:
        if self._image is None:
            return
        if self._mask is not None and self._show_mask:
            display = _blend_mask(self._image, self._mask)
        else:
            display = self._image
        pixmap = numpy_to_qpixmap(display)
        # Lưu persistent ROI để dựng lại sau scene.clear()
        keep_roi = None
        if self._roi_persistent is not None:
            pos = self._roi_persistent.pos()
            keep_roi = (int(pos.x()), int(pos.y()),
                        self._roi_persistent_size[0], self._roi_persistent_size[1])
        self._scene.clear()
        self._segment_item = None
        self._roi_item = None
        self._roi_persistent = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if keep_roi is not None:
            self.set_persistent_roi(*keep_roi)

    def fit_to_view(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self.resetTransform()

    def set_measure_mode(self, enabled: bool) -> None:
        self._measure_mode = enabled
        if enabled:
            self._roi_mode = False
        self.setDragMode(
            QGraphicsView.NoDrag
            if (enabled or self._roi_mode)
            else QGraphicsView.ScrollHandDrag
        )
        self.setCursor(
            Qt.CrossCursor
            if (enabled or self._roi_mode)
            else Qt.ArrowCursor
        )

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        if enabled:
            self._measure_mode = False
        self.setDragMode(
            QGraphicsView.NoDrag
            if (enabled or self._measure_mode)
            else QGraphicsView.ScrollHandDrag
        )
        self.setCursor(
            Qt.CrossCursor
            if (enabled or self._measure_mode)
            else Qt.ArrowCursor
        )

    # ---- Persistent ROI overlay (movable) ----
    def set_persistent_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Hiển thị overlay rect kéo được (cho ROI tool); gọi nhiều lần để cập nhật."""
        if self._image is None or w <= 0 or h <= 0:
            return
        if self._roi_persistent is not None:
            # Update existing item to tránh recreate (giữ trạng thái drag)
            self._roi_persistent_suppress = True
            self._roi_persistent.setRect(QRectF(0, 0, w, h))
            self._roi_persistent.setPos(x, y)
            self._roi_persistent_size = (w, h)
            self._roi_persistent_suppress = False
            return
        pen = QPen(QColor("#36c5d6"), 0)
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        rect_item = _MovableRoiItem(QRectF(0, 0, w, h),
                                     self._on_persistent_roi_change)
        rect_item.setPen(pen)
        rect_item.setBrush(QBrush(QColor(54, 197, 214, 40)))
        rect_item.setPos(x, y)
        rect_item.setZValue(100)
        self._scene.addItem(rect_item)
        self._roi_persistent = rect_item
        self._roi_persistent_size = (w, h)

    def clear_persistent_roi(self) -> None:
        if self._roi_persistent is not None:
            self._scene.removeItem(self._roi_persistent)
            self._roi_persistent = None
            self._roi_persistent_size = (0, 0)

    def _on_persistent_roi_change(self) -> None:
        if self._roi_persistent_suppress or self._roi_persistent is None or self._image is None:
            return
        pos = self._roi_persistent.pos()
        w, h = self._roi_persistent_size
        H, W = self._image.shape[:2]
        # Clamp
        x = max(0, min(int(pos.x()), W - w))
        y = max(0, min(int(pos.y()), H - h))
        if x != int(pos.x()) or y != int(pos.y()):
            self._roi_persistent_suppress = True
            self._roi_persistent.setPos(x, y)
            self._roi_persistent_suppress = False
        self.persistent_roi_changed.emit(x, y, w, h)

    def has_image(self) -> bool:
        return self._image is not None

    def current_image(self) -> Optional[np.ndarray]:
        return self._image

    # -- events -------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._pixmap_item is None:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if (
            self._measure_mode
            and event.button() == Qt.LeftButton
            and self._pixmap_item is not None
        ):
            self._segment_start = self.mapToScene(event.position().toPoint())
            if self._segment_item is not None:
                self._scene.removeItem(self._segment_item)
            pen = QPen(QColor("#ffd166"), 0)
            pen.setCosmetic(True)
            pen.setWidth(2)
            self._segment_item = self._scene.addLine(
                self._segment_start.x(),
                self._segment_start.y(),
                self._segment_start.x(),
                self._segment_start.y(),
                pen,
            )
            return
        if (
            self._roi_mode
            and event.button() == Qt.LeftButton
            and self._pixmap_item is not None
        ):
            self._roi_start = self.mapToScene(event.position().toPoint())
            if self._roi_item is not None:
                self._scene.removeItem(self._roi_item)
            pen = QPen(QColor("#6ea8ff"), 0)
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            self._roi_item = self._scene.addRect(
                QRectF(self._roi_start, self._roi_start), pen
            )
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._pixmap_item is not None:
            scene_pt = self.mapToScene(event.position().toPoint())
            col = int(scene_pt.x())
            row = int(scene_pt.y())
            if self._image is not None and 0 <= row < self._image.shape[0] and 0 <= col < self._image.shape[1]:
                px = self._image[row, col]
                if hasattr(px, "tolist"):
                    px = px.tolist()
                self.mouse_moved.emit(row, col, px)
        if (
            self._measure_mode
            and self._segment_start is not None
            and self._segment_item is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            self._segment_item.setLine(
                self._segment_start.x(), self._segment_start.y(), end.x(), end.y()
            )
        if (
            self._roi_mode
            and self._roi_start is not None
            and self._roi_item is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            self._roi_item.setRect(QRectF(self._roi_start, end).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if (
            self._measure_mode
            and event.button() == Qt.LeftButton
            and self._segment_start is not None
            and self._image is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            r1 = int(np.clip(self._segment_start.y(), 0, self._image.shape[0] - 1))
            c1 = int(np.clip(self._segment_start.x(), 0, self._image.shape[1] - 1))
            r2 = int(np.clip(end.y(), 0, self._image.shape[0] - 1))
            c2 = int(np.clip(end.x(), 0, self._image.shape[1] - 1))
            self._segment_start = None
            self.measure_segment_drawn.emit(r1, c1, r2, c2)
            return
        if (
            self._roi_mode
            and event.button() == Qt.LeftButton
            and self._roi_start is not None
            and self._image is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._roi_start, end).normalized()
            self._roi_start = None
            x = int(np.clip(rect.x(), 0, self._image.shape[1] - 1))
            y = int(np.clip(rect.y(), 0, self._image.shape[0] - 1))
            w = int(np.clip(rect.width(), 1, self._image.shape[1] - x))
            h = int(np.clip(rect.height(), 1, self._image.shape[0] - y))
            self.roi_drawn.emit(x, y, w, h)
            return
        super().mouseReleaseEvent(event)

    def clear_roi(self) -> None:
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
