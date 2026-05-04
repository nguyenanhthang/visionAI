"""Image canvas: hiển thị numpy image, hỗ trợ zoom (wheel), pan (drag),
và vẽ segment cho thao tác Measure 1D."""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
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


class ImageCanvas(QGraphicsView):
    """Canvas hiển thị ảnh với zoom + pan + vẽ segment để measure."""

    measure_segment_drawn = Signal(int, int, int, int)  # row1, col1, row2, col2
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

        # Measure-segment tools
        self._measure_mode = False
        self._segment_start: Optional[QPointF] = None
        self._segment_item: Optional[QGraphicsLineItem] = None

    # -- public API ---------------------------------------------------------
    def set_image(self, img: np.ndarray) -> None:
        self._image = img
        pixmap = numpy_to_qpixmap(img)
        self._scene.clear()
        self._segment_item = None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_to_view()

    def fit_to_view(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self.resetTransform()

    def set_measure_mode(self, enabled: bool) -> None:
        self._measure_mode = enabled
        self.setDragMode(
            QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag
        )
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

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
        super().mouseReleaseEvent(event)
