"""Image canvas with drag-and-drop and aspect-preserving scaling."""

from __future__ import annotations

from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a Pillow RGBA image to a QPixmap."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    # ``data`` must outlive QImage; copying detaches the buffer.
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QFrame):
    """Centered canvas that scales the image while preserving aspect ratio."""

    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ImageCanvas")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setMinimumSize(360, 360)

        self._pixmap: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self._display = QLabel()
        self._display.setObjectName("ImageDisplay")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setText(
            "✨  Drop an image here\nor use “Open” to begin"
        )
        layout.addWidget(self._display)

    # --------------------------------------------------------------- API
    def set_image(self, img: Optional[Image.Image]) -> None:
        if img is None:
            self._pixmap = None
            self._display.setText(
                "✨  Drop an image here\nor use “Open” to begin"
            )
            self._display.setPixmap(QPixmap())
            return
        self._pixmap = pil_to_qpixmap(img)
        self._refresh()

    def clear(self) -> None:
        self.set_image(None)

    # --------------------------------------------------------------- Events
    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        self._refresh()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and self._is_image(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if self._is_image(path):
                    self.file_dropped.emit(path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    # --------------------------------------------------------------- Helpers
    def _refresh(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self._display.size()
        if target.width() <= 2 or target.height() <= 2:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)

    @staticmethod
    def _is_image(path: str) -> bool:
        lowered = path.lower()
        return lowered.endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
        )
