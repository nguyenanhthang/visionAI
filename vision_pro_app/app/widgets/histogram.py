"""Compact RGB histogram rendered with QPainter."""

from __future__ import annotations

from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame


class HistogramWidget(QFrame):
    """Renders RGB intensity distributions as soft glowing curves."""

    BUCKETS = 64

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Histogram")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(110)

        self._channels: Optional[dict[str, list[int]]] = None

    def set_image(self, img: Optional[Image.Image]) -> None:
        if img is None:
            self._channels = None
            self.update()
            return

        rgb = img.convert("RGB")
        full = rgb.histogram()  # length 768: R, G, B back to back
        bucket_size = 256 // self.BUCKETS
        channels = {}
        for idx, name in enumerate(("r", "g", "b")):
            raw = full[idx * 256 : (idx + 1) * 256]
            buckets = [
                sum(raw[i * bucket_size : (i + 1) * bucket_size])
                for i in range(self.BUCKETS)
            ]
            channels[name] = buckets
        self._channels = channels
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().paintEvent(event)
        if not self._channels:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(10, 10, -10, -10)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        peak = max(max(values) for values in self._channels.values()) or 1

        colors = {
            "r": QColor(255, 90, 110, 200),
            "g": QColor(100, 220, 140, 200),
            "b": QColor(110, 150, 255, 200),
        }

        for name, values in self._channels.items():
            path = QPainterPath()
            step = rect.width() / max(len(values) - 1, 1)
            for i, v in enumerate(values):
                x = rect.left() + i * step
                y = rect.bottom() - (v / peak) * rect.height()
                if i == 0:
                    path.moveTo(x, rect.bottom())
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)
            path.lineTo(rect.right(), rect.bottom())
            path.closeSubpath()

            color = colors[name]
            gradient = QLinearGradient(0, rect.top(), 0, rect.bottom())
            gradient.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 160))
            gradient.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 30))
            painter.fillPath(path, gradient)

            outline = QPen(color, 1.4)
            painter.setPen(outline)
            painter.drawPath(path)
