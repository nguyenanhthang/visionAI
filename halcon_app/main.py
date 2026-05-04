"""HALCON Vision Studio - Entry point.

Ứng dụng PySide6 sử dụng MVTec HALCON cho các tác vụ machine vision:
- Load / acquire image
- Threshold + Blob analysis (region features)
- Edge detection (Canny / Sobel)
- Shape-based template matching
- Measure 1D (đo cạnh, khoảng cách)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.styles import HALCON_STYLE


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("HALCON Vision Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VisionAI")

    font = QFont("Segoe UI", 10)
    if not font.exactMatch():
        for family in ("Inter", "SF Pro Display", "Helvetica Neue", "Arial"):
            candidate = QFont(family, 10)
            if candidate.exactMatch():
                font = candidate
                break
    app.setFont(font)
    app.setStyleSheet(HALCON_STYLE)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
