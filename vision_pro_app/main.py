"""Vision Pro Image Studio - Entry point."""
import os
import sys

os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.styles import VISION_PRO_STYLE


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Vision Pro Image Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Vision Pro Studio")

    font = QFont("SF Pro Display", 10)
    if not font.exactMatch():
        for family in ("Segoe UI Variable", "Segoe UI", "Inter", "Helvetica Neue"):
            candidate = QFont(family, 10)
            if candidate.exactMatch():
                font = candidate
                break
    app.setFont(font)
    app.setStyleSheet(VISION_PRO_STYLE)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
