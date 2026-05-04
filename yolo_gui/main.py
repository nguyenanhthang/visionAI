"""
YOLO GUI - Entry Point
Khởi chạy ứng dụng YOLO GUI với PySide6
"""
import sys
import os

# Kích hoạt high DPI scaling trước khi tạo QApplication
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.main_window import MainWindow
from app.resources.styles import DARK_THEME


def main():
    """Hàm chính khởi động ứng dụng."""
    # Bật high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO GUI")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("YOLO GUI")

    # Áp dụng font mặc định
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Áp dụng dark theme
    app.setStyleSheet(DARK_THEME)

    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
