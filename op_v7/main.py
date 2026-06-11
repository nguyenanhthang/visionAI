"""Entry point — PySide6 version of the Riser cable inspection app.

Workflow:
    1. Hiển thị LoginWindow (modal). Đóng → có employee_id hoặc None.
    2. Nếu login thành công → mở MainWindow.

So với bản CustomTkinter cũ: dùng QApplication duy nhất cho cả 2 cửa sổ,
worker chạy trên QThread + emit Signal thay vì queue.Queue + after().
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

import config
import h3u_h5u
import crashlog
from login_window import LoginWindow
from main_window import MainWindow
from settings_window import load_settings_overrides
import ctypes
# Bật dark mode aware cho Windows
try:
    ctypes.windll.dwmapi.DwmSetWindowAttribute
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    pass

def resource_path(rel: str) -> Path:
    """Trỏ đúng đường dẫn cả khi chạy script lẫn khi chạy exe PyInstaller."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    return Path(__file__).resolve().with_name(rel)


def load_stylesheet() -> str:
    qss_path = resource_path("styles.qss")
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    print(f"[WARN] Không tìm thấy stylesheet: {qss_path}", file=sys.stderr)
    return ""


def main():
    crashlog.install()         # ghi crash.log nếu app văng/abort
    load_settings_overrides()  # patch config từ settings.json (nếu có)
    h3u_h5u.configure(port=getattr(config, "PLC_PORT", 502))  # cổng Modbus TCP

    app = QApplication(sys.argv)
    app.setApplicationName("Riser cable")
    app.setOrganizationName("Riser cable")
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet())

    login = LoginWindow()
    if login.exec() != LoginWindow.DialogCode.Accepted:
        sys.exit(0)

    employee = login.result_employee  # {"id": ..., "name": ...}
    if not employee:
        sys.exit(0)

    window = MainWindow(employee_id=employee["id"], employee_name=employee["name"])
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
