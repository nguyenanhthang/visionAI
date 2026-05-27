"""Entry point — PySide6 version of the Vision AI inspection app.

Workflow:
    1. Hiển thị LoginWindow (modal). Đóng → có employee_id hoặc None.
    2. Nếu login thành công → mở MainWindow.

So với bản CustomTkinter cũ: dùng QApplication duy nhất cho cả 2 cửa sổ,
worker chạy trên QThread + emit Signal thay vì queue.Queue + after().
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from login_window import LoginWindow
from main_window import MainWindow


def load_stylesheet() -> str:
    qss_path = Path(__file__).with_name("styles.qss")
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    return ""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Vision AI")
    app.setOrganizationName("Vision AI")
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
