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


# Nhãn build — in vào banner crash.log + tiêu đề cửa sổ. ĐỔI MỖI LẦN BUILD
# để biết chắc máy trạm đang chạy bản nào (log 12/06 toàn dump của build cũ).
APP_BUILD = "op_v7 2026-06-12.2 a11y-block+batch-log+preimport"


def install_a11y_blocker(app):
    """Chặn WM_GETOBJECT → tool UIA/MSAA (remote desktop, AV, agent giám
    sát…) KHÔNG attach được vào cửa sổ app.

    crash.log bắt được GUI đơ cứng ≥15s NGAY TRONG QTextEdit.append:
    mỗi lần widget đổi nội dung, Windows thông báo accessibility ĐỒNG BỘ
    sang client đang attach — client treo là app treo theo (chuỗi
    0x8001010d dày đặc trong log = đúng họ COM input-sync này). Chặn từ
    message ĐẦU TIÊN thì Qt không bao giờ kích hoạt accessibility → cũng
    không bắn event nào nữa. App kiosk xưởng không cần screen-reader.
    Tắt bằng DISABLE_ACCESSIBILITY = False trong config nếu cần.
    """
    if sys.platform != "win32" or not getattr(config, "DISABLE_ACCESSIBILITY", True):
        return None
    try:
        import ctypes.wintypes as wt
        from PySide6.QtCore import QAbstractNativeEventFilter

        class _NoA11y(QAbstractNativeEventFilter):
            _WM_GETOBJECT = 0x003D
            _logged = False

            def nativeEventFilter(self, etype, message):
                try:
                    msg = ctypes.cast(int(message), ctypes.POINTER(wt.MSG)).contents
                    if msg.message == self._WM_GETOBJECT:
                        if not self._logged:
                            # bằng chứng hiện trường: blocker ĐANG chạy và
                            # THẬT SỰ có tool dò accessibility vào app
                            _NoA11y._logged = True
                            crashlog.note("A11Y", "Đã chặn WM_GETOBJECT đầu tiên "
                                          "— có tool UIA/MSAA đang dò app này")
                        return True, 0   # nuốt → không cấp accessibility object
                except Exception:
                    pass
                return False, 0

        flt = _NoA11y()
        app.installNativeEventFilter(flt)
        return flt   # PHẢI giữ ref — Qt không own filter
    except Exception:
        return None


def main():
    crashlog.install(build=APP_BUILD)   # ghi crash.log nếu app văng/treo
    load_settings_overrides()  # patch config từ settings.json (nếu có)
    h3u_h5u.configure(port=getattr(config, "PLC_PORT", 502))  # cổng Modbus TCP

    # Nạp TRƯỚC các lib nặng. Import lười giữa ca (worker import xlrd lúc
    # verdict đầu tiên) bị hook import của PySide6 chạy inspect.getsource →
    # giữ GIL + khóa import-lock nhiều giây, worker khác đứng chờ
    # (crash.log op_s2 06:32 12/06). Nạp ở đây thì tốn lúc khởi động,
    # không tốn giữa ca.
    for _m in ("xlrd", "xlwt", "requests"):
        try:
            __import__(_m)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app._a11y_filter = install_a11y_blocker(app)   # giữ ref suốt vòng đời app
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
    window.setWindowTitle(f"{window.windowTitle()}  ·  {APP_BUILD}")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
