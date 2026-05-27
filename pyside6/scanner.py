"""Badge scanner worker — PySide6 version.

Trước đây dùng threading.Thread + queue.Queue, GUI drain bằng after().
Bây giờ dùng QThread + Signal:
    - Worker (QObject) sống trong QThread riêng.
    - Phát Signal về GUI → Qt tự queue qua event loop main thread.
    - Không cần lock/queue.

Tích hợp:
    self.thread = QThread()
    self.scanner = BadgeScanner(port=..., token_url=...)
    self.scanner.moveToThread(self.thread)
    self.thread.started.connect(self.scanner.run)
    self.scanner.connected.connect(self.on_connected)
    self.scanner.login_ok.connect(self.on_login_ok)
    ...
    self.thread.start()
    # khi muốn dừng:
    self.scanner.stop()
    self.thread.quit(); self.thread.wait()
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal, Slot

try:
    import serial
except ImportError:
    serial = None

try:
    import requests
except ImportError:
    requests = None

import employees


class BadgeScanner(QObject):
    """Đọc badge từ serial scanner, xác thực rồi emit signal."""

    # signal -> GUI (auto queued vì thread khác)
    connected     = Signal(str)         # port name
    disconnected  = Signal()
    scanned       = Signal(str)         # raw badge id
    login_ok      = Signal(dict)        # {"staff_id": ..., "staff_name": ...}
    error         = Signal(str)         # message
    finished      = Signal()            # khi run() kết thúc (dùng để quit thread)

    def __init__(
        self,
        port: str = "COM4",
        baudrate: int = 9600,
        read_size: int = 8,
        serial_timeout: float = 0.5,
        token_url: str = "",
        employee_url_prefix: str = "",
        request_timeout: float = 5.0,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.read_size = read_size
        self.serial_timeout = serial_timeout
        self.token_url = token_url
        self.employee_url_prefix = employee_url_prefix
        self.request_timeout = request_timeout

        self._stop = False
        self._serial = None

    # ---- main loop (chạy trên worker thread sau khi moveToThread) ----
    @Slot()
    def run(self):
        if serial is None:
            self.error.emit("pyserial chưa cài: pip install pyserial")
            self.finished.emit()
            return

        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=8,
                timeout=self.serial_timeout,
            )
        except Exception as exc:
            self.error.emit(f"Mở cổng {self.port} thất bại: {exc}")
            self.finished.emit()
            return

        self.connected.emit(self.port)
        try:
            while not self._stop:
                badge_id = self._read_badge()
                if not badge_id:
                    continue
                self.scanned.emit(badge_id)
                self._validate(badge_id)
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self.disconnected.emit()
            self.finished.emit()

    @Slot()
    def stop(self):
        """An toàn để gọi từ thread khác — chỉ set flag."""
        self._stop = True

    # ---- helpers ----
    def _read_badge(self) -> str:
        try:
            raw = self._serial.readline(self.read_size)
        except Exception as exc:
            self.error.emit(f"Lỗi đọc serial: {exc}")
            # avoid spin-loop on persistent failure
            time.sleep(1.0)
            return ""

        if not raw:
            return ""  # timeout → không có data, vòng lặp tiếp tục

        text = raw.decode("ascii", errors="ignore").strip()
        # Format chuẩn: 8 ký tự bắt đầu bằng V/v
        if len(text) == 8 and text[0].lower() == "v":
            return text
        if text:
            self.error.emit(f"Mã quét không hợp lệ: {text!r}")
        return ""

    def _validate(self, badge_id: str):
        """Ưu tiên API; nếu không config thì fallback local."""
        if not self.token_url or not self.employee_url_prefix or requests is None:
            self._validate_local(badge_id)
        else:
            self._validate_api(badge_id)

    def _validate_local(self, badge_id: str):
        name = employees.lookup(badge_id)
        if name is None:
            self.error.emit(f"Mã nhân viên '{badge_id}' không tồn tại.")
            return
        self.login_ok.emit({"staff_id": badge_id, "staff_name": name})

    def _validate_api(self, badge_id: str):
        try:
            r = requests.get(self.token_url, timeout=self.request_timeout)
            r.raise_for_status()
            token = r.json()["data"]["token"]

            r = requests.get(
                self.employee_url_prefix + badge_id,
                headers={"token": token},
                timeout=self.request_timeout,
            )
            if r.status_code != 200:
                self.error.emit(f"API phản hồi {r.status_code}")
                return

            payload = r.json().get("data")
            if not payload or payload == "null":
                self.error.emit(f"Mã nhân viên '{badge_id}' không tồn tại.")
                return

            self.login_ok.emit({
                "staff_id": payload["staffCode"],
                "staff_name": payload["staffName"],
            })
        except requests.Timeout:
            self.error.emit("API timeout — kiểm tra kết nối mạng.")
        except Exception as exc:
            self.error.emit(f"Lỗi gọi API: {exc}")


class ProductScanner(QObject):
    """Đọc mã sản phẩm từ máy quét tay, gọi API check, ghi PLC D250.

    Mỗi lần worker bóp cò scanner:
      1. Đọc mã từ serial.
      2. GET ``sn_check_prefix + code + sn_check_suffix``.
      3. status 200 → ghi D250 = 1, ngược lại (timeout / lỗi / non-200) → D250 = 2.
      4. Emit ``verdict(code, api_ok, plc_value)`` để UI cập nhật.
    """

    connected    = Signal(str)              # port name
    disconnected = Signal()
    scanned      = Signal(str)              # mã thô vừa quét được
    verdict      = Signal(str, bool, int)   # code, api_ok, plc_value đã ghi
    error        = Signal(str)
    finished     = Signal()

    def __init__(
        self,
        port: str = "COM4",
        baudrate: int = 9600,
        read_size: int = 64,
        serial_timeout: float = 0.5,
        sn_check_prefix: str = "",
        sn_check_suffix: str = "",
        plc_port: str = "",
        plc_baud: int = 9600,
        request_timeout: float = 5.0,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.read_size = read_size
        self.serial_timeout = serial_timeout
        self.sn_check_prefix = sn_check_prefix
        self.sn_check_suffix = sn_check_suffix
        self.plc_port = plc_port
        self.plc_baud = plc_baud
        self.request_timeout = request_timeout

        self._stop = False
        self._serial = None

    @Slot()
    def run(self):
        if serial is None:
            self.error.emit("pyserial chưa cài: pip install pyserial")
            self.finished.emit()
            return

        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=8,
                timeout=self.serial_timeout,
            )
        except Exception as exc:
            self.error.emit(f"Mở cổng {self.port} thất bại: {exc}")
            self.finished.emit()
            return

        self.connected.emit(self.port)
        try:
            while not self._stop:
                code = self._read_code()
                if not code:
                    continue
                self.scanned.emit(code)
                ok = self._check_api(code)
                plc_value = 1 if ok else 2
                self._write_plc(plc_value)
                self.verdict.emit(code, ok, plc_value)
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self.disconnected.emit()
            self.finished.emit()

    @Slot()
    def stop(self):
        self._stop = True

    def _read_code(self) -> str:
        try:
            raw = self._serial.readline(self.read_size)
        except Exception as exc:
            self.error.emit(f"Lỗi đọc serial: {exc}")
            time.sleep(1.0)
            return ""
        if not raw:
            return ""
        return raw.decode("ascii", errors="ignore").strip()

    def _check_api(self, code: str) -> bool:
        if not self.sn_check_prefix or requests is None:
            # Không cấu hình API → coi như "lấy được mã" → D250 = 1.
            return True
        url = self.sn_check_prefix + code + self.sn_check_suffix
        try:
            r = requests.get(url, timeout=self.request_timeout)
            if r.status_code == 200:
                return True
            self.error.emit(f"API trả {r.status_code} cho mã {code}")
            return False
        except requests.Timeout:
            self.error.emit(f"API timeout khi check mã {code}")
            return False
        except Exception as exc:
            self.error.emit(f"Lỗi gọi API check: {exc}")
            return False

    def _write_plc(self, value: int):
        if not self.plc_port:
            return
        try:
            import panasonic
            panasonic.write_data_panasonic(
                self.plc_port, self.plc_baud, "D250", value,
            )
        except Exception as exc:
            self.error.emit(f"Lỗi ghi PLC D250: {exc}")
