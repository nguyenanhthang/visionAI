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
import config
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


def validate_employee_api(token_url: str, employee_url_prefix: str,
                          badge_id: str, request_timeout: float = 5.0):
    """Gọi API xác thực nhân viên — trả về (ok, data).

    ok=True  → data = {"staff_id":..., "staff_name":...}
    ok=False → data = thông báo lỗi (str)
    """
    if requests is None:
        return False, "thư viện requests chưa cài"
    try:
        r = requests.get(token_url, timeout=request_timeout)
        r.raise_for_status()
        token = r.json()["data"]["token"]

        r = requests.get(
            employee_url_prefix + badge_id,
            headers={"token": token},
            timeout=request_timeout,
        )
        if r.status_code != 200:
            return False, f"API phản hồi {r.status_code}"

        payload = r.json().get("data")
        if not payload or payload == "null":
            return False, f"Mã nhân viên '{badge_id}' không tồn tại."

        return True, {
            "staff_id": payload["staffCode"],
            "staff_name": payload["staffName"],
        }
    except requests.Timeout:
        return False, "API timeout — kiểm tra kết nối mạng."
    except Exception as exc:
        return False, f"Lỗi gọi API: {exc}"


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
        # if not getattr(config, "ON_OFF_SFC", True):
        #     self._validate_local(badge_id)
        # else:
        self._validate_api(badge_id)

    def _validate_local(self, badge_id: str):
        name = employees.lookup(badge_id)
        if name is None:
            self.error.emit(f"Mã nhân viên '{badge_id}' không tồn tại.")
            return
        self.login_ok.emit({"staff_id": badge_id, "staff_name": name})

    def _validate_api(self, badge_id: str):
        ok, data = validate_employee_api(
            self.token_url, self.employee_url_prefix,
            badge_id, self.request_timeout,
        )
        if ok:
            self.login_ok.emit(data)
        else:
            self.error.emit(data)


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
        plc_ip: str = "",
        plc_scan_result_addr: int = 250,
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
        self.plc_ip = plc_ip
        self.plc_scan_result_addr = plc_scan_result_addr
        self.request_timeout = request_timeout

        self._stop = False
        self._serial = None
        self._buf = ""   # đệm byte chưa tách hết thành mã hoàn chỉnh

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
                # Tắt quét SN (toggle "Quét SN" OFF) → tự ghi D250=1 (pass),
                # line chạy không cần quét tay.
                if not getattr(config, "SCAN_ENABLED", True):
                    self._write_plc(1)
                    time.sleep(0.03)
                else:
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
        """Trả về 1 mã hoàn chỉnh (đã tách theo CR/LF).

        Scanner kết thúc mỗi lần quét bằng CR (``\\r``). Đọc thô vào buffer
        rồi cắt theo ``\\r``/``\\n``: nếu 2 lần quét dồn vào cùng 1 lần đọc
        (lúc app đang bận) thì mã thứ 2 vẫn nằm lại buffer cho vòng sau —
        không gộp 2 mã, không để ``\\r`` lọt vào giữa mã (gây tên file lỗi).
        """
        # 1) còn mã hoàn chỉnh trong buffer → trả ngay, khỏi chờ serial
        code = self._pop_code()
        if code:
            return code
        # 2) đọc thêm từ serial rồi tách lại
        try:
            raw = self._serial.read(self.read_size)
        except Exception as exc:
            self.error.emit(f"Lỗi đọc serial: {exc}")
            time.sleep(1.0)
            return ""
        if raw:
            self._buf += raw.decode("ascii", errors="ignore")
        return self._pop_code() or ""

    def _pop_code(self) -> str:
        """Lấy mã đầu tiên còn nguyên trong buffer (tới ký tự CR/LF). '' nếu chưa có."""
        while self._buf:
            cands = [i for i in (self._buf.find("\r"), self._buf.find("\n")) if i >= 0]
            if not cands:
                return ""               # chưa có terminator → chờ đọc thêm
            pos = min(cands)
            code = self._buf[:pos].strip()
            j = pos
            while j < len(self._buf) and self._buf[j] in "\r\n":
                j += 1                  # nhảy qua mọi CR/LF liên tiếp
            self._buf = self._buf[j:]
            if code:
                return code             # bỏ qua đoạn rỗng, tìm mã kế tiếp
        return ""

    def _check_api(self, code: str) -> bool:
        if not getattr(config, "ON_OFF_SFC", True):
            # Không cấu hình API → coi như "lấy được mã" → D250 = 1.
            return True
        url = self.sn_check_prefix + code + self.sn_check_suffix
        try:
            r = requests.get(url, timeout=self.request_timeout)
            if r.text == "0":
                return True
            self.error.emit(f"API trả {r.text} cho mã {code}")
            return False
        except requests.Timeout:
            self.error.emit(f"API timeout khi check mã {code}")
            return False
        except Exception as exc:
            self.error.emit(f"Lỗi gọi API check: {exc}")
            return False

    def _write_plc(self, value: int):
        if not self.plc_ip:
            return
        try:
            import h3u_h5u
            if not h3u_h5u.write_data_h3u(self.plc_ip, self.plc_scan_result_addr, value):
                self.error.emit(f"Ghi PLC reg {self.plc_scan_result_addr} thất bại")
        except Exception as exc:
            self.error.emit(f"Lỗi ghi PLC reg {self.plc_scan_result_addr}: {exc}")