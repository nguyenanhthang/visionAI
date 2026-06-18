"""Badge scanner worker - đọc serial + validate API trên background thread.

Lý do tách thread:
    - serial.readline() và requests.get() đều blocking.
    - Nếu chạy trên main Tk thread, GUI sẽ đứng hình.
    - Cùng pattern PLCWorker: thread đẩy event qua queue, GUI drain qua after().

Tích hợp:
    queue = Queue()
    scanner = BadgeScanner(queue, port="COM4", token_url=..., employee_url_prefix=...)
    scanner.start()
    # main thread drain queue, xử lý event
    scanner.stop()   # khi đóng app hoặc login xong
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import serial
except ImportError:
    serial = None

try:
    import requests
except ImportError:
    requests = None

import employees


@dataclass
class ScanEvent:
    type: str  # connected | disconnected | scan | login_ok | error
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class BadgeScanner:
    """Đọc badge từ scanner serial, xác thực rồi emit event qua queue."""

    def __init__(
        self,
        event_queue: "queue.Queue[ScanEvent]",
        port: str = "COM4",
        baudrate: int = 9600,
        read_size: int = 8,
        serial_timeout: float = 0.5,
        token_url: str = "",
        employee_url_prefix: str = "",
        request_timeout: float = 5.0,
    ):
        self.event_queue = event_queue
        self.port = port
        self.baudrate = baudrate
        self.read_size = read_size
        self.serial_timeout = serial_timeout
        self.token_url = token_url
        self.employee_url_prefix = employee_url_prefix
        self.request_timeout = request_timeout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serial = None

    def start(self):
        if serial is None:
            self._emit("error", "pyserial chưa cài: pip install pyserial")
            return
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="BadgeScanner")
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _emit(self, event_type: str, data: Any = None):
        self.event_queue.put(ScanEvent(type=event_type, data=data))

    def _run(self):
        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=8,
                timeout=self.serial_timeout,
            )
            self._emit("connected", self.port)
        except Exception as exc:
            self._emit("error", f"Mở cổng {self.port} thất bại: {exc}")
            return

        try:
            while not self._stop_event.is_set():
                badge_id = self._read_badge()
                if not badge_id:
                    continue
                self._emit("scan", badge_id)
                self._validate(badge_id)
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self._emit("disconnected")

    def _read_badge(self) -> str:
        try:
            raw = self._serial.readline(self.read_size)
        except Exception as exc:
            self._emit("error", f"Lỗi đọc serial: {exc}")
            self._stop_event.wait(1.0)
            return ""

        if not raw:
            return ""  # timeout - không có dữ liệu, vòng lặp tiếp tục

        text = raw.decode("ascii", errors="ignore").strip()
        # Format chuẩn: 8 ký tự bắt đầu bằng V/v (theo code gốc của bạn)
        if len(text) == 8 and text[0].lower() == "v":
            return text
        if text:
            self._emit("error", f"Mã quét không hợp lệ: {text!r}")
        return ""

    def _validate(self, badge_id: str):
        """Xác thực badge. Ưu tiên API; nếu không có config thì fallback local."""
        if not self.token_url or not self.employee_url_prefix or requests is None:
            self._validate_local(badge_id)
            return
        self._validate_api(badge_id)

    def _validate_local(self, badge_id: str):
        name = employees.lookup(badge_id)
        if name is None:
            self._emit("error", f"Mã nhân viên '{badge_id}' không tồn tại.")
            return
        self._emit("login_ok", {"staff_id": badge_id, "staff_name": name})

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
                self._emit("error", f"API phản hồi {r.status_code}")
                return

            payload = r.json().get("data")
            if not payload or payload == "null":
                self._emit("error", f"Mã nhân viên '{badge_id}' không tồn tại.")
                return

            self._emit("login_ok", {
                "staff_id": payload["staffCode"],
                "staff_name": payload["staffName"],
            })
        except requests.Timeout:
            self._emit("error", "API timeout - kiểm tra kết nối mạng.")
        except Exception as exc:
            self._emit("error", f"Lỗi gọi API: {exc}")
