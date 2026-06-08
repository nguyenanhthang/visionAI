"""PLC worker — PySide6 version (QThread + Signal).

So với bản threading + queue.Queue cũ:
    - PLCWorker là QObject, được move sang QThread.
    - Mỗi event là 1 Signal; Qt tự queue về main thread.
    - Stop cooperative: GUI gọi worker.stop() → loop break → finished emit.

Tích hợp PLC thật:
    Subclass PLCWorker, override _connect / _disconnect / _poll.
    Trong _poll dùng self.result.emit(...), self.image.emit(...), v.v.
"""

from __future__ import annotations

import random
import time
from PySide6.QtCore import QObject, Signal, Slot

class PLCWorker(QObject):
    """Base class: vòng lặp poll PLC trên thread riêng."""

    connected    = Signal()
    disconnected = Signal()
    error        = Signal(str)
    fatal        = Signal(str)
    result       = Signal(dict)   # {"product_id": ..., "ok": bool, ...}
    image        = Signal(object) # PIL.Image hoặc QImage
    scan_check   = Signal()       # PLC yêu cầu kiểm tra "đã quét SN chưa" (D500 lên 1)
    finished     = Signal()

    def __init__(self, poll_interval: float = 0.2, parent: QObject | None = None):
        super().__init__(parent)
        self.poll_interval = poll_interval
        self._stop = False

    # ---- main loop ----
    @Slot()
    def run(self):
        try:
            self._connect()
            self.connected.emit()
            
            while not self._stop:
                try:
                    self._poll()
                except Exception as exc:
                    self.error.emit(repr(exc))
                # interruptible-ish sleep
                slept = 0.0
                step = 0.05
                while slept < self.poll_interval and not self._stop:
                    time.sleep(min(step, self.poll_interval - slept))
                    slept += step

        except Exception as exc:
            self.fatal.emit(repr(exc))
        finally:
            try:
                self._disconnect()
            finally:
                self.disconnected.emit()
                self.finished.emit()

    @Slot()
    def stop(self):
        self._stop = True

    # ---- override để giao tiếp PLC thật ----
    def _connect(self):    pass
    def _disconnect(self): pass
    def _poll(self):       pass


class SimulatedPLCWorker(PLCWorker):
    """Worker giả lập để test GUI khi chưa có PLC thật."""

    def __init__(self, poll_interval: float = 1.5, parent: QObject | None = None):
        super().__init__(poll_interval=poll_interval, parent=parent)
        self._counter = 0

    def _poll(self):
        self._counter += 1
        is_ok = random.random() > 0.25
        self.result.emit({
            "ok": is_ok,
            "product_id": f"P{self._counter:04d}",
        })


class H3U_PLCWorker(PLCWorker):
    """Đọc Inovance H3U/H5U qua Modbus TCP (h3u_h5u.py).

    Poll holding register ``result_addr`` (mặc định 300) — AOI ghi verdict:
        1 → OK   → emit {"ok": True,  "result": "PASS"}
        2 → NG   → emit {"ok": False, "result": "FAIL"}
        khác → idle, không emit.

    Đồng thời poll ``scan_check_addr`` (mặc định 500) — PLC bật =1 để hỏi
    "đã quét SN chưa", chỉ bắn scan_check ở sườn lên 0→1.

    Chỉ emit verdict khi giá trị thay đổi; sau khi nhận verdict ghi lại 0.
    """

    def __init__(self, ip: str, poll_interval: float = 0.2,
                 result_addr: int = 300, scan_addr: int = 250,
                 scan_check_addr: int = 500,
                 parent: QObject | None = None):
        super().__init__(poll_interval=poll_interval, parent=parent)
        self.ip = ip
        self.result_addr = result_addr
        self.scan_addr = scan_addr
        self.scan_check_addr = scan_check_addr
        self._prev_val = None
        self._prev_scan_chk = 0

    def _connect(self):
        import h3u_h5u
        h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0)
        h3u_h5u.write_data_h3u(self.ip, self.scan_addr, 2)
        v = h3u_h5u.read_data_h3u(self.ip, self.result_addr)
        if v is None:
            raise RuntimeError(f"PLC {self.ip} không phản hồi (Modbus TCP)")
        self._prev_val = v
        # init D500 để không bắn "chưa quét" ngay lúc khởi động nếu đang =1
        self._prev_scan_chk = h3u_h5u.read_data_h3u(self.ip, self.scan_check_addr) or 0

    def _disconnect(self):
        try:
            import h3u_h5u
            h3u_h5u.close_all()
        except Exception:
            pass

    def _poll(self):
        import h3u_h5u
        # 1) verdict reg 300
        val = h3u_h5u.read_data_h3u(self.ip, self.result_addr)
        if val is not None and val != self._prev_val:
            self._prev_val = val
            if val == 1:
                self.result.emit({"ok": True, "result": "PASS"})
                h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0)
            elif val == 2:
                self.result.emit({"ok": False, "result": "FAIL"})
                h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0)
        # 2) check "đã quét SN chưa" reg 500 — chỉ bắn ở sườn lên 0→1
        chk = h3u_h5u.read_data_h3u(self.ip, self.scan_check_addr)
        if chk is not None:
            if chk == 1 and self._prev_scan_chk != 1:
                self.scan_check.emit()
            self._prev_scan_chk = chk
