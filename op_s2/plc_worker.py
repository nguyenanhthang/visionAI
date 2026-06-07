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


class CP2E_PLCWorker(PLCWorker):
    """Đọc Omron CP2E qua FINS/TCP (cp2e.py).

    Poll DM word ``result_addr`` (mặc định D300) — AOI ghi verdict vào đây:
        1 → OK   → emit {"ok": True,  "result": "PASS"}
        2 → NG   → emit {"ok": False, "result": "FAIL"}
        khác → idle, không emit.

    Chỉ emit khi giá trị thay đổi để tránh push trùng; sau khi app ghi
    nhận verdict thì ghi lại 0 vào register để PLC bắn lần sau.
    """

    def __init__(self, ip: str, poll_interval: float = 0.2,
                 result_addr: int = 300, scan_addr: int = 250,
                 parent: QObject | None = None):
        super().__init__(poll_interval=poll_interval, parent=parent)
        self.ip = ip
        self.result_addr = result_addr
        self.scan_addr = scan_addr
        self._prev_val = None

    def _connect(self):
        import cp2e
        cp2e.write_data_cp2e(self.ip, self.result_addr, 0)
        cp2e.write_data_cp2e(self.ip, self.scan_addr, 2)
        v = cp2e.read_data_cp2e(self.ip, self.result_addr)
        if v is None:
            raise RuntimeError(f"PLC {self.ip} không phản hồi (FINS/TCP)")
        self._prev_val = v

    def _disconnect(self):
        try:
            import cp2e
            cp2e.close_all()
        except Exception:
            pass

    def _poll(self):
        import cp2e
        val = cp2e.read_data_cp2e(self.ip, self.result_addr)
        if val is None or val == self._prev_val:
            return
        self._prev_val = val
        if val == 1:
            self.result.emit({"ok": True, "result": "PASS"})
            cp2e.write_data_cp2e(self.ip, self.result_addr, 0)
        elif val == 2:
            self.result.emit({"ok": False, "result": "FAIL"})
            cp2e.write_data_cp2e(self.ip, self.result_addr, 0)
