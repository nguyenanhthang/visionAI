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


class H3U_PLCWorker(PLCWorker):
    """Đọc Inovance H3U/H5U qua Modbus TCP (h3u_h5u.py).

    Override _poll để decode register thật. Scaffold dưới đọc 4 register
    liên tiếp từ ``trigger_addr`` và phát hiện rising edge ở word đầu
    và cuối — sửa lại theo wiring line của bạn.
    """

    def __init__(self, ip: str, poll_interval: float = 0.2,
                 trigger_addr: int = 0,
                 parent: QObject | None = None):
        super().__init__(poll_interval=poll_interval, parent=parent)
        self.ip = ip
        self.trigger_addr = trigger_addr
        self._prev_trigger_right = 0
        self._prev_trigger_left = 0

    def _connect(self):
        import h3u_h5u
        v = h3u_h5u.read_data_h3u(self.ip, self.trigger_addr)
        if v is None:
            raise RuntimeError(f"PLC {self.ip} không phản hồi (Modbus TCP)")

    def _disconnect(self):
        try:
            import h3u_h5u
            h3u_h5u.close_all()
        except Exception:
            pass

    def _poll(self):
        import h3u_h5u
        regs = h3u_h5u.read_multi_data_h3u(self.ip, self.trigger_addr, 4)
        if not regs:
            return

        trig_right = regs[0]
        trig_left  = regs[3]

        # rising edge → emit result giả (thay bằng logic AOI thật)
        if trig_right and not self._prev_trigger_right:
            self.result.emit({"product_id": "RIGHT", "ok": True})
        if trig_left and not self._prev_trigger_left:
            self.result.emit({"product_id": "LEFT", "ok": True})

        self._prev_trigger_right = trig_right
        self._prev_trigger_left  = trig_left
