"""PLC worker chạy trên background thread, giao tiếp với GUI qua queue.

Pattern:
    - Worker thread: poll PLC, push PLCEvent vào queue (thread-safe).
    - Main (GUI) thread: dùng tk.after() để drain queue và update widget.
    - KHÔNG được sửa widget Tk từ worker thread - Tk không thread-safe.

Tích hợp PLC thật:
    Kế thừa PLCWorker và override _connect / _disconnect / _poll.
    Trong _poll dùng self._emit(type, data) để gửi event về GUI.
"""

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PLCEvent:
    type: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class PLCWorker:
    """Base class: vòng lặp poll PLC trên thread riêng."""

    def __init__(self, event_queue: "queue.Queue[PLCEvent]", poll_interval: float = 0.1):
        self.event_queue = event_queue
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="PLCWorker")
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _emit(self, event_type: str, data: Any = None):
        self.event_queue.put(PLCEvent(type=event_type, data=data))

    def _run(self):
        try:
            self._connect()
            self._emit("connected")
            while not self._stop_event.is_set():
                try:
                    self._poll()
                except Exception as exc:
                    self._emit("error", repr(exc))
                # interruptible sleep - stop() sẽ wake ngay
                if self._stop_event.wait(self.poll_interval):
                    break
        except Exception as exc:
            self._emit("fatal", repr(exc))
        finally:
            try:
                self._disconnect()
            finally:
                self._emit("disconnected")

    # ---- override để giao tiếp PLC thật ----
    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _poll(self):
        pass


class SimulatedPLCWorker(PLCWorker):
    """Worker giả lập để test GUI khi chưa có PLC thật."""

    def __init__(self, event_queue, poll_interval: float = 1.5):
        super().__init__(event_queue, poll_interval)
        self._counter = 0

    def _poll(self):
        self._counter += 1
        is_ok = random.random() > 0.25
        self._emit(
            "result",
            {"ok": is_ok, "product_id": f"P{self._counter:04d}"},
        )
