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
        self._err_last = ""    # throttle error lặp lại → tránh ngập log GUI
        self._err_t = 0.0

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
                    self._emit_error(repr(exc))
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

    def _emit_error(self, msg: str, every: float = 2.0):
        """Lỗi giống nhau lặp liên tục (PLC rớt mạng) → chỉ phát ≤1 lần/2s.
        Không throttle thì 5 poll/giây × hàng giờ = hàng vạn event dồn vào
        GUI thread."""
        now = time.time()
        if msg != self._err_last or now - self._err_t >= every:
            self._err_last = msg
            self._err_t = now
            self.error.emit(msg)

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

    Kèm ``scan_check_addr`` (mặc định 500) — PLC bật =1 để hỏi "đã quét SN
    chưa". Hai thanh ghi được đọc GỘP 1 giao dịch Modbus khi đủ gần nhau
    (≤120 word — giới hạn 1 frame Modbus là 125); xa hơn thì đọc 2 lần.

    Handshake "1 emit / 1 xung" + ACK-TRƯỚC-EMIT-SAU (cờ ``_verdict_armed``):
        thấy ≠0 lần đầu → ghi 0 (ack); ack OK mới emit rồi hạ cờ. PLC còn
        giữ =1 các vòng sau → chỉ re-ack, KHÔNG emit lại. Khi thanh ghi về
        0 → arm lại cho xung kế. Ack lỗi (mạng chập chờn) → giữ cờ, poll sau
        thử lại — không mất verdict.

    Hai lỗi của bản trước, sửa cùng lúc:
      - ``_prev_val`` (so sánh đổi giá trị) + ghi 0 SAU emit, không kiểm tra
        ghi: 1 lần ghi-0 trượt là thanh ghi kẹt 1, mọi PASS sau bị bỏ vĩnh
        viễn → line đứng (giống treo).
      - ack-trước nhưng KHÔNG nhớ đã xử lý xung: nếu PLC giữ =1 vài vòng
        poll thì emit 5 lần/giây → bão thread SFC/ảnh/Excel → cạn tài
        nguyên → ĐƠ CỨNG máy (phải tắt Task Manager). Cờ armed chặn cả hai.
    """

    _SPAN_MAX = 120   # đọc gộp khi 2 thanh ghi cách nhau ≤ 120 word (Modbus max 125)

    def __init__(self, ip: str, poll_interval: float = 0.2,
                 result_addr: int = 300, scan_addr: int = 250,
                 scan_check_addr: int = 500,
                 parent: QObject | None = None):
        super().__init__(poll_interval=poll_interval, parent=parent)
        self.ip = ip
        self.result_addr = result_addr
        self.scan_addr = scan_addr
        self.scan_check_addr = int(scan_check_addr or 0)  # 0 → tắt scan-check
        # "armed" = sẵn sàng nhận XUNG kế tiếp. Đặt False ngay khi đã xử lý 1
        # xung; chỉ arm lại khi thanh ghi đã về 0. Nhờ vậy verdict/scan_check
        # chỉ bắn 1 LẦN cho mỗi xung — kể cả khi PLC GIỮ thanh ghi =1 nhiều
        # vòng poll (5 lần/giây). KHÔNG có cờ này thì giữ-cao = bão verdict →
        # đẻ hàng loạt thread SFC/ảnh/Excel → cạn tài nguyên → đơ cứng máy.
        self._verdict_armed = True
        self._scanchk_armed = True

    def _connect(self):
        import h3u_h5u
        h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0)
        h3u_h5u.write_data_h3u(self.ip, self.scan_addr, 2)
        if self.scan_check_addr:
            h3u_h5u.write_data_h3u(self.ip, self.scan_check_addr, 0)
        v = h3u_h5u.read_data_h3u(self.ip, self.result_addr)
        if v is None:
            raise RuntimeError(f"PLC {self.ip} không phản hồi (Modbus TCP)")

    def _disconnect(self):
        try:
            import h3u_h5u
            h3u_h5u.close_all()
        except Exception:
            pass

    def _read_regs(self):
        """(verdict, scan_check) — phần tử None nếu đọc lỗi/không dùng."""
        import h3u_h5u
        if not self.scan_check_addr:
            return h3u_h5u.read_data_h3u(self.ip, self.result_addr), None
        lo = min(self.result_addr, self.scan_check_addr)
        hi = max(self.result_addr, self.scan_check_addr)
        span = hi - lo + 1
        if span <= self._SPAN_MAX:
            vals = h3u_h5u.read_multi_data_h3u(self.ip, lo, span)
            if not vals or len(vals) < span:
                return None, None
            return vals[self.result_addr - lo], vals[self.scan_check_addr - lo]
        # 2 thanh ghi quá xa nhau (quá 1 frame Modbus) → đành đọc 2 lần
        return (h3u_h5u.read_data_h3u(self.ip, self.result_addr),
                h3u_h5u.read_data_h3u(self.ip, self.scan_check_addr))

    def _poll(self):
        import h3u_h5u
        val, chk = self._read_regs()
        # 1) verdict reg 300 — 1 emit / 1 xung, ack-trước-emit-sau
        if val in (1, 2):
            if self._verdict_armed:
                # xung mới → ghi 0 (ack). Ghi OK mới emit + nhả "armed".
                if h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0):
                    self._verdict_armed = False
                    if val == 1:
                        self.result.emit({"ok": True, "result": "PASS"})
                    else:
                        self.result.emit({"ok": False, "result": "FAIL"})
                else:
                    # ack lỗi → KHÔNG emit, giữ armed, poll sau thử lại (không mất)
                    self._emit_error(
                        f"Ghi ack reg {self.result_addr}=0 thất bại — thử lại poll sau")
            else:
                # đã xử lý xung này rồi, PLC còn giữ =1 → chỉ re-ack, KHÔNG emit lại
                h3u_h5u.write_data_h3u(self.ip, self.result_addr, 0)
        elif val == 0:
            self._verdict_armed = True   # thanh ghi đã nhả → sẵn sàng xung kế
        # 2) check "đã quét SN chưa" reg 500 — cũng 1 emit / 1 xung
        if chk == 1:
            if self._scanchk_armed:
                if h3u_h5u.write_data_h3u(self.ip, self.scan_check_addr, 0):
                    self._scanchk_armed = False
                    self.scan_check.emit()
                else:
                    self._emit_error(
                        f"Ghi ack reg {self.scan_check_addr}=0 thất bại — thử lại poll sau")
            else:
                h3u_h5u.write_data_h3u(self.ip, self.scan_check_addr, 0)
        elif chk == 0:
            self._scanchk_armed = True
