"""Ghi nguyên nhân khi app văng/treo để truy vết (không phụ thuộc GUI).

``install()`` bật:
  - ``faulthandler``: dump Python traceback khi gặp tín hiệu fatal — segfault,
    abort của Qt, hỏng heap (0xC0000374)…
  - ``sys.excepthook`` + ``threading.excepthook``: ghi exception CHƯA BẮT
    (cả ở thread phụ) kèm thời gian.
  - **Watchdog (Python)**: GUI gọi ``heartbeat()`` mỗi giây; quá ``stale``
    giây không nhịp → dump TẤT CẢ thread + ghi ``HANG``. Khi GUI sống lại →
    ghi ``RECOVERED`` kèm số giây kẹt (phân biệt treo thật vs khựng tạm thời).
  - **C-timer (``dump_traceback_later``)**: watchdog Python cần GIL để chạy;
    nếu GUI thread ĐƠ CỨNG mà GIỮ GIL (deadlock C/exhaustion) thì watchdog
    bị đói GIL, KHÔNG ghi được ``HANG``. ``dump_traceback_later`` chạy ở tầng
    C, nổ kể cả khi Python kẹt GIL → vẫn dump được stack lúc đơ cứng. Mỗi
    ``heartbeat()`` re-arm lại; chỉ nổ nếu nhịp tim ngừng ``stale`` giây.

ĐỌC LOG THẾ NÀO:
  - ``===== HANG … =====``      → GUI treo, xem stack để biết kẹt ở đâu.
  - ``===== HANG-STILL … =====`` → vẫn treo, dump lại để so stack (đứng hẳn?).
  - ``===== RECOVERED … =====``  → chỉ khựng tạm thời rồi tự hồi (I/O chậm),
    KHÔNG phải đứng máy.
  - ``Timeout (0:00:15)!``        → C-timer nổ: GUI ĐƠ CỨNG (giữ GIL), stack
    ngay dưới là chỗ kẹt — đây là loại "phải tắt Task Manager".
  - ``Windows fatal exception: code 0x8001010d`` → **nhiễu COM lành tính**
    (RPC_E_CANTCALLOUT_ININPUTSYNCCALL, thường do tool accessibility / AV /
    remote desktop), app VẪN chạy — bỏ qua. Chỉ lo HANG / Timeout /
    ``0xC0000374`` / access violation.

File ``crash.log`` nằm cạnh .exe (frozen) hoặc cạnh script.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

_fault_fp = None          # giữ file mở cho faulthandler suốt vòng đời process
_alive = None             # mốc heartbeat gần nhất (time.monotonic)
_hang_dumped = False      # đã dump cho lần treo hiện tại chưa (tránh spam)
_hang_stale = 15.0        # ngưỡng treo (giây) — dùng cho cả watchdog + C-timer


def log_path() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    return base / "crash.log"


def _write(header: str, text: str):
    try:
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(f"\n===== {header} {datetime.now():%Y-%m-%d %H:%M:%S} =====\n{text}\n")
    except Exception:
        pass


def heartbeat():
    """GUI gọi định kỳ (vd QTimer 1s) để báo 'còn sống'."""
    global _alive
    _alive = time.monotonic()
    # Re-arm C-timer: nổ (dump tất cả thread) nếu nhịp tim ngừng _hang_stale
    # giây. Chạy ở tầng C nên bắt được cả khi GUI giữ GIL → đơ cứng (lúc đó
    # watchdog Python bị đói GIL, không ghi nổi HANG).
    if _fault_fp is not None:
        try:
            faulthandler.dump_traceback_later(
                _hang_stale, repeat=False, file=_fault_fp)
        except Exception:
            pass


def _dump_all_threads():
    try:
        if _fault_fp is not None:
            faulthandler.dump_traceback(file=_fault_fp, all_threads=True)
            _fault_fp.flush()
    except Exception:
        pass


def _watchdog(stale: float):
    """Bắt treo GUI (loại Python còn chạy được). Trong lúc treo, cứ ``stale``
    giây dump lại 1 lần để thấy app có nhúc nhích không (cùng chỗ = deadlock
    thật); khi hồi thì ghi rõ đã kẹt bao lâu (khựng tạm thời, không đứng máy).
    Loại đơ-cứng-giữ-GIL do C-timer ``dump_traceback_later`` lo (xem heartbeat)."""
    global _hang_dumped
    redump_at = 0.0
    hang_since = 0.0
    while True:
        time.sleep(1.0)
        if _alive is None:
            continue
        late = time.monotonic() - _alive
        if late > stale:
            now = time.monotonic()
            if not _hang_dumped:
                _hang_dumped = True
                hang_since = now
                redump_at = now + max(stale, 10.0)
                _write("HANG", f"GUI không phản hồi ~{late:.0f}s — dump tất cả thread:")
                _dump_all_threads()
            elif now >= redump_at:
                redump_at = now + max(stale, 10.0)
                _write("HANG-STILL", f"vẫn treo ~{now - hang_since:.0f}s — dump lại:")
                _dump_all_threads()
        else:
            if _hang_dumped:
                _write("RECOVERED",
                       f"GUI hồi phục sau ~{time.monotonic() - hang_since:.0f}s kẹt "
                       f"(khựng tạm thời, không phải đứng máy)")
            _hang_dumped = False   # GUI sống lại → cho phép dump lần treo sau


def install(hang_stale: float = 15.0):
    """Gọi 1 lần ở đầu main()."""
    global _fault_fp, _hang_stale
    _hang_stale = float(hang_stale)

    # 1) exception chưa bắt ở main thread
    def _excepthook(et, e, tb):
        _write("UNCAUGHT", "".join(traceback.format_exception(et, e, tb)))
        sys.__excepthook__(et, e, tb)
    sys.excepthook = _excepthook

    # 2) exception chưa bắt ở thread phụ (Python 3.8+)
    def _thread_hook(args):
        _write("THREAD", "".join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback)))
    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass

    # 3) crash native (segfault / abort của Qt / hỏng heap) → dump stack vào file
    try:
        _fault_fp = open(log_path(), "a", encoding="utf-8")
        faulthandler.enable(_fault_fp)
    except Exception:
        pass

    # banner mở phiên: tách các lần chạy + nhắc cách đọc log
    base = (sys.executable if getattr(sys, "frozen", False) else __file__)
    _write("START", f"App khởi động — theo dõi treo (>{hang_stale:.0f}s) + crash.\n"
                    f"path: {base}\n"
                    f"(0x8001010d = nhiễu COM lành tính, bỏ qua; "
                    f"chỉ lo HANG / Timeout / 0xC0000374 / access violation)")

    # 4) watchdog bắt treo (no-responding) + arm C-timer lần đầu
    heartbeat()
    try:
        threading.Thread(target=_watchdog, args=(hang_stale,),
                         daemon=True, name="crash-watchdog").start()
    except Exception:
        pass
