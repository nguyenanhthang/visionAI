"""Ghi nguyên nhân khi app văng/treo để truy vết (không phụ thuộc GUI).

``install()`` bật:
  - ``faulthandler``: dump Python traceback khi gặp tín hiệu fatal — segfault,
    hoặc abort của Qt (vd "QThread: Destroyed while thread is still running").
  - ``sys.excepthook`` + ``threading.excepthook``: ghi exception CHƯA BẮT
    (cả ở thread phụ) kèm thời gian.

File ``crash.log`` nằm cạnh .exe (frozen) hoặc cạnh script. Khi app văng,
gửi file này để biết chính xác chỗ lỗi.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

_fault_fp = None  # giữ file mở cho faulthandler suốt vòng đời process


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


def install():
    """Gọi 1 lần ở đầu main()."""
    global _fault_fp

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

    # 3) crash native (segfault / abort của Qt) → dump Python stack vào file
    try:
        _fault_fp = open(log_path(), "a", encoding="utf-8")
        faulthandler.enable(_fault_fp)
    except Exception:
        pass
