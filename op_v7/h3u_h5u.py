"""Inovance H3U/H5U Modbus TCP driver.

Thay `panasonic.py` cho line dùng PLC Inovance. API module-level
được giữ song song với panasonic.py để plc_worker + scanner đổi sang
chỉ mất 1 dòng import:

    read_data_h3u(ip, address)            -> int | None
    write_data_h3u(ip, address, value)    -> bool
    read_multi_data_h3u(ip, start, count) -> list[int] | None
    close_all()                            -> None

Class `H3U_H5U` cho code muốn cầm connection trực tiếp.

Slot quy ước: holding register, slave id 1, port 502. Override bằng
`configure(slave_id=, port=, timeout=)` lúc startup nếu line khác.
"""

from __future__ import annotations

import threading
import time

try:
    from modbus_tk import modbus_tcp
    import modbus_tk.defines as cst
except ImportError:
    modbus_tcp = None
    cst = None


# ── Default config ──────────────────────────────────────────────
PORT      = 502
SLAVE_ID  = 1
TIMEOUT_S = 3.0


def configure(slave_id: int | None = None, port: int | None = None,
              timeout: float | None = None):
    """Đổi default cho mọi connection lazy-open sau call này."""
    global SLAVE_ID, PORT, TIMEOUT_S
    if slave_id is not None: SLAVE_ID = slave_id
    if port is not None:     PORT = port
    if timeout is not None:  TIMEOUT_S = timeout


# ── Connection cache ────────────────────────────────────────────
_lock = threading.Lock()
_masters: dict[str, "modbus_tcp.TcpMaster"] = {}
_fail_until: dict[str, float] = {}   # IP → mốc monotonic được phép reconnect lại
_BACKOFF_S = 2.0


def _get_master(ip: str):
    if modbus_tcp is None:
        raise RuntimeError("modbus-tk chưa cài: pip install modbus-tk")
    m = _masters.get(ip)
    if m is None:
        # PLC đang chết: không mở lại (≈3s timeout) ở MỌI call — 2 thread
        # (poll + scanner) thay nhau ôm _lock chờ timeout làm quét SN
        # nghẽn hàng chục giây. Fail-fast trong cửa sổ backoff.
        if time.monotonic() < _fail_until.get(ip, 0.0):
            raise ConnectionError(f"{ip}: chờ backoff sau lỗi kết nối")
        m = modbus_tcp.TcpMaster(host=ip, port=PORT)
        m.set_timeout(TIMEOUT_S)
        _masters[ip] = m
    return m


def _drop(ip: str):
    """Đóng + bỏ master hỏng để call sau mở connection mới.

    modbus_tk giữ socket bên trong master: sau 1 lần lỗi (PLC reboot, đứt
    mạng, response lệch frame) socket có thể chết/lệch vĩnh viễn — nếu cứ
    cache mãi thì mọi read/write sau đó đều lỗi cho tới khi RESTART APP.
    """
    m = _masters.pop(ip, None)
    if m is not None:
        try:
            m.close()
        except Exception:
            pass


def _execute(ip: str, *args, **kwargs):
    """Chạy 1 giao dịch Modbus: lỗi → drop master, mở lại, thử thêm 1 lần."""
    with _lock:
        try:
            return _get_master(ip).execute(SLAVE_ID, *args, **kwargs)
        except Exception:
            _drop(ip)
            try:
                return _get_master(ip).execute(SLAVE_ID, *args, **kwargs)
            except Exception:
                _drop(ip)
                _fail_until[ip] = time.monotonic() + _BACKOFF_S
                raise


# ── Module-level API (giống panasonic.py) ───────────────────────
def read_data_h3u(ip: str, address: int) -> int | None:
    try:
        v = _execute(ip, cst.READ_HOLDING_REGISTERS, int(address), 1)
        return int(v[0]) if v else None
    except Exception:
        return None


def read_multi_data_h3u(ip: str, start: int, count: int) -> list[int] | None:
    try:
        v = _execute(ip, cst.READ_HOLDING_REGISTERS, int(start), int(count))
        return list(v) if v else None
    except Exception:
        return None


def write_data_h3u(ip: str, address: int, value: int) -> bool:
    try:
        _execute(ip, cst.WRITE_SINGLE_REGISTER, int(address),
                 output_value=int(value))
        return True
    except Exception:
        return False


def close_all():
    with _lock:
        for m in _masters.values():
            try:
                m.close()
            except Exception:
                pass
        _masters.clear()


# ── Class API (tương thích PLC.py gốc) ──────────────────────────
class H3U_H5U:
    """Wrapper class — giữ master riêng (không dùng cache module-level)."""

    def __init__(self, ip: str, slave_id: int = SLAVE_ID, port: int = PORT,
                 timeout: float = TIMEOUT_S):
        self.ip = str(ip)
        self.slave_id = slave_id
        self.port = port
        self.timeout = timeout
        self.master = None

    def connect(self):
        if modbus_tcp is None:
            raise RuntimeError("modbus-tk chưa cài: pip install modbus-tk")
        self.master = modbus_tcp.TcpMaster(host=self.ip, port=self.port)
        self.master.set_timeout(self.timeout)

    def ReadMemory(self, address: int) -> int:
        v = self.master.execute(
            self.slave_id, cst.READ_HOLDING_REGISTERS, int(address), 1,
        )
        return int(v[0])

    def WriteMemory(self, address: int, data: int):
        return self.master.execute(
            self.slave_id, cst.WRITE_SINGLE_REGISTER, int(address),
            output_value=int(data),
        )

    def close(self):
        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None
