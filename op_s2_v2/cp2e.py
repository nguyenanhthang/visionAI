"""Omron CP2E FINS/TCP driver.

Thay `h3u_h5u.py` (Inovance Modbus TCP) cho line chạy PLC **Omron CP2E**.
Giữ nguyên API module-level để `plc_worker` + `scanner` đổi sang chỉ tốn
1 dòng import:

    read_data_cp2e(ip, address)            -> int | None
    write_data_cp2e(ip, address, value)    -> bool
    read_multi_data_cp2e(ip, start, count) -> list[int] | None
    close_all()                            -> None

Mặc định thao tác vùng Data Memory (DM word, FINS area code 0x82) — tức
``address=300`` ↔ ``D300`` trên CP2E. Port FINS/TCP mặc định của Omron là
9600. Đổi bằng ``configure(area=, port=, timeout=)`` lúc startup nếu line
dùng vùng nhớ / cổng khác.

Giao thức dựng theo FINS_TCP.py:
  - Lúc connect gửi NADS (node address rỗng) để PLC cấp node address
    DA1 (server/PLC) + SA1 (client/PC).
  - MEMORY AREA READ (01 01) / WRITE (01 02).
  - Response = 16 byte TCP header + 10 byte FINS header + 2 byte MRC/SRC
    + 2 byte end-code (MRES/SRES) + data → data bắt đầu ở offset 30.

Connection được cache theo IP, có lock cho thread-safe + tự reconnect 1
lần khi lỗi (giống cách _do() của panasonic.py xử lý port chết).
"""

from __future__ import annotations

import socket
import struct
import threading
import time


# ── FINS memory area codes (word access) ─────────────────────────
DM_WORD   = b"\x82"   # Data Memory   (CP2E: vùng D)
CIO_WORD  = b"\xB0"   # CIO
WORK_WORD = b"\xB1"   # Work area (W)
HOLD_WORD = b"\xB2"   # Holding (H)


# ── Default config ───────────────────────────────────────────────
PORT      = 9600      # cổng FINS/TCP mặc định của Omron
AREA      = DM_WORD   # vùng nhớ thao tác mặc định
TIMEOUT_S = 2.0       # giây — timeout socket

_MAGIC = b"FINS"
_DATA_OFFSET = 30     # vị trí byte data đầu tiên trong response FINS/TCP


def configure(area: bytes | None = None, port: int | None = None,
              timeout: float | None = None):
    """Override default cho mọi connection lazy-open sau call này."""
    global AREA, PORT, TIMEOUT_S
    if area is not None:    AREA = area
    if port is not None:    PORT = int(port)
    if timeout is not None: TIMEOUT_S = float(timeout)


# ── byte helpers ─────────────────────────────────────────────────
def _u16(v: int) -> bytes:
    return struct.pack(">H", int(v) & 0xFFFF)


def _u32(v: int) -> bytes:
    return struct.pack(">I", int(v) & 0xFFFFFFFF)


def _check_end_code(resp: bytes):
    """Kiểm tra end-code FINS (MRES/SRES ở offset 28-29). 0000 = OK."""
    if len(resp) < _DATA_OFFSET:
        raise ConnectionError(f"FINS response quá ngắn ({len(resp)} byte)")
    mres, sres = resp[28], resp[29]
    if (mres, sres) != (0, 0):
        raise RuntimeError(f"FINS end code {mres:02X}{sres:02X}")


# ── 1 kết nối FINS/TCP tới 1 PLC ─────────────────────────────────
class _FinsTcp:
    def __init__(self, host: str, port: int = PORT, timeout: float = TIMEOUT_S):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.da1 = b"\x00"   # node PLC  (đích)
        self.sa1 = b"\x00"   # node PC   (nguồn)

    def connect(self):
        self.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        # Frame FINS rất nhỏ → tắt Nagle cho khỏi trễ; keepalive để OS tự
        # phát hiện kết nối chết (PLC reboot/rút cáp) thay vì chờ timeout đọc.
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        try:
            s.connect((self.host, self.port))
            self.sock = s
            # NADS: gửi node address rỗng → PLC trả về client+server node.
            nads = _MAGIC + _u32(12) + _u32(0) + _u32(0) + _u32(0)
            s.sendall(nads)
            resp = self._recv_frame()
            self.sa1 = bytes([resp[19]])   # client node (PC)
            self.da1 = bytes([resp[23]])   # server node (PLC)
        except Exception:
            try: s.close()
            except Exception: pass
            self.sock = None
            raise

    def close(self):
        if self.sock is not None:
            try: self.sock.close()
            except Exception: pass
            self.sock = None

    # ---- low-level frame I/O ----
    def _recv_n(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("FINS: kết nối đóng giữa chừng")
            buf += chunk
        return buf

    def _recv_frame(self) -> bytes:
        """Đọc trọn 1 frame FINS/TCP theo trường length (8 byte đầu)."""
        header = self._recv_n(8)            # 'FINS' + length(4)
        if header[:4] != _MAGIC:
            raise ConnectionError(f"FINS: header lạ {header[:4]!r}")
        length = struct.unpack(">I", header[4:8])[0]
        return header + self._recv_n(length)

    def _fins_header(self) -> bytes:
        # ICF RSV GCT DNA DA1 DA2 SNA SA1 SA2 SID  (10 byte)
        return (b"\x80\x00\x02\x00" + self.da1 + b"\x00"
                + b"\x00" + self.sa1 + b"\x00" + b"\x01")

    def _send_fins(self, fins_cmd: bytes):
        # TCP header: FINS + length(4) + command(=2 FINS frame send) + err(0)
        body = _u32(2) + _u32(0) + fins_cmd
        self.sock.sendall(_MAGIC + _u32(len(body)) + body)

    # ---- memory area read / write ----
    def read(self, address: int, count: int) -> list[int]:
        cmd = b"\x01\x01" + AREA + _u16(address) + b"\x00" + _u16(count)
        self._send_fins(self._fins_header() + cmd)
        resp = self._recv_frame()
        _check_end_code(resp)
        data = resp[_DATA_OFFSET:_DATA_OFFSET + 2 * count]
        return [struct.unpack(">H", data[i:i + 2])[0]
                for i in range(0, len(data) - len(data) % 2, 2)]

    def write(self, address: int, values: list[int]):
        payload = b"".join(_u16(v) for v in values)
        cmd = (b"\x01\x02" + AREA + _u16(address) + b"\x00"
               + _u16(len(values)) + payload)
        self._send_fins(self._fins_header() + cmd)
        _check_end_code(self._recv_frame())


# ── Connection cache (giống panasonic._ports / h3u_h5u._masters) ─
_lock = threading.Lock()
_conns: "dict[str, _FinsTcp]" = {}
_fail_until: "dict[str, float]" = {}   # IP → mốc monotonic được phép reconnect lại
_BACKOFF_S = 2.0


def _get_conn(ip: str) -> _FinsTcp:
    c = _conns.get(ip)
    if c is None or c.sock is None:
        # PLC đang chết: không thử connect (≈2-6s timeout) ở MỌI call —
        # 2 thread (poll + scanner) thay nhau ôm _lock chờ timeout làm
        # quét SN nghẽn hàng chục giây. Fail-fast trong cửa sổ backoff.
        if time.monotonic() < _fail_until.get(ip, 0.0):
            raise ConnectionError(f"{ip}: chờ backoff sau lỗi kết nối")
        c = _FinsTcp(ip, PORT, TIMEOUT_S)
        try:
            c.connect()
        except Exception:
            _fail_until[ip] = time.monotonic() + _BACKOFF_S
            raise
        _fail_until.pop(ip, None)
        _conns[ip] = c
    return c


def _drop(ip: str):
    c = _conns.pop(ip, None)
    if c is not None:
        c.close()


# ── Module-level API (giống h3u_h5u.py) ──────────────────────────
def read_multi_data_cp2e(ip: str, start: int, count: int) -> "list[int] | None":
    with _lock:
        try:
            return _get_conn(ip).read(int(start), int(count))
        except Exception:
            _drop(ip)
            try:
                return _get_conn(ip).read(int(start), int(count))
            except Exception:
                return None


def read_data_cp2e(ip: str, address: int) -> "int | None":
    vals = read_multi_data_cp2e(ip, address, 1)
    return vals[0] if vals else None


def write_data_cp2e(ip: str, address: int, value: int) -> bool:
    with _lock:
        try:
            _get_conn(ip).write(int(address), [int(value)])
            return True
        except Exception:
            _drop(ip)
            try:
                _get_conn(ip).write(int(address), [int(value)])
                return True
            except Exception:
                return False


def close_all():
    with _lock:
        for c in _conns.values():
            c.close()
        _conns.clear()


# ── Class API (tương thích kiểu PLC.py gốc) ──────────────────────
class CP2E:
    """Wrapper giữ connection riêng (không dùng cache module-level)."""

    def __init__(self, ip: str, port: int = PORT, timeout: float = TIMEOUT_S):
        self._conn = _FinsTcp(str(ip), port, timeout)

    def connect(self):
        self._conn.connect()

    def ReadMemory(self, address: int) -> int:
        return self._conn.read(int(address), 1)[0]

    def WriteMemory(self, address: int, data: int):
        self._conn.write(int(address), [int(data)])

    def close(self):
        self._conn.close()
