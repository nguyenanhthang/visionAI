"""plc_panasonic — native Panasonic MEWTOCOL-COM driver (FP-X / FP-XH / FP7).

Thay cho m_plc_panasonic.dll (.NET). Lý do:
  • DLL cũ open/close serial port mỗi call → ~500-600ms/call. Main loop có
    7 reads/iter → 4.2s/iter. Native + persistent connection → ~50ms/call,
    + batch read (1 lệnh đọc nhiều thanh ghi) → tổng ~0.15s/iter.
  • Cross-platform (không cần .NET / pythonnet).
  • Thread-safe (Lock — match Mutex_WR của DLL).

API tương thích code cũ (không cần đổi caller):
  read_data_panasonic(COM, baud, slot)       -> str | None   (decimal value)
  write_data_panasonic(COM, baud, slot, val) -> None         (silent on err)

API mới (tăng tốc):
  read_multi_data_panasonic(COM, baud, start_slot, count) -> list[int] | None
  write_multi_data_panasonic(COM, baud, start_slot, values) -> bool
  close_all()  — gọi lúc shutdown
  configure_serial(parity=, bytesize=, stopbits=, station=, timeout=)

MEWTOCOL-COM format (reference):
  Command :  %<stn(2)>#<cmd(2)><type(1+)><start(5)><end(5)>[data]<BCC><CR>
  Resp OK :  %<stn(2)>$<cmd(2)>[data]<BCC><CR>
  Resp err:  %<stn(2)>!<err(2)><BCC><CR>
  BCC = XOR tất cả byte từ '%' đến hết data, → 2 ký tự ASCII hex uppercase.
  Word data trong response = "LLHH" (low byte trước, high byte sau).
"""

import re
import threading
import time

import serial


# ── Default config ──────────────────────────────────────────────────
# 8N1 = mặc định của .NET SerialPort (DLL cũ dùng cái này nếu không config
# rõ). Panasonic FP factory default thường là 8O1 nhưng đa số máy đã được
# reconfig 8N1. Auto-detect ở `_get_port` sẽ thử cả 2.
PARITY    = serial.PARITY_NONE
BYTESIZE  = serial.EIGHTBITS
STOPBITS  = serial.STOPBITS_ONE
STATION   = 1
TIMEOUT_S = 0.5   # giây — đủ cho cả batch read 100 words ở 9600 baud

# Auto-detect parity lúc connect lần đầu. Sau khi detect xong → cache.
AUTO_DETECT_PARITY = True
_DETECTED_PARITY = None   # = serial.PARITY_NONE / PARITY_ODD sau khi probe

_lock = threading.Lock()
_ports: "dict[tuple, serial.Serial]" = {}


def configure_serial(parity=None, bytesize=None, stopbits=None,
                     station=None, timeout=None, auto_detect=None):
    """Override defaults. Gọi 1 lần lúc startup nếu cần."""
    global PARITY, BYTESIZE, STOPBITS, STATION, TIMEOUT_S, AUTO_DETECT_PARITY
    global _DETECTED_PARITY
    if parity is not None:
        PARITY = parity
        _DETECTED_PARITY = parity   # tin user, bỏ probe
    if bytesize is not None: BYTESIZE  = bytesize
    if stopbits is not None: STOPBITS  = stopbits
    if station  is not None: STATION   = int(station)
    if timeout  is not None: TIMEOUT_S = float(timeout)
    if auto_detect is not None: AUTO_DETECT_PARITY = bool(auto_detect)


# ── MEWTOCOL command/response helpers ───────────────────────────────
def _bcc(body: bytes) -> bytes:
    b = 0
    for c in body:
        b ^= c
    return f"{b:02X}".encode("ascii")


_ADDR_RE = re.compile(r"^\s*(?:DT?)?\s*(\d+)\s*$", re.IGNORECASE)


def _addr(slot) -> int:
    """Parse 'D6000' / 'DT6000' / '6000' → 6000 (int)."""
    if isinstance(slot, int):
        return slot
    m = _ADDR_RE.match(str(slot))
    if not m:
        raise ValueError(f"Invalid PLC address: {slot!r}")
    return int(m.group(1))


def _build_read_cmd(start: int, end: int, station: int = None) -> bytes:
    st = STATION if station is None else int(station)
    body = f"%{st:02d}#RDD{start:05d}{end:05d}".encode("ascii")
    return body + _bcc(body) + b"\r"


def _build_write_cmd(start: int, values, station: int = None) -> bytes:
    st = STATION if station is None else int(station)
    end = start + len(values) - 1
    # Mỗi word: "LLHH" (low byte ASCII hex trước)
    data_parts = []
    for v in values:
        u = int(v) & 0xFFFF
        lo = u & 0xFF
        hi = (u >> 8) & 0xFF
        data_parts.append(f"{lo:02X}{hi:02X}")
    body = (f"%{st:02d}#WDD{start:05d}{end:05d}" + "".join(data_parts)).encode("ascii")
    return body + _bcc(body) + b"\r"


def _send_recv(port: serial.Serial, cmd: bytes) -> bytes:
    """Gửi cmd, đọc response đến CR. Trả bytes (không gồm CR)."""
    try:
        port.reset_input_buffer()
        port.reset_output_buffer()
    except Exception:
        pass
    port.write(cmd)
    out = bytearray()
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        b = port.read(1)
        if not b:
            continue
        if b == b"\r":
            return bytes(out)
        out += b
        # Safety: response MEWTOCOL không bao giờ vượt ~1KB. Tránh
        # spam memory nếu PLC trả về garbage không CR.
        if len(out) > 4096:
            break
    raise TimeoutError("MEWTOCOL response timeout")


def _parse_read_response(resp: bytes, station: int = None):
    st = STATION if station is None else int(station)
    s = resp.decode("ascii", errors="replace")
    pref_ok  = f"%{st:02d}$RD"
    pref_err = f"%{st:02d}!"
    if s.startswith(pref_err):
        raise RuntimeError(f"PLC error response: {s}")
    if not s.startswith(pref_ok):
        raise RuntimeError(f"Bad RD response: {s!r}")
    data = s[len(pref_ok):-2]   # bỏ BCC 2 ký tự cuối
    out = []
    # Mỗi 4 chars = 1 word ("LLHH"). Bỏ phần thừa nếu có (defensive).
    n = (len(data) // 4) * 4
    for i in range(0, n, 4):
        lo = int(data[i:i+2], 16)
        hi = int(data[i+2:i+4], 16)
        out.append((hi << 8) | lo)
    return out


def _parse_write_response(resp: bytes, station: int = None):
    st = STATION if station is None else int(station)
    s = resp.decode("ascii", errors="replace")
    pref_ok  = f"%{st:02d}$WD"
    pref_err = f"%{st:02d}!"
    if s.startswith(pref_err):
        raise RuntimeError(f"PLC error response: {s}")
    if not s.startswith(pref_ok):
        raise RuntimeError(f"Bad WD response: {s!r}")


# ── Connection management ───────────────────────────────────────────
def _open_with_parity(com: str, baud: int, parity) -> serial.Serial:
    return serial.Serial(
        port=com, baudrate=int(baud),
        bytesize=BYTESIZE, parity=parity, stopbits=STOPBITS,
        timeout=TIMEOUT_S, write_timeout=TIMEOUT_S,
    )


def _probe_parity(com: str, baud: int) -> serial.Serial:
    """Mở port, thử PARITY_NONE rồi PARITY_ODD. Trả port có parity đúng.
    Cache kết quả vào _DETECTED_PARITY → lần connect sau dùng thẳng."""
    global _DETECTED_PARITY
    candidates = [serial.PARITY_NONE, serial.PARITY_ODD]
    # Bias theo PARITY default (user-set) — thử cái đó trước.
    if PARITY in candidates and candidates[0] != PARITY:
        candidates.remove(PARITY)
        candidates.insert(0, PARITY)

    last_err = None
    for par in candidates:
        try:
            p = _open_with_parity(com, baud, par)
        except Exception as e:
            last_err = e
            continue
        try:
            # Probe: read DT0 (1 word). Bất kỳ response MEWTOCOL hợp lệ
            # (success $ hoặc error !) đều confirm parity đúng.
            probe = _build_read_cmd(0, 0)
            resp = _send_recv(p, probe)
            s = resp.decode("ascii", errors="ignore")
            if s.startswith(f"%{STATION:02d}$") or s.startswith(f"%{STATION:02d}!"):
                _DETECTED_PARITY = par
                return p
            # Garbage prefix → parity sai. Đóng, thử cái tiếp theo.
            try: p.close()
            except Exception: pass
        except TimeoutError as e:
            last_err = e
            try: p.close()
            except Exception: pass
        except Exception as e:
            last_err = e
            try: p.close()
            except Exception: pass

    raise OSError(f"Cannot probe parity on {com}: {last_err}")


def _get_port(com: str, baud: int) -> serial.Serial:
    key = (str(com).upper(), int(baud))
    p = _ports.get(key)
    if p is not None and p.is_open:
        return p
    if p is not None:
        try: p.close()
        except Exception: pass

    if AUTO_DETECT_PARITY and _DETECTED_PARITY is None:
        p = _probe_parity(com, baud)
    else:
        par = _DETECTED_PARITY if _DETECTED_PARITY is not None else PARITY
        p = _open_with_parity(com, baud, par)

    _ports[key] = p
    return p


def _do(com, baud, cmd: bytes, parse_fn):
    """Gửi cmd dưới lock. Retry 1 lần nếu lỗi serial (port chết)."""
    with _lock:
        try:
            port = _get_port(com, baud)
            resp = _send_recv(port, cmd)
        except (serial.SerialException, OSError, TimeoutError):
            # Port có thể đã chết hoặc PLC tạm offline → đóng + thử lại 1 lần.
            key = (str(com).upper(), int(baud))
            old = _ports.pop(key, None)
            if old is not None:
                try: old.close()
                except Exception: pass
            port = _get_port(com, baud)
            resp = _send_recv(port, cmd)
        return parse_fn(resp)


# ── Public API ──────────────────────────────────────────────────────
def read_multi_data_panasonic(com, baud, start_slot, count):
    """Đọc `count` thanh ghi liên tiếp từ `start_slot`.

    Trả list[int] (mỗi phần tử 16-bit unsigned 0..65535), hoặc None nếu lỗi.

    Ví dụ: read_multi_data_panasonic('COM6', 9600, 'D17800', 4)
           → [val_D17800, val_D17801, val_D17802, val_D17803]
    """
    try:
        start = _addr(start_slot)
        end = start + max(1, int(count)) - 1
        cmd = _build_read_cmd(start, end)
        return _do(com, baud, cmd, _parse_read_response)
    except Exception:
        return None


def read_data_panasonic(com, baud, slot):
    """Đọc 1 thanh ghi. Trả str (decimal) — giữ API cũ. None nếu lỗi.

    Backward-compat: code cũ compare `read == '1'` nên phải trả string.
    """
    vals = read_multi_data_panasonic(com, baud, slot, 1)
    if not vals:
        return None
    return str(vals[0])


def write_data_panasonic(com, baud, slot, value):
    """Ghi 1 thanh ghi. Silent fail (giữ behaviour API cũ — không raise)."""
    try:
        start = _addr(slot)
        cmd = _build_write_cmd(start, [int(value)])
        _do(com, baud, cmd, _parse_write_response)
    except Exception:
        pass


def write_multi_data_panasonic(com, baud, start_slot, values):
    """Ghi liên tiếp `len(values)` thanh ghi từ start_slot. Trả True/False."""
    try:
        start = _addr(start_slot)
        cmd = _build_write_cmd(start, [int(v) for v in values])
        _do(com, baud, cmd, _parse_write_response)
        return True
    except Exception:
        return False


def close_all():
    """Đóng tất cả serial port. Gọi lúc app shutdown."""
    with _lock:
        for p in _ports.values():
            try: p.close()
            except Exception: pass
        _ports.clear()

# NOTE: smoke-test block đã chuyển vào hàm để không tự chạy lúc import.
def _smoke_test(com="COM1", baud=9600):
    b1 = read_multi_data_panasonic(com, baud, "D17800", 4)
    b2 = read_multi_data_panasonic(com, baud, "D17890", 4)
    b3 = read_multi_data_panasonic(com, baud, "D18890", 10)
    v_18803 = read_data_panasonic(com, baud, "D18803")
    print(b1, b2, b3, v_18803)


if __name__ == "__main__":
    _smoke_test()
