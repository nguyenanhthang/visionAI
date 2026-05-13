"""
core/plc.py — PLC connectivity for AOI Vision Pro

Hỗ trợ 3 dòng PLC:
  - Omron CP2E      (FINS over TCP, port 9600)
  - Omron NX1P2     (FINS over UDP, port 9600)
  - Inovance H3U/H5U (Modbus TCP, port 502)

API thống nhất qua lớp ``PLCDriver``:
    connect(), disconnect()
    read_word(area, address)        -> int      (16-bit unsigned)
    write_word(area, address, val)  -> None
    read_bit(area, address, bit)    -> bool
    write_bit(area, address, bit, val) -> None

``PLCManager`` cung cấp polling trigger ở thread riêng và gửi kết quả về PLC.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ── Memory area codes ─────────────────────────────────────────────
class MemoryArea(str, Enum):
    DM_WORD  = "DM_WORD"     # Data Memory word    (D0, D100…)
    CIO_WORD = "CIO_WORD"    # CIO area word
    CIO_BIT  = "CIO_BIT"     # CIO area bit
    DM_BIT   = "DM_BIT"      # DM bit
    W_WORD   = "W_WORD"      # Work word
    H_WORD   = "H_WORD"      # Holding word


# ── FINS hex codes for Omron area access ──────────────────────────
_FINS_AREA_CODES = {
    MemoryArea.DM_WORD:  b'\x82',
    MemoryArea.DM_BIT:   b'\x02',
    MemoryArea.CIO_WORD: b'\xB0',
    MemoryArea.CIO_BIT:  b'\x30',
    MemoryArea.W_WORD:   b'\xB1',
    MemoryArea.H_WORD:   b'\xB2',
}

_FINS_MEMORY_AREA_READ  = b'\x01\x01'
_FINS_MEMORY_AREA_WRITE = b'\x01\x02'


# ── Base driver ───────────────────────────────────────────────────
class PLCDriver(ABC):
    """Giao diện chung cho mọi driver PLC."""

    def __init__(self, ip: str, port: int, timeout: float = 2.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.connected = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_word(self, area: MemoryArea, address: int) -> int: ...

    @abstractmethod
    def write_word(self, area: MemoryArea, address: int, value: int) -> None: ...

    def read_bit(self, area: MemoryArea, address: int, bit: int = 0) -> bool:
        word = self.read_word(self._word_area_for(area), address)
        return bool((word >> bit) & 1)

    def write_bit(self, area: MemoryArea, address: int, bit: int, value: bool) -> None:
        word_area = self._word_area_for(area)
        cur = self.read_word(word_area, address)
        mask = 1 << bit
        new = (cur | mask) if value else (cur & ~mask & 0xFFFF)
        self.write_word(word_area, address, new)

    @staticmethod
    def _word_area_for(area: MemoryArea) -> MemoryArea:
        if area == MemoryArea.CIO_BIT:
            return MemoryArea.CIO_WORD
        if area == MemoryArea.DM_BIT:
            return MemoryArea.DM_WORD
        return area


# ── Omron CP2E — FINS over TCP ────────────────────────────────────
class OmronCP2E(PLCDriver):
    """FINS/TCP driver cho Omron CP2E."""

    _FINS_MAGIC = b'\x46\x49\x4e\x53'   # b'FINS'

    def __init__(self, ip: str, port: int = 9600, timeout: float = 2.0):
        super().__init__(ip, port, timeout)
        self.sock: Optional[socket.socket] = None
        self.da1 = b'\x00'  # destination node (PLC)
        self.sa1 = b'\x00'  # source node (PC)
        self._sid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))

        # FINS/TCP node-address handshake (cmd=0)
        handshake = (self._FINS_MAGIC
                     + (12).to_bytes(4, 'big')      # length = 8 + 4
                     + (0).to_bytes(4, 'big')       # cmd 0 = client→server
                     + (0).to_bytes(4, 'big')       # error code
                     + (0).to_bytes(4, 'big'))      # client node addr (auto)
        self.sock.sendall(handshake)
        resp = self._recv_exact(24)
        # Response: magic(4) + len(4) + cmd(4) + err(4) + clientNode(4) + serverNode(4)
        self.sa1 = resp[19:20]
        self.da1 = resp[23:24]
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_sid(self) -> bytes:
        self._sid = (self._sid + 1) & 0xFF
        return self._sid.to_bytes(1, 'big')

    def _recv_exact(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC closed the connection")
            buf += chunk
        return buf

    def _send_fins(self, command_code: bytes, body: bytes) -> bytes:
        # FINS command header (10 bytes) + command_code (2) + body
        fins_cmd = (
            b'\x80\x00\x02'          # ICF, RSV, GCT
            + b'\x00' + self.da1 + b'\x00'   # DNA, DA1, DA2
            + b'\x00' + self.sa1 + b'\x00'   # SNA, SA1, SA2
            + self._next_sid()
            + command_code
            + body
        )
        # FINS/TCP frame header (cmd=2 = FINS frame send)
        length = 8 + len(fins_cmd)
        frame_hdr = (self._FINS_MAGIC
                     + length.to_bytes(4, 'big')
                     + (2).to_bytes(4, 'big')
                     + (0).to_bytes(4, 'big'))
        with self._lock:
            self.sock.sendall(frame_hdr + fins_cmd)
            # Response: TCP header (16) + FINS header (10) + cmd code (2) + endcode (2) + data
            hdr = self._recv_exact(16)
            resp_len = int.from_bytes(hdr[4:8], 'big')
            body = self._recv_exact(resp_len - 8)
        return body

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = code + address.to_bytes(2, 'big') + b'\x00' + (1).to_bytes(2, 'big')
        resp = self._send_fins(_FINS_MEMORY_AREA_READ, body)
        # resp = FINS header (10) + cmd code (2) + endcode (2) + data (2)
        data = resp[14:16]
        return int.from_bytes(data, 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = (code + address.to_bytes(2, 'big') + b'\x00'
                + (1).to_bytes(2, 'big')
                + (value & 0xFFFF).to_bytes(2, 'big'))
        self._send_fins(_FINS_MEMORY_AREA_WRITE, body)


# ── Omron NX1P2 — FINS over UDP ───────────────────────────────────
class OmronNX1P2(PLCDriver):
    """FINS/UDP driver cho Omron NX1P2."""

    def __init__(self, ip: str, port: int = 9600, timeout: float = 2.0,
                 dest_node: int = 1, src_node: int = 25):
        super().__init__(ip, port, timeout)
        self.dest_node = dest_node
        self.src_node = src_node
        self.sock: Optional[socket.socket] = None
        self._sid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        # Bind ephemeral local port; OS picks one (avoid clash with PLC)
        self.sock.bind(('', 0))
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_sid(self) -> bytes:
        self._sid = (self._sid + 1) & 0xFF
        return self._sid.to_bytes(1, 'big')

    def _send_fins(self, command_code: bytes, body: bytes) -> bytes:
        frame = (
            b'\x80\x00\x07'
            + b'\x00' + self.dest_node.to_bytes(1, 'big') + b'\x00'
            + b'\x00' + self.src_node.to_bytes(1, 'big') + b'\x00'
            + self._next_sid()
            + command_code
            + body
        )
        with self._lock:
            self.sock.sendto(frame, (self.ip, self.port))
            data, _ = self.sock.recvfrom(4096)
        return data

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = code + address.to_bytes(2, 'big') + b'\x00' + (1).to_bytes(2, 'big')
        resp = self._send_fins(_FINS_MEMORY_AREA_READ, body)
        # resp: FINS header (10) + cmd code (2) + endcode (2) + data (2)
        return int.from_bytes(resp[14:16], 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = (code + address.to_bytes(2, 'big') + b'\x00'
                + (1).to_bytes(2, 'big')
                + (value & 0xFFFF).to_bytes(2, 'big'))
        self._send_fins(_FINS_MEMORY_AREA_WRITE, body)


# ── Inovance H3U / H5U — Modbus TCP ───────────────────────────────
class InovanceH3UH5U(PLCDriver):
    """Modbus/TCP driver cho Inovance H3U / H5U.

    Address mapping (Inovance):
      DM_WORD  → D register  (FC 03 / 06)
      CIO_BIT  → M coil      (FC 01 / 05)
    """

    def __init__(self, ip: str, port: int = 502, timeout: float = 2.0,
                 unit_id: int = 1):
        super().__init__(ip, port, timeout)
        self.unit_id = unit_id
        self.sock: Optional[socket.socket] = None
        self._tid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_tid(self) -> int:
        self._tid = (self._tid + 1) & 0xFFFF
        return self._tid

    def _request(self, function_code: int, payload: bytes) -> bytes:
        if not self.connected:
            self.connect()
        tid = self._next_tid()
        pdu = bytes([function_code]) + payload
        adu = (tid.to_bytes(2, 'big')
               + b'\x00\x00'                           # protocol id
               + (len(pdu) + 1).to_bytes(2, 'big')     # length = pdu + unit
               + bytes([self.unit_id])
               + pdu)
        with self._lock:
            self.sock.sendall(adu)
            hdr = self._recv_exact(7)
            length = int.from_bytes(hdr[4:6], 'big') - 1
            body = self._recv_exact(length)
        if body[0] & 0x80:
            raise IOError(f"Modbus exception {body[1]} for FC {function_code}")
        return body[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC closed the connection")
            buf += chunk
        return buf

    def read_word(self, area: MemoryArea, address: int) -> int:
        # FC 03 — Read holding registers
        payload = address.to_bytes(2, 'big') + (1).to_bytes(2, 'big')
        data = self._request(0x03, payload)
        # data: byte_count (1) + register_values (2)
        return int.from_bytes(data[1:3], 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        # FC 06 — Write single register
        payload = address.to_bytes(2, 'big') + (value & 0xFFFF).to_bytes(2, 'big')
        self._request(0x06, payload)

    def read_bit(self, area: MemoryArea, address: int, bit: int = 0) -> bool:
        if area in (MemoryArea.CIO_BIT, MemoryArea.DM_BIT):
            # FC 01 — Read coils
            coil = address * 16 + bit if area == MemoryArea.DM_BIT else address
            payload = coil.to_bytes(2, 'big') + (1).to_bytes(2, 'big')
            data = self._request(0x01, payload)
            return bool(data[1] & 0x01)
        return super().read_bit(area, address, bit)

    def write_bit(self, area: MemoryArea, address: int, bit: int, value: bool) -> None:
        if area in (MemoryArea.CIO_BIT, MemoryArea.DM_BIT):
            # FC 05 — Write single coil
            coil = address * 16 + bit if area == MemoryArea.DM_BIT else address
            payload = coil.to_bytes(2, 'big') + (b'\xFF\x00' if value else b'\x00\x00')
            self._request(0x05, payload)
            return
        super().write_bit(area, address, bit, value)


# ── Registry ──────────────────────────────────────────────────────
DRIVER_BY_MODEL = {
    "Omron CP2E":         OmronCP2E,
    "Omron NX1P2":        OmronNX1P2,
    "Inovance H3U/H5U":   InovanceH3UH5U,
}


# ── Manager / polling thread ──────────────────────────────────────
@dataclass
class PLCConfig:
    model: str = "Omron CP2E"
    ip: str = "192.168.250.1"
    port: int = 9600
    poll_interval_ms: int = 100

    # Trigger (signal AOI bắt đầu chạy)
    trigger_area: MemoryArea = MemoryArea.DM_WORD
    trigger_address: int = 100
    trigger_bit: int = 0           # dùng khi area là *_BIT
    trigger_value: int = 1         # giá trị "ON" với word; bit thì luôn so với 1
    auto_clear_trigger: bool = True

    # Result (gửi PASS/FAIL về PLC)
    result_area: MemoryArea = MemoryArea.DM_WORD
    result_address: int = 101
    result_pass_value: int = 1
    result_fail_value: int = 2

    # Numeric data (gửi số liệu, optional)
    data_area: MemoryArea = MemoryArea.DM_WORD
    data_start_address: int = 110


class PLCManager:
    """Quản lý driver + thread polling trigger.

    Cách dùng (từ GUI):
        mgr = PLCManager()
        mgr.config = PLCConfig(...)
        mgr.connect()
        mgr.start_monitor(on_trigger=lambda: window.run_pipeline())
        # khi pipeline xong:
        mgr.write_result(passed=True, values=[12, 34])
    """

    def __init__(self):
        self.config = PLCConfig()
        self.driver: Optional[PLCDriver] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._on_trigger: Optional[Callable[[], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self._last_trigger: Optional[int] = None

    @property
    def is_connected(self) -> bool:
        return self.driver is not None and self.driver.connected

    @property
    def is_monitoring(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def connect(self) -> None:
        cls = DRIVER_BY_MODEL.get(self.config.model)
        if cls is None:
            raise ValueError(f"Unknown PLC model: {self.config.model}")
        self.disconnect()
        self.driver = cls(self.config.ip, self.config.port)
        self.driver.connect()

    def disconnect(self) -> None:
        self.stop_monitor()
        if self.driver is not None:
            try:
                self.driver.disconnect()
            except Exception:
                pass
        self.driver = None

    # ── Trigger monitor ──
    def start_monitor(self,
                      on_trigger: Callable[[], None],
                      on_error: Optional[Callable[[str], None]] = None) -> None:
        if not self.is_connected:
            raise RuntimeError("PLC not connected")
        if self.is_monitoring:
            return
        self._on_trigger = on_trigger
        self._on_error = on_error
        self._stop.clear()
        self._last_trigger = None
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="PLCMonitor")
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop.set()
        t = self._monitor_thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        cfg = self.config
        is_bit = cfg.trigger_area in (MemoryArea.CIO_BIT, MemoryArea.DM_BIT)
        while not self._stop.is_set():
            try:
                if is_bit:
                    val = 1 if self.driver.read_bit(cfg.trigger_area,
                                                    cfg.trigger_address,
                                                    cfg.trigger_bit) else 0
                else:
                    val = self.driver.read_word(cfg.trigger_area, cfg.trigger_address)
                target = 1 if is_bit else cfg.trigger_value

                # Rising-edge detection
                if self._last_trigger != target and val == target:
                    if cfg.auto_clear_trigger:
                        try:
                            if is_bit:
                                self.driver.write_bit(cfg.trigger_area,
                                                      cfg.trigger_address,
                                                      cfg.trigger_bit, False)
                            else:
                                self.driver.write_word(cfg.trigger_area,
                                                       cfg.trigger_address, 0)
                        except Exception:
                            pass
                    cb = self._on_trigger
                    if cb is not None:
                        try:
                            cb()
                        except Exception as e:
                            if self._on_error:
                                self._on_error(f"Trigger callback error: {e}")
                self._last_trigger = val
            except Exception as e:
                if self._on_error:
                    self._on_error(f"PLC read error: {e}")
                # Sleep a bit longer on error to avoid spamming
                self._stop.wait(0.5)
                continue
            self._stop.wait(max(0.01, cfg.poll_interval_ms / 1000.0))

    # ── Write back ──
    def write_result(self, passed: bool, values: Optional[list] = None) -> None:
        """Ghi PASS/FAIL + (optional) các giá trị số liệu về PLC."""
        if not self.is_connected:
            raise RuntimeError("PLC not connected")
        cfg = self.config
        code = cfg.result_pass_value if passed else cfg.result_fail_value
        self.driver.write_word(cfg.result_area, cfg.result_address, code)
        if values:
            for i, v in enumerate(values):
                try:
                    iv = int(round(float(v)))
                except (TypeError, ValueError):
                    continue
                self.driver.write_word(cfg.data_area,
                                       cfg.data_start_address + i,
                                       iv)
