"""
core/camera.py — Industrial camera wrapper

Hỗ trợ:
  - OpenCV  (USB UVC / RTSP / file index)
  - HikRobot / Do3think GigE & USB3 Vision (qua MVS SDK — Windows only)

API thống nhất qua lớp ``CameraDriver``:
    open(), close()
    grab() -> np.ndarray  (BGR uint8 với color, gray uint8 với mono)
    is_open

CameraRegistry giữ instance persistent — mỗi camera (ENUM index hoặc
serial number) chỉ open 1 lần để tái dùng giữa các lần ``run pipeline``.
"""
from __future__ import annotations

import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import cv2
import numpy as np


# ── Base ──────────────────────────────────────────────────────────
class CameraError(RuntimeError):
    pass


class CameraDriver(ABC):
    def __init__(self):
        self._lock = threading.Lock()
        self.is_open = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def grab(self, timeout_ms: int = 1000) -> np.ndarray: ...


# ── OpenCV ────────────────────────────────────────────────────────
class OpenCVCamera(CameraDriver):
    def __init__(self, index: int = 0, width: int = 0, height: int = 0):
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        with self._lock:
            if self.is_open:
                return
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                raise CameraError(f"OpenCV cannot open camera index {self.index}")
            if self.width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap = cap
            self.is_open = True

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = None
            self.is_open = False

    def grab(self, timeout_ms: int = 1000) -> np.ndarray:
        with self._lock:
            if not self.is_open:
                self.open()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise CameraError(f"OpenCV camera {self.index} read failed")
            return frame


# ── MVS (HikRobot / Do3think) ─────────────────────────────────────
def _import_mvs():
    """Lazy import — chỉ load DLL khi thực sự cần."""
    try:
        from VisionPro.vendor.mvs.MvCameraControl_class import MvCamera
        from VisionPro.vendor.mvs import (
            CameraParams_const as _const,
            CameraParams_header as _hdr,
            PixelType_header as _pix,
        )
    except ImportError:
        # Khi chạy với cwd = VisionPro/
        from vendor.mvs.MvCameraControl_class import MvCamera
        from vendor.mvs import (
            CameraParams_const as _const,
            CameraParams_header as _hdr,
            PixelType_header as _pix,
        )
    return MvCamera, _const, _hdr, _pix


class MVSCamera(CameraDriver):
    """HikRobot / Do3think (MVS SDK).

    Tham số:
        device_index : 0-based, theo thứ tự enumeration
        serial       : nếu set sẽ ưu tiên match theo serial (an toàn hơn index)
        access_mode  : 'exclusive' (mặc định), 'monitor', 'control'
        heartbeat_ms : timeout heartbeat GigE (mặc định 5000)
    """

    _ACCESS = {"exclusive": 1, "monitor": 4, "control": 2}

    def __init__(self, device_index: int = 0, serial: Optional[str] = None,
                 access_mode: str = "exclusive", heartbeat_ms: int = 5000):
        super().__init__()
        self.device_index = device_index
        self.serial = serial
        self.access_mode = access_mode
        self.heartbeat_ms = heartbeat_ms
        self._cam = None
        self._payload_size = 0
        self._buf = None        # ctypes buffer cho raw frame
        self._convert_buf = None  # ctypes buffer cho pixel-converted frame
        self._convert_buf_size = 0

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """Enumerate tất cả MVS camera (GigE + USB3). Trả về list dicts."""
        from ctypes import cast, POINTER
        MvCamera, const, hdr, _pix = _import_mvs()
        dev_list = hdr.MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            const.MV_GIGE_DEVICE | const.MV_USB_DEVICE, dev_list)
        if ret != 0:
            raise CameraError(f"MV_CC_EnumDevices failed 0x{ret:x}")
        out = []
        for i in range(dev_list.nDeviceNum):
            info = cast(dev_list.pDeviceInfo[i],
                        POINTER(hdr.MV_CC_DEVICE_INFO)).contents
            entry = {"index": i}
            if info.nTLayerType == const.MV_GIGE_DEVICE:
                gige = info.SpecialInfo.stGigEInfo
                entry["type"] = "GigE"
                entry["model"] = bytes(gige.chModelName).split(b'\x00', 1)[0].decode(errors='ignore')
                entry["serial"] = bytes(gige.chSerialNumber).split(b'\x00', 1)[0].decode(errors='ignore')
                ip = gige.nCurrentIp
                entry["ip"] = "%d.%d.%d.%d" % (
                    (ip >> 24) & 0xFF, (ip >> 16) & 0xFF,
                    (ip >> 8) & 0xFF, ip & 0xFF)
            elif info.nTLayerType == const.MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                entry["type"] = "USB3"
                entry["model"] = bytes(usb.chModelName).split(b'\x00', 1)[0].decode(errors='ignore')
                entry["serial"] = bytes(usb.chSerialNumber).split(b'\x00', 1)[0].decode(errors='ignore')
            else:
                entry["type"] = "?"
                entry["model"] = ""
                entry["serial"] = ""
            out.append(entry)
        return out

    def open(self) -> None:
        from ctypes import cast, POINTER, c_bool, byref
        with self._lock:
            if self.is_open:
                return
            MvCamera, const, hdr, _pix = _import_mvs()
            dev_list = hdr.MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(
                const.MV_GIGE_DEVICE | const.MV_USB_DEVICE, dev_list)
            if ret != 0:
                raise CameraError(f"MV_CC_EnumDevices failed 0x{ret:x}")
            if dev_list.nDeviceNum == 0:
                raise CameraError("No MVS camera found")

            # Resolve device — by serial if given, else by index
            chosen = None
            if self.serial:
                for i in range(dev_list.nDeviceNum):
                    info = cast(dev_list.pDeviceInfo[i],
                                POINTER(hdr.MV_CC_DEVICE_INFO)).contents
                    sn_field = (info.SpecialInfo.stGigEInfo.chSerialNumber
                                if info.nTLayerType == const.MV_GIGE_DEVICE
                                else info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                    sn = bytes(sn_field).split(b'\x00', 1)[0].decode(errors='ignore')
                    if sn == self.serial:
                        chosen = (i, info)
                        break
                if chosen is None:
                    raise CameraError(f"MVS camera with serial '{self.serial}' not found")
            else:
                if self.device_index >= dev_list.nDeviceNum:
                    raise CameraError(
                        f"Device index {self.device_index} out of range "
                        f"({dev_list.nDeviceNum} devices)")
                info = cast(dev_list.pDeviceInfo[self.device_index],
                            POINTER(hdr.MV_CC_DEVICE_INFO)).contents
                chosen = (self.device_index, info)

            idx, info = chosen
            cam = MvCamera()
            ret = cam.MV_CC_CreateHandle(info)
            if ret != 0:
                raise CameraError(f"MV_CC_CreateHandle failed 0x{ret:x}")

            ret = cam.MV_CC_OpenDevice(self._ACCESS[self.access_mode], 0)
            if ret != 0:
                cam.MV_CC_DestroyHandle()
                raise CameraError(f"MV_CC_OpenDevice failed 0x{ret:x}")

            # GigE-specific tuning
            if info.nTLayerType == const.MV_GIGE_DEVICE:
                pkt = cam.MV_CC_GetOptimalPacketSize()
                if pkt > 0:
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)
                cam.MV_CC_SetIntValue("GevHeartbeatTimeout", self.heartbeat_ms)

            # Free run mode
            cam.MV_CC_SetEnumValue("TriggerMode", 0)  # MV_TRIGGER_MODE_OFF

            # Allocate frame buffer based on PayloadSize
            payload = hdr.MVCC_INTVALUE_EX()
            ret = cam.MV_CC_GetIntValueEx("PayloadSize", payload)
            if ret != 0:
                cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
                raise CameraError(f"GetIntValueEx PayloadSize failed 0x{ret:x}")
            self._payload_size = int(payload.nCurValue)
            from ctypes import c_ubyte
            self._buf = (c_ubyte * self._payload_size)()

            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
                raise CameraError(f"MV_CC_StartGrabbing failed 0x{ret:x}")

            self._cam = cam
            self._mvs = (MvCamera, const, hdr, _pix)
            self.is_open = True

    def close(self) -> None:
        with self._lock:
            if self._cam is None:
                self.is_open = False
                return
            try:
                self._cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self._cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self._cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._cam = None
            self._buf = None
            self._convert_buf = None
            self.is_open = False

    def grab(self, timeout_ms: int = 1000) -> np.ndarray:
        from ctypes import byref, cast, POINTER, c_ubyte, memset, sizeof
        with self._lock:
            if not self.is_open:
                self.open()
            _MvCamera, _const, hdr, pix = self._mvs

            frame_info = hdr.MV_FRAME_OUT_INFO_EX()
            memset(byref(frame_info), 0, sizeof(frame_info))

            ret = self._cam.MV_CC_GetOneFrameTimeout(
                self._buf, self._payload_size, frame_info, timeout_ms)
            if ret != 0:
                raise CameraError(f"MV_CC_GetOneFrameTimeout failed 0x{ret:x}")

            return self._convert_to_ndarray(frame_info, pix, hdr)

    # ── pixel conversion (no msvcrt — works cross-version) ──
    def _convert_to_ndarray(self, frame_info, pix, hdr) -> np.ndarray:
        from ctypes import byref, cast, POINTER, c_ubyte, memset, sizeof
        w, h = int(frame_info.nWidth), int(frame_info.nHeight)
        ptype = frame_info.enPixelType

        # Mono cases — read as gray uint8 directly when possible
        if ptype == pix.PixelType_Gvsp_Mono8:
            arr = np.frombuffer(self._buf, dtype=np.uint8, count=w * h)
            return arr.reshape(h, w).copy()

        # Bayer — convert via OpenCV (faster + correct than SDK convert here)
        bayer_map = {
            pix.PixelType_Gvsp_BayerGB8: cv2.COLOR_BAYER_GB2BGR,
            pix.PixelType_Gvsp_BayerGR8: cv2.COLOR_BAYER_GR2BGR,
            pix.PixelType_Gvsp_BayerRG8: cv2.COLOR_BAYER_RG2BGR,
            pix.PixelType_Gvsp_BayerBG8: cv2.COLOR_BAYER_BG2BGR,
        }
        if ptype in bayer_map:
            arr = np.frombuffer(self._buf, dtype=np.uint8, count=w * h).reshape(h, w)
            return cv2.cvtColor(arr, bayer_map[ptype])

        if ptype == pix.PixelType_Gvsp_BGR8_Packed:
            arr = np.frombuffer(self._buf, dtype=np.uint8, count=w * h * 3)
            return arr.reshape(h, w, 3).copy()  # already BGR

        if ptype == pix.PixelType_Gvsp_RGB8_Packed:
            arr = np.frombuffer(self._buf, dtype=np.uint8, count=w * h * 3)
            return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        # Fallback: dùng SDK pixel converter → BGR8
        need = w * h * 3
        if self._convert_buf is None or self._convert_buf_size < need:
            self._convert_buf = (c_ubyte * need)()
            self._convert_buf_size = need

        cp = hdr.MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(cp), 0, sizeof(cp))
        cp.nWidth = w
        cp.nHeight = h
        cp.pSrcData = cast(self._buf, POINTER(c_ubyte))
        cp.nSrcDataLen = frame_info.nFrameLen
        cp.enSrcPixelType = ptype
        cp.enDstPixelType = pix.PixelType_Gvsp_BGR8_Packed
        cp.pDstBuffer = self._convert_buf
        cp.nDstBufferSize = need
        ret = self._cam.MV_CC_ConvertPixelType(cp)
        if ret != 0:
            raise CameraError(f"MV_CC_ConvertPixelType failed 0x{ret:x}")
        out = np.frombuffer(self._convert_buf, dtype=np.uint8,
                            count=int(cp.nDstLen)).reshape(h, w, 3).copy()
        return out


# ── Persistent registry ───────────────────────────────────────────
class CameraRegistry:
    """Singleton — giữ instance camera giữa các lần grab.

    Key dạng "<backend>:<id>" — vd. "opencv:0", "mvs:idx=0", "mvs:sn=K12345".
    """

    _instance: Optional["CameraRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._cams: Dict[str, CameraDriver] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "CameraRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def get_or_open(self, backend: str, **kwargs) -> CameraDriver:
        backend = backend.lower()
        if backend == "opencv":
            key = f"opencv:{kwargs.get('index', 0)}"
        elif backend == "mvs":
            if kwargs.get("serial"):
                key = f"mvs:sn={kwargs['serial']}"
            else:
                key = f"mvs:idx={kwargs.get('device_index', 0)}"
        else:
            raise CameraError(f"Unknown camera backend: {backend}")

        with self._lock:
            cam = self._cams.get(key)
            if cam is None or not cam.is_open:
                if backend == "opencv":
                    cam = OpenCVCamera(
                        index=int(kwargs.get("index", 0)),
                        width=int(kwargs.get("width", 0)),
                        height=int(kwargs.get("height", 0)))
                else:
                    cam = MVSCamera(
                        device_index=int(kwargs.get("device_index", 0)),
                        serial=kwargs.get("serial"),
                        access_mode=kwargs.get("access_mode", "exclusive"),
                        heartbeat_ms=int(kwargs.get("heartbeat_ms", 5000)))
                cam.open()
                self._cams[key] = cam
            return cam

    def close_all(self) -> None:
        with self._lock:
            for cam in self._cams.values():
                try:
                    cam.close()
                except Exception:
                    pass
            self._cams.clear()
