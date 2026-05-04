"""Image acquisition wrapper.

Bao bọc `open_framegrabber` / `grab_image` của HALCON. Nếu không có HALCON,
fallback sang `cv2.VideoCapture` (USB / webcam) để dev mà không cần licence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .halcon_engine import HALCON_AVAILABLE, _himage_to_numpy

try:
    import halcon as ha  # type: ignore
except Exception:  # pragma: no cover
    ha = None  # type: ignore


@dataclass
class GrabberInfo:
    name: str          # HALCON interface name hoặc "OpenCV"
    devices: list[str] # tên / index thiết bị


def list_grabbers() -> list[GrabberInfo]:
    """Liệt kê các grabber khả dụng."""
    out: list[GrabberInfo] = []
    if HALCON_AVAILABLE:
        try:
            names = ha.info_framegrabber("DirectShow", "info_boards")[1]
            out.append(
                GrabberInfo(name="DirectShow", devices=[str(n) for n in names])
            )
        except Exception:
            pass
        for iface in ("GigEVision2", "USB3Vision", "GenICamTL", "File"):
            try:
                devs = ha.info_framegrabber(iface, "info_boards")[1]
                out.append(
                    GrabberInfo(name=iface, devices=[str(d) for d in devs])
                )
            except Exception:
                continue
    # OpenCV fallback: liệt kê camera index 0..3
    cv_devs: list[str] = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cv_devs.append(f"cam:{idx}")
            cap.release()
    if cv_devs:
        out.append(GrabberInfo(name="OpenCV", devices=cv_devs))
    if not out:
        out.append(GrabberInfo(name="OpenCV", devices=["cam:0"]))
    return out


class Grabber:
    """Wrapper acquisition device — HALCON hoặc OpenCV."""

    def __init__(self):
        self._handle = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend: str = "none"
        self._device_name: str = ""

    @property
    def is_open(self) -> bool:
        return self._handle is not None or (
            self._cap is not None and self._cap.isOpened()
        )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device_name(self) -> str:
        return self._device_name

    def open(self, interface: str, device: str) -> None:
        """Mở grabber. `interface` = 'OpenCV' hoặc tên HALCON interface."""
        self.close()
        if interface == "OpenCV":
            idx = 0
            if device.startswith("cam:"):
                try:
                    idx = int(device.split(":", 1)[1])
                except ValueError:
                    idx = 0
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Không mở được OpenCV camera {device}")
            self._cap = cap
            self._backend = "OpenCV"
            self._device_name = device
            return

        if not HALCON_AVAILABLE:
            raise RuntimeError(
                "HALCON không khả dụng — chỉ có thể dùng backend OpenCV."
            )
        # HALCON open_framegrabber
        self._handle = ha.open_framegrabber(
            interface,
            1, 1,           # HorizontalResolution, VerticalResolution
            0, 0,           # ImageWidth, ImageHeight (0 = default)
            0, 0,           # StartRow, StartColumn
            "default",      # Field
            -1,             # BitsPerChannel
            "default",      # ColorSpace
            -1.0,           # Generic
            "default",      # ExternalTrigger
            device,         # CameraType
            "default",      # Device
            -1, -1,         # Port, LineIn
        )
        self._backend = interface
        self._device_name = device

    def grab(self) -> np.ndarray:
        if not self.is_open:
            raise RuntimeError("Grabber chưa mở.")
        if self._handle is not None:  # HALCON
            himg = ha.grab_image(self._handle)
            return _himage_to_numpy(himg)
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("OpenCV grab thất bại.")
        return frame  # BGR

    def close(self) -> None:
        if self._handle is not None:
            try:
                ha.close_framegrabber(self._handle)
            except Exception:  # pragma: no cover
                pass
            self._handle = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._backend = "none"
        self._device_name = ""
