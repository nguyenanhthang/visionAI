"""
ui/camera_dialog.py — Camera Setup & Live Preview

- Enumerate camera (OpenCV + MVS) và chọn
- Live preview chạy ở thread riêng (không block UI)
- Chỉnh exposure / gain / framerate / trigger mode / pixel format
- Save/Load file config .mfs (chuẩn MVS Studio)
- Snapshot ra file
"""
from __future__ import annotations
import os
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QThread, QSettings
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QSplitter, QWidget, QPlainTextEdit,
)

from core.camera import (
    CameraRegistry, CameraError, MVSCamera, OpenCVCamera,
)


# ── Preview thread ────────────────────────────────────────────────
class _PreviewWorker(QObject):
    frame_ready = Signal(np.ndarray)
    fps_updated = Signal(float)
    error_occured = Signal(str)
    finished = Signal()

    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self._stop = False
        self._timeout_ms = 1000

    def stop(self):
        self._stop = True

    def run(self):
        last = time.perf_counter()
        n = 0
        while not self._stop:
            try:
                frame = self.cam.grab(timeout_ms=self._timeout_ms)
                self.frame_ready.emit(frame)
                n += 1
                now = time.perf_counter()
                if now - last >= 1.0:
                    self.fps_updated.emit(n / (now - last))
                    last = now
                    n = 0
            except CameraError as e:
                self.error_occured.emit(str(e))
                # Backoff tránh spam khi cam offline
                t0 = time.perf_counter()
                while not self._stop and time.perf_counter() - t0 < 0.5:
                    time.sleep(0.05)
        self.finished.emit()


# ── Image viewer widget ───────────────────────────────────────────
class _PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#000; border:1px solid #1e2d45;")
        self.setText("No preview")
        self._last_frame: Optional[np.ndarray] = None

    def show_frame(self, frame: np.ndarray):
        self._last_frame = frame
        if frame.ndim == 2:
            h, w = frame.shape
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, _ = frame.shape
            # BGR → RGB cho QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        # Scale theo widget, giữ tỉ lệ
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, ev):
        if self._last_frame is not None:
            self.show_frame(self._last_frame)
        super().resizeEvent(ev)


# ── Main dialog ───────────────────────────────────────────────────
class CameraSetupDialog(QDialog):
    """Setup + Live Preview cho HikRobot/Do3think (MVS) hoặc OpenCV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Setup & Live Preview")
        self.resize(1180, 760)

        self._cam = None              # current opened CameraDriver
        self._worker: Optional[_PreviewWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()
        self._refresh_devices()
        self._restore_settings()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ── Left: preview ──
        left = QWidget()
        lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        self.preview = _PreviewLabel()
        lv.addWidget(self.preview, 1)

        hb = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start preview")
        self.btn_stop  = QPushButton("■  Stop")
        self.btn_snap  = QPushButton("📷  Snapshot")
        self.btn_trigger = QPushButton("⚡  Software trigger")
        self.btn_stop.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_trigger.setEnabled(False)
        self.btn_start.clicked.connect(self._start_preview)
        self.btn_stop.clicked.connect(self._stop_preview)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_trigger.clicked.connect(self._software_trigger)
        for b in (self.btn_start, self.btn_stop, self.btn_snap, self.btn_trigger):
            hb.addWidget(b)
        hb.addStretch()
        self.lbl_fps = QLabel("0.0 fps")
        self.lbl_fps.setStyleSheet("color:#00d4ff;font-family:'Courier New';")
        hb.addWidget(self.lbl_fps)
        lv.addLayout(hb)
        split.addWidget(left)

        # ── Right: config ──
        right = QWidget()
        rv = QVBoxLayout(right)

        # Device picker
        gb_dev = QGroupBox("Device")
        gd = QGridLayout(gb_dev)
        self.cb_backend = QComboBox()
        self.cb_backend.addItems(["HikRobot/Do3think", "OpenCV"])
        self.cb_backend.currentTextChanged.connect(self._on_backend_changed)
        self.cb_device = QComboBox()
        self.cb_device.setMinimumWidth(280)
        self.btn_refresh = QPushButton("↻  Scan")
        self.btn_open = QPushButton("🔌  Open")
        self.btn_close = QPushButton("✖  Close")
        self.btn_close.setEnabled(False)
        self.btn_refresh.clicked.connect(self._refresh_devices)
        self.btn_open.clicked.connect(self._open_camera)
        self.btn_close.clicked.connect(self._close_camera)

        gd.addWidget(QLabel("Backend:"), 0, 0); gd.addWidget(self.cb_backend, 0, 1, 1, 3)
        gd.addWidget(QLabel("Device:"),  1, 0); gd.addWidget(self.cb_device,  1, 1, 1, 3)
        gd.addWidget(self.btn_refresh, 2, 0)
        gd.addWidget(self.btn_open,    2, 1)
        gd.addWidget(self.btn_close,   2, 2)
        rv.addWidget(gb_dev)

        # Image params
        self.gb_img = QGroupBox("Image acquisition")
        gi = QGridLayout(self.gb_img)
        self.sp_exposure = QDoubleSpinBox(); self.sp_exposure.setRange(1, 1e7); self.sp_exposure.setDecimals(1); self.sp_exposure.setSuffix(" μs"); self.sp_exposure.setValue(10000)
        self.sp_gain = QDoubleSpinBox(); self.sp_gain.setRange(0, 50); self.sp_gain.setDecimals(2); self.sp_gain.setSuffix(" dB"); self.sp_gain.setValue(0)
        self.sp_fps = QDoubleSpinBox(); self.sp_fps.setRange(0.1, 1000); self.sp_fps.setDecimals(1); self.sp_fps.setSuffix(" fps"); self.sp_fps.setValue(30)
        self.cb_pixel_fmt = QComboBox()
        self.cb_pixel_fmt.addItems(["(keep)", "Mono8", "BayerGB8", "BayerGR8", "BayerRG8", "BayerBG8", "RGB8Packed", "BGR8Packed"])
        gi.addWidget(QLabel("Exposure:"),     0, 0); gi.addWidget(self.sp_exposure, 0, 1)
        gi.addWidget(QLabel("Gain:"),         1, 0); gi.addWidget(self.sp_gain,     1, 1)
        gi.addWidget(QLabel("Frame rate:"),   2, 0); gi.addWidget(self.sp_fps,      2, 1)
        gi.addWidget(QLabel("Pixel format:"), 3, 0); gi.addWidget(self.cb_pixel_fmt, 3, 1)
        self.btn_apply_img = QPushButton("Apply image params")
        self.btn_apply_img.clicked.connect(self._apply_image_params)
        gi.addWidget(self.btn_apply_img, 4, 0, 1, 2)
        rv.addWidget(self.gb_img)

        # Trigger
        self.gb_trig = QGroupBox("Trigger")
        gt = QGridLayout(self.gb_trig)
        self.chk_trig = QCheckBox("Trigger mode ON")
        self.cb_trig_src = QComboBox()
        self.cb_trig_src.addItems(["Line0", "Line1", "Line2", "Line3", "Counter0", "Software"])
        self.btn_apply_trig = QPushButton("Apply trigger")
        self.btn_apply_trig.clicked.connect(self._apply_trigger)
        gt.addWidget(self.chk_trig,        0, 0, 1, 2)
        gt.addWidget(QLabel("Source:"),    1, 0); gt.addWidget(self.cb_trig_src, 1, 1)
        gt.addWidget(self.btn_apply_trig,  2, 0, 1, 2)
        rv.addWidget(self.gb_trig)

        # ROI
        self.gb_roi = QGroupBox("ROI")
        gr = QGridLayout(self.gb_roi)
        self.sp_w = QSpinBox(); self.sp_w.setRange(8, 16384)
        self.sp_h = QSpinBox(); self.sp_h.setRange(8, 16384)
        self.sp_x = QSpinBox(); self.sp_x.setRange(0, 16384)
        self.sp_y = QSpinBox(); self.sp_y.setRange(0, 16384)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        gr.addWidget(QLabel("Width:"),  0, 0); gr.addWidget(self.sp_w, 0, 1)
        gr.addWidget(QLabel("Height:"), 0, 2); gr.addWidget(self.sp_h, 0, 3)
        gr.addWidget(QLabel("OffsetX:"),1, 0); gr.addWidget(self.sp_x, 1, 1)
        gr.addWidget(QLabel("OffsetY:"),1, 2); gr.addWidget(self.sp_y, 1, 3)
        gr.addWidget(self.btn_apply_roi, 2, 0, 1, 4)
        rv.addWidget(self.gb_roi)

        # Feature file
        gb_file = QGroupBox("Feature file (.mfs)")
        gf = QHBoxLayout(gb_file)
        self.btn_save_feat = QPushButton("💾  Save…")
        self.btn_load_feat = QPushButton("📂  Load…")
        self.btn_save_feat.clicked.connect(self._save_features)
        self.btn_load_feat.clicked.connect(self._load_features)
        gf.addWidget(self.btn_save_feat); gf.addWidget(self.btn_load_feat)
        rv.addWidget(gb_file)

        # Log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(300)
        self.log.setFixedHeight(120)
        rv.addWidget(self.log, 1)

        btn_close_dlg = QPushButton("Close")
        btn_close_dlg.clicked.connect(self.accept)
        rv.addWidget(btn_close_dlg)

        split.addWidget(right)
        split.setSizes([720, 460])

        self._set_param_enabled(False)

    def _set_param_enabled(self, on: bool):
        for w in (self.gb_img, self.gb_trig, self.gb_roi,
                  self.btn_save_feat, self.btn_load_feat):
            w.setEnabled(on)

    # ── Device enumeration ───────────────────────────────────────
    def _on_backend_changed(self, _):
        self._refresh_devices()

    def _refresh_devices(self):
        self.cb_device.clear()
        backend = self.cb_backend.currentText()
        try:
            if backend == "OpenCV":
                # Probe 0-3 — không có cách thuần để enumerate UVC trong opencv
                for i in range(4):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else 0)
                    ok = cap.isOpened()
                    cap.release()
                    if ok:
                        self.cb_device.addItem(f"OpenCV index {i}", userData={"backend": "opencv", "index": i})
                if self.cb_device.count() == 0:
                    self.cb_device.addItem("(no UVC camera found)", userData=None)
            else:
                devs = MVSCamera.list_devices()
                if not devs:
                    self.cb_device.addItem("(no MVS camera found)", userData=None)
                else:
                    for d in devs:
                        label = f"[{d['index']}] {d['type']} {d['model']}  sn={d['serial']}"
                        if d.get("ip"):
                            label += f"  ip={d['ip']}"
                        self.cb_device.addItem(label, userData={
                            "backend": "mvs",
                            "device_index": d["index"],
                            "serial": d["serial"],
                        })
            self._log(f"Scan {backend}: found {self.cb_device.count()} device(s)")
        except Exception as e:
            self.cb_device.addItem(f"(scan failed: {e})", userData=None)
            self._log(f"Scan error: {e}")

    # ── Camera open / close ──────────────────────────────────────
    def _open_camera(self):
        data = self.cb_device.currentData()
        if not data:
            QMessageBox.warning(self, "No device", "Chọn 1 device hợp lệ trước.")
            return
        self._stop_preview()
        # Không đóng cam cũ thủ công — registry sẽ tái dùng nếu cùng device,
        # hoặc cấp instance mới nếu khác. Cam cũ vẫn nằm trong registry để
        # pipeline / Camera Acquire tool dùng chung.
        self._cam = None

        reg = CameraRegistry.instance()
        try:
            if data["backend"] == "opencv":
                cam = reg.get_or_open("opencv", index=data["index"])
            else:
                kwargs = {"device_index": data["device_index"]}
                if data.get("serial"):
                    kwargs["serial"] = data["serial"]
                cam = reg.get_or_open("mvs", **kwargs)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            self._log(f"Open failed: {e}")
            return

        self._cam = cam
        self._log(f"Opened: {self.cb_device.currentText()}")
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)
        self._set_param_enabled(isinstance(cam, MVSCamera))
        self._pull_current_params()

    def _close_camera(self):
        self._stop_preview()
        if self._cam is not None:
            # Đóng qua registry để pop entry → lần Open kế dùng instance mới
            data = self.cb_device.currentData() or {}
            reg = CameraRegistry.instance()
            try:
                if data.get("backend") == "opencv":
                    reg.close("opencv", index=data.get("index", 0))
                elif data.get("backend") == "mvs":
                    reg.close("mvs",
                              device_index=data.get("device_index", 0),
                              serial=data.get("serial"))
                else:
                    self._cam.close()
            except Exception:
                pass
            self._cam = None
            self._log("Closed camera")
        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)
        self._set_param_enabled(False)

    def _pull_current_params(self):
        """Đọc các param hiện tại của cam và load vào UI."""
        if not isinstance(self._cam, MVSCamera):
            return
        try:
            v = self._cam.get_float("ExposureTime");      self.sp_exposure.setValue(v["current"])
        except CameraError: pass
        try:
            v = self._cam.get_float("Gain");              self.sp_gain.setValue(v["current"])
        except CameraError: pass
        try:
            v = self._cam.get_float("AcquisitionFrameRate"); self.sp_fps.setValue(v["current"])
        except CameraError: pass
        try:
            v = self._cam.get_int("Width");   self.sp_w.setValue(v["current"]); self.sp_w.setRange(v["min"], v["max"])
            v = self._cam.get_int("Height");  self.sp_h.setValue(v["current"]); self.sp_h.setRange(v["min"], v["max"])
            v = self._cam.get_int("OffsetX"); self.sp_x.setValue(v["current"]); self.sp_x.setRange(v["min"], v["max"])
            v = self._cam.get_int("OffsetY"); self.sp_y.setValue(v["current"]); self.sp_y.setRange(v["min"], v["max"])
        except CameraError: pass

    # ── Preview ──────────────────────────────────────────────────
    def _start_preview(self):
        if self._cam is None:
            QMessageBox.warning(self, "Not opened", "Open camera trước.")
            return
        if self._thread and self._thread.isRunning():
            return
        self._worker = _PreviewWorker(self._cam)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self.preview.show_frame)
        self._worker.fps_updated.connect(lambda f: self.lbl_fps.setText(f"{f:5.1f} fps"))
        self._worker.error_occured.connect(self._on_preview_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_snap.setEnabled(True)
        is_mvs = isinstance(self._cam, MVSCamera)
        self.btn_trigger.setEnabled(is_mvs and self.chk_trig.isChecked())
        self._log("Preview started")

    def _stop_preview(self):
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_trigger.setEnabled(False)
        self.lbl_fps.setText("0.0 fps")

    def _on_preview_error(self, msg: str):
        self._log(f"Preview error: {msg}")

    def _snapshot(self):
        frame = self.preview._last_frame
        if frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", f"snapshot_{int(time.time())}.png",
            "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        ok = cv2.imwrite(path, frame)
        self._log(f"{'Saved' if ok else 'Failed save'}: {path}")

    def _software_trigger(self):
        if not isinstance(self._cam, MVSCamera):
            return
        try:
            self._cam.execute_command("TriggerSoftware")
            self._log("Software trigger fired")
        except CameraError as e:
            self._log(f"Trigger failed: {e}")

    # ── Apply param groups ───────────────────────────────────────
    def _apply_image_params(self):
        if not isinstance(self._cam, MVSCamera):
            return
        errs = []
        try: self._cam.set_exposure(self.sp_exposure.value())
        except CameraError as e: errs.append(f"exposure: {e}")
        try: self._cam.set_gain(self.sp_gain.value())
        except CameraError as e: errs.append(f"gain: {e}")
        try: self._cam.set_frame_rate(self.sp_fps.value())
        except CameraError as e: errs.append(f"fps: {e}")
        fmt = self.cb_pixel_fmt.currentText()
        if fmt and fmt != "(keep)":
            try: self._cam.set_pixel_format(fmt)
            except CameraError as e: errs.append(f"pixel format: {e}")
        if errs:
            self._log("Apply image errors: " + "; ".join(errs))
        else:
            self._log("Image params applied")

    def _apply_trigger(self):
        if not isinstance(self._cam, MVSCamera):
            return
        try:
            self._cam.set_trigger_mode(self.chk_trig.isChecked())
            if self.chk_trig.isChecked():
                self._cam.set_enum_str("TriggerSource", self.cb_trig_src.currentText())
            self._log(f"Trigger {'ON ('+self.cb_trig_src.currentText()+')' if self.chk_trig.isChecked() else 'OFF'}")
            self.btn_trigger.setEnabled(
                self.chk_trig.isChecked() and self._thread is not None and self._thread.isRunning())
        except CameraError as e:
            self._log(f"Trigger apply error: {e}")

    def _apply_roi(self):
        if not isinstance(self._cam, MVSCamera):
            return
        # Phải stop grabbing để đổi ROI
        was_preview = self._thread is not None and self._thread.isRunning()
        self._stop_preview()
        try:
            self._cam._cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        errs = []
        # Thứ tự: Width/Height TRƯỚC, sau đó Offset (tránh exceed)
        for key, sp in [("Width", self.sp_w), ("Height", self.sp_h),
                        ("OffsetX", self.sp_x), ("OffsetY", self.sp_y)]:
            try:
                self._cam.set_int(key, sp.value())
            except CameraError as e:
                errs.append(f"{key}: {e}")
        # Realloc buffer (PayloadSize đã đổi)
        try:
            _, _, hdr, _ = self._cam._mvs
            payload = hdr.MVCC_INTVALUE_EX()
            self._cam._cam.MV_CC_GetIntValueEx("PayloadSize", payload)
            from ctypes import c_ubyte
            self._cam._payload_size = int(payload.nCurValue)
            self._cam._buf = (c_ubyte * self._cam._payload_size)()
        except Exception as e:
            errs.append(f"realloc: {e}")
        try:
            self._cam._cam.MV_CC_StartGrabbing()
        except Exception:
            pass
        self._log("ROI applied" + (f" with errors: {errs}" if errs else ""))
        if was_preview:
            self._start_preview()

    # ── Feature file ─────────────────────────────────────────────
    def _save_features(self):
        if not isinstance(self._cam, MVSCamera):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MVS features", "camera.mfs", "MVS Feature (*.mfs)")
        if not path:
            return
        try:
            self._cam.save_features(path)
            self._log(f"Saved features → {path}")
        except CameraError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _load_features(self):
        if not isinstance(self._cam, MVSCamera):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MVS features", "", "MVS Feature (*.mfs);;All Files (*)")
        if not path:
            return
        was_preview = self._thread is not None and self._thread.isRunning()
        self._stop_preview()
        try:
            self._cam.load_features(path)
            self._log(f"Loaded features ← {path}")
            self._pull_current_params()
        except CameraError as e:
            QMessageBox.critical(self, "Load failed", str(e))
        if was_preview:
            self._start_preview()

    # ── Settings persistence ─────────────────────────────────────
    def _log(self, msg: str):
        from datetime import datetime
        self.log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _restore_settings(self):
        s = QSettings(); s.beginGroup("camera_setup")
        idx = self.cb_backend.findText(s.value("backend", "HikRobot/Do3think"))
        if idx >= 0:
            self.cb_backend.setCurrentIndex(idx)
        s.endGroup()

    def closeEvent(self, ev):
        s = QSettings(); s.beginGroup("camera_setup")
        s.setValue("backend", self.cb_backend.currentText())
        s.endGroup()
        self._stop_preview()
        # Không close cam — registry giữ lại để pipeline dùng tiếp
        super().closeEvent(ev)
