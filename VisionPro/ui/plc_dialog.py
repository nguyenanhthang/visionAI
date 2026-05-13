"""
ui/plc_dialog.py — Dialog cấu hình & giám sát kết nối PLC

- Chọn loại PLC: Omron CP2E / Omron NX1P2 / Inovance H3U-H5U
- Cấu hình IP, port, polling interval
- Cấu hình vùng nhớ trigger (DM word / CIO bit / DM bit)
- Cấu hình vùng nhớ result (PASS/FAIL) + data
- Test connection, Read/Write thủ công, Start/Stop monitor
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QSettings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QPushButton, QCheckBox, QPlainTextEdit,
    QMessageBox, QWidget, QFrame,
)

from core.plc import (
    PLCManager, PLCConfig, MemoryArea, DRIVER_BY_MODEL,
)


_AREA_LABELS = {
    "DM Word (D)":  MemoryArea.DM_WORD,
    "DM Bit":       MemoryArea.DM_BIT,
    "CIO Word":     MemoryArea.CIO_WORD,
    "CIO Bit":      MemoryArea.CIO_BIT,
    "Work Word":    MemoryArea.W_WORD,
    "Holding Word": MemoryArea.H_WORD,
}
_AREA_NAME_BY_ENUM = {v: k for k, v in _AREA_LABELS.items()}


class _ManagerBridge(QObject):
    """Bridge để callback từ thread monitor về Qt main thread."""
    trigger_fired = Signal()
    error_occured = Signal(str)


class PLCDialog(QDialog):
    """Dialog cấu hình PLC. Phát signal ``trigger_fired`` khi PLC kích hoạt."""

    trigger_fired = Signal()   # Forward ra MainWindow để chạy pipeline

    def __init__(self, manager: PLCManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("PLC Connection")
        self.resize(560, 640)
        self._mgr = manager
        self._bridge = _ManagerBridge()
        self._bridge.trigger_fired.connect(self._on_trigger_fired)
        self._bridge.error_occured.connect(self._log_error)

        self._build_ui()
        self._load_settings()
        self._refresh_status()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # — Connection group —
        gb_conn = QGroupBox("Connection")
        g = QGridLayout(gb_conn)

        self.cb_model = QComboBox()
        self.cb_model.addItems(list(DRIVER_BY_MODEL.keys()))
        self.cb_model.currentTextChanged.connect(self._on_model_changed)

        self.le_ip = QLineEdit("192.168.250.1")
        self.sp_port = QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(9600)
        self.sp_poll = QSpinBox(); self.sp_poll.setRange(10, 5000); self.sp_poll.setValue(100); self.sp_poll.setSuffix(" ms")

        g.addWidget(QLabel("Model:"),    0, 0); g.addWidget(self.cb_model, 0, 1, 1, 3)
        g.addWidget(QLabel("IP:"),       1, 0); g.addWidget(self.le_ip,    1, 1)
        g.addWidget(QLabel("Port:"),     1, 2); g.addWidget(self.sp_port,  1, 3)
        g.addWidget(QLabel("Poll:"),     2, 0); g.addWidget(self.sp_poll,  2, 1)

        self.btn_connect = QPushButton("🔌  Connect")
        self.btn_disconnect = QPushButton("✖  Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        g.addWidget(self.btn_connect,    3, 0, 1, 2)
        g.addWidget(self.btn_disconnect, 3, 2, 1, 2)
        root.addWidget(gb_conn)

        # — Trigger group —
        gb_trig = QGroupBox("Trigger (PLC → AOI)")
        gt = QGridLayout(gb_trig)

        self.cb_trig_area = QComboBox(); self.cb_trig_area.addItems(list(_AREA_LABELS.keys()))
        self.sp_trig_addr = QSpinBox(); self.sp_trig_addr.setRange(0, 0xFFFF); self.sp_trig_addr.setValue(100)
        self.sp_trig_bit  = QSpinBox(); self.sp_trig_bit.setRange(0, 15); self.sp_trig_bit.setValue(0)
        self.sp_trig_val  = QSpinBox(); self.sp_trig_val.setRange(0, 0xFFFF); self.sp_trig_val.setValue(1)
        self.chk_autoclr  = QCheckBox("Auto-clear sau khi nhận"); self.chk_autoclr.setChecked(True)

        gt.addWidget(QLabel("Area:"),     0, 0); gt.addWidget(self.cb_trig_area, 0, 1)
        gt.addWidget(QLabel("Address:"),  0, 2); gt.addWidget(self.sp_trig_addr, 0, 3)
        gt.addWidget(QLabel("Bit (nếu *_BIT):"), 1, 0); gt.addWidget(self.sp_trig_bit, 1, 1)
        gt.addWidget(QLabel("Value (word):"),    1, 2); gt.addWidget(self.sp_trig_val, 1, 3)
        gt.addWidget(self.chk_autoclr, 2, 0, 1, 4)

        self.btn_start = QPushButton("▶  Start monitor")
        self.btn_stop  = QPushButton("■  Stop monitor")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start_monitor)
        self.btn_stop.clicked.connect(self._on_stop_monitor)
        gt.addWidget(self.btn_start, 3, 0, 1, 2)
        gt.addWidget(self.btn_stop,  3, 2, 1, 2)
        root.addWidget(gb_trig)

        # — Result group —
        gb_res = QGroupBox("Result (AOI → PLC)")
        gr = QGridLayout(gb_res)

        self.cb_res_area = QComboBox(); self.cb_res_area.addItems(list(_AREA_LABELS.keys()))
        self.sp_res_addr = QSpinBox(); self.sp_res_addr.setRange(0, 0xFFFF); self.sp_res_addr.setValue(101)
        self.sp_pass_val = QSpinBox(); self.sp_pass_val.setRange(0, 0xFFFF); self.sp_pass_val.setValue(1)
        self.sp_fail_val = QSpinBox(); self.sp_fail_val.setRange(0, 0xFFFF); self.sp_fail_val.setValue(2)

        gr.addWidget(QLabel("Area:"),         0, 0); gr.addWidget(self.cb_res_area, 0, 1)
        gr.addWidget(QLabel("Address:"),      0, 2); gr.addWidget(self.sp_res_addr, 0, 3)
        gr.addWidget(QLabel("PASS value:"),   1, 0); gr.addWidget(self.sp_pass_val, 1, 1)
        gr.addWidget(QLabel("FAIL value:"),   1, 2); gr.addWidget(self.sp_fail_val, 1, 3)

        self.cb_data_area = QComboBox(); self.cb_data_area.addItems(list(_AREA_LABELS.keys()))
        self.sp_data_addr = QSpinBox(); self.sp_data_addr.setRange(0, 0xFFFF); self.sp_data_addr.setValue(110)
        gr.addWidget(QLabel("Data area:"),    2, 0); gr.addWidget(self.cb_data_area, 2, 1)
        gr.addWidget(QLabel("Data start:"),   2, 2); gr.addWidget(self.sp_data_addr, 2, 3)

        self.btn_send_pass = QPushButton("Send PASS (test)")
        self.btn_send_fail = QPushButton("Send FAIL (test)")
        self.btn_send_pass.clicked.connect(lambda: self._send_test_result(True))
        self.btn_send_fail.clicked.connect(lambda: self._send_test_result(False))
        gr.addWidget(self.btn_send_pass, 3, 0, 1, 2)
        gr.addWidget(self.btn_send_fail, 3, 2, 1, 2)
        root.addWidget(gb_res)

        # — Log —
        self.lbl_status = QLabel("● Disconnected")
        self.lbl_status.setStyleSheet("color:#ff3860;font-weight:700;")
        root.addWidget(self.lbl_status)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(140)
        root.addWidget(self.log)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        root.addWidget(btn_close)

    # ── Helpers ───────────────────────────────────────────────────
    def _on_model_changed(self, model: str):
        if "NX1P2" in model:
            self.sp_port.setValue(9600)
        elif "CP2E" in model:
            self.sp_port.setValue(9600)
        elif "H3U" in model or "H5U" in model:
            self.sp_port.setValue(502)

    def _gather_config(self) -> PLCConfig:
        return PLCConfig(
            model=self.cb_model.currentText(),
            ip=self.le_ip.text().strip(),
            port=self.sp_port.value(),
            poll_interval_ms=self.sp_poll.value(),
            trigger_area=_AREA_LABELS[self.cb_trig_area.currentText()],
            trigger_address=self.sp_trig_addr.value(),
            trigger_bit=self.sp_trig_bit.value(),
            trigger_value=self.sp_trig_val.value(),
            auto_clear_trigger=self.chk_autoclr.isChecked(),
            result_area=_AREA_LABELS[self.cb_res_area.currentText()],
            result_address=self.sp_res_addr.value(),
            result_pass_value=self.sp_pass_val.value(),
            result_fail_value=self.sp_fail_val.value(),
            data_area=_AREA_LABELS[self.cb_data_area.currentText()],
            data_start_address=self.sp_data_addr.value(),
        )

    def _apply_config_to_ui(self, cfg: PLCConfig):
        idx = self.cb_model.findText(cfg.model)
        if idx >= 0: self.cb_model.setCurrentIndex(idx)
        self.le_ip.setText(cfg.ip)
        self.sp_port.setValue(cfg.port)
        self.sp_poll.setValue(cfg.poll_interval_ms)
        self.cb_trig_area.setCurrentText(_AREA_NAME_BY_ENUM[cfg.trigger_area])
        self.sp_trig_addr.setValue(cfg.trigger_address)
        self.sp_trig_bit.setValue(cfg.trigger_bit)
        self.sp_trig_val.setValue(cfg.trigger_value)
        self.chk_autoclr.setChecked(cfg.auto_clear_trigger)
        self.cb_res_area.setCurrentText(_AREA_NAME_BY_ENUM[cfg.result_area])
        self.sp_res_addr.setValue(cfg.result_address)
        self.sp_pass_val.setValue(cfg.result_pass_value)
        self.sp_fail_val.setValue(cfg.result_fail_value)
        self.cb_data_area.setCurrentText(_AREA_NAME_BY_ENUM[cfg.data_area])
        self.sp_data_addr.setValue(cfg.data_start_address)

    def _refresh_status(self):
        if self._mgr.is_connected:
            self.lbl_status.setText(
                f"● Connected to {self._mgr.config.model} @ {self._mgr.config.ip}"
                + ("  [monitoring]" if self._mgr.is_monitoring else ""))
            self.lbl_status.setStyleSheet("color:#39ff14;font-weight:700;")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(not self._mgr.is_monitoring)
            self.btn_stop.setEnabled(self._mgr.is_monitoring)
        else:
            self.lbl_status.setText("● Disconnected")
            self.lbl_status.setStyleSheet("color:#ff3860;font-weight:700;")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(False)

    def _log(self, msg: str):
        from datetime import datetime
        self.log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _log_error(self, msg: str):
        self._log("ERROR: " + msg)

    # ── Actions ───────────────────────────────────────────────────
    def _on_connect(self):
        self._mgr.config = self._gather_config()
        try:
            self._mgr.connect()
            self._log(f"Connected to {self._mgr.config.model} @ {self._mgr.config.ip}:{self._mgr.config.port}")
        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))
            self._log(f"Connect failed: {e}")
        self._refresh_status()
        self._save_settings()

    def _on_disconnect(self):
        self._mgr.disconnect()
        self._log("Disconnected")
        self._refresh_status()

    def _on_start_monitor(self):
        # Cập nhật config (đặc biệt là trigger fields) trước khi start
        self._mgr.config = self._gather_config()
        try:
            self._mgr.start_monitor(
                on_trigger=self._bridge.trigger_fired.emit,
                on_error=self._bridge.error_occured.emit,
            )
            cfg = self._mgr.config
            self._log(f"Monitoring {cfg.trigger_area.value} @ {cfg.trigger_address} "
                      f"every {cfg.poll_interval_ms}ms")
        except Exception as e:
            QMessageBox.critical(self, "Start monitor failed", str(e))
        self._refresh_status()
        self._save_settings()

    def _on_stop_monitor(self):
        self._mgr.stop_monitor()
        self._log("Monitor stopped")
        self._refresh_status()

    def _on_trigger_fired(self):
        self._log("⚡ Trigger received → run pipeline")
        self.trigger_fired.emit()

    def _send_test_result(self, passed: bool):
        self._mgr.config = self._gather_config()
        try:
            self._mgr.write_result(passed=passed, values=None)
            self._log(f"Sent {'PASS' if passed else 'FAIL'} → "
                      f"{self._mgr.config.result_area.value}{self._mgr.config.result_address}")
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))

    # ── Persistence ───────────────────────────────────────────────
    def _save_settings(self):
        s = QSettings()
        s.beginGroup("plc")
        cfg = self._gather_config()
        s.setValue("model", cfg.model)
        s.setValue("ip", cfg.ip)
        s.setValue("port", cfg.port)
        s.setValue("poll", cfg.poll_interval_ms)
        s.setValue("trig_area", cfg.trigger_area.value)
        s.setValue("trig_addr", cfg.trigger_address)
        s.setValue("trig_bit", cfg.trigger_bit)
        s.setValue("trig_val", cfg.trigger_value)
        s.setValue("auto_clr", cfg.auto_clear_trigger)
        s.setValue("res_area", cfg.result_area.value)
        s.setValue("res_addr", cfg.result_address)
        s.setValue("pass_val", cfg.result_pass_value)
        s.setValue("fail_val", cfg.result_fail_value)
        s.setValue("data_area", cfg.data_area.value)
        s.setValue("data_addr", cfg.data_start_address)
        s.endGroup()

    def _load_settings(self):
        s = QSettings()
        s.beginGroup("plc")
        try:
            cfg = PLCConfig(
                model=s.value("model", "Omron CP2E"),
                ip=s.value("ip", "192.168.250.1"),
                port=int(s.value("port", 9600)),
                poll_interval_ms=int(s.value("poll", 100)),
                trigger_area=MemoryArea(s.value("trig_area", MemoryArea.DM_WORD.value)),
                trigger_address=int(s.value("trig_addr", 100)),
                trigger_bit=int(s.value("trig_bit", 0)),
                trigger_value=int(s.value("trig_val", 1)),
                auto_clear_trigger=s.value("auto_clr", True, type=bool),
                result_area=MemoryArea(s.value("res_area", MemoryArea.DM_WORD.value)),
                result_address=int(s.value("res_addr", 101)),
                result_pass_value=int(s.value("pass_val", 1)),
                result_fail_value=int(s.value("fail_val", 2)),
                data_area=MemoryArea(s.value("data_area", MemoryArea.DM_WORD.value)),
                data_start_address=int(s.value("data_addr", 110)),
            )
            self._apply_config_to_ui(cfg)
        except Exception:
            pass
        s.endGroup()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)
