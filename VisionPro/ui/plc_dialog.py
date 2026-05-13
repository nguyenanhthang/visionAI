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

import json

from PySide6.QtCore import Qt, Signal, QObject, QSettings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QPlainTextEdit, QMessageBox, QWidget, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView,
)

from core.plc import (
    PLCManager, PLCConfig, MemoryArea, DRIVER_BY_MODEL, DataMapping,
)
from core.tool_registry import TOOL_BY_ID


_DATA_TYPES = ["int16", "int32", "float32", "scaled_int16", "scaled_int32"]


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

    def __init__(self, manager: PLCManager, graph=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("PLC Connection")
        self.resize(720, 820)
        self._mgr = manager
        self._graph = graph
        self._bridge = _ManagerBridge()
        self._bridge.trigger_fired.connect(self._on_trigger_fired)
        self._bridge.error_occured.connect(self._log_error)

        self._build_ui()
        self._load_settings()
        self._refresh_status()

    def set_graph(self, graph) -> None:
        """Cập nhật reference đến FlowGraph hiện hành & refresh combo node."""
        self._graph = graph
        self._refresh_mapping_rows()

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
        gr.addWidget(QLabel("Default data area:"),  2, 0); gr.addWidget(self.cb_data_area, 2, 1)
        gr.addWidget(QLabel("Default start:"),      2, 2); gr.addWidget(self.sp_data_addr, 2, 3)

        self.cb_word_order = QComboBox(); self.cb_word_order.addItems(["ABCD (high word first)", "CDAB (low word first)"])
        gr.addWidget(QLabel("Float/int32 word order:"), 3, 0); gr.addWidget(self.cb_word_order, 3, 1, 1, 3)

        self.btn_send_pass = QPushButton("Send PASS (test)")
        self.btn_send_fail = QPushButton("Send FAIL (test)")
        self.btn_send_pass.clicked.connect(lambda: self._send_test_result(True))
        self.btn_send_fail.clicked.connect(lambda: self._send_test_result(False))
        gr.addWidget(self.btn_send_pass, 4, 0, 1, 2)
        gr.addWidget(self.btn_send_fail, 4, 2, 1, 2)
        root.addWidget(gb_res)

        # — Data mapping table (length, area, count… → PLC) —
        gb_map = QGroupBox("Data mappings (output của node → vùng nhớ PLC)")
        gm = QVBoxLayout(gb_map)

        self.tbl_map = QTableWidget(0, 7)
        self.tbl_map.setHorizontalHeaderLabels(
            ["Node", "Output", "Area", "Address", "Type", "Scale", ""])
        hdr = self.tbl_map.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (2, 3, 4, 5, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.tbl_map.verticalHeader().setVisible(False)
        self.tbl_map.setFixedHeight(180)
        gm.addWidget(self.tbl_map)

        hb_map = QHBoxLayout()
        self.btn_add_map  = QPushButton("➕  Add mapping")
        self.btn_test_map = QPushButton("⇡  Test send mappings")
        self.btn_refresh_map = QPushButton("↻  Refresh node list")
        self.btn_add_map.clicked.connect(lambda: self._add_mapping_row())
        self.btn_test_map.clicked.connect(self._test_send_mappings)
        self.btn_refresh_map.clicked.connect(self._refresh_mapping_rows)
        hb_map.addWidget(self.btn_add_map)
        hb_map.addWidget(self.btn_test_map)
        hb_map.addWidget(self.btn_refresh_map)
        hb_map.addStretch()
        gm.addLayout(hb_map)
        root.addWidget(gb_map)

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

    # ── Mapping table ─────────────────────────────────────────────
    def _node_options(self) -> list:
        """Trả về list (display_text, node_id) cho combo Node."""
        if not self._graph:
            return []
        opts = []
        for nid, node in self._graph.nodes.items():
            tool = TOOL_BY_ID.get(node.tool_id)
            label = tool.name if tool else node.tool_id
            opts.append((f"{label}  [{nid}]", nid))
        return opts

    def _output_options(self, node_id: str) -> list:
        """Trả về list output name của node hiện tại."""
        if not self._graph or node_id not in self._graph.nodes:
            return []
        node = self._graph.nodes[node_id]
        tool = TOOL_BY_ID.get(node.tool_id)
        names = []
        if tool:
            for port in tool.outputs:
                if port.data_type != "image":   # bỏ image, chỉ numeric
                    names.append(port.name)
        # Bổ sung key đang có trong outputs runtime mà tool không khai báo
        for k, v in node.outputs.items():
            if k not in names and isinstance(v, (int, float)) and not isinstance(v, bool):
                names.append(k)
        return names

    def _add_mapping_row(self, mapping: Optional[DataMapping] = None):
        row = self.tbl_map.rowCount()
        self.tbl_map.insertRow(row)

        cb_node = QComboBox()
        cb_out  = QComboBox()
        cb_area = QComboBox(); cb_area.addItems(list(_AREA_LABELS.keys()))
        sp_addr = QSpinBox(); sp_addr.setRange(0, 0xFFFF)
        cb_type = QComboBox(); cb_type.addItems(_DATA_TYPES)
        sp_scale = QDoubleSpinBox(); sp_scale.setDecimals(4); sp_scale.setRange(-1e6, 1e6); sp_scale.setValue(1.0)
        btn_del = QPushButton("✖")

        # Populate node combo
        for txt, nid in self._node_options():
            cb_node.addItem(txt, userData=nid)

        def _refresh_outputs():
            nid = cb_node.currentData()
            current = cb_out.currentText()
            cb_out.clear()
            cb_out.addItems(self._output_options(nid) if nid else [])
            idx = cb_out.findText(current)
            if idx >= 0:
                cb_out.setCurrentIndex(idx)

        cb_node.currentIndexChanged.connect(lambda _: _refresh_outputs())
        _refresh_outputs()

        # Apply saved mapping values if provided
        if mapping:
            idx = cb_node.findData(mapping.node_id)
            if idx >= 0:
                cb_node.setCurrentIndex(idx)
            _refresh_outputs()
            # Cho phép giữ output_key cũ kể cả khi node hiện tại không có
            if cb_out.findText(mapping.output_key) < 0 and mapping.output_key:
                cb_out.addItem(mapping.output_key)
            cb_out.setCurrentText(mapping.output_key)
            cb_area.setCurrentText(_AREA_NAME_BY_ENUM.get(mapping.area, list(_AREA_LABELS.keys())[0]))
            sp_addr.setValue(mapping.address)
            cb_type.setCurrentText(mapping.data_type)
            sp_scale.setValue(mapping.scale)

        btn_del.setFixedWidth(28)
        btn_del.clicked.connect(lambda: self._remove_mapping_row(btn_del))

        self.tbl_map.setCellWidget(row, 0, cb_node)
        self.tbl_map.setCellWidget(row, 1, cb_out)
        self.tbl_map.setCellWidget(row, 2, cb_area)
        self.tbl_map.setCellWidget(row, 3, sp_addr)
        self.tbl_map.setCellWidget(row, 4, cb_type)
        self.tbl_map.setCellWidget(row, 5, sp_scale)
        self.tbl_map.setCellWidget(row, 6, btn_del)

    def _remove_mapping_row(self, btn: QPushButton):
        for row in range(self.tbl_map.rowCount()):
            if self.tbl_map.cellWidget(row, 6) is btn:
                self.tbl_map.removeRow(row)
                return

    def _read_mappings(self) -> list:
        """Đọc bảng → list[DataMapping]."""
        out = []
        for row in range(self.tbl_map.rowCount()):
            cb_node = self.tbl_map.cellWidget(row, 0)
            cb_out  = self.tbl_map.cellWidget(row, 1)
            cb_area = self.tbl_map.cellWidget(row, 2)
            sp_addr = self.tbl_map.cellWidget(row, 3)
            cb_type = self.tbl_map.cellWidget(row, 4)
            sp_scale = self.tbl_map.cellWidget(row, 5)
            nid = cb_node.currentData() if cb_node else None
            okey = cb_out.currentText() if cb_out else ""
            if not nid or not okey:
                continue
            out.append(DataMapping(
                node_id=nid,
                output_key=okey,
                area=_AREA_LABELS[cb_area.currentText()],
                address=sp_addr.value(),
                data_type=cb_type.currentText(),
                scale=sp_scale.value(),
            ))
        return out

    def _refresh_mapping_rows(self):
        """Refresh combo node ở mỗi row khi graph thay đổi (giữ giá trị hiện tại)."""
        existing = self._read_mappings()
        self.tbl_map.setRowCount(0)
        for m in existing:
            self._add_mapping_row(m)

    def _test_send_mappings(self):
        """Đọc graph hiện tại + gửi mappings về PLC để kiểm tra."""
        if not self._mgr.is_connected:
            QMessageBox.warning(self, "Not connected", "Connect PLC trước đã.")
            return
        if self._graph is None:
            QMessageBox.warning(self, "No graph", "Không có pipeline.")
            return
        self._mgr.config = self._gather_config()
        # results dict: {node_id: outputs_dict}
        results = {nid: dict(n.outputs) for nid, n in self._graph.nodes.items()}
        try:
            report = self._mgr.write_data_mappings(results)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))
            return
        for r in report:
            if "error" in r:
                self._log(f"✗ {r['node_id']}.{r['output_key']} @ {r['address']} — {r['error']}")
            else:
                self._log(f"✓ {r['node_id']}.{r['output_key']} = {r['value']} → {r['address']} [{r['data_type']}]")

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
            float_word_order="ABCD" if self.cb_word_order.currentIndex() == 0 else "CDAB",
            data_mappings=self._read_mappings(),
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
        self.cb_word_order.setCurrentIndex(0 if cfg.float_word_order == "ABCD" else 1)
        self.tbl_map.setRowCount(0)
        for m in cfg.data_mappings:
            self._add_mapping_row(m)

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
            self._mgr.write_result(passed=passed)
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
        s.setValue("word_order", cfg.float_word_order)
        s.setValue("mappings", json.dumps([m.to_dict() for m in cfg.data_mappings]))
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
                float_word_order=s.value("word_order", "ABCD"),
                data_mappings=[
                    DataMapping.from_dict(d)
                    for d in json.loads(s.value("mappings", "[]") or "[]")
                ],
            )
            self._apply_config_to_ui(cfg)
        except Exception:
            pass
        s.endGroup()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)
