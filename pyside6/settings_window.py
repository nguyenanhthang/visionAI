"""Settings dialog cho config.py.

Cơ chế: config.py giữ DEFAULT. Override được lưu vào settings.json
cùng folder; ``load_settings_overrides()`` patch attr lên module
config lúc startup (gọi từ main.py). Dialog cho phép chỉnh +
lưu vào JSON.

Một số trường (PLC_IP, COM port…) cần restart để áp dụng cho
worker thread đang chạy — dialog cảnh báo sau khi save.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

import config


SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"


def load_settings_overrides() -> dict:
    """Đọc settings.json và patch attr lên module config."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for k, v in data.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return data


def save_settings(data: dict):
    SETTINGS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for k, v in data.items():
        if hasattr(config, k):
            setattr(config, k, v)


class _DirField(QWidget):
    """LineEdit + Browse… để chọn folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        self.edit = QLineEdit()
        self.btn = QPushButton("Browse…")
        self.btn.clicked.connect(self._pick)
        h.addWidget(self.edit, 1); h.addWidget(self.btn)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn folder", self.edit.text())
        if d:
            self.edit.setText(d)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, v: str):
        self.edit.setText(v)


class SettingsDialog(QDialog):
    """Tabs: Scanner · PLC · API/SFC · Folders."""

    # (key, label, type) — type ∈ {"str","int","float","bool","dir"}
    SCHEMA = {
        "Scanner": [
            ("SCANNER_PORT",     "COM port",            "str"),
            ("SCANNER_BAUDRATE", "Baudrate",            "int"),
            ("SCANNER_READ_SIZE","Read size badge",     "int"),
            ("SCANNER_TIMEOUT",  "Timeout (s)",         "float"),
            ("PRODUCT_READ_SIZE","Read size SP",        "int"),
            ("Staff",            "Mã NV mặc định",      "str"),
        ],
        "PLC": [
            ("PLC_IP",                "PLC IP",                 "str"),
            ("PLC_RESULT_ADDR",       "Reg đọc verdict (300)",  "int"),
            ("PLC_SCAN_RESULT_ADDR",  "Reg ghi scan (250)",     "int"),
            ("PLC_POLL_HZ",           "Poll Hz",                "int"),
            ("PLC_SIMULATED",         "Chạy giả lập",           "bool"),
        ],
        "API / SFC": [
            ("STATION_NAME",            "Station name",          "str"),
            ("ON_OFF_SFC",              "Bật push SFC",          "bool"),
            ("sn_link1",                "SN check URL prefix",   "str"),
            ("sn_link2",                "SN check URL suffix",   "str"),
            ("link_sfc",                "clipThroughStation",    "str"),
            ("token_link",              "Token URL",             "str"),
            ("emp_link",                "Employee URL prefix",   "str"),
            ("API_TOKEN_URL",           "API_TOKEN_URL (fb)",    "str"),
            ("API_EMPLOYEE_URL_PREFIX", "API_EMPLOYEE_URL_PREFIX","str"),
            ("API_REQUEST_TIMEOUT",     "Request timeout (s)",   "int"),
        ],
        "Folder ảnh": [
            ("OPL_OK_DIR",     "Folder ảnh OK",     "dir"),
            ("OPL_NG_DIR",     "Folder ảnh NG",     "dir"),
            ("link_post_img",  "Folder đẩy ảnh",    "dir"),
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt — Riser cable")
        self.setMinimumSize(720, 560)
        self.setModal(True)
        self._widgets: dict[str, QWidget] = {}
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16); outer.setSpacing(12)

        self.tabs = QTabWidget()
        for tab_name, fields in self.SCHEMA.items():
            self.tabs.addTab(self._build_tab(fields), tab_name)
        outer.addWidget(self.tabs, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _build_tab(self, fields) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.setContentsMargins(12, 14, 12, 12); lay.setSpacing(10)
        for key, label, typ in fields:
            widget = self._make_widget(typ)
            self._widgets[key] = widget
            lay.addRow(QLabel(label), widget)
        return w

    @staticmethod
    def _make_widget(typ: str) -> QWidget:
        if typ == "int":
            w = QSpinBox(); w.setRange(-1_000_000, 1_000_000); return w
        if typ == "float":
            w = QDoubleSpinBox(); w.setDecimals(2); w.setRange(0.0, 1e6); return w
        if typ == "bool":
            return QCheckBox()
        if typ == "dir":
            return _DirField()
        return QLineEdit()

    def _load_values(self):
        for fields in self.SCHEMA.values():
            for key, _, typ in fields:
                v = getattr(config, key, None)
                if v is None:
                    continue
                w = self._widgets[key]
                if typ == "int":
                    try: w.setValue(int(v))
                    except Exception: pass
                elif typ == "float":
                    try: w.setValue(float(v))
                    except Exception: pass
                elif typ == "bool":
                    w.setChecked(bool(v))
                else:  # str, dir
                    w.setText(str(v))

    def _collect(self) -> dict:
        data = {}
        for fields in self.SCHEMA.values():
            for key, _, typ in fields:
                w = self._widgets[key]
                if typ == "int":
                    data[key] = int(w.value())
                elif typ == "float":
                    data[key] = float(w.value())
                elif typ == "bool":
                    data[key] = bool(w.isChecked())
                else:
                    data[key] = w.text()
        return data

    def _on_save(self):
        try:
            save_settings(self._collect())
            QMessageBox.information(
                self, "Đã lưu",
                "Cài đặt đã lưu vào settings.json.\n\n"
                "Một số mục (PLC IP, COM port, Poll Hz…) sẽ chỉ có "
                "hiệu lực sau khi khởi động lại app.",
            )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không lưu được:\n{exc}")
