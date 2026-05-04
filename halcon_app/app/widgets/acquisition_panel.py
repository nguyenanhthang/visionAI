"""Acquisition controls — chọn grabber, connect, live, snapshot."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.operators import GrabberInfo, list_grabbers


class AcquisitionPanel(QWidget):
    refresh_requested = Signal()
    connect_requested = Signal(str, str)  # interface, device
    disconnect_requested = Signal()
    live_toggled = Signal(bool)
    snapshot_requested = Signal()
    fps_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox("Image Acquisition")
        form = QFormLayout(group)

        self.interface_combo = QComboBox()
        self.device_combo = QComboBox()
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.setSuffix(" fps")

        form.addRow("Interface", self.interface_combo)
        form.addRow("Device", self.device_combo)
        form.addRow("Live FPS", self.fps_spin)
        layout.addWidget(group)

        # Buttons
        btns_row1 = QHBoxLayout()
        self.refresh_btn = QPushButton("⟳  Refresh")
        self.refresh_btn.setProperty("secondary", True)
        self.connect_btn = QPushButton("🔌  Connect")
        btns_row1.addWidget(self.refresh_btn)
        btns_row1.addWidget(self.connect_btn)
        layout.addLayout(btns_row1)

        btns_row2 = QHBoxLayout()
        self.live_btn = QPushButton("▶  Live")
        self.live_btn.setCheckable(True)
        self.live_btn.setEnabled(False)
        self.snap_btn = QPushButton("📸  Snapshot")
        self.snap_btn.setEnabled(False)
        self.snap_btn.setProperty("secondary", True)
        btns_row2.addWidget(self.live_btn)
        btns_row2.addWidget(self.snap_btn)
        layout.addLayout(btns_row2)

        self.disconnect_btn = QPushButton("⏏  Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setProperty("secondary", True)
        layout.addWidget(self.disconnect_btn)

        self.status_label = QLabel("Trạng thái: chưa kết nối")
        self.status_label.setProperty("muted", True)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Wire
        self.refresh_btn.clicked.connect(self._refresh)
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn.clicked.connect(self.disconnect_requested.emit)
        self.live_btn.toggled.connect(self.live_toggled.emit)
        self.snap_btn.clicked.connect(self.snapshot_requested.emit)
        self.fps_spin.valueChanged.connect(self.fps_changed.emit)
        self.interface_combo.currentTextChanged.connect(self._on_iface_changed)

        self._grabbers: list[GrabberInfo] = []
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        self._grabbers = list_grabbers()
        self.interface_combo.blockSignals(True)
        self.interface_combo.clear()
        for g in self._grabbers:
            self.interface_combo.addItem(g.name)
        self.interface_combo.blockSignals(False)
        self._on_iface_changed(self.interface_combo.currentText())
        self.refresh_requested.emit()

    def _on_iface_changed(self, iface: str):
        self.device_combo.clear()
        for g in self._grabbers:
            if g.name == iface:
                self.device_combo.addItems(g.devices)
                break

    def _on_connect(self):
        iface = self.interface_combo.currentText()
        device = self.device_combo.currentText()
        if iface and device:
            self.connect_requested.emit(iface, device)

    # ------------------------------------------------------------------
    def set_connected(self, connected: bool, info: str = ""):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.live_btn.setEnabled(connected)
        self.snap_btn.setEnabled(connected)
        self.interface_combo.setEnabled(not connected)
        self.device_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        if not connected:
            self.live_btn.setChecked(False)
        self.status_label.setText(
            f"Trạng thái: {'connected — ' + info if connected else 'chưa kết nối'}"
        )

    def set_live(self, on: bool):
        if self.live_btn.isChecked() != on:
            self.live_btn.blockSignals(True)
            self.live_btn.setChecked(on)
            self.live_btn.blockSignals(False)
        self.live_btn.setText("⏸  Stop Live" if on else "▶  Live")
