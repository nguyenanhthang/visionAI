"""Sidebar gồm các CollapsibleSection cho từng nhóm tool, kiểu Cognex VisionPro.

Sections:
  • Acquisition         (camera connect / live / snapshot)
  • Pre-process         (gauss, median, mean, sharpen)
  • Locate              (Blob, Edges, Pattern Match + ROI sampling)
  • Measure             (Caliper / Measure 1D, Histogram)
  • Identify            (ID Read - QR / Barcode)
  • Inspect             (Color stats trong ROI)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .acquisition_panel import AcquisitionPanel
from .collapsible import CollapsibleSection, HRule, SectionLabel


# =============================================================================
# Tiny helpers
# =============================================================================

def _card(layout: QVBoxLayout | QFormLayout) -> QFrame:
    f = QFrame()
    f.setProperty("card", True)
    if isinstance(layout, QFormLayout):
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
    else:
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
    f.setLayout(layout)
    return f


def _primary(text: str) -> QPushButton:
    b = QPushButton(text)
    return b


def _secondary(text: str, checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setProperty("secondary", True)
    if checkable:
        b.setCheckable(True)
    return b


# =============================================================================
# Section content widgets
# =============================================================================

class _PreProcessContent(QWidget):
    run_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        form = QFormLayout()
        self.method = QComboBox()
        self.method.addItems(["gauss", "median", "mean", "sharpen"])
        self.ksize = QSpinBox()
        self.ksize.setRange(3, 51)
        self.ksize.setSingleStep(2)
        self.ksize.setValue(5)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.5)
        form.addRow("Method", self.method)
        form.addRow("Kernel size", self.ksize)
        form.addRow("Sigma", self.sigma)
        outer.addWidget(_card(form))

        run = _primary("▶  Apply Filter")
        run.clicked.connect(
            lambda: self.run_requested.emit({
                "method": self.method.currentText(),
                "ksize": self.ksize.value(),
                "sigma": self.sigma.value(),
            })
        )
        outer.addWidget(run)


class _BlobContent(QWidget):
    run_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        form = QFormLayout()
        self.min_gray = QSpinBox(); self.min_gray.setRange(0, 255); self.min_gray.setValue(0)
        self.max_gray = QSpinBox(); self.max_gray.setRange(0, 255); self.max_gray.setValue(128)
        self.min_area = QSpinBox(); self.min_area.setRange(1, 10_000_000); self.min_area.setValue(100)
        self.max_area = QSpinBox(); self.max_area.setRange(1, 100_000_000); self.max_area.setValue(10_000_000)
        form.addRow("Min Gray", self.min_gray)
        form.addRow("Max Gray", self.max_gray)
        form.addRow("Min Area", self.min_area)
        form.addRow("Max Area", self.max_area)
        outer.addWidget(_card(form))
        run = _primary("▶  Run Blob")
        run.clicked.connect(lambda: self.run_requested.emit({
            "min_gray": self.min_gray.value(), "max_gray": self.max_gray.value(),
            "min_area": self.min_area.value(), "max_area": self.max_area.value(),
        }))
        outer.addWidget(run)


class _EdgesContent(QWidget):
    run_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        form = QFormLayout()
        self.method = QComboBox(); self.method.addItems(["canny", "sobel", "deriche2", "lanser2"])
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.1, 10.0); self.alpha.setSingleStep(0.1); self.alpha.setValue(1.0)
        self.low = QSpinBox(); self.low.setRange(0, 255); self.low.setValue(40)
        self.high = QSpinBox(); self.high.setRange(0, 255); self.high.setValue(120)
        form.addRow("Method", self.method)
        form.addRow("Alpha", self.alpha)
        form.addRow("Low", self.low)
        form.addRow("High", self.high)
        outer.addWidget(_card(form))
        run = _primary("▶  Run Edges")
        run.clicked.connect(lambda: self.run_requested.emit({
            "method": self.method.currentText(), "alpha": self.alpha.value(),
            "low": self.low.value(), "high": self.high.value(),
        }))
        outer.addWidget(run)


class _ShapeMatchContent(QWidget):
    run_requested = Signal(dict)
    template_load_requested = Signal()
    template_save_requested = Signal()
    template_clear_requested = Signal()
    pick_roi_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Params
        form = QFormLayout()
        self.min_score = QDoubleSpinBox(); self.min_score.setRange(0.1, 1.0); self.min_score.setSingleStep(0.05); self.min_score.setValue(0.6)
        self.num_matches = QSpinBox(); self.num_matches.setRange(1, 100); self.num_matches.setValue(5)
        self.angle_start = QDoubleSpinBox(); self.angle_start.setRange(-3.14, 3.14); self.angle_start.setSingleStep(0.05); self.angle_start.setValue(-0.39)
        self.angle_extent = QDoubleSpinBox(); self.angle_extent.setRange(0.0, 6.28); self.angle_extent.setSingleStep(0.05); self.angle_extent.setValue(0.78)
        form.addRow("Min Score", self.min_score)
        form.addRow("Num Matches", self.num_matches)
        form.addRow("Angle Start", self.angle_start)
        form.addRow("Angle Extent", self.angle_extent)
        outer.addWidget(_card(form))

        # Template area
        tpl_lay = QVBoxLayout()
        tpl_lay.addWidget(SectionLabel("Template"))
        self.template_label = QLabel("(chưa chọn)")
        self.template_label.setProperty("muted", True)
        tpl_lay.addWidget(self.template_label)
        self.template_preview = QLabel()
        self.template_preview.setAlignment(Qt.AlignCenter)
        self.template_preview.setMinimumHeight(110)
        self.template_preview.setStyleSheet(
            "background-color: #161927; border: 1px dashed #363c52; border-radius: 6px; color:#9aa3bd;"
        )
        self.template_preview.setText("(no template)")
        tpl_lay.addWidget(self.template_preview)

        btns = QHBoxLayout()
        self.pick_btn = _secondary("✎  Pick ROI", checkable=True)
        self.pick_btn.toggled.connect(self.pick_roi_toggled.emit)
        load_btn = _secondary("📂  File")
        load_btn.clicked.connect(self.template_load_requested.emit)
        save_btn = _secondary("💾")
        save_btn.setToolTip("Lưu template")
        save_btn.clicked.connect(self.template_save_requested.emit)
        clear_btn = _secondary("✕")
        clear_btn.setToolTip("Xoá template")
        clear_btn.clicked.connect(self.template_clear_requested.emit)
        btns.addWidget(self.pick_btn, 2)
        btns.addWidget(load_btn, 1)
        btns.addWidget(save_btn, 0)
        btns.addWidget(clear_btn, 0)
        tpl_lay.addLayout(btns)
        outer.addWidget(_card(tpl_lay))

        run = _primary("▶  Run Match")
        run.clicked.connect(lambda: self.run_requested.emit({
            "min_score": self.min_score.value(), "num_matches": self.num_matches.value(),
            "angle_start": self.angle_start.value(), "angle_extent": self.angle_extent.value(),
        }))
        outer.addWidget(run)

    def set_template_name(self, name: Optional[str]):
        self.template_label.setText(name or "(chưa chọn)")

    def set_template_preview(self, img: Optional[np.ndarray]):
        if img is None:
            self.template_preview.clear()
            self.template_preview.setText("(no template)")
            return
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = disp.shape
        qimg = QImage(disp.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.template_preview.width() or 220, 130,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.template_preview.setPixmap(pix)

    def reset_pick_button(self):
        self.pick_btn.setChecked(False)


class _CaliperContent(QWidget):
    """Measure 1D — Cognex gọi là Caliper."""
    measure_mode_toggled = Signal(bool)
    run_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        form = QFormLayout()
        self.sigma = QDoubleSpinBox(); self.sigma.setRange(0.1, 10.0); self.sigma.setSingleStep(0.1); self.sigma.setValue(1.0)
        self.threshold = QSpinBox(); self.threshold.setRange(1, 255); self.threshold.setValue(30)
        form.addRow("Sigma", self.sigma)
        form.addRow("Edge Threshold", self.threshold)
        outer.addWidget(_card(form))

        self.draw_btn = _secondary("✎  Vẽ segment trên ảnh", checkable=True)
        self.draw_btn.toggled.connect(self.measure_mode_toggled.emit)
        outer.addWidget(self.draw_btn)
        self.segment_label = QLabel("Segment: (chưa vẽ)")
        self.segment_label.setProperty("muted", True)
        outer.addWidget(self.segment_label)

        run = _primary("▶  Run Caliper")
        run.clicked.connect(self._on_run)
        outer.addWidget(run)

        self._segment: Optional[tuple[int, int, int, int]] = None

    def set_segment(self, r1, c1, r2, c2):
        self._segment = (r1, c1, r2, c2)
        self.segment_label.setText(f"Segment: ({c1},{r1}) → ({c2},{r2})")
        self.draw_btn.setChecked(False)

    def _on_run(self):
        if self._segment is None:
            return
        r1, c1, r2, c2 = self._segment
        self.run_requested.emit({
            "row1": r1, "col1": c1, "row2": r2, "col2": c2,
            "sigma": self.sigma.value(), "threshold": self.threshold.value(),
        })


class _HistogramContent(QWidget):
    run_requested = Signal()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        info = QLabel("Tính histogram + thống kê (mean / median / std / min / max).")
        info.setProperty("muted", True); info.setWordWrap(True)
        outer.addWidget(info)
        run = _primary("▶  Compute Histogram")
        run.clicked.connect(self.run_requested.emit)
        outer.addWidget(run)


class _IDReadContent(QWidget):
    run_requested = Signal()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        info = QLabel("Đọc QR code & barcode 1D (OpenCV detector).\nHALCON: find_data_code_2d.")
        info.setProperty("muted", True); info.setWordWrap(True)
        outer.addWidget(info)
        run = _primary("▶  Decode")
        run.clicked.connect(self.run_requested.emit)
        outer.addWidget(run)


class _ColorContent(QWidget):
    pick_roi_toggled = Signal(bool)
    run_requested = Signal()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        info = QLabel("Mean BGR / RGB / HSV trong ROI. Để trống → toàn ảnh.")
        info.setProperty("muted", True); info.setWordWrap(True)
        outer.addWidget(info)

        self.pick_btn = _secondary("✎  Chọn ROI", checkable=True)
        self.pick_btn.toggled.connect(self.pick_roi_toggled.emit)
        outer.addWidget(self.pick_btn)

        self.roi_label = QLabel("ROI: toàn ảnh")
        self.roi_label.setProperty("muted", True)
        outer.addWidget(self.roi_label)

        run = _primary("▶  Compute Color")
        run.clicked.connect(self.run_requested.emit)
        outer.addWidget(run)

        self.roi: tuple[int, int, int, int] = (0, 0, 0, 0)

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.roi = (x, y, w, h)
        self.roi_label.setText(f"ROI: ({x},{y}) {w}×{h}")
        self.pick_btn.setChecked(False)

    def reset_pick(self):
        self.pick_btn.setChecked(False)


# =============================================================================
# Sidebar
# =============================================================================

class OperatorSidebar(QScrollArea):
    """Sidebar dạng accordion — gộp tất cả tool VisionPro-style."""

    # Acquisition (forwarded từ AcquisitionPanel)
    acq_connect = Signal(str, str)
    acq_disconnect = Signal()
    acq_live = Signal(bool)
    acq_snapshot = Signal()
    acq_fps = Signal(int)

    # Tool runs
    filter_run = Signal(dict)
    threshold_run = Signal(dict)
    edges_run = Signal(dict)
    shape_match_run = Signal(dict)
    shape_template_load = Signal()
    shape_template_save = Signal()
    shape_template_clear = Signal()
    shape_pick_roi_toggled = Signal(bool)
    measure_run = Signal(dict)
    measure_mode_toggled = Signal(bool)
    histogram_run = Signal()
    idread_run = Signal()
    color_pick_roi_toggled = Signal(bool)
    color_run = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

        host = QWidget()
        self.setWidget(host)
        v = QVBoxLayout(host)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Title
        title = QLabel("Tools")
        title.setProperty("heading", True)
        v.addWidget(title)
        sub = QLabel("Chọn nhóm để mở rộng / thu gọn")
        sub.setProperty("muted", True)
        v.addWidget(sub)
        v.addWidget(HRule())

        # --- Acquisition section ---
        self.acq_panel = AcquisitionPanel()
        self.acq_section = CollapsibleSection("Acquisition", icon="📷", expanded=True)
        self.acq_section.set_content(self.acq_panel)
        v.addWidget(self.acq_section)

        # --- Pre-process ---
        self.preproc_content = _PreProcessContent()
        self.preproc_section = CollapsibleSection("Pre-process", icon="🪄", expanded=False)
        self.preproc_section.set_content(self.preproc_content)
        v.addWidget(self.preproc_section)

        # --- Locate (Blob, Edges, Match) ---
        self.blob_content = _BlobContent()
        self.edges_content = _EdgesContent()
        self.shape_content = _ShapeMatchContent()
        self.locate_section = CollapsibleSection("Locate", icon="🎯", expanded=True)
        loc_box = QWidget()
        loc_lay = QVBoxLayout(loc_box)
        loc_lay.setContentsMargins(0, 0, 0, 0); loc_lay.setSpacing(8)
        loc_lay.addWidget(SectionLabel("Blob"))
        loc_lay.addWidget(self.blob_content)
        loc_lay.addWidget(HRule())
        loc_lay.addWidget(SectionLabel("Edges (sub-pixel)"))
        loc_lay.addWidget(self.edges_content)
        loc_lay.addWidget(HRule())
        loc_lay.addWidget(SectionLabel("Pattern Match"))
        loc_lay.addWidget(self.shape_content)
        self.locate_section.set_content(loc_box)
        v.addWidget(self.locate_section)

        # --- Measure ---
        self.caliper_content = _CaliperContent()
        self.histogram_content = _HistogramContent()
        self.measure_section = CollapsibleSection("Measure", icon="📐", expanded=False)
        mes_box = QWidget()
        mes_lay = QVBoxLayout(mes_box); mes_lay.setContentsMargins(0, 0, 0, 0); mes_lay.setSpacing(8)
        mes_lay.addWidget(SectionLabel("Caliper (1D)"))
        mes_lay.addWidget(self.caliper_content)
        mes_lay.addWidget(HRule())
        mes_lay.addWidget(SectionLabel("Histogram"))
        mes_lay.addWidget(self.histogram_content)
        self.measure_section.set_content(mes_box)
        v.addWidget(self.measure_section)

        # --- Identify ---
        self.id_content = _IDReadContent()
        self.id_section = CollapsibleSection("Identify", icon="🔢", expanded=False)
        self.id_section.set_content(self.id_content)
        v.addWidget(self.id_section)

        # --- Inspect ---
        self.color_content = _ColorContent()
        self.inspect_section = CollapsibleSection("Inspect", icon="🎨", expanded=False)
        self.inspect_section.set_content(self.color_content)
        v.addWidget(self.inspect_section)

        v.addStretch()

        self._sections: list[CollapsibleSection] = [
            self.acq_section, self.preproc_section, self.locate_section,
            self.measure_section, self.id_section, self.inspect_section,
        ]

        self._wire()

    # ------------------------------------------------------------------
    def _wire(self):
        # Acquisition
        self.acq_panel.connect_requested.connect(self.acq_connect.emit)
        self.acq_panel.disconnect_requested.connect(self.acq_disconnect.emit)
        self.acq_panel.live_toggled.connect(self.acq_live.emit)
        self.acq_panel.snapshot_requested.connect(self.acq_snapshot.emit)
        self.acq_panel.fps_changed.connect(self.acq_fps.emit)

        # Tools
        self.preproc_content.run_requested.connect(self.filter_run.emit)
        self.blob_content.run_requested.connect(self.threshold_run.emit)
        self.edges_content.run_requested.connect(self.edges_run.emit)
        self.shape_content.run_requested.connect(self.shape_match_run.emit)
        self.shape_content.template_load_requested.connect(self.shape_template_load.emit)
        self.shape_content.template_save_requested.connect(self.shape_template_save.emit)
        self.shape_content.template_clear_requested.connect(self.shape_template_clear.emit)
        self.shape_content.pick_roi_toggled.connect(self.shape_pick_roi_toggled.emit)
        self.caliper_content.run_requested.connect(self.measure_run.emit)
        self.caliper_content.measure_mode_toggled.connect(self.measure_mode_toggled.emit)
        self.histogram_content.run_requested.connect(self.histogram_run.emit)
        self.id_content.run_requested.connect(self.idread_run.emit)
        self.color_content.pick_roi_toggled.connect(self.color_pick_roi_toggled.emit)
        self.color_content.run_requested.connect(self.color_run.emit)

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------
    def collapse_all(self):
        for s in self._sections:
            s.set_expanded(False)

    def expand_all(self):
        for s in self._sections:
            s.set_expanded(True)

    # template
    def set_template_name(self, name: Optional[str]):
        self.shape_content.set_template_name(name)

    def set_template_preview(self, img):
        self.shape_content.set_template_preview(img)

    def reset_pick_button(self):
        self.shape_content.reset_pick_button()

    def focus_match(self):
        self.locate_section.set_expanded(True)

    # measure
    def set_measure_segment(self, r1, c1, r2, c2):
        self.caliper_content.set_segment(r1, c1, r2, c2)
        self.measure_section.set_expanded(True)

    # color roi
    def set_color_roi(self, x, y, w, h):
        self.color_content.set_roi(x, y, w, h)
        self.inspect_section.set_expanded(True)

    def reset_color_pick(self):
        self.color_content.reset_pick()

    # acquisition forwards
    def set_acq_connected(self, connected: bool, info: str = ""):
        self.acq_panel.set_connected(connected, info)

    def set_acq_live(self, on: bool):
        self.acq_panel.set_live(on)

    @property
    def acq_fps_value(self) -> int:
        return self.acq_panel.fps_spin.value()
