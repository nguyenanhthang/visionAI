"""Resources panel — gom các tài nguyên ngoài cho pipeline.

Sections (collapsible):
  • 📷 Acquisition (camera connect / live / snapshot)
  • 🩹 Mask (gen từ gray / HSV / ROI; show / invert / save / load)
  • 🧩 Template (cho Pattern Match: load / pick ROI / save / clear + preview)
  • 📋 Reference (cho Image Diff: load / clear)
  • 📐 Caliper segment + 🎨 Color ROI — chỉ status, vẽ trên canvas

Mọi signal được forward ra ngoài để main_window kết nối.
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


def _card(layout) -> QFrame:
    f = QFrame(); f.setProperty("card", True)
    layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
    f.setLayout(layout)
    return f


def _btn(text: str, secondary: bool = False, checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    if secondary:
        b.setProperty("secondary", True)
    b.setCheckable(checkable)
    return b


# =============================================================================
# Mask content
# =============================================================================

class _MaskBlock(QWidget):
    gen_gray = Signal(int, int)
    gen_hsv = Signal(int, int, int, int, int, int)
    pick_roi_toggled = Signal(bool)
    invert = Signal()
    clear = Signal()
    show_toggled = Signal(bool)
    save = Signal()
    load = Signal()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        self.status_label = QLabel("Mask: (none)")
        self.status_label.setProperty("muted", True)
        outer.addWidget(self.status_label)

        self.show_btn = _btn("👁  Hiện mask", secondary=True, checkable=True)
        self.show_btn.setChecked(True)
        self.show_btn.toggled.connect(self.show_toggled.emit)
        outer.addWidget(self.show_btn)

        # Gray range
        gray_outer = QVBoxLayout()
        gray_outer.addWidget(SectionLabel("From gray range"))
        gf = QFormLayout()
        self.gray_min = QSpinBox(); self.gray_min.setRange(0, 255); self.gray_min.setValue(0)
        self.gray_max = QSpinBox(); self.gray_max.setRange(0, 255); self.gray_max.setValue(128)
        gf.addRow("Min", self.gray_min)
        gf.addRow("Max", self.gray_max)
        gray_outer.addLayout(gf)
        gb = _btn("⬛  Generate")
        gb.clicked.connect(lambda: self.gen_gray.emit(self.gray_min.value(), self.gray_max.value()))
        gray_outer.addWidget(gb)
        outer.addWidget(_card(gray_outer))

        # HSV range
        hsv_outer = QVBoxLayout()
        hsv_outer.addWidget(SectionLabel("From HSV range"))
        hf = QFormLayout()
        self.h_min = QSpinBox(); self.h_min.setRange(0, 179); self.h_min.setValue(0)
        self.h_max = QSpinBox(); self.h_max.setRange(0, 179); self.h_max.setValue(179)
        self.s_min = QSpinBox(); self.s_min.setRange(0, 255); self.s_min.setValue(0)
        self.s_max = QSpinBox(); self.s_max.setRange(0, 255); self.s_max.setValue(255)
        self.v_min = QSpinBox(); self.v_min.setRange(0, 255); self.v_min.setValue(0)
        self.v_max = QSpinBox(); self.v_max.setRange(0, 255); self.v_max.setValue(255)
        hf.addRow("H", self._pair(self.h_min, self.h_max))
        hf.addRow("S", self._pair(self.s_min, self.s_max))
        hf.addRow("V", self._pair(self.v_min, self.v_max))
        hsv_outer.addLayout(hf)
        hb = _btn("🎨  Generate")
        hb.clicked.connect(lambda: self.gen_hsv.emit(
            self.h_min.value(), self.h_max.value(),
            self.s_min.value(), self.s_max.value(),
            self.v_min.value(), self.v_max.value(),
        ))
        hsv_outer.addWidget(hb)
        outer.addWidget(_card(hsv_outer))

        # ROI + actions
        self.roi_btn = _btn("✎  Vẽ ROI làm mask", secondary=True, checkable=True)
        self.roi_btn.toggled.connect(self.pick_roi_toggled.emit)
        outer.addWidget(self.roi_btn)

        actions = QHBoxLayout()
        ib = _btn("↔  Invert", secondary=True); ib.clicked.connect(self.invert.emit)
        cb = _btn("✕  Clear", secondary=True); cb.clicked.connect(self.clear.emit)
        actions.addWidget(ib); actions.addWidget(cb)
        outer.addLayout(actions)

        io = QHBoxLayout()
        lb = _btn("📂  Load", secondary=True); lb.clicked.connect(self.load.emit)
        sb = _btn("💾  Save", secondary=True); sb.clicked.connect(self.save.emit)
        io.addWidget(lb); io.addWidget(sb)
        outer.addLayout(io)

    @staticmethod
    def _pair(a, b):
        w = QWidget(); lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        lay.addWidget(a); lay.addWidget(b)
        return w

    def set_status(self, text: str):
        self.status_label.setText(f"Mask: {text}")

    def reset_pick(self):
        self.roi_btn.setChecked(False)


# =============================================================================
# Template (for Pattern Match)
# =============================================================================

class _TemplateBlock(QWidget):
    load = Signal()
    save = Signal()
    clear = Signal()
    pick_roi_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        self.label = QLabel("(chưa chọn)")
        self.label.setProperty("muted", True)
        outer.addWidget(self.label)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(110)
        self.preview.setStyleSheet(
            "background-color:#161927; border:1px dashed #363c52; border-radius:6px; color:#9aa3bd;"
        )
        self.preview.setText("(no template)")
        outer.addWidget(self.preview)

        row = QHBoxLayout()
        self.pick_btn = _btn("✎  Pick ROI", secondary=True, checkable=True)
        self.pick_btn.toggled.connect(self.pick_roi_toggled.emit)
        ld = _btn("📂  File", secondary=True); ld.clicked.connect(self.load.emit)
        sv = _btn("💾", secondary=True); sv.setToolTip("Save"); sv.clicked.connect(self.save.emit)
        cl = _btn("✕", secondary=True); cl.setToolTip("Clear"); cl.clicked.connect(self.clear.emit)
        row.addWidget(self.pick_btn, 2); row.addWidget(ld, 1); row.addWidget(sv, 0); row.addWidget(cl, 0)
        outer.addLayout(row)

    def set_name(self, name: Optional[str]):
        self.label.setText(name or "(chưa chọn)")

    def set_preview(self, img: Optional[np.ndarray]):
        if img is None:
            self.preview.clear(); self.preview.setText("(no template)")
            return
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = disp.shape
        qimg = QImage(disp.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.width() or 220, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def reset_pick(self):
        self.pick_btn.setChecked(False)


# =============================================================================
# Reference (for Image Diff)
# =============================================================================

class _ReferenceBlock(QWidget):
    load = Signal()
    clear = Signal()

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)
        self.label = QLabel("Reference: (chưa chọn)")
        self.label.setProperty("muted", True)
        outer.addWidget(self.label)
        row = QHBoxLayout()
        ld = _btn("📂  Load reference", secondary=True); ld.clicked.connect(self.load.emit)
        cl = _btn("✕", secondary=True); cl.clicked.connect(self.clear.emit)
        row.addWidget(ld, 1); row.addWidget(cl, 0)
        outer.addLayout(row)

    def set_name(self, name: Optional[str]):
        self.label.setText(f"Reference: {name}" if name else "Reference: (chưa chọn)")


# =============================================================================
# Segment / Color ROI status
# =============================================================================

class _CanvasInputsBlock(QWidget):
    measure_mode_toggled = Signal(bool)
    color_pick_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        outer.addWidget(SectionLabel("Caliper segment"))
        self.draw_seg_btn = _btn("✎  Vẽ segment", secondary=True, checkable=True)
        self.draw_seg_btn.toggled.connect(self.measure_mode_toggled.emit)
        outer.addWidget(self.draw_seg_btn)
        self.seg_label = QLabel("Segment: (chưa vẽ)")
        self.seg_label.setProperty("muted", True)
        outer.addWidget(self.seg_label)

        outer.addWidget(HRule())
        outer.addWidget(SectionLabel("Color ROI"))
        self.color_pick_btn = _btn("✎  Chọn ROI", secondary=True, checkable=True)
        self.color_pick_btn.toggled.connect(self.color_pick_toggled.emit)
        outer.addWidget(self.color_pick_btn)
        self.color_label = QLabel("ROI: toàn ảnh")
        self.color_label.setProperty("muted", True)
        outer.addWidget(self.color_label)

    def set_segment(self, r1, c1, r2, c2):
        self.seg_label.setText(f"Segment: ({c1},{r1}) → ({c2},{r2})")
        self.draw_seg_btn.setChecked(False)

    def set_color_roi(self, x, y, w, h):
        self.color_label.setText(f"ROI: ({x},{y}) {w}×{h}")
        self.color_pick_btn.setChecked(False)


# =============================================================================
# ResourcesPanel
# =============================================================================

class ResourcesPanel(QScrollArea):
    # Acquisition
    acq_connect = Signal(str, str)
    acq_disconnect = Signal()
    acq_live = Signal(bool)
    acq_snapshot = Signal()
    acq_fps = Signal(int)

    # Mask
    mask_gen_gray = Signal(int, int)
    mask_gen_hsv = Signal(int, int, int, int, int, int)
    mask_pick_roi_toggled = Signal(bool)
    mask_invert = Signal()
    mask_clear = Signal()
    mask_show_toggled = Signal(bool)
    mask_save = Signal()
    mask_load = Signal()

    # Template
    template_load = Signal()
    template_save = Signal()
    template_clear = Signal()
    template_pick_roi_toggled = Signal(bool)

    # Reference
    reference_load = Signal()
    reference_clear = Signal()

    # Canvas inputs
    measure_mode_toggled = Signal(bool)
    color_pick_roi_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

        host = QWidget(); self.setWidget(host)
        v = QVBoxLayout(host)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)

        # Acquisition
        self.acq_panel = AcquisitionPanel()
        self.acq_section = CollapsibleSection("Acquisition", icon="📷", expanded=True)
        self.acq_section.set_content(self.acq_panel)
        v.addWidget(self.acq_section)

        # Mask
        self.mask_block = _MaskBlock()
        self.mask_section = CollapsibleSection("Mask / ROI", icon="🩹", expanded=False)
        self.mask_section.set_content(self.mask_block)
        v.addWidget(self.mask_section)

        # Template
        self.template_block = _TemplateBlock()
        self.template_section = CollapsibleSection("Template (for Match)", icon="🧩", expanded=False)
        self.template_section.set_content(self.template_block)
        v.addWidget(self.template_section)

        # Reference
        self.reference_block = _ReferenceBlock()
        self.reference_section = CollapsibleSection("Reference (for Diff)", icon="📋", expanded=False)
        self.reference_section.set_content(self.reference_block)
        v.addWidget(self.reference_section)

        # Canvas inputs (segment, color ROI)
        self.canvas_block = _CanvasInputsBlock()
        self.inputs_section = CollapsibleSection("Canvas inputs", icon="🖱", expanded=False)
        self.inputs_section.set_content(self.canvas_block)
        v.addWidget(self.inputs_section)

        v.addStretch()

        self._sections = [
            self.acq_section, self.mask_section, self.template_section,
            self.reference_section, self.inputs_section,
        ]
        self._wire()

    def _wire(self):
        # Acquisition
        a = self.acq_panel
        a.connect_requested.connect(self.acq_connect.emit)
        a.disconnect_requested.connect(self.acq_disconnect.emit)
        a.live_toggled.connect(self.acq_live.emit)
        a.snapshot_requested.connect(self.acq_snapshot.emit)
        a.fps_changed.connect(self.acq_fps.emit)

        # Mask
        m = self.mask_block
        m.gen_gray.connect(self.mask_gen_gray.emit)
        m.gen_hsv.connect(self.mask_gen_hsv.emit)
        m.pick_roi_toggled.connect(self.mask_pick_roi_toggled.emit)
        m.invert.connect(self.mask_invert.emit)
        m.clear.connect(self.mask_clear.emit)
        m.show_toggled.connect(self.mask_show_toggled.emit)
        m.save.connect(self.mask_save.emit)
        m.load.connect(self.mask_load.emit)

        # Template
        t = self.template_block
        t.load.connect(self.template_load.emit)
        t.save.connect(self.template_save.emit)
        t.clear.connect(self.template_clear.emit)
        t.pick_roi_toggled.connect(self.template_pick_roi_toggled.emit)

        # Reference
        r = self.reference_block
        r.load.connect(self.reference_load.emit)
        r.clear.connect(self.reference_clear.emit)

        # Canvas inputs
        c = self.canvas_block
        c.measure_mode_toggled.connect(self.measure_mode_toggled.emit)
        c.color_pick_toggled.connect(self.color_pick_roi_toggled.emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def expand_all(self):
        for s in self._sections: s.set_expanded(True)

    def collapse_all(self):
        for s in self._sections: s.set_expanded(False)

    # acquisition
    def set_acq_connected(self, c, info=""): self.acq_panel.set_connected(c, info)
    def set_acq_live(self, on): self.acq_panel.set_live(on)
    @property
    def acq_fps_value(self): return self.acq_panel.fps_spin.value()

    # mask
    def set_mask_status(self, text):
        self.mask_block.set_status(text); self.mask_section.set_expanded(True)

    def reset_mask_pick(self): self.mask_block.reset_pick()

    # template
    def set_template_name(self, name): self.template_block.set_name(name)
    def set_template_preview(self, img): self.template_block.set_preview(img)
    def reset_template_pick(self): self.template_block.reset_pick()
    def focus_template(self): self.template_section.set_expanded(True)

    # reference
    def set_reference_name(self, name):
        self.reference_block.set_name(name)
        self.reference_section.set_expanded(True)

    # canvas inputs
    def set_segment(self, r1, c1, r2, c2):
        self.canvas_block.set_segment(r1, c1, r2, c2)
        self.inputs_section.set_expanded(True)

    def set_color_roi(self, x, y, w, h):
        self.canvas_block.set_color_roi(x, y, w, h)
        self.inputs_section.set_expanded(True)
