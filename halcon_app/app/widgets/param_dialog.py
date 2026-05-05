"""Generic param dialog dựng từ schema của ToolSpec."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from app.operators.pipeline import Param, ToolSpec


class ParamDialog(QDialog):
    def __init__(self, spec: ToolSpec, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{spec.icon}  {spec.display}")
        self.setMinimumWidth(360)
        self.spec = spec
        self._widgets: dict[str, tuple[Param, Any]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        if not spec.params:
            outer.addWidget(QLabel(f"Tool '{spec.display}' không có tham số."))
        else:
            form = QFormLayout()
            form.setSpacing(8)
            for p in spec.params:
                w = self._build(p, params.get(p.name, p.default))
                form.addRow(p.label, w)
                self._widgets[p.name] = (p, w)
            outer.addLayout(form)

        if spec.needs:
            note = QLabel(
                "ℹ Tool này cần: " + ", ".join(spec.needs)
                + "  (cấu hình ở sidebar trái trước khi Run)."
            )
            note.setProperty("muted", True)
            note.setWordWrap(True)
            outer.addWidget(note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _build(self, p: Param, value: Any):
        if p.kind == "int":
            w = QSpinBox()
            if p.rng:
                w.setRange(int(p.rng[0]), int(p.rng[1]))
            if p.step:
                w.setSingleStep(int(p.step))
            try:
                w.setValue(int(value))
            except (TypeError, ValueError):
                w.setValue(int(p.default))
            return w
        if p.kind == "float":
            w = QDoubleSpinBox()
            if p.rng:
                w.setRange(float(p.rng[0]), float(p.rng[1]))
            if p.step:
                w.setSingleStep(float(p.step))
            w.setDecimals(4)
            try:
                w.setValue(float(value))
            except (TypeError, ValueError):
                w.setValue(float(p.default))
            return w
        if p.kind == "choice":
            w = QComboBox()
            choices = p.choices or []
            w.addItems(choices)
            if value in choices:
                w.setCurrentText(str(value))
            return w
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        # fallback string
        w = QLineEdit(str(value))
        return w

    def values(self) -> dict:
        out: dict = {}
        for name, (p, w) in self._widgets.items():
            if isinstance(w, QSpinBox):
                out[name] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                out[name] = float(w.value())
            elif isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, QLineEdit):
                out[name] = w.text()
        return out
