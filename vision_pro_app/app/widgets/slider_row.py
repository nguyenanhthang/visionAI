"""Labeled slider used by the adjustment panel."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class SliderRow(QWidget):
    """A label + value readout + horizontal slider, all in one row."""

    valueChanged = Signal(int)

    def __init__(
        self,
        title: str,
        *,
        minimum: int = -100,
        maximum: int = 100,
        default: int = 0,
        unit: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._unit = unit
        self._default = default

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._title = QLabel(title)
        self._title.setObjectName("FieldLabel")
        self._value_label = QLabel(self._format(default))
        self._value_label.setObjectName("FieldValue")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._value_label)
        root.addLayout(header)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(minimum, maximum)
        self._slider.setValue(default)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.valueChanged.connect(self._on_changed)
        root.addWidget(self._slider)

    def _format(self, value: int) -> str:
        if self._unit:
            return f"{value:+d}{self._unit}" if self._unit == "%" else f"{value}{self._unit}"
        return str(value)

    def _on_changed(self, value: int) -> None:
        self._value_label.setText(self._format(value))
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self._slider.value()

    def set_value(self, value: int) -> None:
        self._slider.setValue(value)

    def reset(self) -> None:
        self._slider.setValue(self._default)
