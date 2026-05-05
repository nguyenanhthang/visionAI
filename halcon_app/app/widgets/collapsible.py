"""CollapsibleSection: header click-to-toggle + content area.

Dùng cho sidebar gộp các nhóm tool. Một section gồm:
  • Header (QPushButton checkable) hiển thị icon ▸/▾ + title + (optional) badge
  • Content widget show/hide khi toggle
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    toggled_open = Signal(bool)

    def __init__(
        self,
        title: str,
        icon: str = "",
        expanded: bool = False,
        badge: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._title = title
        self._icon = icon
        self._badge = badge

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self.header = QPushButton(self._render_header_text(expanded))
        self.header.setProperty("section", True)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.toggled.connect(self._on_toggle)
        outer.addWidget(self.header)

        # Content frame (border-less, indented a bit)
        self._content_holder = QFrame()
        self._content_holder.setFrameShape(QFrame.NoFrame)
        self._content_layout = QVBoxLayout(self._content_holder)
        self._content_layout.setContentsMargins(2, 8, 2, 4)
        self._content_layout.setSpacing(8)
        self._content_holder.setVisible(expanded)
        outer.addWidget(self._content_holder)

    # ------------------------------------------------------------------
    def set_content(self, widget: QWidget) -> None:
        # remove previous
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._content_layout.addWidget(widget)

    def add_content(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_badge(self, text: str) -> None:
        self._badge = text
        self.header.setText(self._render_header_text(self.header.isChecked()))

    # ------------------------------------------------------------------
    def _render_header_text(self, expanded: bool) -> str:
        chevron = "▾" if expanded else "▸"
        icon = f"{self._icon}  " if self._icon else ""
        badge = f"   ·  {self._badge}" if self._badge else ""
        return f"{chevron}   {icon}{self._title}{badge}"

    def _on_toggle(self, checked: bool) -> None:
        self._content_holder.setVisible(checked)
        self.header.setText(self._render_header_text(checked))
        self.toggled_open.emit(checked)


class SectionLabel(QLabel):
    """Label nhỏ kiểu sub-heading dùng trong section."""

    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setProperty("subheading", True)


class HRule(QFrame):
    """Đường kẻ ngang mỏng."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Plain)
        self.setStyleSheet("color: #363c52; background: #363c52; max-height: 1px;")
