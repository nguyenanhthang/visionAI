"""
Widget xem log với màu sắc theo mức độ
Hỗ trợ auto-scroll, lưu log và xóa log
"""
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QCheckBox, QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QFont


# Màu sắc theo mức độ log
_LEVEL_COLORS = {
    "INFO":    "#cdd6f4",   # Trắng nhạt
    "WARNING": "#f9e2af",   # Vàng
    "ERROR":   "#f38ba8",   # Đỏ
    "METRIC":  "#a6e3a1",   # Xanh lá
    "DEBUG":   "#6c7086",   # Xám
    "SUCCESS": "#94e2d5",   # Teal
}


class LogViewer(QWidget):
    """Widget hiển thị log với màu và auto-scroll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Thanh công cụ phía trên
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._btn_clear = QPushButton("🗑️ Xóa")
        self._btn_clear.setMaximumWidth(80)
        self._btn_clear.setToolTip("Xóa toàn bộ log")
        self._btn_clear.clicked.connect(self.clear_log)

        self._btn_save = QPushButton("💾 Lưu")
        self._btn_save.setMaximumWidth(80)
        self._btn_save.setToolTip("Lưu log ra file")
        self._btn_save.clicked.connect(self.save_log)

        self._chk_autoscroll = QCheckBox("Auto-scroll")
        self._chk_autoscroll.setChecked(True)
        self._chk_autoscroll.setToolTip("Tự động cuộn xuống log mới nhất")

        toolbar.addWidget(self._btn_clear)
        toolbar.addWidget(self._btn_save)
        toolbar.addStretch()
        toolbar.addWidget(self._chk_autoscroll)

        layout.addLayout(toolbar)

        # Text area hiển thị log
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Courier New", 11))
        self._text_edit.setStyleSheet(
            "background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 5px;"
        )
        self._text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._text_edit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_log(self, message: str, level: str = "INFO"):
        """
        Thêm dòng log với màu theo mức độ.
        level: INFO, WARNING, ERROR, METRIC, DEBUG, SUCCESS
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = _LEVEL_COLORS.get(level.upper(), _LEVEL_COLORS["INFO"])

        # Xây dựng HTML
        html = (
            f'<span style="color:#6c7086">[{timestamp}]</span> '
            f'<span style="color:{color}">{_escape_html(message)}</span>'
        )
        self._text_edit.append(html)

        if self._chk_autoscroll.isChecked():
            self._scroll_to_bottom()

    def clear_log(self):
        """Xóa toàn bộ nội dung log."""
        self._text_edit.clear()

    def save_log(self):
        """Lưu log ra file text."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Log", "training_log.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._text_edit.toPlainText())
            except Exception as e:
                self.append_log(f"Lỗi lưu log: {e}", "ERROR")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _scroll_to_bottom(self):
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())


def _escape_html(text: str) -> str:
    """Escape ký tự đặc biệt HTML."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "<br>")
    )
