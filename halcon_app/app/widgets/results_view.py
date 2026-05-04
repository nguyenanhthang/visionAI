"""Hiển thị kết quả: bảng metrics + log console."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical, self)

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.console = QPlainTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Log...")

        splitter.addWidget(self.table)
        splitter.addWidget(self.console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        rows = list(_flatten(metrics))
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(_fmt(v)))

    def append_log(self, lines: list[str] | str) -> None:
        if isinstance(lines, str):
            self.console.appendPlainText(lines)
            return
        for line in lines:
            self.console.appendPlainText(line)

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.console.clear()


def _flatten(d: dict[str, Any], prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        elif isinstance(v, list):
            yield (key + ".count", len(v))
            for i, item in enumerate(v[:50]):  # cap để tránh bảng quá dài
                if isinstance(item, dict):
                    for kk, vv in item.items():
                        yield (f"{key}[{i}].{kk}", vv)
                else:
                    yield (f"{key}[{i}]", item)
        else:
            yield (key, v)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
