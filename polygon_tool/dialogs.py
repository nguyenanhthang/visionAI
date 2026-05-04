from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ShapeEditDialog(QDialog):
    """Dialog (F1) for editing shape properties: class, color, width, description, and points."""

    def __init__(
        self,
        class_id: int,
        class_name: str,
        color: Tuple[int, int, int],
        width: int,
        description: str,
        points: List[Tuple[float, float]],
        class_map: Optional[dict] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sửa thông tin shape (F1)")
        self.setMinimumWidth(480)

        self._color = QColor(*color)
        self._class_map = class_map or {}

        layout = QVBoxLayout(self)

        # ── Class ──────────────────────────────────────────────────────────────
        class_group = QGroupBox("Thông tin class")
        class_form = QHBoxLayout(class_group)

        class_form.addWidget(QLabel("Class ID:"))
        self._class_id_spin = QSpinBox()
        self._class_id_spin.setRange(0, 9999)
        self._class_id_spin.setValue(class_id)
        self._class_id_spin.valueChanged.connect(self._on_class_id_changed)
        class_form.addWidget(self._class_id_spin)

        class_form.addWidget(QLabel("Class name:"))
        self._class_name_edit = QLineEdit(class_name)
        class_form.addWidget(self._class_name_edit)

        layout.addWidget(class_group)

        # ── Appearance ─────────────────────────────────────────────────────────
        appear_group = QGroupBox("Hiển thị")
        appear_form = QHBoxLayout(appear_group)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(32, 24)
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        appear_form.addWidget(QLabel("Màu:"))
        appear_form.addWidget(self._color_btn)

        appear_form.addWidget(QLabel("Độ dày:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 20)
        self._width_spin.setValue(width)
        appear_form.addWidget(self._width_spin)

        layout.addWidget(appear_group)

        # ── Description ────────────────────────────────────────────────────────
        desc_group = QGroupBox("Mô tả")
        desc_layout = QVBoxLayout(desc_group)
        self._desc_edit = QLineEdit(description)
        desc_layout.addWidget(self._desc_edit)
        layout.addWidget(desc_group)

        # ── Points table ───────────────────────────────────────────────────────
        pts_group = QGroupBox("Tọa độ điểm")
        pts_layout = QVBoxLayout(pts_group)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["#", "X", "Y"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        pts_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._del_pt_btn = QPushButton("Xóa điểm")
        self._del_pt_btn.clicked.connect(self._delete_point)
        self._up_btn = QPushButton("▲ Lên")
        self._up_btn.clicked.connect(self._move_up)
        self._down_btn = QPushButton("▼ Xuống")
        self._down_btn.clicked.connect(self._move_down)
        btn_row.addWidget(self._del_pt_btn)
        btn_row.addWidget(self._up_btn)
        btn_row.addWidget(self._down_btn)
        pts_layout.addLayout(btn_row)
        layout.addWidget(pts_group)

        # ── Dialog buttons ─────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate table
        for pt in points:
            self._add_point_row(pt[0], pt[1])

    # ── Private helpers ─────────────────────────────────────────────────────

    def _add_point_row(self, x: float, y: float) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, idx_item)

        x_spin = QSpinBox()
        x_spin.setRange(-99999, 99999)
        x_spin.setValue(int(round(x)))
        self._table.setCellWidget(row, 1, x_spin)

        y_spin = QSpinBox()
        y_spin.setRange(-99999, 99999)
        y_spin.setValue(int(round(y)))
        self._table.setCellWidget(row, 2, y_spin)

    def _renumber_rows(self) -> None:
        for r in range(self._table.rowCount()):
            self._table.item(r, 0).setText(str(r + 1))

    def _update_color_btn(self) -> None:
        self._color_btn.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #888;"
        )

    def _on_class_id_changed(self, value: int) -> None:
        if value in self._class_map:
            self._class_name_edit.setText(self._class_map[value])

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Chọn màu")
        if chosen.isValid():
            self._color = chosen
            self._update_color_btn()

    def _delete_point(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self._renumber_rows()

    def _move_up(self) -> None:
        row = self._table.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self._table.setCurrentCell(row - 1, self._table.currentColumn())

    def _move_down(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= self._table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self._table.setCurrentCell(row + 1, self._table.currentColumn())

    def _swap_rows(self, a: int, b: int) -> None:
        for col in (1, 2):
            wa = self._table.cellWidget(a, col)
            wb = self._table.cellWidget(b, col)
            va = wa.value()
            vb = wb.value()
            wa.setValue(vb)
            wb.setValue(va)

    def _on_accept(self) -> None:
        if self._table.rowCount() < 3:
            QMessageBox.warning(self, "Lỗi", "Polygon cần ít nhất 3 điểm.")
            return
        self.accept()

    # ── Public API ──────────────────────────────────────────────────────────

    def get_result(self) -> dict:
        points = []
        for r in range(self._table.rowCount()):
            x = self._table.cellWidget(r, 1).value()
            y = self._table.cellWidget(r, 2).value()
            points.append((x, y))
        return {
            "class_id": self._class_id_spin.value(),
            "class_name": self._class_name_edit.text().strip(),
            "color": (self._color.red(), self._color.green(), self._color.blue()),
            "width": self._width_spin.value(),
            "description": self._desc_edit.text().strip(),
            "points": points,
        }
