"""Pipeline panel — danh sách tool kéo-thả, có thumbnail preview cho từng node.

Giống Cognex VisionPro ToolBlock:
- Add Tool từ palette (combobox)
- Drag-drop để sắp xếp
- ☑ enable / ✏ edit / ✕ delete cho mỗi node
- Click node → emit select_node(idx) cho main window hiển thị output node đó
- Run All chạy tuần tự, mỗi node được fill thumbnail
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.operators.pipeline import TOOLS, PipelineNode

from .image_canvas import numpy_to_qpixmap


class PipelineItem(QWidget):
    edit_requested = Signal()
    delete_requested = Signal()
    enable_toggled = Signal(bool)

    def __init__(self, node: PipelineNode, idx: int, parent=None):
        super().__init__(parent)
        self.node = node
        self.idx = idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Drag handle hint
        handle = QLabel("⋮⋮")
        handle.setStyleSheet("color:#454c66; font-size:14pt;")
        handle.setFixedWidth(14)
        layout.addWidget(handle)

        # Thumbnail
        self.thumb = QLabel()
        self.thumb.setFixedSize(60, 60)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(self._thumb_style_idle())
        layout.addWidget(self.thumb)

        # Title + subtitle
        text_lay = QVBoxLayout()
        text_lay.setSpacing(2)
        spec = TOOLS.get(node.tool_id)
        icon = spec.icon if spec else "•"
        self.title = QLabel(f"<b>{idx+1}.</b> {icon}  {node.label}")
        self.title.setStyleSheet("font-size:10pt;")
        self.subtitle = QLabel(self._summary())
        self.subtitle.setStyleSheet("color:#9aa3bd; font-size:9pt;")
        self.subtitle.setWordWrap(True)
        text_lay.addWidget(self.title)
        text_lay.addWidget(self.subtitle)
        layout.addLayout(text_lay, 1)

        # Controls
        ctrl_lay = QVBoxLayout()
        ctrl_lay.setSpacing(4)
        self.enable_chk = QCheckBox("On")
        self.enable_chk.setChecked(node.enabled)
        self.enable_chk.toggled.connect(self.enable_toggled.emit)
        ctrl_lay.addWidget(self.enable_chk)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        self.edit_btn = QPushButton("✏")
        self.edit_btn.setFixedSize(26, 26)
        self.edit_btn.setProperty("secondary", True)
        self.edit_btn.setToolTip("Edit params")
        self.edit_btn.clicked.connect(self.edit_requested.emit)
        self.del_btn = QPushButton("✕")
        self.del_btn.setFixedSize(26, 26)
        self.del_btn.setProperty("secondary", True)
        self.del_btn.setToolTip("Remove")
        self.del_btn.clicked.connect(self.delete_requested.emit)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.del_btn)
        ctrl_lay.addLayout(btn_row)
        layout.addLayout(ctrl_lay)

        self.refresh()

    @staticmethod
    def _thumb_style_idle() -> str:
        return ("background:#0e1117; border:1px solid #2a3140;"
                " border-radius:6px; color:#454c66;")

    @staticmethod
    def _thumb_style_error() -> str:
        return ("background:#3a1f1f; border:1px solid #ff6b6b;"
                " border-radius:6px; color:#ff6b6b; font-size:18pt;")

    def _summary(self) -> str:
        if self.node.error:
            return f"⚠ {self.node.error}"
        if self.node.last_metrics is not None:
            # tóm tắt key chính
            for k in ("count", "defect_count", "edge_count", "edge_pixels",
                      "mask_pixels", "mean", "fg_pixels"):
                if k in self.node.last_metrics:
                    return f"{k} = {self.node.last_metrics[k]}"
        # fallback: hiển thị params
        if self.node.params:
            kvs = list(self.node.params.items())[:3]
            return ", ".join(f"{k}={v}" for k, v in kvs)
        return "—"

    def refresh(self):
        self.subtitle.setText(self._summary())
        self.enable_chk.setChecked(self.node.enabled)
        if self.node.last_thumbnail is not None:
            self.thumb.setPixmap(
                numpy_to_qpixmap(self.node.last_thumbnail).scaled(
                    60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self.thumb.setStyleSheet(self._thumb_style_idle())
        elif self.node.error:
            self.thumb.clear()
            self.thumb.setText("⚠")
            self.thumb.setStyleSheet(self._thumb_style_error())
        else:
            self.thumb.clear()
            self.thumb.setText("∅")
            self.thumb.setStyleSheet(self._thumb_style_idle())


class PipelinePanel(QWidget):
    add_node_requested = Signal(str)             # tool_id
    edit_node_requested = Signal(int)            # idx
    delete_node_requested = Signal(int)          # idx
    enable_node_changed = Signal(int, bool)
    select_node = Signal(int)
    run_requested = Signal()
    clear_requested = Signal()
    reorder_changed = Signal(int, int)           # src, dst

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PipelinePanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header
        title_row = QHBoxLayout()
        title = QLabel("Pipeline")
        title.setProperty("heading", True)
        self.count_badge = QLabel("0 nodes")
        self.count_badge.setProperty("badge", True)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.count_badge)
        layout.addLayout(title_row)

        sub = QLabel("Drag để sắp xếp · click để xem preview · ✏ chỉnh params")
        sub.setProperty("muted", True)
        sub.setWordWrap(True)
        layout.addWidget(sub)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setMovement(QListWidget.Snap)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.itemSelectionChanged.connect(self._on_select)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget, 1)

        # Add row
        add_row = QHBoxLayout()
        self.add_combo = QComboBox()
        for tool_id, spec in TOOLS.items():
            self.add_combo.addItem(f"{spec.icon}  {spec.display}", tool_id)
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._on_add)
        add_row.addWidget(self.add_combo, 1)
        add_row.addWidget(self.add_btn)
        layout.addLayout(add_row)

        # Action row
        actions = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run All")
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.clear_btn = QPushButton("🗑  Clear")
        self.clear_btn.setProperty("secondary", True)
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        actions.addWidget(self.run_btn, 1)
        actions.addWidget(self.clear_btn)
        layout.addLayout(actions)

        # Status line
        self.status = QLabel("Ready.")
        self.status.setProperty("muted", True)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self._suppress_reorder = False

    # ------------------------------------------------------------------
    def _on_add(self):
        tool_id = self.add_combo.currentData()
        if tool_id:
            self.add_node_requested.emit(tool_id)

    def _on_select(self):
        items = self.list_widget.selectedItems()
        if items:
            idx = self.list_widget.row(items[0])
            self.select_node.emit(idx)

    def _on_rows_moved(self, _parent, src_start, _src_end, _dest_parent, dest_row):
        if self._suppress_reorder:
            return
        src = src_start
        dst = dest_row if dest_row < src_start else dest_row - 1
        if src != dst:
            self.reorder_changed.emit(src, dst)

    # ------------------------------------------------------------------
    def rebuild(self, nodes: list[PipelineNode], select_idx: Optional[int] = None) -> None:
        self._suppress_reorder = True
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, node in enumerate(nodes):
            widget = PipelineItem(node, i)
            widget.edit_requested.connect(
                lambda i=i: self.edit_node_requested.emit(i)
            )
            widget.delete_requested.connect(
                lambda i=i: self.delete_node_requested.emit(i)
            )
            widget.enable_toggled.connect(
                lambda v, i=i: self.enable_node_changed.emit(i, v)
            )
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
        self.list_widget.blockSignals(False)
        self._suppress_reorder = False

        n = len(nodes)
        self.count_badge.setText(f"{n} node{'s' if n != 1 else ''}")
        if select_idx is not None and 0 <= select_idx < n:
            self.list_widget.setCurrentRow(select_idx)

    def refresh_node(self, idx: int, node: PipelineNode) -> None:
        item = self.list_widget.item(idx)
        if item is None:
            return
        widget = self.list_widget.itemWidget(item)
        if isinstance(widget, PipelineItem):
            widget.node = node
            widget.refresh()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def select(self, idx: int) -> None:
        self.list_widget.setCurrentRow(idx)
