"""Pipeline panel — danh sách tool kéo-thả, có thumbnail preview cho từng node.

Giống Cognex VisionPro ToolBlock:
- Add Tool từ palette (combobox)
- Drag-drop để sắp xếp
- ☑ enable / ✏ edit / ✕ delete cho mỗi node
- Click node → emit select_node(idx) cho main window hiển thị output node đó
- Properties panel ngay bên dưới list: chỉnh thuộc tính → **live preview**
- Run All chạy tuần tự, mỗi node được fill thumbnail
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.operators.pipeline import TOOLS, Param, PipelineNode, ToolSpec

from .image_canvas import numpy_to_qpixmap


class PipelineItem(QWidget):
    delete_requested = Signal()
    enable_toggled = Signal(bool)

    def __init__(self, node: PipelineNode, idx: int, parent=None):
        super().__init__(parent)
        self.node = node
        self.idx = idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Drag handle
        handle = QLabel("⋮⋮")
        handle.setStyleSheet("color:#454c66; font-size:14pt;")
        handle.setFixedWidth(14)
        handle.setToolTip("Kéo để sắp xếp")
        layout.addWidget(handle)

        # Thumbnail (lớn hơn)
        self.thumb = QLabel()
        self.thumb.setFixedSize(72, 72)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(self._thumb_style_idle())
        layout.addWidget(self.thumb)

        # Title + subtitle
        text_lay = QVBoxLayout()
        text_lay.setSpacing(3)
        spec = TOOLS.get(node.tool_id)
        icon = spec.icon if spec else "•"
        title_text = f"<b>{idx+1}.</b> {icon}  {node.label}"
        if spec and spec.chain:
            title_text += "  <span style='color:#36c5d6; font-size:8pt;'>↪ chain</span>"
        self.title = QLabel(title_text)
        self.title.setStyleSheet("font-size:10pt;")
        self.subtitle = QLabel(self._summary())
        self.subtitle.setStyleSheet("color:#9aa3bd; font-size:9pt;")
        self.subtitle.setWordWrap(True)
        text_lay.addWidget(self.title)
        text_lay.addWidget(self.subtitle)
        text_lay.addStretch()
        layout.addLayout(text_lay, 1)

        # Controls — vertical: enable + remove
        ctrl_lay = QVBoxLayout()
        ctrl_lay.setSpacing(6)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)

        self.enable_chk = QCheckBox("On")
        self.enable_chk.setChecked(node.enabled)
        self.enable_chk.setToolTip("Bật/tắt node")
        self.enable_chk.toggled.connect(self.enable_toggled.emit)
        ctrl_lay.addWidget(self.enable_chk, 0, Qt.AlignRight)

        self.del_btn = QPushButton("✕  Remove")
        self.del_btn.setProperty("secondary", True)
        self.del_btn.setToolTip("Xoá node")
        self.del_btn.clicked.connect(self.delete_requested.emit)
        ctrl_lay.addWidget(self.del_btn)

        ctrl_lay.addStretch()
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
                    72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
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


class NodePropertiesPanel(QWidget):
    """Form chỉnh thuộc tính của node được chọn.

    Mỗi widget có signal valueChanged → debounce → emit `params_changed`
    để main window chạy pipeline live-preview.
    """

    params_changed = Signal(int, dict)        # idx, new_params dict (debounced)
    live_run_requested = Signal()             # bấm nút Apply / dừng pipeline
    pick_canvas_requested = Signal(str)       # purpose: "roi"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._idx: int = -1
        self._spec: Optional[ToolSpec] = None
        self._widgets: dict[str, tuple[Param, QWidget]] = {}
        self._suppress: bool = False
        self._live: bool = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Header row
        header_row = QHBoxLayout()
        self.header_label = QLabel("Properties")
        self.header_label.setProperty("subheading", True)
        self.live_chk = QCheckBox("Live preview")
        self.live_chk.setChecked(True)
        self.live_chk.setToolTip("Tự chạy pipeline khi đổi tham số")
        self.live_chk.toggled.connect(self._on_live_toggled)
        header_row.addWidget(self.header_label)
        header_row.addStretch()
        header_row.addWidget(self.live_chk)
        outer.addLayout(header_row)

        # Empty-state hint
        self.empty_state = QFrame()
        self.empty_state.setProperty("card", True)
        es_lay = QVBoxLayout(self.empty_state)
        es_lay.setContentsMargins(14, 18, 14, 18)
        es_lay.setSpacing(6)
        big_icon = QLabel("⚙")
        big_icon.setStyleSheet("color:#36c5d6; font-size:28pt;")
        big_icon.setAlignment(Qt.AlignCenter)
        es_lay.addWidget(big_icon)
        es_title = QLabel("Chưa chọn node")
        es_title.setProperty("subheading", True)
        es_title.setAlignment(Qt.AlignCenter)
        es_lay.addWidget(es_title)
        es_hint = QLabel("Click 1 node ở danh sách phía trên để xem và chỉnh thuộc tính.\n"
                         "Hoặc bấm <b>+ Add</b> bên dưới để thêm tool mới.")
        es_hint.setProperty("muted", True)
        es_hint.setAlignment(Qt.AlignCenter)
        es_hint.setWordWrap(True)
        es_lay.addWidget(es_hint)
        outer.addWidget(self.empty_state)

        # Form holder (rebuild khi node thay đổi)
        self.form_holder = QFrame()
        self.form_holder.setProperty("card", True)
        self.form_layout = QFormLayout(self.form_holder)
        self.form_layout.setContentsMargins(10, 10, 10, 10)
        self.form_layout.setSpacing(8)
        self.form_holder.setVisible(False)
        outer.addWidget(self.form_holder)

        # Action row (Reset, Pick from canvas, Apply)
        self.action_row = QHBoxLayout()
        self.reset_btn = QPushButton("↺  Reset defaults")
        self.reset_btn.setProperty("secondary", True)
        self.reset_btn.setToolTip("Đặt lại về giá trị mặc định của tool")
        self.reset_btn.clicked.connect(self._on_reset)
        self.reset_btn.setVisible(False)

        self.pick_btn = QPushButton("📌  Pick from canvas")
        self.pick_btn.setProperty("secondary", True)
        self.pick_btn.setToolTip("Vẽ ROI trên ảnh để điền x/y/w/h")
        self.pick_btn.clicked.connect(lambda: self.pick_canvas_requested.emit("roi"))
        self.pick_btn.setVisible(False)

        self.apply_btn = QPushButton("▶  Apply (Run pipeline)")
        self.apply_btn.clicked.connect(self.live_run_requested.emit)
        self.apply_btn.setVisible(False)

        self.action_row.addWidget(self.reset_btn)
        self.action_row.addWidget(self.pick_btn)
        self.action_row.addStretch()
        self.action_row.addWidget(self.apply_btn)
        outer.addLayout(self.action_row)

        outer.addStretch()

        # Debounce timer
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._emit_changes)

    # ------------------------------------------------------------------
    def _on_live_toggled(self, on: bool):
        self._live = on
        self.apply_btn.setVisible(not on)

    def show_node(self, idx: int, node: PipelineNode) -> None:
        self._idx = idx
        spec = TOOLS.get(node.tool_id)
        if spec is None:
            self.clear()
            return
        self._spec = spec
        self.header_label.setText(f"Properties — #{idx+1} {spec.icon} {spec.display}")
        self.empty_state.setVisible(False)

        # Clear current form
        self._suppress = True
        self._widgets.clear()
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)

        if not spec.params:
            empty = QLabel("(tool này không có tham số)")
            empty.setProperty("muted", True)
            self.form_layout.addRow(empty)
        else:
            for p in spec.params:
                w = self._build_widget(p, node.params.get(p.name, p.default))
                self.form_layout.addRow(p.label, w)
                self._widgets[p.name] = (p, w)

        self.form_holder.setVisible(True)
        self.reset_btn.setVisible(bool(spec.params))
        # Pick from canvas chỉ hiện cho tool có x/y/w/h params (vd ROI)
        has_xywh = all(k in [p.name for p in spec.params] for k in ("x", "y", "w", "h"))
        self.pick_btn.setVisible(has_xywh)
        self.apply_btn.setVisible(not self._live)
        self._suppress = False

    def clear(self) -> None:
        self._idx = -1
        self._spec = None
        self._widgets.clear()
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self.form_holder.setVisible(False)
        self.reset_btn.setVisible(False)
        self.pick_btn.setVisible(False)
        self.apply_btn.setVisible(False)
        self.header_label.setText("Properties")
        self.empty_state.setVisible(True)

    def _on_reset(self):
        if self._idx < 0 or self._spec is None:
            return
        defaults = self._spec.default_params()
        self._suppress = True
        for name, (p, w) in self._widgets.items():
            v = defaults.get(name, p.default)
            if isinstance(w, QSpinBox): w.setValue(int(v))
            elif isinstance(w, QDoubleSpinBox): w.setValue(float(v))
            elif isinstance(w, QComboBox):
                if v in (p.choices or []): w.setCurrentText(str(v))
            elif isinstance(w, QCheckBox): w.setChecked(bool(v))
            elif isinstance(w, QLineEdit): w.setText(str(v))
        self._suppress = False
        self._debounce.start()  # trigger emit_changes

    def fill_xywh(self, x: int, y: int, w: int, h: int) -> None:
        """Điền params x/y/w/h từ ROI vẽ trên canvas (cho ROI tool)."""
        if not all(k in self._widgets for k in ("x", "y", "w", "h")):
            return
        self._suppress = True
        for name, val in (("x", x), ("y", y), ("w", w), ("h", h)):
            widget = self._widgets[name][1]
            if isinstance(widget, QSpinBox):
                widget.setValue(int(val))
        self._suppress = False
        self._debounce.start()

    def update_for_idx(self, idx: int, node: PipelineNode) -> None:
        """Sync nếu cùng idx (ví dụ sau reorder hoặc params bị invalidate)."""
        if idx == self._idx:
            self.show_node(idx, node)

    @property
    def live(self) -> bool:
        return self._live

    # ------------------------------------------------------------------
    def _build_widget(self, p: Param, value):
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
            w.valueChanged.connect(self._on_change)
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
            w.valueChanged.connect(self._on_change)
            return w
        if p.kind == "choice":
            w = QComboBox()
            choices = p.choices or []
            w.addItems(choices)
            if value in choices:
                w.setCurrentText(str(value))
            w.currentTextChanged.connect(self._on_change)
            return w
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(self._on_change)
            return w
        w = QLineEdit(str(value))
        w.textChanged.connect(self._on_change)
        return w

    def _on_change(self, *args):
        if self._suppress or self._idx < 0:
            return
        # Debounce — gom các thay đổi rồi mới emit (slider kéo nhanh ổn)
        self._debounce.start()

    def _emit_changes(self):
        if self._idx < 0:
            return
        params: dict = {}
        for name, (p, w) in self._widgets.items():
            if isinstance(w, QSpinBox):
                params[name] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                params[name] = float(w.value())
            elif isinstance(w, QComboBox):
                params[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                params[name] = w.isChecked()
            elif isinstance(w, QLineEdit):
                params[name] = w.text()
        self.params_changed.emit(self._idx, params)


class PipelinePanel(QWidget):
    add_node_requested = Signal(str)             # tool_id
    edit_node_requested = Signal(int)            # idx (legacy dialog path)
    delete_node_requested = Signal(int)          # idx
    enable_node_changed = Signal(int, bool)
    select_node = Signal(int)
    run_requested = Signal()
    clear_requested = Signal()
    reorder_changed = Signal(int, int)           # src, dst
    params_changed = Signal(int, dict)           # idx, new_params (live edit)
    live_apply_requested = Signal()              # khi tắt live mode
    pick_canvas_requested = Signal(int, str)     # idx, purpose ("roi")

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

        # Vertical splitter: list (top) + properties (bottom)
        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setMovement(QListWidget.Snap)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(False)
        self.list_widget.itemSelectionChanged.connect(self._on_select)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        split.addWidget(self.list_widget)

        # Properties panel (live edit)
        self.props_panel = NodePropertiesPanel()
        self.props_panel.params_changed.connect(self.params_changed.emit)
        self.props_panel.live_run_requested.connect(self.live_apply_requested.emit)
        self.props_panel.pick_canvas_requested.connect(
            lambda purpose: self.pick_canvas_requested.emit(self.props_panel._idx, purpose)
        )
        split.addWidget(self.props_panel)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([320, 280])
        layout.addWidget(split, 1)

        # Add tool — menu phân nhóm theo category
        self.add_btn = QToolButton()
        self.add_btn.setText("+  Add Tool")
        self.add_btn.setPopupMode(QToolButton.InstantPopup)
        self.add_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.add_btn.setStyleSheet(
            "QToolButton{background:#36c5d6; color:#0b1220; border:none;"
            " border-radius:6px; padding:8px 16px; font-weight:700;}"
            "QToolButton:hover{background:#5dd5e3;}"
            "QToolButton::menu-indicator{image:none;}"
        )
        menu = QMenu(self.add_btn)
        # group by category
        cats: dict[str, list] = {}
        for tool_id, spec in TOOLS.items():
            cats.setdefault(spec.category, []).append((tool_id, spec))
        for cat in ("Pre-process", "Locate", "Measure", "Identify", "Inspect", "Other"):
            items = cats.get(cat, [])
            if not items:
                continue
            sub = menu.addMenu(cat)
            for tool_id, spec in items:
                act = QAction(f"{spec.icon}  {spec.display}", sub)
                act.triggered.connect(lambda _=False, tid=tool_id: self.add_node_requested.emit(tid))
                sub.addAction(act)
        self.add_btn.setMenu(menu)
        layout.addWidget(self.add_btn)

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
    def _on_select(self):
        items = self.list_widget.selectedItems()
        if items:
            idx = self.list_widget.row(items[0])
            self.select_node.emit(idx)
            # Cập nhật properties panel — chỉ rebuild form nếu đổi sang node khác
            # (giữ widget user đang gõ khi live edit cùng node)
            if idx != self.props_panel._idx:
                item = self.list_widget.item(idx)
                widget = self.list_widget.itemWidget(item) if item else None
                if isinstance(widget, PipelineItem):
                    self.props_panel.show_node(idx, widget.node)
        else:
            self.props_panel.clear()

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
            self.props_panel.show_node(select_idx, nodes[select_idx])
        else:
            self.props_panel.clear()

    def refresh_node(self, idx: int, node: PipelineNode) -> None:
        item = self.list_widget.item(idx)
        if item is None:
            return
        widget = self.list_widget.itemWidget(item)
        if isinstance(widget, PipelineItem):
            widget.node = node
            widget.refresh()
        # Nếu đang chọn node này thì sync props (sau khi reset cache do edit)
        # — nhưng tránh sync khi chính props panel đang phát ra change
        if (self.list_widget.currentRow() == idx and
                not self.props_panel._suppress and
                self.props_panel._debounce.isActive() is False):
            # Không rebuild form nếu cùng spec (giữ giá trị widget user đang gõ)
            pass

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def select(self, idx: int) -> None:
        self.list_widget.setCurrentRow(idx)
