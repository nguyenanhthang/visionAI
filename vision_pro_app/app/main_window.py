"""Main window for the Vision Pro Image Studio."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .processor import FILTERS, ImageProcessor
from .widgets import GlassPanel, HistogramWidget, ImageCanvas, SliderRow


FILTER_LABELS = {
    "none": "Original",
    "grayscale": "Mono",
    "sepia": "Sepia",
    "invert": "Invert",
    "blur": "Soft Blur",
    "sharpen": "Sharpen",
    "edge": "Edges",
    "emboss": "Emboss",
    "cool": "Cool",
    "warm": "Warm",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Pro Image Studio")
        self.resize(1280, 820)
        self.setMinimumSize(1080, 720)

        self.processor = ImageProcessor()

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(self._build_canvas_panel(), stretch=3)
        body.addWidget(self._build_tools_panel(), stretch=2)
        root.addLayout(body, stretch=1)

        root.addWidget(self._build_status_bar())

        self._install_shortcuts()
        self._refresh_states()

    # =================================================================
    # Layout builders
    # =================================================================
    def _build_header(self) -> QWidget:
        header = GlassPanel(variant="default")
        header.setObjectName("HeaderBar")
        header.setFixedHeight(78)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 12, 22, 12)
        layout.setSpacing(14)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Vision Pro Image Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("VISIONOS · IMAGE LAB")
        subtitle.setObjectName("AppSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)

        layout.addStretch(1)

        self.btn_open = QPushButton("Open")
        self.btn_open.clicked.connect(self._on_open)

        self.btn_save = QPushButton("Save As…")
        self.btn_save.clicked.connect(self._on_save)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self._on_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self._on_redo)

        self.btn_revert = QPushButton("Revert")
        self.btn_revert.clicked.connect(self._on_revert)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("PrimaryButton")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_apply.setToolTip("Bake the current adjustments as a new editable layer")

        for btn in (
            self.btn_open,
            self.btn_save,
            self.btn_undo,
            self.btn_redo,
            self.btn_revert,
            self.btn_apply,
        ):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(btn)

        return header

    def _build_canvas_panel(self) -> QWidget:
        wrapper = GlassPanel(variant="default")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        self.canvas = ImageCanvas()
        self.canvas.file_dropped.connect(self._load_path)
        layout.addWidget(self.canvas, stretch=1)

        self.histogram = HistogramWidget()
        layout.addWidget(self.histogram)

        return wrapper

    def _build_tools_panel(self) -> QWidget:
        wrapper = GlassPanel(variant="strong")
        wrapper.setMinimumWidth(360)
        wrapper.setMaximumWidth(460)

        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        heading = QLabel("Studio")
        heading.setObjectName("AppTitle")
        outer.addWidget(heading)

        sub = QLabel("CRAFT · TUNE · EXPORT")
        sub.setObjectName("AppSubtitle")
        outer.addWidget(sub)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_adjust_tab(), "Adjust")
        tabs.addTab(self._build_filters_tab(), "Filters")
        tabs.addTab(self._build_transform_tab(), "Transform")
        outer.addWidget(tabs, stretch=1)

        return wrapper

    def _build_adjust_tab(self) -> QWidget:
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.Shape.NoFrame)

        host = QWidget()
        scroller.setWidget(host)

        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        layout.addWidget(self._section_title("Light"))
        self.slider_brightness = SliderRow("Brightness", default=0)
        self.slider_brightness.valueChanged.connect(self._on_brightness)
        layout.addWidget(self.slider_brightness)

        self.slider_contrast = SliderRow("Contrast", default=0)
        self.slider_contrast.valueChanged.connect(self._on_contrast)
        layout.addWidget(self.slider_contrast)

        layout.addWidget(self._section_title("Color"))
        self.slider_saturation = SliderRow("Saturation", default=0)
        self.slider_saturation.valueChanged.connect(self._on_saturation)
        layout.addWidget(self.slider_saturation)

        layout.addWidget(self._section_title("Detail"))
        self.slider_sharpness = SliderRow("Sharpness", default=0)
        self.slider_sharpness.valueChanged.connect(self._on_sharpness)
        layout.addWidget(self.slider_sharpness)

        layout.addStretch(1)

        reset = QPushButton("Reset Adjustments")
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(self._reset_adjustments)
        layout.addWidget(reset)

        return scroller

    def _build_filters_tab(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        layout.addWidget(self._section_title("Filter"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)

        for index, name in enumerate(FILTERS):
            btn = QPushButton(FILTER_LABELS.get(name, name.title()))
            btn.setObjectName("FilterTile")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setProperty("filter_name", name)
            if name == "none":
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, n=name: self._on_filter(n))
            self._filter_group.addButton(btn)
            row, col = divmod(index, 2)
            grid.addWidget(btn, row, col)

        layout.addLayout(grid)
        layout.addStretch(1)
        return host

    def _build_transform_tab(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        layout.addWidget(self._section_title("Rotate"))
        self.slider_rotation = SliderRow(
            "Rotation", minimum=-180, maximum=180, default=0, unit="°"
        )
        self.slider_rotation.valueChanged.connect(self._on_rotation)
        layout.addWidget(self.slider_rotation)

        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(8)
        for label, delta in (("-90°", -90), ("-15°", -15), ("+15°", 15), ("+90°", 90)):
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, d=delta: self._nudge_rotation(d))
            rotate_row.addWidget(btn)
        layout.addLayout(rotate_row)

        layout.addWidget(self._section_title("Flip"))
        flip_row = QHBoxLayout()
        flip_row.setSpacing(8)
        self.btn_flip_h = QPushButton("Flip Horizontal")
        self.btn_flip_h.setObjectName("ToggleChip")
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flip_h.toggled.connect(self._on_flip_h)
        flip_row.addWidget(self.btn_flip_h)

        self.btn_flip_v = QPushButton("Flip Vertical")
        self.btn_flip_v.setObjectName("ToggleChip")
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_flip_v.toggled.connect(self._on_flip_v)
        flip_row.addWidget(self.btn_flip_v)
        layout.addLayout(flip_row)

        layout.addStretch(1)
        return host

    def _build_status_bar(self) -> QWidget:
        bar = GlassPanel(variant="default", glow=False)
        bar.setObjectName("StatusBar")
        bar.setFixedHeight(46)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(20)

        self.status_file = QLabel("No image loaded")
        self.status_file.setObjectName("StatusText")

        self.status_size = QLabel("—")
        self.status_size.setObjectName("StatusText")

        self.status_filter = QLabel("Filter: Original")
        self.status_filter.setObjectName("StatusText")

        layout.addWidget(self.status_file)
        layout.addStretch(1)
        layout.addWidget(self.status_size)
        layout.addWidget(self._dot())
        layout.addWidget(self.status_filter)

        return bar

    # =================================================================
    # Helpers
    # =================================================================
    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _dot(self) -> QLabel:
        dot = QLabel("•")
        dot.setObjectName("StatusText")
        return dot

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self._on_open)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._on_save)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._on_undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self._on_redo)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._on_revert)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._on_apply)

    # =================================================================
    # File actions
    # =================================================================
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            self.processor.load(path)
        except Exception as exc:  # noqa: BLE001 - surface any IO/decoding failure
            QMessageBox.critical(self, "Failed to open image", str(exc))
            return
        self._reset_controls(silent=True)
        self._render()
        self._refresh_states()

    def _on_save(self) -> None:
        if not self.processor.has_image:
            return
        suggested = self.processor.source_path or "untitled.png"
        base = os.path.splitext(os.path.basename(suggested))[0]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            f"{base}_edited.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;Bitmap (*.bmp)",
        )
        if not path:
            return
        try:
            self.processor.save(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to save", str(exc))
            return
        self.status_file.setText(f"Saved · {os.path.basename(path)}")

    # =================================================================
    # Adjustments
    # =================================================================
    def _on_brightness(self, value: int) -> None:
        self.processor.set_brightness(self._slider_to_factor(value))
        self._render()

    def _on_contrast(self, value: int) -> None:
        self.processor.set_contrast(self._slider_to_factor(value))
        self._render()

    def _on_saturation(self, value: int) -> None:
        self.processor.set_saturation(self._slider_to_factor(value, span=1.0))
        self._render()

    def _on_sharpness(self, value: int) -> None:
        self.processor.set_sharpness(self._slider_to_factor(value))
        self._render()

    def _on_rotation(self, value: int) -> None:
        self.processor.set_rotation(value)
        self._render()

    def _on_flip_h(self, checked: bool) -> None:
        self.processor.adjustments.flip_horizontal = bool(checked)
        self._render()

    def _on_flip_v(self, checked: bool) -> None:
        self.processor.adjustments.flip_vertical = bool(checked)
        self._render()

    def _on_filter(self, name: str) -> None:
        self.processor.set_filter(name)
        self._render()
        self.status_filter.setText(f"Filter: {FILTER_LABELS.get(name, name.title())}")

    def _nudge_rotation(self, delta: int) -> None:
        new_value = self.slider_rotation.value() + delta
        # clamp to slider range
        new_value = max(-180, min(180, new_value))
        self.slider_rotation.set_value(new_value)

    @staticmethod
    def _slider_to_factor(value: int, span: float = 1.0) -> float:
        """Map slider [-100..100] to enhancement factor in [1-span, 1+span]."""
        return 1.0 + (value / 100.0) * span

    # =================================================================
    # History
    # =================================================================
    def _on_undo(self) -> None:
        if not self.processor.can_undo():
            return
        self.processor.undo()
        self._reset_controls(silent=True)
        self._render()
        self._refresh_states()

    def _on_redo(self) -> None:
        if not self.processor.can_redo():
            return
        self.processor.redo()
        self._reset_controls(silent=True)
        self._render()
        self._refresh_states()

    def _on_revert(self) -> None:
        if not self.processor.has_image:
            return
        self.processor.revert_to_load()
        self._reset_controls(silent=True)
        self._render()
        self._refresh_states()

    def _on_apply(self) -> None:
        if not self.processor.has_image:
            return
        self.processor.commit()
        self._reset_controls(silent=True)
        self._render()
        self._refresh_states()

    def _reset_adjustments(self) -> None:
        self.processor.reset_adjustments()
        self._reset_controls(silent=True)
        self._render()

    # =================================================================
    # State sync
    # =================================================================
    def _reset_controls(self, *, silent: bool = False) -> None:
        sliders = (
            self.slider_brightness,
            self.slider_contrast,
            self.slider_saturation,
            self.slider_sharpness,
            self.slider_rotation,
        )
        for slider in sliders:
            slider.blockSignals(silent)
            slider.reset()
            slider.blockSignals(False)

        self.btn_flip_h.blockSignals(silent)
        self.btn_flip_v.blockSignals(silent)
        self.btn_flip_h.setChecked(False)
        self.btn_flip_v.setChecked(False)
        self.btn_flip_h.blockSignals(False)
        self.btn_flip_v.blockSignals(False)

        for btn in self._filter_group.buttons():
            if btn.property("filter_name") == "none":
                btn.setChecked(True)
                break
        self.status_filter.setText("Filter: Original")

    def _render(self) -> None:
        rendered = self.processor.render()
        self.canvas.set_image(rendered)
        self.histogram.set_image(rendered)
        self._refresh_states()
        if rendered is not None:
            self.status_size.setText(f"{rendered.width} × {rendered.height} px")

    def _refresh_states(self) -> None:
        has = self.processor.has_image
        self.btn_save.setEnabled(has)
        self.btn_apply.setEnabled(has)
        self.btn_revert.setEnabled(has)
        self.btn_undo.setEnabled(self.processor.can_undo())
        self.btn_redo.setEnabled(self.processor.can_redo())

        if has and self.processor.source_path:
            self.status_file.setText(os.path.basename(self.processor.source_path))
        elif not has:
            self.status_file.setText("No image loaded")
            self.status_size.setText("—")
