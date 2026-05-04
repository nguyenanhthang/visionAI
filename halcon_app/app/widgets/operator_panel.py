"""Panel chứa các tab parameter cho từng operator HALCON."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class _ThresholdTab(QWidget):
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox("Threshold + Blob Analysis")
        form = QFormLayout(group)
        self.min_gray = QSpinBox()
        self.min_gray.setRange(0, 255)
        self.min_gray.setValue(0)
        self.max_gray = QSpinBox()
        self.max_gray.setRange(0, 255)
        self.max_gray.setValue(128)
        self.min_area = QSpinBox()
        self.min_area.setRange(1, 10_000_000)
        self.min_area.setValue(100)
        self.max_area = QSpinBox()
        self.max_area.setRange(1, 100_000_000)
        self.max_area.setValue(10_000_000)
        form.addRow("Min Gray", self.min_gray)
        form.addRow("Max Gray", self.max_gray)
        form.addRow("Min Area", self.min_area)
        form.addRow("Max Area", self.max_area)
        layout.addWidget(group)

        hint = QLabel(
            "HALCON: threshold(Image, MinGray, MaxGray) → connection → "
            "select_shape('area', ...)."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        run_btn = QPushButton("▶  Run Threshold")
        run_btn.clicked.connect(self._on_run)
        layout.addWidget(run_btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit(
            {
                "min_gray": self.min_gray.value(),
                "max_gray": self.max_gray.value(),
                "min_area": self.min_area.value(),
                "max_area": self.max_area.value(),
            }
        )


class _EdgesTab(QWidget):
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("Edges (sub-pixel)")
        form = QFormLayout(group)
        self.method = QComboBox()
        self.method.addItems(["canny", "sobel", "deriche2", "lanser2"])
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.1, 10.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1.0)
        self.low = QSpinBox()
        self.low.setRange(0, 255)
        self.low.setValue(40)
        self.high = QSpinBox()
        self.high.setRange(0, 255)
        self.high.setValue(120)
        form.addRow("Method", self.method)
        form.addRow("Alpha (smooth)", self.alpha)
        form.addRow("Low threshold", self.low)
        form.addRow("High threshold", self.high)
        layout.addWidget(group)

        hint = QLabel("HALCON: edges_sub_pix(Image, Edges, Method, Alpha, Low, High).")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn = QPushButton("▶  Run Edges")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit(
            {
                "method": self.method.currentText(),
                "alpha": self.alpha.value(),
                "low": self.low.value(),
                "high": self.high.value(),
            }
        )


class _ShapeMatchTab(QWidget):
    run_requested = Signal(dict)
    template_load_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("Shape-based Matching")
        form = QFormLayout(group)
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.1, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setValue(0.6)
        self.num_matches = QSpinBox()
        self.num_matches.setRange(1, 100)
        self.num_matches.setValue(5)
        self.angle_start = QDoubleSpinBox()
        self.angle_start.setRange(-3.14, 3.14)
        self.angle_start.setValue(-0.39)
        self.angle_start.setSingleStep(0.05)
        self.angle_extent = QDoubleSpinBox()
        self.angle_extent.setRange(0.0, 6.28)
        self.angle_extent.setValue(0.78)
        self.angle_extent.setSingleStep(0.05)
        form.addRow("Min Score", self.min_score)
        form.addRow("Num Matches", self.num_matches)
        form.addRow("Angle Start (rad)", self.angle_start)
        form.addRow("Angle Extent (rad)", self.angle_extent)
        layout.addWidget(group)

        self.template_label = QLabel("Template: <chưa chọn>")
        self.template_label.setProperty("muted", True)
        layout.addWidget(self.template_label)

        row = QHBoxLayout()
        load_btn = QPushButton("📂  Load Template…")
        load_btn.setProperty("secondary", True)
        load_btn.clicked.connect(self.template_load_requested.emit)
        run_btn = QPushButton("▶  Run Match")
        run_btn.clicked.connect(self._on_run)
        row.addWidget(load_btn)
        row.addWidget(run_btn)
        layout.addLayout(row)

        hint = QLabel(
            "HALCON: create_shape_model + find_shape_model. Fallback dùng "
            "matchTemplate (NCC) — không xoay."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

    def set_template_name(self, name: str | None):
        self.template_label.setText(
            f"Template: {name}" if name else "Template: <chưa chọn>"
        )

    def _on_run(self):
        self.run_requested.emit(
            {
                "min_score": self.min_score.value(),
                "num_matches": self.num_matches.value(),
                "angle_start": self.angle_start.value(),
                "angle_extent": self.angle_extent.value(),
            }
        )


class _MeasureTab(QWidget):
    measure_mode_toggled = Signal(bool)
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("Measure 1D")
        form = QFormLayout(group)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 255)
        self.threshold.setValue(30)
        form.addRow("Sigma", self.sigma)
        form.addRow("Edge Threshold", self.threshold)
        layout.addWidget(group)

        self.draw_btn = QPushButton("✎  Vẽ segment trên ảnh")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.measure_mode_toggled.emit)
        layout.addWidget(self.draw_btn)

        self.segment_label = QLabel("Segment: <chưa vẽ>")
        self.segment_label.setProperty("muted", True)
        layout.addWidget(self.segment_label)

        run_btn = QPushButton("▶  Run Measure")
        run_btn.clicked.connect(self._on_run)
        layout.addWidget(run_btn)

        hint = QLabel(
            "HALCON: gen_measure_rectangle2 + measure_pairs. Click & drag để "
            "vẽ đường đo."
        )
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

        self._segment: tuple[int, int, int, int] | None = None

    def set_segment(self, r1: int, c1: int, r2: int, c2: int):
        self._segment = (r1, c1, r2, c2)
        self.segment_label.setText(f"Segment: ({c1},{r1}) → ({c2},{r2})")
        self.draw_btn.setChecked(False)

    def _on_run(self):
        if self._segment is None:
            return
        r1, c1, r2, c2 = self._segment
        self.run_requested.emit(
            {
                "row1": r1,
                "col1": c1,
                "row2": r2,
                "col2": c2,
                "sigma": self.sigma.value(),
                "threshold": self.threshold.value(),
            }
        )


class OperatorPanel(QTabWidget):
    threshold_run = Signal(dict)
    edges_run = Signal(dict)
    shape_match_run = Signal(dict)
    shape_template_load = Signal()
    measure_run = Signal(dict)
    measure_mode_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.threshold_tab = _ThresholdTab()
        self.edges_tab = _EdgesTab()
        self.shape_tab = _ShapeMatchTab()
        self.measure_tab = _MeasureTab()

        self.addTab(self.threshold_tab, "Blob")
        self.addTab(self.edges_tab, "Edges")
        self.addTab(self.shape_tab, "Match")
        self.addTab(self.measure_tab, "Measure")

        self.threshold_tab.run_requested.connect(self.threshold_run.emit)
        self.edges_tab.run_requested.connect(self.edges_run.emit)
        self.shape_tab.run_requested.connect(self.shape_match_run.emit)
        self.shape_tab.template_load_requested.connect(self.shape_template_load.emit)
        self.measure_tab.run_requested.connect(self.measure_run.emit)
        self.measure_tab.measure_mode_toggled.connect(self.measure_mode_toggled.emit)

    def set_template_name(self, name: str | None):
        self.shape_tab.set_template_name(name)

    def set_measure_segment(self, r1: int, c1: int, r2: int, c2: int):
        self.measure_tab.set_segment(r1, c1, r2, c2)
