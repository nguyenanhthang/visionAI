"""Main window cho HALCON Vision Studio."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.operators import (
    HALCON_AVAILABLE,
    HALCON_VERSION,
    edges_sub_pix,
    measure_pairs,
    read_image,
    shape_match,
    threshold_blob,
)
from app.widgets import ImageCanvas, OperatorPanel, ResultsView


SUPPORTED_EXTS = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HALCON Vision Studio")
        self.resize(1400, 880)
        self.setMinimumSize(QSize(1100, 700))

        self._image_path: Optional[Path] = None
        self._original_image: Optional[np.ndarray] = None
        self._template_path: Optional[Path] = None
        self._template_image: Optional[np.ndarray] = None

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal, central)

        # Left: image canvas
        canvas_holder = QWidget()
        cv_lay = QVBoxLayout(canvas_holder)
        cv_lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Image Viewer")
        title.setProperty("heading", True)
        cv_lay.addWidget(title)
        self.canvas = ImageCanvas()
        cv_lay.addWidget(self.canvas, 1)

        # Right: operator panel + results
        right = QSplitter(Qt.Vertical)
        self.operator_panel = OperatorPanel()
        self.results_view = ResultsView()
        right.addWidget(self.operator_panel)
        right.addWidget(self.results_view)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)

        splitter.addWidget(canvas_holder)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("&File")
        act_open = QAction("Open Image…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open_image)
        m_file.addAction(act_open)

        act_template = QAction("Open Template…", self)
        act_template.triggered.connect(self._on_open_template)
        m_file.addAction(act_template)

        m_file.addSeparator()
        act_save = QAction("Save Result Image…", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save_result)
        m_file.addAction(act_save)

        m_file.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_view = bar.addMenu("&View")
        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(self.canvas.fit_to_view)
        m_view.addAction(act_fit)
        act_reset = QAction("Reset Zoom", self)
        act_reset.setShortcut("Ctrl+1")
        act_reset.triggered.connect(self.canvas.reset_zoom)
        m_view.addAction(act_reset)

        m_help = bar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        m_help.addAction(act_about)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("📂  Open Image", self)
        act_open.triggered.connect(self._on_open_image)
        tb.addAction(act_open)

        act_template = QAction("🧩  Open Template", self)
        act_template.triggered.connect(self._on_open_template)
        tb.addAction(act_template)

        tb.addSeparator()

        act_reset_img = QAction("↺  Reset Image", self)
        act_reset_img.triggered.connect(self._on_reset_image)
        tb.addAction(act_reset_img)

        act_fit = QAction("🔍  Fit", self)
        act_fit.triggered.connect(self.canvas.fit_to_view)
        tb.addAction(act_fit)

        tb.addSeparator()
        act_save = QAction("💾  Save Result", self)
        act_save.triggered.connect(self._on_save_result)
        tb.addAction(act_save)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        engine = (
            f"HALCON {HALCON_VERSION} ✓" if HALCON_AVAILABLE else "HALCON unavailable — using OpenCV fallback"
        )
        self._engine_label = QLabel(engine)
        self._image_label = QLabel("No image")
        self._cursor_label = QLabel("")
        sb.addPermanentWidget(self._image_label, 2)
        sb.addPermanentWidget(self._cursor_label, 2)
        sb.addPermanentWidget(self._engine_label, 1)

    def _wire_signals(self):
        self.operator_panel.threshold_run.connect(self._run_threshold)
        self.operator_panel.edges_run.connect(self._run_edges)
        self.operator_panel.shape_match_run.connect(self._run_shape_match)
        self.operator_panel.shape_template_load.connect(self._on_open_template)
        self.operator_panel.measure_run.connect(self._run_measure)
        self.operator_panel.measure_mode_toggled.connect(self.canvas.set_measure_mode)

        self.canvas.measure_segment_drawn.connect(
            self.operator_panel.set_measure_segment
        )
        self.canvas.mouse_moved.connect(self._on_cursor_moved)

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), SUPPORTED_EXTS
        )
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được ảnh:\n{exc}")
            return
        self._image_path = Path(path)
        self._original_image = img
        self.canvas.set_image(img)
        self._image_label.setText(
            f"{self._image_path.name}  •  {img.shape[1]}×{img.shape[0]}"
        )
        self.results_view.append_log(f"Loaded image: {self._image_path.name}")

    def _on_open_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Template", str(Path.home()), SUPPORTED_EXTS
        )
        if not path:
            return
        try:
            tpl = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được template:\n{exc}")
            return
        self._template_path = Path(path)
        self._template_image = tpl
        self.operator_panel.set_template_name(self._template_path.name)
        self.results_view.append_log(f"Loaded template: {self._template_path.name}")

    def _on_reset_image(self):
        if self._original_image is not None:
            self.canvas.set_image(self._original_image)

    def _on_save_result(self):
        img = self.canvas.current_image()
        if img is None:
            QMessageBox.information(self, "Save", "Chưa có ảnh để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", str(Path.home() / "result.png"), SUPPORTED_EXTS
        )
        if not path:
            return
        import cv2

        ok = cv2.imwrite(path, img)
        if ok:
            self.results_view.append_log(f"Saved → {path}")
        else:
            QMessageBox.warning(self, "Save", "Lưu thất bại.")

    def _on_about(self):
        msg = (
            "<b>HALCON Vision Studio</b><br>"
            "PySide6 GUI wrapping MVTec HALCON operators.<br><br>"
            f"Engine: <b>{'HALCON ' + HALCON_VERSION if HALCON_AVAILABLE else 'OpenCV fallback'}</b>"
        )
        QMessageBox.about(self, "About", msg)

    def _on_cursor_moved(self, row: int, col: int, value):
        self._cursor_label.setText(f"({col}, {row})  px={value}")

    # ------------------------------------------------------------------
    # Operator runs
    # ------------------------------------------------------------------
    def _ensure_image(self) -> bool:
        if self._original_image is None:
            QMessageBox.information(self, "No Image", "Hãy mở một ảnh trước.")
            return False
        return True

    def _run_threshold(self, params: dict):
        if not self._ensure_image():
            return
        result = threshold_blob(self._original_image, **params)
        self._show_result(result, title="Threshold + Blob")

    def _run_edges(self, params: dict):
        if not self._ensure_image():
            return
        result = edges_sub_pix(self._original_image, **params)
        self._show_result(result, title="Edges")

    def _run_shape_match(self, params: dict):
        if not self._ensure_image():
            return
        if self._template_image is None:
            QMessageBox.information(
                self, "Template", "Hãy load template trước khi match."
            )
            return
        result = shape_match(self._original_image, self._template_image, **params)
        self._show_result(result, title="Shape Match")

    def _run_measure(self, params: dict):
        if not self._ensure_image():
            return
        result = measure_pairs(self._original_image, **params)
        self._show_result(result, title="Measure 1D")

    def _show_result(self, result, title: str):
        self.canvas.set_image(result.image)
        self.results_view.set_metrics(result.metrics)
        self.results_view.append_log(f"=== {title} ===")
        self.results_view.append_log(result.log)
