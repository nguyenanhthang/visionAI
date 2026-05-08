"""
ui/patmax_dialog.py
PatMax Training & Search Dialog — mô phỏng CogPatMaxPatternAlignTool
Cognex VisionPro style:
  - Tab TRAIN: vẽ ROI trên ảnh, set origin, train, xem model preview
  - Tab SEARCH: chạy search, xem kết quả, score map, multi-result table
  - Tab SETTINGS: các params search (angle range, scale, threshold, ...)
  - Lưu/Load model file .patmax
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QTabWidget, QWidget, QScrollArea, QFrame,
                                QSplitter, QPushButton, QGroupBox,
                                QSizePolicy, QFileDialog, QMessageBox,
                                QSpinBox, QDoubleSpinBox, QCheckBox,
                                QComboBox, QSlider, QTableWidget,
                                QTableWidgetItem, QHeaderView, QLineEdit,
                                QProgressBar)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer, QThread
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
                            QPen, QBrush, QCursor, QMouseEvent)

from core.flow_graph import NodeInstance, FlowGraph
from core.patmax_engine import (PatMaxModel, PatMaxResult, train_patmax,
                                  run_patmax, draw_patmax_results,
                                  save_model, load_model)


# ── Reuse InteractiveImageLabel từ node_detail_dialog ────────────
from ui.node_detail_dialog import InteractiveImageLabel


# ════════════════════════════════════════════════════════════════════
#  Model preview widget
# ════════════════════════════════════════════════════════════════════
class ModelPreviewWidget(QWidget):
    """Hiển thị thông tin model đã train."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setStyleSheet(
            "background:#0a0e1a; border:1px solid #1e2d45; border-radius:6px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(12)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(80, 80)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:4px;")
        self._thumb.setText("No Model")
        lay.addWidget(self._thumb)

        # Info
        info = QWidget()
        il = QVBoxLayout(info); il.setContentsMargins(0,0,0,0); il.setSpacing(3)
        self._name_lbl = QLabel("No model trained")
        self._name_lbl.setStyleSheet(
            "color:#64748b; font-size:13px; font-weight:700;")
        self._info1 = QLabel("")
        self._info1.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._info2 = QLabel("")
        self._info2.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._hash_lbl = QLabel("")
        self._hash_lbl.setStyleSheet(
            "color:#1e2d45; font-size:10px; font-family:'Courier New';")
        il.addWidget(self._name_lbl)
        il.addWidget(self._info1)
        il.addWidget(self._info2)
        il.addWidget(self._hash_lbl)
        il.addStretch()
        lay.addWidget(info, 1)

        # Status badge
        self._badge = QLabel("UNTRAINED")
        self._badge.setFixedSize(90, 28)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            "background:#1e2d45; color:#64748b; border-radius:4px;"
            "font-size:11px; font-weight:700; letter-spacing:1px;")
        lay.addWidget(self._badge)

    def update_model(self, model: Optional[PatMaxModel]):
        if model is None or not model.trained:
            self._name_lbl.setText("No model trained")
            self._name_lbl.setStyleSheet("color:#64748b; font-size:13px; font-weight:700;")
            self._info1.setText("")
            self._info2.setText("")
            self._hash_lbl.setText("")
            self._badge.setText("UNTRAINED")
            self._badge.setStyleSheet(
                "background:#1e2d45; color:#64748b; border-radius:4px;"
                "font-size:11px; font-weight:700;")
            self._thumb.setText("No Model")
            self._thumb.setPixmap(QPixmap())
            return

        self._name_lbl.setText("Model Trained ✔")
        self._name_lbl.setStyleSheet(
            "color:#39ff14; font-size:13px; font-weight:700;")
        x,y,w,h = model.train_roi or (0,0,0,0)
        self._info1.setText(
            f"ROI: ({x},{y})  {w}×{h} px   "
            f"Edges: {model.edge_count}")
        self._info2.setText(
            f"Origin: ({model.origin_x:.1f}, {model.origin_y:.1f})")
        self._hash_lbl.setText(f"Hash: {model.model_hash}")
        self._badge.setText("TRAINED")
        self._badge.setStyleSheet(
            "background:#1b4332; color:#39ff14; border-radius:4px;"
            "border:1px solid #39ff14; font-size:11px; font-weight:700;")

        # Thumbnail
        if model.thumbnail is not None:
            import cv2
            arr = model.thumbnail.copy()
            if len(arr.shape)==2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2]==3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h2,w2,ch = arr.shape
            qimg = QImage(arr.data.tobytes(), w2, h2, ch*w2, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg).scaled(
                78, 78, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._thumb.setPixmap(pix)
        else:
            self._thumb.setText("No thumb")


# ════════════════════════════════════════════════════════════════════
#  Result table
# ════════════════════════════════════════════════════════════════════
class ResultTable(QTableWidget):
    result_selected = Signal(int)   # row index

    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels(
            ["#", "Score", "X", "Y", "Angle (°)", "Scale"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget { background:#0a0e1a; color:#e2e8f0;
                           gridline-color:#1e2d45; border:1px solid #1e2d45;
                           font-size:11px; font-family:'Courier New'; }
            QTableWidget::item:selected { background:#1a2236; color:#00d4ff; }
            QTableWidget::item:alternate { background:#0d1220; }
            QHeaderView::section { background:#0d1220; color:#64748b;
                                   border:1px solid #1e2d45; padding:3px; }
        """)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self):
        rows = self.selectionModel().selectedRows()
        if rows:
            self.result_selected.emit(rows[0].row())

    def populate(self, results: List[PatMaxResult]):
        self.setRowCount(0)
        for i, r in enumerate(results):
            self.insertRow(i)
            col = "#39ff14" if r.score >= 0.7 else "#ffd700" if r.score >= 0.5 else "#ff3860"
            for j, (val, fmt) in enumerate([
                (str(i+1),         "{}"),
                (r.score,          "{:.4f}"),
                (r.origin_x,       "{:.2f}"),
                (r.origin_y,       "{:.2f}"),
                (r.angle,          "{:+.2f}"),
                (r.scale,          "{:.3f}"),
            ]):
                txt = fmt.format(val) if not isinstance(val, str) else val
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 1:   # score
                    item.setForeground(QColor(col))
                self.setItem(i, j, item)


# ════════════════════════════════════════════════════════════════════
#  Main PatMax Dialog
# ════════════════════════════════════════════════════════════════════
class PatMaxDialog(QDialog):
    model_trained   = Signal()    # emit khi train xong
    run_requested   = Signal(str) # node_id

    def __init__(self, node: NodeInstance, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._node    = node
        self._graph   = graph
        self._model: PatMaxModel = node.params.get("_patmax_model") or PatMaxModel()
        self._results: List[PatMaxResult] = []
        self._score_map_img: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None

        self.setWindowTitle(f"🎯  PatMax Pattern Align Tool  —  {node.tool.name}")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 820)
        self.setModal(False)
        self.setStyleSheet("""
            QDialog { background:#0a0e1a; color:#e2e8f0; }
            QGroupBox { border:1px solid #1e2d45; border-radius:6px;
                        margin-top:8px; padding-top:8px;
                        color:#64748b; font-size:11px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QScrollArea { border:none; }
            QLabel { color:#e2e8f0; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(52)
        hdr.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #16213e,stop:1 #0a0e1a);"
            "border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16,0,16,0)
        logo = QLabel("🎯  CogPatMaxPatternAlignTool")
        logo.setStyleSheet(
            "color:#00d4ff; font-size:14px; font-weight:700; "
            "letter-spacing:1px; background:transparent;")
        hl.addWidget(logo)
        hl.addStretch()

        # Status chip
        self._status_chip = QLabel("● UNTRAINED")
        self._status_chip.setStyleSheet(
            "color:#64748b; font-size:12px; font-weight:700; background:transparent;")
        hl.addWidget(self._status_chip)
        root.addWidget(hdr)

        # ── Main layout: left panel | right image ─────────────────
        main_spl = QSplitter(Qt.Horizontal)
        main_spl.setHandleWidth(1)
        main_spl.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(main_spl, 1)

        # ════ LEFT PANEL ══════════════════════════════════════════
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(10,10,10,10); ll.setSpacing(8)

        # Model preview
        ll.addWidget(QLabel("MODEL STATUS"))
        self._model_preview = ModelPreviewWidget()
        self._model_preview.update_model(self._model)
        ll.addWidget(self._model_preview)

        # Tabs: Train | Search | Settings
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#0a0e1a;color:#64748b;padding:7px 12px;
                         border:none;font-size:11px;font-weight:700;}
            QTabBar::tab:selected{color:#00d4ff;border-bottom:2px solid #00d4ff;}
            QTabBar::tab:hover{color:#e2e8f0;}
        """)

        self._tabs.addTab(self._build_train_tab(), "⚙ Train")
        self._tabs.addTab(self._build_search_tab(), "▶ Search")
        self._tabs.addTab(self._build_settings_tab(), "⚒ Settings")
        ll.addWidget(self._tabs)

        # Save/Load model buttons
        savload = QWidget()
        sl = QHBoxLayout(savload); sl.setContentsMargins(0,0,0,0); sl.setSpacing(6)
        btn_save = self._mk_btn("💾 Save Model", "#1b4332", "#39ff14")
        btn_load = self._mk_btn("📂 Load Model", "#0f3460", "#00d4ff")
        btn_save.clicked.connect(self._save_model)
        btn_load.clicked.connect(self._load_model)
        sl.addWidget(btn_save); sl.addWidget(btn_load)
        ll.addWidget(savload)

        # Result table
        ll.addWidget(QLabel("RESULTS"))
        self._result_table = ResultTable()
        self._result_table.setFixedHeight(140)
        self._result_table.result_selected.connect(self._on_result_selected)
        ll.addWidget(self._result_table)

        main_spl.addWidget(left)

        # ════ RIGHT PANEL — Image ════════════════════════════════
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6,6,6,6); rl.setSpacing(4)

        # Image toolbar
        img_tb = QWidget()
        img_tb.setFixedHeight(36)
        img_tb.setStyleSheet("background:#060a14; border-radius:4px;")
        itl = QHBoxLayout(img_tb); itl.setContentsMargins(8,4,8,4); itl.setSpacing(8)

        lbl_view = QLabel("VIEW:")
        lbl_view.setStyleSheet("color:#64748b; font-size:11px;")
        itl.addWidget(lbl_view)

        self._btn_view_image    = self._mk_small_btn("Image", active=True)
        self._btn_view_scoremap = self._mk_small_btn("Score Map")
        self._btn_view_edges    = self._mk_small_btn("Edge Model")
        self._btn_view_image.clicked.connect(lambda: self._set_view("image"))
        self._btn_view_scoremap.clicked.connect(lambda: self._set_view("scoremap"))
        self._btn_view_edges.clicked.connect(lambda: self._set_view("edges"))
        for b in (self._btn_view_image, self._btn_view_scoremap, self._btn_view_edges):
            itl.addWidget(b)

        itl.addSpacing(12)
        self._btn_refresh = self._mk_small_btn("🔄 Refresh")
        self._btn_refresh.setToolTip(
            "Lấy ảnh mới từ upstream — xoá result_vis/score_map cũ, "
            "giữ ROI + origin")
        self._btn_refresh.clicked.connect(self._on_refresh_click)
        itl.addWidget(self._btn_refresh)

        itl.addStretch()
        self._view_info = QLabel("")
        self._view_info.setStyleSheet("color:#1e2d45; font-size:10px; font-family:'Courier New';")
        itl.addWidget(self._view_info)
        rl.addWidget(img_tb)

        # Interactive image
        self._img_label = InteractiveImageLabel(mode="roi")
        self._img_label.roi_changed.connect(self._on_roi_drawn)
        self._img_label.origin_changed.connect(self._on_origin_dragged)
        self._img_label.shape_drawn.connect(self._on_shape_drawn)
        self._img_label.setMinimumSize(600, 400)
        rl.addWidget(self._img_label, 1)
        self._current_shape: str = "rect"
        self._current_shape_data: Optional[dict] = None

        # Status bar
        self._img_status = QLabel("Load image source → Run pipeline → Draw ROI → Train")
        self._img_status.setStyleSheet(
            "color:#64748b; font-size:11px; font-family:'Courier New';"
            "padding:4px 8px; background:#060a14; border-radius:3px;")
        rl.addWidget(self._img_status)

        main_spl.addWidget(right)
        main_spl.setSizes([340, 940])

        # ── Bottom bar ─────────────────────────────────────────────
        bot = QWidget(); bot.setFixedHeight(44)
        bot.setStyleSheet("background:#060a14; border-top:1px solid #1e2d45;")
        bl = QHBoxLayout(bot); bl.setContentsMargins(12,6,12,6); bl.setSpacing(8)

        self._btn_close = self._mk_btn("Close", "#1e2d45", "#94a3b8")
        self._btn_close.clicked.connect(self.close)
        bl.addWidget(self._btn_close)
        bl.addStretch()

        self._btn_run_pipeline = self._mk_btn("▶  Run Pipeline Node", "#0f3460", "#00d4ff")
        self._btn_run_pipeline.clicked.connect(self._run_pipeline_node)
        bl.addWidget(self._btn_run_pipeline)
        root.addWidget(bot)

        self._current_view = "image"
        self._current_roi: Optional[Tuple[int,int,int,int]] = None

        # Load existing model ROI if any
        if self._model.train_roi:
            self._current_roi = self._model.train_roi

        # Load image from upstream
        QTimer.singleShot(200, self._load_upstream_image)

    # ════════════════════════════════════════════════════════════════
    #  Build tabs
    # ════════════════════════════════════════════════════════════════
    def _build_train_tab(self) -> QWidget:
        inner = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)

        w = inner
        lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        hint = QLabel("1. Chọn loại shape ROI (Rect/Circle/Ellipse/Polygon)\n"
                       "2. Vẽ vùng Pattern trên ảnh — Polygon: click từng đỉnh,\n"
                       "    double-click để đóng, right-click để huỷ\n"
                       "3. Chỉnh tham số Train → ▶ Train Pattern")
        hint.setStyleSheet(
            "background:#0d1a2a; color:#ffd700; font-size:11px;"
            "padding:8px; border-radius:4px; line-height:1.5;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # Shape selector
        shape_grp = QGroupBox("ROI Shape")
        sg2 = QVBoxLayout(shape_grp); sg2.setContentsMargins(8, 22, 8, 8); sg2.setSpacing(6)
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(["Rectangle", "Circle", "Ellipse", "Polygon"])
        self._shape_combo.setStyleSheet(
            "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:3px 6px;border-radius:4px;}"
            "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
            "selection-background-color:#1a2236;}")
        self._shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        sg2.addWidget(self._shape_combo)
        lay.addWidget(shape_grp)

        # Origin setting — layout gọn 1 hàng combo + 1 hàng X/Y
        orig_grp = QGroupBox("Pattern Origin Point")
        og = QVBoxLayout(orig_grp); og.setContentsMargins(10, 26, 10, 10); og.setSpacing(8)

        # Hàng 1: Preset combo
        preset_row = QWidget(); pr_lay = QHBoxLayout(preset_row)
        pr_lay.setContentsMargins(0, 0, 0, 0); pr_lay.setSpacing(6)
        lbl_pre = QLabel("Preset:")
        lbl_pre.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl_pre.setMinimumWidth(46)
        self._origin_combo = QComboBox()
        self._origin_combo.addItems([
            "Center (50%, 50%)",
            "Top-Left (0%, 0%)",
            "Top-Center (50%, 0%)",
            "Top-Right (100%, 0%)",
            "Middle-Left (0%, 50%)",
            "Middle-Right (100%, 50%)",
            "Bottom-Center (50%, 100%)",
            "Custom...",
        ])
        self._origin_combo.setStyleSheet(
            "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:3px 6px;border-radius:4px;}"
            "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
            "selection-background-color:#1a2236;}")
        pr_lay.addWidget(lbl_pre); pr_lay.addWidget(self._origin_combo, 1)
        og.addWidget(preset_row)

        # Hàng 2: X / Y cùng dòng
        xy_row = QWidget(); xy_lay = QHBoxLayout(xy_row)
        xy_lay.setContentsMargins(0, 0, 0, 0); xy_lay.setSpacing(6)
        sp_style = ("QDoubleSpinBox{background:#0a0e1a;border:1px solid #1e2d45;"
                    "color:#e2e8f0;padding:2px 4px;border-radius:3px;}")
        lbl_x = QLabel("X:"); lbl_x.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._sp_origin_x = QDoubleSpinBox()
        self._sp_origin_x.setRange(-99999.0, 99999.0)
        self._sp_origin_x.setDecimals(2); self._sp_origin_x.setSingleStep(1.0)
        self._sp_origin_x.setStyleSheet(sp_style)
        lbl_y = QLabel("Y:"); lbl_y.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._sp_origin_y = QDoubleSpinBox()
        self._sp_origin_y.setRange(-99999.0, 99999.0)
        self._sp_origin_y.setDecimals(2); self._sp_origin_y.setSingleStep(1.0)
        self._sp_origin_y.setStyleSheet(sp_style)
        xy_lay.addWidget(lbl_x); xy_lay.addWidget(self._sp_origin_x, 1)
        xy_lay.addSpacing(6)
        xy_lay.addWidget(lbl_y); xy_lay.addWidget(self._sp_origin_y, 1)
        og.addWidget(xy_row)

        # Hint
        hint_o = QLabel("💡 Kéo dấu O vàng — có thể kéo ra ngoài ROI")
        hint_o.setStyleSheet("color:#ffd700; font-size:10px;")
        hint_o.setWordWrap(True)
        og.addWidget(hint_o)

        # Reset Image button
        btn_reset = QPushButton("🔄  Reset Image")
        btn_reset.setFixedHeight(28)
        btn_reset.setStyleSheet(
            "QPushButton{background:#1a2236;border:1px solid #1e2d45;"
            "border-radius:4px;color:#94a3b8;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#0f3460;color:#00d4ff;border-color:#00d4ff;}")
        btn_reset.clicked.connect(self._reset_image)
        og.addWidget(btn_reset)

        self._origin_combo.currentIndexChanged.connect(self._on_origin_preset_changed)
        self._sp_origin_x.valueChanged.connect(self._on_origin_spin_changed)
        self._sp_origin_y.valueChanged.connect(self._on_origin_spin_changed)
        self._origin_updating = False  # guard chống loop tín hiệu
        lay.addWidget(orig_grp)

        # Canny params
        canny_grp = QGroupBox("Edge Detection (Canny)")
        cg = QVBoxLayout(canny_grp); cg.setContentsMargins(8,12,8,8); cg.setSpacing(4)
        self._canny_low  = self._mk_spin("Low Threshold",  50,  0, 500, cg)
        self._canny_high = self._mk_spin("High Threshold", 150, 0, 500, cg)
        lay.addWidget(canny_grp)

        # Train Mode
        tm_grp = QGroupBox("Train Mode")
        tmg = QVBoxLayout(tm_grp); tmg.setContentsMargins(10, 26, 10, 10); tmg.setSpacing(6)
        self._train_mode_combo = QComboBox()
        self._train_mode_combo.addItems([
            "Evaluate DOFs At Runtime",
            "Create DOF Templates",
        ])
        self._train_mode_combo.setStyleSheet(
            "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:3px 6px;border-radius:4px;}"
            "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
            "selection-background-color:#1a2236;}")
        tmg.addWidget(self._train_mode_combo)
        tm_hint = QLabel(
            "• Evaluate DOFs At Runtime: train nhanh, search xử lý DOF khi run\n"
            "• Create DOF Templates: precompute mọi (angle, scale) — search\n"
            "  nhanh hơn nhưng train tốn thời gian + bộ nhớ")
        tm_hint.setStyleSheet("color:#94a3b8; font-size:10px;")
        tm_hint.setWordWrap(True)
        tmg.addWidget(tm_hint)
        lay.addWidget(tm_grp)

        # Train button
        self._btn_train = QPushButton("⚙  Train Pattern")
        self._btn_train.setFixedHeight(38)
        self._btn_train.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1b4332,stop:1 #2d6a4f);"
            "border:1px solid #39ff14;border-radius:5px;color:#39ff14;"
            "font-size:13px;font-weight:700;letter-spacing:1px;}"
            "QPushButton:hover{background:#2d6a4f;}"
            "QPushButton:disabled{background:#1e2d45;color:#1e2d45;border-color:#1e2d45;}")
        self._btn_train.clicked.connect(self._train)
        lay.addWidget(self._btn_train)

        self._train_status = QLabel("")
        self._train_status.setStyleSheet(
            "color:#64748b; font-size:11px; font-family:'Courier New';")
        self._train_status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._train_status)
        lay.addStretch()
        return scroll

    def _build_search_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        # Search button
        self._btn_search = QPushButton("▶  Run Search")
        self._btn_search.setFixedHeight(38)
        self._btn_search.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #0f3460,stop:1 #16213e);"
            "border:1px solid #00d4ff;border-radius:5px;color:#00d4ff;"
            "font-size:13px;font-weight:700;letter-spacing:1px;}"
            "QPushButton:hover{background:#16213e;}"
            "QPushButton:disabled{background:#1e2d45;color:#1e2d45;border-color:#1e2d45;}")
        self._btn_search.clicked.connect(self._run_search)
        lay.addWidget(self._btn_search)

        # Search results summary
        self._search_summary = QLabel("No search run yet.")
        self._search_summary.setStyleSheet(
            "background:#0d1220; color:#94a3b8; font-size:11px; "
            "font-family:'Courier New'; padding:8px; border-radius:4px;")
        self._search_summary.setWordWrap(True)
        lay.addWidget(self._search_summary)

        # Progress
        self._search_progress = QProgressBar()
        self._search_progress.setRange(0, 100)
        self._search_progress.setValue(0)
        self._search_progress.setFixedHeight(8)
        self._search_progress.setStyleSheet(
            "QProgressBar{background:#0a0e1a;border:none;border-radius:4px;}"
            "QProgressBar::chunk{background:#00d4ff;border-radius:4px;}")
        self._search_progress.hide()
        lay.addWidget(self._search_progress)

        # View score map button
        self._btn_show_score = QPushButton("🗺  View Score Map")
        self._btn_show_score.setFixedHeight(30)
        self._btn_show_score.setStyleSheet(
            "QPushButton{background:#111827;border:1px solid #1e2d45;"
            "border-radius:4px;color:#94a3b8;font-size:11px;}"
            "QPushButton:hover{background:#00d4ff22;color:#00d4ff;border-color:#00d4ff;}")
        self._btn_show_score.clicked.connect(lambda: self._set_view("scoremap"))
        lay.addWidget(self._btn_show_score)

        lay.addStretch()
        return w

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(w)

        lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        # Threshold
        thr_grp = QGroupBox("Acceptance Threshold")
        tg = QVBoxLayout(thr_grp); tg.setContentsMargins(8,12,8,8); tg.setSpacing(4)
        self._sp_threshold = self._mk_dspin("Min Score", 0.5, 0.0, 1.0, 0.01, tg)
        lay.addWidget(thr_grp)

        # Angle
        ang_grp = QGroupBox("Angle Search")
        ag = QVBoxLayout(ang_grp); ag.setContentsMargins(8,12,8,8); ag.setSpacing(4)
        self._chk_angle = QCheckBox("Enable angle search")
        self._chk_angle.setChecked(False)
        self._chk_angle.stateChanged.connect(self._on_angle_toggle)
        ag.addWidget(self._chk_angle)
        self._sp_ang_low  = self._mk_dspin("Angle Low (°)",  -30.0, -180, 180, 1.0, ag)
        self._sp_ang_high = self._mk_dspin("Angle High (°)",  30.0, -180, 180, 1.0, ag)
        self._sp_ang_step = self._mk_dspin("Angle Step (°)",   5.0,  0.5,  45, 0.5, ag)
        self._sp_ang_low.setEnabled(False)
        self._sp_ang_high.setEnabled(False)
        self._sp_ang_step.setEnabled(False)
        lay.addWidget(ang_grp)

        # Scale
        sc_grp = QGroupBox("Scale Search")
        sg = QVBoxLayout(sc_grp); sg.setContentsMargins(8,12,8,8); sg.setSpacing(4)
        self._chk_scale = QCheckBox("Enable scale search")
        self._chk_scale.setChecked(False)
        self._chk_scale.stateChanged.connect(self._on_scale_toggle)
        sg.addWidget(self._chk_scale)
        self._sp_sc_low  = self._mk_dspin("Scale Low",  0.9, 0.1, 3.0, 0.05, sg)
        self._sp_sc_high = self._mk_dspin("Scale High", 1.1, 0.1, 3.0, 0.05, sg)
        self._sp_sc_step = self._mk_dspin("Scale Step", 0.1, 0.01, 1.0, 0.01, sg)
        self._sp_sc_low.setEnabled(False)
        self._sp_sc_high.setEnabled(False)
        self._sp_sc_step.setEnabled(False)
        lay.addWidget(sc_grp)

        # Multi-result
        mr_grp = QGroupBox("Multi-Result")
        mg = QVBoxLayout(mr_grp); mg.setContentsMargins(8,12,8,8); mg.setSpacing(4)
        self._sp_num_results = self._mk_spin("Max Results",    1, 1, 50, mg)
        self._sp_overlap     = self._mk_dspin("Overlap Thresh",0.5, 0.0, 1.0, 0.05, mg)
        lay.addWidget(mr_grp)

        # Search region
        sr_grp = QGroupBox("Search Region (optional)")
        srg = QVBoxLayout(sr_grp); srg.setContentsMargins(8,12,8,8); srg.setSpacing(4)
        hint_sr = QLabel("0 = search entire image")
        hint_sr.setStyleSheet("color:#1e2d45; font-size:10px;")
        srg.addWidget(hint_sr)
        self._sp_sr_x = self._mk_spin("X",  0, 0, 8192, srg)
        self._sp_sr_y = self._mk_spin("Y",  0, 0, 8192, srg)
        self._sp_sr_w = self._mk_spin("W",  0, 0, 8192, srg)
        self._sp_sr_h = self._mk_spin("H",  0, 0, 8192, srg)
        lay.addWidget(sr_grp)
        lay.addStretch()

        outer = QWidget(); ol = QVBoxLayout(outer)
        ol.setContentsMargins(0,0,0,0); ol.addWidget(scroll)
        return outer

    # ════════════════════════════════════════════════════════════════
    #  Actions
    # ════════════════════════════════════════════════════════════════
    def _on_refresh_click(self):
        """Force refresh — luôn re-fetch và clear stale state."""
        self._image_sig = None  # bypass equality check
        if self._refresh_image_if_changed():
            self._set_view("image")
        else:
            self._img_status.setText("⚠ Không có ảnh upstream để refresh.")

    def _img_signature(self, arr) -> Optional[str]:
        """Hash ngắn cho ảnh (down-sample 16x16 + md5) — phát hiện ảnh mới."""
        if arr is None or not isinstance(arr, np.ndarray):
            return None
        try:
            import hashlib, cv2
            small = cv2.resize(arr, (16, 16))
            return hashlib.md5(small.tobytes()).hexdigest()[:12]
        except Exception:
            return f"{arr.shape}-{arr.dtype}"

    def _refresh_image_if_changed(self) -> bool:
        """Nếu ảnh upstream khác ảnh hiện tại → xoá kết quả cũ, cập nhật canvas.
        ROI + origin marker vẫn giữ lại (đọc từ model). Trả về True nếu đã refresh.
        """
        img = self._get_input_image()
        if img is None or not isinstance(img, np.ndarray):
            return False
        new_sig = self._img_signature(img)
        old_sig = getattr(self, "_image_sig", None)
        if new_sig == old_sig:
            return False
        self._image_sig = new_sig
        self._current_image = img
        # Xoá render cũ — không phụ thuộc ảnh
        self._results = []
        self._score_map_img = None
        self._result_vis = None
        if hasattr(self, "_result_table"):
            self._result_table.setRowCount(0)
        if hasattr(self, "_search_summary"):
            self._search_summary.setText("Image mới — chạy search lại.")
            self._search_summary.setStyleSheet(
                "background:#0d1220; color:#94a3b8; font-size:11px;"
                "font-family:'Courier New'; padding:8px; border-radius:4px;")
        # Cập nhật canvas: ảnh mới + restore ROI/origin từ model
        self._img_label.set_image(img)
        if self._model and self._model.train_roi:
            x, y, w2, h2 = self._model.train_roi
            st = getattr(self._model, "shape_type", "rect") or "rect"
            sd = getattr(self._model, "shape_data", None)
            self._img_label.set_shape_mode(st)
            if sd:
                self._img_label.set_shape_data(st, sd)
            else:
                self._img_label.set_rect_from_params(x, y, w2, h2)
            self._img_label.set_origin(x + self._model.origin_x,
                                        y + self._model.origin_y)
        else:
            self._img_label.set_origin(None, None)
        h, w = img.shape[:2]
        self._img_status.setText(
            f"🔄 Image refreshed: {w}×{h}  —  drawings cũ đã xoá")
        # View phải về raw image (vì result_vis đã clear)
        self._current_view = "image"
        return True

    def _load_upstream_image(self):
        """Lấy ảnh từ upstream node."""
        img = self._get_input_image()
        if img is not None and isinstance(img, np.ndarray):
            self._current_image = img
            self._image_sig = self._img_signature(img)
            self._img_label.set_image(img)
            h, w = img.shape[:2]
            self._img_status.setText(
                f"Image loaded: {w}×{h}  —  Kéo chuột để vẽ vùng Pattern")
            # Restore train_mode combo nếu model đã có
            tm = getattr(self._model, "train_mode", "evaluate") or "evaluate"
            self._train_mode_combo.setCurrentIndex(1 if tm == "create" else 0)
            # Restore ROI nếu có
            if self._model.train_roi:
                x,y,w2,h2 = self._model.train_roi
                st = getattr(self._model, "shape_type", "rect") or "rect"
                sd = getattr(self._model, "shape_data", None)
                self._current_shape = st
                self._current_shape_data = dict(sd) if sd else None
                shape_idx = {"rect":0,"circle":1,"ellipse":2,"polygon":3}.get(st, 0)
                self._shape_combo.setCurrentIndex(shape_idx)
                self._img_label.set_shape_mode(st)
                if sd:
                    QTimer.singleShot(100, lambda: self._img_label.set_shape_data(st, sd))
                else:
                    QTimer.singleShot(100, lambda: self._img_label.set_rect_from_params(x,y,w2,h2))
                # Restore origin marker từ model
                ox = x + float(self._model.origin_x)
                oy = y + float(self._model.origin_y)
                QTimer.singleShot(120, lambda: self._set_origin(ox, oy, from_user=False))
                QTimer.singleShot(125, lambda: self._origin_combo.setCurrentIndex(7))
        else:
            self._img_status.setText(
                "⚠  Không có ảnh.  "
                "Kết nối node Image Source → Run pipeline node trước.")

    def _on_roi_drawn(self, x, y, w, h):
        self._current_roi = (x, y, w, h)
        self._img_status.setText(
            f"ROI đã vẽ: ({x},{y})  {w}×{h} px  ({self._current_shape})  "
            f"—  Nhấn ⚙ Train Pattern")
        # Cập nhật điểm origin theo preset hiện tại trong ROI mới
        self._apply_origin_preset_to_roi()

    def _on_shape_drawn(self, shape_type: str, data: dict):
        self._current_shape = shape_type
        self._current_shape_data = dict(data)

    def _on_shape_changed(self, idx: int):
        m = {0: "rect", 1: "circle", 2: "ellipse", 3: "polygon"}
        s = m.get(idx, "rect")
        self._current_shape = s
        self._current_shape_data = None
        self._current_roi = None
        if hasattr(self, "_img_label"):
            self._img_label.set_shape_mode(s)
        hints = {
            "rect":     "Kéo chuột vẽ Rectangle",
            "circle":   "Kéo từ tâm ra rìa để vẽ Circle",
            "ellipse":  "Kéo bounding-box cho Ellipse",
            "polygon":  "Click từng đỉnh — double-click đóng, right-click huỷ",
        }
        self._img_status.setText(f"Shape: {s} — {hints[s]}")

    # ── Origin point handling ──────────────────────────────────────
    def _preset_offset(self, idx: int) -> Optional[Tuple[float, float]]:
        m = {
            0: (0.5, 0.5), 1: (0.0, 0.0), 2: (0.5, 0.0), 3: (1.0, 0.0),
            4: (0.0, 0.5), 5: (1.0, 0.5), 6: (0.5, 1.0),
        }
        return m.get(idx)   # 7 = Custom → None

    def _apply_origin_preset_to_roi(self):
        """Khi đổi preset hoặc vẽ ROI mới: đặt origin theo preset."""
        if self._current_roi is None:
            return
        idx = self._origin_combo.currentIndex()
        off = self._preset_offset(idx)
        if off is None:
            # Custom — giữ nguyên giá trị spinbox, chỉ đảm bảo trong ROI
            ox = self._sp_origin_x.value()
            oy = self._sp_origin_y.value()
        else:
            x, y, w, h = self._current_roi
            ox = x + w * off[0]
            oy = y + h * off[1]
        self._set_origin(ox, oy, from_user=False)

    def _set_origin(self, ox: float, oy: float, from_user: bool):
        """Cập nhật origin: spinboxes + canvas marker. from_user=True khi user nhập tay."""
        self._origin_updating = True
        try:
            self._sp_origin_x.setValue(float(ox))
            self._sp_origin_y.setValue(float(oy))
            self._img_label.set_origin(ox, oy)
        finally:
            self._origin_updating = False

    def _on_origin_preset_changed(self, idx: int):
        if self._preset_offset(idx) is not None:
            self._apply_origin_preset_to_roi()
        # Nếu Custom → không reset, để user kéo/nhập tay

    def _on_origin_spin_changed(self, _val):
        if self._origin_updating:
            return
        ox = self._sp_origin_x.value()
        oy = self._sp_origin_y.value()
        self._origin_updating = True
        try:
            self._img_label.set_origin(ox, oy)
            # Nếu user chỉnh tay → chuyển combo về Custom
            if self._origin_combo.currentIndex() != 7:
                self._origin_combo.setCurrentIndex(7)
        finally:
            self._origin_updating = False

    def _transform_origin(self, result) -> Tuple[float, float]:
        """Biến đổi origin (lưu trong model) sang vị trí match được."""
        import math
        m = self._model
        if not m or not m.train_roi:
            return float(result.x), float(result.y)
        w = float(m.pattern_w); h = float(m.pattern_h)
        # Offset origin so với tâm pattern (toạ độ pattern, có thể âm/>w,h)
        dx = float(m.origin_x) - w / 2.0
        dy = float(m.origin_y) - h / 2.0
        rad = math.radians(-float(result.angle))
        ca = math.cos(rad); sa = math.sin(rad)
        s  = float(result.scale) if result.scale else 1.0
        ox = float(result.x) + s * (dx * ca - dy * sa)
        oy = float(result.y) + s * (dx * sa + dy * ca)
        return ox, oy

    def _reset_image(self):
        """Reload ảnh từ upstream, xoá ROI/origin/results/score-map hiện tại."""
        # Clear state
        self._current_roi = None
        self._current_shape_data = None
        self._results = []
        self._score_map_img = None
        if hasattr(self, "_result_vis"):
            self._result_vis = None
        # Clear canvas markers
        self._img_label.set_origin(None, None)
        self._img_label.set_shape_mode(self._current_shape)  # clears _rect/_poly/_data
        # Reset origin spinboxes & combo
        self._origin_updating = True
        try:
            self._sp_origin_x.setValue(0.0)
            self._sp_origin_y.setValue(0.0)
            self._origin_combo.setCurrentIndex(0)
        finally:
            self._origin_updating = False
        # Reload ảnh upstream
        img = self._get_input_image()
        if img is not None and isinstance(img, np.ndarray):
            self._current_image = img
            self._img_label.set_image(img)
            h, w = img.shape[:2]
            self._img_status.setText(
                f"🔄 Image reset: {w}×{h}  —  Vẽ ROI mới để train")
        else:
            self._img_status.setText("⚠ Không có ảnh upstream để reset.")
        # Clear search summary
        if hasattr(self, "_search_summary"):
            self._search_summary.setText("Image reset — chạy search lại nếu cần.")
            self._search_summary.setStyleSheet(
                "background:#0d1220; color:#94a3b8; font-size:11px;"
                "font-family:'Courier New'; padding:8px; border-radius:4px;")
        if hasattr(self, "_result_table"):
            self._result_table.setRowCount(0)
        self._set_view("image")

    def _on_origin_dragged(self, ox: float, oy: float):
        """Callback khi user kéo origin trên canvas."""
        self._origin_updating = True
        try:
            self._sp_origin_x.setValue(float(ox))
            self._sp_origin_y.setValue(float(oy))
            if self._origin_combo.currentIndex() != 7:
                self._origin_combo.setCurrentIndex(7)
        finally:
            self._origin_updating = False

    def _train(self):
        if self._current_image is None:
            QMessageBox.warning(self, "Train", "Chưa có ảnh. Hãy chạy pipeline trước.")
            return
        if self._current_roi is None:
            QMessageBox.warning(self, "Train",
                                "Hãy kéo chuột trên ảnh để vẽ vùng Pattern trước.")
            return

        x, y, w, h = self._current_roi
        if w < 8 or h < 8:
            QMessageBox.warning(self, "Train", "Vùng ROI quá nhỏ (min 8×8 px).")
            return

        # Origin offset — tính từ toạ độ origin (x,y) thực tế trên ảnh.
        # KHÔNG clamp: cho phép origin nằm ngoài ROI (offset âm hoặc >1).
        idx = self._origin_combo.currentIndex()
        preset = self._preset_offset(idx)
        if preset is not None:
            origin_off = preset
        else:
            ox = self._sp_origin_x.value()
            oy = self._sp_origin_y.value()
            denom_w = float(w) if w > 0 else 1.0
            denom_h = float(h) if h > 0 else 1.0
            fx = (ox - x) / denom_w
            fy = (oy - y) / denom_h
            origin_off = (fx, fy)

        self._btn_train.setEnabled(False)
        self._train_status.setText("Training...")
        self._train_status.setStyleSheet("color:#ffd700; font-size:11px;")

        try:
            tm = "create" if self._train_mode_combo.currentIndex() == 1 else "evaluate"
            ang_low_t  = self._sp_ang_low.value()  if self._chk_angle.isChecked() else 0.0
            ang_high_t = self._sp_ang_high.value() if self._chk_angle.isChecked() else 0.0
            sc_low_t   = self._sp_sc_low.value()   if self._chk_scale.isChecked() else 1.0
            sc_high_t  = self._sp_sc_high.value()  if self._chk_scale.isChecked() else 1.0
            model = train_patmax(
                self._current_image,
                self._current_roi,
                origin_offset=origin_off,
                canny_low=self._canny_low.value(),
                canny_high=self._canny_high.value(),
                shape_type=self._current_shape,
                shape_data=self._current_shape_data,
                train_mode=tm,
                angle_low=ang_low_t, angle_high=ang_high_t,
                angle_step=self._sp_ang_step.value(),
                scale_low=sc_low_t, scale_high=sc_high_t,
                scale_step=self._sp_sc_step.value(),
            )
            # Lưu thêm các params còn lại
            model.accept_threshold = self._sp_threshold.value()
            model.num_results = self._sp_num_results.value()

            self._model = model
            self._node.params["_patmax_model"] = model

            # Cập nhật marker origin (đã trained) — toạ độ tuyệt đối ảnh
            self._set_origin(x + model.origin_x, y + model.origin_y, from_user=False)

            self._model_preview.update_model(model)
            self._status_chip.setText(f"● TRAINED  hash:{model.model_hash}")
            self._status_chip.setStyleSheet(
                "color:#39ff14; font-size:12px; font-weight:700; background:transparent;")

            self._train_status.setText(
                f"✔ Trained OK  |  {model.edge_count} edge pixels  |  "
                f"Pattern {model.pattern_w}×{model.pattern_h}")
            self._train_status.setStyleSheet("color:#39ff14; font-size:11px;")

            # Show edge model
            self._set_view("edges")
            self.model_trained.emit()

        except Exception as e:
            self._train_status.setText(f"✖ Error: {e}")
            self._train_status.setStyleSheet("color:#ff3860; font-size:11px;")
        finally:
            self._btn_train.setEnabled(True)

    def _run_search(self):
        if not self._model.is_valid():
            QMessageBox.warning(self, "Search", "Chưa train model.")
            return
        # Tự refresh nếu upstream đổi ảnh
        self._refresh_image_if_changed()
        img = self._current_image if self._current_image is not None else self._get_input_image()
        if img is None:
            QMessageBox.warning(self, "Search", "Không có ảnh để search.")
            return
        self._current_image = img
        self._image_sig = self._img_signature(img)

        self._search_progress.show()
        self._search_progress.setValue(30)
        self._btn_search.setEnabled(False)

        try:
            ang_low  = self._sp_ang_low.value()  if self._chk_angle.isChecked() else 0.0
            ang_high = self._sp_ang_high.value() if self._chk_angle.isChecked() else 0.0
            sc_low   = self._sp_sc_low.value()   if self._chk_scale.isChecked() else 1.0
            sc_high  = self._sp_sc_high.value()  if self._chk_scale.isChecked() else 1.0

            self._search_progress.setValue(50)

            results, score_map_vis = run_patmax(
                img, self._model,
                accept_threshold=self._sp_threshold.value(),
                angle_low=ang_low, angle_high=ang_high,
                angle_step=self._sp_ang_step.value(),
                scale_low=sc_low, scale_high=sc_high,
                scale_step=self._sp_sc_step.value(),
                num_results=self._sp_num_results.value(),
                overlap_threshold=self._sp_overlap.value(),
            )

            self._search_progress.setValue(80)
            self._results = results
            self._score_map_img = score_map_vis

            # Draw results on image
            result_vis = draw_patmax_results(img, results, self._model)
            # Save for display
            self._result_vis = result_vis

            self._result_table.populate(results)

            n = len(results)
            if n > 0:
                best = results[0]
                self._search_summary.setText(
                    f"Found: {n} result(s)\n"
                    f"Best: score={best.score:.4f}  "
                    f"origin=({best.origin_x:.2f}, {best.origin_y:.2f})  "
                    f"angle={best.angle:+.2f}°")
                self._search_summary.setStyleSheet(
                    "background:#0d2a1a; color:#39ff14; font-size:11px;"
                    "font-family:'Courier New'; padding:8px; border-radius:4px;"
                    "border:1px solid #39ff1433;")
                # Cập nhật suggested threshold
                self._sp_threshold.setValue(round(best.score * 0.85, 3))
            else:
                # Tìm best score từ console log (chạy lại với threshold=0 để biết)
                results_all, _ = run_patmax(
                    img, self._model,
                    accept_threshold=0.0,
                    angle_low=ang_low, angle_high=ang_high,
                    angle_step=self._sp_ang_step.value(),
                    scale_low=sc_low, scale_high=sc_high,
                    scale_step=self._sp_sc_step.value(),
                    num_results=1,
                )
                if results_all:
                    best_s = results_all[0].score
                    suggest = round(best_s * 0.80, 3)
                    self._search_summary.setText(
                        f"NOT FOUND above threshold={self._sp_threshold.value():.3f}\n"
                        f"Best score found: {best_s:.4f}\n"
                        f"➡ Suggested threshold: {suggest:.3f}\n"
                        f"Click 'Apply Suggest' hoặc giảm threshold.")
                    self._search_summary.setStyleSheet(
                        "background:#2a1a0d; color:#ffd700; font-size:11px;"
                        "font-family:'Courier New'; padding:8px; border-radius:4px;"
                        "border:1px solid #ffd70033;")
                    # Auto-fill suggested threshold
                    self._sp_threshold.setValue(suggest)
                    self._img_status.setText(
                        f"⚠ Threshold tự động điều chỉnh xuống {suggest:.3f} — "
                        f"nhấn Run Search lại")
                else:
                    self._search_summary.setText(
                        "NOT FOUND\nModel không match với ảnh này.\n"
                        "Thử: retrain với ảnh này, hoặc giảm Canny threshold khi train.")
                    self._search_summary.setStyleSheet(
                        "background:#2a0d0d; color:#ff3860; font-size:11px;"
                        "font-family:'Courier New'; padding:8px; border-radius:4px;"
                        "border:1px solid #ff386033;")

            # Update node outputs
            node = self._node
            node.outputs["found"]     = n > 0
            node.outputs["num_found"] = n
            node.outputs["image"]     = result_vis
            if n > 0:
                best = results[0]
                # x, y = toạ độ điểm tham chiếu (yellow) — chính là output chính
                node.outputs["score"]    = best.score
                node.outputs["x"]        = best.origin_x
                node.outputs["y"]        = best.origin_y
                node.outputs["origin_x"] = best.origin_x
                node.outputs["origin_y"] = best.origin_y
                node.outputs["center_x"] = best.x
                node.outputs["center_y"] = best.y
                node.outputs["angle"]    = best.angle
                node.outputs["scale"]    = best.scale
                # Cập nhật marker tham chiếu trên ảnh kết quả (theo template)
                self._set_origin(best.origin_x, best.origin_y, from_user=False)
            node.status = "pass" if n > 0 else "fail"

            self._set_view("image")
            self.run_requested.emit(node.node_id)

        except Exception as e:
            self._search_summary.setText(f"Error: {e}")
            self._search_summary.setStyleSheet(
                "background:#2a0d0d; color:#ff3860; font-size:11px; padding:8px;")
        finally:
            self._search_progress.setValue(100)
            QTimer.singleShot(500, self._search_progress.hide)
            self._btn_search.setEnabled(True)

    def _on_result_selected(self, row: int):
        if row < 0 or row >= len(self._results):
            return
        # Highlight selected result
        r = self._results[row]
        self._img_status.setText(
            f"Result #{row+1}:  score={r.score:.4f}  "
            f"x={r.x:.2f}  y={r.y:.2f}  "
            f"angle={r.angle:+.2f}°  scale={r.scale:.3f}")

    def _set_view(self, view: str):
        self._current_view = view
        btns = {
            "image": self._btn_view_image,
            "scoremap": self._btn_view_scoremap,
            "edges": self._btn_view_edges,
        }
        active_style = (
            "QPushButton{background:#0f3460;border:1px solid #00d4ff;"
            "border-radius:3px;color:#00d4ff;font-size:11px;padding:2px 8px;}")
        idle_style = (
            "QPushButton{background:#111827;border:1px solid #1e2d45;"
            "border-radius:3px;color:#64748b;font-size:11px;padding:2px 8px;}"
            "QPushButton:hover{color:#e2e8f0;}")
        for k, b in btns.items():
            b.setStyleSheet(active_style if k == view else idle_style)

        if view == "image":
            _rv = getattr(self, "_result_vis", None)
            img = _rv if _rv is not None else self._current_image
            if img is not None:
                self._img_label.set_image(img)
            # Khi đang xem result_vis: marker origin đã vẽ trong ảnh, ẩn overlay
            # để không chồng. Khi xem raw image: bật overlay theo spinbox hiện tại.
            if _rv is not None:
                self._img_label.set_origin(None, None)
            else:
                self._img_label.set_origin(self._sp_origin_x.value(),
                                           self._sp_origin_y.value())
            self._view_info.setText("Output image with results")
        elif view == "scoremap":
            self._img_label.set_origin(None, None)
            img = getattr(self, "_score_map_img", None)
            if img is not None:
                self._img_label.set_image(img)
                self._view_info.setText("Score heatmap (red=high match)")
            else:
                self._img_label.set_image(None)
                self._view_info.setText("Run search first")
        elif view == "edges":
            self._img_label.set_origin(None, None)
            if self._model.is_valid() and self._model.edge_image is not None:
                import cv2
                edge_vis = cv2.cvtColor(self._model.edge_image, cv2.COLOR_GRAY2BGR)
                # Colorize edges
                edge_vis[self._model.edge_image > 0] = [0, 220, 80]
                # Overlay model size info
                cv2.putText(edge_vis,
                    f"Pattern {self._model.pattern_w}×{self._model.pattern_h}  "
                    f"Edges:{self._model.edge_count}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 1)
                self._img_label.set_image(edge_vis)
                self._view_info.setText("Edge model (green=boundaries used for matching)")
            else:
                self._img_label.set_image(None)
                self._view_info.setText("Train model first")

    def _save_model(self):
        if not self._model.is_valid():
            QMessageBox.warning(self, "Save", "Chưa có model để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PatMax Model", "model.patmax",
            "PatMax Model (*.patmax);;All Files (*)")
        if path:
            try:
                save_model(self._model, path)
                QMessageBox.information(self, "Save", f"Model saved:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load PatMax Model", "",
            "PatMax Model (*.patmax);;JSON (*.json);;All Files (*)")
        if path:
            model = load_model(path)
            if model and model.is_valid():
                self._model = model
                self._node.params["_patmax_model"] = model
                self._model_preview.update_model(model)
                # Restore ROI + origin marker + shape từ model
                if model.train_roi:
                    rx, ry, rw, rh = model.train_roi
                    self._current_roi = model.train_roi
                    st = getattr(model, "shape_type", "rect") or "rect"
                    sd = getattr(model, "shape_data", None)
                    self._current_shape = st
                    self._current_shape_data = dict(sd) if sd else None
                    self._shape_combo.setCurrentIndex(
                        {"rect":0,"circle":1,"ellipse":2,"polygon":3}.get(st, 0))
                    self._img_label.set_shape_mode(st)
                    if sd:
                        self._img_label.set_shape_data(st, sd)
                    else:
                        self._img_label.set_rect_from_params(rx, ry, rw, rh)
                    self._set_origin(rx + model.origin_x, ry + model.origin_y,
                                     from_user=False)
                    self._origin_combo.setCurrentIndex(7)
                self._status_chip.setText(f"● LOADED  hash:{model.model_hash}")
                self._status_chip.setStyleSheet(
                    "color:#00d4ff; font-size:12px; font-weight:700; background:transparent;")
                self._set_view("edges")
                QMessageBox.information(self, "Load",
                    f"Model loaded OK\n"
                    f"Pattern: {model.pattern_w}×{model.pattern_h}  "
                    f"Edges: {model.edge_count}")
            else:
                QMessageBox.critical(self, "Error", "Cannot load model or model invalid.")

    def _run_pipeline_node(self):
        """Chạy pipeline node PatMax (dùng model đã train)."""
        node = self._node
        # Refresh nếu ảnh upstream đã đổi
        self._refresh_image_if_changed()
        img  = self._current_image if self._current_image is not None else self._get_input_image()
        if img is None:
            QMessageBox.warning(self, "Run", "Không có ảnh.")
            return
        # Ensure model is in params
        node.params["_patmax_model"] = self._model
        # Call engine directly
        if self._model.is_valid():
            self._run_search()
        else:
            QMessageBox.warning(self, "Run", "Train model trước.")

    def _on_angle_toggle(self, state):
        en = bool(state)
        self._sp_ang_low.setEnabled(en)
        self._sp_ang_high.setEnabled(en)
        self._sp_ang_step.setEnabled(en)

    def _on_scale_toggle(self, state):
        en = bool(state)
        self._sp_sc_low.setEnabled(en)
        self._sp_sc_high.setEnabled(en)
        self._sp_sc_step.setEnabled(en)

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════
    def _get_input_image(self) -> Optional[np.ndarray]:
        node = self._node
        # Từ output của chính node (đã chạy)
        img = node.outputs.get("image")
        if img is not None and isinstance(img, np.ndarray):
            return img
        # Từ upstream
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id and conn.dst_port == "image":
                src = self._graph.nodes.get(conn.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _mk_btn(self, text, bg, fg) -> QPushButton:
        b = QPushButton(text)
        b.setFixedHeight(32)
        b.setStyleSheet(
            f"QPushButton{{background:{bg};border:1px solid {fg}33;"
            f"border-radius:4px;color:{fg};font-size:12px;font-weight:600;}}"
            f"QPushButton:hover{{background:{fg}22;border-color:{fg};}}")
        return b

    def _mk_small_btn(self, text, active=False) -> QPushButton:
        b = QPushButton(text)
        b.setFixedHeight(26)
        active_s = ("QPushButton{background:#0f3460;border:1px solid #00d4ff;"
                    "border-radius:3px;color:#00d4ff;font-size:11px;padding:0 8px;}")
        idle_s = ("QPushButton{background:#111827;border:1px solid #1e2d45;"
                  "border-radius:3px;color:#64748b;font-size:11px;padding:0 8px;}"
                  "QPushButton:hover{color:#e2e8f0;}")
        b.setStyleSheet(active_s if active else idle_s)
        return b

    def _mk_spin(self, label, default, mn, mx, parent_lay) -> QSpinBox:
        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
        lbl = QLabel(label); lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(100)
        sp = QSpinBox()
        sp.setRange(mn, mx); sp.setValue(default)
        sp.setStyleSheet("QSpinBox{background:#0a0e1a;border:1px solid #1e2d45;"
                         "color:#e2e8f0;padding:2px 4px;border-radius:3px;}")
        rl.addWidget(lbl); rl.addWidget(sp)
        parent_lay.addWidget(row)
        return sp

    def _mk_dspin(self, label, default, mn, mx, step, parent_lay) -> QDoubleSpinBox:
        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
        lbl = QLabel(label); lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(100)
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx); sp.setValue(default); sp.setSingleStep(step)
        sp.setDecimals(3)
        sp.setStyleSheet("QDoubleSpinBox{background:#0a0e1a;border:1px solid #1e2d45;"
                         "color:#e2e8f0;padding:2px 4px;border-radius:3px;}")
        rl.addWidget(lbl); rl.addWidget(sp)
        parent_lay.addWidget(row)
        return sp
