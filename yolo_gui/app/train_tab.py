"""
Tab huấn luyện YOLO - Tab chính của ứng dụng
Cung cấp đầy đủ tham số huấn luyện và theo dõi tiến trình
"""
import os
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QScrollArea, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer

from app.widgets.log_viewer import LogViewer
from app.widgets.metrics_viewer import MetricsViewer
from app.widgets.progress_widget import ProgressWidget
from app.utils.yolo_utils import get_available_models, get_device_info, TASK_MAP
from app.utils.config import get_default_config, load_config, save_config
from app.workers.train_worker import TrainWorker


class TrainTab(QWidget):
    """Tab cấu hình và thực hiện huấn luyện YOLO."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[TrainWorker] = None
        self._train_start_time: float = 0.0
        self._best_metric_score = 0.0
        self._setup_ui()
        self._load_device_info()

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # === Panel trái (cấu hình) ===
        left_panel = self._build_left_panel()
        main_layout.addWidget(left_panel, 35)  # 35% chiều rộng

        # === Panel phải (log + biểu đồ) ===
        right_panel = self._build_right_panel()
        main_layout.addWidget(right_panel, 65)

    def _build_left_panel(self) -> QScrollArea:
        """Xây dựng panel trái chứa tất cả tham số."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Task & Model ---
        layout.addWidget(self._build_task_model_group())
        # --- Dataset ---
        layout.addWidget(self._build_dataset_group())
        # --- Hyperparameters ---
        layout.addWidget(self._build_hyperparams_group())
        # --- Augmentation ---
        layout.addWidget(self._build_augmentation_group())
        # --- Device ---
        layout.addWidget(self._build_device_group())
        # --- Output ---
        layout.addWidget(self._build_output_group())
        # --- Buttons ---
        layout.addWidget(self._build_action_buttons())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_task_model_group(self) -> QGroupBox:
        grp = QGroupBox("🎯 Task & Model")
        form = QFormLayout(grp)
        form.setSpacing(8)

        # Task selection
        self.combo_task = QComboBox()
        self.combo_task.addItems([
            "Classification", "Object Detection",
            "OBB Detection", "Segmentation"
        ])
        self.combo_task.setCurrentText("Object Detection")
        self.combo_task.currentTextChanged.connect(self.on_task_changed)
        form.addRow("Task:", self.combo_task)

        # Model selection
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        self.combo_model = QComboBox()
        self.combo_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_browse_model = QPushButton("📁")
        btn_browse_model.setMaximumWidth(36)
        btn_browse_model.setToolTip("Chọn file model (.pt)")
        btn_browse_model.clicked.connect(self.browse_model)
        model_layout.addWidget(self.combo_model)
        model_layout.addWidget(btn_browse_model)
        form.addRow("Model:", model_row)

        # Khởi tạo danh sách model
        self.update_model_list("Object Detection")
        return grp

    def _build_dataset_group(self) -> QGroupBox:
        grp = QGroupBox("📂 Dataset")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        # Dataset config file
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        self.edit_dataset = QLineEdit()
        self.edit_dataset.setPlaceholderText("Chọn file dataset.yaml...")
        btn_browse_ds = QPushButton("📁 Browse")
        btn_browse_ds.setMaximumWidth(90)
        btn_browse_ds.clicked.connect(self.browse_dataset)
        row_layout.addWidget(self.edit_dataset)
        row_layout.addWidget(btn_browse_ds)
        layout.addWidget(QLabel("File cấu hình dataset (.yaml):"))
        layout.addWidget(row)

        # Dataset info
        self._lbl_dataset_info = QLabel("Chưa chọn dataset")
        self._lbl_dataset_info.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._lbl_dataset_info.setWordWrap(True)
        layout.addWidget(self._lbl_dataset_info)

        return grp

    def _build_hyperparams_group(self) -> QGroupBox:
        grp = QGroupBox("⚙️ Siêu tham số Huấn luyện")
        form = QFormLayout(grp)
        form.setSpacing(6)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.spin_epochs.setToolTip("Số epoch huấn luyện")
        form.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(16)
        self.spin_batch.setToolTip("Kích thước batch (-1 = auto)")
        form.addRow("Batch Size:", self.spin_batch)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 1280)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setToolTip("Kích thước ảnh đầu vào")
        form.addRow("Image Size:", self.spin_imgsz)

        self.dspin_lr = QDoubleSpinBox()
        self.dspin_lr.setRange(0.0001, 0.1)
        self.dspin_lr.setValue(0.01)
        self.dspin_lr.setSingleStep(0.001)
        self.dspin_lr.setDecimals(4)
        self.dspin_lr.setToolTip("Learning rate ban đầu")
        form.addRow("Learning Rate:", self.dspin_lr)

        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["SGD", "Adam", "AdamW", "auto"])
        self.combo_optimizer.setToolTip("Thuật toán tối ưu")
        form.addRow("Optimizer:", self.combo_optimizer)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(8)
        self.spin_workers.setToolTip("Số luồng tải dữ liệu")
        form.addRow("Workers:", self.spin_workers)

        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 100)
        self.spin_patience.setValue(50)
        self.spin_patience.setToolTip("Early stopping patience")
        form.addRow("Patience:", self.spin_patience)

        self.dspin_weight_decay = QDoubleSpinBox()
        self.dspin_weight_decay.setRange(0.0, 0.1)
        self.dspin_weight_decay.setValue(0.0005)
        self.dspin_weight_decay.setSingleStep(0.0001)
        self.dspin_weight_decay.setDecimals(5)
        self.dspin_weight_decay.setToolTip("Hệ số weight decay (L2)")
        form.addRow("Weight Decay:", self.dspin_weight_decay)

        return grp

    def _build_augmentation_group(self) -> QGroupBox:
        grp = QGroupBox("🎨 Augmentation")
        form = QFormLayout(grp)
        form.setSpacing(6)

        self.dspin_hsv_h = QDoubleSpinBox()
        self.dspin_hsv_h.setRange(0.0, 1.0); self.dspin_hsv_h.setValue(0.015); self.dspin_hsv_h.setDecimals(3)
        form.addRow("HSV Hue:", self.dspin_hsv_h)

        self.dspin_hsv_s = QDoubleSpinBox()
        self.dspin_hsv_s.setRange(0.0, 1.0); self.dspin_hsv_s.setValue(0.7); self.dspin_hsv_s.setDecimals(3)
        form.addRow("HSV Saturation:", self.dspin_hsv_s)

        self.dspin_hsv_v = QDoubleSpinBox()
        self.dspin_hsv_v.setRange(0.0, 1.0); self.dspin_hsv_v.setValue(0.4); self.dspin_hsv_v.setDecimals(3)
        form.addRow("HSV Value:", self.dspin_hsv_v)

        self.dspin_degrees = QDoubleSpinBox()
        self.dspin_degrees.setRange(0.0, 180.0); self.dspin_degrees.setValue(0.0); self.dspin_degrees.setDecimals(1)
        form.addRow("Rotation (°):", self.dspin_degrees)

        self.dspin_translate = QDoubleSpinBox()
        self.dspin_translate.setRange(0.0, 0.9); self.dspin_translate.setValue(0.1); self.dspin_translate.setDecimals(2)
        form.addRow("Translation:", self.dspin_translate)

        self.dspin_scale = QDoubleSpinBox()
        self.dspin_scale.setRange(0.0, 0.9); self.dspin_scale.setValue(0.5); self.dspin_scale.setDecimals(2)
        form.addRow("Scale:", self.dspin_scale)

        self.chk_flipud = QCheckBox("Lật dọc (flipud)")
        self.chk_fliplr = QCheckBox("Lật ngang (fliplr)")
        self.chk_fliplr.setChecked(True)
        form.addRow("Flip:", self.chk_flipud)
        form.addRow("", self.chk_fliplr)

        self.dspin_mosaic = QDoubleSpinBox()
        self.dspin_mosaic.setRange(0.0, 1.0); self.dspin_mosaic.setValue(1.0); self.dspin_mosaic.setDecimals(2)
        form.addRow("Mosaic:", self.dspin_mosaic)

        self.dspin_mixup = QDoubleSpinBox()
        self.dspin_mixup.setRange(0.0, 1.0); self.dspin_mixup.setValue(0.0); self.dspin_mixup.setDecimals(2)
        form.addRow("Mixup:", self.dspin_mixup)

        return grp

    def _build_device_group(self) -> QGroupBox:
        grp = QGroupBox("💻 Thiết bị")
        form = QFormLayout(grp)
        form.setSpacing(6)

        self.combo_device = QComboBox()
        self.combo_device.addItem("auto")
        form.addRow("Device:", self.combo_device)

        self._lbl_device_info = QLabel("Đang kiểm tra thiết bị...")
        self._lbl_device_info.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._lbl_device_info.setWordWrap(True)
        form.addRow("", self._lbl_device_info)

        return grp

    def _build_output_group(self) -> QGroupBox:
        grp = QGroupBox("💾 Output")
        form = QFormLayout(grp)
        form.setSpacing(6)

        # Project directory
        proj_row = QWidget()
        proj_layout = QHBoxLayout(proj_row)
        proj_layout.setContentsMargins(0, 0, 0, 0)
        proj_layout.setSpacing(4)
        self.edit_project = QLineEdit("runs")
        btn_browse_proj = QPushButton("📁")
        btn_browse_proj.setMaximumWidth(36)
        btn_browse_proj.clicked.connect(self.browse_output)
        proj_layout.addWidget(self.edit_project)
        proj_layout.addWidget(btn_browse_proj)
        form.addRow("Project Dir:", proj_row)

        self.edit_exp_name = QLineEdit("train")
        self.edit_exp_name.setPlaceholderText("Tên thí nghiệm...")
        form.addRow("Experiment:", self.edit_exp_name)

        self.spin_save_period = QSpinBox()
        self.spin_save_period.setRange(-1, 100)
        self.spin_save_period.setValue(-1)
        self.spin_save_period.setToolTip("-1 = chỉ lưu best/last")
        form.addRow("Save Period:", self.spin_save_period)

        # Resume
        resume_row = QWidget()
        resume_layout = QHBoxLayout(resume_row)
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_layout.setSpacing(4)
        self.chk_resume = QCheckBox("Resume")
        self.edit_resume_path = QLineEdit()
        self.edit_resume_path.setPlaceholderText("Chọn checkpoint để resume...")
        self.edit_resume_path.setEnabled(False)
        btn_browse_resume = QPushButton("📁")
        btn_browse_resume.setMaximumWidth(36)
        btn_browse_resume.clicked.connect(self._browse_resume)
        self.chk_resume.toggled.connect(self.edit_resume_path.setEnabled)
        self.chk_resume.toggled.connect(btn_browse_resume.setEnabled)
        resume_layout.addWidget(self.chk_resume)
        resume_layout.addWidget(self.edit_resume_path)
        resume_layout.addWidget(btn_browse_resume)
        form.addRow("Resume:", resume_row)

        return grp

    def _build_action_buttons(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        # Hàng 1: Start / Stop
        row1 = QHBoxLayout()
        self.btn_start_train = QPushButton("🚀 Bắt đầu Huấn luyện")
        self.btn_start_train.setObjectName("btn_start_train")
        self.btn_start_train.setMinimumHeight(36)
        self.btn_start_train.clicked.connect(self.start_training)

        self.btn_stop_train = QPushButton("⏹️ Dừng")
        self.btn_stop_train.setObjectName("btn_stop_train")
        self.btn_stop_train.setMinimumHeight(36)
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self.stop_training)

        row1.addWidget(self.btn_start_train)
        row1.addWidget(self.btn_stop_train)
        layout.addLayout(row1)

        # Hàng 2: Save / Load Config
        row2 = QHBoxLayout()
        btn_save_cfg = QPushButton("💾 Lưu Config")
        btn_save_cfg.clicked.connect(self.save_config)
        btn_load_cfg = QPushButton("📂 Tải Config")
        btn_load_cfg.clicked.connect(self.load_config)
        row2.addWidget(btn_save_cfg)
        row2.addWidget(btn_load_cfg)
        layout.addLayout(row2)

        return container

    def _build_right_panel(self) -> QSplitter:
        """Xây dựng panel phải: log + progress + metrics."""
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Phần trên: Log + Progress (ngang)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # Log viewer
        log_container = QGroupBox("📋 Training Log")
        log_layout = QVBoxLayout(log_container)
        self.log_viewer = LogViewer()
        log_layout.addWidget(self.log_viewer)
        top_layout.addWidget(log_container, 60)

        # Progress widget
        self.progress_widget = ProgressWidget()
        top_layout.addWidget(self.progress_widget, 40)

        splitter.addWidget(top_widget)

        # Phần dưới: Metrics charts
        metrics_container = QGroupBox("📊 Metrics")
        metrics_layout = QVBoxLayout(metrics_container)
        self.metrics_viewer = MetricsViewer()
        metrics_layout.addWidget(self.metrics_viewer)

        # Best results label
        self._lbl_best = QLabel("🏆 Best: --")
        self._lbl_best.setStyleSheet(
            "color: #a6e3a1; font-size: 12px; font-family: 'Courier New';"
        )
        self._lbl_best.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(self._lbl_best)

        splitter.addWidget(metrics_container)

        splitter.setSizes([400, 350])
        return splitter

    # ==================================================================
    # Slots & Logic
    # ==================================================================

    def on_task_changed(self, task: str):
        """Cập nhật danh sách model khi task thay đổi."""
        self.update_model_list(task)
        # Cập nhật image size mặc định cho classification
        if task == "Classification":
            self.spin_imgsz.setValue(224)
        else:
            self.spin_imgsz.setValue(640)

    def update_model_list(self, task: str):
        """Cập nhật ComboBox model theo task."""
        models = get_available_models(task)
        self.combo_model.clear()
        self.combo_model.addItems(models)

    def _load_device_info(self):
        """Tải thông tin thiết bị."""
        try:
            info = get_device_info()
            self.combo_device.clear()
            self.combo_device.addItem("auto")
            for dev in info.get("devices", []):
                self.combo_device.addItem(dev)

            if info["device"] != "cpu":
                device_text = f"GPU: {info['name']} ({info['vram']} GB VRAM)"
                self._lbl_device_info.setStyleSheet("color: #a6e3a1; font-size: 11px;")
            else:
                device_text = "CPU only - Không phát hiện GPU"
                self._lbl_device_info.setStyleSheet("color: #f9e2af; font-size: 11px;")
            self._lbl_device_info.setText(device_text)
        except Exception as e:
            self._lbl_device_info.setText(f"Lỗi: {e}")

    def browse_dataset(self):
        """Mở hộp thoại chọn file YAML dataset."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Dataset Config", "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.edit_dataset.setText(path)
            self._update_dataset_info(path)

    def _update_dataset_info(self, path: str):
        """Cập nhật thông tin dataset."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            nc = data.get("nc", "?")
            task = data.get("task", "?")
            self._lbl_dataset_info.setText(
                f"✅ Task: {task} | {nc} classes | {Path(path).name}"
            )
            self._lbl_dataset_info.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        except Exception as e:
            self._lbl_dataset_info.setText(f"❌ Lỗi đọc YAML: {e}")
            self._lbl_dataset_info.setStyleSheet("color: #f38ba8; font-size: 11px;")

    def browse_model(self):
        """Mở hộp thoại chọn file model .pt."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Model Weights", "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)"
        )
        if path:
            self.combo_model.insertItem(0, path)
            self.combo_model.setCurrentIndex(0)

    def browse_output(self):
        """Mở hộp thoại chọn thư mục output."""
        path = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Project")
        if path:
            self.edit_project.setText(path)

    def _browse_resume(self):
        """Chọn checkpoint để resume."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Checkpoint", "",
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.edit_resume_path.setText(path)

    def get_training_params(self) -> dict:
        """Thu thập tất cả tham số từ UI thành dict."""
        task = self.combo_task.currentText()
        return {
            "task":         TASK_MAP.get(task, "detect"),
            "model":        self.combo_model.currentText(),
            "data":         self.edit_dataset.text().strip(),
            "epochs":       self.spin_epochs.value(),
            "batch":        self.spin_batch.value(),
            "imgsz":        self.spin_imgsz.value(),
            "lr0":          self.dspin_lr.value(),
            "optimizer":    self.combo_optimizer.currentText(),
            "workers":      self.spin_workers.value(),
            "patience":     self.spin_patience.value(),
            "weight_decay": self.dspin_weight_decay.value(),
            "hsv_h":        self.dspin_hsv_h.value(),
            "hsv_s":        self.dspin_hsv_s.value(),
            "hsv_v":        self.dspin_hsv_v.value(),
            "degrees":      self.dspin_degrees.value(),
            "translate":    self.dspin_translate.value(),
            "scale":        self.dspin_scale.value(),
            "flipud":       1.0 if self.chk_flipud.isChecked() else 0.0,
            "fliplr":       0.5 if self.chk_fliplr.isChecked() else 0.0,
            "mosaic":       self.dspin_mosaic.value(),
            "mixup":        self.dspin_mixup.value(),
            "device":       self.combo_device.currentText(),
            "project":      self.edit_project.text().strip() or "runs",
            "name":         self.edit_exp_name.text().strip() or "train",
            "save_period":  self.spin_save_period.value(),
            "resume":       self.chk_resume.isChecked(),
            "resume_path":  self.edit_resume_path.text().strip(),
        }

    def start_training(self):
        """Khởi động quá trình huấn luyện."""
        params = self.get_training_params()

        # Kiểm tra đầu vào
        if not params["data"]:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng chọn file dataset!")
            return
        if not params["model"]:
            QMessageBox.warning(self, "Thiếu model", "Vui lòng chọn model!")
            return
        if not Path(params["data"]).exists():
            QMessageBox.warning(self, "Lỗi", f"Không tìm thấy file dataset:\n{params['data']}")
            return

        # Reset UI
        self.log_viewer.clear_log()
        self.metrics_viewer.clear()
        self.progress_widget.reset()
        self.progress_widget.start_timer()
        self._best_metric_score = 0.0
        self._lbl_best.setText("🏆 Best: --")
        self._train_start_time = time.time()

        # Cập nhật trạng thái buttons
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)

        self.log_viewer.append_log("🚀 Khởi động huấn luyện...", "INFO")

        # Tạo và chạy worker
        self._worker = TrainWorker(params)
        self._worker.log.connect(self.on_training_log)
        self._worker.progress.connect(self.on_training_progress)
        self._worker.metrics.connect(self.on_training_metrics)
        self._worker.finished.connect(self.on_training_finished)
        self._worker.error.connect(self.on_training_error)
        self._worker.start()

    def stop_training(self):
        """Dừng huấn luyện."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.log_viewer.append_log("⏹️ Đang dừng huấn luyện...", "WARNING")
        self.btn_stop_train.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def on_training_log(self, msg: str):
        """Nhận log từ worker."""
        level = "INFO"
        if "error" in msg.lower() or "❌" in msg:
            level = "ERROR"
        elif "warning" in msg.lower() or "⚠️" in msg:
            level = "WARNING"
        elif any(k in msg for k in ["mAP", "loss", "📊", "METRIC"]):
            level = "METRIC"
        self.log_viewer.append_log(msg, level)

    def on_training_progress(self, current: int, total: int, info: dict):
        """Cập nhật thanh tiến trình."""
        elapsed = info.get("elapsed", 0)
        eta = info.get("eta", 0)
        speed = info.get("speed", "--")
        best_str = f"mAP50={self._best_metric_score:.4f}" if self._best_metric_score > 0 else "--"
        self.progress_widget.update_progress(
            current, total, elapsed, eta, speed, best_str
        )

    def on_training_metrics(self, metrics_dict: dict):
        """Cập nhật biểu đồ metrics."""
        self.metrics_viewer.update_metrics(metrics_dict)
        # Cập nhật best metric
        for key in ["metrics/mAP50", "metrics/mAP50-95", "metrics/accuracy_top1"]:
            if key in metrics_dict:
                val = float(metrics_dict[key])
                if val > self._best_metric_score:
                    self._best_metric_score = val
                    self._lbl_best.setText(
                        f"🏆 Best {key.split('/')[-1]}: {self._best_metric_score:.4f} "
                        f"(Epoch {metrics_dict.get('epoch', '?')})"
                    )

    def on_training_finished(self, result_path: str):
        """Xử lý khi huấn luyện hoàn thành."""
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)

        elapsed = time.time() - self._train_start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)

        if result_path:
            self.log_viewer.append_log(
                f"✅ Huấn luyện hoàn thành! Thời gian: {h:02d}:{m:02d}:{s:02d}\n"
                f"📁 Kết quả: {result_path}", "SUCCESS"
            )
            QMessageBox.information(
                self, "Hoàn thành",
                f"Huấn luyện hoàn thành!\nThời gian: {h:02d}:{m:02d}:{s:02d}\nKết quả: {result_path}"
            )
        else:
            self.log_viewer.append_log("⏹️ Huấn luyện đã dừng.", "WARNING")

    def on_training_error(self, error_msg: str):
        """Xử lý khi có lỗi huấn luyện."""
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.log_viewer.append_log(f"❌ {error_msg}", "ERROR")
        QMessageBox.critical(self, "Lỗi Huấn luyện", error_msg[:500])

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def save_config(self):
        """Lưu cấu hình hiện tại ra file JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Cấu hình", "train_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            params = self.get_training_params()
            if save_config(params, path):
                QMessageBox.information(self, "Đã lưu", f"Đã lưu cấu hình: {path}")
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể lưu cấu hình!")

    def load_config(self):
        """Tải cấu hình từ file JSON."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Tải Cấu hình", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                cfg = load_config(path)
                self._apply_config(cfg)
                QMessageBox.information(self, "Đã tải", f"Đã tải cấu hình: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể tải cấu hình:\n{e}")

    def _apply_config(self, cfg: dict):
        """Áp dụng cấu hình lên UI."""
        task_map_rev = {"classify": "Classification", "detect": "Object Detection",
                        "obb": "OBB Detection", "segment": "Segmentation"}
        task = task_map_rev.get(cfg.get("task", "detect"), "Object Detection")
        idx = self.combo_task.findText(task)
        if idx >= 0:
            self.combo_task.setCurrentIndex(idx)

        self.spin_epochs.setValue(cfg.get("epochs", 100))
        self.spin_batch.setValue(cfg.get("batch", 16))
        self.spin_imgsz.setValue(cfg.get("imgsz", 640))
        self.dspin_lr.setValue(cfg.get("lr0", 0.01))
        opt_idx = self.combo_optimizer.findText(cfg.get("optimizer", "SGD"))
        if opt_idx >= 0:
            self.combo_optimizer.setCurrentIndex(opt_idx)
        self.spin_workers.setValue(cfg.get("workers", 8))
        self.spin_patience.setValue(cfg.get("patience", 50))
        self.dspin_weight_decay.setValue(cfg.get("weight_decay", 0.0005))
        self.dspin_hsv_h.setValue(cfg.get("hsv_h", 0.015))
        self.dspin_hsv_s.setValue(cfg.get("hsv_s", 0.7))
        self.dspin_hsv_v.setValue(cfg.get("hsv_v", 0.4))
        self.dspin_degrees.setValue(cfg.get("degrees", 0.0))
        self.dspin_translate.setValue(cfg.get("translate", 0.1))
        self.dspin_scale.setValue(cfg.get("scale", 0.5))
        self.chk_flipud.setChecked(cfg.get("flipud", 0.0) > 0)
        self.chk_fliplr.setChecked(cfg.get("fliplr", 0.5) > 0)
        self.dspin_mosaic.setValue(cfg.get("mosaic", 1.0))
        self.dspin_mixup.setValue(cfg.get("mixup", 0.0))
        if cfg.get("project"):
            self.edit_project.setText(cfg["project"])
        if cfg.get("name"):
            self.edit_exp_name.setText(cfg["name"])
        self.spin_save_period.setValue(cfg.get("save_period", -1))
