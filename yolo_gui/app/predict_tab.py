"""
Tab dự đoán YOLO
Hỗ trợ ảnh đơn, thư mục (duyệt từng ảnh bằng Next/Prev), video và webcam
"""
import json
from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QScrollArea, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QButtonGroup,
    QRadioButton, QSizePolicy, QAbstractItemView
)
from PySide6.QtCore import Qt

from app.widgets.image_viewer import ImageViewer
from app.workers.predict_worker import PredictWorker
from app.utils.yolo_utils import get_available_models, get_device_info, TASK_MAP
from app.utils.file_utils import save_results_csv, save_results_json


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
}


class PredictTab(QWidget):
    """Tab dự đoán — folder mode dùng Next/Prev predict từng ảnh."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[PredictWorker] = None
        self._last_detections: list = []
        self._all_results: list = []

        # ── Folder mode ──
        self._folder_images: List[Path] = []
        self._folder_results: dict = {}     # {index: (result_image, detections)}
        self._current_index: int = -1
        self._is_folder_mode: bool = False
        self._is_predicting: bool = False    # Đang predict 1 ảnh?

        self._setup_ui()
        self._load_device_info()

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel trái
        left_scroll = self._build_left_panel()
        h_splitter.addWidget(left_scroll)

        # Panel giữa (ảnh gốc)
        center_widget = QGroupBox("🖼️ Ảnh / Frame Gốc")
        center_layout = QVBoxLayout(center_widget)
        self.viewer_original = ImageViewer()
        center_layout.addWidget(self.viewer_original)
        h_splitter.addWidget(center_widget)

        # Panel phải (kết quả)
        right_widget = QGroupBox("🔍 Kết quả Dự đoán")
        right_layout = QVBoxLayout(right_widget)
        self.viewer_result = ImageViewer()
        right_layout.addWidget(self.viewer_result)
        h_splitter.addWidget(right_widget)

        h_splitter.setSizes([300, 500, 500])
        main_layout.addWidget(h_splitter, 60)

        # Thanh điều hướng folder
        self._nav_widget = self._build_navigation_bar()
        main_layout.addWidget(self._nav_widget)
        self._nav_widget.setVisible(False)

        # Bảng kết quả
        bottom_widget = self._build_results_table()
        main_layout.addWidget(bottom_widget, 30)

    # ------------------------------------------------------------------
    # Navigation bar
    # ------------------------------------------------------------------

    def _build_navigation_bar(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 8px;
            }
        """)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Nút Previous + Predict
        self.btn_prev = QPushButton("◀ Trước")
        self.btn_prev.setMinimumHeight(34)
        self.btn_prev.setMinimumWidth(90)
        self.btn_prev.clicked.connect(self._go_prev_and_predict)
        self.btn_prev.setStyleSheet(self._nav_btn_style("#89b4fa"))
        layout.addWidget(self.btn_prev)

        layout.addStretch()

        # Thông tin vị trí
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self._lbl_nav_info = QLabel("0 / 0")
        self._lbl_nav_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_nav_info.setStyleSheet(
            "color: #cdd6f4; font-size: 14px; font-weight: bold; border: none;"
        )
        info_layout.addWidget(self._lbl_nav_info)

        self._lbl_nav_filename = QLabel("")
        self._lbl_nav_filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_nav_filename.setStyleSheet(
            "color: #a6adc8; font-size: 11px; border: none;"
        )
        info_layout.addWidget(self._lbl_nav_filename)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Nút Next + Predict
        self.btn_next = QPushButton("Sau ▶")
        self.btn_next.setMinimumHeight(34)
        self.btn_next.setMinimumWidth(90)
        self.btn_next.clicked.connect(self._go_next_and_predict)
        self.btn_next.setStyleSheet(self._nav_btn_style("#a6e3a1"))
        layout.addWidget(self.btn_next)

        return container

    @staticmethod
    def _nav_btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}; color: #1e1e2e;
                border: none; border-radius: 6px;
                padding: 6px 16px; font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ background-color: #94e2d5; }}
            QPushButton:disabled {{ background-color: #45475a; color: #6c7086; }}
        """

    # ------------------------------------------------------------------
    # Navigation logic — mỗi lần bấm Next/Prev = predict 1 ảnh
    # ------------------------------------------------------------------

    def _go_prev_and_predict(self):
        """Về ảnh trước → nếu chưa predict thì predict, nếu rồi thì hiển thị cache."""
        if self._is_predicting:
            return
        if self._current_index > 0:
            self._current_index -= 1
            self._show_or_predict_current()

    def _go_next_and_predict(self):
        """Sang ảnh tiếp → nếu chưa predict thì predict, nếu rồi thì hiển thị cache."""
        if self._is_predicting:
            return
        if self._current_index < len(self._folder_images) - 1:
            self._current_index += 1
            self._show_or_predict_current()

    def _show_or_predict_current(self):
        """Hiển thị ảnh hiện tại. Nếu chưa predict → predict. Nếu rồi → hiện cache."""
        idx = self._current_index
        if idx < 0 or idx >= len(self._folder_images):
            return

        img_path = self._folder_images[idx]

        # Hiển thị ảnh gốc
        self.viewer_original.load_image(str(img_path))

        # Kiểm tra cache
        if idx in self._folder_results:
            # Đã predict rồi → hiện cache, không predict lại
            result_img, detections = self._folder_results[idx]
            self.viewer_result.load_image(result_img)
            self._last_detections = detections
            self._update_table(detections)
            self._update_nav_ui()
        else:
            # Chưa predict → predict ảnh này
            self._predict_current_image()

    def _predict_current_image(self):
        """Predict ảnh tại _current_index."""
        idx = self._current_index
        if idx < 0 or idx >= len(self._folder_images):
            return

        img_path = self._folder_images[idx]
        params = self._get_predict_params()
        params["source_type"] = "single_file"   # Chế độ predict 1 file
        params["source"] = str(img_path)

        self._is_predicting = True
        self._set_nav_buttons_enabled(False)
        self._update_nav_ui(predicting=True)

        # Xóa kết quả cũ trên viewer
        self.viewer_result.clear_image()
        self.table_results.setRowCount(0)
        self._lbl_summary.setText("⏳ Đang predict...")

        self._worker = PredictWorker(params)
        self._worker.result.connect(self._on_folder_single_result)
        self._worker.error.connect(self._on_folder_single_error)
        self._worker.finished.connect(self._on_folder_single_finished)
        self._worker.start()

    def _on_folder_single_result(self, image, detections: list):
        """Nhận kết quả predict 1 ảnh trong folder."""
        idx = self._current_index
        # Lưu vào cache
        self._folder_results[idx] = (image, detections)
        self._all_results.extend(detections)

        # Hiển thị
        self.viewer_result.load_image(image)
        self._last_detections = detections
        self._update_table(detections)

    def _on_folder_single_error(self, msg: str):
        """Lỗi khi predict 1 ảnh."""
        self._lbl_summary.setText(f"❌ Lỗi: {msg[:100]}")

    def _on_folder_single_finished(self):
        """Predict 1 ảnh xong → mở khóa nút Next/Prev."""
        self._is_predicting = False
        self._set_nav_buttons_enabled(True)
        self._update_nav_ui()

    def _set_nav_buttons_enabled(self, enabled: bool):
        """Enable/disable nút nav."""
        self.btn_prev.setEnabled(enabled and self._current_index > 0)
        self.btn_next.setEnabled(
            enabled and self._current_index < len(self._folder_images) - 1
        )

    def _update_nav_ui(self, predicting: bool = False):
        """Cập nhật label trên navigation bar."""
        total = len(self._folder_images)
        current = self._current_index + 1 if self._current_index >= 0 else 0
        self._lbl_nav_info.setText(f"{current} / {total}")

        if 0 <= self._current_index < total:
            filename = self._folder_images[self._current_index].name
            has_cache = self._current_index in self._folder_results
            if predicting:
                status = "⏳ Đang predict..."
            elif has_cache:
                n_det = len(self._folder_results[self._current_index][1])
                status = f"✅ {n_det} detections"
            else:
                status = "📷 Chưa predict"
            self._lbl_nav_filename.setText(f"{filename}  —  {status}")
        else:
            self._lbl_nav_filename.setText("")

        # Enable/disable nút
        if not predicting:
            self.btn_prev.setEnabled(self._current_index > 0)
            self.btn_next.setEnabled(self._current_index < total - 1)

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setMaximumWidth(320)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_inference_group())
        layout.addWidget(self._build_viz_group())
        layout.addWidget(self._build_action_buttons())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("🎯 Model")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self.combo_task = QComboBox()
        self.combo_task.addItems([
            "Classification", "Object Detection",
            "OBB Detection", "Segmentation"
        ])
        self.combo_task.setCurrentText("Object Detection")
        self.combo_task.currentTextChanged.connect(self.on_task_changed)
        layout.addWidget(QLabel("Task:"))
        layout.addWidget(self.combo_task)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        self.edit_model = QLineEdit()
        self.edit_model.setPlaceholderText("Chọn file .pt...")
        btn_browse = QPushButton("📁")
        btn_browse.setMaximumWidth(36)
        btn_browse.clicked.connect(self.browse_model)
        model_layout.addWidget(self.edit_model)
        model_layout.addWidget(btn_browse)
        layout.addWidget(QLabel("Model weights:"))
        layout.addWidget(model_row)

        self._lbl_model_info = QLabel("Chưa chọn model")
        self._lbl_model_info.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._lbl_model_info.setWordWrap(True)
        layout.addWidget(self._lbl_model_info)

        return grp

    def _build_source_group(self) -> QGroupBox:
        grp = QGroupBox("📥 Nguồn đầu vào")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._source_group = QButtonGroup(self)
        self.radio_image = QRadioButton("Ảnh đơn")
        self.radio_folder = QRadioButton("Thư mục (Next/Prev)")
        self.radio_video = QRadioButton("Video")
        self.radio_webcam = QRadioButton("Webcam")
        self.radio_image.setChecked(True)

        for i, rb in enumerate([self.radio_image, self.radio_folder,
                                self.radio_video, self.radio_webcam]):
            self._source_group.addButton(rb, i)
            layout.addWidget(rb)

        self._source_group.buttonClicked.connect(self.on_source_changed)

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(4)
        self.edit_source = QLineEdit()
        self.edit_source.setPlaceholderText("Chọn file/thư mục...")
        btn_browse_src = QPushButton("📁")
        btn_browse_src.setMaximumWidth(36)
        btn_browse_src.clicked.connect(self.browse_input)
        file_layout.addWidget(self.edit_source)
        file_layout.addWidget(btn_browse_src)
        layout.addWidget(file_row)

        webcam_row = QWidget()
        webcam_layout = QHBoxLayout(webcam_row)
        webcam_layout.setContentsMargins(0, 0, 0, 0)
        webcam_layout.setSpacing(4)
        webcam_layout.addWidget(QLabel("Webcam ID:"))
        self.spin_cam_id = QSpinBox()
        self.spin_cam_id.setRange(0, 10)
        self.spin_cam_id.setValue(0)
        self.spin_cam_id.setEnabled(False)
        webcam_layout.addWidget(self.spin_cam_id)
        layout.addWidget(webcam_row)

        self._lbl_source_info = QLabel("")
        self._lbl_source_info.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._lbl_source_info.setWordWrap(True)
        layout.addWidget(self._lbl_source_info)

        return grp

    def _build_inference_group(self) -> QGroupBox:
        grp = QGroupBox("⚙️ Tham số Inference")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Confidence Threshold:"))
        conf_row = QWidget()
        conf_layout = QHBoxLayout(conf_row)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(4)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(25)
        self.dspin_conf = QDoubleSpinBox()
        self.dspin_conf.setRange(0.0, 1.0)
        self.dspin_conf.setValue(0.25)
        self.dspin_conf.setSingleStep(0.01)
        self.dspin_conf.setDecimals(2)
        self.dspin_conf.setMaximumWidth(70)
        self.slider_conf.valueChanged.connect(
            lambda v: self.dspin_conf.setValue(v / 100))
        self.dspin_conf.valueChanged.connect(
            lambda v: self.slider_conf.setValue(int(v * 100)))
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.dspin_conf)
        layout.addWidget(conf_row)

        layout.addWidget(QLabel("IoU Threshold:"))
        iou_row = QWidget()
        iou_layout = QHBoxLayout(iou_row)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.setSpacing(4)
        self.slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_iou.setRange(0, 100)
        self.slider_iou.setValue(70)
        self.dspin_iou = QDoubleSpinBox()
        self.dspin_iou.setRange(0.0, 1.0)
        self.dspin_iou.setValue(0.7)
        self.dspin_iou.setSingleStep(0.01)
        self.dspin_iou.setDecimals(2)
        self.dspin_iou.setMaximumWidth(70)
        self.slider_iou.valueChanged.connect(
            lambda v: self.dspin_iou.setValue(v / 100))
        self.dspin_iou.valueChanged.connect(
            lambda v: self.slider_iou.setValue(int(v * 100)))
        iou_layout.addWidget(self.slider_iou)
        iou_layout.addWidget(self.dspin_iou)
        layout.addWidget(iou_row)

        max_row = QWidget()
        max_layout = QHBoxLayout(max_row)
        max_layout.setContentsMargins(0, 0, 0, 0)
        max_layout.addWidget(QLabel("Max Detections:"))
        self.spin_max_det = QSpinBox()
        self.spin_max_det.setRange(1, 1000)
        self.spin_max_det.setValue(300)
        max_layout.addWidget(self.spin_max_det)
        layout.addWidget(max_row)

        imgsz_row = QWidget()
        imgsz_layout = QHBoxLayout(imgsz_row)
        imgsz_layout.setContentsMargins(0, 0, 0, 0)
        imgsz_layout.addWidget(QLabel("Image Size:"))
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 1280)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        imgsz_layout.addWidget(self.spin_imgsz)
        layout.addWidget(imgsz_row)

        self.chk_half = QCheckBox("Half Precision (FP16)")
        layout.addWidget(self.chk_half)

        self.chk_agnostic = QCheckBox("Agnostic NMS")
        layout.addWidget(self.chk_agnostic)

        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItem("auto")
        device_layout.addWidget(self.combo_device)
        layout.addWidget(device_row)

        return grp

    def _build_viz_group(self) -> QGroupBox:
        grp = QGroupBox("🎨 Hiển thị")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self.chk_show_labels = QCheckBox("Hiện nhãn class")
        self.chk_show_labels.setChecked(True)
        layout.addWidget(self.chk_show_labels)

        self.chk_show_conf = QCheckBox("Hiện confidence score")
        self.chk_show_conf.setChecked(True)
        layout.addWidget(self.chk_show_conf)

        self.chk_show_boxes = QCheckBox("Hiện bounding boxes")
        self.chk_show_boxes.setChecked(True)
        layout.addWidget(self.chk_show_boxes)

        self.chk_show_masks = QCheckBox("Hiện masks (Segmentation)")
        self.chk_show_masks.setChecked(True)
        layout.addWidget(self.chk_show_masks)

        lw_row = QWidget()
        lw_layout = QHBoxLayout(lw_row)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line Width:"))
        self.spin_line_width = QSpinBox()
        self.spin_line_width.setRange(1, 10)
        self.spin_line_width.setValue(2)
        lw_layout.addWidget(self.spin_line_width)
        layout.addWidget(lw_row)

        return grp

    def _build_action_buttons(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        self.btn_run_predict = QPushButton("🔍 Chạy Dự đoán")
        self.btn_run_predict.setObjectName("btn_run_predict")
        self.btn_run_predict.setMinimumHeight(36)
        self.btn_run_predict.clicked.connect(self.run_predict)

        self.btn_stop_predict = QPushButton("⏹️ Dừng")
        self.btn_stop_predict.setObjectName("btn_stop_predict")
        self.btn_stop_predict.setMinimumHeight(36)
        self.btn_stop_predict.setEnabled(False)
        self.btn_stop_predict.clicked.connect(self.stop_predict)

        row1.addWidget(self.btn_run_predict)
        row1.addWidget(self.btn_stop_predict)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_save = QPushButton("💾 Lưu Kết quả")
        btn_save.clicked.connect(self.save_results)
        row2.addWidget(btn_save)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        btn_csv = QPushButton("📤 Export CSV")
        btn_csv.clicked.connect(self.export_csv)
        btn_json = QPushButton("📤 Export JSON")
        btn_json.clicked.connect(self.export_json)
        row3.addWidget(btn_csv)
        row3.addWidget(btn_json)
        layout.addLayout(row3)

        return container

    def _build_results_table(self) -> QGroupBox:
        grp = QGroupBox("📋 Kết quả Phát hiện")
        layout = QVBoxLayout(grp)

        self._lbl_summary = QLabel("")
        self._lbl_summary.setStyleSheet(
            "color: #a6e3a1; font-size: 11px; font-weight: bold;")
        layout.addWidget(self._lbl_summary)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(5)
        self.table_results.setHorizontalHeaderLabels(
            ["ID", "Class", "Confidence", "BBox", "Area"])
        self.table_results.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table_results.setAlternatingRowColors(True)
        self.table_results.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setMaximumHeight(200)
        layout.addWidget(self.table_results)

        return grp

    # ==================================================================
    # Slots
    # ==================================================================

    def on_task_changed(self, task: str):
        if task == "Classification":
            self.spin_imgsz.setValue(224)
        else:
            self.spin_imgsz.setValue(640)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Model Weights", "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*)")
        if path:
            self.edit_model.setText(path)
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            self._lbl_model_info.setText(
                f"✅ {Path(path).name} ({size_mb:.1f} MB)")
            self._lbl_model_info.setStyleSheet(
                "color: #a6e3a1; font-size: 11px;")

    def on_source_changed(self, button):
        is_webcam = self.radio_webcam.isChecked()
        self.edit_source.setEnabled(not is_webcam)
        self.spin_cam_id.setEnabled(is_webcam)

        if not self.radio_folder.isChecked():
            self._nav_widget.setVisible(False)
            self._is_folder_mode = False

    def browse_input(self):
        if self.radio_image.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Chọn Ảnh", "",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All (*)")
        elif self.radio_folder.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Ảnh")
        elif self.radio_video.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Chọn Video", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All (*)")
        else:
            return

        if path:
            self.edit_source.setText(path)
            if self.radio_image.isChecked():
                self.viewer_original.load_image(path)
                self._nav_widget.setVisible(False)
                self._is_folder_mode = False
            elif self.radio_folder.isChecked():
                self._scan_folder(path)

    def _scan_folder(self, folder_path: str):
        """Quét folder, hiện nav bar, preview ảnh đầu."""
        folder = Path(folder_path)
        self._folder_images = sorted([
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        self._folder_results.clear()
        self._current_index = 0
        self._is_folder_mode = True

        total = len(self._folder_images)
        if total == 0:
            self._lbl_source_info.setText(
                "❌ Không tìm thấy ảnh trong thư mục!")
            self._lbl_source_info.setStyleSheet(
                "color: #f38ba8; font-size: 11px;")
            self._nav_widget.setVisible(False)
            return

        self._lbl_source_info.setText(
            f"📂 {total} ảnh | Bấm ▶ Dự đoán rồi dùng ◀ Trước / Sau ▶")
        self._lbl_source_info.setStyleSheet("color: #a6e3a1; font-size: 11px;")

        # Hiện nav bar + preview ảnh đầu tiên (chưa predict)
        self._nav_widget.setVisible(True)
        self.viewer_original.load_image(str(self._folder_images[0]))
        self.viewer_result.clear_image()
        self._update_nav_ui()

    def _load_device_info(self):
        try:
            info = get_device_info()
            self.combo_device.clear()
            self.combo_device.addItem("auto")
            for dev in info.get("devices", []):
                self.combo_device.addItem(dev)
        except Exception:
            pass

    def _get_predict_params(self) -> dict:
        if self.radio_image.isChecked():
            source_type = "image"
        elif self.radio_folder.isChecked():
            source_type = "folder"
        elif self.radio_video.isChecked():
            source_type = "video"
        else:
            source_type = "webcam"

        return {
            "model":        self.edit_model.text().strip(),
            "source":       self.edit_source.text().strip(),
            "source_type":  source_type,
            "cam_id":       self.spin_cam_id.value(),
            "conf":         self.dspin_conf.value(),
            "iou":          self.dspin_iou.value(),
            "imgsz":        self.spin_imgsz.value(),
            "max_det":      self.spin_max_det.value(),
            "half":         self.chk_half.isChecked(),
            "agnostic_nms": self.chk_agnostic.isChecked(),
            "show_labels":  self.chk_show_labels.isChecked(),
            "show_conf":    self.chk_show_conf.isChecked(),
            "line_width":   self.spin_line_width.value(),
            "device":       self.combo_device.currentText(),
        }

    # ==================================================================
    # Run Predict
    # ==================================================================

    def run_predict(self):
        """Nút 🔍 Chạy Dự đoán."""
        params = self._get_predict_params()

        if not params["model"]:
            QMessageBox.warning(self, "Thiếu model",
                                "Vui lòng chọn model weights!")
            return
        if not Path(params["model"]).exists():
            QMessageBox.warning(self, "Lỗi",
                                f"Không tìm thấy model:\n{params['model']}")
            return

        # ── Folder mode: predict ảnh đầu tiên, chờ user bấm Next ──
        if self._is_folder_mode and self.radio_folder.isChecked():
            if not self._folder_images:
                QMessageBox.warning(self, "Lỗi", "Thư mục không có ảnh!")
                return
            self._all_results = []
            self._folder_results.clear()
            self._current_index = 0
            self._show_or_predict_current()
            return

        # ── Các mode khác: chạy bình thường ──
        if params["source_type"] != "webcam" and not params["source"]:
            QMessageBox.warning(self, "Thiếu nguồn",
                                "Vui lòng chọn file/thư mục nguồn!")
            return

        self._last_detections = []
        self._all_results = []
        self.table_results.setRowCount(0)
        self._lbl_summary.setText("")
        self.viewer_result.clear_image()

        self.btn_run_predict.setEnabled(False)
        self.btn_stop_predict.setEnabled(True)

        self._worker = PredictWorker(params)
        self._worker.result.connect(self.on_predict_result)
        self._worker.frame.connect(self.on_predict_frame)
        self._worker.finished.connect(self.on_predict_finished)
        self._worker.error.connect(self.on_predict_error)
        self._worker.start()

    def stop_predict(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self.btn_stop_predict.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker callbacks (cho image/video/webcam — KHÔNG phải folder)
    # ------------------------------------------------------------------

    def on_predict_result(self, image, detections: list):
        self.viewer_result.load_image(image)
        self._last_detections = detections
        self._all_results.extend(detections)
        self._update_table(detections)

    def on_predict_frame(self, frame):
        self.viewer_result.load_image(frame)

    def on_predict_finished(self):
        self.btn_run_predict.setEnabled(True)
        self.btn_stop_predict.setEnabled(False)

    def on_predict_error(self, msg: str):
        self.btn_run_predict.setEnabled(True)
        self.btn_stop_predict.setEnabled(False)
        QMessageBox.critical(self, "Lỗi Dự đoán", msg[:500])

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _update_table(self, detections: list):
        self.table_results.setRowCount(0)
        for det in detections:
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(
                row, 0, QTableWidgetItem(str(det.get("id", ""))))
            self.table_results.setItem(
                row, 1, QTableWidgetItem(str(det.get("class", ""))))
            conf = det.get("confidence", "")
            self.table_results.setItem(
                row, 2, QTableWidgetItem(
                    f"{conf:.4f}" if isinstance(conf, float) else str(conf)))
            self.table_results.setItem(
                row, 3, QTableWidgetItem(str(det.get("bbox", ""))))
            self.table_results.setItem(
                row, 4, QTableWidgetItem(str(det.get("area", ""))))

        n = len(detections)
        classes = set(d.get("class", "") for d in detections)
        cached = len(self._folder_results) if self._is_folder_mode else 0
        total_imgs = len(self._folder_images) if self._is_folder_mode else 0
        if self._is_folder_mode:
            self._lbl_summary.setText(
                f"Ảnh hiện tại: {n} detections | "
                f"Đã predict: {cached}/{total_imgs} ảnh")
        elif n > 0:
            self._lbl_summary.setText(
                f"{n} detections | {len(classes)} classes: "
                + ", ".join(sorted(classes)))

    # ------------------------------------------------------------------
    # Export / Save
    # ------------------------------------------------------------------

    def save_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Kết quả", "result.jpg",
            "Image Files (*.jpg *.png);;All Files (*)")
        if not path:
            return
        try:
            for item in self.viewer_result._scene.items():
                if hasattr(item, "pixmap"):
                    pixmap = item.pixmap()
                    if pixmap and not pixmap.isNull():
                        pixmap.save(path)
                        QMessageBox.information(
                            self, "Đã lưu", f"Đã lưu: {path}")
                        return
            QMessageBox.information(
                self, "Thông báo", "Chưa có ảnh kết quả!")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi", f"Không thể lưu: {e}")

    def export_csv(self):
        if not self._all_results:
            QMessageBox.information(
                self, "Thông báo", "Chưa có kết quả để xuất!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất CSV", "detections.csv",
            "CSV Files (*.csv);;All Files (*)")
        if path and save_results_csv(self._all_results, path):
            QMessageBox.information(self, "Đã xuất", f"Đã xuất: {path}")

    def export_json(self):
        if not self._all_results:
            QMessageBox.information(
                self, "Thông báo", "Chưa có kết quả để xuất!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất JSON", "detections.json",
            "JSON Files (*.json);;All Files (*)")
        if path and save_results_json(self._all_results, path):
            QMessageBox.information(self, "Đã xuất", f"Đã xuất: {path}")

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if self._is_folder_mode and not self._is_predicting:
            key = event.key()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
                self._go_prev_and_predict()
                return
            elif key in (Qt.Key.Key_Right, Qt.Key.Key_D):
                self._go_next_and_predict()
                return
        super().keyPressEvent(event)