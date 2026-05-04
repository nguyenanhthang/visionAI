"""
Cửa sổ chính của YOLO GUI
"""
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar,
    QMenuBar, QMenu, QMessageBox, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from app.train_tab import TrainTab
from app.predict_tab import PredictTab
from app.utils.yolo_utils import get_device_info


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng YOLO GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()

    # ==================================================================
    # Setup
    # ==================================================================

    def setup_ui(self):
        """Thiết lập giao diện chính."""
        self.setWindowTitle("YOLO GUI - Train & Predict")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # Tab widget trung tâm
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        # Tạo các tab
        self.train_tab = TrainTab()
        self.predict_tab = PredictTab()

        self.tab_widget.addTab(self.train_tab, "🚀 Train")
        self.tab_widget.addTab(self.predict_tab, "🔍 Predict")

        self.setCentralWidget(self.tab_widget)

    def setup_menu(self):
        """Thiết lập menu bar."""
        menubar = self.menuBar()

        # === File Menu ===
        file_menu = menubar.addMenu("📁 File")

        act_open = QAction("📂 Mở Project", self)
        act_open.setShortcut("Ctrl+O")
        act_open.setStatusTip("Mở thư mục project")
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save_cfg = QAction("💾 Lưu Config", self)
        act_save_cfg.setShortcut("Ctrl+S")
        act_save_cfg.setStatusTip("Lưu cấu hình hiện tại")
        act_save_cfg.triggered.connect(self._save_config)
        file_menu.addAction(act_save_cfg)

        file_menu.addSeparator()

        act_exit = QAction("🚪 Thoát", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.setStatusTip("Thoát ứng dụng")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # === Settings Menu ===
        settings_menu = menubar.addMenu("⚙️ Settings")

        act_theme = QAction("🎨 Theme", self)
        act_theme.setStatusTip("Thay đổi giao diện (hiện chỉ hỗ trợ dark theme)")
        act_theme.triggered.connect(self._show_theme_info)
        settings_menu.addAction(act_theme)

        act_gpu = QAction("💻 GPU Settings", self)
        act_gpu.setStatusTip("Xem thông tin GPU")
        act_gpu.triggered.connect(self._show_gpu_info)
        settings_menu.addAction(act_gpu)

        # === Help Menu ===
        help_menu = menubar.addMenu("❓ Help")

        act_about = QAction("ℹ️ About", self)
        act_about.setShortcut("F1")
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

        act_docs = QAction("📚 Tài liệu YOLO", self)
        act_docs.setStatusTip("Mở tài liệu Ultralytics YOLO")
        act_docs.triggered.connect(self._open_docs)
        help_menu.addAction(act_docs)

    def setup_status_bar(self):
        """Thiết lập status bar với thông tin GPU và phiên bản."""
        status_bar = self.statusBar()

        # Thông tin YOLO version
        yolo_label = QLabel()
        try:
            import ultralytics
            yolo_ver = ultralytics.__version__
        except ImportError:
            yolo_ver = "not installed"
        yolo_label.setText(f"  Ultralytics YOLO v{yolo_ver}  ")
        yolo_label.setStyleSheet("color: #89b4fa;")

        # Thông tin GPU
        gpu_label = QLabel()
        try:
            info = get_device_info()
            if info["device"] != "cpu":
                gpu_text = f"  🟢 GPU: {info['name']} ({info['vram']} GB)  "
                gpu_label.setStyleSheet("color: #a6e3a1;")
            else:
                gpu_text = "  🟡 CPU Only  "
                gpu_label.setStyleSheet("color: #f9e2af;")
            gpu_label.setText(gpu_text)
        except Exception:
            gpu_label.setText("  Device: Unknown  ")

        status_bar.addPermanentWidget(gpu_label)
        status_bar.addPermanentWidget(yolo_label)
        status_bar.showMessage("Sẵn sàng - YOLO GUI")

    # ==================================================================
    # Menu Actions
    # ==================================================================

    def _open_project(self):
        """Mở thư mục project."""
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Mở Thư mục Project")
        if path:
            self.statusBar().showMessage(f"Project: {path}")

    def _save_config(self):
        """Lưu config từ tab đang active."""
        current = self.tab_widget.currentIndex()
        if current == 0:
            self.train_tab.save_config()
        else:
            QMessageBox.information(self, "Thông báo", "Lưu config chỉ áp dụng cho tab Train.")

    def _show_theme_info(self):
        QMessageBox.information(
            self, "Theme",
            "Ứng dụng sử dụng Dark Theme (Catppuccin Mocha).\n"
            "Hiện tại chỉ hỗ trợ dark theme."
        )

    def _show_gpu_info(self):
        """Hiển thị thông tin GPU chi tiết."""
        try:
            import torch
            if torch.cuda.is_available():
                lines = [f"Số GPU: {torch.cuda.device_count()}"]
                for i in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(i)
                    vram = prop.total_memory / (1024 ** 3)
                    lines.append(
                        f"GPU {i}: {prop.name}\n"
                        f"  VRAM: {vram:.1f} GB\n"
                        f"  CUDA: {prop.major}.{prop.minor}"
                    )
                msg = "\n".join(lines)
            else:
                msg = "Không phát hiện GPU CUDA.\nSử dụng CPU."
        except ImportError:
            msg = "PyTorch chưa được cài đặt."
        QMessageBox.information(self, "GPU Information", msg)

    def _open_docs(self):
        """Mở tài liệu YOLO trong trình duyệt."""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://docs.ultralytics.com"))

    def show_about(self):
        """Hiển thị hộp thoại About."""
        try:
            import ultralytics
            yolo_ver = ultralytics.__version__
        except ImportError:
            yolo_ver = "not installed"

        try:
            from PySide6 import __version__ as pyside_ver
        except ImportError:
            pyside_ver = "unknown"

        QMessageBox.about(
            self,
            "About YOLO GUI",
            f"""<h2>🚀 YOLO GUI</h2>
            <p>Giao diện đồ họa cho Ultralytics YOLO</p>
            <hr>
            <p><b>Phiên bản:</b> 1.0.0</p>
            <p><b>Ultralytics YOLO:</b> {yolo_ver}</p>
            <p><b>PySide6:</b> {pyside_ver}</p>
            <hr>
            <p>Hỗ trợ các task:</p>
            <ul>
                <li>🏷️ Classification (Phân loại)</li>
                <li>📦 Object Detection (Phát hiện đối tượng)</li>
                <li>🔄 OBB Detection (Phát hiện với bounding box xoay)</li>
                <li>✂️ Segmentation (Phân đoạn)</li>
            </ul>
            <p><a href="https://docs.ultralytics.com">📚 Tài liệu YOLO</a></p>
            """
        )
