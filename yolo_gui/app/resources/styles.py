# Giao diện tối (Dark Theme) cho toàn bộ ứng dụng
# Bảng màu Catppuccin Mocha

DARK_THEME = """
/* ===== Nền chính ===== */
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* ===== GroupBox ===== */
QGroupBox {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #89b4fa;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #89b4fa;
    font-weight: bold;
}

/* ===== QPushButton ===== */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 13px;
    min-height: 28px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
    color: #89b4fa;
}

QPushButton:pressed {
    background-color: #585b70;
    border-color: #74c7ec;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}

QPushButton#btn_start_train, QPushButton#btn_run_predict {
    background-color: #40a02b;
    color: #e6e9ef;
    border-color: #40a02b;
    font-weight: bold;
}

QPushButton#btn_start_train:hover, QPushButton#btn_run_predict:hover {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border-color: #a6e3a1;
}

QPushButton#btn_stop_train, QPushButton#btn_stop_predict {
    background-color: #d20f39;
    color: #e6e9ef;
    border-color: #d20f39;
    font-weight: bold;
}

QPushButton#btn_stop_train:hover, QPushButton#btn_stop_predict:hover {
    background-color: #f38ba8;
    color: #1e1e2e;
    border-color: #f38ba8;
}

/* ===== QComboBox ===== */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: #45475a;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox:focus {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #45475a;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background-color: #45475a;
}

QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
    outline: none;
}

/* ===== QSpinBox / QDoubleSpinBox ===== */
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 18px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}

/* ===== QLineEdit ===== */
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: #45475a;
}

QLineEdit:hover {
    border-color: #89b4fa;
}

QLineEdit:focus {
    border-color: #89b4fa;
}

QLineEdit:read-only {
    background-color: #181825;
    color: #a6adc8;
}

/* ===== QTextEdit / QPlainTextEdit ===== */
QTextEdit, QPlainTextEdit {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 4px;
    selection-background-color: #45475a;
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #89b4fa;
}

/* ===== QTabWidget ===== */
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
    border-radius: 0 6px 6px 6px;
}

QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-bottom: none;
    padding: 8px 20px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-size: 13px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #313244;
    color: #89b4fa;
    border-color: #89b4fa;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #313244;
    color: #cdd6f4;
}

/* ===== QTableWidget ===== */
QTableWidget {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    gridline-color: #313244;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
    alternate-background-color: #181825;
}

QTableWidget::item {
    padding: 4px 8px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #45475a;
    color: #cdd6f4;
}

QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 6px 8px;
    font-weight: bold;
}

QHeaderView::section:hover {
    background-color: #45475a;
}

/* ===== QProgressBar ===== */
QProgressBar {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    min-height: 20px;
    font-size: 12px;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}

QProgressBar#epoch_bar::chunk {
    background-color: #a6e3a1;
}

QProgressBar#total_bar::chunk {
    background-color: #89b4fa;
}

/* ===== QScrollArea / QScrollBar ===== */
QScrollArea {
    background-color: #1e1e2e;
    border: none;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #45475a;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ===== QSplitter ===== */
QSplitter::handle {
    background-color: #45475a;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #89b4fa;
}

/* ===== QMenuBar / QMenu ===== */
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #45475a;
    padding: 2px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #313244;
    color: #89b4fa;
}

QMenuBar::item:pressed {
    background-color: #45475a;
}

QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 0;
}

QMenu::item {
    padding: 6px 20px 6px 12px;
    border-radius: 4px;
    margin: 1px 4px;
}

QMenu::item:selected {
    background-color: #45475a;
    color: #89b4fa;
}

QMenu::separator {
    height: 1px;
    background-color: #45475a;
    margin: 4px 8px;
}

/* ===== QStatusBar ===== */
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #45475a;
    font-size: 12px;
}

QStatusBar::item {
    border: none;
}

/* ===== QCheckBox ===== */
QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

/* ===== QRadioButton ===== */
QRadioButton {
    color: #cdd6f4;
    spacing: 8px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 8px;
    background-color: #313244;
}

QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QRadioButton::indicator:hover {
    border-color: #89b4fa;
}

/* ===== QSlider ===== */
QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
    border: 1px solid #45475a;
}

QSlider::handle:horizontal {
    background-color: #89b4fa;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #74c7ec;
}

QSlider::sub-page:horizontal {
    background-color: #89b4fa;
    border-radius: 3px;
}

/* ===== QLabel ===== */
QLabel {
    color: #cdd6f4;
    background-color: transparent;
}

QLabel#label_title {
    color: #89b4fa;
    font-size: 15px;
    font-weight: bold;
}

QLabel#label_info {
    color: #a6adc8;
    font-size: 12px;
}

QLabel#label_metric {
    color: #a6e3a1;
    font-family: "Courier New", monospace;
    font-size: 12px;
}

/* ===== QGraphicsView (ImageViewer) ===== */
QGraphicsView {
    background-color: #11111b;
    border: 1px solid #45475a;
    border-radius: 5px;
}

/* ===== ToolBar ===== */
QToolBar {
    background-color: #181825;
    border: none;
    padding: 2px;
    spacing: 4px;
}

QToolButton {
    background-color: transparent;
    color: #cdd6f4;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
}

QToolButton:hover {
    background-color: #313244;
    border-color: #45475a;
}

/* ===== Tooltip ===== */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #89b4fa;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
"""
