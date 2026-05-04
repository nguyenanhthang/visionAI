"""Stylesheet (Qt CSS) cho HALCON Vision Studio - tone xanh dương công nghiệp."""

HALCON_STYLE = """
* { font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; }

QMainWindow, QWidget {
    background-color: #1b1f27;
    color: #e6eaf2;
}

QMenuBar {
    background-color: #11141a;
    color: #cfd6e4;
    border-bottom: 1px solid #2a3140;
    padding: 4px;
}
QMenuBar::item:selected { background-color: #2a3140; border-radius: 4px; }
QMenu {
    background-color: #1b1f27;
    color: #e6eaf2;
    border: 1px solid #2a3140;
    padding: 6px;
}
QMenu::item { padding: 6px 24px; border-radius: 4px; }
QMenu::item:selected { background-color: #2a78ff; color: white; }

QToolBar {
    background-color: #161a22;
    border-bottom: 1px solid #2a3140;
    padding: 6px;
    spacing: 6px;
}
QToolButton {
    background-color: #232936;
    color: #e6eaf2;
    border: 1px solid #2a3140;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
}
QToolButton:hover { background-color: #2d3442; border-color: #3a4456; }
QToolButton:pressed { background-color: #2a78ff; border-color: #2a78ff; }
QToolButton:checked { background-color: #2a78ff; border-color: #2a78ff; color: white; }

QPushButton {
    background-color: #2a78ff;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover { background-color: #3d87ff; }
QPushButton:pressed { background-color: #1f63db; }
QPushButton:disabled { background-color: #2a3140; color: #6b7280; }
QPushButton[secondary="true"] {
    background-color: #232936;
    color: #e6eaf2;
    border: 1px solid #2a3140;
}
QPushButton[secondary="true"]:hover { background-color: #2d3442; }

QGroupBox {
    background-color: #1f242e;
    border: 1px solid #2a3140;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    font-weight: 600;
    color: #cfd6e4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #6ea8ff;
}

QLabel { color: #cfd6e4; }
QLabel[heading="true"] { font-size: 13pt; font-weight: 700; color: #ffffff; }
QLabel[muted="true"] { color: #8a93a6; font-size: 9pt; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #11141a;
    color: #e6eaf2;
    border: 1px solid #2a3140;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #2a78ff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2a78ff;
}

QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #1b1f27;
    color: #e6eaf2;
    border: 1px solid #2a3140;
    selection-background-color: #2a78ff;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #2a3140;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2a78ff;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QTabWidget::pane {
    border: 1px solid #2a3140;
    border-radius: 8px;
    background-color: #1f242e;
    top: -1px;
}
QTabBar::tab {
    background: #161a22;
    color: #8a93a6;
    padding: 8px 16px;
    border: 1px solid #2a3140;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1f242e;
    color: #6ea8ff;
    border-color: #2a3140;
}
QTabBar::tab:hover:!selected { color: #cfd6e4; }

QTextEdit, QPlainTextEdit {
    background-color: #0e1117;
    color: #b8c2d6;
    border: 1px solid #2a3140;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
}

QTableWidget {
    background-color: #11141a;
    alternate-background-color: #161a22;
    color: #e6eaf2;
    gridline-color: #2a3140;
    border: 1px solid #2a3140;
    border-radius: 6px;
    selection-background-color: #2a78ff;
}
QHeaderView::section {
    background-color: #1f242e;
    color: #6ea8ff;
    padding: 6px;
    border: none;
    border-right: 1px solid #2a3140;
    font-weight: 600;
}

QStatusBar {
    background-color: #11141a;
    color: #8a93a6;
    border-top: 1px solid #2a3140;
}
QStatusBar QLabel { color: #8a93a6; padding: 0 6px; }

QScrollBar:vertical {
    background: #161a22;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #2a3140;
    min-height: 24px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover { background: #3a4456; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #161a22;
    height: 12px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #2a3140;
    min-width: 24px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal:hover { background: #3a4456; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSplitter::handle { background: #2a3140; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
"""
