"""Stylesheet (Qt CSS) cho HALCON Vision Studio.

Theme tối dịu lấy cảm hứng từ Cognex VisionPro: nền graphite, accent cyan/teal,
typography rõ, spacing rộng để dễ đọc.
"""

# Color tokens
BG_BASE       = "#1c2030"
BG_SURFACE    = "#252a3b"
BG_SURFACE_2  = "#2d334a"
BG_SUNKEN     = "#161927"
BORDER        = "#363c52"
BORDER_STRONG = "#454c66"
TEXT          = "#e6ecf5"
TEXT_MUTED    = "#9aa3bd"
TEXT_HEADING  = "#ffffff"
ACCENT        = "#36c5d6"   # teal/cyan, gần Cognex
ACCENT_HOVER  = "#5dd5e3"
ACCENT_PRESS  = "#1ea8b8"
WARN          = "#ffb454"
DANGER        = "#ff6b6b"
OK            = "#6cd989"

HALCON_STYLE = f"""
* {{ font-family: "Segoe UI", "Inter", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; }}

QMainWindow, QWidget {{
    background-color: {BG_BASE};
    color: {TEXT};
    font-size: 10pt;
}}

QMenuBar {{
    background-color: {BG_SUNKEN};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
}}
QMenuBar::item {{ padding: 6px 12px; border-radius: 4px; }}
QMenuBar::item:selected {{ background-color: {BG_SURFACE_2}; }}
QMenu {{
    background-color: {BG_SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px;
}}
QMenu::item {{ padding: 7px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {ACCENT}; color: #0b1220; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolBar {{
    background-color: {BG_SUNKEN};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    spacing: 4px;
}}
QToolButton {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 500;
}}
QToolButton:hover {{ background-color: {BG_SURFACE}; border-color: {BORDER}; }}
QToolButton:pressed, QToolButton:checked {{
    background-color: {ACCENT};
    color: #0b1220;
    border-color: {ACCENT};
}}

QPushButton {{
    background-color: {ACCENT};
    color: #0b1220;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 18px;
}}
QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background-color: {ACCENT_PRESS}; }}
QPushButton:disabled {{ background-color: {BG_SURFACE_2}; color: {TEXT_MUTED}; }}
QPushButton[secondary="true"] {{
    background-color: {BG_SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QPushButton[secondary="true"]:hover {{
    background-color: {BG_SURFACE_2};
    border-color: {BORDER_STRONG};
}}
QPushButton[secondary="true"]:checked {{
    background-color: {ACCENT};
    color: #0b1220;
    border-color: {ACCENT};
}}
QPushButton[ghost="true"] {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: none;
    padding: 4px 8px;
    text-align: left;
    font-weight: 500;
}}
QPushButton[ghost="true"]:hover {{ color: {TEXT}; background-color: {BG_SURFACE}; }}

QGroupBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 12px 10px 12px;
    font-weight: 600;
    color: {TEXT_HEADING};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT};
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

QLabel {{ color: {TEXT}; }}
QLabel[heading="true"] {{ font-size: 13pt; font-weight: 700; color: {TEXT_HEADING}; }}
QLabel[subheading="true"] {{ font-size: 10pt; font-weight: 600; color: {TEXT_HEADING}; }}
QLabel[muted="true"] {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QLabel[badge="true"] {{
    background-color: {BG_SURFACE_2};
    color: {ACCENT};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 8pt;
    font-weight: 700;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_SUNKEN};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {ACCENT};
    selection-color: #0b1220;
    min-height: 18px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {BG_SURFACE_2};
    border: none;
    width: 18px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BORDER_STRONG};
}}

QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #0b1220;
    padding: 4px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 16px; height: 16px;
    margin: -7px 0;
    border-radius: 8px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {BG_SUNKEN};
    color: #c4ccdf;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
    selection-background-color: {ACCENT};
    selection-color: #0b1220;
}}

QTableWidget {{
    background-color: {BG_SUNKEN};
    alternate-background-color: {BG_SURFACE};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: #0b1220;
}}
QHeaderView::section {{
    background-color: {BG_SURFACE_2};
    color: {ACCENT};
    padding: 8px 10px;
    border: none;
    border-right: 1px solid {BORDER};
    font-weight: 700;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.6px;
}}

QStatusBar {{
    background-color: {BG_SUNKEN};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
}}
QStatusBar QLabel {{ color: {TEXT_MUTED}; padding: 2px 10px; }}
QStatusBar::item {{ border: none; }}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    min-height: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    min-width: 30px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}

/* Sidebar card containers */
QFrame#sidebarCard {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* Collapsible section header (rendered as QPushButton[section="true"]) */
QPushButton[section="true"] {{
    background-color: {BG_SURFACE_2};
    color: {TEXT_HEADING};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    text-align: left;
    font-weight: 700;
    font-size: 10pt;
}}
QPushButton[section="true"]:hover {{
    background-color: {BORDER};
    border-color: {BORDER_STRONG};
}}
QPushButton[section="true"]:checked {{
    background-color: {BG_SURFACE_2};
    border-color: {ACCENT};
    color: {ACCENT};
}}

QFrame[card="true"] {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px;
}}

QToolTip {{
    background-color: {BG_SUNKEN};
    color: {TEXT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 6px 10px;
}}
"""
