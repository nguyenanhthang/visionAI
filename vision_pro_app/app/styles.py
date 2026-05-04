"""Vision Pro inspired Qt stylesheet.

The look is built around translucent "glass" surfaces, soft glows, generous
rounding and a deep cosmic gradient as the canvas. All values are tuned to
feel close to visionOS while remaining performant on a regular desktop.
"""

VISION_PRO_STYLE = """
/* ---------- Window ---------- */
QMainWindow, QDialog {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #06010f,
        stop:0.45 #160a32,
        stop:0.85 #0a0420,
        stop:1 #02010a);
}

QWidget {
    color: rgba(255, 255, 255, 0.92);
    font-family: "SF Pro Display", "Segoe UI Variable", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ---------- Glass surfaces ---------- */
QFrame#GlassPanel {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 26px;
}

QFrame#GlassPanel[variant="strong"] {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

QFrame#HeaderBar {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 22px;
}

QFrame#StatusBar {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
}

QLabel#AppTitle {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.4px;
    color: white;
}

QLabel#AppSubtitle {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.55);
    letter-spacing: 0.6px;
}

QLabel#SectionTitle {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: rgba(255, 255, 255, 0.55);
    text-transform: uppercase;
    padding: 4px 2px;
}

QLabel#FieldLabel {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.70);
}

QLabel#FieldValue {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.95);
    font-weight: 600;
}

QLabel#StatusText {
    color: rgba(255, 255, 255, 0.65);
    font-size: 12px;
}

/* ---------- Buttons ---------- */
QPushButton {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    padding: 9px 18px;
    color: rgba(255, 255, 255, 0.95);
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background: rgba(255, 255, 255, 0.18);
    border-color: rgba(255, 255, 255, 0.32);
    color: white;
}

QPushButton:pressed {
    background: rgba(255, 255, 255, 0.26);
    border-color: rgba(255, 255, 255, 0.45);
}

QPushButton:disabled {
    color: rgba(255, 255, 255, 0.30);
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.06);
}

QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(120, 132, 255, 0.85),
        stop:1 rgba(208, 120, 255, 0.85));
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: white;
    font-weight: 600;
}

QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(140, 150, 255, 0.95),
        stop:1 rgba(228, 140, 255, 0.95));
}

QPushButton#PrimaryButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(100, 112, 235, 1.0),
        stop:1 rgba(188, 100, 235, 1.0));
}

QPushButton#IconButton {
    padding: 8px 12px;
    border-radius: 14px;
    min-width: 36px;
}

QPushButton#FilterTile {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 14px 6px;
    text-align: center;
    font-size: 12px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.85);
    min-height: 64px;
}

QPushButton#FilterTile:hover {
    background: rgba(255, 255, 255, 0.14);
    border-color: rgba(255, 255, 255, 0.28);
    color: white;
}

QPushButton#FilterTile:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(120, 132, 255, 0.55),
        stop:1 rgba(208, 120, 255, 0.55));
    border: 1px solid rgba(255, 255, 255, 0.50);
    color: white;
}

QPushButton#ToggleChip {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 14px;
    padding: 8px 14px;
    font-size: 12px;
}

QPushButton#ToggleChip:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(120, 132, 255, 0.7),
        stop:1 rgba(208, 120, 255, 0.7));
    border: 1px solid rgba(255, 255, 255, 0.45);
    color: white;
}

/* ---------- Sliders ---------- */
QSlider::groove:horizontal {
    background: rgba(255, 255, 255, 0.10);
    height: 6px;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(120, 132, 255, 0.85),
        stop:1 rgba(208, 120, 255, 0.85));
    border-radius: 3px;
}

QSlider::add-page:horizontal {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: white;
    border: 2px solid rgba(255, 255, 255, 0.85);
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #f4f0ff;
    border: 2px solid rgba(208, 120, 255, 0.8);
}

/* ---------- Tabs ---------- */
QTabWidget::pane {
    border: none;
    background: transparent;
    margin-top: 6px;
}

QTabBar {
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.65);
    padding: 8px 18px;
    margin-right: 6px;
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
    border-bottom-left-radius: 14px;
    border-bottom-right-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 12px;
    font-weight: 500;
}

QTabBar::tab:hover {
    background: rgba(255, 255, 255, 0.10);
    color: white;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(120, 132, 255, 0.55),
        stop:1 rgba(208, 120, 255, 0.55));
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.40);
}

/* ---------- Scroll area + tab pages ----------
   Force every container inside the tab/scroll machinery to be transparent so
   the underlying glass panel bleeds through. */
QScrollArea {
    background: transparent;
    border: none;
}

QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {
    background: transparent;
}

QTabWidget,
QTabWidget > QStackedWidget,
QTabWidget > QStackedWidget > QWidget {
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.30);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    border: none;
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 4px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(255, 255, 255, 0.30);
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
    border: none;
    width: 0;
}

/* ---------- Combo box ---------- */
QComboBox {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 12px;
    padding: 6px 12px;
    min-height: 26px;
    color: white;
}

QComboBox:hover {
    border-color: rgba(255, 255, 255, 0.30);
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background: #15102a;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 10px;
    color: white;
    selection-background-color: rgba(120, 132, 255, 0.55);
    padding: 4px;
}

/* ---------- Tooltip ---------- */
QToolTip {
    background: rgba(20, 12, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: white;
    padding: 6px 10px;
    border-radius: 10px;
    font-size: 12px;
}

/* ---------- Image canvas ---------- */
QFrame#ImageCanvas {
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 24px;
}

QLabel#ImageDisplay {
    background: transparent;
    color: rgba(255, 255, 255, 0.40);
    font-size: 14px;
}

QLabel#DropHint {
    color: rgba(255, 255, 255, 0.45);
    font-size: 13px;
    background: transparent;
}

/* ---------- Histogram ---------- */
QFrame#Histogram {
    background: rgba(0, 0, 0, 0.30);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
}
"""
