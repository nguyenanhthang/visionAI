"""Login window — PySide6 QDialog.

Hiển thị form đăng nhập, kết quả lưu vào self.result_employee
({"id": ..., "name": ...}) khi accept(), None khi cancel.

Scanner chạy trên QThread riêng, emit Signal → slot trên main thread.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QObject, QSize, QThread, Signal, Slot
from PySide6.QtGui import QIcon, QPainter, QColor, QBrush, QPen, QPixmap, QFont
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, QTextEdit,
)

import config
import employees
from scanner import BadgeScanner, validate_employee_api
from settings_window import SettingsDialog, gear_icon


class ManualLoginWorker(QObject):
    """Gọi API xác thực cho mã gõ tay (chạy trên QThread riêng)."""

    login_ok = Signal(dict)   # {"staff_id":..., "staff_name":...}
    error    = Signal(str)
    finished = Signal()

    def __init__(self, token_url: str, employee_url_prefix: str,
                 badge_id: str, request_timeout: float = 5.0,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.token_url = token_url
        self.employee_url_prefix = employee_url_prefix
        self.badge_id = badge_id
        self.request_timeout = request_timeout

    @Slot()
    def run(self):
        try:
            ok, data = validate_employee_api(
                self.token_url, self.employee_url_prefix,
                self.badge_id, self.request_timeout,
            )
            if ok:
                self.login_ok.emit(data)
            else:
                self.error.emit(data)
        finally:
            self.finished.emit()


# ── helpers ──────────────────────────────────────────────────
def make_brand_pixmap(size: int = 38) -> QPixmap:
    """Vẽ logo Riser cable (eye-icon trên gradient cyan) ra QPixmap."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # background rounded rect with gradient
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QLinearGradient
    grad = QLinearGradient(QPointF(0, 0), QPointF(size, size))
    grad.setColorAt(0.0, QColor("#3fb6f0"))
    grad.setColorAt(1.0, QColor("#1f6aa5"))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.26, size * 0.26)

    # eye glyph
    pen = QPen(QColor("#06121b"))
    pen.setWidthF(size * 0.12)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    cx, cy = size / 2, size / 2
    rx, ry = size * 0.35, size * 0.22

    # eye outline (lens-shape via 2 arcs)
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.moveTo(cx - rx, cy)
    path.quadTo(cx, cy - ry * 1.6, cx + rx, cy)
    path.quadTo(cx, cy + ry * 1.6, cx - rx, cy)
    p.drawPath(path)

    # pupil
    p.setBrush(QBrush(QColor("#06121b")))
    p.setPen(Qt.PenStyle.NoPen)
    pr = size * 0.12
    p.drawEllipse(QPointF(cx, cy), pr, pr)

    p.end()
    return pm


class StatusDot(QWidget):
    """Chấm trạng thái có pulse animation nhẹ."""

    def __init__(self, color: str = "#7d8590", parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(10, 10))
        self._color = QColor(color)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # glow
        glow = QColor(self._color); glow.setAlpha(80)
        p.setBrush(QBrush(glow)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, 10, 10)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(2, 2, 6, 6)
        p.end()


class LoginWindow(QDialog):
    SCANNER_COLOR_OK   = "#2ea043"
    SCANNER_COLOR_OFF  = "#7d8590"
    SCANNER_COLOR_ERR  = "#f85149"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result_employee: dict | None = None

        self.setWindowTitle("Đăng nhập — Riser cable")
        self.setWindowIcon(QIcon(make_brand_pixmap(64)))
        self.setFixedSize(480, 620)
        self.setModal(True)

        self._build_ui()
        self._setup_scanner_thread()

    # ── UI ───────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 24)
        root.setSpacing(0)

        card = QFrame(); card.setObjectName("Card")
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(36, 32, 36, 28)
        card_l.setSpacing(14)

        # brand row
        brand_row = QHBoxLayout(); brand_row.setSpacing(12)
        logo = QLabel(); logo.setPixmap(make_brand_pixmap(40))
        logo.setFixedSize(40, 40)
        brand_row.addWidget(logo)
        name = QLabel(); name.setObjectName("BrandName")
        name.setText("RISER <span style='color:#3fb6f0;'> CABLE</span>")
        name.setTextFormat(Qt.TextFormat.RichText)
        brand_row.addWidget(name)
        brand_row.addStretch(1)

        self.settings_btn = QPushButton()
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setToolTip("Cài đặt")
        self.settings_btn.setFixedSize(34, 34)
        self.settings_btn.setIcon(QIcon(gear_icon(20, "#9aa4ae")))
        self.settings_btn.setIconSize(QSize(20, 20))
        self.settings_btn.setStyleSheet(
            "QPushButton{background:#101820;border:1px solid #2a3540;border-radius:8px;}"
            "QPushButton:hover{border-color:#3fb6f0;}"
        )
        self.settings_btn.clicked.connect(self._open_settings)
        brand_row.addWidget(self.settings_btn)

        card_l.addLayout(brand_row)

        sub = QLabel("Quét thẻ nhân viên hoặc nhập mã thủ công.")
        sub.setObjectName("Muted")
        card_l.addWidget(sub)
        card_l.addSpacing(8)

        # employee id field
        lbl = QLabel("MÃ NHÂN VIÊN"); lbl.setObjectName("FieldLabel")
        card_l.addWidget(lbl)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Nhập hoặc quét thẻ…")
        self.entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entry.setMinimumHeight(42)
        self.entry.returnPressed.connect(self._handle_manual_login)
        self.entry.textChanged.connect(self._clear_field_error)
        card_l.addWidget(self.entry)

        # login button
        self.login_btn = QPushButton("ĐĂNG NHẬP →")
        self.login_btn.setObjectName("Primary")
        self.login_btn.setMinimumHeight(42)
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.clicked.connect(self._handle_manual_login)
        card_l.addWidget(self.login_btn)
        card_l.addSpacing(6)

        # scanner pill
        scanner = QFrame(); scanner.setObjectName("CardSunken")
        scanner.setProperty("chip", True)
        sl = QHBoxLayout(scanner)
        sl.setContentsMargins(12, 10, 12, 10); sl.setSpacing(10)
        self.scanner_dot = StatusDot(self.SCANNER_COLOR_OFF)
        sl.addWidget(self.scanner_dot)
        self.scanner_label = QLabel("Scanner")
        self.scanner_label.setObjectName("Muted")
        sl.addWidget(self.scanner_label)
        sl.addStretch(1)
        self.scanner_value = QLabel("đang khởi động…")
        self.scanner_value.setObjectName("Mono")
        self.scanner_value.setStyleSheet("color:#7d8590; font-size:12px;")
        sl.addWidget(self.scanner_value)
        card_l.addWidget(scanner)

        # log header
        log_hdr = QHBoxLayout()
        lh = QLabel("SYSTEM LOG"); lh.setObjectName("FieldLabel")
        log_hdr.addWidget(lh); log_hdr.addStretch(1)
        card_l.addLayout(log_hdr)

        # log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        card_l.addWidget(self.log, 1)

        root.addWidget(card, 1)

    # ── scanner wiring ───────────────────────────────────────
    def _setup_scanner_thread(self):
        self._scanner_thread = QThread(self)
        self.scanner = BadgeScanner(
            port=config.SCANNER_PORT,
            baudrate=config.SCANNER_BAUDRATE,
            read_size=config.SCANNER_READ_SIZE,
            serial_timeout=config.SCANNER_TIMEOUT,
            token_url=config.token_link,
            employee_url_prefix=config.emp_link,
            request_timeout=config.API_REQUEST_TIMEOUT,
        )
        self.scanner.moveToThread(self._scanner_thread)
        self._scanner_thread.started.connect(self.scanner.run)
        self.scanner.connected.connect(self._on_scanner_connected)
        self.scanner.disconnected.connect(self._on_scanner_disconnected)
        self.scanner.scanned.connect(self._on_scan)
        self.scanner.error.connect(self._on_scanner_error)
        self.scanner.login_ok.connect(self._on_login_ok)
        self.scanner.finished.connect(self._scanner_thread.quit)
        self._scanner_thread.start()

    # ── settings ─────────────────────────────────────────────
    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._log("Đã lưu cài đặt — đăng nhập lại để áp dụng", "ok")

    # ── logging helper ───────────────────────────────────────
    def _log(self, message: str, kind: str = "info"):
        colors = {"info": "#79c0ff", "ok": "#56d364", "err": "#ff7b72"}
        ts = datetime.now().strftime("%H:%M:%S")
        col = colors.get(kind, "#e6edf3")
        html = (
            f"<span style='color:#5b6772;'>[{ts}]</span> "
            f"<span style='color:{col};'>{message}</span>"
        )
        self.log.append(html)

    # ── scanner slots ────────────────────────────────────────
    def _on_scanner_connected(self, port: str):
        self.scanner_dot.set_color(self.SCANNER_COLOR_OK)
        self.scanner_value.setText(f"{port} · 9600 8N1")
        self.scanner_value.setStyleSheet("color:#2ea043; font-size:12px;")
        self._log(f"Scanner sẵn sàng ({port})", "ok")

    def _on_scanner_disconnected(self):
        self.scanner_dot.set_color(self.SCANNER_COLOR_OFF)
        self.scanner_value.setText("ngắt kết nối")
        self.scanner_value.setStyleSheet("color:#7d8590; font-size:12px;")

    def _on_scan(self, badge_id: str):
        self._log(f"Đã quét: {badge_id}", "info")

    def _on_scanner_error(self, msg: str):
        self._log(msg, "err")

    def _on_login_ok(self, payload: dict):
        sid = payload.get("staff_id")
        name = payload.get("staff_name", "")
        if sid:
            self._log(f"Xác thực OK: {sid} — {name}", "ok")
            self._finish_login(sid, name)

    # ── manual entry ─────────────────────────────────────────
    def _flag_field_error(self):
        self.entry.setProperty("invalid", True)
        self.entry.style().unpolish(self.entry)
        self.entry.style().polish(self.entry)

    def _clear_field_error(self, *_):
        if self.entry.property("invalid"):
            self.entry.setProperty("invalid", False)
            self.entry.style().unpolish(self.entry)
            self.entry.style().polish(self.entry)

    def _handle_manual_login(self):
        eid = self.entry.text().strip()
        if not eid:
            self._flag_field_error()
            self._log("Mã nhân viên đang để trống.", "err")
            return
        if getattr(self, "_manual_thread", None) is not None:
            return  # đang gọi API, bỏ qua click thừa

        # Không config API → fallback local lookup
        # if not config.API_TOKEN_URL or not config.API_EMPLOYEE_URL_PREFIX:
        #     name = employees.lookup(eid)
        #     if name is None:
        #         self._flag_field_error()
        #         self._log(f"Mã nhân viên '{eid}' không tồn tại.", "err")
        #         return
        #     self._finish_login(eid, name)
        #     return

        # Có config API → gọi API trên worker thread
        self.login_btn.setEnabled(False)
        self.entry.setEnabled(False)
        self._log(f"Xác thực {eid} qua API…", "info")

        self._manual_thread = QThread(self)
        self._manual_worker = ManualLoginWorker(
            token_url=config.token_link,
            employee_url_prefix=config.emp_link,
            badge_id=eid,
            request_timeout=config.API_REQUEST_TIMEOUT,
        )
        self._manual_worker.moveToThread(self._manual_thread)
        self._manual_thread.started.connect(self._manual_worker.run)
        self._manual_worker.login_ok.connect(self._on_login_ok)
        self._manual_worker.error.connect(self._on_manual_error)
        self._manual_worker.finished.connect(self._on_manual_finished)
        self._manual_worker.finished.connect(self._manual_thread.quit)
        self._manual_thread.start()

    def _on_manual_error(self, msg: str):
        self._flag_field_error()
        self._log(msg, "err")

    def _on_manual_finished(self):
        self.login_btn.setEnabled(True)
        self.entry.setEnabled(True)
        self._manual_thread = None
        self._manual_worker = None

    def _finish_login(self, employee_id: str, employee_name: str):
        self.result_employee = {"id": employee_id, "name": employee_name}
        self._stop_scanner()
        self.accept()

    # ── teardown ─────────────────────────────────────────────
    def _stop_scanner(self):
        try:
            self.scanner.stop()
            self._scanner_thread.quit()
            self._scanner_thread.wait(1500)
        except Exception:
            pass
        # cleanup manual login worker nếu còn chạy
        try:
            if getattr(self, "_manual_thread", None) is not None:
                self._manual_thread.quit()
                self._manual_thread.wait(2000)
                self._manual_thread = None
                self._manual_worker = None
        except Exception:
            pass

    def reject(self):
        self._stop_scanner()
        super().reject()

    def closeEvent(self, e):
        self._stop_scanner()
        super().closeEvent(e)
