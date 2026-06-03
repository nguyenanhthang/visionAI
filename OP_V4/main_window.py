"""Main window — PySide6 QMainWindow.

Layout giống mockup HTML:
    ┌──────────────── topbar (logo · station · status chips · clock · user) ────────┐
    │ ┌──────────────────────┐ ┌─ Phiên làm việc ───────────┐                       │
    │ │                      │ │  info rows                  │                       │
    │ │   Image panel        │ │ ── stats: TOTAL · OK · NG ──│                       │
    │ │                      │ │  yield bar                  │                       │
    │ │                      │ ├─ Log hoạt động ─────────────┤                       │
    │ │                      │ │  log textbox                │                       │
    │ └──────────────────────┘ └─────────────────────────────┘                       │
    └─────────────────── statusbar (PLC · scanner · uptime) ────────────────────────┘
"""

from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QSize, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush, QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

import config
from login_window import StatusDot, make_brand_pixmap
from plc_worker import SimulatedPLCWorker, H3U_PLCWorker
from scanner import ProductScanner
from settings_window import SettingsDialog, gear_icon


# ── tiny widgets ─────────────────────────────────────────────────
class StatusChip(QFrame):
    """Pill ngang ở topbar: ● label."""

    def __init__(self, text: str, color: str = "#2ea043", parent=None):
        super().__init__(parent)
        self.setProperty("chip", True)
        self.setObjectName("Chip")
        self.setStyleSheet(
            "QFrame#Chip{background:#101820;border:1px solid #2a3540;border-radius:13px;}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 12, 5); lay.setSpacing(8)
        self.dot = StatusDot(color)
        lay.addWidget(self.dot)
        self.label = QLabel(text)
        self.label.setStyleSheet("color:#9aa4ae;font-size:12px;background:transparent;")
        lay.addWidget(self.label)

    def set_state(self, text: str, color: str):
        self.label.setText(text)
        self.dot.set_color(color)



class OplImageWorker(QObject):
    """Tìm subfolder mới nhất + ảnh mới nhất → load + upload.

    - Subfolder + ảnh đều chọn theo mtime giảm dần để không lệ
      thuộc vào quy ước đặt tên (YYYYMMDD, test1/test2, …).
    - Upload = shutil.copy2 sang upload_dir (UNC share của hệ thống).
    - File đích đổi tên: ``{sn}_{YYYY.MM.DD HH.MM.SS}_{verdict_label}{ext}``
      ví dụ ``P1715102-98-D_SFVN26139D00055_2026.05.21 07.25.36_Passed.png``.
      Đặt vào subfolder ngày hôm nay (``YYYYMMDD``).
    """

    image_ready = Signal(str, QImage)   # filename, QImage
    upload_done = Signal(bool, str)     # ok, message
    error       = Signal(str)
    finished    = Signal()

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root_dir: str, upload_dir: str = "",
                 sn: str = "", verdict_label: str = "",
                 parent: QObject | None = None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.upload_dir = upload_dir
        self.sn = sn or "UNKNOWN"
        self.verdict_label = verdict_label or "Unknown"

    @Slot()
    def run(self):
        try:
            root = Path(self.root_dir)
            if not root.exists():
                self.error.emit(f"Folder gốc không tồn tại: {root}")
                return

            subfolders = sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not subfolders:
                self.error.emit(f"{root.name} không có subfolder nào")
                return
            latest_folder = subfolders[0]

            imgs = sorted(
                (p for p in latest_folder.iterdir()
                 if p.is_file() and p.suffix.lower() in self._IMG_EXTS),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not imgs:
                self.error.emit(f"Folder {latest_folder.name} không có ảnh")
                return
            latest_img = imgs[0]

            qimg = QImage(str(latest_img))
            if qimg.isNull():
                self.error.emit(f"Không load được ảnh {latest_img.name}")
                return
            self.image_ready.emit(f"{latest_folder.name}/{latest_img.name}", qimg)

            if self.upload_dir:
                try:
                    now = datetime.now()
                    today = now.strftime("%Y%m%d")
                    ts = now.strftime("%Y.%m.%d %H.%M.%S")
                    new_name = f"{self.sn}_{ts}_{self.verdict_label}{latest_img.suffix}"
                    dest_parent = Path(self.upload_dir) / today
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    dest = dest_parent / new_name
                    shutil.copy2(str(latest_img), str(dest))
                    self.upload_done.emit(True, f"đã copy → {dest.name}")
                except Exception as exc:
                    self.upload_done.emit(False, f"upload lỗi: {exc}")
        except Exception as exc:
            self.error.emit(repr(exc))
        finally:
            self.finished.emit()


class SfcPushWorker(QObject):
    """POST kết quả verdict lên MES clipThroughStation.

    Payload: {sn, stationName, empNo, result}. Response code=200 → ok,
    khác → log lỗi (vd 406 "下一制程为 EOL-S").
    """

    done     = Signal(bool, str)   # ok, message
    finished = Signal()

    def __init__(self, url: str, payload: dict, timeout: float = 5.0,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.url = url
        self.payload = payload
        self.timeout = timeout

    @Slot()
    def run(self):
        sn = self.payload.get("sn", "")
        result = self.payload.get("result", "")
        try:
            import requests
            r = requests.post(self.url, json=self.payload, timeout=self.timeout)
            try:
                body = r.json()
            except Exception:
                body = {}
            code = body.get("code")
            msg  = body.get("msg", "")
            if code == 200:
                self.done.emit(True, f"SFC {sn} {result} ok")
            else:
                self.done.emit(False, f"SFC {sn} {result} fail code={code} {msg}")
        except Exception as exc:
            self.done.emit(False, f"SFC {sn} {result} lỗi: {exc}")
        finally:
            self.finished.emit()


class DataExportWorker(QObject):
    """Append 1 dòng đo sang file .xls log theo ngày khi có verdict PLC.

    - Nguồn: ``<src_dir>/<ngày>.xls`` — lấy cột B→I (8 giá trị) của
      DÒNG DỮ LIỆU MỚI NHẤT (dòng cuối còn dữ liệu).
    - Đích:  ``<dst_dir>/<ngày>.xls`` — append dòng
      ``[times, SN, L1-1, L1-2, L2-1, L2-2, L3-1, L3-2, L4-1, L4-2]``;
      header ghi 1 lần ở đầu file.
    - ``times`` = giờ nhận verdict, ``SN`` = mã sản phẩm đang quét.

    xlwt không append trực tiếp .xls → đọc lại toàn bộ file đích rồi
    ghi lại (header + dòng cũ + dòng mới) dưới ``_file_lock`` để 2 verdict
    sát nhau không ghi đè lẫn nhau.
    """

    done     = Signal(bool, str)   # ok, message
    finished = Signal()

    HEADER = ["times", "SN", "L1-1", "L1-2", "L2-1", "L2-2",
              "L3-1", "L3-2", "L4-1", "L4-2"]
    _SRC_COLS = range(1, 9)          # cột B..I (0-based: 1..8)
    _file_lock = threading.Lock()    # serialize ghi file đích

    def __init__(self, src_dir: str, dst_dir: str, sn: str = "",
                 date_fmt: str = "%Y%m%d", parent: QObject | None = None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.sn = sn or ""
        self.date_fmt = date_fmt or "%Y%m%d"

    @Slot()
    def run(self):
        try:
            import xlrd
            import xlwt
        except ImportError:
            self.done.emit(False, "Thiếu thư viện: pip install xlrd xlwt")
            self.finished.emit()
            return
        try:
            now = datetime.now()
            day = now.strftime(self.date_fmt)
            src = Path(self.src_dir) / f"{day}.xls"
            if not src.exists():
                self.done.emit(False, f"File .xls nguồn không tồn tại: {src.name}")
                return

            measures = self._read_latest_measures(xlrd, src)
            if measures is None:
                self.done.emit(False, f"{src.name} không có dòng dữ liệu")
                return

            row = [now.strftime("%Y-%m-%d %H:%M:%S"), self.sn, *measures]

            dst = Path(self.dst_dir) / f"{day}.xls"
            with self._file_lock:
                Path(self.dst_dir).mkdir(parents=True, exist_ok=True)
                old_rows = self._read_existing(xlrd, dst) if dst.exists() else []
                self._write_all(xlwt, dst, old_rows + [row])

            self.done.emit(True, f"Lưu data → {dst.name} (+1 dòng, SN {self.sn or '—'})")
        except Exception as exc:
            self.done.emit(False, f"Export .xls lỗi: {exc}")
        finally:
            self.finished.emit()

    # ── helpers ──────────────────────────────────────────────
    @classmethod
    def _read_latest_measures(cls, xlrd, path: Path):
        """Cột B→I của dòng cuối còn dữ liệu. None nếu file rỗng."""
        sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
        for r in range(sheet.nrows - 1, -1, -1):
            cells = [cls._norm(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            if any(v != "" for v in cells):
                return [cells[c] if c < len(cells) else "" for c in cls._SRC_COLS]
        return None

    @classmethod
    def _read_existing(cls, xlrd, path: Path) -> list[list]:
        """Đọc lại các dòng dữ liệu cũ của file đích (bỏ header)."""
        try:
            sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
        except Exception:
            return []
        rows = []
        for r in range(sheet.nrows):
            vals = [cls._norm(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            if r == 0 and vals[:1] == [cls.HEADER[0]]:
                continue  # bỏ header cũ — sẽ ghi lại
            if any(v != "" for v in vals):
                rows.append(vals)
        return rows

    @classmethod
    def _write_all(cls, xlwt, path: Path, rows: list[list]):
        wb = xlwt.Workbook(encoding="utf-8")
        ws = wb.add_sheet("data")
        for c, h in enumerate(cls.HEADER):
            ws.write(0, c, h)
        for ri, row in enumerate(rows, start=1):
            for c, val in enumerate(row):
                ws.write(ri, c, val)
        wb.save(str(path))

    @staticmethod
    def _norm(v):
        """xls trả số dạng float — số nguyên thì bỏ đuôi '.0'; None → ''."""
        if v is None:
            return ""
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v


# ── main window ──────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self, employee_id: str, employee_name: str = "", parent=None):
        super().__init__(parent)
        self.employee_id = employee_id
        self.employee_name = employee_name or "—"
        self._started_at = time.time()
        self._current_sn = ""

        self.setWindowTitle("Riser cable — Giao diện chính")
        self.setWindowIcon(QIcon(make_brand_pixmap(64)))
        self.resize(560, 780)
        self.setMinimumSize(460, 600)

        self._build_ui()
        self._setup_clock()
        self._setup_plc()
        self._setup_product_scanner()

        self._log("Đăng nhập thành công", "SYS",
                  detail=f"{employee_id} — {self.employee_name}")

    # ── UI build ─────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        root.addWidget(self._build_topbar())

        body = QFrame(); body.setStyleSheet("background:#0b1015;border:none;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 14, 14, 14); bl.setSpacing(14)
        body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        bl.addWidget(self._build_info_card(), 0)
        bl.addWidget(self._build_log_card(), 1)

        root.addWidget(body, 1)
        root.addWidget(self._build_statusbar())

    # ── topbar ───────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame(); bar.setObjectName("TopBar")
        bar.setStyleSheet(
            "QFrame#TopBar{background:#0a0e13;border-bottom:1px solid #2a3540;}"
            "QLabel{background:transparent;}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 12, 20, 12); lay.setSpacing(14)

        logo = QLabel(); logo.setPixmap(make_brand_pixmap(34))
        logo.setFixedSize(34, 34)
        lay.addWidget(logo)

        name = QLabel("RISER <span style='color:#3fb6f0;'> CABLE</span>")
        name.setTextFormat(Qt.TextFormat.RichText)
        name.setStyleSheet("font-size:16px;font-weight:700;letter-spacing:2px;")
        lay.addWidget(name)

        self.station_lbl = QLabel(config.STATION_NAME)
        self.station_lbl.setObjectName("Station")
        self.station_lbl.setStyleSheet(
            "font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;"
            "color:#9aa4ae;background:#101820;border:1px solid #2a3540;"
            "padding:5px 10px;border-radius:6px;"
        )
        lay.addWidget(self.station_lbl)

        lay.addStretch(1)

        self.plc_chip     = StatusChip("PLC offline", "#7d8590")
        self.scanner_chip = StatusChip("Scanner offline", "#7d8590")
        self.cycle_chip   = StatusChip("Cycle —", "#3fb6f0")
        for c in (self.plc_chip, self.scanner_chip, self.cycle_chip):
            lay.addWidget(c)

        self.clock_lbl = QLabel("--:--:--")
        self.clock_lbl.setStyleSheet(
            "font-family:'JetBrains Mono',Consolas,monospace;font-size:13px;"
            "padding-left:10px;border-left:1px solid #2a3540;"
        )
        lay.addWidget(self.clock_lbl)

        # settings button (gear icon)
        self.settings_btn = QPushButton()
        self.settings_btn.setObjectName("Ghost")
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setToolTip("Cài đặt")
        self.settings_btn.setFixedSize(34, 30)
        self.settings_btn.setIcon(QIcon(gear_icon(20, "#9aa4ae")))
        self.settings_btn.setIconSize(QSize(20, 20))
        self.settings_btn.setAutoDefault(False)
        self.settings_btn.setDefault(False)
        self.settings_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings_btn.setStyleSheet(
            "QPushButton{background:#101820;border:1px solid #2a3540;border-radius:6px;}"
            "QPushButton:hover{border-color:#3fb6f0;}"
        )
        self.settings_btn.clicked.connect(self._open_settings)
        lay.addWidget(self.settings_btn)

        # user
        user_box = QFrame(); user_box.setStyleSheet("background:transparent;border:none;")
        ul = QHBoxLayout(user_box); ul.setContentsMargins(14, 0, 0, 0); ul.setSpacing(10)
        user_box.setStyleSheet("QFrame{border:none;background:transparent;}")
        # avatar
        avatar = QLabel(); avatar.setFixedSize(30, 30)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        initial = (self.employee_name or "?")[:1].upper()
        pm = QPixmap(30, 30); pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        from PySide6.QtCore import QPointF
        grad = QLinearGradient(QPointF(0, 0), QPointF(30, 30))
        grad.setColorAt(0.0, QColor("#3fb6f0")); grad.setColorAt(1.0, QColor("#1f6aa5"))
        p.setBrush(QBrush(grad)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, 30, 30)
        p.setPen(QPen(QColor("#06121b")))
        f = QFont("Inter"); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, initial)
        p.end()
        avatar.setPixmap(pm)
        ul.addWidget(avatar)

        umeta = QVBoxLayout(); umeta.setSpacing(0); umeta.setContentsMargins(0, 0, 0, 0)
        un = QLabel(self.employee_name); un.setStyleSheet("font-size:13px;font-weight:600;")
        ui_ = QLabel(f"ID {self.employee_id}")
        ui_.setStyleSheet("font-size:11px;color:#7d8590;font-family:'JetBrains Mono',Consolas,monospace;")
        umeta.addWidget(un); umeta.addWidget(ui_)
        ul.addLayout(umeta)
        user_box.setStyleSheet("QFrame{background:transparent;border:none;}QLabel{background:transparent;}")
        # left border
        sep = QFrame(); sep.setFixedWidth(1)
        sep.setStyleSheet("background:#2a3540;border:none;")
        lay.addWidget(sep)
        lay.addWidget(user_box)

        return bar

    # ── info card ────────────────────────────────────────────
    def _build_info_card(self) -> QWidget:
        card = QFrame(); card.setObjectName("Card")
        card.setStyleSheet(
            "QFrame#Card{background:#141b22;border:1px solid #2a3540;border-radius:12px;}"
            "QLabel{background:transparent;}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        lay.addWidget(self._panel_header("PHIÊN LÀM VIỆC", f"bắt đầu {datetime.now().strftime('%H:%M')}", _user_icon))

        rows = QFrame(); rl = QVBoxLayout(rows)
        rl.setContentsMargins(18, 4, 18, 14); rl.setSpacing(0)
        self._product_lbl = self._info_row(rl, "Product ID", "—")
        self._info_row(rl, "Mã nhân viên", self.employee_id)
        self._info_row(rl, "Tên nhân viên", self.employee_name)
        self._info_row(rl, "Trạm", config.STATION_NAME)
        lay.addWidget(rows)

        return card

    def _info_row(self, parent_layout, key: str, value: str) -> QLabel:
        row = QHBoxLayout(); row.setContentsMargins(0, 9, 0, 9); row.setSpacing(8)
        k = QLabel(key); k.setStyleSheet("font-size:12px;color:#7d8590;")
        v = QLabel(value); v.setStyleSheet(
            "font-size:13px;font-weight:600;font-family:'JetBrains Mono',Consolas,monospace;color:#e6edf3;"
        )
        v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(k); row.addStretch(1); row.addWidget(v)
        # divider
        wrap = QVBoxLayout(); wrap.setSpacing(0); wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addLayout(row)
        sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet("background:#1f2933;border:none;")
        wrap.addWidget(sep)
        parent_layout.addLayout(wrap)
        return v

    # ── log card ─────────────────────────────────────────────
    def _build_log_card(self) -> QWidget:
        card = QFrame(); card.setObjectName("Card")
        card.setStyleSheet(
            "QFrame#Card{background:#141b22;border:1px solid #2a3540;border-radius:12px;}"
            "QLabel{background:transparent;}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        lay.addWidget(self._panel_header("LOG HOẠT ĐỘNG", "auto-scroll", _list_icon))

        wrap = QFrame()
        wl = QVBoxLayout(wrap); wl.setContentsMargins(14, 4, 14, 14); wl.setSpacing(0)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setStyleSheet(
            "QTextEdit{background:#0a0f14;border:1px solid #2a3540;border-radius:8px;"
            "padding:8px;color:#e6edf3;"
            "font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;}"
        )
        wl.addWidget(self.log_view)
        lay.addWidget(wrap, 1)
        return card

    # ── panel header ─────────────────────────────────────────
    def _panel_header(self, title: str, meta: str, icon_fn, border_top=False) -> QWidget:
        h = QFrame()
        style = "background:transparent;"
        if border_top:
            style += "border-top:1px solid #2a3540;"
        h.setStyleSheet(f"QFrame{{{style}border-left:none;border-right:none;border-bottom:1px solid #2a3540;}}"
                        "QLabel{background:transparent;border:none;}")
        l = QHBoxLayout(h); l.setContentsMargins(16, 14, 16, 12); l.setSpacing(10)
        ic = QLabel(); ic.setPixmap(icon_fn(18, "#3fb6f0")); ic.setFixedSize(18, 18)
        l.addWidget(ic)
        t = QLabel(title)
        t.setStyleSheet("color:#9aa4ae;font-size:11px;font-weight:700;letter-spacing:3px;")
        l.addWidget(t); l.addStretch(1)
        m = QLabel(meta)
        m.setStyleSheet("color:#7d8590;font-size:11px;font-family:'JetBrains Mono',Consolas,monospace;")
        l.addWidget(m)
        return h

    # ── statusbar ────────────────────────────────────────────
    def _build_statusbar(self) -> QWidget:
        bar = QFrame(); bar.setObjectName("StatusBar")
        bar.setStyleSheet(
            "QFrame#StatusBar{background:#0a0e13;border-top:1px solid #2a3540;}"
            "QLabel{background:transparent;color:#7d8590;font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;}"
        )
        l = QHBoxLayout(bar); l.setContentsMargins(20, 8, 20, 8); l.setSpacing(20)

        self.sb_plc     = self._sb_item("PLC offline", "#7d8590")
        self.sb_scanner = self._sb_item("Scanner offline", "#7d8590")
        self.sb_cycle   = self._sb_item("Cycle —", "#3fb6f0")
        for w in (self.sb_plc, self.sb_scanner, self.sb_cycle):
            l.addWidget(w)
        l.addStretch(1)
        self.sb_uptime = QLabel("Uptime 0s")
        l.addWidget(self.sb_uptime)
        return bar

    def _sb_item(self, text: str, color: str) -> QWidget:
        w = QFrame(); w.setStyleSheet("background:transparent;border:none;")
        h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(7)
        dot = StatusDot(color); h.addWidget(dot)
        lbl = QLabel(text); h.addWidget(lbl)
        w.dot = dot; w.lbl = lbl
        return w

    # ── clock + uptime ───────────────────────────────────────
    def _setup_clock(self):
        self._clock_timer = QTimer(self); self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._tick)
        self._clock_timer.start(); self._tick()

    def _tick(self):
        now = datetime.now()
        self.clock_lbl.setText(now.strftime("%H:%M:%S · %d/%m/%Y"))
        up = int(time.time() - self._started_at)
        h, m, s = up // 3600, (up % 3600) // 60, up % 60
        self.sb_uptime.setText(f"Uptime {h}h {m:02d}m {s:02d}s")

    # ── PLC wiring ───────────────────────────────────────────
    def _setup_plc(self):
        self._plc_thread = QThread(self)
        if config.PLC_SIMULATED:
            self.plc = SimulatedPLCWorker(poll_interval=1.5)
            self._set_chip(self.scanner_chip, "Scanner offline", "#7d8590")
            self.sb_scanner.dot.set_color("#7d8590")
        else:
            self.plc = H3U_PLCWorker(
                ip=config.PLC_IP,
                result_addr=config.PLC_RESULT_ADDR,
                poll_interval=1.0 / max(config.PLC_POLL_HZ, 1),
            )
        self.plc.moveToThread(self._plc_thread)
        self._plc_thread.started.connect(self.plc.run)
        self.plc.connected.connect(self._on_plc_connected)
        self.plc.disconnected.connect(self._on_plc_disconnected)
        self.plc.error.connect(lambda e: self._log(e, "PLC", level="err"))
        self.plc.fatal.connect(lambda e: self._log(e, "PLC", level="err"))
        self.plc.result.connect(self._on_plc_result)
        self.plc.finished.connect(self._plc_thread.quit)
        self._plc_thread.start()
        self._last_result_ts = time.time()

    def _on_plc_connected(self):
        self._set_chip(self.plc_chip, "PLC online", "#2ea043")
        self.sb_plc.lbl.setText(f"PLC · {config.PLC_IP}:502")
        self.sb_plc.dot.set_color("#2ea043")
        self._log("PLC connected", "PLC", level="ok")

    def _on_plc_disconnected(self):
        self._set_chip(self.plc_chip, "PLC offline", "#7d8590")
        self.sb_plc.dot.set_color("#7d8590")
        self._log("PLC disconnected", "PLC")

    # ── product scanner wiring ───────────────────────────────
    def _setup_product_scanner(self):
        self._scan_thread = QThread(self)
        self.product_scanner = ProductScanner(
            port=config.SCANNER_PORT,
            baudrate=config.SCANNER_BAUDRATE,
            read_size=config.PRODUCT_READ_SIZE,
            serial_timeout=config.SCANNER_TIMEOUT,
            sn_check_prefix=config.sn_link1,
            sn_check_suffix=config.sn_link2,
            plc_ip=config.PLC_IP,
            plc_scan_result_addr=config.PLC_SCAN_RESULT_ADDR,
            request_timeout=config.API_REQUEST_TIMEOUT,
        )
        self.product_scanner.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self.product_scanner.run)
        self.product_scanner.connected.connect(self._on_scanner_connected)
        self.product_scanner.disconnected.connect(self._on_scanner_disconnected)
        self.product_scanner.scanned.connect(self._on_product_scanned)
        self.product_scanner.verdict.connect(self._on_product_verdict)
        self.product_scanner.error.connect(lambda e: self._log(e, "SYS", level="err"))
        self.product_scanner.finished.connect(self._scan_thread.quit)
        self._scan_thread.start()

    def _on_scanner_connected(self, port: str):
        self._set_chip(self.scanner_chip, f"Scanner {port}", "#2ea043")
        self.sb_scanner.dot.set_color("#2ea043")
        self.sb_scanner.lbl.setText(f"Scanner · {port}")
        self._log(f"Scanner sản phẩm sẵn sàng ({port})", "SYS", level="ok")

    def _on_scanner_disconnected(self):
        self._set_chip(self.scanner_chip, "Scanner offline", "#7d8590")
        self.sb_scanner.dot.set_color("#7d8590")

    def _on_product_scanned(self, code: str):
        self._current_sn = code
        self._product_lbl.setText(code)
        self._log(f"Đã quét mã SP: {code}", "SYS")

    def _on_product_verdict(self, code: str, api_ok: bool, plc_value: int):
        tag = "OK" if api_ok else "NG"
        level = "ok" if api_ok else "err"
        api_txt = "200 OK" if api_ok else "fail"
        self._log(
            f"{code} → API {api_txt}, ghi reg {config.PLC_SCAN_RESULT_ADDR}={plc_value}",
            tag, level=level,
        )

    # ── OPL upload (auto trigger sau mỗi PLC verdict) ────────
    def _trigger_opl_upload(self, sn: str, verdict_label: str, root_dir: str):
        if not getattr(config, "ON_OFF_SFC", True):
            return
        if getattr(self, "_opl_thread", None) is not None:
            return  # đang chạy, bỏ qua trigger trùng
        self._log(f"Tìm ảnh {verdict_label} mới nhất cho {sn} trong {root_dir}…", "SYS")

        self._opl_thread = QThread(self)
        self._opl_worker = OplImageWorker(
            root_dir=root_dir,
            upload_dir=config.link_post_img,
            sn=sn,
            verdict_label=verdict_label,
        )
        self._opl_worker.moveToThread(self._opl_thread)
        self._opl_thread.started.connect(self._opl_worker.run)
        self._opl_worker.image_ready.connect(self._on_opl_image_ready)
        self._opl_worker.upload_done.connect(self._on_opl_upload_done)
        self._opl_worker.error.connect(lambda e: self._log(e, "SYS", level="err"))
        self._opl_worker.finished.connect(self._on_opl_finished)
        self._opl_worker.finished.connect(self._opl_thread.quit)
        self._opl_thread.start()

    def _on_opl_image_ready(self, name: str, _qimg: QImage):
        self._log(f"Đã load ảnh: {name}", "SYS", level="ok")

    def _on_opl_upload_done(self, ok: bool, msg: str):
        level = "ok" if ok else "err"
        self._log(f"Upload: {msg}", "SYS", level=level)

    def _on_opl_finished(self):
        self._opl_thread = None
        self._opl_worker = None

    # ── Data export .xls (auto trigger sau mỗi PLC verdict) ───
    def _trigger_data_export(self, sn: str):
        src = getattr(config, "DATA_SRC_DIR", "")
        dst = getattr(config, "DATA_EXPORT_DIR", "")
        if not src or not dst:
            return  # chưa cấu hình folder → bỏ qua
        thread = QThread(self)
        worker = DataExportWorker(
            src_dir=src,
            dst_dir=dst,
            sn=sn or self._current_sn,
            date_fmt=getattr(config, "DATA_FILE_DATEFMT", "%Y%m%d"),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_data_export_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._data_jobs = getattr(self, "_data_jobs", [])
        self._data_jobs = [(t, w) for (t, w) in self._data_jobs if t.isRunning()]
        self._data_jobs.append((thread, worker))
        thread.start()

    def _on_data_export_done(self, ok: bool, msg: str):
        self._log(msg, "SYS", level=("ok" if ok else "err"))

    def _on_plc_result(self, data: dict):
        pid = data.get("product_id", "") or self._current_sn
        ok = data.get("ok")
        result = data.get("result", "")  # "PASS" / "FAIL" cho clipThroughStation
        # cycle
        now = time.time()
        cycle = now - self._last_result_ts
        self._last_result_ts = now
        self._set_chip(self.cycle_chip, f"Cycle {cycle:.2f}s", "#3fb6f0")
        self.sb_cycle.lbl.setText(f"Cycle {cycle:.2f}s")
        # info
        if pid:
            self._product_lbl.setText(pid)
        # log verdict
        label = result or ("PASS" if ok else "FAIL")
        tag, level = ("OK", "ok") if ok else ("NG", "err")
        self._log(f"AOI verdict: {pid or '—'} → {label}", tag, level=level)
        # đẩy kết quả lên SFC + đẩy ảnh OPL
        if result in ("PASS", "FAIL"):
            if result == "PASS":
                verdict_label = "Passed"
                root_dir = config.OPL_OK_DIR
            else:
                verdict_label = "Failed"
                root_dir = config.OPL_NG_DIR
            self._push_sfc_result(pid, result)
            self._trigger_opl_upload(pid, verdict_label, root_dir)
            self._trigger_data_export(pid)

    # ── SFC clipThroughStation push ──────────────────────────
    def _push_sfc_result(self, sn: str, result: str):
        if not getattr(config, "ON_OFF_SFC", True):
            self._log(f"SFC tắt — bỏ qua push {sn} {result}", "SYS")
            return
        if not config.link_sfc:
            return
        payload = {
            "sn": sn,
            "stationName": config.STATION_NAME,
            "empNo": self.employee_id,
            "result": result,
        }
        thread = QThread(self)
        worker = SfcPushWorker(config.link_sfc, payload,
                               timeout=config.API_REQUEST_TIMEOUT)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_sfc_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._sfc_jobs = getattr(self, "_sfc_jobs", [])
        self._sfc_jobs.append((thread, worker))
        thread.start()

    def _on_sfc_done(self, ok: bool, msg: str):
        self._log(msg, "SYS", level=("ok" if ok else "err"))

    # ── settings ─────────────────────────────────────────────
    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            # vài thứ apply được ngay
            self.station_lbl.setText(config.STATION_NAME)
            self._log("Đã lưu cài đặt", "SYS", level="ok")

    # ── log ──────────────────────────────────────────────────
    def _log(self, message: str, tag: str = "INFO", level: str = "info", detail: str = ""):
        tag_colors = {
            "OK":  "#7ee787",
            "NG":  "#ffa198",
            "PLC": "#d2a8ff",
            "SYS": "#9aa4ae",
            "INFO":"#79c0ff",
        }
        msg_colors = {"info":"#e6edf3","ok":"#56d364","err":"#ff7b72","warn":"#d29922"}
        ts = datetime.now().strftime("%H:%M:%S")
        tcol = tag_colors.get(tag, "#79c0ff")
        mcol = msg_colors.get(level, "#e6edf3")
        det = f" <span style='color:#7d8590'>· {detail}</span>" if detail else ""
        html = (
            f"<span style='color:#5b6772;'>[{ts}]</span>&nbsp;"
            f"<span style='color:{tcol};font-weight:700;'>{tag}</span>&nbsp;&nbsp;"
            f"<span style='color:{mcol}'>{message}</span>{det}"
        )
        self.log_view.append(html)

    # ── chip helper ──────────────────────────────────────────
    def _set_chip(self, chip: StatusChip, text: str, color: str):
        chip.set_state(text, color)

    # ── close ────────────────────────────────────────────────
    def closeEvent(self, e):
        try:
            self.product_scanner.stop()
            self._scan_thread.quit()
            self._scan_thread.wait(1500)
        except Exception:
            pass
        try:
            self.plc.stop()
            self._plc_thread.quit()
            self._plc_thread.wait(2000)
        except Exception:
            pass
        super().closeEvent(e)


# ── extra icons (drawn vector) ───────────────────────────────────
def _user_icon(size: int, color: str) -> QPixmap:
    pm = QPixmap(size, size); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.6); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    from PySide6.QtCore import QRectF
    p.drawEllipse(QRectF(size*0.30, size*0.14, size*0.40, size*0.40))
    p.drawArc(QRectF(size*0.14, size*0.50, size*0.72, size*0.72), 0, 180 * 16)
    p.end()
    return pm


def _list_icon(size: int, color: str) -> QPixmap:
    pm = QPixmap(size, size); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color)); pen.setWidthF(1.6); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    from PySide6.QtCore import QRectF, QPointF
    p.drawRoundedRect(QRectF(size*0.18, size*0.18, size*0.64, size*0.64), 2, 2)
    for i, y in enumerate((0.34, 0.50, 0.66)):
        x2 = 0.72 if i < 2 else 0.58
        p.drawLine(QPointF(size*0.30, size*y), QPointF(size*x2, size*y))
    p.end()
    return pm
