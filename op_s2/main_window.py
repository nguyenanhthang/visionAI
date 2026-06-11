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

import json
import re
import shutil
import threading
import time
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QSize, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

import config
import crashlog
from login_window import StatusDot, make_brand_pixmap
from plc_worker import SimulatedPLCWorker, CP2E_PLCWorker
from scanner import ProductScanner
from settings_window import SettingsDialog, gear_icon


# ── helpers ──────────────────────────────────────────────────────
_ILLEGAL_FN = re.compile(r'[\x00-\x1f<>:"/\\|?*]')   # ký tự cấm trong tên file Windows


def safe_filename(name: str, fallback: str = "UNKNOWN") -> str:
    """Bỏ ký tự điều khiển (\\r, \\n…) + ký tự cấm để tên file không lỗi.

    Phòng khi mã quét lọt ký tự lạ (vd \\r làm Windows báo Errno 22 lúc
    copy ảnh). Rỗng sau khi làm sạch → trả ``fallback``.
    """
    cleaned = _ILLEGAL_FN.sub("", str(name)).strip(" .")
    return cleaned or fallback


# ── đọc số đo từ file nguồn (.xls/.csv) — dùng chung SFC + data export ──
SRC_MEASURE_COLS = range(1, 6)   # cột B..F: Yellow, Orange, Black, Red, OK NG


def _norm(v):
    """xls trả số dạng float — số nguyên thì bỏ đuôi '.0'; None → ''."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _norm_text(v):
    """Chuẩn hoá 1 ô đọc từ text: chuỗi số → int/float, còn lại giữ str."""
    if v is None:
        return ""
    s = str(v).strip()
    if s == "":
        return ""
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def find_source_file(src_dir: str, day: str):
    """Tìm file nguồn theo ngày: ưu tiên .xls, sau đó .csv. None nếu không có."""
    if not src_dir:
        return None
    for ext in (".xls", ".csv"):
        p = Path(src_dir) / f"{day}{ext}"
        if p.exists():
            return p
    return None


_TAIL_BYTES = 262144   # file text chỉ đọc 256KB cuối — file nguồn phình cả ngày


def _last_text_rows(path: Path) -> list:
    """Các dòng CUỐI của file text (tab/comma) — không đọc cả file.

    File nguồn được máy đo append liên tục cả ca; đọc nguyên file cho MỖI
    verdict nghĩa là chi phí tăng tuyến tính theo giờ chạy (giữ GIL khi
    parse → GUI khựng dần). Ta chỉ cần dòng cuối → seek đọc khúc đuôi.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
            data = f.read()
        lines = data.decode("utf-8-sig", errors="ignore").splitlines()
        if size > _TAIL_BYTES and lines:
            lines = lines[1:]   # dòng đầu có thể bị cắt giữa chừng → bỏ
        rows = []
        for line in lines:
            if line.strip() == "":
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            rows.append([_norm_text(c) for c in parts])
        return rows
    except Exception:
        return []


def read_source_rows(path: Path) -> list:
    """Đọc file nguồn thành list dòng (mỗi dòng = list cột).

    Hỗ trợ: (1) .xls (BIFF) thật, (2) đuôi .xls nhưng nội dung text
    tab-separated, (3) .csv comma-separated. Thử BIFF trước; không phải
    thì đọc text tách theo tab HOẶC comma (chỉ đọc khúc đuôi — đủ cho
    dòng mới nhất).
    """
    try:
        import xlrd
        try:
            sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
            return [
                [_norm(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
                for r in range(sheet.nrows)
            ]
        except Exception:
            pass  # không phải .xls BIFF → đọc text bên dưới
    except ImportError:
        pass
    return _last_text_rows(path)


def latest_measures(path: Path, cols=SRC_MEASURE_COLS):
    """Cột `cols` của dòng dữ liệu cuối còn dữ liệu. None nếu file rỗng."""
    for cells in reversed(read_source_rows(path)):
        if any(v != "" for v in cells):
            return [cells[c] if c < len(cells) else "" for c in cols]
    return None


def read_latest_measures(src_dir: str, date_fmt: str = "%Y%m%d",
                         cols=SRC_MEASURE_COLS):
    """Tìm file nguồn theo ngày hôm nay rồi lấy số đo của dòng mới nhất."""
    src = find_source_file(src_dir, datetime.now().strftime(date_fmt or "%Y%m%d"))
    return latest_measures(src, cols) if src else None


def _fmt_mm(v) -> str:
    """Định dạng 1 giá trị đo kèm đơn vị mm (3 số lẻ)."""
    try:
        return f"{float(v):.3f}mm"
    except (TypeError, ValueError):
        s = str(v).strip()
        return f"{s}mm" if s else ""


def format_timer(measures) -> str:
    """Chuỗi thông số cho payload SFC từ [Yellow, Orange, Black, Red, …].

    Trả ``'Black: ..mm; Orange: ..mm; Red: ..mm; Yellow: ..mm'`` (đúng thứ
    tự yêu cầu). measures thiếu/rỗng → ''.
    """
    if not measures or len(measures) < 4:
        return ""
    yellow, orange, black, red = measures[0], measures[1], measures[2], measures[3]
    return (f"Black: {_fmt_mm(black)}; Orange: {_fmt_mm(orange)}; "
            f"Red: {_fmt_mm(red)}; Yellow: {_fmt_mm(yellow)}")


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
    """Tìm subfolder mới nhất + ảnh mới nhất → upload.

    - Subfolder + ảnh đều chọn theo mtime giảm dần để không lệ
      thuộc vào quy ước đặt tên (YYYYMMDD, test1/test2, …).
    - Upload = shutil.copy2 sang upload_dir (UNC share của hệ thống).
    - File đích đổi tên: ``{sn}_{YYYY.MM.DD HH.MM.SS}_{verdict_label}{ext}``
      ví dụ ``P1715102-98-D_SFVN26139D00055_2026.05.21 07.25.36_Passed.png``.
      Đặt vào subfolder ngày hôm nay (``YYYYMMDD``).
    - KHÔNG decode ảnh (bản cũ load QImage vài chục MB mỗi sản phẩm chỉ
      để log tên rồi vứt) — copy file là đủ.
    """

    image_ready = Signal(str)           # "folder/filename" vừa tìm thấy
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
        self.sn = safe_filename(sn, "UNKNOWN")
        self.verdict_label = safe_filename(verdict_label, "Unknown")

    @staticmethod
    def _latest(dir_path: Path, want_dir: bool, exts=()) -> "Path | None":
        """Entry mtime mới nhất trong dir (scandir → stat cache, nhanh hơn
        iterdir+stat khi folder tích cả nghìn ảnh sau nhiều giờ chạy)."""
        import os
        best, best_m = None, -1.0
        try:
            with os.scandir(dir_path) as it:
                for e in it:
                    try:
                        if want_dir:
                            if not e.is_dir():
                                continue
                        else:
                            if not e.is_file():
                                continue
                            if exts and not e.name.lower().endswith(exts):
                                continue
                        m = e.stat().st_mtime
                    except OSError:
                        continue
                    if m > best_m:
                        best, best_m = e.path, m
        except OSError:
            return None
        return Path(best) if best else None

    @Slot()
    def run(self):
        try:
            root = Path(self.root_dir)
            if not root.exists():
                self.error.emit(f"Folder gốc không tồn tại: {root}")
                return

            latest_folder = self._latest(root, want_dir=True)
            if latest_folder is None:
                self.error.emit(f"{root.name} không có subfolder nào")
                return

            latest_img = self._latest(latest_folder, want_dir=False,
                                      exts=self._IMG_EXTS)
            if latest_img is None:
                self.error.emit(f"Folder {latest_folder.name} không có ảnh")
                return
            self.image_ready.emit(f"{latest_folder.name}/{latest_img.name}")

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

    Payload: {sn, stationName, empNo, result}. Nếu có ``src_dir`` thì đọc
    thêm số đo (Yellow/Orange/Black/Red) của dòng mới nhất trong file
    ``<src_dir>/<ngày>.(xls|csv)`` rồi chèn field ``timer`` dạng
    ``"Black: 0.522mm; Orange: 0.526mm; Red: 0.529mm; Yellow: 0.624mm"``.
    Response code=200 → ok, khác → log lỗi (vd 406 "下一制程为 EOL-S").
    """

    done     = Signal(bool, str)   # ok, message
    finished = Signal()

    def __init__(self, url: str, payload: dict, timeout: float = 5.0,
                 src_dir: str = "", date_fmt: str = "%Y%m%d",
                 parent: QObject | None = None):
        super().__init__(parent)
        self.url = url
        self.payload = payload
        self.timeout = timeout
        self.src_dir = src_dir
        self.date_fmt = date_fmt or "%Y%m%d"

    @Slot()
    def run(self):
        sn = self.payload.get("sn", "")
        result = self.payload.get("result", "")
        # Chèn thông số đo đọc từ CSV/xls vào payload (field "timer")
        if self.src_dir:
            try:
                timer = format_timer(read_latest_measures(self.src_dir, self.date_fmt))
                if timer:
                    self.payload["timer"] = timer
            except Exception:
                pass
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

    - Nguồn: ``<src_dir>/<ngày>.xls`` (hoặc ``.csv``) — lấy cột B→F
      (5 giá trị) của DÒNG DỮ LIỆU MỚI NHẤT (dòng cuối còn dữ liệu).
    - Đích:  ``<dst_dir>/<ngày>.xls`` — append dòng
      ``[times, SN, Yellow, Orange, Black, Red, OK NG, result]``;
      header ghi 1 lần ở đầu file.
    - ``times`` = giờ nhận verdict, ``SN`` = mã sản phẩm đang quét,
      ``result`` = "OK" / "NG".

    xlwt không append trực tiếp .xls → đọc lại toàn bộ file đích rồi
    ghi lại (header + dòng cũ + dòng mới) dưới ``_file_lock`` để 2 verdict
    sát nhau không ghi đè lẫn nhau.
    """

    done     = Signal(bool, str)   # ok, message
    finished = Signal()

    HEADER = ["times", "SN", "Yellow", "Orange", "Black", "Red",
              "OK NG", "result"]
    _file_lock = threading.Lock()    # serialize ghi file đích
    WRITE_RETRIES = 5                # số lần thử ghi lại khi file bị khóa
    RETRY_DELAY   = 0.4              # giây giữa các lần thử

    def __init__(self, src_dir: str, dst_dir: str, sn: str = "",
                 result: str = "", date_fmt: str = "%Y%m%d",
                 parent: QObject | None = None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.sn = sn or ""
        self.result = result or ""
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
            src = find_source_file(self.src_dir, day)
            if src is None:
                self.done.emit(False, f"File nguồn {day}.(xls/csv) không tồn tại")
                return

            measures = latest_measures(src)
            if measures is None:
                self.done.emit(False, f"{src.name} không có dòng dữ liệu")
                return

            row = [now.strftime("%Y-%m-%d %H:%M:%S"), self.sn, *measures, self.result]

            dst = Path(self.dst_dir) / f"{day}.xls"
            pending = Path(self.dst_dir) / f"{day}.pending.json"

            ok_save = False
            reason = ""
            with self._file_lock:
                Path(self.dst_dir).mkdir(parents=True, exist_ok=True)
                # các dòng còn đệm từ lần trước (chưa ghi được) + dòng mới
                pending_rows = self._read_pending(pending)
                try:
                    old_rows = self._read_existing(xlrd, dst) if dst.exists() else []
                    self._write_all_retry(xlwt, dst, old_rows + pending_rows + [row])
                    self._clear_pending(pending)   # ghi xong → xoá đệm
                    ok_save = True
                except PermissionError:
                    # file đang mở trong Excel → đệm dòng mới, KHÔNG mất dữ liệu
                    self._append_pending(pending, [row])
                    reason = "đang mở trong Excel"
                except Exception as exc:
                    # lỗi ghi khác → vẫn đệm để thử lại lần sau, không mất dòng
                    self._append_pending(pending, [row])
                    reason = f"lỗi ghi ({exc})"

            if ok_save:
                flushed = len(pending_rows)
                extra = f" (gồm {flushed} dòng đệm trước đó)" if flushed else ""
                self.done.emit(
                    True,
                    f"Lưu data → {dst.name} (+{flushed + 1} dòng, "
                    f"SN {self.sn or '—'}){extra}",
                )
            else:
                n = len(pending_rows) + 1
                self.done.emit(
                    False,
                    f"{day}.xls {reason} — đã đệm {n} dòng, tự lưu khi file rảnh",
                )
        except Exception as exc:
            self.done.emit(False, f"Export .xls lỗi: {exc}")
        finally:
            self.finished.emit()

    # ── helpers ──────────────────────────────────────────────
    @classmethod
    def _read_existing(cls, xlrd, path: Path) -> list[list]:
        """Đọc lại các dòng dữ liệu cũ của file ĐÍCH (bỏ header).

        File đích luôn do chính worker ghi bằng xlwt → là .xls thật.
        Vì vậy KHÔNG dùng fallback text ở đây: nếu file tồn tại mà đọc
        không được thì ném lỗi để run() báo và TUYỆT ĐỐI không ghi đè
        (tránh mất toàn bộ dữ liệu cũ).
        """
        sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
        out = []
        for r in range(sheet.nrows):
            vals = [_norm(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            if r == 0 and vals[:1] == [cls.HEADER[0]]:
                continue  # bỏ header cũ — sẽ ghi lại
            if any(v != "" for v in vals):
                out.append(vals)
        return out

    @classmethod
    def _write_all(cls, xlwt, path: Path, rows: list[list]):
        wb = xlwt.Workbook(encoding="utf-8")
        ws = wb.add_sheet("data")
        for c, h in enumerate(cls.HEADER):
            ws.write(0, c, h)
        for ri, row in enumerate(rows, start=1):
            for c, val in enumerate(row):
                ws.write(ri, c, val)
        # Ghi ra file tạm rồi thay thế: nếu ghi dở (mất điện, lỗi) thì
        # file đích cũ vẫn còn nguyên, không bị cụt.
        tmp = path.with_suffix(path.suffix + ".tmp")
        wb.save(str(tmp))
        import os
        os.replace(str(tmp), str(path))  # atomic trên cùng ổ đĩa

    @classmethod
    def _write_all_retry(cls, xlwt, path: Path, rows: list[list]):
        """Như _write_all nhưng thử lại vài lần khi file bị khóa
        (vd Excel vừa nhả ra). Hết số lần thử mà vẫn khóa → ném
        PermissionError để tầng trên đệm lại."""
        last_err = None
        for _ in range(cls.WRITE_RETRIES):
            try:
                cls._write_all(xlwt, path, rows)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(cls.RETRY_DELAY)
        raise last_err if last_err else PermissionError(str(path))

    # ── file đệm (.pending.json): các dòng chưa ghi được vào .xls ──
    @staticmethod
    def _read_pending(path: Path) -> list[list]:
        """Đọc các dòng đang đệm (mỗi dòng 1 JSON array). Đệm hỏng → []."""
        if not path.exists():
            return []
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            return []   # đệm lỗi thì bỏ qua, không chặn việc ghi
        return rows

    @staticmethod
    def _append_pending(path: Path, rows: list[list]):
        """Nối thêm các dòng vào cuối file đệm (text, Excel không khóa)."""
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def _clear_pending(path: Path):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


# ── main window ──────────────────────────────────────────────────
class MainWindow(QMainWindow):

    # _log gọi từ thread phụ → re-dispatch về GUI thread qua signal này
    _log_queued = Signal(str, str, str, str)

    def __init__(self, employee_id: str, employee_name: str = "", parent=None):
        super().__init__(parent)
        self.employee_id = employee_id
        self.employee_name = employee_name or "—"
        self._started_at = time.time()
        self._current_sn = ""
        self._jobs = []          # giữ ref (thread, worker) các job đang chạy
        self._opl_busy = False   # đang upload ảnh OPL?
        self._opl_busy_since = 0.0   # mốc bắt đầu upload (chẩn đoán kẹt share)
        self._opl_warn_t = 0.0       # throttle cảnh báo upload kẹt
        self._scan_warn_t = 0.0      # throttle cảnh báo "chưa quét hàng"

        self.setWindowTitle("Riser cable — Giao diện chính")
        self.setWindowIcon(QIcon(make_brand_pixmap(64)))
        self.resize(560, 780)
        self.setMinimumSize(460, 600)

        self._build_ui()
        self._log_queued.connect(self._append_log)   # bound method → queued
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

        bl.addWidget(self._build_controls(), 0)
        bl.addWidget(self._build_info_card(), 0)
        bl.addWidget(self._build_log_card(), 1)

        root.addWidget(body, 1)
        root.addWidget(self._build_statusbar())

    # ── controls (toggle bật/tắt) ────────────────────────────
    _TOGGLE_QSS = (
        "QPushButton{background:#101820;color:#7d8590;border:1px solid #2a3540;"
        "border-radius:8px;padding:6px 14px;font-weight:600;}"
        "QPushButton:hover{border-color:#3fb6f0;}"
        "QPushButton:checked{background:#13351f;color:#56d364;border-color:#2ea043;}"
    )

    def _build_controls(self) -> QWidget:
        card = QFrame(); card.setObjectName("Card")
        card.setStyleSheet(
            "QFrame#Card{background:#141b22;border:1px solid #2a3540;border-radius:12px;}"
            "QLabel{background:transparent;}"
        )
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10); lay.setSpacing(10)
        title = QLabel("ĐIỀU KHIỂN")
        title.setStyleSheet("color:#9aa4ae;font-size:11px;font-weight:700;letter-spacing:3px;")
        lay.addWidget(title); lay.addStretch(1)
        self.scan_toggle  = self._make_toggle("Quét SN", "SCAN_ENABLED")
        self.excel_toggle = self._make_toggle("Lưu Excel", "SAVE_EXCEL")
        self.image_toggle = self._make_toggle("Lưu ảnh", "SAVE_IMAGE")
        for b in (self.scan_toggle, self.excel_toggle, self.image_toggle):
            lay.addWidget(b)
        return card

    def _make_toggle(self, label: str, attr: str) -> QPushButton:
        btn = QPushButton()
        btn.setCheckable(True)
        on = bool(getattr(config, attr, True))
        btn.setChecked(on)
        btn.setText(f"{label}: {'ON' if on else 'OFF'}")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setMinimumHeight(32)
        btn.setStyleSheet(self._TOGGLE_QSS)
        btn.toggled.connect(
            lambda c, b=btn, a=attr, l=label: self._on_toggle(b, a, l, c))
        return btn

    def _on_toggle(self, btn: QPushButton, attr: str, label: str, checked: bool):
        setattr(config, attr, checked)
        btn.setText(f"{label}: {'ON' if checked else 'OFF'}")
        self._log(f"{label} {'BẬT' if checked else 'TẮT'}", "SYS",
                  level=("ok" if checked else "warn"))

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
        # giữ tối đa 2000 dòng — tránh document phình vô hạn theo thời gian
        self.log_view.document().setMaximumBlockCount(2000)
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
        crashlog.heartbeat()   # báo watchdog GUI còn sống
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
            self.plc = CP2E_PLCWorker(
                ip=config.PLC_IP,
                result_addr=config.PLC_RESULT_ADDR,
                scan_addr=config.PLC_SCAN_RESULT_ADDR,
                scan_check_addr=getattr(config, "PLC_SCAN_CHECK_ADDR", 500),
                poll_interval=1.0 / max(config.PLC_POLL_HZ, 1),
            )
        self.plc.moveToThread(self._plc_thread)
        self._plc_thread.started.connect(self.plc.run)
        self.plc.connected.connect(self._on_plc_connected)
        self.plc.disconnected.connect(self._on_plc_disconnected)
        self.plc.error.connect(self._on_plc_error)
        self.plc.fatal.connect(self._on_plc_error)
        self.plc.result.connect(self._on_plc_result)
        self.plc.scan_check.connect(self._on_scan_check)
        self.plc.finished.connect(self._plc_thread.quit)
        self._plc_thread.start()
        self._last_result_ts = time.time()

    # NOTE: phải connect signal worker → BOUND METHOD của self (QObject GUI),
    # KHÔNG dùng lambda. Lambda không có QObject context → Qt dùng
    # DirectConnection → slot chạy NGAY trên thread worker → _log đụng QTextEdit
    # từ thread sai → hỏng heap (0xC0000374) → văng app. Bound method →
    # AutoConnection → queued về GUI thread → an toàn.
    def _on_plc_error(self, msg: str):
        self._log(msg, "PLC", level="err")

    def _on_worker_error(self, msg: str):
        self._log(msg, "SYS", level="err")

    def _on_plc_connected(self):
        self._set_chip(self.plc_chip, "PLC online", "#2ea043")
        self.sb_plc.lbl.setText(f"PLC · {config.PLC_IP}:{getattr(config, 'PLC_PORT', 9600)}")
        self.sb_plc.dot.set_color("#2ea043")
        self._log("PLC connected", "PLC", level="ok")

    def _on_plc_disconnected(self):
        self._set_chip(self.plc_chip, "PLC offline", "#7d8590")
        self.sb_plc.dot.set_color("#7d8590")
        self._log("PLC disconnected", "PLC")

    def _on_scan_check(self):
        """PLC bật D500=1 để hỏi 'đã quét SN chưa'.

        Chỉ kiểm tra khi đang BẬT quét SN: nếu SN còn rỗng (operator quên
        quét) → báo lỗi 'chưa quét hàng'. Tắt quét SN thì bỏ qua.
        """
        if not getattr(config, "SCAN_ENABLED", True):
            return
        if not self._current_sn:
            # PLC có thể hỏi lại mỗi vòng poll → throttle để log không ngập
            now = time.time()
            if now - self._scan_warn_t >= 2.0:
                self._scan_warn_t = now
                self._log("CHƯA QUÉT HÀNG — quét SN trước khi qua trạm",
                          "NG", level="err")

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
        self.product_scanner.error.connect(self._on_worker_error)
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
    # ── chạy worker 1-lần, tự dọn sạch (tránh rò thread/handle) ──
    def _run_worker(self, worker, busy_attr: str | None = None):
        """Chạy worker QObject trên 1 QThread dùng-một-lần rồi DỌN SẠCH.

        Quan trọng: gọi thread.deleteLater() khi xong để giải phóng handle +
        cửa sổ nội bộ Win32 của QThread. Nếu không, MỖI sản phẩm rò 1 thread
        → sau ~10 phút cạn USER handle của Windows → app đơ dù CPU/RAM thấp.
        """
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)   # xoá worker (mẫu chuẩn Qt)
        job = (thread, worker)
        self._jobs.append(job)
        if len(self._jobs) > 32:
            # job xong là tự rút khỏi _jobs — dồn nhiều = worker đang kẹt
            # (share mạng/Excel khóa file…) → báo sớm trước khi cạn tài nguyên
            self._log(f"Cảnh báo: {len(self._jobs)} job nền chưa xong — "
                      "kiểm tra mạng/share ảnh/file Excel", "SYS", level="warn")

        def _cleanup():
            thread.deleteLater()
            try:
                self._jobs.remove(job)   # buông ref → worker được GC
            except ValueError:
                pass
            if busy_attr:
                setattr(self, busy_attr, False)

        thread.finished.connect(_cleanup)
        thread.start()

    def _trigger_opl_upload(self, sn: str, verdict_label: str, root_dir: str):
        if not getattr(config, "SAVE_IMAGE", True):
            self._log("Save image tắt — bỏ qua đẩy ảnh", "SYS")
            return
        if self._opl_busy:
            # đang upload ảnh, bỏ qua trigger trùng. Kẹt quá lâu (share
            # mạng chết → copy2 treo nhiều phút) thì báo để biết đường xử lý.
            now = time.time()
            stuck = now - self._opl_busy_since
            if stuck > 300 and now - self._opl_warn_t >= 60:
                self._opl_warn_t = now
                self._log(f"Upload ảnh kẹt {stuck/60:.0f} phút — kiểm tra "
                          f"share {config.link_post_img}", "SYS", level="warn")
            return
        self._log(f"Tìm ảnh {verdict_label} mới nhất cho {sn} trong {root_dir}…", "SYS")

        worker = OplImageWorker(
            root_dir=root_dir,
            upload_dir=config.link_post_img,
            sn=sn,
            verdict_label=verdict_label,
        )
        worker.image_ready.connect(self._on_opl_image_ready)
        worker.upload_done.connect(self._on_opl_upload_done)
        worker.error.connect(self._on_worker_error)
        self._opl_busy = True
        self._opl_busy_since = time.time()
        self._run_worker(worker, busy_attr="_opl_busy")

    def _on_opl_image_ready(self, name: str):
        self._log(f"Đã tìm thấy ảnh: {name}", "SYS", level="ok")

    def _on_opl_upload_done(self, ok: bool, msg: str):
        level = "ok" if ok else "err"
        self._log(f"Upload: {msg}", "SYS", level=level)

    # ── Data export .xls (auto trigger sau mỗi PLC verdict) ───
    def _trigger_data_export(self, sn: str, result: str = ""):
        if not getattr(config, "SAVE_EXCEL", True):
            self._log("Save Excel tắt — bỏ qua ghi .xls", "SYS")
            return
        src = getattr(config, "DATA_SRC_DIR", "")
        dst = getattr(config, "DATA_EXPORT_DIR", "")
        if not src or not dst:
            return  # chưa cấu hình folder → bỏ qua
        worker = DataExportWorker(
            src_dir=src,
            dst_dir=dst,
            sn=sn or self._current_sn,
            result=result,
            date_fmt=getattr(config, "DATA_FILE_DATEFMT", "%Y%m%d"),
        )
        worker.done.connect(self._on_data_export_done)
        self._run_worker(worker)

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
            verdict = "OK" if result == "PASS" else "NG"
            self._push_sfc_result(pid, result)
            self._trigger_opl_upload(pid, verdict_label, root_dir)
            self._trigger_data_export(pid, verdict)
        # xử lý xong tín hiệu D300 → xoá SN, chờ lần quét kế tiếp
        self._current_sn = ""
        self._product_lbl.setText("—")

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
        worker = SfcPushWorker(
            config.link_sfc, payload,
            timeout=config.API_REQUEST_TIMEOUT,
            src_dir=getattr(config, "DATA_SRC_DIR", ""),
            date_fmt=getattr(config, "DATA_FILE_DATEFMT", "%Y%m%d"),
        )
        worker.done.connect(self._on_sfc_done)
        self._run_worker(worker)

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
        """Ghi 1 dòng log — AN TOÀN THREAD.

        crash.log thực tế (build cũ) cho thấy app văng 0xC0000374 (hỏng
        heap) vì signal worker nối qua lambda → slot chạy NGAY trên thread
        worker → _log đụng QTextEdit từ thread sai. Mọi connect giờ đã là
        bound method, nhưng để lỗi kiểu đó KHÔNG BAO GIỜ sập app nữa:
        gọi _log từ thread khác GUI → tự re-dispatch qua signal (queued).
        """
        if QThread.currentThread() is not self.thread():
            self._log_queued.emit(str(message), str(tag), str(level), str(detail))
            return
        self._append_log(str(message), str(tag), str(level), str(detail))

    def _append_log(self, message: str, tag: str, level: str, detail: str):
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
        # escape: message/detail (vd lỗi API, repr exception) có thể chứa
        # < > & làm vỡ HTML → khi đó log không hiển thị/không cuộn được.
        det = (f" <span style='color:#7d8590'>· {escape(str(detail))}</span>"
               if detail else "")
        html = (
            f"<span style='color:#5b6772;'>[{ts}]</span>&nbsp;"
            f"<span style='color:{tcol};font-weight:700;'>{escape(str(tag))}</span>&nbsp;&nbsp;"
            f"<span style='color:{mcol}'>{escape(str(message))}</span>{det}"
        )
        self.log_view.append(html)
        # luôn cuộn xuống dòng mới nhất (kể cả khi log lỗi dồn dập)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

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
        # dọn các job 1-lần còn chạy (SFC/OPL/export)
        for thread, _worker in list(self._jobs):
            try:
                thread.quit()
                thread.wait(1500)
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