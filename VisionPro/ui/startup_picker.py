"""
ui/startup_picker.py — Startup dialog hiển thị khi mở app, cho user pick:
  • 1 file AOI gần đây (list QSettings["recent_files"])
  • Browse mở file bất kỳ
  • New blank project

Sau khi user chọn, MainWindow được tạo + load đúng file user pick (nếu có).
"""
from __future__ import annotations
import os, time
from typing import Optional

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QListWidget, QListWidgetItem, QPushButton,
                                QFileDialog, QFrame, QWidget, QSizePolicy,
                                QStyle)
from PySide6.QtCore import Qt, QSettings, QSize, QDateTime
from PySide6.QtGui import QFont, QFontMetrics


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _humanize_mtime(mtime: float) -> str:
    """Thời gian tương đối: 'Just now', '5 min ago', '2h ago', 'Yesterday',
    '3 days ago', 'YYYY-MM-DD' cho > 1 tuần."""
    now = time.time()
    diff = now - mtime
    if diff < 60:
        return "Just now"
    if diff < 3600:
        m = int(diff // 60)
        return f"{m} min ago"
    if diff < 86400:
        h = int(diff // 3600)
        return f"{h}h ago"
    if diff < 86400 * 2:
        return "Yesterday"
    if diff < 86400 * 7:
        return f"{int(diff // 86400)} days ago"
    return time.strftime("%Y-%m-%d", time.localtime(mtime))


class RecentFileCard(QWidget):
    """Custom card cho 1 recent file. Bao gồm:
      • Icon block màu cyan (📄) bên trái
      • Filename bold + path elided ở giữa
      • Meta chips (mtime humanized + size) bên phải

    Hỗ trợ trạng thái selected (border + bg sáng hơn) qua set_selected()."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path
        self._selected = False
        self.setAttribute(Qt.WA_StyledBackground, True)

        # Layout: [Icon] [Name + Path stacked] [stretch] [Meta chips stacked]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(14)

        # Icon block — fixed size, accent color
        self._icon = QLabel("📄")
        self._icon.setFixedSize(46, 46)
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet(
            "QLabel{background:#0f3460;color:#00d4ff;border-radius:8px;"
            "font-size:22px;border:1px solid #1a2236;}")
        lay.addWidget(self._icon)

        # Center column: filename + path
        center = QVBoxLayout(); center.setSpacing(3)
        self._name_lbl = QLabel(os.path.basename(path))
        self._name_lbl.setStyleSheet(
            "color:#e2e8f0;font-size:13px;font-weight:700;")
        center.addWidget(self._name_lbl)

        # Path elided khi quá dài (dùng QFontMetrics khi resize)
        self._path_lbl = QLabel()
        self._path_lbl.setStyleSheet("color:#64748b;font-size:10px;")
        self._path_lbl.setToolTip(path)
        center.addWidget(self._path_lbl)
        lay.addLayout(center, 1)

        # Meta chips: mtime humanized + size
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
            time_str = _humanize_mtime(mtime)
            size_str = _fmt_size(size)
            time_tt = time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(mtime))
        except OSError:
            time_str, size_str, time_tt = "—", "—", ""

        chips = QHBoxLayout(); chips.setSpacing(6)
        chip_time = self._make_chip(f"🕐 {time_str}", tip=time_tt)
        chip_size = self._make_chip(f"📦 {size_str}")
        chips.addWidget(chip_time); chips.addWidget(chip_size)
        lay.addLayout(chips)

        self._apply_style()

    @staticmethod
    def _make_chip(text: str, tip: str = "") -> QLabel:
        c = QLabel(text)
        c.setStyleSheet(
            "QLabel{background:#131a2a;color:#94a3b8;font-size:10px;"
            "padding:4px 8px;border-radius:9px;font-weight:600;}")
        if tip:
            c.setToolTip(tip)
        return c

    def set_selected(self, selected: bool):
        if selected == self._selected:
            return
        self._selected = selected
        self._apply_style()

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet(
                "RecentFileCard{background:#162033;border:1px solid #00d4ff;"
                "border-radius:10px;}"
                "RecentFileCard:hover{background:#1a2640;}")
            self._name_lbl.setStyleSheet(
                "color:#00d4ff;font-size:13px;font-weight:700;")
        else:
            self.setStyleSheet(
                "RecentFileCard{background:#0d1220;border:1px solid #1e2d45;"
                "border-radius:10px;}"
                "RecentFileCard:hover{background:#131a2a;"
                "border:1px solid #2a3d5e;}")
            self._name_lbl.setStyleSheet(
                "color:#e2e8f0;font-size:13px;font-weight:700;")

    def resizeEvent(self, ev):
        # Path label elided theo width thực — tránh wrap xuống nhiều dòng làm
        # card cao bất thường.
        fm = QFontMetrics(self._path_lbl.font())
        avail = max(50, self._path_lbl.width())
        self._path_lbl.setText(
            fm.elidedText(self._path, Qt.ElideMiddle, avail))
        super().resizeEvent(ev)


class StartupAOIPicker(QDialog):
    """Splash-style picker: chọn file AOI để load hoặc New blank project.

    Public API sau khi exec():
      - result()        → QDialog.Accepted nếu user chọn file/New, Rejected nếu đóng
      - chosen_path()   → str path nếu user pick file/recent; "" nếu New blank
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VisionPro — Load AOI")
        self.setMinimumSize(720, 520)
        self.setStyleSheet("""
            QDialog{background:#0a0e1a;color:#e2e8f0;}
            QLabel{color:#e2e8f0;}
            QListWidget{background:transparent;border:none;
                        color:#e2e8f0;padding:0;outline:none;}
            QListWidget::item{padding:0;margin:0 0 6px 0;
                              border:none;background:transparent;}
            /* Card được render bởi RecentFileCard widget, không cần
               selection bg ở QListWidget level — card tự highlight. */
            QListWidget::item:selected{background:transparent;}
            QScrollBar:vertical{background:#0a0e1a;width:8px;border:none;}
            QScrollBar::handle:vertical{background:#1e2d45;border-radius:4px;
                                         min-height:30px;}
            QScrollBar::handle:vertical:hover{background:#2a3d5e;}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{
                background:none;border:none;height:0;}
            QPushButton{background:#1e2d45;color:#e2e8f0;border:none;
                        border-radius:6px;padding:10px 20px;font-weight:600;
                        font-size:12px;}
            QPushButton:hover{background:#00d4ff;color:#000;}
            QPushButton#primary{background:#0f3460;color:#00d4ff;}
            QPushButton#primary:hover{background:#00d4ff;color:#000;}
            QPushButton#ghost{background:transparent;color:#94a3b8;
                              border:1px solid #1e2d45;}
            QPushButton#ghost:hover{background:#131a2a;color:#00d4ff;
                                     border:1px solid #00d4ff;}
        """)

        self._chosen: str = ""   # "" = New blank; non-empty = file path
        self._build_ui()

    # ── Public ────────────────────────────────────────────────────────
    def chosen_path(self) -> str:
        return self._chosen

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 22)
        root.setSpacing(16)

        # Hero header: app brand + tagline
        hero = QVBoxLayout(); hero.setSpacing(2)
        title = QLabel("👁  VisionPro AOI")
        title.setStyleSheet(
            "color:#00d4ff;font-size:26px;font-weight:800;letter-spacing:2px;")
        hero.addWidget(title)
        sub = QLabel("Chọn file AOI gần đây để tiếp tục, hoặc bắt đầu mới.")
        sub.setStyleSheet("color:#64748b;font-size:12px;")
        hero.addWidget(sub)
        root.addLayout(hero)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;background:#1e2d45;max-height:1px;")
        root.addWidget(sep)

        # Section header — "Recent files" + count
        sec_row = QHBoxLayout()
        hdr = QLabel("📂  RECENT FILES")
        hdr.setStyleSheet(
            "color:#94a3b8;font-size:10px;font-weight:800;letter-spacing:2px;")
        sec_row.addWidget(hdr)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(
            "color:#475569;font-size:10px;font-weight:700;letter-spacing:1px;")
        sec_row.addWidget(self._count_lbl)
        sec_row.addStretch(1)
        root.addLayout(sec_row)

        # Card list — QListWidget với custom setItemWidget cho mỗi row
        self._list = QListWidget()
        self._list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._list.setSpacing(0)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        self._populate_recents()
        root.addWidget(self._list, 1)

        # Action buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)

        btn_browse = QPushButton("📁  Browse…")
        btn_browse.setObjectName("ghost")
        btn_browse.setToolTip("Mở file .aoi/.json bất kỳ qua file dialog.")
        btn_browse.clicked.connect(self._on_browse)
        btn_row.addWidget(btn_browse)

        btn_new = QPushButton("✨  New blank project")
        btn_new.setObjectName("ghost")
        btn_new.setToolTip("Tạo pipeline trống, không load file.")
        btn_new.clicked.connect(self._on_new)
        btn_row.addWidget(btn_new)

        btn_row.addStretch(1)

        self._btn_open = QPushButton("Open Selected →")
        self._btn_open.setObjectName("primary")
        self._btn_open.setToolTip("Mở file đang chọn trong recent list. "
                                   "Double-click vào file cũng được.")
        self._btn_open.clicked.connect(self._on_open_selected)
        self._btn_open.setEnabled(False)   # disable nếu chưa có selection
        btn_row.addWidget(self._btn_open)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        root.addLayout(btn_row)

        # Sync btn state với selection hiện tại (populate đã setCurrentRow(0)
        # trước khi btn được tạo nên cần sync sau).
        self._on_selection_changed(self._list.currentItem(), None)

    def _populate_recents(self):
        """Fill list từ QSettings['recent_files']. Bỏ file đã bị xóa khỏi
        disk. Render mỗi row là card widget (RecentFileCard) qua setItemWidget."""
        s = QSettings()
        raw = s.value("recent_files", []) or []
        if isinstance(raw, str):
            raw = [raw]
        recents = [p for p in raw if isinstance(p, str) and os.path.isfile(p)]
        if len(recents) != len(raw):
            s.setValue("recent_files", recents)

        self._list.clear()
        self._count_lbl.setText(
            f"  ·  {len(recents)} file{'s' if len(recents) != 1 else ''}")

        if not recents:
            # Empty state — card-style placeholder, không selectable
            placeholder = QWidget()
            pl_lay = QVBoxLayout(placeholder)
            pl_lay.setContentsMargins(24, 36, 24, 36); pl_lay.setSpacing(8)
            icon = QLabel("📭")
            icon.setStyleSheet("font-size:36px;")
            icon.setAlignment(Qt.AlignCenter)
            pl_lay.addWidget(icon)
            msg = QLabel("Chưa có file AOI nào gần đây")
            msg.setStyleSheet("color:#64748b;font-size:13px;font-weight:600;")
            msg.setAlignment(Qt.AlignCenter)
            pl_lay.addWidget(msg)
            hint = QLabel("Bấm <b>Browse…</b> để mở file đầu tiên, "
                            "hoặc <b>New blank project</b>.")
            hint.setStyleSheet("color:#475569;font-size:11px;")
            hint.setAlignment(Qt.AlignCenter)
            pl_lay.addWidget(hint)
            placeholder.setStyleSheet(
                "QWidget{background:#0d1220;border:1px dashed #1e2d45;"
                "border-radius:10px;}")
            item = QListWidgetItem()
            item.setSizeHint(placeholder.sizeHint())
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable
                          & ~Qt.ItemIsEnabled)
            self._list.addItem(item)
            self._list.setItemWidget(item, placeholder)
            return

        for path in recents:
            card = RecentFileCard(path)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(QSize(0, card.sizeHint().height()))
            self._list.addItem(item)
            self._list.setItemWidget(item, card)
        self._list.setCurrentRow(0)

    def _on_selection_changed(self, current, previous):
        # Update card visual + enable/disable Open button. Guard btn_open vì
        # _populate_recents có thể trigger trước khi nút được tạo (build_ui
        # call _populate_recents giữa chừng).
        for i in range(self._list.count()):
            it = self._list.item(i)
            w = self._list.itemWidget(it)
            if isinstance(w, RecentFileCard):
                w.set_selected(it is current)
        btn = getattr(self, "_btn_open", None)
        if btn is not None:
            btn.setEnabled(
                current is not None and current.data(Qt.UserRole) is not None)

    # ── Slots ─────────────────────────────────────────────────────────
    def _on_double_click(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if path:
            self._chosen = path
            self.accept()

    def _on_open_selected(self):
        item = self._list.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if path:
            self._chosen = path
            self.accept()

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AOI File", "",
            "AOI Pipeline (*.aoi *.json);;All Files (*)")
        if path:
            self._chosen = path
            self.accept()

    def _on_new(self):
        self._chosen = ""   # signal "New blank" to caller
        self.accept()
