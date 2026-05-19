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
                                QFileDialog, QFrame, QWidget, QSizePolicy)
from PySide6.QtCore import Qt, QSettings, QSize
from PySide6.QtGui import QFont


class StartupAOIPicker(QDialog):
    """Splash-style picker: chọn file AOI để load hoặc New blank project.

    Public API sau khi exec():
      - result()        → QDialog.Accepted nếu user chọn file/New, Rejected nếu đóng
      - chosen_path()   → str path nếu user pick file/recent; "" nếu New blank
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VisionPro — Load AOI")
        self.setMinimumSize(640, 440)
        self.setStyleSheet("""
            QDialog{background:#0a0e1a;color:#e2e8f0;}
            QLabel{color:#e2e8f0;}
            QListWidget{background:#0d1220;border:1px solid #1e2d45;
                        color:#e2e8f0;font-size:12px;padding:4px;
                        outline:none;}
            QListWidget::item{padding:8px 10px;border-bottom:1px solid #131a2a;}
            QListWidget::item:selected{background:#1a2236;color:#00d4ff;}
            QListWidget::item:hover{background:#131a2a;}
            QPushButton{background:#1e2d45;color:#e2e8f0;border:none;
                        border-radius:5px;padding:9px 18px;font-weight:600;
                        font-size:12px;}
            QPushButton:hover{background:#00d4ff;color:#000;}
            QPushButton#primary{background:#0f3460;color:#00d4ff;}
            QPushButton#primary:hover{background:#00d4ff;color:#000;}
        """)

        self._chosen: str = ""   # "" = New blank; non-empty = file path
        self._build_ui()

    # ── Public ────────────────────────────────────────────────────────
    def chosen_path(self) -> str:
        return self._chosen

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)

        # Header
        title = QLabel("👁  VisionPro AOI")
        title.setStyleSheet(
            "color:#00d4ff;font-size:22px;font-weight:700;letter-spacing:2px;")
        root.addWidget(title)

        sub = QLabel("Chọn file AOI để mở, hoặc bắt đầu project mới.")
        sub.setStyleSheet("color:#94a3b8;font-size:11px;")
        root.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        root.addWidget(sep)

        # Recent files list
        hdr = QLabel("📂  Recent files")
        hdr.setStyleSheet(
            "color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:1px;"
            "margin-top:4px;")
        root.addWidget(hdr)

        self._list = QListWidget()
        self._list.setIconSize(QSize(0, 0))
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._populate_recents()
        root.addWidget(self._list, 1)

        # Action buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)

        btn_browse = QPushButton("📁  Browse…")
        btn_browse.setToolTip("Mở file .aoi/.json bất kỳ qua file dialog.")
        btn_browse.clicked.connect(self._on_browse)
        btn_row.addWidget(btn_browse)

        btn_new = QPushButton("✨  New blank project")
        btn_new.setToolTip("Tạo pipeline trống, không load file.")
        btn_new.clicked.connect(self._on_new)
        btn_row.addWidget(btn_new)

        btn_row.addStretch(1)

        btn_open = QPushButton("Open Selected")
        btn_open.setObjectName("primary")
        btn_open.setToolTip("Mở file đang chọn trong recent list. "
                             "Double-click vào file cũng được.")
        btn_open.clicked.connect(self._on_open_selected)
        btn_row.addWidget(btn_open)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        root.addLayout(btn_row)

    def _populate_recents(self):
        """Fill list từ QSettings['recent_files']. Bỏ file đã bị xóa khỏi disk."""
        s = QSettings()
        raw = s.value("recent_files", []) or []
        if isinstance(raw, str):
            raw = [raw]
        recents = [p for p in raw if isinstance(p, str) and os.path.isfile(p)]
        # Re-save sau filter để clean stale entries
        if len(recents) != len(raw):
            s.setValue("recent_files", recents)

        self._list.clear()
        if not recents:
            placeholder = QListWidgetItem(
                "  (Chưa có file nào — bấm Browse để mở file lần đầu)")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsSelectable
                                 & ~Qt.ItemIsEnabled)
            placeholder.setForeground(Qt.gray)
            self._list.addItem(placeholder)
            return
        for path in recents:
            self._list.addItem(self._make_item(path))
        self._list.setCurrentRow(0)

    def _make_item(self, path: str) -> QListWidgetItem:
        name = os.path.basename(path)
        try:
            mt = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(os.path.getmtime(path)))
            size = os.path.getsize(path)
            sz_str = self._fmt_size(size)
            meta = f"{mt}  ·  {sz_str}"
        except OSError:
            meta = "(metadata unavailable)"
        item = QListWidgetItem()
        item.setText(f"  📄  {name}\n       {path}\n       {meta}")
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        return item

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

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
