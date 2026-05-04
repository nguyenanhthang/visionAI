"""
Widget hiển thị tiến trình huấn luyện
Bao gồm progress bar cho epoch và tổng thể, thông tin ETA và tốc độ
"""
import time
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt


class ProgressWidget(QWidget):
    """Hiển thị tiến trình epoch và tổng thể khi huấn luyện."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_time: Optional[float] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # --- Tiến trình epoch ---
        grp_epoch = QGroupBox("Epoch hiện tại")
        epoch_layout = QVBoxLayout(grp_epoch)

        self._lbl_epoch = QLabel("Epoch: --/--")
        self._lbl_epoch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_epoch.setStyleSheet("font-weight: bold; color: #a6e3a1;")

        self.epoch_progress = QProgressBar()
        self.epoch_progress.setObjectName("epoch_bar")
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setTextVisible(True)
        self.epoch_progress.setFormat("%v / %m")
        self.epoch_progress.setMinimumHeight(22)

        epoch_layout.addWidget(self._lbl_epoch)
        epoch_layout.addWidget(self.epoch_progress)
        layout.addWidget(grp_epoch)

        # --- Tiến trình tổng thể ---
        grp_total = QGroupBox("Tổng tiến trình")
        total_layout = QVBoxLayout(grp_total)

        self.total_progress = QProgressBar()
        self.total_progress.setObjectName("total_bar")
        self.total_progress.setRange(0, 100)
        self.total_progress.setValue(0)
        self.total_progress.setTextVisible(True)
        self.total_progress.setFormat("%p%")
        self.total_progress.setMinimumHeight(22)

        total_layout.addWidget(self.total_progress)
        layout.addWidget(grp_total)

        # --- Thống kê ---
        grp_stats = QGroupBox("Thống kê")
        stats_layout = QGridLayout(grp_stats)
        stats_layout.setSpacing(6)

        def _make_stat_row(label_text: str, row: int):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #a6adc8;")
            val = QLabel("--")
            val.setStyleSheet("color: #cdd6f4; font-family: 'Courier New';")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            stats_layout.addWidget(lbl, row, 0)
            stats_layout.addWidget(val, row, 1)
            return val

        self._lbl_elapsed = _make_stat_row("⏱ Thời gian đã qua:", 0)
        self._lbl_eta = _make_stat_row("⏳ Dự kiến còn lại:", 1)
        self._lbl_speed = _make_stat_row("⚡ Tốc độ:", 2)
        self._lbl_best_metric = _make_stat_row("🏆 Tốt nhất:", 3)

        layout.addWidget(grp_stats)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_progress(
        self,
        current_epoch: int,
        total_epochs: int,
        elapsed: float = 0.0,
        eta: float = 0.0,
        speed: str = "--",
        best_metric: str = "--",
    ):
        """Cập nhật thanh tiến trình và thông tin thống kê."""
        if total_epochs <= 0:
            return

        # Tiến trình epoch
        self.epoch_progress.setRange(0, total_epochs)
        self.epoch_progress.setValue(current_epoch)
        self._lbl_epoch.setText(f"Epoch: {current_epoch} / {total_epochs}")

        # Tiến trình phần trăm
        pct = int(current_epoch / total_epochs * 100)
        self.total_progress.setValue(pct)

        # Thời gian
        self._lbl_elapsed.setText(_format_time(elapsed))
        self._lbl_eta.setText(_format_time(eta))
        self._lbl_speed.setText(speed)
        self._lbl_best_metric.setText(best_metric)

    def reset(self):
        """Đặt lại về trạng thái ban đầu."""
        self.epoch_progress.setValue(0)
        self.total_progress.setValue(0)
        self._lbl_epoch.setText("Epoch: --/--")
        self._lbl_elapsed.setText("--")
        self._lbl_eta.setText("--")
        self._lbl_speed.setText("--")
        self._lbl_best_metric.setText("--")
        self._start_time = None

    def start_timer(self):
        """Bắt đầu đếm thời gian."""
        self._start_time = time.time()

    def get_elapsed(self) -> float:
        """Lấy thời gian đã trôi qua (giây)."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


def _format_time(seconds: float) -> str:
    """Định dạng thời gian từ giây sang HH:MM:SS."""
    if seconds <= 0:
        return "--"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
