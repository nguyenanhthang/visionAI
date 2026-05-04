"""
Widget hiển thị biểu đồ metrics trong quá trình huấn luyện.
Sử dụng matplotlib embedded trong PySide6.
"""
from collections import defaultdict
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QSizePolicy, QFileDialog
)
from PySide6.QtCore import Qt

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Nhóm metrics cần vẽ theo từng loại chart
LOSS_KEYS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "train/obj_loss", "train/seg_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "val/obj_loss", "val/seg_loss",
]

PERFORMANCE_KEYS = [
    "metrics/mAP50", "metrics/mAP50-95",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "metrics/precision", "metrics/recall",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/accuracy_top1", "metrics/accuracy_top5",
]

# Màu cho từng metric (đẹp trên nền tối)
COLORS = [
    "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
    "#cba6f7", "#94e2d5", "#fab387", "#74c7ec",
    "#b4befe", "#f2cdcd",
]


class MetricsViewer(QWidget):
    """Widget hiển thị biểu đồ loss và metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Lưu trữ dữ liệu ──
        # Mỗi key → list of (epoch, value)  — luôn đảm bảo x, y cùng length
        self._data: Dict[str, List[tuple]] = defaultdict(list)
        self._seen_epochs: Dict[str, set] = defaultdict(set)  # tránh duplicate

        self._setup_ui()

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        if not HAS_MATPLOTLIB:
            layout.addWidget(QLabel(
                "⚠️ matplotlib chưa cài đặt.\n"
                "Chạy: pip install matplotlib"
            ))
            return

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._combo_view = QComboBox()
        self._combo_view.addItems(["Loss", "Performance", "Tất cả"])
        self._combo_view.setCurrentText("Tất cả")
        self._combo_view.currentTextChanged.connect(lambda _: self._redraw())
        toolbar.addWidget(QLabel("Hiển thị:"))
        toolbar.addWidget(self._combo_view)

        toolbar.addStretch()

        btn_export = QPushButton("💾 Export")
        btn_export.setMaximumWidth(80)
        btn_export.clicked.connect(self._export_chart)
        toolbar.addWidget(btn_export)

        btn_clear = QPushButton("🗑️ Xoá")
        btn_clear.setMaximumWidth(60)
        btn_clear.clicked.connect(self.clear)
        toolbar.addWidget(btn_clear)

        layout.addLayout(toolbar)

        # Canvas matplotlib
        self._fig = Figure(figsize=(8, 4), dpi=100, facecolor="#1e1e2e")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._canvas)

    # ==================================================================
    # Public API
    # ==================================================================

    def update_metrics(self, metrics_dict: dict):
        """
        Nhận dict metrics từ worker và lưu vào bộ nhớ.
        Mỗi lần gọi tương ứng 1 "event" (cuối epoch train hoặc val).
        """
        epoch = metrics_dict.get("epoch", None)
        if epoch is None:
            return

        epoch = int(epoch)

        for key, value in metrics_dict.items():
            if key == "epoch":
                continue

            # Chỉ lưu các key thuộc loss hoặc performance
            if not any(key.startswith(p) for p in
                       ("train/", "val/", "metrics/", "lr/")):
                continue

            try:
                val = float(value)
            except (TypeError, ValueError):
                continue

            # ── Chống duplicate: mỗi (key, epoch) chỉ lưu 1 lần ──
            if epoch in self._seen_epochs[key]:
                # Cập nhật giá trị mới nhất cho epoch đó (thay vì bỏ qua)
                for i, (e, _) in enumerate(self._data[key]):
                    if e == epoch:
                        self._data[key][i] = (epoch, val)
                        break
            else:
                self._seen_epochs[key].add(epoch)
                self._data[key].append((epoch, val))

        # Vẽ lại biểu đồ
        try:
            self._redraw()
        except Exception:
            pass  # Không crash UI nếu vẽ lỗi

    def clear(self):
        """Xoá toàn bộ dữ liệu và biểu đồ."""
        self._data.clear()
        self._seen_epochs.clear()
        if HAS_MATPLOTLIB:
            self._fig.clear()
            self._canvas.draw_idle()

    # ==================================================================
    # Drawing
    # ==================================================================

    def _redraw(self):
        """Vẽ lại toàn bộ biểu đồ."""
        if not HAS_MATPLOTLIB:
            return

        self._fig.clear()

        view = self._combo_view.currentText()

        # Lọc keys theo view
        if view == "Loss":
            target_keys = [k for k in self._data if
                           any(k.startswith(p) for p in ("train/", "val/"))
                           and "loss" in k.lower()]
        elif view == "Performance":
            target_keys = [k for k in self._data if
                           k.startswith("metrics/")]
        else:  # Tất cả
            target_keys = list(self._data.keys())

        if not target_keys:
            ax = self._fig.add_subplot(111, facecolor="#1e1e2e")
            ax.text(0.5, 0.5, "Chưa có dữ liệu",
                    ha="center", va="center", color="#6c7086", fontsize=14)
            ax.set_facecolor("#1e1e2e")
            self._style_ax(ax)
            self._canvas.draw_idle()
            return

        # Chia thành 2 subplot nếu cả loss + perf đều có
        loss_keys = [k for k in target_keys if "loss" in k.lower()]
        perf_keys = [k for k in target_keys if k.startswith("metrics/")]
        other_keys = [k for k in target_keys
                      if k not in loss_keys and k not in perf_keys]

        num_plots = sum([
            1 if (loss_keys or other_keys) else 0,
            1 if perf_keys else 0,
        ])
        if num_plots == 0:
            num_plots = 1

        plot_idx = 1

        # ── Subplot 1: Loss ──
        if loss_keys or other_keys:
            ax = self._fig.add_subplot(1, num_plots, plot_idx,
                                       facecolor="#1e1e2e")
            ax.set_title("Loss", color="#cdd6f4", fontsize=11, pad=8)
            color_idx = 0
            for key in sorted(loss_keys + other_keys):
                pairs = self._data[key]
                if not pairs:
                    continue
                # Sắp xếp theo epoch
                pairs_sorted = sorted(pairs, key=lambda p: p[0])
                x = [p[0] for p in pairs_sorted]
                y = [p[1] for p in pairs_sorted]
                color = COLORS[color_idx % len(COLORS)]
                label = key.replace("train/", "T/").replace("val/", "V/")
                ax.plot(x, y, color=color, linewidth=1.5,
                        marker=".", markersize=3, label=label)
                color_idx += 1
            ax.legend(fontsize=7, facecolor="#313244", edgecolor="#45475a",
                      labelcolor="#cdd6f4", loc="upper right")
            self._style_ax(ax)
            plot_idx += 1

        # ── Subplot 2: Performance ──
        if perf_keys:
            ax = self._fig.add_subplot(1, num_plots, plot_idx,
                                       facecolor="#1e1e2e")
            ax.set_title("Performance", color="#cdd6f4", fontsize=11, pad=8)
            color_idx = 0
            for key in sorted(perf_keys):
                pairs = self._data[key]
                if not pairs:
                    continue
                pairs_sorted = sorted(pairs, key=lambda p: p[0])
                x = [p[0] for p in pairs_sorted]
                y = [p[1] for p in pairs_sorted]
                color = COLORS[color_idx % len(COLORS)]
                label = key.replace("metrics/", "")
                ax.plot(x, y, color=color, linewidth=1.5,
                        marker=".", markersize=3, label=label)
                color_idx += 1
            ax.legend(fontsize=7, facecolor="#313244", edgecolor="#45475a",
                      labelcolor="#cdd6f4", loc="lower right")
            ax.set_ylim(-0.05, 1.05)
            self._style_ax(ax)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

    @staticmethod
    def _style_ax(ax):
        """Áp dụng dark-theme styling cho axis."""
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="#6c7086", labelsize=8)
        ax.xaxis.label.set_color("#6c7086")
        ax.yaxis.label.set_color("#6c7086")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, alpha=0.15, color="#585b70")
        ax.set_xlabel("Epoch", fontsize=9)

    # ==================================================================
    # Export
    # ==================================================================

    def _export_chart(self):
        """Lưu biểu đồ ra file ảnh."""
        if not HAS_MATPLOTLIB:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu biểu đồ", "metrics.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if path:
            self._fig.savefig(path, dpi=150, facecolor="#1e1e2e",
                              bbox_inches="tight")