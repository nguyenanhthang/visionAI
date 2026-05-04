"""
Widget hiển thị thông tin dataset YAML
"""
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt


class DatasetBrowser(QWidget):
    """Hiển thị thông tin dataset từ file YAML."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grp = QGroupBox("Thông tin Dataset")
        grp_layout = QVBoxLayout(grp)

        self._lbl_path = QLabel("Chưa chọn dataset")
        self._lbl_path.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self._lbl_path.setWordWrap(True)

        self._lbl_summary = QLabel("")
        self._lbl_summary.setStyleSheet("color: #a6e3a1; font-weight: bold;")

        self._txt_info = QTextEdit()
        self._txt_info.setReadOnly(True)
        self._txt_info.setMaximumHeight(120)
        self._txt_info.setStyleSheet(
            "background-color: #11111b; color: #cdd6f4; "
            "border: 1px solid #45475a; font-family: 'Courier New'; font-size: 11px;"
        )

        grp_layout.addWidget(self._lbl_path)
        grp_layout.addWidget(self._lbl_summary)
        grp_layout.addWidget(self._txt_info)
        layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, yaml_path: str):
        """Đọc và hiển thị thông tin từ file YAML của dataset."""
        try:
            import yaml
            path = Path(yaml_path)
            if not path.exists():
                self._show_error(f"Không tìm thấy file: {yaml_path}")
                return

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self._lbl_path.setText(f"📄 {path.name}")
            self._display_info(data, path)

        except ImportError:
            self._show_error("Thiếu thư viện PyYAML")
        except Exception as e:
            self._show_error(f"Lỗi đọc dataset: {e}")

    def clear(self):
        """Xóa thông tin hiển thị."""
        self._lbl_path.setText("Chưa chọn dataset")
        self._lbl_summary.setText("")
        self._txt_info.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _display_info(self, data: dict, path: Path):
        """Hiển thị thông tin dataset."""
        nc = data.get("nc", "?")
        names = data.get("names", {})
        task = data.get("task", "detect")
        dataset_path = data.get("path", "?")
        train = data.get("train", "?")
        val = data.get("val", "?")

        # Đếm số ảnh nếu có thể
        n_train = self._count_images(path.parent / dataset_path if dataset_path != "?" else path.parent, train)
        n_val = self._count_images(path.parent / dataset_path if dataset_path != "?" else path.parent, val)

        summary = f"Task: {task} | {nc} classes"
        if n_train:
            summary += f" | Train: {n_train} ảnh"
        if n_val:
            summary += f" | Val: {n_val} ảnh"
        self._lbl_summary.setText(summary)

        # Chi tiết
        lines = [f"Task: {task}", f"Path: {dataset_path}",
                 f"Train: {train}", f"Val: {val}", f"Số classes: {nc}", ""]
        if isinstance(names, dict):
            lines.append("Classes:")
            for idx, name in list(names.items())[:20]:
                lines.append(f"  {idx}: {name}")
            if len(names) > 20:
                lines.append(f"  ... và {len(names) - 20} classes nữa")
        elif isinstance(names, list):
            lines.append("Classes:")
            for i, name in enumerate(names[:20]):
                lines.append(f"  {i}: {name}")

        self._txt_info.setText("\n".join(str(l) for l in lines))

    def _count_images(self, base_path: Path, sub: str) -> int:
        """Đếm số file ảnh trong thư mục."""
        try:
            from app.utils.file_utils import get_image_files
            full = base_path / sub
            if full.exists():
                return len(get_image_files(str(full)))
        except Exception:
            pass
        return 0

    def _show_error(self, msg: str):
        self._lbl_path.setText(f"❌ {msg}")
        self._lbl_summary.setText("")
        self._txt_info.clear()
