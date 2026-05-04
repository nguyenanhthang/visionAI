from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from canvas import DrawingCanvas
from dialogs import ShapeEditDialog


class LabelDrawingUI(QMainWindow):
    """Main window: canvas + controls + multi-image cache management."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Polygon Label Tool")
        self.resize(1280, 800)

        # Multi-image cache: path → list of shape dicts
        self.image_cache: Dict[str, List[dict]] = {}
        self._current_image: str = ""

        # Class map: class_id → class_name
        self._class_map: Dict[int, str] = {}

        # Drawing defaults
        self._draw_color: Tuple[int, int, int] = (255, 0, 0)
        self._draw_width: int = 2
        self._draw_class_id: int = 0
        self._draw_class_name: str = "object"

        self._build_ui()
        self._auto_load()

    # ── UI Build ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ── Canvas ─────────────────────────────────────────────────────────
        self._canvas = DrawingCanvas()
        self._canvas.setMinimumSize(600, 500)
        splitter.addWidget(self._canvas)

        # Connect canvas signals
        self._canvas.shape_created.connect(self._on_shape_created)
        self._canvas.shape_deleted.connect(self._on_shape_deleted)
        self._canvas.shape_updated.connect(self._on_shape_updated)
        self._canvas.undo_changed.connect(self._on_undo_changed)
        self._canvas.redo_changed.connect(self._on_redo_changed)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._canvas.navigate_prev.connect(self._prev_image)
        self._canvas.navigate_next.connect(self._next_image)

        # ── Right panel ─────────────────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(340)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        splitter.setSizes([940, 340])

        # Zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_row = QHBoxLayout(zoom_group)
        self._zoom_label = QLabel("100%")
        btn_zi = QPushButton("+")
        btn_zi.setFixedWidth(28)
        btn_zi.clicked.connect(self._canvas.zoom_in)
        btn_zo = QPushButton("−")
        btn_zo.setFixedWidth(28)
        btn_zo.clicked.connect(self._canvas.zoom_out)
        btn_zr = QPushButton("Fit")
        btn_zr.clicked.connect(self._canvas.zoom_reset)
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addWidget(btn_zi)
        zoom_row.addWidget(btn_zo)
        zoom_row.addWidget(btn_zr)
        right_layout.addWidget(zoom_group)

        # Undo/Redo
        undo_group = QGroupBox("Undo / Redo")
        undo_row = QHBoxLayout(undo_group)
        self._btn_undo = QPushButton("↩ Undo")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._canvas.undo)
        self._btn_redo = QPushButton("↪ Redo")
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._canvas.redo)
        undo_row.addWidget(self._btn_undo)
        undo_row.addWidget(self._btn_redo)
        right_layout.addWidget(undo_group)

        # Load image
        img_group = QGroupBox("Ảnh")
        img_layout = QVBoxLayout(img_group)
        load_row = QHBoxLayout()
        btn_load_img = QPushButton("📂 Tải ảnh")
        btn_load_img.clicked.connect(self._load_image)
        load_row.addWidget(btn_load_img)
        btn_load_folder = QPushButton("📁 Mở Folder")
        btn_load_folder.clicked.connect(self._load_folder)
        load_row.addWidget(btn_load_folder)
        img_layout.addLayout(load_row)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Previous")
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next = QPushButton("▶ Next")
        self._btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        img_layout.addLayout(nav_row)

        self._img_nav_label = QLabel("")
        self._img_nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_layout.addWidget(self._img_nav_label)

        right_layout.addWidget(img_group)

        # Class settings
        cls_group = QGroupBox("Class")
        cls_layout = QVBoxLayout(cls_group)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("ID:"))
        self._class_id_spin = QSpinBox()
        self._class_id_spin.setRange(0, 9999)
        self._class_id_spin.setValue(self._draw_class_id)
        self._class_id_spin.valueChanged.connect(self._on_class_id_changed)
        id_row.addWidget(self._class_id_spin)
        cls_layout.addLayout(id_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._class_name_edit = QLineEdit(self._draw_class_name)
        self._class_name_edit.textChanged.connect(self._on_class_name_changed)
        name_row.addWidget(self._class_name_edit)
        cls_layout.addLayout(name_row)

        right_layout.addWidget(cls_group)

        # Drawing settings
        draw_group = QGroupBox("Cài đặt vẽ")
        draw_layout = QVBoxLayout(draw_group)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Màu:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(32, 24)
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_btn)
        draw_layout.addLayout(color_row)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Độ dày:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 20)
        self._width_spin.setValue(self._draw_width)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        width_row.addWidget(self._width_spin)
        draw_layout.addLayout(width_row)

        right_layout.addWidget(draw_group)

        # Shapes list
        shapes_group = QGroupBox("Shapes")
        shapes_layout = QVBoxLayout(shapes_group)
        self._shapes_list = QListWidget()
        self._shapes_list.setMaximumHeight(180)
        self._shapes_list.currentRowChanged.connect(self._on_shapes_list_row_changed)
        shapes_layout.addWidget(self._shapes_list)

        shape_btns = QHBoxLayout()
        self._btn_del_shape = QPushButton("🗑 Xóa")
        self._btn_del_shape.clicked.connect(self._delete_selected_shape)
        self._btn_edit_shape = QPushButton("✏️ F1")
        self._btn_edit_shape.clicked.connect(self._open_edit_dialog)
        self._btn_insert_pt = QPushButton("🖱 F2")
        self._btn_insert_pt.setCheckable(True)
        self._btn_insert_pt.clicked.connect(self._toggle_insert_mode)
        self._btn_clear_shapes = QPushButton("🗑 Xóa hết")
        self._btn_clear_shapes.clicked.connect(self._clear_all_shapes)
        shape_btns.addWidget(self._btn_del_shape)
        shape_btns.addWidget(self._btn_edit_shape)
        shape_btns.addWidget(self._btn_insert_pt)
        shape_btns.addWidget(self._btn_clear_shapes)
        shapes_layout.addLayout(shape_btns)
        right_layout.addWidget(shapes_group)

        # Image list (multi-image management)
        img_list_group = QGroupBox("Danh sách ảnh")
        img_list_layout = QVBoxLayout(img_list_group)
        self._image_list = QListWidget()
        self._image_list.setMaximumHeight(160)
        self._image_list.itemDoubleClicked.connect(self._on_image_list_dbl_click)
        img_list_layout.addWidget(self._image_list)
        right_layout.addWidget(img_list_group)

        # Save / Load / Export
        io_group = QGroupBox("Save / Load / Export")
        io_layout = QVBoxLayout(io_group)

        row1 = QHBoxLayout()
        btn_save = QPushButton("💾 Save JSON")
        btn_save.clicked.connect(self._save_json)
        btn_load = QPushButton("📂 Load JSON")
        btn_load.clicked.connect(self._load_json)
        row1.addWidget(btn_save)
        row1.addWidget(btn_load)
        io_layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_save_proj = QPushButton("💾 Lưu Project")
        btn_save_proj.clicked.connect(self._save_project)
        btn_load_proj = QPushButton("📂 Mở Project")
        btn_load_proj.clicked.connect(self._load_project)
        row2.addWidget(btn_save_proj)
        row2.addWidget(btn_load_proj)
        io_layout.addLayout(row2)

        btn_export = QPushButton("📤 Export YOLO")
        btn_export.clicked.connect(self._export_yolo)
        io_layout.addWidget(btn_export)

        right_layout.addWidget(io_group)
        right_layout.addStretch()

    # ── Image management ─────────────────────────────────────────────────────

    def _load_image(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Tải ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        if not paths:
            return
        # Switch to first selected, add all to cache
        for p in paths:
            if p not in self.image_cache:
                self.image_cache[p] = []
        self._switch_to_image(paths[0])
        self._refresh_image_list()

    def _switch_to_image(self, path: str) -> None:
        # Save current shapes to cache
        if self._current_image:
            self._save_current_to_cache()

        # Load new image
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Lỗi", f"File không tồn tại:\n{path}")
            if path in self.image_cache:
                del self.image_cache[path]
            self._refresh_image_list()
            return

        self._current_image = path
        self._canvas.set_image(path, keep_shapes=False)
        self._load_from_cache(path)
        self._refresh_image_list()
        self._refresh_shapes_list()
        self._update_nav_label()
        self._update_nav_buttons()

    def _save_current_to_cache(self) -> None:
        if self._current_image:
            self.image_cache[self._current_image] = self._canvas.get_shapes()

    def _load_from_cache(self, path: str) -> None:
        shapes = self.image_cache.get(path, [])
        if shapes:
            self._canvas.load_shapes(shapes)

    def _load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Mở Folder ảnh")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
        files = sorted(
            (
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in exts
            ),
            key=lambda p: os.path.basename(p).lower(),
        )
        if not files:
            QMessageBox.information(self, "Thông báo", "Không tìm thấy ảnh nào trong thư mục.")
            return
        for p in files:
            if p not in self.image_cache:
                self.image_cache[p] = []
        self._switch_to_image(files[0])
        self._refresh_image_list()

    def _get_image_paths(self) -> list:
        """Return list of image paths in the cache (insertion order)."""
        return list(self.image_cache.keys())

    def _prev_image(self) -> None:
        paths = self._get_image_paths()
        if not paths or not self._current_image:
            return
        try:
            idx = paths.index(self._current_image)
        except ValueError:
            return
        new_idx = (idx - 1) % len(paths)
        self._switch_to_image(paths[new_idx])

    def _next_image(self) -> None:
        paths = self._get_image_paths()
        if not paths or not self._current_image:
            return
        try:
            idx = paths.index(self._current_image)
        except ValueError:
            return
        new_idx = (idx + 1) % len(paths)
        self._switch_to_image(paths[new_idx])

    def _update_nav_label(self) -> None:
        paths = self._get_image_paths()
        if not paths or not self._current_image:
            self._img_nav_label.setText("")
            return
        try:
            idx = paths.index(self._current_image)
            self._img_nav_label.setText(f"{idx + 1} / {len(paths)}")
        except ValueError:
            self._img_nav_label.setText("")

    def _update_nav_buttons(self) -> None:
        has_nav = len(self.image_cache) > 1
        self._btn_prev.setEnabled(has_nav)
        self._btn_next.setEnabled(has_nav)

    # ── Image list widget ────────────────────────────────────────────────────

    def _refresh_image_list(self) -> None:
        self._image_list.blockSignals(True)
        self._image_list.clear()
        for path, shapes in self.image_cache.items():
            name = os.path.basename(path)
            count = len(shapes)
            item = QListWidgetItem(f"{name}  [{count}]")
            item.setData(Qt.ItemDataRole.UserRole, path)
            if path == self._current_image:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QColor(0, 180, 0))
            self._image_list.addItem(item)
        self._image_list.blockSignals(False)
        self._update_nav_buttons()

    def _on_image_list_dbl_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and path != self._current_image:
            self._switch_to_image(path)

    # ── Shapes list widget ───────────────────────────────────────────────────

    def _refresh_shapes_list(self) -> None:
        self._shapes_list.blockSignals(True)
        self._shapes_list.clear()
        for sd in self._canvas.get_shapes():
            label = f"[{sd['shape_id']}] {sd['class_name']} ({len(sd['points'])}pt)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, sd["shape_id"])
            self._shapes_list.addItem(item)
        self._shapes_list.blockSignals(False)

    def _on_shapes_list_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._shapes_list.item(row)
        if item:
            sid = item.data(Qt.ItemDataRole.UserRole)
            self._canvas.select_shape(sid)

    # ── Canvas signal handlers ────────────────────────────────────────────────

    def _on_shape_created(self, data: dict) -> None:
        self._class_map[data["class_id"]] = data["class_name"]
        self._save_current_to_cache()
        self._refresh_shapes_list()
        self._refresh_image_list()

    def _on_shape_deleted(self, shape_id: int) -> None:
        self._save_current_to_cache()
        self._refresh_shapes_list()
        self._refresh_image_list()

    def _on_shape_updated(self, data: dict) -> None:
        if data.get("__EDIT__"):
            self._open_edit_dialog()
            return
        self._save_current_to_cache()
        self._refresh_shapes_list()
        self._refresh_image_list()

    def _on_undo_changed(self, can_undo: bool) -> None:
        self._btn_undo.setEnabled(can_undo)

    def _on_redo_changed(self, can_redo: bool) -> None:
        self._btn_redo.setEnabled(can_redo)

    def _on_zoom_changed(self, scale: float) -> None:
        self._zoom_label.setText(f"{scale * 100:.0f}%")

    # ── Shape actions ────────────────────────────────────────────────────────

    def _delete_selected_shape(self) -> None:
        shape = self._canvas.get_selected_shape()
        if shape:
            self._canvas.delete_shape(shape.shape_id)

    def _open_edit_dialog(self) -> None:
        shape = self._canvas.get_selected_shape()
        if shape is None:
            QMessageBox.information(self, "Thông báo", "Chưa chọn shape nào.")
            return
        dlg = ShapeEditDialog(
            class_id=shape.class_id,
            class_name=shape.class_name,
            color=shape.color,
            width=shape.width,
            description=shape.description,
            points=shape.points,
            class_map=self._class_map,
            parent=self,
        )
        if dlg.exec():
            result = dlg.get_result()
            self._canvas.update_shape_from_dict(shape.shape_id, result)
            self._class_map[result["class_id"]] = result["class_name"]

    def _toggle_insert_mode(self, checked: bool) -> None:
        self._canvas.set_insert_mode(checked)
        self._canvas.setFocus()

    def _clear_all_shapes(self) -> None:
        reply = QMessageBox.question(
            self, "Xóa hết", "Xóa tất cả shapes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._canvas.clear_shapes()

    # ── Drawing default controls ──────────────────────────────────────────────

    def _update_color_btn(self) -> None:
        r, g, b = self._draw_color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(*self._draw_color), self, "Chọn màu vẽ")
        if chosen.isValid():
            self._draw_color = (chosen.red(), chosen.green(), chosen.blue())
            self._update_color_btn()
            self._canvas.set_drawing_defaults(
                self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
            )

    def _on_class_id_changed(self, value: int) -> None:
        self._draw_class_id = value
        if value in self._class_map:
            self._class_name_edit.blockSignals(True)
            self._class_name_edit.setText(self._class_map[value])
            self._class_name_edit.blockSignals(False)
            self._draw_class_name = self._class_map[value]
        self._canvas.set_drawing_defaults(
            self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
        )

    def _on_class_name_changed(self, name: str) -> None:
        self._draw_class_name = name
        self._class_map[self._draw_class_id] = name
        self._canvas.set_drawing_defaults(
            self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
        )

    def _on_width_changed(self, value: int) -> None:
        self._draw_width = value
        self._canvas.set_drawing_defaults(
            self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
        )

    # ── Save / Load JSON ──────────────────────────────────────────────────────

    def _save_json(self) -> None:
        self._save_current_to_cache()
        path, _ = QFileDialog.getSaveFileName(self, "Lưu JSON", "", "JSON (*.json)")
        if not path:
            return
        data = {
            "class_map": {str(k): v for k, v in self._class_map.items()},
            "images": {
                img_path: shapes
                for img_path, shapes in self.image_cache.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Lưu JSON", f"Đã lưu: {path}")

    def _load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Tải JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            QMessageBox.warning(self, "Lỗi JSON", f"Không thể đọc file:\n{exc}")
            return

        self._class_map = {int(k): v for k, v in data.get("class_map", {}).items()}
        raw_images: dict = data.get("images", {})

        # Filter images that no longer exist and warn
        missing = [p for p in raw_images if not os.path.isfile(p)]
        self.image_cache = {p: v for p, v in raw_images.items() if os.path.isfile(p)}
        if missing:
            QMessageBox.warning(
                self, "Ảnh bị thiếu",
                "Các ảnh sau không còn tồn tại và sẽ bị bỏ qua:\n" + "\n".join(missing),
            )

        # Update UI
        self._current_image = ""
        self._canvas.reset()
        self._refresh_shapes_list()
        self._refresh_image_list()

        if not self.image_cache:
            QMessageBox.information(self, "Thông báo", "Không có ảnh nào hợp lệ sau khi tải.")
            return

        # Auto-load first valid image
        for img_path in self.image_cache:
            self._switch_to_image(img_path)
            break

        # Update class controls to first class in map
        if self._class_map:
            first_id = min(self._class_map)
            self._class_id_spin.blockSignals(True)
            self._class_id_spin.setValue(first_id)
            self._class_id_spin.blockSignals(False)
            self._class_name_edit.blockSignals(True)
            self._class_name_edit.setText(self._class_map[first_id])
            self._class_name_edit.blockSignals(False)
            self._draw_class_id = first_id
            self._draw_class_name = self._class_map[first_id]
            self._canvas.set_drawing_defaults(
                self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
            )

    # ── Export YOLO ───────────────────────────────────────────────────────────

    def _export_yolo(self) -> None:
        self._save_current_to_cache()
        out_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục Export YOLO")
        if not out_dir:
            return

        count = 0
        for img_path, shapes in self.image_cache.items():
            if not shapes:
                continue
            from PySide6.QtGui import QImage
            qimg = QImage(img_path)
            if qimg.isNull():
                continue
            iw, ih = qimg.width(), qimg.height()
            lines = []
            for sd in shapes:
                pts = sd["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2 / iw
                cy = (y_min + y_max) / 2 / ih
                bw = (x_max - x_min) / iw
                bh = (y_max - y_min) / ih
                lines.append(f"{sd['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                base = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(out_dir, base + ".txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                count += 1

        QMessageBox.information(self, "Export YOLO", f"Đã xuất {count} file vào:\n{out_dir}")

    # ── Auto-save / Auto-load ─────────────────────────────────────────────────

    _AUTOSAVE_PATH = Path.home() / ".polygon_tool_autosave.json"

    def _auto_save(self) -> None:
        """Silently save full state to ~/.polygon_tool_autosave.json."""
        try:
            self._save_current_to_cache()
            data = {
                "class_map": {str(k): v for k, v in self._class_map.items()},
                "images": dict(self.image_cache),
                "current_image": self._current_image,
                "draw_color": list(self._draw_color),
                "draw_width": self._draw_width,
                "draw_class_id": self._draw_class_id,
                "draw_class_name": self._draw_class_name,
            }
            with open(self._AUTOSAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Never block exit

    def _auto_load(self) -> None:
        """Restore state from autosave file if it exists."""
        if not self._AUTOSAVE_PATH.is_file():
            return
        try:
            with open(self._AUTOSAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            import logging
            logging.warning("polygon_tool: autosave file is corrupt, skipping.")
            return

        self._class_map = {int(k): v for k, v in data.get("class_map", {}).items()}

        # Restore drawing defaults
        try:
            raw_color = data["draw_color"]
            if isinstance(raw_color, (list, tuple)) and len(raw_color) == 3:
                self._draw_color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
            self._draw_width = int(data["draw_width"])
            self._draw_class_id = int(data["draw_class_id"])
            self._draw_class_name = str(data["draw_class_name"])
        except (KeyError, TypeError, ValueError):
            pass

        # Restore image cache, skipping missing files
        raw_images: dict = data.get("images", {})
        self.image_cache = {p: v for p, v in raw_images.items() if os.path.isfile(p)}

        # Update class controls
        if self._class_map:
            self._class_id_spin.blockSignals(True)
            self._class_id_spin.setValue(self._draw_class_id)
            self._class_id_spin.blockSignals(False)
            self._class_name_edit.blockSignals(True)
            self._class_name_edit.setText(self._draw_class_name)
            self._class_name_edit.blockSignals(False)
        self._update_color_btn()
        self._width_spin.blockSignals(True)
        self._width_spin.setValue(self._draw_width)
        self._width_spin.blockSignals(False)
        self._canvas.set_drawing_defaults(
            self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
        )

        self._refresh_image_list()

        if not self.image_cache:
            return

        # Try to restore current image, fall back to first available
        current = data.get("current_image", "")
        if current and os.path.isfile(current) and current in self.image_cache:
            self._switch_to_image(current)
        else:
            self._switch_to_image(next(iter(self.image_cache)))

    # ── closeEvent ────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        self._auto_save()
        super().closeEvent(event)

    # ── Save / Load Project ───────────────────────────────────────────────────

    def _save_project(self) -> None:
        """Save full project (shapes + settings) to a JSON file."""
        self._save_current_to_cache()
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Project", "", "JSON (*.json)"
        )
        if not path:
            return
        data = {
            "version": 1,
            "class_map": {str(k): v for k, v in self._class_map.items()},
            "images": dict(self.image_cache),
            "current_image": self._current_image,
            "settings": {
                "draw_color": list(self._draw_color),
                "draw_width": self._draw_width,
                "draw_class_id": self._draw_class_id,
                "draw_class_name": self._draw_class_name,
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Lỗi", f"Không thể ghi file:\n{exc}")
            return
        QMessageBox.information(self, "Lưu Project", f"Đã lưu: {path}")

    def _load_project(self) -> None:
        """Load a full project file (shapes + settings)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở Project", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            QMessageBox.warning(self, "Lỗi JSON", f"Không thể đọc file:\n{exc}")
            return

        self._class_map = {int(k): v for k, v in data.get("class_map", {}).items()}

        # Restore settings if present
        settings = data.get("settings", {})
        try:
            raw_color = settings["draw_color"]
            if isinstance(raw_color, (list, tuple)) and len(raw_color) == 3:
                self._draw_color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
            self._draw_width = int(settings["draw_width"])
            self._draw_class_id = int(settings["draw_class_id"])
            self._draw_class_name = str(settings["draw_class_name"])
        except (KeyError, TypeError, ValueError):
            pass

        # Restore image cache, filtering missing files
        raw_images: dict = data.get("images", {})
        missing = [p for p in raw_images if not os.path.isfile(p)]
        self.image_cache = {p: v for p, v in raw_images.items() if os.path.isfile(p)}
        if missing:
            QMessageBox.warning(
                self, "Ảnh bị thiếu",
                "Các ảnh sau không còn tồn tại và sẽ bị bỏ qua:\n" + "\n".join(missing),
            )

        # Update controls
        if self._class_map:
            self._class_id_spin.blockSignals(True)
            self._class_id_spin.setValue(self._draw_class_id)
            self._class_id_spin.blockSignals(False)
            self._class_name_edit.blockSignals(True)
            self._class_name_edit.setText(self._draw_class_name)
            self._class_name_edit.blockSignals(False)
        self._update_color_btn()
        self._width_spin.blockSignals(True)
        self._width_spin.setValue(self._draw_width)
        self._width_spin.blockSignals(False)
        self._canvas.set_drawing_defaults(
            self._draw_color, self._draw_width, self._draw_class_id, self._draw_class_name
        )

        self._current_image = ""
        self._canvas.reset()
        self._refresh_shapes_list()
        self._refresh_image_list()

        if not self.image_cache:
            QMessageBox.information(self, "Thông báo", "Không có ảnh nào hợp lệ sau khi tải.")
            return

        # Try current_image from project, else first available
        current = data.get("current_image", "")
        if current and os.path.isfile(current) and current in self.image_cache:
            self._switch_to_image(current)
        else:
            self._switch_to_image(next(iter(self.image_cache)))
