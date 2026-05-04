from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from models import Shape


class DrawingCanvas(QWidget):
    """Canvas widget for polygon drawing with zoom, pan, and point-insertion (F2)."""

    shape_created = Signal(dict)
    shape_deleted = Signal(int)   # shape_id
    shape_updated = Signal(dict)
    undo_changed = Signal(bool)   # can_undo
    redo_changed = Signal(bool)   # can_redo
    zoom_changed = Signal(float)  # zoom_level
    navigate_prev = Signal()
    navigate_next = Signal()

    # ── Init ────────────────────────────────────────────────────────────────

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Image
        self._pixmap: Optional[QPixmap] = None
        self._img_path: str = ""

        # View transform
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Pan state
        self._pan_active: bool = False
        self._pan_last: Optional[QPointF] = None

        # Shapes
        self._shapes: List[Shape] = []
        self._next_id: int = 1

        # Current drawing
        self._drawing_points: List[Tuple[float, float]] = []
        self._mouse_pos: Optional[Tuple[float, float]] = None

        # Selection / drag
        self._selected_id: Optional[int] = None
        self._drag_vertex_idx: Optional[int] = None

        # F2 insert-point mode
        self._insert_mode: bool = False
        self._insert_edge_index: Optional[int] = None
        self._insert_preview_pt: Optional[Tuple[float, float]] = None

        # Hover tracking
        self._hover_vertex_idx: Optional[int] = None
        self._hover_shape_id: Optional[int] = None

        # Drawing settings
        self._default_color: Tuple[int, int, int] = (255, 0, 0)
        self._default_width: int = 2
        self._default_class_id: int = 0
        self._default_class_name: str = "object"

        # Undo / Redo stacks (list of shape-list snapshots)
        self._undo_stack: List[List[dict]] = []
        self._redo_stack: List[List[dict]] = []

    # ── Public helpers ──────────────────────────────────────────────────────

    def set_drawing_defaults(
        self,
        color: Tuple[int, int, int],
        width: int,
        class_id: int,
        class_name: str,
    ) -> None:
        self._default_color = color
        self._default_width = width
        self._default_class_id = class_id
        self._default_class_name = class_name

    def set_image(self, path: str, keep_shapes: bool = False) -> None:
        self._pixmap = QPixmap(path)
        self._img_path = path
        if not keep_shapes:
            self._shapes.clear()
            self._next_id = 1
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._emit_undo_redo()
        self._drawing_points.clear()
        self._selected_id = None
        self._fit_to_window()
        self.update()

    def load_shapes(self, shapes_data: List[dict]) -> None:
        self._shapes = [Shape.from_dict(d) for d in shapes_data]
        max_id = max((s.shape_id for s in self._shapes), default=0)
        self._next_id = max_id + 1
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_undo_redo()
        self.update()

    def get_shapes(self) -> List[dict]:
        return [s.to_dict() for s in self._shapes]

    def get_selected_shape(self) -> Optional[Shape]:
        if self._selected_id is None:
            return None
        return next((s for s in self._shapes if s.shape_id == self._selected_id), None)

    def select_shape(self, shape_id: Optional[int]) -> None:
        self._selected_id = shape_id
        self.update()

    def reset(self) -> None:
        """Clear all shapes and reset state without emitting per-shape signals."""
        self._shapes.clear()
        self._next_id = 1
        self._drawing_points.clear()
        self._selected_id = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_undo_redo()
        self.update()

    def update_shape_from_dict(self, shape_id: int, data: dict) -> None:
        self._push_undo()
        for i, s in enumerate(self._shapes):
            if s.shape_id == shape_id:
                self._shapes[i] = Shape(
                    shape_id=shape_id,
                    class_id=data["class_id"],
                    class_name=data["class_name"],
                    shape_type=s.shape_type,
                    points=[tuple(p) for p in data["points"]],
                    color=tuple(data["color"]),
                    width=data["width"],
                    description=data["description"],
                )
                self.shape_updated.emit(self._shapes[i].to_dict())
                break
        self.update()

    def delete_shape(self, shape_id: int) -> None:
        self._push_undo()
        self._shapes = [s for s in self._shapes if s.shape_id != shape_id]
        if self._selected_id == shape_id:
            self._selected_id = None
        self.shape_deleted.emit(shape_id)
        self.update()

    def clear_shapes(self) -> None:
        self._push_undo()
        ids = [s.shape_id for s in self._shapes]
        self._shapes.clear()
        self._selected_id = None
        for sid in ids:
            self.shape_deleted.emit(sid)
        self.update()

    def set_insert_mode(self, enabled: bool) -> None:
        self._insert_mode = enabled
        self._insert_edge_index = None
        self._insert_preview_pt = None
        self.update()

    # ── Zoom / Pan ──────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(0.8)

    def zoom_reset(self) -> None:
        self._fit_to_window()
        self.update()

    def _zoom_by(self, factor: float, center: Optional[QPointF] = None) -> None:
        new_scale = max(0.05, min(50.0, self._scale * factor))
        if center is None:
            center = QPointF(self.width() / 2, self.height() / 2)
        # Zoom toward cursor
        self._offset_x = center.x() - (center.x() - self._offset_x) * (new_scale / self._scale)
        self._offset_y = center.y() - (center.y() - self._offset_y) * (new_scale / self._scale)
        self._scale = new_scale
        self.zoom_changed.emit(self._scale)
        self.update()

    def _fit_to_window(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width() or 800, self.height() or 600
        sx = ww / pw if pw else 1.0
        sy = wh / ph if ph else 1.0
        self._scale = min(sx, sy) * 0.95
        self._offset_x = (ww - pw * self._scale) / 2
        self._offset_y = (wh - ph * self._scale) / 2
        self.zoom_changed.emit(self._scale)

    # ── Coordinate helpers ──────────────────────────────────────────────────

    def _to_img(self, wx: float, wy: float) -> Tuple[float, float]:
        """Widget coords → image coords."""
        return (wx - self._offset_x) / self._scale, (wy - self._offset_y) / self._scale

    def _to_widget(self, ix: float, iy: float) -> Tuple[float, float]:
        """Image coords → widget coords."""
        return ix * self._scale + self._offset_x, iy * self._scale + self._offset_y

    def _clamp_to_image(self, ix: float, iy: float) -> Tuple[float, float]:
        """Giới hạn tọa độ trong phạm vi ảnh."""
        if self._pixmap is None or self._pixmap.isNull():
            return ix, iy
        return (
            max(0.0, min(float(self._pixmap.width() - 1), ix)),
            max(0.0, min(float(self._pixmap.height() - 1), iy)),
        )

    # ── Undo / Redo ─────────────────────────────────────────────────────────

    def _snapshot(self) -> List[dict]:
        return [s.to_dict() for s in self._shapes]

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self._emit_undo_redo()

    def _emit_undo_redo(self) -> None:
        self.undo_changed.emit(bool(self._undo_stack))
        self.redo_changed.emit(bool(self._redo_stack))

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        state = self._undo_stack.pop()
        self._restore_snapshot(state)
        self._emit_undo_redo()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        state = self._redo_stack.pop()
        self._restore_snapshot(state)
        self._emit_undo_redo()

    def _restore_snapshot(self, state: List[dict]) -> None:
        self._shapes = [Shape.from_dict(d) for d in state]
        max_id = max((s.shape_id for s in self._shapes), default=0)
        self._next_id = max_id + 1
        self._selected_id = None
        self.update()

    # ── Hit testing ─────────────────────────────────────────────────────────

    def _hit_vertex(
        self, shape: Shape, ix: float, iy: float, radius: float = 6.0
    ) -> Optional[int]:
        r2 = (radius / self._scale) ** 2
        for idx, (px, py) in enumerate(shape.points):
            if (px - ix) ** 2 + (py - iy) ** 2 <= r2:
                return idx
        return None

    def _point_in_polygon(
        self, pts: List[Tuple[float, float]], x: float, y: float
    ) -> bool:
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _find_nearest_edge(
        self, shape: Shape, ix: float, iy: float
    ) -> Tuple[int, float]:
        """Return (edge_index, dist²) for the nearest edge to (ix, iy)."""
        best_idx, best_d2 = 0, float("inf")
        pts = shape.points
        n = len(pts)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            d2 = self._dist2_point_to_segment(ix, iy, ax, ay, bx, by)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx, best_d2

    @staticmethod
    def _dist2_point_to_segment(
        px: float, py: float,
        ax: float, ay: float,
        bx: float, by: float,
    ) -> float:
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap and not self._pixmap.isNull():
            pw = self._pixmap.width() * self._scale
            ph = self._pixmap.height() * self._scale
            painter.drawPixmap(
                int(self._offset_x), int(self._offset_y), int(pw), int(ph),
                self._pixmap,
            )

        # Draw finished shapes
        for shape in self._shapes:
            self._draw_shape(painter, shape)

        # Draw in-progress polygon
        if self._drawing_points:
            self._draw_in_progress(painter)

    def _draw_shape(self, painter: QPainter, shape: Shape) -> None:
        pts_w = [self._to_widget(x, y) for x, y in shape.points]
        color = QColor(*shape.color)
        is_selected = shape.shape_id == self._selected_id

        # F2 insert preview: replace nearest edge with two dashed segments
        if self._insert_mode and is_selected and self._insert_edge_index is not None and self._insert_preview_pt:
            ei = self._insert_edge_index
            mouse_w = self._to_widget(*self._insert_preview_pt)
            n = len(pts_w)

            # Draw polygon edges, substituting the preview edge
            pen = QPen(color, shape.width)
            painter.setPen(pen)
            for i in range(n):
                a = pts_w[i]
                b = pts_w[(i + 1) % n]
                if i == ei:
                    # Dashed A→mouse and mouse→B
                    dash_pen = QPen(color, shape.width, Qt.PenStyle.DashLine)
                    painter.setPen(dash_pen)
                    painter.drawLine(
                        QPointF(*a), QPointF(*mouse_w)
                    )
                    painter.drawLine(
                        QPointF(*mouse_w), QPointF(*b)
                    )
                    painter.setPen(pen)
                else:
                    painter.drawLine(QPointF(*a), QPointF(*b))

            # Preview point (green)
            preview_pen = QPen(QColor(0, 220, 0), 2)
            painter.setPen(preview_pen)
            painter.setBrush(QColor(0, 220, 0, 180))
            mx, my = mouse_w
            painter.drawEllipse(QPointF(mx, my), 6, 6)
        else:
            # Normal polygon draw
            pen = QPen(color, shape.width)
            if is_selected:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))

            from PySide6.QtGui import QPolygonF
            poly = QPolygonF([QPointF(wx, wy) for wx, wy in pts_w])
            painter.drawPolygon(poly)

        # Vertices
        v_pen = QPen(QColor(255, 255, 255), 1)
        painter.setPen(v_pen)
        painter.setBrush(color)
        for idx, (wx, wy) in enumerate(pts_w):
            is_hover = (
                shape.shape_id == self._hover_shape_id
                and idx == self._hover_vertex_idx
            )
            if is_hover:
                # Hover: yellow border, larger radius
                painter.setPen(QPen(QColor(255, 220, 0), 2))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(wx, wy), 8, 8)
                painter.setPen(v_pen)
                painter.setBrush(color)
            elif is_selected:
                # Selected: white border, larger radius
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(wx, wy), 6, 6)
                painter.setPen(v_pen)
                painter.setBrush(color)
            else:
                painter.drawEllipse(QPointF(wx, wy), 4, 4)

        # Label
        if pts_w:
            lx, ly = pts_w[0]
            painter.setPen(QColor(0, 0, 0))
            font = QFont("Arial", 12)
            font.setWeight(QFont.Bold)  # hoặc QFont.Black
            painter.setFont(font)
            painter.drawText(QPointF(lx + 4, ly - 4), f"{shape.class_name}:{shape.shape_id} class:{shape.class_id}")

    def _draw_in_progress(self, painter: QPainter) -> None:
        pts_w = [self._to_widget(x, y) for x, y in self._drawing_points]
        color = QColor(*self._default_color)
        pen = QPen(color, self._default_width)
        painter.setPen(pen)

        # Draw existing segments
        for i in range(len(pts_w) - 1):
            painter.drawLine(QPointF(*pts_w[i]), QPointF(*pts_w[i + 1]))

        # Preview: last point → mouse → first point (dashed)
        if self._mouse_pos and len(pts_w) >= 1:
            mouse_w = self._to_widget(*self._mouse_pos)
            dash_pen = QPen(color, self._default_width, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(QPointF(*pts_w[-1]), QPointF(*mouse_w))
            if len(pts_w) >= 2:
                painter.drawLine(QPointF(*mouse_w), QPointF(*pts_w[0]))

        # Vertices
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(color)
        for wx, wy in pts_w:
            painter.drawEllipse(QPointF(wx, wy), 5, 5)

    # ── Mouse events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        ix, iy = self._to_img(pos.x(), pos.y())

        # Middle/Right drag → pan
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan_active = True
            self._pan_last = pos
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        # F2 insert mode
        if self._insert_mode:
            sel = self.get_selected_shape()
            if sel and self._insert_edge_index is not None:
                self._push_undo()
                ix, iy = self._clamp_to_image(ix, iy)
                pts = list(sel.points)
                pts.insert(self._insert_edge_index + 1, (ix, iy))
                sel.points = pts
                self.shape_updated.emit(sel.to_dict())
                # Update preview to reflect new state
                self._update_insert_preview(ix, iy)
                self.update()
            return

        # Ctrl+Click → delete vertex
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            sel = self.get_selected_shape()
            if sel:
                vi = self._hit_vertex(sel, ix, iy)
                if vi is not None:
                    self._push_undo()
                    sel.points = [p for i, p in enumerate(sel.points) if i != vi]
                    if len(sel.points) < 3:
                        self.delete_shape(sel.shape_id)
                    else:
                        self.shape_updated.emit(sel.to_dict())
                    self.update()
                    return

        # Start drag on existing vertex
        sel = self.get_selected_shape()
        if sel:
            vi = self._hit_vertex(sel, ix, iy)
            if vi is not None:
                self._drag_vertex_idx = vi
                self._push_undo()
                return

        # Click inside a shape → select
        for shape in reversed(self._shapes):
            if self._point_in_polygon(shape.points, ix, iy):
                self._selected_id = shape.shape_id
                self.update()
                return

        # Deselect
        if self._selected_id is not None:
            self._selected_id = None
            self.update()
            return

        # Add point to current polygon
        self._drawing_points.append(self._clamp_to_image(ix, iy))
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        ix, iy = self._to_img(pos.x(), pos.y())

        # Pan
        if self._pan_active and self._pan_last is not None:
            dx = pos.x() - self._pan_last.x()
            dy = pos.y() - self._pan_last.y()
            self._offset_x += dx
            self._offset_y += dy
            self._pan_last = pos
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            self.update()
            return

        # Drag vertex
        if self._drag_vertex_idx is not None:
            sel = self.get_selected_shape()
            if sel:
                pts = list(sel.points)
                pts[self._drag_vertex_idx] = self._clamp_to_image(ix, iy)
                sel.points = pts
                self.update()
            return

        self._mouse_pos = (ix, iy)

        # F2 preview
        if self._insert_mode:
            self._update_insert_preview(ix, iy)

        # Hover vertex tracking
        hover_radius = 8.0
        new_hover_shape_id: Optional[int] = None
        new_hover_vertex_idx: Optional[int] = None

        # Check selected shape first, then all shapes
        shapes_to_check = []
        sel = self.get_selected_shape()
        if sel:
            shapes_to_check.append(sel)
        for s in self._shapes:
            if s not in shapes_to_check:
                shapes_to_check.append(s)

        for s in shapes_to_check:
            vi = self._hit_vertex(s, ix, iy, radius=hover_radius)
            if vi is not None:
                new_hover_shape_id = s.shape_id
                new_hover_vertex_idx = vi
                break

        self._hover_shape_id = new_hover_shape_id
        self._hover_vertex_idx = new_hover_vertex_idx

        # Update cursor
        if new_hover_shape_id is not None:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        elif self._insert_mode:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif self._drawing_points:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

        self.update()

    def _update_insert_preview(self, ix: float, iy: float) -> None:
        sel = self.get_selected_shape()
        if sel and len(sel.points) >= 2:
            edge_idx, _ = self._find_nearest_edge(sel, ix, iy)
            self._insert_edge_index = edge_idx
            self._insert_preview_pt = (ix, iy)  # actual mouse pos, not projected

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan_active = False
            self._pan_last = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        if event.button() == Qt.MouseButton.LeftButton and self._drag_vertex_idx is not None:
            self._drag_vertex_idx = None
            sel = self.get_selected_shape()
            if sel:
                self.shape_updated.emit(sel.to_dict())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self._zoom_by(factor, center=event.position())

    # ── Keyboard events ──────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()

        # Undo / Redo
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                self.undo()
                return
            if key == Qt.Key.Key_Y:
                self.redo()
                return

        # Finish polygon
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_polygon()
            return

        # Cancel polygon
        if key == Qt.Key.Key_Escape:
            if self._drawing_points:
                self._drawing_points.clear()
                self.update()
            elif self._selected_id is not None:
                self._selected_id = None
                self.update()
            return

        # Remove last point
        if key == Qt.Key.Key_Backspace:
            if self._drawing_points:
                self._drawing_points.pop()
                self.update()
            return

        # F1 → request edit dialog (handled upstream via shape_updated signal)
        if key == Qt.Key.Key_F1:
            self.shape_updated.emit({"__EDIT__": True, "shape_id": self._selected_id})
            return

        # F2 → toggle insert mode
        if key == Qt.Key.Key_F2:
            self.set_insert_mode(not self._insert_mode)
            return

        # A / D → navigate previous / next image
        if key == Qt.Key.Key_A:
            self.navigate_prev.emit()
            return
        if key == Qt.Key.Key_D:
            self.navigate_next.emit()
            return

        super().keyPressEvent(event)

    def _finish_polygon(self) -> None:
        if len(self._drawing_points) < 3:
            return
        self._push_undo()
        shape = Shape(
            shape_id=self._next_id,
            class_id=self._default_class_id,
            class_name=self._default_class_name,
            shape_type="polygon",
            points=list(self._drawing_points),
            color=self._default_color,
            width=self._default_width,
        )
        self._next_id += 1
        self._shapes.append(shape)
        self._selected_id = shape.shape_id
        self._drawing_points.clear()
        self.shape_created.emit(shape.to_dict())
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self._pixmap and not self._pixmap.isNull():
            self._fit_to_window()
        super().resizeEvent(event)
