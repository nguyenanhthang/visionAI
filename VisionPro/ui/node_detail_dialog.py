"""
ui/node_detail_dialog.py — v5
Fix crop_roi:
  - Không có port kết nối → kéo chuột vẽ thủ công → lưu _drawn_roi
  - Có port kết nối → hiển thị readonly, vẽ rect từ port value
  - Mode hint thông minh hiển thị đang dùng mode nào
"""
from __future__ import annotations
from typing import Optional, Any, List, Tuple
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QTabWidget, QWidget, QScrollArea, QFrame,
                                QSplitter, QPushButton, QGroupBox,
                                QSizePolicy, QApplication, QFileDialog,
                                QMessageBox)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
                            QPen, QBrush, QCursor, QMouseEvent)

from core.flow_graph import NodeInstance, FlowGraph
# PatMaxDialog imported lazily to avoid circular import
from core.tool_registry import ToolDef, ParamDef
from ui.properties_panel import ParamRow


# ════════════════════════════════════════════════════════════════════
#  Interactive image label
# ════════════════════════════════════════════════════════════════════
class InteractiveImageLabel(QLabel):
    """
    Hiển thị ảnh + overlay tương tác.
    mode="roi"      → kéo chuột chọn vùng (màu cyan)
    mode="template" → kéo chuột chọn template (màu cam)
    mode="pick"     → click lấy pixel color
    mode="view"     → chỉ xem, không tương tác
    mode="readonly" → xem + hiển thị rect cố định (port connected)
    """
    roi_changed    = Signal(int, int, int, int)
    pixel_picked   = Signal(int, int)
    template_drawn = Signal(int, int, int, int)
    origin_changed       = Signal(float, float)   # image coords (float)
    origin_angle_changed = Signal(float)          # degrees
    shape_drawn    = Signal(str, dict)      # shape_type, data (image coords)
    shapes_changed = Signal(list)           # multi-mode: list of {"type", **data}

    def __init__(self, mode="view", parent=None):
        super().__init__(parent)
        self.mode        = mode
        self._arr        = None
        self._scale      = 1.0      # = fit_scale * user_zoom (effective)
        self._fit_scale  = 1.0      # fit-to-widget scale (no user zoom)
        self._user_zoom  = 1.0      # multiplier ≥ 1 = phóng to, <1 = thu nhỏ
        self._pan_dx     = 0        # pan offset (widget coords) cộng vào _off_x
        self._pan_dy     = 0
        self._off_x      = 0
        self._off_y      = 0
        self._rect: Optional[QRect] = None
        self._drag_start: Optional[QPoint] = None
        self._dragging   = False
        self._pick_pos: Optional[Tuple[int,int]] = None
        self._readonly_rect: Optional[Tuple[int,int,int,int]] = None
        # Origin marker (PatMax pattern reference point) — image coords
        self._origin_xy: Optional[Tuple[float, float]] = None
        self._show_origin: bool = False
        self._dragging_origin: bool = False
        # Hệ trục XY tại origin — xoay được quanh tâm
        self._origin_angle: float = 0.0   # độ, 0 = X→phải, Y↓
        self._dragging_origin_rot: bool = False
        # Shape ROI ("rect" | "circle" | "ellipse" | "polygon")
        self._shape: str = "rect"
        self._shape_data: dict = {}                                # toạ độ ảnh
        self._poly_drawing: list = []                              # [(x,y), ...] đang vẽ
        # Edit-mode state cho shape đã vẽ xong (move / resize qua corner handles)
        self._edit_action: Optional[str] = None    # "move" | "tl" | "tr" | "bl" | "br"
        self._edit_anchor_w: Optional[QPoint] = None
        self._edit_orig_data: dict = {}
        # Multi-shape (opt-in qua set_multi_shape(True)). Khi tắt, behaviour
        # giống single-shape (chỉ dùng _shape_data). Khi bật, _shapes là
        # nguồn lưu trữ chính, _shape_data + _active_idx là shape đang active.
        self._multi: bool = False
        self._shapes: List[dict] = []     # mỗi entry: {"type": str, "data": dict}
        self._active_idx: Optional[int] = None

        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Optional QScrollArea parent — biết viewport size khi zoom > 1.
        self._scroll_area = None
        self._base_min_size = (400, 300)
        self.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:6px;")

        cur_map = {
            "roi": Qt.CrossCursor, "template": Qt.CrossCursor,
            "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
            "readonly": Qt.ArrowCursor,
        }
        self.setCursor(QCursor(cur_map.get(mode, Qt.ArrowCursor)))

    def set_scroll_area(self, area):
        """Liên kết label với QScrollArea cha → zoom > 1 hiện scrollbars."""
        self._scroll_area = area
        self._base_min_size = (400, 300)

    # ── Image ──────────────────────────────────────────────────────
    def set_image(self, arr: Optional[np.ndarray]):
        # Reset zoom/pan khi đổi sang ảnh có kích thước khác (fit lại từ đầu)
        if arr is None or self._arr is None or arr.shape[:2] != self._arr.shape[:2]:
            self._user_zoom = 1.0
            self._pan_dx = 0
            self._pan_dy = 0
        self._arr = arr
        self._render()

    def set_rect_from_params(self, x, y, w, h):
        """Hiển thị rect khởi tạo từ params/port (image coords)."""
        if self._arr is None:
            return
        ih, iw = self._arr.shape[:2]
        x = max(0, min(int(x), iw-1))
        y = max(0, min(int(y), ih-1))
        w = max(1, min(int(w), iw-x))
        h = max(1, min(int(h), ih-y))
        wx, wy = self._img_to_widget(x, y)
        ww = int(w * self._scale)
        wh = int(h * self._scale)
        self._rect = QRect(wx, wy, ww, wh)
        self._render()

    def set_readonly_rect(self, x, y, w, h):
        """Hiển thị rect từ port (không cho phép kéo thay đổi)."""
        self._readonly_rect = (x, y, w, h)
        self._render()

    def set_shape_mode(self, shape: str):
        """Đặt loại shape: 'rect' | 'circle' | 'ellipse' | 'polygon'.
        XOÁ shape_data hiện tại (single-mode reset). Multi-mode dùng
        set_next_shape_type() để giữ list."""
        if shape not in ("rect", "circle", "ellipse", "polygon"):
            shape = "rect"
        self._shape = shape
        self._poly_drawing = []
        self._rect = None
        self._shape_data = {}
        self._dragging = False
        self._render()

    def set_next_shape_type(self, shape: str):
        """Multi-mode: chỉ đổi loại shape sẽ vẽ tiếp, KHÔNG xoá list/active."""
        if shape not in ("rect", "circle", "ellipse", "polygon"):
            shape = "rect"
        self._shape = shape
        self._poly_drawing = []
        self._dragging = False
        self._render()

    def set_shape_data(self, shape: str, data: dict):
        """Khôi phục shape đã train (toạ độ ảnh)."""
        self._shape = shape
        self._shape_data = dict(data) if data else {}
        self._poly_drawing = []
        # Cập nhật _rect (bbox widget) cho rendering tham chiếu
        if shape == "rect" and data:
            wx, wy = self._img_to_widget(data["x"], data["y"])
            self._rect = QRect(wx, wy,
                                int(data["w"] * self._scale),
                                int(data["h"] * self._scale))
        else:
            self._rect = None
        self._render()

    def get_shape(self) -> Tuple[str, dict]:
        return self._shape, dict(self._shape_data)

    def cancel_polygon(self):
        if self._poly_drawing:
            self._poly_drawing = []
            self._render()

    # ── Multi-shape API ────────────────────────────────────────────
    def set_multi_shape(self, enable: bool):
        """Bật/tắt multi-shape. Tắt → xoá list, giữ shape hiện tại."""
        if bool(enable) == self._multi:
            return
        self._multi = bool(enable)
        if not self._multi:
            self._shapes = []
            self._active_idx = None
        elif self._shape_data:
            # Bật — đẩy shape hiện tại (nếu có) vào list để hiển thị nhất quán
            self._shapes = [{"type": self._shape, "data": dict(self._shape_data)}]
            self._active_idx = 0
        self._render()

    def get_shapes(self) -> List[dict]:
        """List shapes cho caller. Mỗi entry: {"type", **data}."""
        if self._multi:
            out = []
            for s in self._shapes:
                e = {"type": s["type"]}
                e.update(s["data"])
                out.append(e)
            return out
        if self._shape_data:
            e = {"type": self._shape}
            e.update(self._shape_data)
            return [e]
        return []

    def set_shapes(self, shapes: List[dict]):
        """Khôi phục danh sách shapes. Mỗi entry: {"type", **data}."""
        self._shapes = []
        for s in shapes or []:
            t = s.get("type", "rect")
            d = {k: v for k, v in s.items() if k != "type"}
            self._shapes.append({"type": t, "data": d})
        if self._shapes:
            self._active_idx = len(self._shapes) - 1
            last = self._shapes[self._active_idx]
            self._shape = last["type"]
            self._shape_data = dict(last["data"])
        else:
            self._active_idx = None
            self._shape_data = {}
        self._render()

    def clear_shapes(self):
        self._shapes = []
        self._active_idx = None
        self._shape_data = {}
        self._poly_drawing = []
        self._rect = None
        self._render()
        self._emit_shapes_changed()

    def delete_active_shape(self):
        if not self._multi:
            if self._shape_data:
                self._shape_data = {}
                self._render()
                self._emit_shapes_changed()
            return
        if self._active_idx is None or not (0 <= self._active_idx < len(self._shapes)):
            return
        self._shapes.pop(self._active_idx)
        if self._shapes:
            self._active_idx = min(self._active_idx, len(self._shapes) - 1)
            last = self._shapes[self._active_idx]
            self._shape = last["type"]
            self._shape_data = dict(last["data"])
        else:
            self._active_idx = None
            self._shape_data = {}
        self._render()
        self._emit_shapes_changed()

    def _emit_shapes_changed(self):
        self.shapes_changed.emit(self.get_shapes())

    def _commit_active_to_list(self):
        """Đồng bộ _shape_data → _shapes[_active_idx] (multi-mode only)."""
        if not self._multi or self._active_idx is None:
            return
        if 0 <= self._active_idx < len(self._shapes):
            self._shapes[self._active_idx] = {
                "type": self._shape, "data": dict(self._shape_data)}

    def _bbox_of(self, stype: str, sd: dict) -> Optional[QRect]:
        if not sd or stype not in ("rect", "ellipse", "circle"):
            return None
        if stype == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            return QRect(wx - wr, wy - wr, 2 * wr, 2 * wr)
        wx, wy = self._img_to_widget(sd["x"], sd["y"])
        return QRect(wx, wy,
                      int(sd["w"] * self._scale),
                      int(sd["h"] * self._scale))

    def _hit_test_shapes_list(self, wx: int, wy: int
                                ) -> Optional[Tuple[int, str]]:
        """Hit-test trên _shapes (multi-mode). Trả (idx, action)."""
        if not self._multi:
            return None
        for i in reversed(range(len(self._shapes))):
            entry = self._shapes[i]
            if entry["type"] not in ("rect", "ellipse", "circle"):
                continue
            bb = self._bbox_of(entry["type"], entry["data"])
            if bb is None:
                continue
            for name, (cx, cy) in (
                ("tl", (bb.left(),  bb.top())),
                ("tr", (bb.right(), bb.top())),
                ("bl", (bb.left(),  bb.bottom())),
                ("br", (bb.right(), bb.bottom())),
            ):
                if abs(wx - cx) <= 8 and abs(wy - cy) <= 8:
                    return i, name
            if bb.contains(wx, wy):
                return i, "move"
        return None

    def _draw_one_shape(self, p: QPainter, stype: str, sd: dict,
                         label_num: int = 0):
        """Vẽ 1 shape không có handles — dùng cho các shape không-active trong list."""
        if not sd:
            return
        if stype == "rect":
            wx, wy = self._img_to_widget(sd["x"], sd["y"])
            ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
            p.drawRect(wx, wy, ww, wh)
            if label_num:
                p.drawText(wx + 4, wy + 14, f"#{label_num}")
        elif stype == "ellipse":
            wx, wy = self._img_to_widget(sd["x"], sd["y"])
            ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
            p.drawEllipse(wx, wy, ww, wh)
            if label_num:
                p.drawText(wx + 4, wy + 14, f"#{label_num}")
        elif stype == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            p.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
            if label_num:
                p.drawText(wx - wr + 4, wy - wr + 14, f"#{label_num}")
        elif stype == "polygon" and sd.get("pts"):
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF
            pts_w = [QPointF(*self._img_to_widget(px, py)) for px, py in sd["pts"]]
            p.drawPolygon(QPolygonF(pts_w))
            if label_num and pts_w:
                p.drawText(int(pts_w[0].x()) + 4,
                            int(pts_w[0].y()) + 14, f"#{label_num}")

    def set_origin(self, x: Optional[float], y: Optional[float]):
        """Đặt điểm tham chiếu (origin) trên ảnh. Truyền (None,None) để ẩn."""
        if x is None or y is None:
            self._origin_xy = None
            self._show_origin = False
        else:
            self._origin_xy = (float(x), float(y))
            self._show_origin = True
        self._render()

    def set_origin_angle(self, angle: float):
        """Đặt góc xoay trục XY tại origin (độ)."""
        self._origin_angle = float(angle) % 360.0
        self._render()

    # ── Origin helpers (toạ độ widget của tâm + 2 đầu trục) ──────
    AXIS_LEN  = 32
    ROT_HANDLE_OFFSET = 8        # bán kính vòng tròn xoay sau X-arrow tip
    CENTER_HIT_RADIUS = 8

    def _origin_widget_pos(self) -> Optional[Tuple[int, int]]:
        if not self._show_origin or self._origin_xy is None:
            return None
        ox, oy = self._origin_xy
        return self._img_to_widget(ox, oy)

    def _origin_axis_endpoints(self) -> Optional[Tuple[Tuple[int,int],
                                                        Tuple[int,int],
                                                        Tuple[int,int]]]:
        """Trả (center, x_end, y_end) widget coords; None nếu không hiển thị."""
        c = self._origin_widget_pos()
        if c is None:
            return None
        import math as _m
        a = _m.radians(self._origin_angle)
        cx, cy = c
        L = self.AXIS_LEN
        # X-axis: pointing along angle (image-space: y-axis downwards)
        x_end = (int(cx + _m.cos(a) * L), int(cy + _m.sin(a) * L))
        # Y-axis: 90° clockwise (theo chuẩn image: Y xuống)
        y_end = (int(cx - _m.sin(a) * L), int(cy + _m.cos(a) * L))
        return (cx, cy), x_end, y_end

    def _rot_handle_pos(self) -> Optional[Tuple[int, int]]:
        eps = self._origin_axis_endpoints()
        if eps is None:
            return None
        import math as _m
        a = _m.radians(self._origin_angle)
        cx, cy = eps[0]
        d = self.AXIS_LEN + self.ROT_HANDLE_OFFSET
        return (int(cx + _m.cos(a) * d), int(cy + _m.sin(a) * d))

    def _hit_origin_center(self, wx: int, wy: int) -> bool:
        c = self._origin_widget_pos()
        if c is None:
            return False
        dx = wx - c[0]; dy = wy - c[1]
        return (dx * dx + dy * dy) <= self.CENTER_HIT_RADIUS ** 2

    def _hit_origin_rot(self, wx: int, wy: int, radius: int = 10) -> bool:
        r = self._rot_handle_pos()
        if r is None:
            return False
        dx = wx - r[0]; dy = wy - r[1]
        return (dx * dx + dy * dy) <= radius ** 2

    def _hit_origin_handle(self, wx: int, wy: int, radius: int = 12) -> bool:
        # Backwards-compat: gộp center + rotate handle.
        return self._hit_origin_center(wx, wy) or self._hit_origin_rot(wx, wy)

    def _img_to_widget(self, ix, iy):
        return int(ix * self._scale + self._off_x), int(iy * self._scale + self._off_y)

    def _widget_to_img(self, wx, wy):
        if self._scale == 0:
            return 0, 0
        return int((wx - self._off_x) / self._scale), int((wy - self._off_y) / self._scale)

    # ── Wheel zoom (giữ pixel dưới chuột cố định) ──────────────────
    def wheelEvent(self, event):
        if self._arr is None or self._fit_scale <= 0:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(0.2, min(20.0, self._user_zoom * factor))
        if abs(new_zoom - self._user_zoom) < 1e-4:
            return
        try:
            mp = event.position()
            mx = float(mp.x()); my = float(mp.y())
        except AttributeError:
            mx = float(event.x()); my = float(event.y())
        scale_old = self._scale
        if scale_old <= 0:
            return
        # Pixel ảnh dưới con trỏ trước khi zoom (toạ độ ảnh)
        ix = (mx - self._off_x) / scale_old
        iy = (my - self._off_y) / scale_old
        self._user_zoom = new_zoom

        if self._scroll_area is not None:
            # Render lại để label resize theo zoom mới
            self._render()
            # Sau render: tính vị trí pixel cũ trên label mới, scroll để đặt
            # nó dưới con trỏ (mx, my là toạ độ trên label cũ — vẫn ổn vì
            # con trỏ đang ở vị trí widget-space của label).
            new_scale = self._scale
            new_px = ix * new_scale + self._off_x
            new_py = iy * new_scale + self._off_y
            # Lệch so với mouse → cuộn theo
            hbar = self._scroll_area.horizontalScrollBar()
            vbar = self._scroll_area.verticalScrollBar()
            hbar.setValue(hbar.value() + int(new_px - mx))
            vbar.setValue(vbar.value() + int(new_py - my))
        else:
            # Pan-offset mode (không scroll area) — giữ pixel dưới chuột
            h, w = self._arr.shape[:2]
            new_scale = self._fit_scale * new_zoom
            new_dw = int(w * new_scale); new_dh = int(h * new_scale)
            base_off_x = (self.width()  - new_dw) // 2
            base_off_y = (self.height() - new_dh) // 2
            target_off_x = mx - ix * new_scale
            target_off_y = my - iy * new_scale
            self._pan_dx = int(round(target_off_x - base_off_x))
            self._pan_dy = int(round(target_off_y - base_off_y))
            self._render()
        event.accept()

    def reset_zoom(self):
        """Reset về fit-to-widget (zoom=1, pan=0)."""
        self._user_zoom = 1.0
        self._pan_dx = 0
        self._pan_dy = 0
        self._render()

    def mouseDoubleClickEvent(self, event):
        """Double-click giữa khoảng trống (không trên shape) → reset zoom."""
        if (self._arr is not None and self._user_zoom != 1.0
                and event.button() == Qt.MiddleButton):
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ── Edit-mode helpers (move / resize shape đã vẽ xong) ─────────
    def _shape_widget_bbox(self) -> Optional[QRect]:
        """Bbox (widget coords) của shape đang lưu trong _shape_data."""
        sd = self._shape_data
        if not sd or self._shape not in ("rect", "ellipse", "circle"):
            return None
        if self._shape == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            return QRect(wx - wr, wy - wr, 2 * wr, 2 * wr)
        wx, wy = self._img_to_widget(sd["x"], sd["y"])
        ww = int(sd["w"] * self._scale)
        wh = int(sd["h"] * self._scale)
        return QRect(wx, wy, ww, wh)

    def _hit_corner(self, wx: int, wy: int, tol: int = 8) -> Optional[str]:
        bb = self._shape_widget_bbox()
        if bb is None:
            return None
        for name, (cx, cy) in (
            ("tl", (bb.left(),  bb.top())),
            ("tr", (bb.right(), bb.top())),
            ("bl", (bb.left(),  bb.bottom())),
            ("br", (bb.right(), bb.bottom())),
        ):
            if abs(wx - cx) <= tol and abs(wy - cy) <= tol:
                return name
        return None

    def _hit_body(self, wx: int, wy: int) -> bool:
        bb = self._shape_widget_bbox()
        return bb is not None and bb.contains(wx, wy)

    def _apply_edit(self, pos: QPoint):
        """Áp delta widget→ảnh, cập nhật _shape_data theo _edit_action."""
        if self._edit_anchor_w is None or not self._edit_orig_data \
                or self._scale <= 0:
            return
        dx_i = (pos.x() - self._edit_anchor_w.x()) / self._scale
        dy_i = (pos.y() - self._edit_anchor_w.y()) / self._scale
        sd = dict(self._edit_orig_data)
        if self._arr is not None:
            H, W = self._arr.shape[:2]
        else:
            H = W = 10 ** 9

        if self._shape in ("rect", "ellipse"):
            x = sd["x"]; y = sd["y"]; w = sd["w"]; h = sd["h"]
            act = self._edit_action
            if act == "move":
                x = max(0, min(int(round(x + dx_i)), W - w))
                y = max(0, min(int(round(y + dy_i)), H - h))
            elif act == "tl":
                nx = int(round(x + dx_i)); ny = int(round(y + dy_i))
                nw = w + (x - nx); nh = h + (y - ny)
                if nw >= 4 and nh >= 4 and nx >= 0 and ny >= 0:
                    x, y, w, h = nx, ny, nw, nh
            elif act == "tr":
                ny = int(round(y + dy_i))
                nh = h + (y - ny); nw = int(round(w + dx_i))
                if nw >= 4 and nh >= 4 and ny >= 0 and (x + nw) <= W:
                    y, w, h = ny, nw, nh
            elif act == "bl":
                nx = int(round(x + dx_i))
                nw = w + (x - nx); nh = int(round(h + dy_i))
                if nw >= 4 and nh >= 4 and nx >= 0 and (y + nh) <= H:
                    x, w, h = nx, nw, nh
            elif act == "br":
                nw = int(round(w + dx_i)); nh = int(round(h + dy_i))
                if nw >= 4 and nh >= 4 and (x + nw) <= W and (y + nh) <= H:
                    w, h = nw, nh
            sd.update({"x": x, "y": y, "w": w, "h": h})

        elif self._shape == "circle":
            cx = sd["cx"]; cy = sd["cy"]; r = sd["r"]
            if self._edit_action == "move":
                cx = max(r, min(int(round(cx + dx_i)), W - r))
                cy = max(r, min(int(round(cy + dy_i)), H - r))
            else:
                # Bất kỳ corner nào → resize bán kính theo khoảng cách tới tâm
                wx_c, wy_c = self._img_to_widget(cx, cy)
                d_w = ((pos.x() - wx_c) ** 2 + (pos.y() - wy_c) ** 2) ** 0.5
                r = max(4, int(round(d_w / self._scale)))
                r = min(r, cx, cy, W - cx, H - cy)
            x = max(0, cx - r); y = max(0, cy - r)
            sd.update({"cx": cx, "cy": cy, "r": r,
                        "x": x, "y": y, "w": 2 * r, "h": 2 * r})

        self._shape_data = sd
        self._render()

    def _render(self):
        if self._arr is None:
            self.setText("No Image\nRun pipeline first or load image source.")
            return
        import cv2
        arr = self._arr.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        h, w, ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        # Khi có scroll area: dùng viewport size làm tham chiếu fit, không phải
        # self.width()/height() (chính label có thể đã grow theo zoom).
        if self._scroll_area is not None:
            vp = self._scroll_area.viewport()
            pw = max(1, vp.width() - 4)
            ph = max(1, vp.height() - 4)
        else:
            pw = max(1, self.width() - 4)
            ph = max(1, self.height() - 4)
        sx = pw / w; sy = ph / h
        self._fit_scale = min(sx, sy)
        self._scale = self._fit_scale * self._user_zoom
        dw = int(w * self._scale); dh = int(h * self._scale)

        if self._scroll_area is not None and self._user_zoom > 1.0:
            # Zoom > 1 → label rộng hơn viewport → QScrollArea hiện scrollbars.
            need_w = max(self._base_min_size[0], dw + 8)
            need_h = max(self._base_min_size[1], dh + 8)
            if self.width() != need_w or self.height() != need_h:
                self.setMinimumSize(need_w, need_h)
                self.resize(need_w, need_h)
            self._off_x = (self.width() - dw) // 2
            self._off_y = (self.height() - dh) // 2
        else:
            # Fit-to-viewport: kéo label về kích thước viewport, center image.
            if self._scroll_area is not None:
                vp = self._scroll_area.viewport()
                if self.width() != vp.width() or self.height() != vp.height():
                    self.setMinimumSize(self._base_min_size[0],
                                        self._base_min_size[1])
                    self.resize(vp.width(), vp.height())
            self._off_x = (self.width() - dw) // 2 + self._pan_dx
            self._off_y = (self.height() - dh) // 2 + self._pan_dy

        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(5, 8, 16))
        p = QPainter(canvas)
        p.drawPixmap(self._off_x, self._off_y, dw, dh, pix)

        # Readonly rect (port connected) — màu xanh lá
        if self._readonly_rect:
            rx, ry, rw, rh = self._readonly_rect
            wx, wy = self._img_to_widget(rx, ry)
            ww = int(rw * self._scale); wh = int(rh * self._scale)
            col = QColor(57, 255, 20)  # bright green
            p.setPen(QPen(col, 2, Qt.SolidLine))
            p.setBrush(QBrush(QColor(57, 255, 20, 25)))
            p.drawRect(wx, wy, ww, wh)
            p.setPen(QPen(col))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(wx + 4, wy - 6, f"[PORT] ({rx},{ry}) {rw}×{rh}")

        # Interactive / drawn shape — màu cyan (roi) hoặc cam (template)
        if self.mode == "template":
            col = QColor(255, 140, 50)
        else:
            col = QColor(0, 212, 255)
        fill = QColor(col.red(), col.green(), col.blue(), 28)

        # Multi-mode: vẽ tất cả shapes đã commit (ngoài active đang edit) — dim hơn
        if self._multi and self._shapes:
            other_col = QColor(col.red(), col.green(), col.blue(), 200)
            other_fill = QColor(col.red(), col.green(), col.blue(), 14)
            p.setPen(QPen(other_col, 2, Qt.DotLine))
            p.setBrush(QBrush(other_fill))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            for i, entry in enumerate(self._shapes):
                if i == self._active_idx:
                    continue
                self._draw_one_shape(p, entry["type"], entry["data"], i + 1)

        # Shape đã hoàn tất (saved)
        sd = self._shape_data
        if sd:
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            if self._shape == "rect":
                wx, wy = self._img_to_widget(sd["x"], sd["y"])
                ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
                p.drawRect(wx, wy, ww, wh)
            elif self._shape == "ellipse":
                wx, wy = self._img_to_widget(sd["x"], sd["y"])
                ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
                p.drawEllipse(wx, wy, ww, wh)
            elif self._shape == "circle":
                wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
                wr = int(sd["r"] * self._scale)
                p.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
                p.setPen(QPen(col, 1))
                p.drawLine(wx - 5, wy, wx + 5, wy)
                p.drawLine(wx, wy - 5, wx, wy + 5)
            elif self._shape == "polygon" and sd.get("pts"):
                from PySide6.QtCore import QPointF
                from PySide6.QtGui import QPolygonF
                pts_w = [QPointF(*self._img_to_widget(px, py)) for px, py in sd["pts"]]
                p.drawPolygon(QPolygonF(pts_w))

            # Corner handles cho rect/ellipse/circle — chỉ hiện khi
            # KHÔNG đang vẽ rubber-band, để user biết có thể move/resize.
            if (self.mode in ("roi", "template")
                    and self._shape in ("rect", "ellipse", "circle")
                    and not self._dragging):
                bb = self._shape_widget_bbox()
                if bb is not None:
                    p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
                    for cx, cy in ((bb.left(),  bb.top()),
                                    (bb.right(), bb.top()),
                                    (bb.left(),  bb.bottom()),
                                    (bb.right(), bb.bottom())):
                        p.drawRect(cx - 4, cy - 4, 8, 8)

        # Đang drag rect/circle/ellipse
        draw_rect = self._rect
        if draw_rect and not draw_rect.isNull() and self._dragging:
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            if self._shape == "ellipse":
                p.drawEllipse(draw_rect)
            elif self._shape == "circle":
                p.drawEllipse(draw_rect)
            else:
                p.drawRect(draw_rect)
            # Corner handles + label
            p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
            for cx, cy in [(draw_rect.left(), draw_rect.top()),
                           (draw_rect.right(), draw_rect.top()),
                           (draw_rect.left(), draw_rect.bottom()),
                           (draw_rect.right(), draw_rect.bottom())]:
                p.drawRect(cx - 4, cy - 4, 8, 8)
            ix, iy = self._widget_to_img(draw_rect.left(), draw_rect.top())
            iw2 = int(draw_rect.width() / self._scale)
            ih2 = int(draw_rect.height() / self._scale)
            p.setPen(QPen(col)); p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(draw_rect.left() + 4, draw_rect.top() - 6,
                       f"({ix},{iy})  {iw2}×{ih2}")
        elif draw_rect and not draw_rect.isNull() and self._shape == "rect" and not sd:
            # ROI rect đã set qua set_rect_from_params (legacy)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            p.drawRect(draw_rect)

        # Polygon đang vẽ
        if self._shape == "polygon" and self._poly_drawing:
            p.setPen(QPen(col, 2, Qt.DashLine)); p.setBrush(Qt.NoBrush)
            pts_w = [self._img_to_widget(px, py) for px, py in self._poly_drawing]
            for i in range(len(pts_w) - 1):
                p.drawLine(pts_w[i][0], pts_w[i][1],
                           pts_w[i+1][0], pts_w[i+1][1])
            p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
            for px, py in pts_w:
                p.drawEllipse(px - 4, py - 4, 8, 8)
            p.setPen(QPen(QColor(255, 215, 0)))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            if pts_w:
                p.drawText(pts_w[0][0] + 8, pts_w[0][1] - 6,
                           f"Polygon: {len(pts_w)} pt — double-click để đóng")

        # Origin marker — hệ trục XY xoay được quanh tâm
        if self._show_origin and self._origin_xy is not None:
            eps = self._origin_axis_endpoints()
            if eps is not None:
                (cx, cy), x_end, y_end = eps
                rot = self._rot_handle_pos()
                # X axis — đỏ
                col_x = QColor(255, 70, 70)
                p.setPen(QPen(col_x, 2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(cx, cy, x_end[0], x_end[1])
                # Mũi tên X
                import math as _m
                a = _m.radians(self._origin_angle)
                ah = 6.0
                ax1 = (int(x_end[0] - ah * _m.cos(a - _m.radians(25))),
                        int(x_end[1] - ah * _m.sin(a - _m.radians(25))))
                ax2 = (int(x_end[0] - ah * _m.cos(a + _m.radians(25))),
                        int(x_end[1] - ah * _m.sin(a + _m.radians(25))))
                p.drawLine(x_end[0], x_end[1], ax1[0], ax1[1])
                p.drawLine(x_end[0], x_end[1], ax2[0], ax2[1])
                p.setPen(QPen(col_x))
                p.setFont(QFont("Segoe UI", 9, QFont.Bold))
                p.drawText(x_end[0] + 4, x_end[1] + 4, "X")
                # Y axis — xanh lá
                col_y = QColor(80, 220, 100)
                p.setPen(QPen(col_y, 2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(cx, cy, y_end[0], y_end[1])
                b = a + _m.radians(90)  # góc Y
                ay1 = (int(y_end[0] - ah * _m.cos(b - _m.radians(25))),
                        int(y_end[1] - ah * _m.sin(b - _m.radians(25))))
                ay2 = (int(y_end[0] - ah * _m.cos(b + _m.radians(25))),
                        int(y_end[1] - ah * _m.sin(b + _m.radians(25))))
                p.drawLine(y_end[0], y_end[1], ay1[0], ay1[1])
                p.drawLine(y_end[0], y_end[1], ay2[0], ay2[1])
                p.setPen(QPen(col_y))
                p.drawText(y_end[0] + 4, y_end[1] + 4, "Y")
                # Tâm
                p.setPen(QPen(QColor(0, 212, 255), 1))
                p.setBrush(QBrush(QColor(0, 212, 255)))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)
                # Rotate handle
                if rot is not None:
                    rh_col = QColor(255, 215, 0)
                    p.setPen(QPen(rh_col, 2))
                    p.setBrush(QBrush(QColor(255, 215, 0, 80)))
                    p.drawEllipse(rot[0] - 5, rot[1] - 5, 10, 10)
                    # Vòng cung gợi ý xoay
                    p.setPen(QPen(rh_col, 1, Qt.DotLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawArc(rot[0] - 8, rot[1] - 8, 16, 16, 0, 270 * 16)
                # Label toạ độ + góc
                p.setPen(QPen(QColor(0, 212, 255)))
                p.setFont(QFont("Courier New", 9, QFont.Bold))
                p.drawText(cx + 18, cy - 12,
                           f"O ({self._origin_xy[0]:.1f},{self._origin_xy[1]:.1f})  "
                           f"{self._origin_angle:+.1f}deg")

        # Pixel pick marker
        if self._pick_pos and self.mode == "pick":
            px2, py2 = self._img_to_widget(*self._pick_pos)
            p.setPen(QPen(QColor(255, 255, 0), 2))
            p.drawLine(px2 - 10, py2, px2 + 10, py2)
            p.drawLine(px2, py2 - 10, px2, py2 + 10)
            p.drawEllipse(px2 - 6, py2 - 6, 12, 12)

        p.end()
        self.setPixmap(canvas)

    # ── Mouse ──────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        if self.mode == "pick":
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            if self._arr is not None:
                h2, w2 = self._arr.shape[:2]
                ix = max(0, min(ix, w2-1)); iy = max(0, min(iy, h2-1))
                self._pick_pos = (ix, iy)
                self._render()
                self.pixel_picked.emit(ix, iy)
        elif self.mode in ("roi", "template"):
            # Ưu tiên: rotate handle → center handle → kéo bình thường
            if self._show_origin and self._hit_origin_rot(pos.x(), pos.y()):
                self._dragging_origin_rot = True
                self._update_origin_angle_from_widget(pos.x(), pos.y())
                return
            if self._show_origin and self._hit_origin_center(pos.x(), pos.y()):
                self._dragging_origin = True
                self._update_origin_from_widget(pos.x(), pos.y())
                return
            if event.button() == Qt.RightButton and self._shape == "polygon":
                # Right-click: huỷ polygon đang vẽ
                self.cancel_polygon()
                return
            if self._shape == "polygon":
                ix_f = (pos.x() - self._off_x) / self._scale if self._scale else 0.0
                iy_f = (pos.y() - self._off_y) / self._scale if self._scale else 0.0
                if self._arr is not None:
                    H2, W2 = self._arr.shape[:2]
                    ix_f = max(0.0, min(ix_f, W2 - 1.0))
                    iy_f = max(0.0, min(iy_f, H2 - 1.0))
                self._poly_drawing.append((ix_f, iy_f))
                self._render()
                return
            # rect / circle / ellipse — multi: scan list trước; single: shape_data
            self.setFocus()
            if self._shape in ("rect", "ellipse", "circle"):
                if self._multi:
                    hit = self._hit_test_shapes_list(pos.x(), pos.y())
                    if hit is not None:
                        idx, action = hit
                        # Commit active hiện tại trước khi đổi sang shape khác
                        self._commit_active_to_list()
                        entry = self._shapes[idx]
                        self._active_idx = idx
                        self._shape = entry["type"]
                        self._shape_data = dict(entry["data"])
                        self._edit_action = action  # "move"|"tl"|"tr"|"bl"|"br"
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(entry["data"])
                        self._render()
                        return
                # Single-mode (hoặc multi không hit) → check edit trên _shape_data
                if self._shape_data:
                    corner = self._hit_corner(pos.x(), pos.y())
                    if corner:
                        self._edit_action = corner
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(self._shape_data)
                        return
                    if self._hit_body(pos.x(), pos.y()):
                        self._edit_action = "move"
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(self._shape_data)
                        return
            # Click ngoài shape (hoặc chưa có shape) → vẽ mới
            if self._multi:
                # Commit active hiện tại; multi-mode KHÔNG xoá list — append mới
                self._commit_active_to_list()
                self._active_idx = None
            self._drag_start = pos
            self._dragging   = True
            self._rect = QRect(pos, QSize(0, 0))
            self._shape_data = {}    # xoá shape cũ (single mode); multi → tạo entry mới khi release

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.mode in ("roi", "template") and self._shape == "polygon" \
                and len(self._poly_drawing) >= 3:
            pts = list(self._poly_drawing)
            self._poly_drawing = []
            xs = [px for px, _ in pts]; ys = [py for _, py in pts]
            x = int(min(xs)); y = int(min(ys))
            w = max(1, int(max(xs) - min(xs)))
            h = max(1, int(max(ys) - min(ys)))
            if self._arr is not None:
                H2, W2 = self._arr.shape[:2]
                x = max(0, min(x, W2 - 1)); y = max(0, min(y, H2 - 1))
                w = max(1, min(w, W2 - x)); h = max(1, min(h, H2 - y))
            self._shape_data = {"pts": pts, "x": x, "y": y, "w": w, "h": h}
            if self._multi:
                self._shapes.append({"type": "polygon",
                                      "data": dict(self._shape_data)})
                self._active_idx = len(self._shapes) - 1
            self.shape_drawn.emit("polygon", dict(self._shape_data))
            self.roi_changed.emit(x, y, w, h)
            self._emit_shapes_changed()
            self._render()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        if self._dragging_origin_rot:
            self._update_origin_angle_from_widget(pos.x(), pos.y())
            return
        if self._dragging_origin:
            self._update_origin_from_widget(pos.x(), pos.y())
            return
        if self._edit_action:
            self._apply_edit(pos)
            return
        if not self._dragging or self._drag_start is None:
            return
        if self._shape == "circle":
            cx = self._drag_start.x(); cy = self._drag_start.y()
            dx = pos.x() - cx; dy = pos.y() - cy
            r = int(max(1, (dx * dx + dy * dy) ** 0.5))
            self._rect = QRect(cx - r, cy - r, 2 * r, 2 * r)
        else:
            self._rect = QRect(self._drag_start, pos).normalized()
        self._render()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_origin_rot:
            self._dragging_origin_rot = False
            return
        if self._dragging_origin:
            self._dragging_origin = False
            return
        if self._edit_action:
            # Kết thúc move/resize — phát signal cho dialog cha cập nhật ROI
            self._edit_action = None
            self._edit_anchor_w = None
            self._edit_orig_data = {}
            sd = self._shape_data
            if sd and self._shape in ("rect", "ellipse", "circle") \
                    and "x" in sd and "y" in sd:
                if self._multi:
                    self._commit_active_to_list()
                self.shape_drawn.emit(self._shape, dict(sd))
                self.roi_changed.emit(int(sd["x"]), int(sd["y"]),
                                       int(sd["w"]), int(sd["h"]))
                if self.mode == "template":
                    self.template_drawn.emit(int(sd["x"]), int(sd["y"]),
                                              int(sd["w"]), int(sd["h"]))
                self._emit_shapes_changed()
            self._render()
            return
        if not self._dragging:
            return
        self._dragging = False
        if not (self._rect and self._rect.width() > 4 and self._rect.height() > 4):
            return
        ix, iy = self._widget_to_img(self._rect.left(), self._rect.top())
        iw2 = max(1, int(self._rect.width() / self._scale))
        ih2 = max(1, int(self._rect.height() / self._scale))
        if self._arr is not None:
            H2, W2 = self._arr.shape[:2]
            ix  = max(0, min(ix, W2-1)); iy  = max(0, min(iy, H2-1))
            iw2 = max(1, min(iw2, W2-ix)); ih2 = max(1, min(ih2, H2-iy))

        if self._shape == "rect":
            self._shape_data = {"x": ix, "y": iy, "w": iw2, "h": ih2}
        elif self._shape == "ellipse":
            self._shape_data = {"x": ix, "y": iy, "w": iw2, "h": ih2}
        elif self._shape == "circle":
            cx_w = self._drag_start.x() if self._drag_start else 0
            cy_w = self._drag_start.y() if self._drag_start else 0
            cx_i, cy_i = self._widget_to_img(cx_w, cy_w)
            r_i = max(1, int(self._rect.width() / 2 / self._scale))
            ix = max(0, cx_i - r_i); iy = max(0, cy_i - r_i)
            iw2 = 2 * r_i; ih2 = 2 * r_i
            if self._arr is not None:
                H2, W2 = self._arr.shape[:2]
                iw2 = min(iw2, W2 - ix); ih2 = min(ih2, H2 - iy)
            self._shape_data = {"cx": cx_i, "cy": cy_i, "r": r_i,
                                 "x": ix, "y": iy, "w": iw2, "h": ih2}

        if self._multi and self._shape_data:
            # Append shape mới vào list, set active = last
            self._shapes.append({"type": self._shape,
                                  "data": dict(self._shape_data)})
            self._active_idx = len(self._shapes) - 1

        if self._shape in ("rect", "ellipse", "circle"):
            self.shape_drawn.emit(self._shape, dict(self._shape_data))
        self.roi_changed.emit(ix, iy, iw2, ih2)
        if self.mode == "template":
            self.template_drawn.emit(ix, iy, iw2, ih2)
        self._emit_shapes_changed()
        self._render()

    def _update_origin_from_widget(self, wx: int, wy: int):
        """Convert widget pos → image coords, clamp ảnh, emit signal.
        Cho phép kéo origin ra ngoài ROI rect (chỉ clamp vào ảnh)."""
        ix_f = (wx - self._off_x) / self._scale if self._scale else 0.0
        iy_f = (wy - self._off_y) / self._scale if self._scale else 0.0
        if self._arr is not None:
            H2, W2 = self._arr.shape[:2]
            ix_f = max(0.0, min(ix_f, W2 - 1.0))
            iy_f = max(0.0, min(iy_f, H2 - 1.0))
        self._origin_xy = (ix_f, iy_f)
        self._show_origin = True
        self._render()
        self.origin_changed.emit(ix_f, iy_f)

    def _update_origin_angle_from_widget(self, wx: int, wy: int):
        """Tính góc xoay axes theo vị trí con trỏ so với tâm origin."""
        c = self._origin_widget_pos()
        if c is None:
            return
        import math as _m
        dx = wx - c[0]; dy = wy - c[1]
        if dx == 0 and dy == 0:
            return
        angle = _m.degrees(_m.atan2(dy, dx))
        self._origin_angle = angle % 360.0
        self._render()
        self.origin_angle_changed.emit(self._origin_angle)

    def keyPressEvent(self, event):
        if self._multi and event.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                and self._active_idx is not None:
            self.delete_active_shape()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ════════════════════════════════════════════════════════════════════
#  Main Dialog
# ════════════════════════════════════════════════════════════════════
class NodeDetailDialog(QDialog):
    run_requested = Signal(str)

    def __init__(self, node: NodeInstance, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._node  = node
        self._graph = graph
        tool: ToolDef = node.tool

        self.setWindowTitle(f"{tool.icon}  {tool.name}  —  Detail")
        self.setMinimumSize(1000, 650)
        self.resize(1120, 740)
        self.setModal(False)
        self.setStyleSheet("""
            QDialog { background:#0a0e1a; color:#e2e8f0; }
            QGroupBox { border:1px solid #1e2d45; border-radius:6px;
                        margin-top:8px; padding-top:8px;
                        color:#64748b; font-size:11px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QScrollArea { border:none; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(54)
        hdr.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {tool.color},stop:1 #0a0e1a);"
            f"border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16, 0, 16, 0)

        icon_l = QLabel(tool.icon)
        icon_l.setStyleSheet("font-size:28px; background:transparent;")
        hl.addWidget(icon_l)

        tc = QVBoxLayout()
        t1 = QLabel(tool.name)
        t1.setStyleSheet("color:#fff; font-size:16px; font-weight:700; background:transparent;")
        cog = f"  {tool.cognex_equiv}" if tool.cognex_equiv else ""
        t2 = QLabel(f"{tool.category}{cog}  •  {tool.description}")
        t2.setStyleSheet("color:#ffffff88; font-size:11px; background:transparent;")
        tc.addWidget(t1); tc.addWidget(t2)
        hl.addLayout(tc, 1)

        self._run_btn = QPushButton("▶  Run Node")
        self._run_btn.setFixedSize(120, 34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#00d4ff;border:none;border-radius:5px;"
            "color:#000;font-weight:700;font-size:13px;}"
            "QPushButton:hover{background:#33ddff;}"
            "QPushButton:pressed{background:#0099bb;}")
        self._run_btn.clicked.connect(self._on_run)
        hl.addWidget(self._run_btn)
        root.addWidget(hdr)

        # ── Mode hint bar ─────────────────────────────────────────
        self._mode_hint = QLabel("")
        self._mode_hint.setStyleSheet(
            "background:#0d1a2a; color:#ffd700; font-size:11px;"
            "padding:5px 16px; border-bottom:1px solid #1e2d45;")
        self._mode_hint.hide()
        root.addWidget(self._mode_hint)

        # ── Splitter ──────────────────────────────────────────────
        spl = QSplitter(Qt.Horizontal)
        spl.setHandleWidth(1)
        spl.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(spl, 1)

        # LEFT — params
        left = QWidget(); left.setMaximumWidth(300); left.setMinimumWidth(240)
        ll   = QVBoxLayout(left); ll.setContentsMargins(10, 10, 10, 10); ll.setSpacing(8)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#0a0e1a;color:#64748b;padding:6px 10px;
                         border:none;font-size:11px;font-weight:600;}
            QTabBar::tab:selected{color:#00d4ff;border-bottom:2px solid #00d4ff;}
        """)
        self._params_scroll = QScrollArea(); self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(self._build_params_widget())
        tabs.addTab(self._params_scroll, "⚙ Params")
        tabs.addTab(self._build_ports_widget(), "🔌 Ports")
        ll.addWidget(tabs)

        self._out_group = QGroupBox("Output Values")
        og = QVBoxLayout(self._out_group)
        og.setContentsMargins(8, 12, 8, 8); og.setSpacing(4)
        self._out_labels = {}
        self._build_output_labels(og)
        ll.addWidget(self._out_group)

        self._status_lbl = QLabel("Status: IDLE")
        self._status_lbl.setStyleSheet("color:#64748b; font-size:11px; padding:2px;")
        ll.addWidget(self._status_lbl)
        spl.addWidget(left)

        # RIGHT — image
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6, 6, 6, 6); rl.setSpacing(4)

        img_hdr = QHBoxLayout()
        img_lbl = QLabel("OUTPUT IMAGE")
        img_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        img_hdr.addWidget(img_lbl); img_hdr.addStretch()

        # Reset Image button — chỉ hiện cho Crop ROI: xoá drawn_roi + đưa
        # x/y/w/h về full ảnh nguồn để vẽ lại từ đầu.
        if tool.tool_id == "crop_roi":
            btn_reset = QPushButton("🔄  Reset Image")
            btn_reset.setFixedHeight(26)
            btn_reset.setStyleSheet(
                "QPushButton{background:#1e2d45;border:1px solid #3a4b6a;"
                "color:#94a3b8;font-size:11px;padding:0 12px;border-radius:4px;}"
                "QPushButton:hover{background:#2c3e60;color:#00d4ff;}")
            btn_reset.clicked.connect(self._on_reset_crop_image)
            img_hdr.addWidget(btn_reset)

        # ── Chọn mode interactive theo tool ──────────────────────
        self._roi_port_connected = False

        if tool.tool_id == "crop_roi":
            self._roi_port_connected = self._check_roi_ports_connected()
            if self._roi_port_connected:
                # Port đang kết nối → readonly, hiển thị rect từ port
                mode_str = "readonly"
                self._mode_hint.setText(
                    "🔗  Port x/y/w/h đang được kết nối — ROI tự động theo upstream tool.")
            else:
                # Không kết nối → vẽ thủ công
                mode_str = "roi"
                self._mode_hint.setText(
                    "✏  Kéo chuột trên ảnh để vẽ vùng ROI thủ công  "
                    "—  kết nối port x/y/w/h để tracking tự động.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode=mode_str)
            if mode_str == "roi":
                self._img_label.roi_changed.connect(self._on_roi_changed)
                # Init rect từ _drawn_roi hoặc params
                drawn = node.params.get("_drawn_roi")
                if drawn:
                    x2, y2, w2, h2 = drawn
                else:
                    x2 = node.params.get("x", 0); y2 = node.params.get("y", 0)
                    w2 = node.params.get("crop_w", 320); h2 = node.params.get("crop_h", 240)
                QTimer.singleShot(120, lambda: self._img_label.set_rect_from_params(x2, y2, w2, h2))

        elif tool.tool_id in ("patmax", "patmax_align", "patfind"):
            # PatMax/PatFind → mở PatMaxDialog chuyên dụng
            self._mode_hint.setText(
                "🎯  PatMax/PatFind — Cửa sổ Train & Search chuyên dụng đang mở...")
            self._mode_hint.setStyleSheet(
                "background:#0d1a2a; color:#00d4ff; font-size:11px;"
                "padding:5px 16px; border-bottom:1px solid #1e2d45;")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="view")
            # Mở PatMaxDialog sau khi dialog này xuất hiện
            QTimer.singleShot(150, self._open_patmax_dialog)

        elif tool.tool_id == "color_picker":
            self._mode_hint.setText(
                "🎨  Click chuột vào ảnh để lấy màu tại điểm đó.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="pick")
            self._img_label.pixel_picked.connect(self._on_pixel_picked)

        else:
            self._img_label = InteractiveImageLabel(mode="view")

        img_hdr.addWidget(QWidget())   # spacer placeholder
        rl.addLayout(img_hdr)
        rl.addWidget(self._img_label, 1)

        self._img_info = QLabel("")
        self._img_info.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New'; padding:2px;")
        self._img_info.setAlignment(Qt.AlignCenter)
        rl.addWidget(self._img_info)

        self._pixel_bar = QLabel("")
        self._pixel_bar.setStyleSheet(
            "color:#ffd700; font-size:11px; font-family:'Courier New';"
            "background:#111827; border-radius:4px; padding:3px 10px;")
        self._pixel_bar.hide()
        rl.addWidget(self._pixel_bar)

        spl.addWidget(right)
        spl.setSizes([270, 830])

        cb = QPushButton("Close")
        cb.setFixedHeight(30)
        cb.setStyleSheet(
            "QPushButton{background:#1e2d45;border:none;border-radius:4px;"
            "color:#94a3b8;font-size:12px;margin:4px 12px;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        cb.clicked.connect(self.close)
        root.addWidget(cb)

        self.refresh_outputs()

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════
    def _check_roi_ports_connected(self) -> bool:
        """Trả về True nếu ít nhất một port x/y/w/h được kết nối."""
        node = self._node
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id and conn.dst_port in ("x","y","w","h"):
                return True
        return False

    def _get_input_image(self) -> Optional[np.ndarray]:
        for conn in self._graph.connections:
            if conn.dst_id == self._node.node_id and conn.dst_port == "image":
                src = self._graph.nodes.get(conn.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    # ════════════════════════════════════════════════════════════════
    #  Build sub-widgets
    # ════════════════════════════════════════════════════════════════
    def _build_params_widget(self) -> QWidget:
        node = self._node; tool = node.tool
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(5)
        if not tool.params:
            lbl = QLabel("No parameters.")
            lbl.setStyleSheet("color:#1e2d45; font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter); lay.addWidget(lbl)
        else:
            self._param_rows = {}
            for param in tool.params:
                # Conditional visibility (visible_if)
                if getattr(param, "visible_if", None):
                    ok = True
                    for k, v in param.visible_if.items():
                        if node.params.get(k) != v:
                            ok = False; break
                    if not ok:
                        continue
                pr = ParamRow(param, node.params.get(param.name, param.default))
                if param.tooltip:
                    pr.setToolTip(param.tooltip)
                pr.value_changed.connect(
                    lambda name, val, nid=node.node_id: self._on_param(nid, name, val))
                lay.addWidget(pr)
                self._param_rows[param.name] = pr
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)
        note = QLabel("▶ Run Node để áp dụng thay đổi")
        note.setStyleSheet("color:#1e2d45; font-size:10px;")
        note.setAlignment(Qt.AlignCenter)
        lay.addWidget(note); lay.addStretch()
        return w

    def _build_ports_widget(self) -> QWidget:
        tool = self._node.tool; w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)

        def section(label, color):
            h = QLabel(label)
            h.setStyleSheet(f"color:{color}; font-size:10px; font-weight:700; "
                            f"letter-spacing:1.5px; margin-top:4px;")
            lay.addWidget(h)

        if tool.inputs:
            section("INPUTS", "#00d4ff")
            for p in tool.inputs:
                connected = any(c.dst_id == self._node.node_id and c.dst_port == p.name
                                for c in self._graph.connections)
                status = "🔗" if connected else ("○" if not p.required else "●")
                r = QLabel(f"  {status}  {p.name}  [{p.data_type}]"
                           f"{'  (opt)' if not p.required else ''}")
                col = "#39ff14" if connected else "#00b4d8"
                r.setStyleSheet(
                    f"color:{col}; font-size:11px; font-family:'Courier New';"
                    f"background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        if tool.outputs:
            section("OUTPUTS", "#ff8c42")
            for p in tool.outputs:
                r = QLabel(f"  ⬤  {p.name}  [{p.data_type}]")
                r.setStyleSheet(
                    "color:#ff8c42; font-size:11px; font-family:'Courier New';"
                    "background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        # Crop ROI special: show connection status
        if tool.tool_id == "crop_roi":
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#1e2d45;"); lay.addWidget(sep)
            ports_status = []
            for pname in ("x","y","w","h"):
                conn = any(c.dst_id == self._node.node_id and c.dst_port == pname
                           for c in self._graph.connections)
                ports_status.append(f"{pname}:{'🔗' if conn else '✏'}")
            note = QLabel("  ".join(ports_status))
            note.setStyleSheet(
                "color:#ffd700; font-size:11px; font-family:'Courier New';"
                "padding:4px 6px; background:#0d1a2a; border-radius:4px;")
            lay.addWidget(note)

        lay.addStretch()
        return w

    def _build_output_labels(self, layout):
        self._out_labels = {}
        for port in self._node.tool.outputs:
            if port.name == "image":
                continue
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
            k = QLabel(port.name)
            k.setStyleSheet("color:#64748b; font-size:11px; font-family:'Courier New';")
            k.setMinimumWidth(80)
            v = QLabel("—")
            v.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            rl.addWidget(k); rl.addWidget(v, 1)
            layout.addWidget(row)
            self._out_labels[port.name] = v

    # ════════════════════════════════════════════════════════════════
    #  Interactive callbacks
    # ════════════════════════════════════════════════════════════════
    def _on_roi_changed(self, x, y, w, h):
        """
        Kéo chuột vẽ ROI thủ công → lưu vào _drawn_roi.
        Sync luôn x/y/crop_w/crop_h vào params + spinbox để hiển thị
        đúng vùng đang được cắt.
        """
        node = self._node
        node.params["_drawn_roi"] = (x, y, w, h)
        node.params["x"]      = int(x)
        node.params["y"]      = int(y)
        node.params["crop_w"] = int(w)
        node.params["crop_h"] = int(h)

        # Cập nhật spinbox params để hiển thị (nhưng _drawn_roi là nguồn truth)
        for name, val in [("x", x), ("y", y), ("crop_w", w), ("crop_h", h)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)

    def _on_reset_crop_image(self):
        """Reset Crop ROI về full ảnh nguồn — xoá _drawn_roi + set
        x=y=0, w/h = kích thước input image."""
        node = self._node
        # Tìm input image: ưu tiên upstream output, fallback node.outputs
        src_img = None
        if self._graph:
            for c in self._graph.connections:
                if c.dst_id == node.node_id and c.dst_port == "image":
                    s = self._graph.nodes.get(c.src_id)
                    if s and "image" in s.outputs:
                        src_img = s.outputs["image"]; break
        if src_img is None:
            src_img = node.outputs.get("image")
        if src_img is None:
            QMessageBox.information(self, "Reset Image",
                "Chưa có ảnh nguồn — chạy pipeline trước rồi reset.")
            return
        h, w = src_img.shape[:2]
        node.params.pop("_drawn_roi", None)
        node.params["x"]      = 0
        node.params["y"]      = 0
        node.params["crop_w"] = int(w)
        node.params["crop_h"] = int(h)
        node.params["_crop_initialized"] = True
        # Sync spinbox
        for name, val in [("x", 0), ("y", 0), ("crop_w", w), ("crop_h", h)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        # Clear vẽ trên ảnh
        if hasattr(self, "_img_label"):
            self._img_label.set_rect_from_params(0, 0, w, h)
        self._img_info.setText(f"🔄 Reset → full image ({w}x{h})")

        # Cập nhật info label
        self._img_info.setText(
            f"Manual ROI: ({x},{y})  {w}×{h} px  —  ▶ Run Node để crop")
        self._img_info.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _open_patmax_dialog(self):
        """Mở cửa sổ PatMax chuyên dụng."""
        from ui.patmax_dialog import PatMaxDialog
        # Nếu đã mở rồi, bring to front
        existing = getattr(self, '_patmax_dlg', None)
        if existing and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        dlg = PatMaxDialog(self._node, self._graph, self)
        dlg.run_requested.connect(self.run_requested)
        dlg.model_trained.connect(self._on_patmax_model_trained)
        self._patmax_dlg = dlg
        dlg.show()
        # Update mode hint
        self._mode_hint.setText("🎯  PatMax dialog mở — Train & Search tại đó.")

    def _on_patmax_model_trained(self):
        """Callback khi PatMax train xong — refresh image preview."""
        self.refresh_outputs()

    def _on_template_drawn(self, x, y, w, h):
        """Vẽ ROI → cắt template → lưu vào params."""
        node = self._node
        img = node.outputs.get("image") or self._get_input_image()
        if img is None:
            QMessageBox.warning(self, "Template",
                                "Cần ảnh để cắt template.\n"
                                "Kết nối node Image Source và Run Node trước.")
            return
        H2, W2 = img.shape[:2]
        x = max(0, min(x, W2-1)); y = max(0, min(y, H2-1))
        w = max(1, min(w, W2-x)); h = max(1, min(h, H2-y))
        templ = img[y:y+h, x:x+w]
        node.params["_template_array"] = templ
        node.params["_template_rect"]  = (x, y, w, h)
        self._img_info.setText(
            f"✔ Template saved: ({x},{y})  {w}×{h} px  —  ▶ Run Node")
        self._img_info.setStyleSheet(
            "color:#ffd700; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _on_pixel_picked(self, x, y):
        """Click → lấy màu pixel."""
        node = self._node
        node.params["pick_x"] = x; node.params["pick_y"] = y
        for name, val in [("pick_x", x), ("pick_y", y)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        self._pixel_bar.show()
        self._on_run()

    def _on_param(self, node_id, name, value):
        if not (self._graph and node_id in self._graph.nodes):
            return
        node = self._graph.nodes[node_id]
        node.params[name] = value
        # Nếu có param khác phụ thuộc tên này → rebuild Params tab để cập nhật.
        # Defer qua event loop: rebuild ngay trong slot sẽ xoá chính widget
        # đang phát signal (vd QComboBox source_mode toggle Folder ↔ File
        # nhiều lần) → next toggle hit C++ deleted object → crash app.
        if any(getattr(p, "visible_if", None) and name in p.visible_if
                for p in node.tool.params):
            QTimer.singleShot(0, self._rebuild_params_tab)

    def _rebuild_params_tab(self):
        """Rebuild Params tab — dùng khi visible_if của param khác đổi.
        An toàn khi dialog đã đóng / scroll widget đã bị Qt xoá."""
        try:
            scroll = getattr(self, "_params_scroll", None)
            if scroll is None:
                return
            scroll.setWidget(self._build_params_widget())
        except RuntimeError:
            # Underlying C++ widget đã bị xoá — ignore.
            pass

    # ════════════════════════════════════════════════════════════════
    #  Run
    # ════════════════════════════════════════════════════════════════
    def _on_run(self):
        node = self._node
        # Build inputs: defaults + upstream outputs
        inputs = {p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id:
                src = self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port] = src.outputs[conn.src_port]

        try:
            out = node.tool.process_fn(inputs, node.params)
            node.outputs  = out or {}
            node.status   = "pass"
            if "pass" in node.outputs:
                node.status = "pass" if node.outputs["pass"] else "fail"
            node.error_msg = ""
        except Exception as e:
            node.outputs  = {}
            node.status   = "error"
            node.error_msg = str(e)

        self.refresh_outputs()
        self.run_requested.emit(node.node_id)

    # ════════════════════════════════════════════════════════════════
    #  Refresh
    # ════════════════════════════════════════════════════════════════
    def refresh_outputs(self):
        node = self._node
        sc = {"pass":"#39ff14","fail":"#ff3860","error":"#ff3860",
              "idle":"#64748b","running":"#ffd700"}.get(node.status, "#64748b")
        self._status_lbl.setText(f"Status: {node.status.upper()}")
        self._status_lbl.setStyleSheet(
            f"color:{sc}; font-size:12px; font-weight:700; padding:2px;")

        if node.error_msg:
            self._img_info.setText(f"Error: {node.error_msg}")
            self._img_info.setStyleSheet(
                "color:#ff3860; font-size:10px; font-family:'Courier New'; padding:2px;")

        # Scalar outputs
        for name, lbl in self._out_labels.items():
            val = node.outputs.get(name)
            if val is None:
                lbl.setText("—"); lbl.setStyleSheet("color:#1e2d45; font-size:11px;")
            elif isinstance(val, bool):
                lbl.setText("✔ TRUE" if val else "✖ FALSE")
                lbl.setStyleSheet(
                    f"color:{'#39ff14' if val else '#ff3860'}; font-size:11px; font-weight:700;")
            elif isinstance(val, float):
                lbl.setText(f"{val:.5f}")
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            elif isinstance(val, int):
                lbl.setText(str(val))
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            else:
                lbl.setText(str(val)[:50])
                lbl.setStyleSheet("color:#e2e8f0; font-size:11px;")

        # Color picker
        if node.tool.tool_id == "color_picker" and node.outputs:
            r2 = node.outputs.get("r", 0); g2 = node.outputs.get("g", 0)
            b2 = node.outputs.get("b", 0); H2 = node.outputs.get("h", 0)
            S2 = node.outputs.get("s", 0); V2 = node.outputs.get("v", 0)
            self._pixel_bar.setText(
                f"  ({node.params.get('pick_x',0)}, {node.params.get('pick_y',0)})  "
                f"  RGB {r2},{g2},{b2}  #{r2:02X}{g2:02X}{b2:02X}  "
                f"  HSV {H2},{S2},{V2}")
            self._pixel_bar.setStyleSheet(
                f"color:rgb({r2},{g2},{b2}); font-size:11px; font-family:'Courier New';"
                f"background:#111827; border-radius:4px; padding:3px 10px;"
                f"border:1px solid rgb({r2},{g2},{b2});")
            self._pixel_bar.show()

        # Image
        img = node.outputs.get("image")
        if img is not None and isinstance(img, np.ndarray):
            h2, w2 = img.shape[:2]
            if not node.error_msg:
                self._img_info.setText(
                    f"{w2}×{h2} px  |  {img.dtype}  |  {node.status.upper()}")
                self._img_info.setStyleSheet(
                    f"color:{sc}; font-size:10px; font-family:'Courier New'; padding:2px;")
            self._img_label.set_image(img)

            # crop_roi: hiển thị rect
            if node.tool.tool_id == "crop_roi":
                if self._roi_port_connected:
                    # Lấy giá trị thực từ output (sau khi proc_crop chạy)
                    ox = node.outputs.get("x", 0); oy = node.outputs.get("y", 0)
                    ow = node.outputs.get("w", 0); oh = node.outputs.get("h", 0)
                    self._img_label.set_readonly_rect(ox, oy, ow, oh)
                else:
                    drawn = node.params.get("_drawn_roi")
                    if drawn:
                        self._img_label.set_rect_from_params(*drawn)

            # patmax/patfind: hiển thị output image từ engine
            elif node.tool.tool_id in ("patmax", "patmax_align", "patfind"):
                pass   # PatMaxDialog tự quản lý display

        elif not node.error_msg:
            self._img_label.set_image(None)
