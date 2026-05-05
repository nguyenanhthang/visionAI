"""
ui/node_item.py — Cognex VisionPro style
Hiển thị Cognex tool name, tooltip params, port colors.
"""
from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING

from PySide6.QtWidgets import QGraphicsItem, QGraphicsEllipseItem, QMenu, QApplication
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                            QLinearGradient, QPainterPath, QCursor)

from core.flow_graph import NodeInstance
from core.tool_registry import ToolDef

PORT_R        = 7
PORT_D        = PORT_R * 2
NODE_MIN_W    = 190
NODE_HEADER_H = 42
NODE_PORT_ROW = 22
NODE_PADDING  = 8

C_BG       = QColor(13, 18, 30)
C_BORDER   = QColor(30, 45, 69)
C_SEL      = QColor(0, 212, 255)
C_PASS     = QColor(57, 255, 20)
C_FAIL     = QColor(255, 56, 96)
C_WARN     = QColor(255, 215, 0)
C_DIM      = QColor(100, 116, 139)
C_PORT_IN  = QColor(0, 180, 220)
C_PORT_OUT = QColor(255, 140, 50)


class PortItem(QGraphicsEllipseItem):
    """Port hitbox — scene xử lý drag connection."""
    def __init__(self, node_item: "NodeItem", port_name: str,
                 is_output: bool, index: int, parent=None):
        super().__init__(-PORT_R, -PORT_R, PORT_D, PORT_D, parent)
        self.node_item  = node_item
        self.port_name  = port_name
        self.is_output  = is_output
        self.port_index = index
        self._hovered   = False

        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setZValue(20)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self._update_brush()

    def _update_brush(self):
        base = C_PORT_OUT if self.is_output else C_PORT_IN
        if self._hovered:
            self.setBrush(QBrush(base))
            self.setPen(QPen(Qt.white, 2))
        else:
            self.setBrush(QBrush(base.darker(200)))
            self.setPen(QPen(base, 1.5))

    def hoverEnterEvent(self, event):
        self._hovered = True
        self._update_brush()
        self.setScale(1.4)
        # Show port name tooltip
        self.setToolTip(f"{'OUT' if self.is_output else 'IN'}: {self.port_name}")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._update_brush()
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeSignals(QObject):
    selected   = Signal(str)
    moved      = Signal(str, float, float)
    delete_req = Signal(str)
    open_props = Signal(str)


class NodeItem(QGraphicsItem):
    def __init__(self, node: NodeInstance, signals: NodeSignals):
        super().__init__()
        self.node    = node
        self.signals = signals

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setPos(node.pos_x, node.pos_y)
        self.setZValue(10)

        tool: ToolDef = node.tool
        self._color       = QColor(tool.color)
        self._icon        = tool.icon
        self._name        = tool.name
        self._cognex_name = tool.cognex_equiv

        n_ports = max(len(tool.inputs), len(tool.outputs), 1)
        self._w = max(NODE_MIN_W,
                      len(tool.name) * 7 + 60)
        self._h = NODE_HEADER_H + NODE_PADDING + n_ports * NODE_PORT_ROW + NODE_PADDING

        self._in_ports:  List[PortItem] = []
        self._out_ports: List[PortItem] = []
        self._build_ports()

        # Tooltip
        tip = f"<b>{tool.name}</b>"
        if tool.cognex_equiv:
            tip += f"<br><span style='color:#00d4ff'>{tool.cognex_equiv}</span>"
        tip += f"<br>{tool.description}"
        self.setToolTip(tip)

    def _build_ports(self):
        tool = self.node.tool
        for i, port in enumerate(tool.inputs):
            p = PortItem(self, port.name, False, i, self)
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            p.setPos(0, y)
            self._in_ports.append(p)
        for i, port in enumerate(tool.outputs):
            p = PortItem(self, port.name, True, i, self)
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            p.setPos(self._w, y)
            self._out_ports.append(p)

    def boundingRect(self) -> QRectF:
        m = PORT_R + 4
        return QRectF(-m, -m, self._w + m * 2, self._h + m * 2)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        status = self.node.status

        if self.isSelected():
            border_col, border_w = C_SEL, 2.5
        elif status == "pass":
            border_col, border_w = C_PASS, 2.0
        elif status == "fail":
            border_col, border_w = C_FAIL, 2.0
        elif status == "running":
            border_col, border_w = C_WARN, 2.0
        elif status == "error":
            border_col, border_w = C_FAIL, 2.0
        else:
            border_col, border_w = C_BORDER, 1.5

        # Shadow
        shadow = QPainterPath()
        shadow.addRoundedRect(3, 3, self._w, self._h, 8, 8)
        painter.fillPath(shadow, QBrush(QColor(0, 0, 0, 80)))

        # Body
        body = QPainterPath()
        body.addRoundedRect(0, 0, self._w, self._h, 8, 8)
        painter.fillPath(body, QBrush(C_BG))
        painter.setPen(QPen(border_col, border_w))
        painter.drawPath(body)

        # Header gradient
        hdr = QPainterPath()
        hdr.addRoundedRect(0, 0, self._w, NODE_HEADER_H, 8, 8)
        cut = QPainterPath()
        cut.addRect(0, NODE_HEADER_H // 2, self._w, NODE_HEADER_H)
        hdr = hdr.united(cut)
        grad = QLinearGradient(0, 0, self._w, NODE_HEADER_H)
        grad.setColorAt(0, self._color.lighter(140))
        grad.setColorAt(1, self._color.darker(110))
        painter.fillPath(hdr, QBrush(grad))

        # Icon
        painter.setFont(QFont("Segoe UI Emoji", 14))
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(6, 0, 30, NODE_HEADER_H), Qt.AlignCenter, self._icon)

        # Tool name
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(36, 2, self._w - 42, NODE_HEADER_H // 2 + 2),
                         Qt.AlignVCenter | Qt.AlignLeft, self._name)

        # Cognex equiv name (small, cyan)
        if self._cognex_name:
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(0, 212, 255, 180)))
            painter.drawText(QRectF(36, NODE_HEADER_H // 2, self._w - 42, NODE_HEADER_H // 2),
                             Qt.AlignVCenter | Qt.AlignLeft, self._cognex_name)

        # Status badge
        if status in ("pass", "fail", "error", "running"):
            colors = {"pass": C_PASS, "fail": C_FAIL, "error": C_FAIL, "running": C_WARN}
            texts  = {"pass": "✔ PASS", "fail": "✖ FAIL", "error": "ERR", "running": "…"}
            badge_col = colors[status]
            badge_txt = texts[status]
            painter.setPen(QPen(badge_col))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(QRectF(0, self._h - 18, self._w - 6, 14),
                             Qt.AlignRight | Qt.AlignVCenter, badge_txt)

        # Port labels
        tool = self.node.tool
        painter.setFont(QFont("Segoe UI", 7))
        for i, port in enumerate(tool.inputs):
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            painter.setPen(QPen(C_PORT_IN.lighter(120)))
            painter.drawText(QRectF(10, y - 8, self._w // 2 - 14, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, port.name)

        for i, port in enumerate(tool.outputs):
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            painter.setPen(QPen(C_PORT_OUT.lighter(120)))
            painter.drawText(QRectF(self._w // 2, y - 8, self._w // 2 - 12, 16),
                             Qt.AlignRight | Qt.AlignVCenter, port.name)

        # Output value previews
        if self.node.outputs:
            painter.setFont(QFont("Courier New", 7))
            painter.setPen(QPen(C_DIM))
            y_off = NODE_HEADER_H + NODE_PADDING + 2
            for key, val in list(self.node.outputs.items())[:3]:
                if isinstance(val, bool):
                    txt = f"{key}:{'✔' if val else '✖'}"
                elif isinstance(val, float):
                    txt = f"{key}:{val:.3g}"
                elif isinstance(val, int):
                    txt = f"{key}:{val}"
                elif isinstance(val, str) and len(val) < 20:
                    txt = f"{key}:{val[:12]}"
                else:
                    continue
                painter.drawText(QRectF(8, y_off + 2, self._w - 16, 12),
                                 Qt.AlignLeft | Qt.AlignVCenter, txt)
                y_off += 12
                if y_off > self._h - 20:
                    break

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.pos_x = self.pos().x()
            self.node.pos_y = self.pos().y()
            self.signals.moved.emit(self.node.node_id, self.pos().x(), self.pos().y())
            if self.scene():
                self.scene().update()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.signals.selected.emit(self.node.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.signals.open_props.emit(self.node.node_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(
            "QMenu{background:#0d1220;color:#e2e8f0;border:1px solid #1e2d45;font-size:12px;}"
            "QMenu::item:selected{background:#1a2236;color:#00d4ff;}"
            "QMenu::separator{height:1px;background:#1e2d45;}")
        act_props  = menu.addAction(f"{self.node.tool.icon}  Properties / Detail")
        act_run    = menu.addAction("▶  Run this node")
        menu.addSeparator()
        act_viewer = menu.addAction("👁  View output in Image Viewer")
        menu.addSeparator()
        act_del    = menu.addAction("🗑  Delete")

        chosen = menu.exec(event.screenPos())
        if chosen == act_props:
            self.signals.open_props.emit(self.node.node_id)
        elif chosen == act_del:
            self.signals.delete_req.emit(self.node.node_id)
        elif chosen == act_run:
            if self.scene() and hasattr(self.scene(), "run_single_node"):
                self.scene().run_single_node(self.node.node_id)
        elif chosen == act_viewer:
            if self.scene() and hasattr(self.scene(), "view_in_viewer"):
                self.scene().view_in_viewer(self.node.node_id)

    def get_port_scene_pos(self, port_name: str, is_output: bool) -> Optional[QPointF]:
        ports = self._out_ports if is_output else self._in_ports
        for p in ports:
            if p.port_name == port_name:
                return p.scene_center()
        return None

    def node_width(self) -> float:
        return self._w

    def node_height(self) -> float:
        return self._h
