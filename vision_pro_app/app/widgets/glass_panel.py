"""A frosted-glass styled QFrame with a subtle drop shadow."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect


class GlassPanel(QFrame):
    """Translucent panel with rounded corners and a soft glow shadow.

    The QSS rules in :mod:`app.styles` style ``QFrame#GlassPanel`` and react to
    the ``variant`` dynamic property (``"strong"`` for slightly more opaque
    surfaces).
    """

    def __init__(self, parent=None, *, variant: str = "default", glow: bool = True):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.setProperty("variant", variant)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if glow:
            self._install_glow()

    def _install_glow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 14)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)
