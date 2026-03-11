"""step_node_widget.py — Circular node widget showing a step's run status."""

from __future__ import annotations

from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Status → (fill colour, border colour)
_COLORS: dict[str, tuple[str, str]] = {
    "not_run": ("#3c3c3c", "#5a5a5a"),
    "running": ("#b87a00", "#e8a800"),
    "success": ("#1e6e2e", "#28a745"),
    "failed": ("#6e1e1e", "#dc3545"),
    "stub": ("#2a2a2a", "#404040"),
}

_DIAMETER = 52


class StepNodeWidget(QWidget):
    """Single circular node.

    Parameters
    ----------
    step_id:
        The step identifier used in signals.
    label:
        Short display label (≤ ~6 chars looks best at this size).
    stub:
        When True the node is displayed faded and is not interactive.
    """

    clicked: Signal = Signal(str)  # step_id

    def __init__(
        self,
        step_id: str,
        label: str,
        stub: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.step_id = step_id
        self.label = label
        self.stub = stub

        self._status = "stub" if stub else "not_run"
        self._metadata: str | None = None  # optional text override (e.g., "5 epochs")
        self._pulse = False  # amber blink state
        self._pulse_on = False

        self._timer = QTimer(self)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._toggle_pulse)

        self.setFixedSize(QSize(_DIAMETER, _DIAMETER))
        self.setToolTip(step_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        """Update visual state.  status must be one of the _COLORS keys."""
        if self.stub:
            return
        self._status = status
        if status == "running":
            self._pulse_on = True
            self._pulse = True
            self._timer.start()
        else:
            self._timer.stop()
            self._pulse = False
        self.update()

    def set_metadata(self, text: str | None) -> None:
        """Set optional text override (e.g., 'N epochs' for train step)."""
        self._metadata = text
        self.update()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self.update()

    def _fill_color(self) -> QColor:
        fill, _ = _COLORS.get(self._status, _COLORS["not_run"])
        if self._pulse and self._status == "running":
            fill = "#e8a800" if self._pulse_on else "#b87a00"
        return QColor(fill)

    def _border_color(self) -> QColor:
        _, border = _COLORS.get(self._status, _COLORS["not_run"])
        return QColor(border)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 3
        d = _DIAMETER - margin * 2

        # Fill
        painter.setBrush(self._fill_color())
        pen = QPen(self._border_color(), 2)
        painter.setPen(pen)
        painter.drawEllipse(margin, margin, d, d)

        # Status icon / label
        painter.setPen(QColor("#e8e8e8") if not self.stub else QColor("#505050"))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)

        icon = self._icon()
        painter.drawText(self.rect(), 0x84, icon)  # AlignHCenter | AlignVCenter

    def _icon(self) -> str:
        # If metadata is set (e.g., epoch count for Train), show it instead of status icon
        if self._metadata is not None:
            return self._metadata
        if self.stub:
            return "-"
        icons = {
            "not_run": "-",
            "running": "⟳",
            "success": "✓",
            "failed": "✗",
        }
        return icons.get(self._status, self.label)

    def sizeHint(self) -> QSize:
        return QSize(_DIAMETER, _DIAMETER)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.stub:
            self.clicked.emit(self.step_id)
        super().mousePressEvent(event)
