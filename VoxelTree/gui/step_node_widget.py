"""step_node_widget.py — Circular node widget showing a step's run status.

This widget now also supports a fractional ``progress`` value (0‑1) for the
"running" state; when set a coloured ring is drawn around the node and a
numeric percentage appears at the centre.  Progress updates are normally
pumped from the GUI via ``RunWorker`` signals and stored in the
``RunRegistry`` metadata.
"""

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
    "stale": ("#8a7a00", "#c0a000"),  # warning yellow for outdated success
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
    # Emitted when the user requests a context menu (right-click) on the node.
    # Carries (step_id, global_position: QPoint).
    context_menu_requested: Signal = Signal(str, object)

    def __init__(
        self,
        step_id: str,
        label: str,
        stub: bool = False,
        server_required: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.step_id = step_id
        self.label = label
        self.stub = stub
        self.server_required = server_required

        self._status = "stub" if stub else "not_run"
        self._metadata: str | None = None  # optional text override (e.g., "5 epochs")
        self._progress: float | None = None  # 0.0–1.0 running progress, None if unknown
        self._pulse = False  # amber blink state
        self._pulse_on = False
        self._runnable = False  # highlight as next runnable step

        self._timer = QTimer(self)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._toggle_pulse)

        self.setFixedSize(QSize(_DIAMETER, _DIAMETER))
        self.setToolTip(step_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        """Update visual state.  status must be one of the _COLORS keys.

        Transitioning away from ``running`` clears any progress value so that
        an old percentage cannot linger on a completed/failed node.
        """
        if self.stub:
            return
        self._status = status
        if status == "running":
            # visible pulse only if we have no concrete progress yet
            if self._progress is None:
                self._pulse_on = True
                self._pulse = True
                self._timer.start()
        else:
            self._timer.stop()
            self._pulse = False
            self._progress = None
        self.update()

    def set_metadata(self, text: str | None) -> None:
        """Set optional text override (e.g., 'N epochs' for train step)."""
        self._metadata = text
        self.update()

    def set_progress(self, fraction: float | None) -> None:
        """Indicate that the node is *running* and currently at the given
        fractional progress (0.0–1.0).  ``None`` removes the indicator.

        The widget will continue to show any ``_metadata`` text at its centre
        (epochs, etc.); the progress ring is drawn around the perimeter.  If
        a non-``None`` progress value is specified the amber pulse is
        suppressed.
        """
        # clamp the value for safety
        if fraction is not None:
            fraction = max(0.0, min(1.0, fraction))
        self._progress = fraction
        # when progress is set we don't blink; pause timer if running
        if self._progress is not None:
            self._timer.stop()
            self._pulse = False
        self.update()

    def set_runnable(self, runnable: bool) -> None:
        """Highlight this node as the next runnable step."""
        self._runnable = runnable
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
        if self._runnable:
            pen = QPen(QColor("#ffdd00"), 3)  # yellow highlight for next runnable
        else:
            pen = QPen(self._border_color(), 2)
        painter.setPen(pen)
        painter.drawEllipse(margin, margin, d, d)

        # Status icon / label
        painter.setPen(QColor("#e8e8e8") if not self.stub else QColor("#505050"))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)

        icon = self._icon()
        painter.drawText(self.rect(), 0x84, icon)  # AlignHCenter | AlignVCenter

        # Progress ring (drawn after icon so it appears on top of fill but
        # underneath text). Only show when we have a valid progress value and
        # the step is running.
        if self._progress is not None and self._status == "running":
            # determine arc colour; using a blue accent for visibility
            pen = QPen(QColor("#4a90e2"), 4)
            painter.setPen(pen)
            # arc rect inside border margin
            arc_margin = 4
            arc_d = d - arc_margin * 2
            # start at 12 o'clock (90 degrees) and proceed clockwise
            start_angle = 90 * 16
            span_angle = -int(self._progress * 360 * 16)
            painter.drawArc(
                margin + arc_margin,
                margin + arc_margin,
                arc_d,
                arc_d,
                start_angle,
                span_angle,
            )

        # Server-required indicator: small blue dot in top-right
        if self.server_required and not self.stub:
            dot_r = 5
            dot_x = _DIAMETER - margin - dot_r - 1
            dot_y = margin + dot_r + 1
            painter.setBrush(QColor("#4a9fd4"))
            painter.setPen(QPen(QColor("#1e1e1e"), 1))
            painter.drawEllipse(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)

    def _icon(self) -> str:
        # If metadata is set (e.g., epoch count for Train), show it first
        if self._metadata is not None:
            return self._metadata
        # Next, if we know a progress fraction show a percentage.
        # We display two significant figures: use one decimal place for
        # values <10, otherwise show an integer.
        if self._progress is not None:
            pct = self._progress * 100.0
            if pct < 10:
                return f"{pct:.1f}%"
            else:
                return f"{int(pct)}%"
        # Otherwise show the node's label directly.
        if self.stub:
            return "-"
        return self.label

    def sizeHint(self) -> QSize:
        return QSize(_DIAMETER, _DIAMETER)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Only treat left-clicks as normal clicks.  ``event.Button`` was a
        # typo; ``button()`` returns a Qt.MouseButton enum value.
        from PySide6.QtCore import Qt

        if not self.stub and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.step_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Show a context menu for the node.

        The widget itself merely notifies its parent that a menu was requested;
        the parent (``ProfileRow``) builds the actual ``QMenu`` so that it can
        make use of the profile name/registry data when enabling or disabling
        actions.
        """
        if self.stub:
            return
        # forward the step id and the global position to whoever cares
        self.context_menu_requested.emit(self.step_id, event.globalPos())
        # do not call super() because we have handled the event
