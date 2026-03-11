"""profile_row.py — One row in the dashboard table (profile + step nodes + Details button)."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from gui.run_registry import RunRegistry
from gui.step_definitions import ACTIVE_STEPS, STUB_STEPS
from gui.step_node_widget import StepNodeWidget

_NODE_DIAM = 52
_CONNECTOR_H = 2


class ProfileRow(QWidget):
    """Horizontal strip: [ Profile Name ]──( N )══( N )══…══( S )  [ Details ]

    Signals
    -------
    details_clicked(str)
        Emitted when the Details button is pressed.  Carries profile_name.
    node_clicked(str, str)
        Emitted when a node is clicked.  Carries (profile_name, step_id).
    """

    details_clicked: Signal = Signal(str)
    node_clicked: Signal = Signal(str, str)

    def __init__(
        self, profile_name: str, registry: RunRegistry, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.profile_name = profile_name
        self.registry = registry

        self._nodes: dict[str, StepNodeWidget] = {}
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)

        # Profile name label
        name_lbl = QLabel(self.profile_name)
        name_lbl.setFixedWidth(90)
        name_lbl.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 12px;")
        layout.addWidget(name_lbl)
        layout.addSpacing(8)

        # Step nodes with connecting lines drawn in paintEvent
        # We use a sub-widget so connector lines are contained
        self._nodes_container = _NodesWidget(self)
        self._nodes_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._nodes_container.setFixedHeight(_NODE_DIAM + 12)

        nodes_layout = QHBoxLayout(self._nodes_container)
        nodes_layout.setContentsMargins(0, 6, 0, 6)
        nodes_layout.setSpacing(0)  # connectors drawn manually

        all_steps = list(ACTIVE_STEPS) + list(STUB_STEPS)
        for i, step in enumerate(all_steps):
            is_stub = not step.enabled

            # Connector line before non-first nodes
            if i > 0:
                conn = _ConnectorWidget(stub=is_stub, parent=self._nodes_container)
                nodes_layout.addWidget(conn)

            node = StepNodeWidget(step.id, step.label, stub=is_stub, parent=self._nodes_container)
            node.clicked.connect(self._on_node_clicked)
            self._nodes[step.id] = node
            nodes_layout.addWidget(node)

        layout.addWidget(self._nodes_container, stretch=1)
        layout.addSpacing(8)

        # Details button
        btn = QPushButton("Details")
        btn.setFixedWidth(70)
        btn.setStyleSheet(
            "QPushButton { background: #2a4a6e; color: #cce0ff; border: 1px solid #4a7abf;"
            " border-radius: 4px; padding: 4px 8px; }"
            "QPushButton:hover { background: #3a5a8e; }"
        )
        btn.clicked.connect(lambda: self.details_clicked.emit(self.profile_name))
        layout.addWidget(btn)

    def _on_node_clicked(self, step_id: str) -> None:
        self.node_clicked.emit(self.profile_name, step_id)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read registry and update all node colours and metadata."""
        self.registry.reload()
        for step_id, node in self._nodes.items():
            status = self.registry.get_status(step_id)
            node.set_status(status)

            # For train step, show epoch count if available
            if step_id == "train":
                epochs = self.registry.get_metadata(step_id, "epochs_completed")
                if epochs is not None:
                    node.set_metadata(f"{epochs}e")
                else:
                    node.set_metadata(None)


class _ConnectorWidget(QWidget):
    """A small horizontal line connecting two nodes."""

    def __init__(self, stub: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stub = stub
        self.setFixedSize(QSize(28, _NODE_DIAM))

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#404040") if self._stub else QColor("#5a8abf")
        pen = QPen(color, _CONNECTOR_H)
        if self._stub:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        cy = self.height() // 2
        painter.drawLine(0, cy, self.width(), cy)


class _NodesWidget(QWidget):
    """Container that draws the row's background tint."""

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
