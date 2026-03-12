"""profile_row.py — One row in the dashboard table (profile + step nodes + Details button)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from gui.run_registry import RunRegistry
from gui.step_definitions import ACTIVE_STEPS, PIPELINE_STEPS, StepDef
from gui.step_node_widget import StepNodeWidget

_NODE_W = 52
_NODE_H = 52
_COL_GAP = 28   # horizontal gap between columns
_ROW_GAP = 8    # vertical gap between rows
_COL_W = _NODE_W + _COL_GAP
_ROW_H = _NODE_H + _ROW_GAP
_V_PAD = 6      # vertical padding inside the nodes container


def _compute_dag_layout(steps: list[StepDef]) -> dict[str, tuple[int, int]]:
    """Compute (col, row) positions for each step from topological depth.

    Column = max depth from any root (longest prereq chain).
    Row = order within that column (preserving PIPELINE_STEPS ordering).
    """
    depth: dict[str, int] = {}
    for step in steps:  # steps is already in topological order
        if not step.prereqs:
            depth[step.id] = 0
        else:
            depth[step.id] = max(depth.get(p, 0) for p in step.prereqs) + 1

    col_count: dict[int, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    for step in steps:
        col = depth[step.id]
        row = col_count.get(col, 0)
        col_count[col] = row + 1
        positions[step.id] = (col, row)
    return positions


class ProfileRow(QWidget):
    """Horizontal strip showing pipeline DAG nodes for one profile.

    Signals
    -------
    details_clicked(str)
        Emitted when the Details button is pressed.  Carries profile_name.
    delete_clicked(str)
        Emitted when the Delete button is pressed.  Carries profile_name.
    node_clicked(str, str)
        Emitted when a runnable node is clicked.  Carries (profile_name, step_id).
    """

    details_clicked: Signal = Signal(str)
    delete_clicked: Signal = Signal(str)
    node_clicked: Signal = Signal(str, str)

    def __init__(
        self, profile_name: str, registry: RunRegistry, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.profile_name = profile_name
        self.registry = registry

        self._nodes: dict[str, StepNodeWidget] = {}
        self._runnable_steps: set[str] = set()
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

        # DAG nodes container
        all_steps = list(PIPELINE_STEPS)  # includes stubs
        self._nodes_container = _NodesWidget(self)
        self._nodes_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Create node widgets (not yet parented — _NodesWidget will own them)
        for step in all_steps:
            is_stub = not step.enabled
            node = StepNodeWidget(
                step.id, step.label,
                stub=is_stub,
                server_required=getattr(step, "server_required", False),
            )
            node.clicked.connect(self._on_node_clicked)
            self._nodes[step.id] = node

        # Hand nodes + steps to the DAG container (it positions them)
        self._nodes_container.layout_nodes(all_steps, self._nodes)
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

        layout.addSpacing(4)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.setStyleSheet(
            "QPushButton { background: #5a3a3a; color: #ff9999; border: 1px solid #8a5a5a;"
            " border-radius: 4px; padding: 2px 4px; font-weight: bold; }"
            "QPushButton:hover { background: #7a4a4a; }"
        )
        del_btn.setToolTip(f"Delete profile '{self.profile_name}'")
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.profile_name))
        layout.addWidget(del_btn)

    def _on_node_clicked(self, step_id: str) -> None:
        if step_id in self._runnable_steps:
            self.node_clicked.emit(self.profile_name, step_id)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read registry and update all node colours and metadata."""
        self.registry.reload()

        # Determine which steps are currently runnable (may be multiple)
        self._runnable_steps = set(self.registry.get_runnable_steps())

        for step_id, node in self._nodes.items():
            status = self.registry.get_status(step_id)
            node.set_status(status)
            node.set_runnable(step_id in self._runnable_steps)

            # Show epoch count for any train step
            if step_id.startswith("train_"):
                epochs = self.registry.get_metadata(step_id, "epochs_completed")
                node.set_metadata(f"{epochs}e" if epochs is not None else None)
            else:
                node.set_metadata(None)

        # Repaint edges to reflect new statuses
        self._nodes_container.update()


class _NodesWidget(QWidget):
    """DAG node-graph container.  Positions child nodes absolutely (no layout
    manager) and draws Bezier edges between them in ``paintEvent``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_positions: dict[str, tuple[int, int]] = {}  # step_id → (col, row)
        self._node_widgets: dict[str, StepNodeWidget] = {}
        self._active_ids: set[str] = set()

    def layout_nodes(
        self,
        steps: list[StepDef],
        nodes: dict[str, StepNodeWidget],
    ) -> None:
        """Compute DAG positions and place child node widgets."""
        self._node_widgets = nodes
        self._active_ids = {s.id for s in ACTIVE_STEPS}
        self._step_positions = _compute_dag_layout(steps)

        if not self._step_positions:
            return

        max_col = max(col for col, _ in self._step_positions.values())
        max_row = max(row for _, row in self._step_positions.values())

        total_w = (max_col + 1) * _COL_W - _COL_GAP
        total_h = (max_row + 1) * _ROW_H - _ROW_GAP + 2 * _V_PAD
        self.setFixedSize(total_w, total_h)

        # Position each node widget as an absolute child
        for step_id, (col, row) in self._step_positions.items():
            node = nodes.get(step_id)
            if node is None:
                continue
            x = col * _COL_W
            y = row * _ROW_H + _V_PAD
            node.setParent(self)
            node.move(x, y)
            node.show()

    def paintEvent(self, _event) -> None:  # noqa: N802
        """Draw background fill and Bezier edges between connected nodes."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        for step in PIPELINE_STEPS:
            if step.id not in self._step_positions:
                continue
            dst_col, dst_row = self._step_positions[step.id]
            dx = dst_col * _COL_W
            dy = dst_row * _ROW_H + _V_PAD + _NODE_H // 2

            for prereq_id in step.prereqs:
                if prereq_id not in self._step_positions:
                    continue
                src_col, src_row = self._step_positions[prereq_id]
                sx = src_col * _COL_W + _NODE_W
                sy = src_row * _ROW_H + _V_PAD + _NODE_H // 2

                # Determine edge colour from node statuses
                is_stub = (
                    step.id not in self._active_ids
                    or prereq_id not in self._active_ids
                )
                if is_stub:
                    pen = QPen(QColor("#404040"), 1.5, Qt.PenStyle.DashLine)
                else:
                    src_node = self._node_widgets.get(prereq_id)
                    dst_node = self._node_widgets.get(step.id)
                    ss = src_node._status if src_node else "not_run"
                    ds = dst_node._status if dst_node else "not_run"
                    if ss == "running" or ds == "running":
                        color = QColor("#e8a800")
                    elif ss == "success" and ds == "success":
                        color = QColor("#28a745")
                    elif ss == "success":
                        color = QColor("#5a8abf")
                    elif ss == "failed" or ds == "failed":
                        color = QColor("#dc3545")
                    else:
                        color = QColor("#404040")
                    pen = QPen(color, 1.5)

                painter.setPen(pen)

                # Cubic Bezier: control points offset horizontally
                offset = (dx - sx) * 0.45
                path = QPainterPath()
                path.moveTo(sx, sy)
                path.cubicTo(sx + offset, sy, dx - offset, dy, dx, dy)
                painter.drawPath(path)
