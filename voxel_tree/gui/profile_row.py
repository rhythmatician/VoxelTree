"""profile_row.py — One row in the dashboard table (profile + step nodes + Details button)."""

from __future__ import annotations

from graphlib import TopologicalSorter
from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QDrag, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QWidget

from voxel_tree.gui.run_registry import RunRegistry
from voxel_tree.gui.step_definitions import PIPELINE_STEPS, STEP_BY_ID, TRACK_BY_ID, TRACK_ORDER, StepDef
from voxel_tree.gui.step_node_widget import StepNodeWidget

_NODE_W = 52
_NODE_H = 52
_COL_GAP = 28  # horizontal gap between columns
_ROW_GAP = 8  # vertical gap between rows
_COL_W = _NODE_W + _COL_GAP
_ROW_H = _NODE_H + _ROW_GAP
_V_PAD = 6  # vertical padding inside the nodes container

_ASSET_DIR = Path(__file__).resolve().parent / "assets"
# MIME type used for drag-and-drop profile reordering
MIME_TYPE_PROFILE = "application/x-voxeltree-profile"


def _make_drag_handle_pixmap(size: int = 18, color: str = "#666666") -> QPixmap:
    """Render the drag-handle SVG in *color* at *size*×*size* pixels."""
    svg_path = _ASSET_DIR / "drag-handle-svgrepo-com.svg"
    svg_bytes = svg_path.read_bytes().decode("utf-8")
    # Replace the dark default fill with the requested colour
    svg_bytes = svg_bytes.replace('fill="#121923"', f'fill="{color}"')
    renderer = QSvgRenderer(QByteArray(svg_bytes.encode()))
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix


class DragHandle(QWidget):
    """Small drag-handle widget that initiates a QDrag for row reordering.

    The parent ProfileRow's *profile_name* is encoded into the MIME data so
    that the receiving container knows which row is being dragged.
    """

    def __init__(self, profile_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile_name = profile_name
        self._drag_start: QPoint | None = None

        self.setFixedSize(20, 40)
        self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
        self.setToolTip("Drag to reorder")

        # Pre-render both states
        self._pix_normal = _make_drag_handle_pixmap(size=16, color="#555555")
        self._pix_hover = _make_drag_handle_pixmap(size=16, color="#aaaaaa")
        self._hovered = False

    # ------------------------------------------------------------------
    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        pix = self._pix_hover if self._hovered else self._pix_normal
        painter = QPainter(self)
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        painter.drawPixmap(x, y, pix)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            return
        if (event.pos() - self._drag_start).manhattanLength() < 6:
            return

        # Build drag pixmap: a small translucent badge with the profile name
        badge = QPixmap(160, 28)
        badge.fill(QColor(0, 0, 0, 0))
        p = QPainter(badge)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#2a4a6e"))
        p.setPen(QColor("#5a9aff"))
        p.drawRoundedRect(0, 0, badge.width() - 1, badge.height() - 1, 5, 5)
        p.setPen(QColor("#cce0ff"))
        p.drawText(badge.rect(), Qt.AlignmentFlag.AlignCenter, self.profile_name)
        p.end()

        mime = QMimeData()
        mime.setData(MIME_TYPE_PROFILE, QByteArray(self.profile_name.encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(badge)
        drag.setHotSpot(QPoint(badge.width() // 2, badge.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_start = None

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start = None
        super().mouseReleaseEvent(event)


def _compute_dag_layout(
    steps: list[StepDef],
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Compute swim-lane-aware (col, row) positions for each step.

    Returns
    -------
    positions  : step_id → (col, row)
    band_info  : track_id → (start_row, band_height)

    Layout rules
    ------------
    * Column = topological depth (longest prereq chain from any root).
      Only edges between steps in the provided list are counted.
    * Row = determined by track membership.  Each track gets a contiguous
      horizontal band.  Tracks are ordered by TRACK_ORDER from step_definitions.
      Within a band, steps at the same column get stacked vertically.
    * data_acq (no track field) → first band; loopback → last band.
    """
    # Restrict graphs to steps visible in this DAG
    step_ids = {s.id for s in steps}
    graph = {s.id: {p for p in s.prereqs if p in step_ids} for s in steps}

    sorter = TopologicalSorter(graph)
    depth: dict[str, int] = {}
    for node in sorter.static_order():
        preds = graph[node]
        depth[node] = (max(depth[p] for p in preds) + 1) if preds else 0

    def _eff_track(s: StepDef) -> str:
        # Give each Voxy level its own lane: train/export/deploy for L4..L0.
        # Step ids are generated as "{phase}_voxy_l{N}".
        if s.track == "voxy":
            sid = s.id
            marker = "_voxy_l"
            ix = sid.find(marker)
            if ix != -1:
                level = sid[ix + len(marker) :]
                if level.isdigit():
                    return "voxy_l" + level
        if s.track:
            return s.track
        return "loopback" if s.phase == "loopback" else "data_acq"

    steps_by_track: dict[str, list[StepDef]] = {}
    for s in steps:
        steps_by_track.setdefault(_eff_track(s), []).append(s)

    # Band height = max nodes sharing the same column within a track
    band_height: dict[str, int] = {}
    for tid, tsteps in steps_by_track.items():
        col_counts: dict[int, int] = {}
        for s in tsteps:
            d = depth[s.id]
            col_counts[d] = col_counts.get(d, 0) + 1
        band_height[tid] = max(col_counts.values()) if col_counts else 1

    # Assign band starting rows, respecting TRACK_ORDER
    band_start: dict[str, int] = {}
    cur = 0
    for tid in TRACK_ORDER:
        if tid in steps_by_track:
            band_start[tid] = cur
            cur += band_height[tid]
    for tid in steps_by_track:  # tracks not in TRACK_ORDER go at the end
        if tid not in band_start:
            band_start[tid] = cur
            cur += band_height.get(tid, 1)

    # Assign final (col, row) — preserve PIPELINE_STEPS order within same depth
    sub_row: dict[str, dict[int, int]] = {t: {} for t in steps_by_track}
    positions: dict[str, tuple[int, int]] = {}
    for s in steps:
        tid = _eff_track(s)
        col = depth[s.id]
        sr = sub_row[tid].get(col, 0)
        sub_row[tid][col] = sr + 1
        positions[s.id] = (col, band_start[tid] + sr)

    band_info: dict[str, tuple[int, int]] = {
        tid: (band_start[tid], band_height[tid]) for tid in band_start
    }
    return positions, band_info


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
    # Context menu actions
    run_from_requested: Signal = Signal(str, str)
    cancel_requested: Signal = Signal(str, str)
    continue_training_requested: Signal = Signal(str, str)  # (profile_name, step_id)

    def __init__(
        self,
        profile_name: str,
        registry: RunRegistry,
        steps: list[StepDef] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.profile_name = profile_name
        self.registry = registry
        # Per-profile step list.  None → fall back to global PIPELINE_STEPS.
        self._steps: list[StepDef] | None = steps

        self._nodes: dict[str, StepNodeWidget] = {}
        self._runnable_steps: set[str] = set()
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)

        # Drag handle (leftmost)
        handle = DragHandle(self.profile_name, self)
        layout.addWidget(handle)
        layout.addSpacing(4)

        # Profile name label
        name_lbl = QLabel(self.profile_name)
        name_lbl.setFixedWidth(90)
        name_lbl.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 12px;")
        layout.addWidget(name_lbl)
        layout.addSpacing(8)

        # DAG nodes container — use per-profile steps if provided, else global list
        all_steps: list[StepDef] = self._steps if self._steps is not None else list(PIPELINE_STEPS)
        self._nodes_container = _NodesWidget(self)
        self._nodes_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Create node widgets (not yet parented — _NodesWidget will own them)
        for step in all_steps:
            is_stub = not step.enabled
            node = StepNodeWidget(
                step.id,
                step.label,
                stub=is_stub,
                server_required=getattr(step, "server_required", False),
                client_required=getattr(step, "client_required", False),
            )
            node.clicked.connect(self._on_node_clicked)
            node.context_menu_requested.connect(self._on_node_contextmenu)
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
            " border-radius: 4px; padding: 4px 8px; } "
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
            " border-radius: 4px; padding: 2px 4px; font-weight: bold; } "
            "QPushButton:hover { background: #7a4a4a; }"
        )
        del_btn.setToolTip(f"Delete profile '{self.profile_name}'")
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.profile_name))
        layout.addWidget(del_btn)

    def _on_node_clicked(self, step_id: str) -> None:
        # Always emit clicks (even if the step is not currently marked runnable),
        # so users can re-run steps or see logs even when the DAG isn't in a
        # runnable state.
        self.node_clicked.emit(self.profile_name, step_id)

    def _on_node_contextmenu(self, step_id: str, global_pos) -> None:
        """Build and display the per-step context menu when a node is right-clicked."""
        # Refresh state so that menu enablement matches current registry info.
        self.refresh()

        menu = QMenu(self)
        run_act = menu.addAction("Run")
        from_act = menu.addAction("Run From Here")
        cancel_act = menu.addAction("Cancel")

        # compute enablement
        can_run = False
        is_running = False
        status = "not_run"
        if self.registry:
            can_run = self.registry.can_run(step_id)
            status = self.registry.get_status(step_id)
            is_running = status == "running"
        run_act.setEnabled(can_run)
        from_act.setEnabled(can_run)
        cancel_act.setEnabled(is_running)

        # "Continue training" — only on the voxy train step when a
        # checkpoint is likely present (success or failed after at least one run).
        continue_act = None
        voxy_train_step = STEP_BY_ID.get(step_id)
        if voxy_train_step and voxy_train_step.track == "voxy" and voxy_train_step.phase == "train":
            menu.addSeparator()
            continue_act = menu.addAction("Continue training...")
            continue_act.setEnabled(status in ("success", "failed"))
        chosen = menu.exec_(global_pos)
        if chosen is run_act and run_act.isEnabled():
            self.node_clicked.emit(self.profile_name, step_id)
        elif chosen is from_act and from_act.isEnabled():
            self.run_from_requested.emit(self.profile_name, step_id)
        elif chosen is cancel_act and cancel_act.isEnabled():
            self.cancel_requested.emit(self.profile_name, step_id)
        elif continue_act is not None and chosen is continue_act:
            self.continue_training_requested.emit(self.profile_name, step_id)

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
            # explicit check for staleness: a succesful step whose prereqs
            # are no longer successful should be shown in a warning state.
            if status == "success" and self.registry.is_stale(step_id):
                # ``status`` is typed as the four normal literals; ``stale`` is
                # a GUI-only augmentation.  Ignore the type mismatch here.
                status = "stale"  # type: ignore[assignment]
            node.set_status(status)
            node.set_runnable(step_id in self._runnable_steps)

            # Show epoch count for any train step
            if step_id.startswith("train_"):
                epochs = self.registry.get_metadata(step_id, "epochs_completed")
                node.set_metadata(f"{epochs}e" if epochs is not None else None)
            else:
                node.set_metadata(None)

            # Progress updates stored separately in metadata under "progress".
            prog = self.registry.get_metadata(step_id, "progress")
            if isinstance(prog, (int, float)):
                node.set_progress(float(prog))
            else:
                node.set_progress(None)

        # Repaint edges to reflect new statuses
        self._nodes_container.update()


class _NodesWidget(QWidget):
    """DAG node-graph container.  Positions child nodes absolutely (no layout
    manager) and draws Bezier edges between them in ``paintEvent``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_positions: dict[str, tuple[int, int]] = {}  # step_id → (col, row)
        self._band_info: dict[str, tuple[int, int]] = {}  # track_id → (start_row, height)
        self._node_widgets: dict[str, StepNodeWidget] = {}
        self._active_ids: set[str] = set()
        self._steps: list[StepDef] = []  # stored for paintEvent

    def layout_nodes(
        self,
        steps: list[StepDef],
        nodes: dict[str, StepNodeWidget],
    ) -> None:
        """Compute DAG positions and place child node widgets."""
        self._node_widgets = nodes
        self._steps = steps  # store for paintEvent
        # active IDs = all enabled steps in the provided list
        self._active_ids = {s.id for s in steps if s.enabled}
        self._step_positions, self._band_info = _compute_dag_layout(steps)

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

        # Swim-lane backgrounds — one tinted strip per model track
        for tid, (start_row, height) in self._band_info.items():
            track = TRACK_BY_ID.get(tid)
            if track is None and tid.startswith("voxy_l"):
                track = TRACK_BY_ID.get("voxy")
            if track is None:
                continue  # data_acq and loopback have no ModelTrack entry
            y = start_row * _ROW_H + _V_PAD - _ROW_GAP // 2
            h = height * _ROW_H + _ROW_GAP
            painter.fillRect(0, y, self.width(), h, QColor(track.swim_lane_color))

        for step in self._steps:
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
                is_stub = step.id not in self._active_ids or prereq_id not in self._active_ids
                if is_stub:
                    pen = QPen(QColor("#404040"), 1.5, Qt.PenStyle.DashLine)
                else:
                    src_node = self._node_widgets.get(prereq_id)
                    dst_node = self._node_widgets.get(step.id)
                    ss = src_node._status if src_node else "not_run"
                    ds = dst_node._status if dst_node else "not_run"
                    # running always takes precedence
                    if ss == "running" or ds == "running":
                        color = QColor("#e8a800")
                    # any failure overrides
                    elif ss == "failed" or ds == "failed":
                        color = QColor("#dc3545")
                    # stale nodes get a warning colour (yellow)
                    elif ss == "stale" or ds == "stale":
                        color = QColor("#ffc107")
                    # both success → green
                    elif ss == "success" and ds == "success":
                        color = QColor("#28a745")
                    # prerequisite succeeded but dst not yet
                    elif ss == "success":
                        color = QColor("#5a8abf")
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
                offset = (dx - sx) * 0.45
                path = QPainterPath()
                path.moveTo(sx, sy)
                path.cubicTo(sx + offset, sy, dx - offset, dy, dx, dy)
                painter.drawPath(path)
