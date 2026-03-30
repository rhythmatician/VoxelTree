"""dashboard_table.py — Scrollable table of profile rows with column headers."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voxel_tree.gui.profile_editor import save_profile_order
from voxel_tree.gui.profile_row import MIME_TYPE_PROFILE, ProfileRow
from voxel_tree.gui.run_registry import RunRegistry
from voxel_tree.gui.step_definitions import StepDef


class _RowsContainer(QWidget):
    """VBox container that accepts drag-and-drop reordering of ProfileRow children.

    Emits ``order_changed`` with the new ordered list of profile names whenever the
    user drops a row into a different position.
    """

    order_changed: Signal = Signal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("background: #1a1a1a;")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(2)
        self._layout.addStretch()  # keeps rows pushed to the top

        self._order: list[str] = []
        self._name_to_widget: dict[str, QWidget] = {}
        self._drop_line_y: int | None = None

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def add_row(self, name: str, widget: QWidget) -> None:
        """Append *widget* for *name* at the end (before the trailing stretch)."""
        idx = self._layout.count() - 1  # just before stretch
        self._layout.insertWidget(idx, widget)
        self._order.append(name)
        self._name_to_widget[name] = widget

    def insert_row(self, idx: int, name: str, widget: QWidget) -> None:
        """Insert *widget* for *name* at display position *idx*."""
        self._layout.insertWidget(idx, widget)
        self._order.insert(idx, name)
        self._name_to_widget[name] = widget

    def remove_row(self, name: str) -> None:
        """Remove the row for *name*, deleting the widget."""
        widget = self._name_to_widget.pop(name, None)
        if widget:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        if name in self._order:
            self._order.remove(name)

    def move_row(self, name: str, to_idx: int) -> None:
        """Move an already-added row to position *to_idx* without deleting it."""
        if name not in self._name_to_widget:
            return
        cur_idx = self._order.index(name)
        if cur_idx == to_idx:
            return
        widget = self._name_to_widget[name]
        self._layout.removeWidget(widget)
        self._order.pop(cur_idx)
        if to_idx > cur_idx:
            to_idx -= 1
        self._layout.insertWidget(to_idx, widget)
        self._order.insert(to_idx, name)

    def current_order(self) -> list[str]:
        """Return the current display order of profile names."""
        return list(self._order)

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_TYPE_PROFILE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_TYPE_PROFILE):
            drop_idx = self._find_drop_index(event.pos().y())
            self._drop_line_y = self._y_for_drop_index(drop_idx)
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drop_line_y = None
        self.update()

    def dropEvent(self, event) -> None:  # noqa: N802
        self._drop_line_y = None
        self.update()

        if not event.mimeData().hasFormat(MIME_TYPE_PROFILE):
            event.ignore()
            return

        profile_name = bytes(event.mimeData().data(MIME_TYPE_PROFILE)).decode("utf-8")
        if profile_name not in self._name_to_widget:
            event.ignore()
            return

        target_idx = self._find_drop_index(event.pos().y())
        src_idx = self._order.index(profile_name)

        # Dropped in the same logical slot — no-op
        if target_idx == src_idx or target_idx == src_idx + 1:
            event.ignore()
            return

        widget = self._name_to_widget[profile_name]
        self._layout.removeWidget(widget)
        self._order.pop(src_idx)
        if target_idx > src_idx:
            target_idx -= 1
        self._layout.insertWidget(target_idx, widget)
        self._order.insert(target_idx, profile_name)

        event.acceptProposedAction()
        self.order_changed.emit(list(self._order))

    def _find_drop_index(self, y: int) -> int:
        """Return the insertion index (0…len) for a drop at y-coordinate *y*."""
        for i, name in enumerate(self._order):
            w = self._name_to_widget.get(name)
            if w and y < w.y() + w.height() // 2:
                return i
        return len(self._order)

    def _y_for_drop_index(self, idx: int) -> int:
        """Return the y pixel position at which to draw the drop-indicator line."""
        if idx < len(self._order):
            w = self._name_to_widget.get(self._order[idx])
            if w:
                return w.y() - 1
        if self._order:
            w = self._name_to_widget.get(self._order[-1])
            if w:
                return w.y() + w.height() + 1
        return 4

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._drop_line_y is None:
            return
        p = QPainter(self)
        p.setPen(QPen(QColor("#5a9aff"), 2))
        p.drawLine(8, self._drop_line_y, self.width() - 8, self._drop_line_y)


class DashboardTable(QWidget):
    """Main dashboard widget: column headers + one ProfileRow per profile.

    Signals
    -------
    details_clicked(str)        profile_name
    node_clicked(str, str)      profile_name, step_id
    new_profile_requested()
    """

    details_clicked: Signal = Signal(str)
    node_clicked: Signal = Signal(str, str)
    node_run_from: Signal = Signal(str, str)
    node_cancel: Signal = Signal(str, str)
    new_profile_requested: Signal = Signal()
    delete_profile_requested: Signal = Signal(str)
    continue_training_requested: Signal = Signal(str, str)  # (profile_name, step_id)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, ProfileRow] = {}
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Scrollable rows area ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self._rows_container = _RowsContainer()
        self._rows_container.order_changed.connect(self._on_order_changed)

        scroll.setWidget(self._rows_container)
        root.addWidget(scroll, stretch=1)

        # ── Bottom toolbar: New + ──
        toolbar = QWidget()
        toolbar.setFixedHeight(40)
        toolbar.setStyleSheet("background: #202020; border-top: 1px solid #333;")
        t_layout = QHBoxLayout(toolbar)
        t_layout.setContentsMargins(8, 4, 8, 4)

        new_btn = QPushButton("＋  New Profile")
        new_btn.setFixedWidth(130)
        new_btn.setStyleSheet(
            "QPushButton { background: #2d4a2d; color: #90ee90; border: 1px solid #4a8a4a; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background: #3a6a3a; }"
        )
        new_btn.clicked.connect(self.new_profile_requested)
        t_layout.addWidget(new_btn)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setFixedWidth(110)
        refresh_btn.setStyleSheet(
            "QPushButton { background: #2e5a6e; color: #cce0ff; border: 1px solid #4a7abf; "
            "border-radius: 4px; padding: 4px 10px; font-weight: bold; } "
            "QPushButton:hover { background: #3a7a9e; }"
        )
        refresh_btn.clicked.connect(self.refresh_all)
        t_layout.addWidget(refresh_btn)

        t_layout.addStretch()

        root.addWidget(toolbar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_profile(
        self,
        profile_name: str,
        registry: RunRegistry,
        steps: list[StepDef] | None = None,
    ) -> None:
        """Add a profile row to the dashboard.

        Parameters
        ----------
        profile_name:
            Display name and unique key.
        registry:
            Run state storage for this profile.
        steps:
            Optional custom step list from a per-profile ``ProfileDag``.  When
            *None* the global ``PIPELINE_STEPS`` list is used.
        """
        if profile_name in self._rows:
            return
        row = ProfileRow(profile_name, registry, steps=steps)
        row.details_clicked.connect(self.details_clicked)
        row.delete_clicked.connect(self.delete_profile_requested)
        row.node_clicked.connect(self.node_clicked)
        row.run_from_requested.connect(self.node_run_from)
        row.cancel_requested.connect(self.node_cancel)
        row.continue_training_requested.connect(self.continue_training_requested)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.setStyleSheet("ProfileRow { background: #232323; border-bottom: 1px solid #2e2e2e; }")

        # Insert before the stretch item at the end
        self._rows_container.add_row(profile_name, row)
        self._rows[profile_name] = row

    def update_profile_steps(self, profile_name: str, steps: list[StepDef] | None) -> None:
        """Replace the step list for an already-loaded profile row.

        This rebuilds the row widget in place so that the new DAG topology
        is reflected immediately.  If *steps* is *None* the global default
        is restored.
        """
        row = self._rows.get(profile_name)
        if row is None:
            return
        registry = row.registry
        # Remember position so the rebuilt row lands in the same slot
        order = self._rows_container.current_order()
        idx = order.index(profile_name) if profile_name in order else None
        self._rows_container.remove_row(profile_name)
        del self._rows[profile_name]
        self.add_profile(profile_name, registry, steps=steps)
        if idx is not None:
            self._rows_container.move_row(profile_name, idx)

    def remove_profile(self, profile_name: str) -> None:
        self._rows.pop(profile_name, None)
        self._rows_container.remove_row(profile_name)

    def refresh_profile(self, profile_name: str) -> None:
        row = self._rows.get(profile_name)
        if row:
            row.refresh()

    def refresh_all(self) -> None:
        for row in self._rows.values():
            row.refresh()

    def profile_names(self) -> list[str]:
        return self._rows_container.current_order()

    def _on_order_changed(self, new_order: list[str]) -> None:
        """Persist the new order whenever the user drag-drops a row."""
        save_profile_order(new_order)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def set_step_queue(self, queue: list[tuple[str, str]]) -> None:
        """Compatibility stub: store queued server steps for future use."""
        # This GUI currently doesn't display the queue; keep it for possible
        # future use and to avoid crashes when callers invoke it.
        self._step_queue = queue

    def set_server_status(self, status: str) -> None:
        """Compatibility stub: record last server status."""
        # No-op (server status is already shown in the ServerStatusBar)
        self._server_status = status
