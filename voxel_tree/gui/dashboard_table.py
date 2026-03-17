"""dashboard_table.py — Scrollable table of profile rows with column headers."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voxel_tree.gui.profile_row import ProfileRow
from voxel_tree.gui.run_registry import RunRegistry
from voxel_tree.gui.step_definitions import StepDef


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

        self._rows_container = QWidget()
        self._rows_container.setStyleSheet("background: #1a1a1a;")
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 4, 0, 4)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()  # pushes rows to the top

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
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.setStyleSheet("ProfileRow { background: #232323; border-bottom: 1px solid #2e2e2e; }")

        # Insert before the stretch item at the end
        count = self._rows_layout.count()
        self._rows_layout.insertWidget(count - 1, row)
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
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        del self._rows[profile_name]
        self.add_profile(profile_name, registry, steps=steps)

    def remove_profile(self, profile_name: str) -> None:
        row = self._rows.pop(profile_name, None)
        if row:
            self._rows_layout.removeWidget(row)
            row.deleteLater()

    def refresh_profile(self, profile_name: str) -> None:
        row = self._rows.get(profile_name)
        if row:
            row.refresh()

    def refresh_all(self) -> None:
        for row in self._rows.values():
            row.refresh()

    def profile_names(self) -> list[str]:
        return list(self._rows.keys())

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
