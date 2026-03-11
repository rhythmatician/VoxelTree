"""dashboard_table.py — Scrollable table of profile rows with column headers."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.profile_row import ProfileRow
from gui.run_registry import RunRegistry
from gui.step_definitions import ACTIVE_STEPS, STUB_STEPS


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
    new_profile_requested: Signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, ProfileRow] = {}
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Column header bar ──
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background: #252525;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(0)

        # Spacer matching profile-name label width
        spacer_lbl = QLabel()
        spacer_lbl.setFixedWidth(90 + 8)
        h_layout.addWidget(spacer_lbl)

        all_steps = list(ACTIVE_STEPS) + list(STUB_STEPS)
        # Each node is 52px wide; connectors are 28px wide
        for i, step in enumerate(all_steps):
            if i > 0:
                conn_spacer = QLabel()
                conn_spacer.setFixedWidth(28)
                h_layout.addWidget(conn_spacer)

            lbl = QLabel(step.label)
            lbl.setFixedWidth(52)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # Set alignment via code
            color = "#505050" if not step.enabled else "#8899bb"
            lbl.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold;"
            )
            h_layout.addWidget(lbl)

        h_layout.addStretch()

        # Spacer matching "Details" button width (70px + 8px gap)
        btn_spacer = QLabel()
        btn_spacer.setFixedWidth(78)
        h_layout.addWidget(btn_spacer)

        root.addWidget(header)

        # ── Scrollable rows area ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self._rows_container = QWidget()
        self._rows_container.setStyleSheet("background: #1a1a1a;")
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 4, 0, 4)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()   # pushes rows to the top

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
        t_layout.addStretch()

        root.addWidget(toolbar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_profile(self, profile_name: str, registry: RunRegistry) -> None:
        if profile_name in self._rows:
            return
        row = ProfileRow(profile_name, registry)
        row.details_clicked.connect(self.details_clicked)
        row.node_clicked.connect(self.node_clicked)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.setStyleSheet(
            "ProfileRow { background: #232323; border-bottom: 1px solid #2e2e2e; }"
        )

        # Insert before the stretch item at the end
        count = self._rows_layout.count()
        self._rows_layout.insertWidget(count - 1, row)
        self._rows[profile_name] = row

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
