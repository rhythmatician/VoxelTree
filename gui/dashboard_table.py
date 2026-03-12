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

from gui.profile_row import ProfileRow, _COL_W, _NODE_W, _compute_dag_layout
from gui.run_registry import RunRegistry
from gui.step_definitions import PIPELINE_STEPS


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

        # ── Column header bar (absolute-positioned per DAG column/row) ──
        positions = _compute_dag_layout(list(PIPELINE_STEPS))
        max_row = max(r for _, r in positions.values())
        _LABEL_H = 14
        _LABEL_ROW_STEP = 18
        _LEFT = 8 + 90 + 8   # layout-margin + name-label + spacing
        header_h = (max_row + 1) * _LABEL_ROW_STEP + 8

        header = QWidget()
        header.setFixedHeight(header_h)
        header.setStyleSheet("background: #252525;")

        for step in PIPELINE_STEPS:
            col, row = positions[step.id]
            x = _LEFT + col * _COL_W
            y = row * _LABEL_ROW_STEP + 4
            lbl = QLabel(step.label, header)
            lbl.move(x, y)
            lbl.setFixedSize(_NODE_W, _LABEL_H)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            color = "#505050" if not step.enabled else "#8899bb"
            lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")

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
        row.delete_clicked.connect(self.delete_profile_requested)
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
