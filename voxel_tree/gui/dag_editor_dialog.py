"""dag_editor_dialog.py — Dialog to edit a profile's active pipeline DAG.

Users can:
  • Enable / disable individual steps (checkboxes in the step list).
  • Preview the resulting topological order in a summary line.
  • Accept to save, or Cancel to discard changes.

Prerequisites are derived automatically from the artifact dependency graph
and cannot be overridden per-profile.

The dialog returns a ``ProfileDag`` via ``result_dag()``; the caller is
responsible for persisting it (typically into the ``dag:`` section of the
profile YAML).
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from voxel_tree.gui.dag_definition import DagStepEntry, ProfileDag
from voxel_tree.gui.step_definitions import PIPELINE_STEPS


class DagEditorDialog(QDialog):
    """Modal dialog for editing per-profile pipeline DAG topology.

    Parameters
    ----------
    profile_name:
        Displayed in the title bar for context.
    dag:
        Current ``ProfileDag`` (or *None* / empty → start from the full
        default PIPELINE_STEPS set).
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        profile_name: str,
        dag: ProfileDag | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit Pipeline DAG — {profile_name}")
        self.setMinimumSize(520, 540)
        self.setStyleSheet(
            "QDialog { background: #1e1e1e; color: #cccccc; }"
            "QLabel { color: #aaaaaa; }"
            "QGroupBox { color: #8899bb; font-weight: bold; border: 1px solid #333;"
            " border-radius: 4px; margin-top: 8px; }"
            "QCheckBox { color: #cccccc; }"
        )

        # ── Initialise internal state from incoming dag ──
        if dag is None or dag.is_empty:
            self._active: set[str] = {s.id for s in PIPELINE_STEPS if s.enabled}
        else:
            self._active = set(dag.active_ids)

        self._result: ProfileDag | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title / instructions
        intro = QLabel(
            "Check which steps to include in this profile. "
            "Prerequisites are determined automatically from the artifact graph."
        )
        intro.setStyleSheet("color: #7799bb; font-size: 11px;")
        root.addWidget(intro)

        # ──────────────────────── Step list ──────────────────────
        root.addWidget(_header_label("Available Steps"))

        self._step_list = QListWidget()
        self._step_list.setStyleSheet(
            "QListWidget { background: #252525; border: 1px solid #444; color: #cccccc; }"
            "QListWidget::item:selected { background: #2a4a6e; color: #ffffff; }"
            "QListWidget::item:hover { background: #2a3a2a; }"
        )

        for step in PIPELINE_STEPS:
            label = f"  {step.label}  ({step.id})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, step.id)

            if not step.enabled:
                item.setForeground(Qt.GlobalColor.darkGray)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(
                    Qt.CheckState.Checked if step.id in self._active else Qt.CheckState.Unchecked
                )

            self._step_list.addItem(item)

        self._step_list.itemChanged.connect(self._on_step_checked)
        root.addWidget(self._step_list, stretch=1)

        # ── Preview bar ──
        self._preview_label = QLabel()
        self._preview_label.setStyleSheet(
            "color: #7799bb; font-size: 10px; background: #252525;"
            " padding: 5px 6px; border: 1px solid #444; border-radius: 3px;"
        )
        self._preview_label.setWordWrap(True)
        root.addWidget(self._preview_label)
        self._update_preview()

        # ── Dialog buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(
            "QPushButton { background: #2a3a2a; color: #ccc; border: 1px solid #4a8a4a;"
            " border-radius: 4px; padding: 4px 16px; } "
            "QPushButton:hover { background: #3a5a3a; }"
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_step_checked(self, item: QListWidgetItem) -> None:
        step_id: str = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._active.add(step_id)
        else:
            self._active.discard(step_id)
        self._update_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        dag = self._build_current_dag()
        try:
            steps = dag.resolve_steps()
        except Exception as exc:
            self._preview_label.setText(f"⚠  Error resolving DAG: {exc}")
            return

        # Quick cycle check
        graph = {s.id: set(s.prereqs) for s in steps}
        try:
            topo_order = list(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            self._preview_label.setText(f"⚠  Cycle detected: {exc}")
            return

        id_to_label = {s.id: s.label for s in steps}
        ordered_labels = [id_to_label[nid] for nid in topo_order if nid in id_to_label]
        self._preview_label.setText(f"DAG  ({len(steps)} steps):  " + "  →  ".join(ordered_labels))

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_current_dag(self) -> ProfileDag:
        """Assemble a ProfileDag from the current UI state."""
        ordered = [
            DagStepEntry(id=s.id)
            for s in PIPELINE_STEPS
            if s.id in self._active
        ]
        return ProfileDag(entries=ordered)

    def _on_accept(self) -> None:
        dag = self._build_current_dag()
        # Validate: no cycles
        try:
            steps = dag.resolve_steps()
            graph = {s.id: set(s.prereqs) for s in steps}
            list(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Cycle Detected", f"The DAG contains a cycle:\n{exc}")
            return
        self._result = dag
        self.accept()

    # ------------------------------------------------------------------
    # Public result accessor
    # ------------------------------------------------------------------

    def result_dag(self) -> ProfileDag | None:
        """Return the edited ProfileDag, or *None* if the dialog was cancelled."""
        return self._result


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------


def _header_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #8899bb; font-weight: bold; font-size: 11px; padding: 2px 0;")
    return lbl
