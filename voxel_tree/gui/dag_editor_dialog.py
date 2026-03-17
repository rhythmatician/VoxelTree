"""dag_editor_dialog.py — Dialog to edit a profile's active pipeline DAG.

Users can:
  • Enable / disable individual steps (checkboxes in the left list).
  • Override the prerequisites of any step (checkboxes in the right panel).
  • Preview the resulting topological order in a summary line.
  • Accept to save, or Cancel to discard changes.

The dialog returns a ``ProfileDag`` via ``result_dag()``; the caller is
responsible for persisting it (typically into the ``dag:`` section of the
profile YAML).
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from voxel_tree.gui.dag_definition import DagStepEntry, ProfileDag
from voxel_tree.gui.step_definitions import PIPELINE_STEPS, STEP_BY_ID


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
        self.setMinimumSize(720, 540)
        self.setStyleSheet(
            "QDialog { background: #1e1e1e; color: #cccccc; }"
            "QLabel { color: #aaaaaa; }"
            "QGroupBox { color: #8899bb; font-weight: bold; border: 1px solid #333;"
            " border-radius: 4px; margin-top: 8px; }"
            "QCheckBox { color: #cccccc; }"
        )

        # ── Initialise internal state from incoming dag ──
        if dag is None or dag.is_empty:
            # Default: all enabled steps from the registry
            self._active: set[str] = {s.id for s in PIPELINE_STEPS if s.enabled}
            self._entries: dict[str, DagStepEntry] = {
                s.id: DagStepEntry(id=s.id, prereqs=None) for s in PIPELINE_STEPS if s.enabled
            }
        else:
            self._active = set(dag.active_ids)
            self._entries = {e.id: DagStepEntry(id=e.id, prereqs=e.prereqs) for e in dag.entries}

        self._result: ProfileDag | None = None
        self._prereq_checkboxes: dict[str, QCheckBox] = {}  # prereq_id → widget
        self._current_step_id: str | None = None

        self._build_ui()
        # Select first item to populate the prereq panel immediately
        if self._step_list.count() > 0:
            self._step_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title / instructions
        intro = QLabel(
            "Check which steps to include. "
            "Select a step and adjust its prerequisites on the right."
        )
        intro.setStyleSheet("color: #7799bb; font-size: 11px;")
        root.addWidget(intro)

        # ── Main splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #333; width: 4px; }")

        # ──────────────────────── Left: step list ──────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(4)

        lv.addWidget(_header_label("Available Steps"))

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
                # Future stub — shown greyed, not interactive
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

        self._step_list.currentItemChanged.connect(self._on_step_selected)
        self._step_list.itemChanged.connect(self._on_step_checked)
        lv.addWidget(self._step_list, stretch=1)

        splitter.addWidget(left)

        # ──────────────────────── Right: prereq editor ─────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(4)

        self._prereq_title = _header_label("Prerequisites for: (select a step)")
        rv.addWidget(self._prereq_title)

        prereq_scroll = QScrollArea()
        prereq_scroll.setWidgetResizable(True)
        prereq_scroll.setStyleSheet("QScrollArea { background: #252525; border: 1px solid #444; }")
        self._prereq_container = QWidget()
        self._prereq_container.setStyleSheet("background: #252525;")
        self._prereqs_layout = QVBoxLayout(self._prereq_container)
        self._prereqs_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._prereqs_layout.setSpacing(4)
        prereq_scroll.setWidget(self._prereq_container)
        rv.addWidget(prereq_scroll, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([320, 400])
        root.addWidget(splitter, stretch=1)

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
            " border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:hover { background: #3a5a3a; }"
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_step_selected(self, current: QListWidgetItem | None, _prev) -> None:
        step_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._current_step_id = step_id
        self._refresh_prereq_panel(step_id)

    def _on_step_checked(self, item: QListWidgetItem) -> None:
        step_id: str = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._active.add(step_id)
            if step_id not in self._entries:
                self._entries[step_id] = DagStepEntry(id=step_id, prereqs=None)
        else:
            self._active.discard(step_id)
        self._update_preview()
        # Refresh prereq panel if the toggled step is currently selected
        if self._current_step_id == step_id:
            self._refresh_prereq_panel(step_id)
        # Also refresh other panels (the candidate list may have changed)
        elif self._current_step_id is not None:
            self._refresh_prereq_panel(self._current_step_id)

    def _on_prereq_toggled(self, step_id: str, prereq_id: str, checked: bool) -> None:
        entry = self._entries.setdefault(step_id, DagStepEntry(id=step_id, prereqs=None))
        if entry.prereqs is None:
            # Materialise the default prereqs before editing
            tmpl = STEP_BY_ID.get(step_id)
            entry.prereqs = list(tmpl.prereqs) if tmpl else []
        if checked:
            if prereq_id not in entry.prereqs:
                entry.prereqs.append(prereq_id)
        else:
            if prereq_id in entry.prereqs:
                entry.prereqs.remove(prereq_id)
        self._update_preview()

    # ------------------------------------------------------------------
    # Panel refresh
    # ------------------------------------------------------------------

    def _refresh_prereq_panel(self, step_id: str | None) -> None:
        """Rebuild the right-hand prereq checkboxes for *step_id*."""
        # Clear existing widgets
        while self._prereqs_layout.count():
            item = self._prereqs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._prereq_checkboxes.clear()

        if step_id is None:
            self._prereq_title.setText("Prerequisites for: (select a step)")
            return

        step = STEP_BY_ID.get(step_id)
        if step is None:
            self._prereq_title.setText(f"Prerequisites for: {step_id}  [not in registry]")
            return

        self._prereq_title.setText(f"Prerequisites for:  {step.label}  ({step_id})")

        # Determine currently active prereqs for this step
        entry = self._entries.get(step_id)
        if entry and entry.prereqs is not None:
            current_prereqs: list[str] = entry.prereqs
        else:
            current_prereqs = [p for p in (step.prereqs or []) if p in self._active]

        # Candidate list: all other currently active registry steps
        candidates = [
            s for s in PIPELINE_STEPS if s.id != step_id and s.id in self._active and s.enabled
        ]

        if not candidates:
            lbl = QLabel("No other active steps to use as prerequisites.")
            lbl.setStyleSheet("color: #666666; font-size: 11px; padding: 4px;")
            self._prereqs_layout.addWidget(lbl)
            return

        for candidate in candidates:
            cb = QCheckBox(f"  {candidate.label}  ({candidate.id})")
            cb.setChecked(candidate.id in current_prereqs)
            cb.setStyleSheet("color: #cccccc; padding: 2px 0;")
            cb.toggled.connect(
                lambda checked, sid=step_id, pid=candidate.id: self._on_prereq_toggled(
                    sid, pid, checked
                )
            )
            self._prereqs_layout.addWidget(cb)
            self._prereq_checkboxes[candidate.id] = cb

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
            self._entries[s.id]
            for s in PIPELINE_STEPS
            if s.id in self._active and s.id in self._entries
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
