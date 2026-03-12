"""detail_panel.py — Right-side dock panel: run controls + live log for one profile."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from VoxelTree.gui.run_registry import RunRegistry
from VoxelTree.gui.run_worker import RunWorker
from VoxelTree.gui.step_definitions import ACTIVE_STEPS, STEP_BY_ID


class DetailPanel(QDockWidget):
    """Dockable panel shown when the user clicks "Details" on a profile row.

    Shows:
    - Profile name + edit button (edit triggers an external callback)
    - Per-step run buttons (Run, Run From Here, Cancel)
    - Live log output (QTextEdit)
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Details", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        self._profile_name: str | None = None
        self._registry: RunRegistry | None = None
        self._workers: dict[str, RunWorker] = {}  # step_id → worker
        self._run_from_targets: set[str] = set()  # step_ids queued by 'run from'
        self._edit_callback = None  # callable(profile_name) → open editor

        self._build_ui()

        # Poll registry to refresh button states while a step runs
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._refresh_buttons)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root.setStyleSheet("background: #1e1e1e; color: #cccccc;")
        self.setWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Profile header ──
        hdr = QHBoxLayout()
        self._name_lbl = QLabel("(no profile selected)")
        self._name_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaccff;")
        hdr.addWidget(self._name_lbl)
        hdr.addStretch()
        self._edit_btn = QPushButton("✎ Edit Profile")
        self._edit_btn.setFixedWidth(100)
        self._edit_btn.setStyleSheet(
            "QPushButton { background: #2a3a4a; color: #aaccff; border: 1px solid #4a6a8a; "
            "border-radius: 4px; padding: 3px 8px; } "
            "QPushButton:hover { background: #3a4a5a; }"
        )
        self._edit_btn.clicked.connect(self._on_edit)
        hdr.addWidget(self._edit_btn)
        layout.addLayout(hdr)

        # ── Step selector & controls ──
        step_scroll = QScrollArea()
        step_scroll.setWidgetResizable(True)
        step_scroll.setFixedHeight(160)
        step_scroll.setStyleSheet("QScrollArea { border: 1px solid #333; }")

        step_container = QWidget()
        step_container.setStyleSheet("background: #252525;")
        self._step_layout = QVBoxLayout(step_container)
        self._step_layout.setContentsMargins(6, 6, 6, 6)
        self._step_layout.setSpacing(3)

        self._step_rows: dict[str, _StepControlRow] = {}
        for step in ACTIVE_STEPS:
            row = _StepControlRow(step.id, step.label)
            row.run_clicked.connect(self._run_step)
            row.run_from_clicked.connect(self._run_from)
            row.cancel_clicked.connect(self._cancel)
            self._step_layout.addWidget(row)
            self._step_rows[step.id] = row

        step_scroll.setWidget(step_container)
        layout.addWidget(step_scroll)

        # ── Log area ──
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Log Output"))
        log_hdr.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(55)
        clear_btn.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: 1px solid #444; "
            "border-radius: 3px; padding: 2px 6px; } "
            "QPushButton:hover { background: #444; }"
        )
        clear_btn.clicked.connect(self._clear_log)
        log_hdr.addWidget(clear_btn)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(55)
        copy_btn.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: 1px solid #444; "
            "border-radius: 3px; padding: 2px 6px; } "
            "QPushButton:hover { background: #444; }"
        )
        copy_btn.clicked.connect(self._copy_log)
        log_hdr.addWidget(copy_btn)

        layout.addLayout(log_hdr)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            "QTextEdit { background: #141414; color: #b0c4b0; border: 1px solid #333; }"
        )
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._log, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_profile(self, profile_name: str, registry: RunRegistry) -> None:
        """Switch panel to show the given profile."""
        self._profile_name = profile_name
        self._registry = registry
        self._name_lbl.setText(profile_name)
        self._log.clear()
        self._refresh_buttons()
        self.setWindowTitle(f"Details — {profile_name}")
        self.show()

    def set_edit_callback(self, cb) -> None:
        """Set a callable(profile_name) invoked when Edit Profile is clicked."""
        self._edit_callback = cb

    def append_log(self, line: str) -> None:
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertPlainText(line + "\n")
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @Slot(str, float)
    def _on_progress(self, step_id: str, fraction: float) -> None:
        """Handle a progress update from a RunWorker.

        We store the fractional value in the registry and notify the parent window
        so that the dashboard row can refresh immediately.
        """
        if self._registry:
            self._registry.set_progress(step_id, fraction)
        parent = self.parent()
        if hasattr(parent, "on_step_progress"):
            parent.on_step_progress(self._profile_name, step_id)  # type: ignore[arg-type]
        elif hasattr(parent, "_dashboard"):
            # fallback to directly refreshing the table row
            parent._dashboard.refresh_profile(self._profile_name)  # type: ignore[arg-type]

    def _on_edit(self) -> None:
        if self._edit_callback and self._profile_name:
            self._edit_callback(self._profile_name)

    def _refresh_buttons(self) -> None:
        if not self._registry:
            return
        running_ids = self._running_step_ids()
        for step_id, row in self._step_rows.items():
            status = self._registry.get_status(step_id)
            can_run = self._registry.can_run(step_id) and step_id not in running_ids
            row.update_state(
                status=status,
                can_run=can_run,
                is_running=(step_id in running_ids),
            )

    def _running_step_ids(self) -> set[str]:
        return {sid for sid, w in self._workers.items() if w.isRunning()}

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def run_step(self, step_id: str) -> None:
        """Public method to run a single step (if prerequisites are met)."""
        self._run_step(step_id)

    def run_from_step(self, step_id: str) -> None:
        """Public wrapper for the internal _run_from method.

        This mirrors the behaviour of the "Run From Here" button in the UI
        and allows other widgets (e.g. context menus) to invoke it.
        """
        self._run_from(step_id)

    def cancel(self) -> None:
        """Public method to cancel any running steps for the current profile."""
        self._cancel()

    def _run_step(self, step_id: str) -> None:
        if not self._registry:
            return
        if step_id in self._running_step_ids():
            return  # already running
        step = STEP_BY_ID.get(step_id)
        if not step:
            return
        # We need the profile dict to build the command — fetched from parent window
        profile_dict = self._get_profile_dict()
        if profile_dict is None:
            return
        cmd = step.cmd_factory(profile_dict)
        self._launch_worker(step_id, cmd)

    def _run_from(self, step_id: str) -> None:
        """Run step_id and all downstream steps reachable through the DAG."""
        if not self._registry:
            return
        # BFS from step_id through the DAG to collect all reachable step IDs
        reachable: set[str] = set()
        queue = [step_id]
        while queue:
            current = queue.pop()
            if current in reachable:
                continue
            reachable.add(current)
            # Add any active step that lists current as a prereq
            for step in ACTIVE_STEPS:
                if current in step.prereqs and step.id not in reachable:
                    queue.append(step.id)
        self._run_from_targets = reachable
        # Launch all currently runnable steps within reachable set
        for sid in list(self._run_from_targets):
            if self._registry.can_run(sid) and sid not in self._running_step_ids():
                status = self._registry.get_status(sid)
                if status not in ("running", "success"):
                    self._run_step(sid)

    def _auto_advance(self, completed_step_id: str, exit_code: int) -> None:
        """Superseded by _run_from_targets logic in _on_step_finished.  Kept for safety."""
        pass

    def _launch_worker(self, step_id: str, cmd: list[str]) -> None:
        assert self._registry is not None
        self._registry.mark_started(step_id)
        self._refresh_buttons()

        ts = datetime.now().strftime("%H:%M:%S")
        self.append_log(f"\n{'─'*60}")
        self.append_log(f"[{ts}] Starting: {step_id}")
        self.append_log("$ " + " ".join(cmd))
        self.append_log(f"{'─'*60}")

        worker = RunWorker(step_id, cmd)
        worker.log_line.connect(self._on_log_line)
        worker.step_finished.connect(self._on_step_finished)
        worker.progress.connect(self._on_progress)
        self._workers[step_id] = worker
        worker.start()
        self._poll_timer.start()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self.append_log(line)

    @Slot(str, int)
    def _on_step_finished(self, step_id: str, exit_code: int) -> None:
        assert self._registry is not None
        ts = datetime.now().strftime("%H:%M:%S")
        if exit_code == 0:
            self._registry.mark_success(step_id)
            self.append_log(f"\n[{ts}] ✓  {step_id} completed successfully.")
        elif exit_code == -2:
            self._registry.mark_failed(step_id, exit_code)
            self.append_log(f"\n[{ts}] ⚠  {step_id} cancelled.")
        else:
            self._registry.mark_failed(step_id, exit_code)
            self.append_log(f"\n[{ts}] ✗  {step_id} failed (exit {exit_code}).")

        # Remove completed worker; stop poll timer only when all workers done
        # also clear any stored progress metadata for this step
        if self._registry:
            self._registry.set_metadata(step_id, "progress", None)
        self._workers.pop(step_id, None)
        if not self._workers:
            self._poll_timer.stop()

        self._refresh_buttons()

        # Auto-advance: if this step succeeded and 'run from' mode is active,
        # launch any downstream steps that are now eligible
        if exit_code == 0 and self._run_from_targets:
            self._run_from_targets.discard(step_id)
            for sid in list(self._run_from_targets):
                if (
                    self._registry.can_run(sid)
                    and sid not in self._running_step_ids()
                    and self._registry.get_status(sid) not in ("running", "success")
                ):
                    self._run_step(sid)

        # Notify the parent window so the dashboard row refreshes
        parent = self.parent()
        if hasattr(parent, "on_step_finished"):
            parent.on_step_finished(self._profile_name, step_id)

    def _cancel(self) -> None:
        for worker in list(self._workers.values()):
            if worker.isRunning():
                worker.cancel()
        self._run_from_targets.clear()

    def _clear_log(self) -> None:
        self._log.clear()

    def _copy_log(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self._log.toPlainText())

    def _get_profile_dict(self) -> dict | None:
        """Retrieve the loaded profile dict from the main window."""
        parent = self.parent()
        if hasattr(parent, "get_profile_dict"):
            return parent.get_profile_dict(self._profile_name)
        return None


# ---------------------------------------------------------------------------
# Per-step control row inside the detail panel
# ---------------------------------------------------------------------------


class _StepControlRow(QWidget):
    """One row: status indicator + step label + Run / Run From / Cancel buttons."""

    run_clicked = Signal(str)  # step_id
    run_from_clicked = Signal(str)
    cancel_clicked = Signal(str)

    _STATUS_COLORS = {
        "not_run": "#555",
        "running": "#e8a800",
        "success": "#28a745",
        "failed": "#dc3545",
        "stale": "#ffc107",  # warn when the step is outdated
    }

    def __init__(self, step_id: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self.step_id = step_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(14)
        self._dot.setStyleSheet("color: #555; font-size: 12px;")
        layout.addWidget(self._dot)

        lbl = QLabel(label)
        lbl.setFixedWidth(70)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(lbl)

        layout.addStretch()

        self._run_btn = self._make_btn("▶  Run", "#1e3d1e", "#4a8a4a")
        self._run_btn.clicked.connect(lambda: self.run_clicked.emit(self.step_id))
        layout.addWidget(self._run_btn)

        self._from_btn = self._make_btn("⏩ From Here", "#1e2d3d", "#4a6a8a")
        self._from_btn.clicked.connect(lambda: self.run_from_clicked.emit(self.step_id))
        layout.addWidget(self._from_btn)

        self._cancel_btn = self._make_btn("■ Cancel", "#3d1e1e", "#8a4a4a")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(lambda: self.cancel_clicked.emit(self.step_id))
        layout.addWidget(self._cancel_btn)

    @staticmethod
    def _make_btn(text: str, bg: str, border: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedWidth(90)
        stylesheet = (
            f"QPushButton {{ background: {bg}; color: #ccc; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px 6px; font-size: 10px; }} "
            f"QPushButton:hover {{ background: {border}; }} "
            f"QPushButton:disabled {{ color: #555; border-color: #333; background: #252525; }}"
        )
        btn.setStyleSheet(stylesheet)
        return btn

    def update_state(self, status: str, can_run: bool, is_running: bool) -> None:
        color = self._STATUS_COLORS.get(status, "#555")
        self._dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._run_btn.setEnabled(can_run)
        self._from_btn.setEnabled(can_run)
        self._cancel_btn.setEnabled(is_running)
