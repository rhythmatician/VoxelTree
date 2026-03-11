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

from gui.run_registry import RunRegistry
from gui.run_worker import RunWorker
from gui.step_definitions import ACTIVE_STEPS, STEP_BY_ID


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
        self._worker: RunWorker | None = None
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

    def _on_edit(self) -> None:
        if self._edit_callback and self._profile_name:
            self._edit_callback(self._profile_name)

    def _refresh_buttons(self) -> None:
        if not self._registry:
            return
        running_id = self._running_step_id()
        for step_id, row in self._step_rows.items():
            status = self._registry.get_status(step_id)
            can_run = self._registry.can_run(step_id) and running_id is None
            row.update_state(status=status, can_run=can_run, is_running=(step_id == running_id))

    def _running_step_id(self) -> str | None:
        if self._worker and self._worker.isRunning():
            return self._worker.step_id
        return None

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def run_step(self, step_id: str) -> None:
        """Public method to run a single step (if prerequisites are met)."""
        self._run_step(step_id)

    def _run_step(self, step_id: str) -> None:
        if not self._registry or self._worker and self._worker.isRunning():
            return
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
        """Run all steps from step_id through the last step, in sequence."""
        # Find index of step_id and run from there
        ids = [s.id for s in ACTIVE_STEPS]
        if step_id not in ids:
            return
        start_idx = ids.index(step_id)
        remaining = [ACTIVE_STEPS[i] for i in range(start_idx, len(ACTIVE_STEPS))]
        if remaining:
            self._run_step(remaining[0].id)
            # Remaining steps will be chained via step_finished → _auto_advance

    def _auto_advance(self, completed_step_id: str, exit_code: int) -> None:
        """After a step finishes, automatically run the next step if all went well."""
        if not self._registry or exit_code != 0:
            return
        ids = [s.id for s in ACTIVE_STEPS]
        if completed_step_id not in ids:
            return
        idx = ids.index(completed_step_id)
        if idx + 1 < len(ids):
            next_id = ids[idx + 1]
            if hasattr(self, "_run_from_active") and self._run_from_active:
                self._run_step(next_id)

    def _launch_worker(self, step_id: str, cmd: list[str]) -> None:
        assert self._registry is not None
        self._registry.mark_started(step_id)
        self._refresh_buttons()

        ts = datetime.now().strftime("%H:%M:%S")
        self.append_log(f"\n{'─'*60}")
        self.append_log(f"[{ts}] Starting: {step_id}")
        self.append_log("$ " + " ".join(cmd))
        self.append_log(f"{'─'*60}")

        self._worker = RunWorker(step_id, cmd)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.step_finished.connect(self._on_step_finished)
        self._worker.start()
        self._poll_timer.start()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self.append_log(line)

    @Slot(str, int)
    def _on_step_finished(self, step_id: str, exit_code: int) -> None:
        assert self._registry is not None
        self._poll_timer.stop()
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

        self._worker = None
        self._refresh_buttons()

        # Notify the parent window so the dashboard row refreshes
        parent = self.parent()
        if hasattr(parent, "on_step_finished"):
            parent.on_step_finished(self._profile_name, step_id)

    def _cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

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
