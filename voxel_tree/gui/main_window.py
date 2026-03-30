"""main_window.py — Top-level application window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # noqa: F401  (reserved for future profile migrations)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from voxel_tree.gui.dashboard_table import DashboardTable
from voxel_tree.gui.dag_definition import ProfileDag
from voxel_tree.gui.detail_panel import DetailPanel
from voxel_tree.gui.profile_editor import (
    ProfileEditorDialog,
    delete_profile,
    list_profiles,
    load_profile,
    load_profile_order,
)
from voxel_tree.gui.run_registry import RunRegistry
from voxel_tree.gui.server_manager import ServerManager
from voxel_tree.gui.server_status_bar import ServerStatusBar
from voxel_tree.gui.step_definitions import PIPELINE_STEPS, StepDef
from voxel_tree.gui.training_popup import TrainingResultsPopup
from voxel_tree.gui.continue_training_dialog import ContinueTrainingDialog

# same logic as profile_editor: we want the workspace/project root, not the
# interior Python package folder.  ``parents[2]`` handles both development and
# installed-package layouts.
_PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

# Server-required steps run automatically when the user clicks “Run Server Session”.
# Order matters: pregen must complete before harvest (client needed) can start.
_SERVER_SESSION_STEPS = ["pregen", "harvest", "dumpnoise"]


class MainWindow(QMainWindow):
    """Main application window.

    Layout
    ------
    - Central widget: DashboardTable
    - Right dock:     DetailPanel
    """

    def __init__(self) -> None:
        super().__init__()
        print("[MW.init.1] WindowTitle setting...", flush=True)
        self.setWindowTitle("VoxelTree Pipeline Manager")
        print("[MW.init.2] Resize...", flush=True)
        self.resize(1100, 460)
        print("[MW.init.3] Stylesheet...", flush=True)
        self.setStyleSheet(
            "QMainWindow { background: #1a1a1a; }" "QMenuBar { background: #252525; color: #ccc; }"
        )

        print("[MW.init.4] Cache dicts...", flush=True)
        # Registry cache: profile_name → RunRegistry
        self._registries: dict[str, RunRegistry] = {}
        # Profile dict cache: profile_name → loaded YAML dict
        self._profiles: dict[str, Any] = {}
        # Step queue for server session: list of (profile_name, step_id)
        self._step_queue: list[tuple[str, str]] = []
        # Server session queue state (step IDs to run sequentially)
        self._server_session_queue: list[str] = []
        self._server_session_current: str | None = None
        self._server_session_active = False

        print("[MW.init.4b] Creating ServerManager...", flush=True)
        self._server = ServerManager()
        self._server.log_line.connect(self._on_server_log)
        self._server.status_changed.connect(self._on_server_status_changed)

        print("[MW.init.5] Creating DashboardTable...", flush=True)
        # ── Widgets ──
        self._server_bar = ServerStatusBar(self._server)
        self._dashboard = DashboardTable()
        print("[MW.init.6] Connecting dashboard signals...", flush=True)
        self._dashboard.details_clicked.connect(self._on_details_clicked)
        self._dashboard.delete_profile_requested.connect(self._on_delete_profile)
        self._dashboard.new_profile_requested.connect(self._on_new_profile)
        self._dashboard.node_clicked.connect(self._on_node_clicked)
        self._dashboard.node_run_from.connect(self._on_node_run_from)
        self._dashboard.node_cancel.connect(self._on_node_cancel)
        self._dashboard.continue_training_requested.connect(self._on_continue_training)
        self._server_bar.run_server_session_requested.connect(self._on_run_server_session)

        print("[MW.init.7] Setting central widget...", flush=True)
        # Wrap server bar + dashboard in a single central widget
        _central = QWidget()
        _vbox = QVBoxLayout(_central)
        _vbox.setContentsMargins(0, 0, 0, 0)
        _vbox.setSpacing(0)
        _vbox.addWidget(self._server_bar)
        _vbox.addWidget(self._dashboard)
        self.setCentralWidget(_central)

        print("[MW.init.8] Creating DetailPanel...", flush=True)
        self._detail = DetailPanel(self)
        print("[MW.init.9] Setting edit callback...", flush=True)
        self._detail.set_edit_callback(self._on_edit_profile)
        print("[MW.init.10] Adding dock widget...", flush=True)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._detail)
        print("[MW.init.11] Hiding detail panel...", flush=True)
        self._detail.hide()

        print("[MW.init.12] Creating refresh timer...", flush=True)
        # ── Periodic refresh (catch external file changes) ──
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(3000)
        self._refresh_timer.timeout.connect(self._dashboard.refresh_all)
        self._refresh_timer.start()
        print("[MW.init.12b] Timer started", flush=True)

        # ── Deferred profile loading ──
        # Load profiles AFTER show() to avoid rendering crashes
        QTimer.singleShot(100, self._load_all_profiles)
        print("[MW.init.12c] Profile loading deferred until after show()", flush=True)
        print("[MW.init.13] MainWindow.__init__ complete!", flush=True)

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def _load_all_profiles(self) -> None:
        all_names = set(list_profiles())
        saved_order = load_profile_order()
        # Profiles that appear in the saved order come first; the rest follow alphabetically
        ordered = [n for n in saved_order if n in all_names]
        remaining = sorted(all_names - set(ordered))
        for name in ordered + remaining:
            self._load_profile(name)

    def _load_profile(self, name: str) -> None:
        data = load_profile(name)
        if not data:
            return
        self._profiles[name] = data
        if name not in self._registries:
            self._registries[name] = RunRegistry(name)
        # Reconcile registry state against on-disk artifacts (e.g. noise dumps) so
        # the UI can show correct success statuses even if the user ran the CLI.
        self._registries[name].reconcile_with_profile(data)

        # Resolve per-profile DAG if present, else pass None (→ global default)
        dag = ProfileDag.from_profile_dict(data)
        steps = dag.resolve_steps() if dag is not None else None
        self._sync_epoch_targets(self._registries[name], data, steps)
        self._dashboard.add_profile(name, self._registries[name], steps=steps)

    def _sync_epoch_targets(
        self,
        registry: RunRegistry,
        profile: dict[str, Any],
        steps: list[StepDef] | None,
    ) -> None:
        """Store configured epoch targets per train step for dashboard badges."""
        train_cfg = profile.get("train", {})
        default_epochs = train_cfg.get("epochs")
        step_list = steps if steps is not None else list(PIPELINE_STEPS)
        for step in step_list:
            sid = step.id
            if not sid.startswith("train_"):
                continue
            target = None
            if sid.startswith("train_voxy_l"):
                try:
                    level = int(sid.removeprefix("train_voxy_l"))
                except ValueError:
                    level = None
                if level is not None:
                    target = train_cfg.get(f"epochs_l{level}", default_epochs)
            else:
                target = default_epochs

            try:
                target_int = int(target) if target is not None else None
            except (TypeError, ValueError):
                target_int = None
            registry.set_metadata(sid, "epochs_target", target_int)

    def get_profile_dict(self, profile_name: str) -> dict[str, Any] | None:
        """Called by DetailPanel to get CLI arguments for a step."""
        return self._profiles.get(profile_name)

    def on_step_progress(self, profile_name: str, step_id: str) -> None:
        """Refresh the dashboard row for a profile reporting live progress."""
        self._dashboard.refresh_profile(profile_name)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_details_clicked(self, profile_name: str) -> None:
        registry = self._registries.get(profile_name)
        if registry:
            self._detail.load_profile(profile_name, registry)
            self._detail.show()
            self.resizeDocks([self._detail], [360], Qt.Orientation.Horizontal)

        # Server settings (seed, level-name, RCON password) are now global,
        # set once in server.properties, not patched per-profile.
        # Profiles only affect pipeline configuration.
        self._server_bar.set_active_profile(profile_name)

    @Slot()
    def _on_new_profile(self) -> None:
        dlg = ProfileEditorDialog(parent=self)
        if dlg.exec():
            name = dlg.profile_name()
            self._load_profile(name)

    def _on_edit_profile(self, profile_name: str) -> None:
        dlg = ProfileEditorDialog(profile_name=profile_name, parent=self)
        if dlg.exec():
            # Reload profile data
            new_name = dlg.profile_name()
            if new_name != profile_name:
                # Name changed — add new, keep old registry
                self._load_profile(new_name)
            else:
                self._profiles[profile_name] = load_profile(profile_name)
                # Reconcile the registry so statuses reflect on-disk outputs
                self._registries[profile_name].reconcile_with_profile(self._profiles[profile_name])
                # Re-apply DAG in case the ``dag:`` section changed
                data = self._profiles[profile_name]
                dag = ProfileDag.from_profile_dict(data)
                steps = dag.resolve_steps() if dag is not None else None
                self._sync_epoch_targets(self._registries[profile_name], data, steps)
                self._dashboard.update_profile_steps(profile_name, steps)
                self._dashboard.refresh_profile(profile_name)

    @Slot(str)
    def _on_delete_profile(self, profile_name: str, reason: str = "delete") -> None:
        # The dashboard currently only emits the profile name. The optional
        # *reason* is here for future compatibility if we want to support
        # `archive` or other actions.
        if reason == "delete":
            delete_profile(profile_name)
            self._dashboard.remove_profile(profile_name)
            self._profiles.pop(profile_name, None)
            self._registries.pop(profile_name, None)
        elif reason == "archive":
            # TODO: archive profiles to a (versioned) history folder
            pass

    @Slot(str, str)
    def _on_node_clicked(self, profile_name: str, step_id: str) -> None:
        if step_id == "new_profile":
            self._on_new_profile()
            return

        if step_id == "cancel":
            self._queue_clear()
            return

        if step_id == "run":
            self._on_run_server_session(profile_name)
            return

        # Open details and run the clicked step directly (rather than silently
        # queuing it) so the user sees status/log output immediately.
        self._on_details_clicked(profile_name)
        if hasattr(self, "_detail") and self._detail:
            self._detail.run_step(step_id)

    @Slot(str, str)
    def _on_node_run_from(self, profile_name: str, step_id: str) -> None:
        """Handle dashboard context-menu 'Run From Here' actions."""
        self._on_details_clicked(profile_name)
        if hasattr(self, "_detail") and self._detail:
            self._detail.run_from_step(step_id)

    @Slot(str, str)
    def _on_node_cancel(self, profile_name: str, step_id: str) -> None:
        """Handle dashboard context-menu cancel actions."""
        self._queue_clear()
        self._on_details_clicked(profile_name)
        if hasattr(self, "_detail") and self._detail:
            self._detail.cancel()

    @Slot(str, str)
    def _on_continue_training(self, profile_name: str, step_id: str) -> None:
        """Handle 'Continue training...' context-menu action."""
        profile_dict = self._profiles.get(profile_name)
        if not profile_dict:
            return

        preselected_levels: list[int] | None = None
        if step_id.startswith("train_voxy_l"):
            try:
                preselected_levels = [int(step_id.removeprefix("train_voxy_l"))]
            except ValueError:
                preselected_levels = None

        dlg = ContinueTrainingDialog(
            profile_dict,
            preselected_levels=preselected_levels,
            parent=self,
        )
        if not dlg.exec():
            return

        levels = dlg.selected_levels()
        additional = dlg.additional_epochs()
        if not levels:
            return

        # Shallow-copy so we don't mutate the stored profile
        augmented = dict(profile_dict)
        augmented["_continue_levels"] = levels
        augmented["_continue_epochs"] = additional

        self._on_details_clicked(profile_name)
        if hasattr(self, "_detail") and self._detail:
            self._detail.run_step_with_profile("continue_train_voxy", augmented)

    def _queue_append(self, profile_name: str, step_id: str) -> None:
        if step_id not in {"cancel", "run"}:
            self._step_queue.append((profile_name, step_id))
            self._dashboard.set_step_queue(self._step_queue)

    def _queue_clear(self) -> None:
        self._step_queue.clear()
        self._dashboard.set_step_queue(self._step_queue)

    @Slot(str)
    def _on_server_log(self, line: str) -> None:
        # Display server log output where the user will notice it.
        # Primary target is the detail panel (when open), otherwise show a
        # short snippet in the status bar and print to stdout.
        if hasattr(self, "_detail") and self._detail.isVisible():
            self._detail.append_log(line)
        else:
            print(line, flush=True)

        if hasattr(self, "_server_bar"):
            self._server_bar.append_log(line)

        # Show critical errors to the user even if the log panel isn't visible.
        if "[Server] ERROR" in line or "ERROR:" in line:
            QMessageBox.critical(self, "Server Error", line)

    @Slot(str)
    def _on_server_status_changed(self, status: str) -> None:
        self._dashboard.set_server_status(status)

    def _on_run_server_session(self, profile_name: str) -> None:
        if profile_name not in self._profiles:
            return

        # Configure the server for the world/role selected in the status bar,
        # then start.  The role is a server-level setting (shared across all
        # profiles), not a per-profile one.
        self._server.configure_for_role(self._server_bar.selected_role)
        self._server.start()

        # Load the detail panel for this profile (so we can run steps).
        registry = self._registries.get(profile_name)
        if registry:
            self._detail.load_profile(profile_name, registry)

        # Build the list of steps to run.  If the user hasn't queued any, run the
        # full required server session.
        if not self._step_queue:
            self._step_queue.extend((profile_name, step_id) for step_id in _SERVER_SESSION_STEPS)

        # Keep only steps for this profile (ignore others) and flatten to ids.
        self._server_session_queue = [
            step_id for p, step_id in self._step_queue if p == profile_name
        ]
        self._step_queue.clear()
        self._dashboard.set_step_queue(self._step_queue)

        self._server_session_active = bool(self._server_session_queue)
        self._run_next_server_session_step()

    def _run_next_server_session_step(self) -> None:
        if not self._server_session_queue:
            self._server_session_active = False
            self._server_session_current = None
            return

        next_step = self._server_session_queue.pop(0)
        self._server_session_current = next_step
        self._detail.run_from_step(next_step)

    def on_step_finished(
        self,
        profile_name: str | None,
        step_id: str,
        exit_code: int = 0,
        training_summary: dict[str, Any] | None = None,
    ) -> None:
        """Called by DetailPanel when a step completes."""
        if exit_code == 0 and training_summary:
            history = training_summary.get("history")
            metric_key = training_summary.get("metric_key")
            popup = TrainingResultsPopup(
                title=training_summary["title"],
                summary_text=training_summary["text"],
                history=history,
                metric_key=metric_key,
                parent=self,
            )
            popup.exec()

        if not self._server_session_active:
            return
        if step_id != self._server_session_current:
            return

        # Continue to the next queued step.
        self._run_next_server_session_step()

    def closeEvent(self, event) -> None:
        # Ensure the Fabric server is stopped cleanly when the GUI exits.
        self._server.stop()
        event.accept()
