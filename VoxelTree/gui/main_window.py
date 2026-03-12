"""main_window.py — Top-level application window."""

from __future__ import annotations

from pathlib import Path

import yaml  # noqa: F401  (reserved for future profile migrations)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from VoxelTree.gui.dashboard_table import DashboardTable
from VoxelTree.gui.detail_panel import DetailPanel
from VoxelTree.gui.profile_editor import (ProfileDeleteDialog,
                                          ProfileEditorDialog,
                                          delete_profile_data, list_profiles,
                                          load_profile)
from VoxelTree.gui.run_registry import RunRegistry
from VoxelTree.gui.server_manager import ServerManager
from VoxelTree.gui.server_status_bar import ServerStatusBar
from VoxelTree.gui.step_definitions import STEP_BY_ID

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

# Server-required steps in the order they should be queued for a session
_SERVER_SESSION_STEPS = ["pregen", "voxy_import", "dumpnoise"]


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
        self._profiles: dict[str, dict] = {}
        # Step queue for server session: list of (profile_name, step_id)
        self._step_queue: list[tuple[str, str]] = []

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
        for name in list_profiles():
            self._load_profile(name)

    def _load_profile(self, name: str) -> None:
        data = load_profile(name)
        if not data:
            return
        self._profiles[name] = data
        if name not in self._registries:
            self._registries[name] = RunRegistry(name)
        # reconcile state against on-disk data to clean up any stale failures
        try:
            self._registries[name].reconcile_with_profile(data)
        except Exception:
            # reconciliation should never crash the GUI; log and continue silently
            print(f"[MW] warning: failed to reconcile registry for profile {name}")
        self._dashboard.add_profile(name, self._registries[name])

    def get_profile_dict(self, profile_name: str) -> dict | None:
        """Called by DetailPanel to get CLI arguments for a step."""
        return self._profiles.get(profile_name)

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
            self._dashboard.refresh_profile(profile_name)

    @Slot(str, str)
    def _on_node_clicked(self, profile_name: str, step_id: str) -> None:
        """Run a step when its node is clicked."""
        self._server_bar.set_active_profile(profile_name)
        registry = self._registries.get(profile_name)
        if not registry or not registry.can_run(step_id):
            return

        # Check if the step requires a running server
        step_def = STEP_BY_ID.get(step_id)
        if step_def and step_def.server_required and not self._server.is_running():
            reply = QMessageBox.question(
                self,
                "Server Required",
                f"Step ‘{step_def.label}’ requires the Fabric server to be running.\n\n"
                "Start the server now? You can click the step again once it’s ready.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._configure_server_rcon(profile_name)
                self._server.start()
            return

        self._detail.load_profile(profile_name, registry)
        self._detail.run_step(step_id)

    @Slot(str, str)
    def _on_node_run_from(self, profile_name: str, step_id: str) -> None:
        """User requested **Run From Here** via context menu or detail panel."""
        self._server_bar.set_active_profile(profile_name)
        registry = self._registries.get(profile_name)
        if not registry:
            return
        # server requirement check is same as for single step
        step_def = STEP_BY_ID.get(step_id)
        if step_def and step_def.server_required and not self._server.is_running():
            reply = QMessageBox.question(
                self,
                "Server Required",
                f"Step ‘{step_def.label}’ requires the Fabric server to be running.\n\n"
                "Start the server now? You can click the step again once it’s ready.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._configure_server_rcon(profile_name)
                self._server.start()
            return

        self._detail.load_profile(profile_name, registry)
        self._detail.run_from_step(step_id)

    @Slot(str, str)
    def _on_node_cancel(self, profile_name: str, step_id: str) -> None:
        """Cancel running step(s) for the profile if the detail panel is focused."""
        self._server_bar.set_active_profile(profile_name)
        # only cancel if the detail panel is currently showing this profile
        if self._detail._profile_name == profile_name:
            self._detail.cancel()

    # ------------------------------------------------------------------
    # Server session
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_run_server_session(self, profile_name: str) -> None:
        """Queue all pending server steps for profile_name and run them."""
        registry = self._registries.get(profile_name)
        if not registry:
            return

        # Configure RCON from profile settings
        self._configure_server_rcon(profile_name)

        # Figure out which server steps still need to run.  A step that has
        # already succeeded but is now stale should also be treated as pending
        # because the user explicitly requested a fresh server session.
        pending = [
            step_id
            for step_id in _SERVER_SESSION_STEPS
            if registry.get_status(step_id) != "success" or registry.is_stale(step_id)
        ]

        if not pending:
            QMessageBox.information(
                self,
                "Server Session",
                f"All server steps are already complete for profile ‘{profile_name}’.\n\n"
                "Nothing to do.",
            )
            return

        self._step_queue = [(profile_name, step_id) for step_id in pending]
        self._run_next_queued_step()

    def _configure_server_rcon(self, profile_name: str) -> None:
        """Update ServerManager’s RCON connection details from profile data."""
        profile_data = self._profiles.get(profile_name, {})
        rcon = profile_data.get("rcon", {})
        self._server.configure_rcon(
            host=str(rcon.get("host", "localhost")),
            port=int(rcon.get("port", 25575)),
            password=str(rcon.get("password", "")),
        )

    def _run_next_queued_step(self) -> None:
        """Pop and execute the next step in _step_queue."""
        if not self._step_queue:
            return
        profile_name, step_id = self._step_queue[0]  # peek, not pop — pop on success
        registry = self._registries.get(profile_name)
        if not registry:
            self._step_queue.clear()
            return
        self._detail.load_profile(profile_name, registry)
        self._detail.run_step(step_id)

    @Slot(str)
    def _on_server_log(self, line: str) -> None:
        """Forward server log lines to the detail panel if it’s visible."""
        if self._detail.isVisible():
            self._detail.append_log(f"[Server] {line}")

    @Slot(str)
    def _on_server_status_changed(self, status: str) -> None:
        """React to server state transitions."""
        # Nothing extra needed here for now; server_bar handles its own display
        pass

    @Slot(str)
    def _on_delete_profile(self, profile_name: str) -> None:
        profile_data = self._profiles.get(profile_name, {})
        dlg = ProfileDeleteDialog(profile_name, profile_data, parent=self)
        if not dlg.exec():
            return

        delete_profile_data(dlg.selected_paths())

        # Remove from UI and caches
        self._dashboard.remove_profile(profile_name)
        self._profiles.pop(profile_name, None)
        self._registries.pop(profile_name, None)

        # Hide detail panel if it was showing this profile
        if self._detail.isVisible():
            self._detail.hide()

    # ------------------------------------------------------------------
    # Called by DetailPanel after a step finishes
    # ------------------------------------------------------------------

    def on_step_finished(self, profile_name: str | None, step_id: str) -> None:
        if profile_name:
            self._dashboard.refresh_profile(profile_name)

    def on_step_progress(self, profile_name: str | None, step_id: str) -> None:
        """Called by DetailPanel when a running step emits a progress update.

        The default behaviour is simply to refresh the corresponding dashboard
        row so that the progress ring updates in near‑real time.
        """
        if profile_name:
            self._dashboard.refresh_profile(profile_name)

        # Advance step queue (server session)
        if self._step_queue and profile_name:
            queued_profile, queued_step = self._step_queue[0]
            if queued_profile == profile_name and queued_step == step_id:
                registry = self._registries.get(profile_name)
                succeeded = registry and registry.get_status(step_id) == "success"
                self._step_queue.pop(0)
                if succeeded and self._step_queue:
                    self._run_next_queued_step()
                elif not succeeded:
                    # Abort remaining queue on failure
                    self._step_queue.clear()
                    self._detail.append_log(
                        f"[Session] Step ‘{step_id}’ failed — remaining session steps cancelled."
                    )
                else:
                    # Queue complete
                    self._detail.append_log("[Session] Server session complete.")
