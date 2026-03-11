"""main_window.py — Top-level application window."""

from __future__ import annotations

from pathlib import Path

import yaml  # noqa: F401  (reserved for future profile migrations)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QMainWindow

from gui.dashboard_table import DashboardTable
from gui.detail_panel import DetailPanel
from gui.profile_editor import (
    ProfileDeleteDialog,
    ProfileEditorDialog,
    delete_profile_data,
    list_profiles,
    load_profile,
)
from gui.run_registry import RunRegistry

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


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
        self.resize(1100, 420)
        print("[MW.init.3] Stylesheet...", flush=True)
        self.setStyleSheet(
            "QMainWindow { background: #1a1a1a; }" "QMenuBar { background: #252525; color: #ccc; }"
        )

        print("[MW.init.4] Cache dicts...", flush=True)
        # Registry cache: profile_name → RunRegistry
        self._registries: dict[str, RunRegistry] = {}
        # Profile dict cache: profile_name → loaded YAML dict
        self._profiles: dict[str, dict] = {}

        print("[MW.init.5] Creating DashboardTable...", flush=True)
        # ── Widgets ──
        self._dashboard = DashboardTable()
        print("[MW.init.6] Connecting dashboard signals...", flush=True)
        self._dashboard.details_clicked.connect(self._on_details_clicked)
        self._dashboard.delete_profile_requested.connect(self._on_delete_profile)
        self._dashboard.new_profile_requested.connect(self._on_new_profile)
        print("[MW.init.7] Setting central widget...", flush=True)
        self.setCentralWidget(self._dashboard)

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
