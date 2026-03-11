"""server_status_bar.py — Toolbar widget for Fabric server lifecycle control.

Shows:
  ● Server: Running / Starting... / Stopped     [Start]  [Stop]   |  [▶ Run Server Steps – <profile>]

The "Run Server Steps" button triggers a server session for the currently
active profile, running whichever of pregen → voxy_import / dumpnoise
haven't succeeded yet.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.server_manager import ServerManager

_STATUS_COLORS = {
    "stopped": "#888888",
    "starting": "#e8a800",
    "running": "#28a745",
    "stopping": "#e8a800",
}

_STATUS_LABELS = {
    "stopped": "Server: Stopped",
    "starting": "Server: Starting…",
    "running": "Server: Running",
    "stopping": "Server: Stopping…",
}


class ServerStatusBar(QWidget):
    """Horizontal toolbar row for server lifecycle management.

    Signals
    -------
    run_server_session_requested(str)
        Emitted when the user clicks "Run Server Steps" with a valid active
        profile.  Carries the profile_name.
    """

    run_server_session_requested: Signal = Signal(str)

    def __init__(self, server_manager: ServerManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._server = server_manager
        self._active_profile: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        # ── Status dot ──
        self._dot = QLabel("●")
        self._dot.setFixedWidth(14)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ── Status text ──
        self._status_lbl = QLabel("Server: Stopped")

        sep1 = _Sep()

        # ── Start / Stop buttons ──
        self._start_btn = QPushButton("Start Server")
        self._start_btn.setFixedHeight(24)
        self._start_btn.setStyleSheet(
            "QPushButton { background:#2e6e36; color:white; border-radius:4px; padding:0 10px; }"
            "QPushButton:hover { background:#28a745; }"
            "QPushButton:disabled { background:#2a2a2a; color:#555; }"
        )

        self._stop_btn = QPushButton("Stop Server")
        self._stop_btn.setFixedHeight(24)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background:#6e2e2e; color:white; border-radius:4px; padding:0 10px; }"
            "QPushButton:hover { background:#dc3545; }"
            "QPushButton:disabled { background:#2a2a2a; color:#555; }"
        )

        sep2 = _Sep()

        # ── Server session button ──
        self._session_btn = QPushButton("Run Server Steps")
        self._session_btn.setFixedHeight(24)
        self._session_btn.setEnabled(False)
        self._session_btn.setToolTip(
            "Run the server-required pipeline steps for the active profile.\n\n"
            "Order: pregen → voxy_import → dumpnoise\n"
            "Already-completed steps are skipped.\n\n"
            "Click a profile row's Details button or a step node to set the active profile."
        )
        self._session_btn.setStyleSheet(
            "QPushButton { background:#2e4e8e; color:white; border-radius:4px; padding:0 10px; }"
            "QPushButton:hover { background:#4a7fd4; }"
            "QPushButton:disabled { background:#2a2a2a; color:#555; }"
        )

        for w in (
            self._dot,
            self._status_lbl,
            sep1,
            self._start_btn,
            self._stop_btn,
            sep2,
            self._session_btn,
        ):
            layout.addWidget(w)
        layout.addStretch()

        self.setStyleSheet("background:#1e1e1e; border-bottom:1px solid #333;")
        self.setFixedHeight(36)

        # ── Connections ──
        self._start_btn.clicked.connect(self._server.start)
        self._stop_btn.clicked.connect(self._server.stop)
        self._session_btn.clicked.connect(self._on_session_clicked)
        self._server.status_changed.connect(self._on_status_changed)

        self._on_status_changed("stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_profile(self, profile_name: str | None) -> None:
        """Update the active profile (shown on the session button)."""
        self._active_profile = profile_name
        self._refresh_session_btn()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        color = _STATUS_COLORS.get(status, "#888888")
        self._dot.setStyleSheet(f"color:{color}; font-size:14px;")
        self._status_lbl.setText(_STATUS_LABELS.get(status, status))
        self._status_lbl.setStyleSheet(f"color:{color};")

        is_stopped = status == "stopped"
        is_running = status == "running"
        self._start_btn.setEnabled(is_stopped)
        self._stop_btn.setEnabled(is_running or status == "starting")
        self._refresh_session_btn()

    @Slot()
    def _on_session_clicked(self) -> None:
        if self._active_profile:
            self.run_server_session_requested.emit(self._active_profile)

    # ------------------------------------------------------------------

    def _refresh_session_btn(self) -> None:
        can_run = self._server.is_running() and bool(self._active_profile)
        self._session_btn.setEnabled(can_run)
        if self._active_profile:
            self._session_btn.setText(f"▶ Run Server Steps  [{self._active_profile}]")
        else:
            self._session_btn.setText("Run Server Steps  (select a profile)")


class _Sep(QLabel):
    """Thin vertical separator."""

    def __init__(self) -> None:
        super().__init__("|")
        self.setStyleSheet("color:#444444; padding:0 2px;")
