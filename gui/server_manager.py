"""server_manager.py — Lifecycle management for the Fabric Minecraft server.

Manages the server JVM subprocess using Qt's QProcess for proper signal-safe
stdout/stderr capture.  Detects startup completion via RCON port polling
(no log-line parsing needed — works regardless of RCON credential changes).

Usage
-----
    mgr = ServerManager(rcon_host="localhost", rcon_port=25575)
    mgr.log_line.connect(print)
    mgr.status_changed.connect(lambda s: print("Status:", s))
    mgr.start()
    # ... later ...
    mgr.stop()
"""

from __future__ import annotations

import socket
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Slot

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "fabric-server"
_RUNTIME_DIR = _TOOLS_DIR / "runtime"

# Locate the server JAR in tools/fabric-server (not inside runtime/)
_JAR_CANDIDATES = list(_TOOLS_DIR.glob("*.jar"))
_JAR_PATH: Path | None = _JAR_CANDIDATES[0] if _JAR_CANDIDATES else None

# Grace period (ms) after /stop before we force-kill
_STOP_GRACE_MS = 20_000
# How often to poll RCON during startup (ms)
_STARTUP_POLL_MS = 2_000
# Max startup wait before giving up (ms)
_STARTUP_TIMEOUT_MS = 120_000


def _rcon_ping(host: str, port: int) -> bool:
    """Return True if something is accepting TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


class ServerManager(QObject):
    """Manages the Fabric server subprocess.

    States
    ------
    "stopped"   — not running
    "starting"  — process launched, waiting for RCON to become available
    "running"   — RCON is accepting connections (server is ready)
    "stopping"  — /stop sent, waiting for process to exit

    Signals
    -------
    status_changed(str)  — new state string (see above)
    log_line(str)        — one line of server stdout/stderr
    """

    status_changed: Signal = Signal(str)
    log_line: Signal = Signal(str)

    def __init__(
        self,
        rcon_host: str = "localhost",
        rcon_port: int = 25575,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._rcon_host = rcon_host
        self._rcon_port = rcon_port
        self._rcon_password: str = ""
        self._status = "stopped"

        # QProcess handles stdout/stderr in a Qt-friendly way
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)

        # Polls RCON port during startup
        self._startup_poll = QTimer(self)
        self._startup_poll.setInterval(_STARTUP_POLL_MS)
        self._startup_poll.timeout.connect(self._check_rcon_ready)

        # Startup timeout — give up after 2 minutes
        self._startup_timeout = QTimer(self)
        self._startup_timeout.setSingleShot(True)
        self._startup_timeout.setInterval(_STARTUP_TIMEOUT_MS)
        self._startup_timeout.timeout.connect(self._on_startup_timeout)

        # Grace timer after /stop
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.setInterval(_STOP_GRACE_MS)
        self._stop_timer.timeout.connect(self._force_kill)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    def is_running(self) -> bool:
        return self._status == "running"

    def configure_rcon(self, host: str, port: int, password: str = "") -> None:
        """Update RCON connection details (call before start())."""
        self._rcon_host = host
        self._rcon_port = port
        self._rcon_password = password

    def ping(self) -> bool:
        """Check whether the RCON port is currently accepting connections."""
        return _rcon_ping(self._rcon_host, self._rcon_port)

    def start(self) -> None:
        """Launch the Fabric server subprocess."""
        if self._status != "stopped":
            self.log_line.emit("[Server] Already running or stopping — ignoring start request.")
            return
        if not _JAR_PATH or not _JAR_PATH.exists():
            self.log_line.emit(
                f"[Server] ERROR: Server JAR not found in {_TOOLS_DIR}\n"
                "Expected: tools/fabric-server/fabric-server-mc.*.jar"
            )
            return
        if not _RUNTIME_DIR.exists():
            self.log_line.emit(f"[Server] ERROR: Runtime directory not found: {_RUNTIME_DIR}")
            return

        self._set_status("starting")
        self._process.setWorkingDirectory(str(_RUNTIME_DIR))
        self._process.start("java", ["-jar", str(_JAR_PATH), "--nogui"])

        if not self._process.waitForStarted(5000):
            self._set_status("stopped")
            self.log_line.emit("[Server] ERROR: Failed to launch JVM (is Java installed?)")
            return

        self.log_line.emit(f"[Server] Launched: java -jar {_JAR_PATH.name} --nogui")
        self.log_line.emit(f"[Server] Runtime: {_RUNTIME_DIR}")
        self.log_line.emit("[Server] Waiting for RCON to become available…")

        self._startup_poll.start()
        self._startup_timeout.start()

    def stop(self) -> None:
        """Gracefully stop the server via RCON /stop, then force-kill if needed."""
        if self._status not in ("running", "starting"):
            return

        self._set_status("stopping")
        self._startup_poll.stop()
        self._startup_timeout.stop()

        # Try RCON /stop first
        try:
            from rcon import RconClient  # noqa: PLC0415

            with RconClient(
                self._rcon_host, self._rcon_port, self._rcon_password, timeout=3.0
            ) as rcon:
                rcon.command("stop")
            self.log_line.emit("[Server] Sent /stop via RCON — waiting for shutdown…")
        except Exception as exc:
            self.log_line.emit(f"[Server] RCON /stop failed ({exc}); waiting for process to exit…")

        self._stop_timer.start()

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_stdout(self) -> None:
        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            if line.strip():
                self.log_line.emit(line)

    @Slot()
    def _on_stderr(self) -> None:
        raw = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            if line.strip():
                self.log_line.emit(line)

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._startup_poll.stop()
        self._startup_timeout.stop()
        self._stop_timer.stop()
        self._set_status("stopped")
        self.log_line.emit(f"[Server] Process exited (code {exit_code})")

    @Slot()
    def _check_rcon_ready(self) -> None:
        if _rcon_ping(self._rcon_host, self._rcon_port):
            self._startup_poll.stop()
            self._startup_timeout.stop()
            self._set_status("running")
            self.log_line.emit("[Server] RCON is accepting connections — server is ready!")

    @Slot()
    def _on_startup_timeout(self) -> None:
        self._startup_poll.stop()
        self.log_line.emit(
            "[Server] WARNING: Startup timeout reached (2 min) but RCON is still not responding. "
            "Check server logs for errors. Server process is still running."
        )
        # Don't kill the process — it might still be loading.  Leave in "starting".

    @Slot()
    def _force_kill(self) -> None:
        self.log_line.emit("[Server] Grace period expired — force-killing server process.")
        self._process.kill()

    # ------------------------------------------------------------------

    def _set_status(self, status: str) -> None:
        if self._status != status:
            self._status = status
            self.status_changed.emit(status)
