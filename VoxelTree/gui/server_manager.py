"""server_manager.py — Lifecycle management for the Fabric Minecraft server.

Manages the server JVM subprocess using Qt's QProcess for proper signal-safe
stdout/stderr capture.  Detects startup completion via RCON port polling
(no log-line parsing needed — works regardless of RCON credential changes).

The single source of truth for RCON credentials (host, port, password) is
``tools/fabric-server/runtime/server.properties``.  Call
:meth:`ServerManager.configure_for_role` (or set ``server_role`` in the profile
YAML) **before** calling :meth:`ServerManager.start`; the manager will patch
``server.properties`` with the role's seed, level-name, and network ports.

Usage
-----
    mgr = ServerManager()
    mgr.log_line.connect(print)
    mgr.status_changed.connect(lambda s: print("Status:", s))
    mgr.configure_for_role("train")   # patches server.properties
    mgr.start()
    # ... later ...
    mgr.stop()
"""

from __future__ import annotations

import socket
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Slot

from VoxelTree.gui.server_config import get_role
from VoxelTree.utils.rcon import RconClient

# ---------------------------------------------------------------------------
# World-freeze gamerule commands sent automatically after server startup.
# Format: (rcon_command, human_description)
# ---------------------------------------------------------------------------
FREEZE_COMMANDS: list[tuple[str, str]] = [
    ("gamerule doFireTick false", "disable fire spread"),
    ("gamerule doMobSpawning false", "disable mob spawning"),
    ("gamerule doWeatherCycle false", "freeze weather"),
    ("gamerule doDaylightCycle false", "freeze time"),
    ("gamerule randomTickSpeed 0", "disable random ticks"),
    ("gamerule doEntityDrops false", "disable entity drops"),
    ("gamerule doTileDrops false", "disable tile drops"),
    ("gamerule doMobLoot false", "disable mob loot"),
    ("difficulty peaceful", "set peaceful difficulty"),
    ("time set 6000", "set noon"),
    ("weather clear", "clear weather"),
]

UNFREEZE_COMMANDS: list[tuple[str, str]] = [
    ("gamerule doFireTick true", "restore fire spread"),
    ("gamerule doMobSpawning true", "restore mob spawning"),
    ("gamerule doWeatherCycle true", "restore weather cycle"),
    ("gamerule doDaylightCycle true", "restore day/night cycle"),
    ("gamerule randomTickSpeed 3", "restore random ticks"),
    ("gamerule doEntityDrops true", "restore entity drops"),
    ("gamerule doTileDrops true", "restore tile drops"),
    ("gamerule doMobLoot true", "restore mob loot"),
]


def _find_fabric_tools_dir() -> Path:
    """Find the repository's tools/fabric-server directory.

    The GUI may be executed from an installed package or directly from the repo.
    Walk up the parent chain looking for a `tools/fabric-server` directory and
    return the first match.
    """

    cur = Path(__file__).resolve()
    for _ in range(6):
        candidate = cur.parent / "tools" / "fabric-server"
        if candidate.exists():
            return candidate
        cur = cur.parent

    # Fallback (best-effort): assume repo root is two levels up.
    return Path(__file__).resolve().parents[2] / "tools" / "fabric-server"


_TOOLS_DIR = _find_fabric_tools_dir()
_RUNTIME_DIR = _TOOLS_DIR / "runtime"
_SERVER_PROPERTIES = _RUNTIME_DIR / "server.properties"

# Locate the server JAR in tools/fabric-server (not inside runtime/)
_JAR_CANDIDATES = list(_TOOLS_DIR.glob("*.jar"))
_JAR_PATH: Path | None = _JAR_CANDIDATES[0] if _JAR_CANDIDATES else None

# Grace period (ms) after /stop before we force-kill
_STOP_GRACE_MS = 20_000
# How often to poll RCON during startup (ms)
_STARTUP_POLL_MS = 2_000
# Max startup wait before giving up (ms)
_STARTUP_TIMEOUT_MS = 120_000

# Default RCON password baked into server.properties when no profile overrides it
_DEFAULT_RCON_PASSWORD = "voxeltree"


def _rcon_ping(host: str, port: int) -> bool:
    """Return True if something is accepting TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


# ------------------------------------------------------------------
# server.properties helpers
# ------------------------------------------------------------------


def read_server_property(key: str, default: str = "") -> str:
    """Read a single property from the runtime server.properties file."""
    try:
        text = _SERVER_PROPERTIES.read_text(encoding="utf-8")
    except Exception:
        return default

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return default


def _patch_server_properties(patches: dict[str, str]) -> None:
    """Update specific key=value pairs in server.properties, preserving comments/order.

    Keys not already present are appended at the end.
    """
    try:
        text = _SERVER_PROPERTIES.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""

    remaining = dict(patches)
    new_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}")
                continue
        new_lines.append(line)

    # Append any keys that weren't found in the original file
    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")

    _SERVER_PROPERTIES.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def get_rcon_settings() -> dict[str, str | int]:
    """Return the current RCON settings from server.properties.

    This is the canonical way for step command factories to obtain RCON
    credentials — profiles no longer store them.

    Returns
    -------
    dict with keys ``host``, ``port``, ``password``.
    """
    return {
        "host": read_server_property("server-ip", "localhost") or "localhost",
        "port": int(read_server_property("rcon.port", "25575")),
        "password": read_server_property("rcon.password", _DEFAULT_RCON_PASSWORD),
    }


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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status = "stopped"
        self._active_role: str | None = None

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

    @property
    def active_role(self) -> str | None:
        """The server role currently configured in server.properties."""
        return self._active_role

    def configure_for_role(self, role_name: str | None = None) -> None:
        """Patch ``server.properties`` for the given role before :meth:`start`.

        Reads the matching :class:`~VoxelTree.gui.server_config.ServerRole` from
        ``servers.yaml`` and writes ``level-name``, ``level-seed``,
        ``server-port``, ``rcon.port``, ``rcon.password``, and
        ``enable-rcon=true`` into ``server.properties``.

        Parameters
        ----------
        role_name:
            One of the role keys in ``servers.yaml`` (e.g. ``"train"``,
            ``"validate"``).  Defaults to ``"train"``.
        """
        role = get_role(role_name)
        self._active_role = role.name
        _patch_server_properties(
            {
                "level-name": role.level_name,
                "level-seed": str(role.seed),
                "server-port": str(role.server_port),
                "rcon.port": str(role.rcon_port),
                "rcon.password": role.rcon_password,
                "enable-rcon": "true",
            }
        )
        self.log_line.emit(
            f"[Server] Role '{role.name}': level={role.level_name!r}  "
            f"seed={role.seed}  port={role.server_port}  rcon={role.rcon_port}"
        )

    def configure_for_profile(self, profile: dict) -> None:
        """Configure the server from the ``server_role`` key in a profile dict.

        Reads ``profile.get("server_role")`` and delegates to
        :meth:`configure_for_role`.  Falls back to ``"train"`` when the key is
        absent.
        """
        if not isinstance(profile, dict):
            raise TypeError(f"Expected dict profile, got {type(profile)}")
        role_name = profile.get("server_role")
        self.configure_for_role(role_name)

    # Legacy shim — old callers may still call configure_rcon(); this now
    # reads/writes server.properties instead of storing state in-memory.
    def configure_rcon(self, host: str, port: int, password: str = "") -> None:  # noqa: D401
        """Deprecated — prefer :meth:`configure_for_profile`.

        Kept for backward compatibility.  Updates server.properties directly.
        """
        patches: dict[str, str] = {}
        if password:
            patches["rcon.password"] = password
        if port != 25575:
            patches["rcon.port"] = str(port)
        if patches:
            _patch_server_properties(patches)

    def ping(self) -> bool:
        """Check whether the RCON port is currently accepting connections."""
        rcon = get_rcon_settings()
        return _rcon_ping(str(rcon["host"]), int(rcon["port"]))

    def start(self) -> None:
        """Launch the Fabric server subprocess."""
        if self._status != "stopped":
            self.log_line.emit("[Server] Already running or stopping — ignoring start request.")
            return
        if not _JAR_PATH or not _JAR_PATH.exists():
            self.log_line.emit(
                f"[Server] ERROR: Server JAR not found in {_TOOLS_DIR}\n"
                "Expected: tools/fabric-server/fabric-server-mc.*.jar\n"
                "Tip: run the Fabric installer (or ensure the jar is checked out/copied into tools/fabric-server)."
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

        # Always read credentials from server.properties — the single source of truth.
        rcon = get_rcon_settings()
        host = str(rcon["host"])
        port = int(rcon["port"])
        password = str(rcon["password"])

        try:
            from VoxelTree.utils.rcon import RconClient  # noqa: PLC0415

            with RconClient(host, port, password, timeout=3.0) as rc:
                rc.command("stop")
            self.log_line.emit("[Server] Sent /stop via RCON — waiting for shutdown…")
        except Exception as exc:
            self.log_line.emit(f"[Server] RCON /stop failed ({exc}); waiting for process to exit…")

        self._stop_timer.start()

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def stop_server(self) -> None:
        """Legacy alias for :meth:`stop` kept for backwards compatibility."""
        self.stop()

    def start_server(self) -> None:
        """Legacy alias for :meth:`start` kept for backwards compatibility."""
        self.start()

    def start_session(self, profile_name: str, steps: list[tuple[str, str]]) -> None:
        """Start a 'server session'.

        This method is invoked by the GUI but the session orchestration is
        currently handled elsewhere; this implementation ensures the server is
        running and avoids AttributeError crashes.
        """
        # Ensure the server is started before attempting any profile-specific
        # actions. The GUI will drive step execution via other components.
        self.start()

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_stdout(self) -> None:
        raw_bytes = self._process.readAllStandardOutput().data()
        if isinstance(raw_bytes, memoryview):
            raw = bytes(raw_bytes).decode("utf-8", errors="replace")
        elif isinstance(raw_bytes, (bytes, bytearray)):
            raw = raw_bytes.decode("utf-8", errors="replace")
        else:
            raw = str(raw_bytes)
        for line in raw.splitlines():
            if line.strip():
                self.log_line.emit(line)

    @Slot()
    def _on_stderr(self) -> None:
        raw_bytes = self._process.readAllStandardError().data()
        if isinstance(raw_bytes, memoryview):
            raw = bytes(raw_bytes).decode("utf-8", errors="replace")
        elif isinstance(raw_bytes, (bytes, bytearray)):
            raw = raw_bytes.decode("utf-8", errors="replace")
        else:
            raw = str(raw_bytes)
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
        rcon = get_rcon_settings()
        if _rcon_ping(str(rcon["host"]), int(rcon["port"])):
            self._startup_poll.stop()
            self._startup_timeout.stop()
            self._set_status("running")
            self.log_line.emit("[Server] RCON is accepting connections — server is ready!")
            self._auto_freeze(rcon)

    def _auto_freeze(self, rcon: dict) -> None:
        """Send freeze commands immediately after server becomes ready."""
        self.log_line.emit("[Server] Auto-freezing world state...")
        try:
            with RconClient(str(rcon["host"]), int(rcon["port"]), str(rcon["password"])) as client:
                for cmd, desc in FREEZE_COMMANDS:
                    client.command(cmd.lstrip("/"))
                    self.log_line.emit(f"[Server]   ✓ {desc}")
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(f"[Server] WARNING: auto-freeze failed: {exc}")
        else:
            self.log_line.emit("[Server] World is frozen and ready for data capture.")

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
