#!/usr/bin/env python3
"""
LODiffusion Data CLI — freeze & pregen pipeline orchestrator.

Automates the Minecraft world freeze + Chunky chunk pregeneration pipeline
needed to produce clean, deterministic training data for LODiffusion.

Requires a Minecraft server running with RCON enabled, Carpet mod, and
Chunky installed.  Use --dry-run to print commands without sending them.

Usage examples
--------------
  # Print all commands that would be sent (no server needed):
  python data-cli.py freeze --dry-run
  python data-cli.py pregen --radius 2048 --dry-run

  # Execute against a running server (RCON must be enabled):
  # Full pipeline order:
  #   1) freeze → 2) pregen → 3) extract → 4) build-pairs → 5) python pipeline.py train

  # Steps 1-2 (RCON required):
  python data-cli.py pregen --host localhost --port 25575 --password secret \\
      --world minecraft:overworld --center 0 0 --radius 2048

  # Step 3 (no server needed, auto-detects Voxy DB):
  python data-cli.py extract
  python data-cli.py extract --saves-dir C:/path/to/saves  # explicit path
  python data-cli.py extract --clean                        # wipe existing NPZs first

  # Step 4 (no server needed, runs Mipper once over all chunks):
  python data-cli.py build-pairs
  python data-cli.py build-pairs --clean                    # force rebuild

  # Other RCON helpers:
  python data-cli.py freeze --host localhost --port 25575 --password secret
  python data-cli.py status --host localhost --port 25575 --password secret
  python data-cli.py unfreeze --host localhost --port 25575 --password secret

Enabling RCON in server.properties
-----------------------------------
  enable-rcon=true
  rcon.port=25575
  rcon.password=<your-password>
"""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Absolute path to THIS file's directory (VoxelTree root)
_HERE = Path(__file__).resolve().parent
# Default Minecraft saves folder (relative to the standard MC/ repo layout)
_DEFAULT_SAVES_DIR = _HERE.parent / "LODiffusion" / "run" / "saves"

# ---------------------------------------------------------------------------
# Minimal RCON client (no external dependencies)
# ---------------------------------------------------------------------------

RCON_LOGIN = 3
RCON_COMMAND = 2
RCON_RESPONSE = 2
_MAX_PAYLOAD = 4096


class RconError(Exception):
    pass


class RconClient:
    """Thread-unsafe, synchronous RCON client over TCP."""

    def __init__(self, host: str, port: int, password: str, timeout: float = 10.0) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._req_id = 1

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection((self._host, self._port), self._timeout)
        except OSError as exc:
            raise RconError(
                f"Cannot connect to {self._host}:{self._port} — {exc}\n"
                "Make sure the server is running with enable-rcon=true in server.properties."
            ) from exc
        self._send_packet(RCON_LOGIN, self._password)
        resp_id, _, _ = self._recv_packet()
        if resp_id == -1:
            raise RconError("RCON authentication failed — wrong password?")

    def command(self, cmd: str) -> str:
        if self._sock is None:
            raise RconError("Not connected — call connect() first.")
        self._send_packet(RCON_COMMAND, cmd)
        _, _, payload = self._recv_packet()
        return payload

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    def _send_packet(self, ptype: int, payload: str) -> None:
        assert self._sock is not None
        data = payload.encode("utf-8") + b"\x00\x00"
        header = struct.pack("<iii", len(data) + 8, self._req_id, ptype)
        self._sock.sendall(header + data)
        self._req_id += 1

    def _recv_packet(self) -> tuple[int, int, str]:
        raw_len = self._recv_exact(4)
        (pkt_len,) = struct.unpack("<i", raw_len)
        body = self._recv_exact(pkt_len)
        req_id, ptype = struct.unpack("<ii", body[:8])
        payload = body[8:-2].decode("utf-8", errors="replace")
        return req_id, ptype, payload

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise RconError("Connection closed by server.")
            buf += chunk
        return buf

    def __enter__(self) -> "RconClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    host: str = "localhost"
    port: int = 25575
    password: str = ""
    dry_run: bool = False
    world: str = "minecraft:overworld"
    center_x: int = 0
    center_z: int = 0
    radius: int = 2048
    shape: str = "square"
    verbose: bool = False
    saves_dir: Path = field(default_factory=lambda: _DEFAULT_SAVES_DIR)
    clean: bool = False


# Gamerule commands that freeze the world for deterministic data collection.
FREEZE_COMMANDS: list[tuple[str, str]] = [
    (
        "/carpet randomTickSpeed 0",
        "Disable random block ticks (crops, fire spread, etc.)",
    ),
    ("/gamerule doFireTick false", "Prevent fire spread"),
    ("/gamerule mobGriefing false", "Prevent mobs modifying terrain"),
    ("/gamerule doMobSpawning false", "Disable mob spawning during pregen"),
    ("/gamerule doDaylightCycle false", "Freeze time of day"),
    ("/gamerule doWeatherCycle false", "Freeze weather"),
    ("/gamerule randomTickSpeed 0", "Vanilla random tick speed override"),
]

UNFREEZE_COMMANDS: list[tuple[str, str]] = [
    ("/carpet randomTickSpeed 3", "Restore random tick speed"),
    ("/gamerule doFireTick true", "Re-enable fire spread"),
    ("/gamerule mobGriefing true", "Re-enable mob griefing"),
    ("/gamerule doMobSpawning true", "Re-enable mob spawning"),
    ("/gamerule doDaylightCycle true", "Unfreeze day/night cycle"),
    ("/gamerule doWeatherCycle true", "Unfreeze weather"),
    ("/gamerule randomTickSpeed 3", "Restore vanilla random tick speed"),
]


def _run_commands(
    commands: list[tuple[str, str]],
    cfg: PipelineConfig,
    section: str,
) -> None:
    """Send a list of (command, description) pairs, or print them in dry-run mode."""

    print(f"\n{'='*60}")
    print(f"  {section}")
    print(f"{'='*60}")

    if cfg.dry_run:
        print("  [DRY-RUN] Commands that would be sent:\n")
        for cmd, desc in commands:
            print(f"    {cmd:50s}  # {desc}")
        print()
        return

    if not cfg.password:
        print("ERROR: --password required (or use --dry-run to preview commands).")
        sys.exit(1)

    with RconClient(cfg.host, cfg.port, cfg.password) as rcon:
        print(f"  Connected to {cfg.host}:{cfg.port}\n")
        for cmd, desc in commands:
            # Strip leading slash — RCON commands don't need it on some servers,
            # but Minecraft's RCON accepts both forms.
            response = rcon.command(cmd.lstrip("/"))
            status = response.strip() or "(no response)"
            if cfg.verbose:
                print(f"  > {cmd}")
                print(f"    {status}")
            else:
                tick = "✓" if status != "(no response)" or "Error" not in status else "✗"
                print(f"  {tick}  {desc}")


def build_pregen_commands(cfg: PipelineConfig) -> list[tuple[str, str]]:
    """Build the Chunky command sequence for pregeneration."""
    return [
        (f"/chunky world {cfg.world}", f"Target world: {cfg.world}"),
        (
            f"/chunky center {cfg.center_x} {cfg.center_z}",
            f"Center: ({cfg.center_x}, {cfg.center_z})",
        ),
        (f"/chunky radius {cfg.radius}", f"Radius: {cfg.radius} blocks"),
        (f"/chunky shape {cfg.shape}", f"Shape: {cfg.shape}"),
        ("/chunky start", "Start pregeneration"),
    ]


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_freeze(cfg: PipelineConfig) -> None:
    """Apply world-freeze gamerules to stop world state from evolving."""
    _run_commands(FREEZE_COMMANDS, cfg, "Freeze — locking world state")
    if not cfg.dry_run:
        print("  World is now frozen. Run 'pregen' to start chunk generation.")


def cmd_unfreeze(cfg: PipelineConfig) -> None:
    """Restore default gamerules after data collection is complete."""
    _run_commands(UNFREEZE_COMMANDS, cfg, "Unfreeze — restoring world state")
    if not cfg.dry_run:
        print("  World gamerules restored to defaults.")


def cmd_pregen(cfg: PipelineConfig) -> None:
    """Configure Chunky and start pregeneration. Optionally polls for completion."""
    # 1. Freeze first
    print("\nStep 1: Freeze world state...")
    _run_commands(FREEZE_COMMANDS, cfg, "Freeze")

    # 2. Configure and start Chunky
    print("\nStep 2: Configure and start Chunky...")
    pregen_cmds = build_pregen_commands(cfg)
    _run_commands(pregen_cmds, cfg, "Chunky Pregen")

    if cfg.dry_run:
        return

    # 3. Poll Chunky for progress
    print("\nStep 3: Monitoring progress (Ctrl-C to stop polling)...")
    try:
        with RconClient(cfg.host, cfg.port, cfg.password) as rcon:
            while True:
                resp = rcon.command("chunky progress")
                if resp:
                    print(f"  [{time.strftime('%H:%M:%S')}] {resp.strip()}")
                    if "complete" in resp.lower() or "done" in resp.lower() or "100%" in resp:
                        print("\n  Pregeneration complete!")
                        break
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Polling stopped. Chunky continues running in the background.")
        print("  Run 'status' to check progress, or 'cancel' in-game via /chunky cancel.")
    if not cfg.dry_run:
        print(
            "\n  Next step: extract Voxy data into training NPZ files.\n"
            f'    python data-cli.py extract --saves-dir "{cfg.saves_dir}"'
        )


def cmd_status(cfg: PipelineConfig) -> None:
    """Query Chunky pregeneration progress."""
    if cfg.dry_run:
        print("\n  [DRY-RUN] Would send: chunky progress")
        return

    if not cfg.password:
        print("ERROR: --password required.")
        sys.exit(1)

    with RconClient(cfg.host, cfg.port, cfg.password) as rcon:
        resp = rcon.command("chunky progress")
        print(f"\n  Chunky status: {resp.strip() or '(no response)'}")


def cmd_info(cfg: PipelineConfig) -> None:  # noqa: ARG001
    """Print the full pipeline plan without connecting."""
    print(
        textwrap.dedent(
            f"""
        LODiffusion Freeze + Pregen Pipeline
        =====================================
        World       : {cfg.world}
        Center      : ({cfg.center_x}, {cfg.center_z})
        Radius      : {cfg.radius} blocks
        Shape       : {cfg.shape}

        Estimated chunks: ~{int((cfg.radius / 16) ** 2 * 3.14159):,}
            (for a circular radius; square has ~{int((cfg.radius * 2 / 16) ** 2):,})

        Step 1 — freeze ({len(FREEZE_COMMANDS)} gamerule commands)
        Step 2 — Chunky pregen ({len(build_pregen_commands(cfg))} Chunky commands)
        Step 3 — poll for completion

        Run with --dry-run to preview all commands.
        Run with --host / --port / --password to execute against a live server.
    """
        ).strip()
    )


def cmd_build_pairs(cfg: PipelineConfig) -> None:
    """Pre-compute LOD pair caches (Phase 1b) via pipeline.py build-pairs."""
    pipeline_py = _HERE / "pipeline.py"
    if not pipeline_py.exists():
        print(f"ERROR: pipeline.py not found at {pipeline_py}")
        sys.exit(1)

    cmd: list[str] = [sys.executable, str(pipeline_py), "build-pairs"]
    if cfg.clean:
        cmd.append("--clean")

    if cfg.dry_run:
        print("\n  [DRY-RUN] Would run:")
        print("   ", " ".join(cmd))
        return

    print("\nBuilding LOD training pair caches ...")
    result = subprocess.run(cmd, cwd=str(_HERE))
    if result.returncode != 0:
        print(f"\nERROR: build-pairs failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print("\n  Pair cache written to data/voxy/")
    print("  Next step: python pipeline.py train --epochs 20")


def cmd_extract(cfg: PipelineConfig) -> None:
    """Extract Voxy RocksDB training data → NPZ files via pipeline.py extract."""
    saves_dir = cfg.saves_dir
    if not saves_dir.exists():
        print(f"ERROR: saves directory not found: {saves_dir}")
        print("  Use --saves-dir to specify the path explicitly.")
        sys.exit(1)

    pipeline_py = _HERE / "pipeline.py"
    if not pipeline_py.exists():
        print(f"ERROR: pipeline.py not found at {pipeline_py}")
        sys.exit(1)

    cmd: list[str] = [sys.executable, str(pipeline_py), "extract", "--voxy-dir", str(saves_dir)]
    if cfg.clean:
        cmd.append("--clean")

    if cfg.dry_run:
        print("\n  [DRY-RUN] Would run:")
        print("   ", " ".join(cmd))
        return

    print(f"\nExtracting Voxy training data from {saves_dir} ...")
    result = subprocess.run(cmd, cwd=str(_HERE))
    if result.returncode != 0:
        print(f"\nERROR: extraction failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print("\n  Extraction complete. Training data written to data/voxy/")
    print("  Next step: python data-cli.py build-pairs")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--host",
        default="localhost",
        metavar="HOST",
        help="RCON host (default: localhost)",
    )
    shared.add_argument(
        "--port",
        type=int,
        default=25575,
        metavar="PORT",
        help="RCON port (default: 25575)",
    )
    shared.add_argument("--password", default="", metavar="PASS", help="RCON password")
    shared.add_argument(
        "--dry-run", action="store_true", help="Print commands instead of sending them"
    )
    shared.add_argument("--verbose", action="store_true", help="Show raw RCON responses")

    pregen_args = argparse.ArgumentParser(add_help=False)
    pregen_args.add_argument(
        "--world",
        default="minecraft:overworld",
        metavar="WORLD",
        help="Target world identifier (default: minecraft:overworld)",
    )
    pregen_args.add_argument(
        "--center",
        nargs=2,
        type=int,
        default=[0, 0],
        metavar=("X", "Z"),
        help="Pregen center in block coordinates (default: 0 0)",
    )
    pregen_args.add_argument(
        "--radius",
        type=int,
        default=2048,
        metavar="R",
        help="Pregen radius in blocks (default: 2048)",
    )
    pregen_args.add_argument(
        "--shape",
        default="square",
        choices=[
            "square",
            "rectangle",
            "circle",
            "oval",
            "diamond",
            "triangle",
            "star",
        ],
        help="Chunky selection shape (default: square)",
    )

    parser = argparse.ArgumentParser(
        prog="data-cli",
        description="LODiffusion freeze + pregen pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Quickstart (development / dry-run):
              python data-cli.py info
              python data-cli.py pregen --radius 512 --dry-run

            Quickstart (live server with RCON):
              python data-cli.py pregen --password secret --radius 2048
        """
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("freeze", parents=[shared], help="Apply world-freeze gamerules")
    sub.add_parser("unfreeze", parents=[shared], help="Restore default gamerules")
    sub.add_parser(
        "pregen",
        parents=[shared, pregen_args],
        help="Freeze world + run Chunky pregeneration (with progress polling)",
    )
    sub.add_parser("status", parents=[shared], help="Query Chunky pregeneration progress")
    sub.add_parser(
        "info",
        parents=[pregen_args],
        help="Print pipeline plan (no server connection needed)",
    )

    extract_p = sub.add_parser(
        "extract",
        help="Extract Voxy RocksDB data → NPZ training files (no server needed)",
        parents=[],
    )
    extract_p.add_argument(
        "--saves-dir",
        type=Path,
        default=_DEFAULT_SAVES_DIR,
        metavar="DIR",
        help=f"Minecraft saves directory (default: {_DEFAULT_SAVES_DIR})",
    )
    extract_p.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing data/voxy/ before extracting",
    )
    extract_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command that would be run without executing it",
    )

    bp_p = sub.add_parser(
        "build-pairs",
        help="Pre-compute LOD pair caches from extracted NPZ chunks (no server needed)",
    )
    bp_p.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing *_pairs_v1.npz files before rebuilding",
    )
    bp_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command that would be run without executing it",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(
        host=getattr(args, "host", "localhost"),
        port=getattr(args, "port", 25575),
        password=getattr(args, "password", ""),
        dry_run=getattr(args, "dry_run", False),
        world=getattr(args, "world", "minecraft:overworld"),
        center_x=getattr(args, "center", [0, 0])[0],
        center_z=getattr(args, "center", [0, 0])[1],
        radius=getattr(args, "radius", 2048),
        shape=getattr(args, "shape", "square"),
        verbose=getattr(args, "verbose", False),
        saves_dir=getattr(args, "saves_dir", _DEFAULT_SAVES_DIR),
        clean=getattr(args, "clean", False),
    )

    dispatch = {
        "freeze": cmd_freeze,
        "unfreeze": cmd_unfreeze,
        "pregen": cmd_pregen,
        "status": cmd_status,
        "info": cmd_info,
        "extract": cmd_extract,
        "build-pairs": cmd_build_pairs,
    }
    dispatch[args.subcommand](cfg)


if __name__ == "__main__":
    main()
