#!/usr/bin/env python3
"""
harvest_voxy_data.py — Automated Voxy training data harvester.

Orchestrates the full pipeline for batch-extracting Voxy LOD data from a
Fabric server using the DataHarvester bot client:

  1. (Optional) Start the Fabric server
  2. Pregen chunks with Chunky via RCON
  3. Wait for the DataHarvester client to connect (manual launch or automated)
  4. Teleport the bot player in a spiral pattern via RCON
  5. Wait for VoxyWorldGen to finish generating chunks
  6. Monitor the Voxy RocksDB until it stabilises
  7. Signal completion

Usage
-----
  # Full harvest: pregen 2048-block radius, then teleport spiral
  python harvest_voxy_data.py --password voxeltree --radius 2048

  # Skip pregen (chunks already exist), just do teleport spiral + wait
  python harvest_voxy_data.py --password voxeltree --radius 2048 --skip-pregen

  # Just run the teleport spiral (server + client already running)
  python harvest_voxy_data.py --password voxeltree --radius 2048 --spiral-only

  # Monitor Voxy DB growth without doing anything else
  python harvest_voxy_data.py --monitor-only

Prerequisites
-------------
  1. Fabric server running at localhost:25565 with RCON enabled (port 25575)
     Mods: Chunky, VoxyWorldGen v2, Fabric API
  2. Minecraft client running with: Voxy, VoxyWorldGen v2 (client), DataHarvester
     (or connect manually)
  3. server.properties: online-mode=false, gamemode=spectator, allow-flight=true
"""

from __future__ import annotations

import argparse  # noqa: E402
import os
import sys
import time
from pathlib import Path

# Allow running from anywhere — ensure the VoxelTree package is importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent.parent  # VoxelTree/ repo root
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from VoxelTree.preprocessing.rcon import RconClient, RconError  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPDATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
MODRINTH_VOXY_SAVES = (
    _APPDATA / "ModrinthApp" / "profiles" / "LODiffusion dependencies" / ".voxy" / "saves"
)
DEFAULT_VOXY_SERVER = "localhost_25565"
DEFAULT_VOXY_DIR = MODRINTH_VOXY_SAVES / DEFAULT_VOXY_SERVER

RCON_HOST = "localhost"
RCON_PORT = 25575


# ---------------------------------------------------------------------------
# Spiral generator
# ---------------------------------------------------------------------------


def spiral_positions(
    radius_blocks: int,
    step_blocks: int = 256,
    center_x: int = 0,
    center_z: int = 0,
) -> list[tuple[int, int]]:
    """Generate a square spiral of (x, z) teleport positions.

    Covers the area from ``(center - radius)`` to ``(center + radius)``
    in a spiral pattern starting from the center.

    Parameters
    ----------
    radius_blocks : int
        Half-side of the square area in blocks.
    step_blocks : int
        Distance between teleport points.  256 blocks = 16 chunks wide,
        which matches VoxyWorldGen's view-distance coverage.
    center_x, center_z : int
        Center of the spiral.

    Returns
    -------
    List of (x, z) block positions.
    """
    positions = [(center_x, center_z)]
    x, z = center_x, center_z
    step = step_blocks
    direction = 0  # 0=+x, 1=+z, 2=-x, 3=-z
    side_length = 1
    side_count = 0
    steps_taken = 0

    dx = [step, 0, -step, 0]
    dz = [0, step, 0, -step]

    max_positions = ((2 * radius_blocks // step) + 1) ** 2

    while len(positions) < max_positions:
        x += dx[direction]
        z += dz[direction]
        steps_taken += 1

        if abs(x - center_x) <= radius_blocks and abs(z - center_z) <= radius_blocks:
            positions.append((x, z))

        if steps_taken >= side_length:
            steps_taken = 0
            direction = (direction + 1) % 4
            side_count += 1
            if side_count >= 2:
                side_count = 0
                side_length += 1

        # Safety: stop if we've gone well beyond the radius
        if abs(x - center_x) > radius_blocks * 2 and abs(z - center_z) > radius_blocks * 2:
            break

    return positions


# ---------------------------------------------------------------------------
# Voxy DB monitoring
# ---------------------------------------------------------------------------


def find_voxy_databases(base_dir: Path) -> list[Path]:
    """Find RocksDB storage directories under *base_dir*."""
    dbs: list[Path] = []
    if not base_dir.is_dir():
        return dbs
    for child in base_dir.iterdir():
        if child.is_dir():
            storage = child / "storage"
            if storage.is_dir():
                # Check for RocksDB markers
                has_current = (storage / "CURRENT").exists()
                has_sst = any(storage.glob("*.sst"))
                if has_current or has_sst:
                    dbs.append(storage)
    return dbs


def get_db_size(db_paths: list[Path]) -> int:
    """Total size of all SST files across all databases."""
    total = 0
    for db in db_paths:
        for f in db.glob("*.sst"):
            total += f.stat().st_size
    return total


def wait_for_voxy_db(
    voxy_dir: Path,
    stable_seconds: int = 60,
    poll_interval: int = 10,
    timeout: int = 3600,
) -> bool:
    """Wait for the Voxy database to exist and stabilise.

    Returns True if the DB stabilised, False on timeout.
    """
    print(f"\n  Monitoring Voxy DB at: {voxy_dir}")
    start = time.time()
    last_size = -1
    stable_since: float | None = None

    while (time.time() - start) < timeout:
        dbs = find_voxy_databases(voxy_dir)
        if not dbs:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:>4}s] No Voxy DB yet...")
            time.sleep(poll_interval)
            continue

        current_size = get_db_size(dbs)
        size_mb = current_size / (1024 * 1024)
        elapsed = int(time.time() - start)

        if current_size == last_size:
            if stable_since is None:
                stable_since = time.time()
            stable_for = int(time.time() - stable_since)
            print(
                f"  [{elapsed:>4}s] DB size: {size_mb:.1f} MB (stable for {stable_for}s / {stable_seconds}s)"
            )
            if stable_for >= stable_seconds:
                print(f"\n  Voxy DB stabilised at {size_mb:.1f} MB.")
                return True
        else:
            growth = (current_size - last_size) / 1024 if last_size > 0 else 0
            print(f"  [{elapsed:>4}s] DB size: {size_mb:.1f} MB (+{growth:.0f} KB)")
            stable_since = None

        last_size = current_size
        time.sleep(poll_interval)

    print(f"\n  Timed out after {timeout}s waiting for Voxy DB to stabilise.")
    return False


# ---------------------------------------------------------------------------
# RCON helpers
# ---------------------------------------------------------------------------


def rcon_command(rcon: RconClient, cmd: str, quiet: bool = False) -> str:
    """Send an RCON command and return the response."""
    resp = rcon.command(cmd).strip()
    if not quiet:
        # Safely encode for the terminal
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = resp.encode(enc, errors="replace").decode(enc) if resp else "(no response)"
        print(f"    > {cmd}")
        print(f"      {safe}")
    return resp


def wait_for_player(rcon: RconClient, timeout: int = 600, poll: int = 5) -> bool:
    """Poll ``/list`` until at least one player is connected."""
    print("\n  Waiting for a player to connect...")
    start = time.time()
    while (time.time() - start) < timeout:
        resp = rcon.command("list")
        # Typical response: "There are 1 of a max of 1 players online: Steve"
        if "0 of" not in resp.lower() and "players online" in resp.lower():
            print(f"  Player connected! ({resp.strip()})")
            return True
        elapsed = int(time.time() - start)
        if elapsed % 30 == 0 and elapsed > 0:
            print(f"  [{elapsed:>4}s] Still waiting for player...")
        time.sleep(poll)
    print(f"\n  Timed out after {timeout}s waiting for player connection.")
    return False


def run_chunky_pregen(
    rcon: RconClient,
    radius_blocks: int,
    center_x: int = 0,
    center_z: int = 0,
    poll_interval: int = 10,
    timeout: int = 7200,
) -> bool:
    """Run Chunky pregeneration and wait for completion."""
    chunk_radius = max(1, (radius_blocks + 15) // 16)

    print(f"\n  Starting Chunky pregen: radius={radius_blocks} blocks ({chunk_radius} chunks)")
    total_chunks = (2 * chunk_radius + 1) ** 2
    print(f"  Expected chunks: ~{total_chunks:,}")

    rcon_command(rcon, f"chunky radius {chunk_radius}")
    rcon_command(rcon, f"chunky center {center_x // 16} {center_z // 16}")
    rcon_command(rcon, "chunky shape square")
    rcon_command(rcon, "chunky start")

    print(f"\n  Monitoring Chunky progress (timeout {timeout}s)...")
    start = time.time()
    while (time.time() - start) < timeout:
        time.sleep(poll_interval)
        resp = rcon.command("chunky progress")
        elapsed = int(time.time() - start)

        if not resp:
            continue

        # Chunky progress looks like: "Task running for world. 50.5% complete. ETA: 1m30s"
        resp_lower = resp.lower()
        if "100" in resp and ("complete" in resp_lower or "done" in resp_lower):
            print(f"  [{elapsed:>4}s] Chunky complete! {resp.strip()}")
            return True
        if "no task" in resp_lower or "not running" in resp_lower:
            print(f"  [{elapsed:>4}s] Chunky finished (no active task).")
            return True

        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = resp.strip().encode(enc, errors="replace").decode(enc)
        print(f"  [{elapsed:>4}s] {safe}")

    print(f"\n  Chunky pregen timed out after {timeout}s.")
    return False


def run_teleport_spiral(
    rcon: RconClient,
    radius_blocks: int,
    step_blocks: int = 256,
    center_x: int = 0,
    center_z: int = 0,
    dwell_seconds: float = 8.0,
    player: str = "@a",
) -> None:
    """Teleport the bot player through a spiral pattern.

    After each teleport, waits ``dwell_seconds`` for VoxyWorldGen to
    process chunks around the new position.
    """
    positions = spiral_positions(radius_blocks, step_blocks, center_x, center_z)
    print(
        f"\n  Teleport spiral: {len(positions)} positions, "
        f"step={step_blocks}b, dwell={dwell_seconds}s"
    )
    estimated_time = len(positions) * dwell_seconds
    print(f"  Estimated time: {estimated_time / 60:.1f} minutes")

    for i, (x, z) in enumerate(positions, 1):
        cmd = f"tp {player} {x} 200 {z}"
        rcon.command(cmd)

        if i % 20 == 0 or i == len(positions):
            pct = i / len(positions) * 100
            print(f"  [{i}/{len(positions)}] ({pct:.0f}%) Teleported to ({x}, {z})")

        time.sleep(dwell_seconds)

    print("  Teleport spiral complete!")


def run_ingestall(
    rcon: RconClient,
    radius_blocks: int,
    poll_interval: float = 10.0,
    timeout: int = 7200,
) -> bool:
    """Send ``/ingestall`` via RCON and poll until complete.

    The ``/ingestall`` command scans pre-generated region files on the
    server, serialises each chunk's block-state/biome/light data, and
    sends it to connected clients as network packets.  The client-side
    Voxy then ingests them through its standard pipeline — identical
    to normal VoxyWorldGen gameplay, but without any teleporting.

    Returns ``True`` when the ingest finishes (or was already done),
    ``False`` on timeout or error.
    """
    radius_chunks = max(1, radius_blocks // 16)
    resp = rcon_command(rcon, f"ingestall {radius_chunks}")
    print(f"  Server: {resp}")

    if "no chunks found" in resp.lower():
        print("  No pre-generated chunks found. Run Chunky pregen first.")
        return False
    if "already running" in resp.lower():
        print("  Ingest already in progress — attaching to monitor.")

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(poll_interval)
        status = rcon_command(rcon, "ingestall status", quiet=True)
        print(f"  {status}")

        if "no ingest running" in status.lower():
            return True
        if "complete" in status.lower():
            return True

    print(f"\n  /ingestall timed out after {timeout}s.")
    # Stop the ingest gracefully
    rcon_command(rcon, "ingestall stop")
    return False


# ---------------------------------------------------------------------------
# Freeze helpers (from cli.py)
# ---------------------------------------------------------------------------

FREEZE_COMMANDS = [
    "gamerule doFireTick false",
    "gamerule doMobSpawning false",
    "gamerule doWeatherCycle false",
    "gamerule doDaylightCycle false",
    "gamerule randomTickSpeed 0",
    "gamerule doEntityDrops false",
    "gamerule doTileDrops false",
    "gamerule doMobLoot false",
    "difficulty peaceful",
    "time set 6000",
    "weather clear",
]


def freeze_world(rcon: RconClient) -> None:
    """Freeze all dynamic world state for deterministic data collection."""
    print("\n  Freezing world state...")
    for cmd in FREEZE_COMMANDS:
        rcon_command(rcon, cmd, quiet=True)
    print("  World frozen.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_harvest(args: argparse.Namespace) -> None:
    """Execute the harvest pipeline.

    In ``--pregen-only`` mode only the freeze + Chunky pregen step runs.
    Otherwise, the full ingest pipeline runs (optionally skipping pregen
    when the ``pregen`` DAG step already completed it).
    """
    sep = "=" * 65
    print(f"\n{sep}")
    if args.pregen_only:
        print("  CHUNK PREGEN — Chunky Pregeneration")
    else:
        print("  DATA HARVESTER — Automated Voxy Training Data Extraction")
    print(sep)
    print(f"  Server:   {args.host}:{args.port}")
    print(f"  RCON:     {args.host}:{args.rcon_port}")
    print(f"  Radius:   {args.radius} blocks ({args.radius // 16} chunks)")
    if not args.pregen_only:
        print(f"  Voxy DB:  {args.voxy_dir}")
    print(sep)

    rcon = None
    try:
        rcon = RconClient(args.host, args.rcon_port, args.password, timeout=15.0)
        rcon.connect()
    except RconError as e:
        print(f"\n  ERROR: Cannot connect to RCON: {e}")
        print("  Make sure the server is running with enable-rcon=true.")
        sys.exit(1)

    try:
        # Step 1: Freeze world
        freeze_world(rcon)

        # Step 2: Chunky pregen (optional)
        if not args.skip_pregen and not args.spiral_only:
            print(f"\n{'─' * 65}")
            print("  STEP 1: Chunky Pregeneration")
            print(f"{'─' * 65}")
            ok = run_chunky_pregen(
                rcon,
                args.radius,
                center_x=args.center_x,
                center_z=args.center_z,
                timeout=args.chunky_timeout,
            )
            if not ok:
                print("  WARNING: Chunky pregen may not have completed.")
                if not args.force:
                    print("  Use --force to continue anyway.")
                    return

        # In pregen-only mode, stop here.
        if args.pregen_only:
            print(f"\n{sep}")
            print("  PREGEN COMPLETE")
            print(sep)
            return

        # Step 3: Wait for player (bot client)
        if not args.spiral_only:
            print(f"\n{'─' * 65}")
            print("  STEP 2: Waiting for Bot Client Connection")
            print(f"{'─' * 65}")
            print("  Launch the Minecraft client with the DataHarvester mod,")
            print("  or connect manually to localhost:25565.")
            if not wait_for_player(rcon, timeout=args.player_timeout):
                print("  No player connected. Aborting.")
                return
        else:
            # Verify player is connected
            resp = rcon.command("list")
            if "0 of" in resp.lower():
                print("  ERROR: No player connected. Connect a client first.")
                return

        # Step 4: Ingest chunks into client-side Voxy
        print(f"\n{'─' * 65}")
        if args.legacy_spiral:
            print("  STEP 3: Teleport Spiral (legacy mode)")
            print(f"{'─' * 65}")
            run_teleport_spiral(
                rcon,
                radius_blocks=args.radius,
                step_blocks=args.step,
                center_x=args.center_x,
                center_z=args.center_z,
                dwell_seconds=args.dwell,
            )
        else:
            print("  STEP 3: /ingestall — Server-to-Client Chunk Broadcast")
            print(f"{'─' * 65}")
            print("  Sending pre-generated chunks directly to client Voxy.")
            print("  No teleporting needed — player can sit still.")
            ok = run_ingestall(
                rcon,
                radius_blocks=args.radius,
                timeout=args.voxy_timeout,
            )
            if not ok and not args.force:
                print("  /ingestall did not complete. Use --force to continue.")
                return

        # Step 5: Wait for Voxy DB to stabilise
        print(f"\n{'─' * 65}")
        print("  STEP 4: Waiting for Voxy DB to Stabilise")
        print(f"{'─' * 65}")
        voxy_dir = Path(args.voxy_dir)
        ok = wait_for_voxy_db(
            voxy_dir,
            stable_seconds=args.stable_seconds,
            timeout=args.voxy_timeout,
        )

        if ok:
            # Print final stats
            dbs = find_voxy_databases(voxy_dir)
            total = get_db_size(dbs)
            print(f"\n{sep}")
            print("  HARVEST COMPLETE")
            print(f"  Voxy DB size: {total / (1024*1024):.1f} MB")
            print("  DB locations:")
            for db in dbs:
                print(f"    {db}")

    finally:
        if rcon is not None:
            rcon.close()


def run_monitor(args: argparse.Namespace) -> None:
    """Just monitor Voxy DB growth — no harvesting steps."""

    voxy_dir = Path(args.voxy_dir)
    wait_for_voxy_db(
        voxy_dir,
        stable_seconds=args.stable_seconds,
        poll_interval=5,
        timeout=args.voxy_timeout,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automated Voxy training data harvester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Connection
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=25565, help="Server port (default: 25565)")
    parser.add_argument(
        "--rcon-port", type=int, default=RCON_PORT, help="RCON port (default: 25575)"
    )
    parser.add_argument(
        "--password", default="voxeltree", help="RCON password (default: voxeltree)"
    )

    # Area
    parser.add_argument(
        "--radius", type=int, default=2048, help="Harvest radius in blocks (default: 2048)"
    )
    parser.add_argument("--center-x", type=int, default=0, help="Center X coordinate (default: 0)")
    parser.add_argument("--center-z", type=int, default=0, help="Center Z coordinate (default: 0)")
    parser.add_argument(
        "--step", type=int, default=256, help="Teleport step size in blocks (default: 256)"
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=8.0,
        help="Seconds to dwell at each teleport position (default: 8)",
    )

    # Voxy
    parser.add_argument(
        "--voxy-dir",
        default=str(DEFAULT_VOXY_DIR),
        help=f"Voxy saves directory (default: {DEFAULT_VOXY_DIR})",
    )
    parser.add_argument(
        "--stable-seconds",
        type=int,
        default=60,
        help="DB must be stable for this many seconds (default: 60)",
    )

    # Timeouts
    parser.add_argument(
        "--chunky-timeout",
        type=int,
        default=7200,
        help="Chunky pregen timeout in seconds (default: 7200)",
    )
    parser.add_argument(
        "--player-timeout", type=int, default=600, help="Wait for player timeout (default: 600)"
    )
    parser.add_argument(
        "--voxy-timeout",
        type=int,
        default=3600,
        help="Voxy DB stabilisation timeout (default: 3600)",
    )

    # Mode flags
    parser.add_argument(
        "--skip-pregen",
        action="store_true",
        help="Skip Chunky pregeneration (chunks already exist)",
    )
    parser.add_argument(
        "--pregen-only",
        action="store_true",
        help="Only run freeze + Chunky pregen, then exit (no client needed)",
    )
    parser.add_argument(
        "--spiral-only",
        action="store_true",
        help="Skip pregen and player wait; just run spiral + monitor",
    )
    parser.add_argument(
        "--legacy-spiral",
        action="store_true",
        help="Use teleport spiral instead of /ingestall (slower, for debugging)",
    )
    parser.add_argument(
        "--monitor-only", action="store_true", help="Only monitor Voxy DB growth, no RCON"
    )
    parser.add_argument(
        "--force", action="store_true", help="Continue even if pregen appears incomplete"
    )

    args = parser.parse_args(argv)

    if args.monitor_only:
        run_monitor(args)
    else:
        run_harvest(args)


if __name__ == "__main__":
    main()
