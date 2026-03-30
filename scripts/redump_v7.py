#!/usr/bin/env python
"""Automated v7 noise re-dump and NPZ rebuild.

This script:
  1. Starts the Minecraft Fabric server
  2. Waits for RCON readiness
  3. Sends `/dumpnoise v7 <radius>` via RCON
  4. Polls for completion
  5. Gracefully stops the server
  6. Rebuilds the training NPZ from the fresh v7 dumps

Usage:
  python scripts/redump_v7.py                     # default radius=4 (~2K sections)
  python scripts/redump_v7.py --radius 100        # 100-chunk radius (~966K sections)
  python scripts/redump_v7.py --skip-server       # only rebuild NPZ (if dumps already exist)
  python scripts/redump_v7.py --jvm-xmx 8g        # override max heap size
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
SERVER_RUNTIME = REPO / "tools" / "fabric-server" / "runtime"
SERVER_JAR = next((REPO / "tools" / "fabric-server").glob("fabric-server-mc.*.jar"), None)
V7_DUMPS_DIR = SERVER_RUNTIME / "v7_dumps"
VOXY_DIR = REPO / "data" / "voxy_octree"
NPZ_OUTPUT = REPO / "noise_training_data" / "voxy_pairs_v7.npz"

# RCON defaults (from server.properties)
RCON_HOST = "localhost"
RCON_PORT = 25575
RCON_PASSWORD = "voxeltree"

# JVM defaults
_JVM_FLAGS_TEMPLATE = [
    "-Xmx{xmx}",
    "-Xms4g",
    "-XX:+UseG1GC",
    "-XX:+ParallelRefProcEnabled",
    "-XX:MaxGCPauseMillis=200",
    "-XX:+UnlockExperimentalVMOptions",
    "-XX:+DisableExplicitGC",
    "-XX:G1NewSizePercent=30",
    "-XX:G1MaxNewSizePercent=40",
    "-XX:G1HeapRegionSize=8M",
    "-XX:G1ReservePercent=20",
    "-XX:InitiatingHeapOccupancyPercent=15",
]


# ---------------------------------------------------------------------------
# RCON (inline, dependency-free)
# ---------------------------------------------------------------------------


def _rcon_ping(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if RCON port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _rcon_command(cmd: str) -> str:
    """Send a single RCON command and return the response."""
    sys.path.insert(0, str(REPO))
    from voxel_tree.utils.rcon import RconClient

    with RconClient(RCON_HOST, RCON_PORT, RCON_PASSWORD, timeout=30.0) as rcon:
        return rcon.command(cmd)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(xmx: str = "12g") -> subprocess.Popen:
    """Start the Fabric server and return the process handle."""
    if SERVER_JAR is None:
        print("ERROR: No fabric-server-mc.*.jar found in tools/fabric-server/")
        sys.exit(1)

    jvm_flags = [f.format(xmx=xmx) for f in _JVM_FLAGS_TEMPLATE]
    cmd = ["java", *jvm_flags, "-jar", str(SERVER_JAR), "--nogui"]
    print(f"  Starting server: java -Xmx{xmx} -jar {SERVER_JAR.name} --nogui")
    print(f"  Working dir: {SERVER_RUNTIME}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(SERVER_RUNTIME),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_rcon(timeout: float = 180.0) -> None:
    """Block until RCON port is ready."""
    print(f"  Waiting for RCON on {RCON_HOST}:{RCON_PORT}...", end="", flush=True)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _rcon_ping(RCON_HOST, RCON_PORT):
            print(" ready!")
            return
        print(".", end="", flush=True)
        time.sleep(2.0)
    print(" TIMEOUT!")
    raise TimeoutError(f"RCON not ready after {timeout:.0f}s")


def stop_server() -> None:
    """Gracefully stop the server via RCON."""
    print("  Sending /stop...")
    try:
        _rcon_command("stop")
    except Exception:
        pass  # Server may close before response
    time.sleep(3.0)


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------


def run_v7_dump(radius: int) -> None:
    """Send /dumpnoise v7 <radius> and poll for completion."""
    total = (2 * radius + 1) ** 2 * 24
    print(f"\n  /dumpnoise v7 {radius}  →  {total:,} sections expected")

    # Clear old v7 dumps
    if V7_DUMPS_DIR.exists():
        old_count = len(list(V7_DUMPS_DIR.glob("section_*.json")))
        if old_count > 0:
            print(f"  Clearing {old_count:,} old v7 dump files...")
            shutil.rmtree(V7_DUMPS_DIR)
            V7_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    response = _rcon_command(f"dumpnoise v7 {radius}")
    print(f"  Server: {response}")

    # Poll for completion by counting output files
    print("  Polling progress...", flush=True)
    last_count = 0
    stall_count = 0
    while True:
        time.sleep(5.0)
        current = len(list(V7_DUMPS_DIR.glob("section_*.json")))
        pct = 100 * current / max(total, 1)

        if current != last_count:
            rate = (current - last_count) / 5.0
            print(f"    {current:>8,}/{total:,} ({pct:5.1f}%)  ~{rate:.0f}/s", flush=True)
            last_count = current
            stall_count = 0
        else:
            stall_count += 1

        if current >= total:
            print(f"  ✓ All {total:,} sections dumped!")
            break

        if stall_count > 24:  # 2 minutes no progress
            print(f"  ⚠️  Stalled at {current:,}/{total:,} — checking RCON...")
            try:
                resp = _rcon_command("list")
                print(f"    Server alive: {resp}")
            except Exception as e:
                print(f"    Server unreachable: {e}")
                break


# ---------------------------------------------------------------------------
# Rebuild NPZ
# ---------------------------------------------------------------------------


def rebuild_npz() -> None:
    """Run build_voxy_pairs on the v7 dumps."""
    print(f"\n{'='*60}")
    print("  Rebuilding NPZ from v7 dumps")
    print(f"{'='*60}")

    n_dumps = len(list(V7_DUMPS_DIR.glob("section_*.json")))
    print(f"  v7 dumps: {n_dumps:,} files in {V7_DUMPS_DIR}")

    sys.path.insert(0, str(REPO))
    from voxel_tree.tasks.voxy.build_voxy_pairs import build_pairs

    # Back up existing NPZ
    if NPZ_OUTPUT.exists():
        backup = NPZ_OUTPUT.with_suffix(".npz.bak")
        print(f"  Backing up existing NPZ → {backup.name}")
        shutil.copy2(NPZ_OUTPUT, backup)

    n, stats = build_pairs(V7_DUMPS_DIR, VOXY_DIR, NPZ_OUTPUT)
    print(f"\n  ✓ Built {n:,} training pairs")
    print(f"    Matched sections: {stats['matched_sections']:,}")
    print(f"    Skipped (no Voxy): {stats['skipped_no_voxy']:,}")
    print(f"    Skipped (no dump): {stats['skipped_no_dump']:,}")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_npz() -> None:
    """Quick sanity check on the rebuilt NPZ."""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"  Verifying {NPZ_OUTPUT.name}")
    print(f"{'='*60}")

    npz = np.load(NPZ_OUTPUT)
    for key in sorted(npz.files):
        a = npz[key]
        print(f"  {key:20s}  shape={str(a.shape):25s}  range=[{a.min():.3f}, {a.max():.3f}]")

    n3d = npz["noise_3d"]
    n_ch = n3d.shape[1]
    print(f"\n  Noise channels: {n_ch}  {'✓ v7 (15-ch)' if n_ch == 15 else '✗ legacy'}")

    hm = npz["heightmap5"]
    hm_nonzero = (hm != 0).any()
    print(f"  Heightmaps non-zero: {hm_nonzero}  {'✓' if hm_nonzero else '✗ STILL ALL ZEROS'}")

    yvals = npz["block_y_min"]
    print(f"  block_y_min range: [{yvals.min()}, {yvals.max()}]  ({len(np.unique(yvals))} unique)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--radius", type=int, default=4, help="Chunk radius for /dumpnoise v7 (default: 4)"
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip server start/dump — only rebuild NPZ from existing v7 dumps",
    )
    parser.add_argument("--jvm-xmx", default="12g", help="JVM max heap size (default: 12g)")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing NPZ — no dump or rebuild"
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_npz()
        return

    if not args.skip_server:
        print("=" * 60)
        print("  V7 Noise Re-dump + NPZ Rebuild")
        print("=" * 60)

        proc = start_server(args.jvm_xmx)
        try:
            wait_for_rcon()
            run_v7_dump(args.radius)
        finally:
            stop_server()
            proc.wait(timeout=30)
            print("  Server stopped.")

    rebuild_npz()
    verify_npz()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

