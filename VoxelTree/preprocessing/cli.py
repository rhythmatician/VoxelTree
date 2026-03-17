#!/usr/bin/env python3
"""
LODiffusion Data CLI — data preparation pipeline orchestrator.

Unified tool for all data preparation: RCON world management, Voxy extraction,
column-height enrichment, and octree pair cache building.

Canonical pipeline steps (run all, or start from any step):
  1) pregen              — RCON: freeze world + Chunky chunk generation
  2) voxy-import         — MANUAL: connect Modrinth client; Voxy auto-populates server LOD DB
  3) dumpnoise           — RCON: /dumpnoise stage1 + sparse_root → all training formats
  4) extract-octree      — Voxy RocksDB → data/voxy_octree/level_N/*.npz
  5) column-heights-octree — Merge vanilla heightmaps from dumpnoise JSON into NPZs
  6) build-octree-pairs  — NPZ → octree training pair caches (*_octree_pairs.npz)

Data Flow & Composition
-----------------------
Steps logically group into three phases:

  WORLD & NOISE GENERATION (Steps 1–3):
    └─ RCON commands: freeze/pregen/dumpnoise
    └─ Outputs: Voxy RocksDB + stage1_dumps/*.json + sparse_root_dumps/*.json
    └─ Only needed if regenerating data or switching world seeds

  DATA EXTRACTION (Step 4):
    └─ Convert: Voxy RocksDB → raw octree sections (*.npz per LOD level)
    └─ Input: Voxy database(s) from LODiffusion/run/saves
    └─ Output: data/voxy_octree/level_N/*.npz (32³ block grids)
    └─ Must run when: New world is generated

  TRAINING PREPARATION (Steps 5–6):
    └─ Steps 5–6 form a pair: ALWAYS run together
    └─ Step 5: Add 5-plane 32×32 heightmaps to octree NPZs
    └─ Step 6: Build octree parent/child pairs (input for train_octree.py)
    └─ Output: data/voxy_octree/*_octree_pairs.npz (ready for training)

Common workflows:
  • Full pipeline (new world):       from-step pregen         (all steps 1–6)
  • Existing Voxy DB:                from-step extract-octree (only 4–6)
  • Cached NPZs:                     from-step column-heights-octree (only 5–6)

Usage examples
--------------
  # Full dataprep from pregen through build-octree-pairs:
  python data-cli.py dataprep --from-step pregen \\
      --password secret --world-name "New World" \\
      --voxy-dir LODiffusion/run/saves

  # Start from extraction (skips RCON steps, checks Voxy DBs exist):
  python data-cli.py dataprep --from-step extract-octree \\
      --voxy-dir LODiffusion/run/saves

  # Just column-heights + build-pairs (checks NPZ files exist):
  python data-cli.py dataprep --from-step column-heights-octree

  # Individual RCON commands:
  python data-cli.py freeze --dry-run
  python data-cli.py pregen --password secret --radius 2048
  python data-cli.py voxy-import --world-name "New World" --password secret

  # Other RCON helpers:
  python data-cli.py status --password secret
  python data-cli.py unfreeze --password secret

Enabling RCON in server.properties
-----------------------------------
  enable-rcon=true
  rcon.port=25575
  rcon.password=<your-password>
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from VoxelTree.preprocessing.rcon import RconClient


def _safe_unicode(char: str, fallback: str) -> str:
    """Return *char* if stdout encoding supports it, else return *fallback*."""

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        char.encode(encoding)
        return char
    except Exception:
        return fallback


# Path anchors — cli.py lives at VoxelTree/preprocessing/
_PREPROCESSING_DIR = Path(__file__).resolve().parent  # VoxelTree/preprocessing/
_PKG_DIR = _PREPROCESSING_DIR.parent  # VoxelTree/
_REPO_ROOT = _PKG_DIR.parent  # repo root

# ---------------------------------------------------------------------------
# Dataprep constants
# ---------------------------------------------------------------------------

VOXY_VOCAB_PATH = _PKG_DIR / "config" / "voxy_vocab.json"
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "voxy_octree"

# Voxy stores server-connection databases under:
#   %APPDATA%\ModrinthApp\profiles\<profile>\.voxy\saves\<host_port>\<hash>\storage
# Singleplayer worlds use:
#   <minecraft-saves>\<world>\voxy\<hash>\storage
#
# We detect the Modrinth profile path at import time and prefer it if present.
_APPDATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
_MODRINTH_VOXY_SAVES = (
    _APPDATA / "ModrinthApp" / "profiles" / "LODiffusion dependencies" / ".voxy" / "saves"
)
DEFAULT_VOXY_SERVER = "localhost_25565"  # Voxy's directory name for a local server connection
DEFAULT_MODRINTH_VOXY_DIR = _MODRINTH_VOXY_SAVES / DEFAULT_VOXY_SERVER

# Fall back to LODiffusion dev saves if the Modrinth profile isn't present.
DEFAULT_VOXY_DIR = (
    DEFAULT_MODRINTH_VOXY_DIR
    if DEFAULT_MODRINTH_VOXY_DIR.is_dir()
    else _REPO_ROOT.parent / "LODiffusion" / "run" / "saves"
)

#: Ordered list of dataprep steps (pregen is now handled by the standalone 'pregen' GUI step).
DATAPREP_STEPS = [
    "voxy-import",
    "dumpnoise",
    "extract-octree",
    "column-heights-octree",
    "build-octree-pairs",
]

#: Fabric server runtime directory (contains server.properties, world folders, etc.).
DEFAULT_SERVER_DIR = _REPO_ROOT / "tools" / "fabric-server" / "runtime"

#: Default noise-dump output directory (relative to repo root).
DEFAULT_NOISE_DUMP_DIR = DEFAULT_SERVER_DIR / "noise_dumps"

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
    verbose: bool = False
    voxy_import_world: str = ""
    voxy_import_timeout: int = 300


# Gamerule commands that freeze the world for deterministic data collection.
FREEZE_COMMANDS: list[tuple[str, str]] = [
    # Carpet: stops ALL game ticks — blocks fluid flow, lava spread, block updates.
    # This MUST be first; gamerule commands alone do NOT stop lava from flowing.
    ("/tick freeze", "Carpet: freeze all game ticks (stops lava, fluid flow, block updates)"),
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
    ("/tick unfreeze", "Carpet: resume normal tick execution"),
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
                ok = status != "(no response)" or "Error" not in status
                tick = _safe_unicode("✓", "[OK]") if ok else _safe_unicode("✗", "[FAIL]")
                print(f"  {tick}  {desc}")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_voxy_import(cfg: PipelineConfig) -> None:
    """Wait for the Voxy server-connection database to be populated.

    When a Minecraft client connects to the server, Voxy automatically builds
    LOD data for every chunk it receives and stores it in::

        %APPDATA%\\ModrinthApp\\profiles\\<profile>\\.voxy\\saves\\localhost_25565\\<hash>\\storage\\

    No in-game command is required for server worlds \u2014 Voxy simply needs a
    connected client.  This step:

    1. Prints connection instructions.
    2. Polls ``DEFAULT_VOXY_DIR`` every 10 s until the RocksDB storage appears,
       then returns so the pipeline can continue to extraction.

    .. note::
        ``/voxy import world "<name>/"`` imports FROM a singleplayer save INTO
        Voxy \u2014 the wrong direction for server-based training data.  Do **not**
        run that command when using a dedicated server.
    """
    voxy_base = Path(DEFAULT_VOXY_DIR)

    sep = "-" * 60
    print(f"\n{sep}")
    print("  VOXY IMPORT -- Voxy server-connection database")
    print(sep)
    print("  Voxy auto-populates its LOD database as you observe chunks.")
    print()
    print("  OPTION A — Automated (recommended):")
    print("    1. Build & install the DataHarvester mod:")
    print("         VoxelTree\\tools\\data-harvester\\build_and_install.bat")
    print("    2. Run the harvest script (handles teleportation via RCON):")
    print("         python -m VoxelTree.preprocessing.harvest --password voxeltree --radius 2048")
    print("    3. Launch Minecraft (Modrinth: LODiffusion dependencies profile)")
    print("       The DataHarvester mod auto-connects to localhost:25565.")
    print()
    print("  OPTION B — Manual:")
    print("    1. Launch Minecraft (Modrinth: LODiffusion dependencies profile)")
    print("    2. Connect to:  localhost:25565")
    print("    3. Teleport around to cover the pre-generated area: /tp 0 100 0")
    print()
    print(f"  Voxy DB location: {voxy_base}")
    print(sep)

    if cfg.dry_run:
        print(f"\n  [DRY-RUN] Would poll: {voxy_base}")
        return

    timeout = cfg.voxy_import_timeout
    interval = 10
    start = time.time()

    while time.time() - start < timeout:
        time.sleep(interval)
        elapsed = int(time.time() - start)

        dbs = _find_voxy_databases(voxy_base)
        if dbs:
            print(f"\n  [{elapsed:>4}s] Voxy DB found ({len(dbs)} dimension(s)):")
            for db in dbs:
                print(f"              {db}")
            print("  Voxy DB ready!")
            return

        print(f"  [{elapsed:>4}s] Waiting for Voxy DB under {voxy_base}...")

    print(
        f"\n  Timed out after {timeout}s."
        "\n  Once Voxy has data, re-run: dataprep --from-step extract-octree"
    )


def cmd_dumpnoise(cfg: PipelineConfig) -> None:
    """Consolidate all training noise dumps via single /dumpnoise command.

    Sends both /dumpnoise stage1 and /dumpnoise sparse_root in sequence,
    generating all noise data needed for the full training pipeline:
    - stage1 format (4×48×4 cells): 12 input features + final_density per chunk
    - sparse_root format (4×2×4 cells): 13 noise channels + biome_ids per section

    Output:
      <game_dir>/stage1_dumps/chunk_<cx>_<cz>.json
      <game_dir>/sparse_root_dumps/section_<cx>_<sy>_<cz>.json
    """
    # Convert block-radius to chunk-radius (round up to cover the requested area)
    chunk_radius = max(1, (cfg.radius + 15) // 16)

    if cfg.dry_run:
        print("\n  [DRY-RUN] Would send:")
        print("    (freeze commands — see 'freeze' subcommand for full list)")
        print("    /kill @e[type=!player]  # clear spawn-bloat mobs before dump")
        print(f"    /dumpnoise stage1 {chunk_radius}")
        print(f"    /dumpnoise sparse_root {chunk_radius}")
        print(f"  (block radius {cfg.radius} -> chunk radius {chunk_radius})")
        return

    if not cfg.password:
        print("ERROR: --password required (or use --dry-run to preview).")
        sys.exit(1)

    # Freeze world state. FREEZE_COMMANDS is idempotent — safe to run even
    # if a prior pregen step already applied it, and critical when dumpnoise
    # is executed standalone (otherwise mobs tick and lava flows during dump).
    _run_commands(FREEZE_COMMANDS, cfg, "Freeze world before noise dump")

    total_chunks = (2 * chunk_radius + 1) ** 2
    total_sections = total_chunks * 24  # 24 sections per chunk column
    print("\n  Consolidating training noise dumps:")
    print(f"    Stage1:    {total_chunks:,} chunks @ 4×48×4 resolution")
    print(f"    SparseRoot: {total_sections:,} sections @ 4×2×4 resolution + biome_ids")

    timeout = cfg.voxy_import_timeout
    interval = 5

    with RconClient(cfg.host, cfg.port, cfg.password) as rcon:
        # Kill all non-player entities before dumping.
        #
        # Chunky pregen loads 128 K+ chunks, each of which auto-spawns mobs.
        # This can produce 100 K+ live entity objects in the JVM heap.  When
        # the 4 concurrent Stage1Dumper-Worker threads start making 8 MB
        # allocations, the GC cannot evacuate fast enough (GCLocker thrash)
        # and the server crashes with OutOfMemoryError: Java heap space —
        # even with a 16 GB heap.
        #
        # Killing all mobs now and sleeping briefly lets G1GC reclaim that
        # memory before the allocation-intensive dump begins.
        print("\n  Killing all entities to reduce heap pressure...")
        kill_resp = rcon.command("kill @e[type=!player]")
        kill_text = kill_resp.strip() or "(no response)"
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            kill_text = kill_text.encode(enc, errors="replace").decode(enc)
        except Exception:
            kill_text = kill_text.encode("utf-8", errors="replace").decode("utf-8")
        print(f"  Server: {kill_text}")
        print("  Sleeping 5 s for GC to reclaim freed entity memory...")
        time.sleep(5)

        # ── Stage 1 dump ────────────────────────────────────────────────────
        print("\n  [1/2] Dumping Stage1 noise...")
        cmd_str = f"dumpnoise stage1 {chunk_radius}"
        resp = rcon.command(cmd_str)
        resp_text = resp.strip() or "(no response)"
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            resp_text = resp_text.encode(enc, errors="replace").decode(enc)
        except Exception:
            resp_text = resp_text.encode("utf-8", errors="replace").decode("utf-8")
        print(f"        Server: {resp_text}")

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(interval)
            elapsed = time.time() - start
            # Heuristic: ~200 chunks/sec is typical for Stage1
            estimate = total_chunks / 200.0 + 5.0
            if elapsed >= estimate:
                break
            print(
                f"        [{time.strftime('%H:%M:%S')}] Waiting "
                f"(~{int(estimate - elapsed)}s remaining)..."
            )

        # ── SparseRoot dump ─────────────────────────────────────────────────
        print("\n  [2/2] Dumping SparseRoot noise...")
        cmd_str = f"dumpnoise sparse_root {chunk_radius}"
        resp = rcon.command(cmd_str)
        resp_text = resp.strip() or "(no response)"
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            resp_text = resp_text.encode(enc, errors="replace").decode(enc)
        except Exception:
            resp_text = resp_text.encode("utf-8", errors="replace").decode("utf-8")
        print(f"        Server: {resp_text}")

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(interval)
            elapsed = time.time() - start
            # Heuristic: ~400 sections/sec is typical for SparseRoot (more fine-grained, smaller writes)
            estimate = total_sections / 400.0 + 5.0
            if elapsed >= estimate:
                break
            print(
                f"        [{time.strftime('%H:%M:%S')}] Waiting "
                f"(~{int(estimate - elapsed)}s remaining)..."
            )

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n  Noise consolidation complete:")
    print(f"    [OK] Stage1:    <game_dir>/stage1_dumps/ ({total_chunks:,} files)")
    print(f"    [OK] SparseRoot: <game_dir>/sparse_root_dumps/ ({total_sections:,} files)")


# ---------------------------------------------------------------------------
# Dataprep pipeline — prerequisite checks + step runners
# ---------------------------------------------------------------------------


def _find_voxy_databases(saves_dir: Path) -> list[Path]:
    """Discover Voxy RocksDB storage directories under *saves_dir*.

    Handles two directory layouts:

    **Server-connection layout** (Modrinth / multiplayer):
        ``<saves_dir>/<hash>/storage/``
        Voxy creates one hash-named subdirectory per dimension when connected
        to a server.  ``saves_dir`` is typically the per-server directory:
        ``%APPDATA%/ModrinthApp/profiles/<profile>/.voxy/saves/localhost_25565``

    **Singleplayer / dev-client layout**:
        ``<saves_dir>/<world>/voxy/<hash>/storage/``
        Voxy embeds data inside the world's save folder.
    """
    found: list[Path] = []

    # Layout 1: server-connection — <hash>/storage directly under saves_dir
    direct = [d for d in sorted(saves_dir.glob("*/storage")) if d.is_dir()]
    # Only accept these if they look like RocksDB (have a CURRENT or *.sst file)
    for d in direct:
        if any(d.glob("CURRENT")) or any(d.glob("*.sst")):
            found.append(d)

    # Layout 2: singleplayer — <world>/voxy/<hash>/storage
    for d in sorted(saves_dir.glob("*/voxy/*/storage")):
        if d.is_dir() and d not in found:
            found.append(d)

    return found


def _count_source_npz(data_dir: Path) -> int:
    """Count non-cache NPZ files in *data_dir* (including train/val subdirs)."""
    count = 0
    for subdir in [data_dir / "train", data_dir / "val"]:
        if subdir.is_dir():
            count += len(list(subdir.glob("*.npz")))
    if count == 0:
        # Flat layout fallback
        count = sum(
            1
            for f in data_dir.glob("*.npz")
            if not f.name.endswith(("_pairs_v1.npz", "_pairs_v2.npz"))
        )
    return count


def _sample_npz_files(data_dir: Path, n: int = 5) -> list[Path]:
    """Return up to *n* source NPZ files for spot-checking."""
    for subdir in [data_dir / "train", data_dir / "val"]:
        if subdir.is_dir():
            files = sorted(subdir.glob("*.npz"))[:n]
            if files:
                return files
    files = sorted(
        f for f in data_dir.glob("*.npz") if not f.name.endswith(("_pairs_v1.npz", "_pairs_v2.npz"))
    )
    return files[:n]


def _npz_has_key(path: Path, key: str) -> bool:
    """Check whether an NPZ archive contains *key* (stdlib-only, no numpy)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return f"{key}.npy" in zf.namelist()
    except Exception:
        return False


def _check_prerequisites(from_step: str, args: argparse.Namespace) -> bool:
    """Verify that the immediately-prior step's output exists.

    Each ``from_step`` only checks what it *directly* needs — we don't
    cascade through all prior steps, since earlier artefacts may have been
    cleaned up after being consumed.

    Returns True if everything looks good, False with a diagnostic message
    if a prerequisite is missing.
    """
    data_dir: Path = getattr(args, "data_dir", DEFAULT_DATA_DIR)

    # extract-octree → evidence that voxy-import ran: Voxy databases exist
    if from_step == "extract-octree":
        voxy_dir: Path | None = getattr(args, "voxy_dir", None) or DEFAULT_VOXY_DIR
        dbs = _find_voxy_databases(voxy_dir)
        if not dbs:
            print(f"ERROR: No Voxy databases found under {voxy_dir}")
            print("  Expected: <saves>/<world>/voxy/<hash>/storage/")
            print("  Did pregen + voxy-import complete?")
            return False
        print(f"  {_safe_unicode('✓', '[OK]')} Found {len(dbs)} Voxy database(s)")

    # column-heights-octree → evidence that extract-octree ran AND noise dumps exist
    if from_step == "column-heights-octree":
        # Check for level_N subdirectories
        has_levels = any((data_dir / f"level_{i}").is_dir() for i in range(5))
        if not has_levels:
            print(f"ERROR: No level_N/ directories found in {data_dir}")
            print("  Run:  python data-cli.py dataprep --octree --from-step extract-octree ...")
            return False
        total_npz = sum(
            len(list((data_dir / f"level_{i}").glob("*.npz")))
            for i in range(5)
            if (data_dir / f"level_{i}").is_dir()
        )
        if total_npz == 0:
            print(f"ERROR: No NPZ files in level_N/ directories under {data_dir}")
            return False
        print(
            f"  {_safe_unicode('✓', '[OK]')} {total_npz:,} octree NPZ files across level_N/ directories"
        )

        noise_dump_dir = getattr(args, "noise_dump_dir", DEFAULT_NOISE_DUMP_DIR)
        jsons = list(noise_dump_dir.glob("chunk_*.json")) if noise_dump_dir.is_dir() else []
        if not jsons:
            print(f"ERROR: No noise dump JSON files in {noise_dump_dir}")
            print("  Run:  python data-cli.py dataprep --from-step dumpnoise ...")
            return False
        print(f"  {_safe_unicode('✓', '[OK]')} {len(jsons):,} noise dump JSON file(s)")

    # build-pairs → evidence that column-heights ran: heightmap_surface arrays
    if from_step == "build-pairs":
        samples = _sample_npz_files(data_dir)
        if not samples:
            print(f"ERROR: No NPZ files to check in {data_dir}")
            return False
        missing = [f for f in samples if not _npz_has_key(f, "heightmap_surface")]
        if missing:
            print("ERROR: NPZ files are missing heightmap_surface:")
            for f in missing:
                print(f"  - {f.name}")
            print("  Run: python data-cli.py dataprep --from-step column-heights ...")
            return False
        print(f"  {_safe_unicode('✓', '[OK]')} NPZ files have heightmap_surface")

    # build-octree-pairs → evidence that column-heights-octree ran: heightmap32 arrays
    if from_step == "build-octree-pairs":
        # Sample check a few files from any available level
        sample_found = False
        for lvl in range(5):
            level_dir = data_dir / f"level_{lvl}"
            if not level_dir.is_dir():
                continue
            samples = sorted(level_dir.glob("*.npz"))[:3]
            if not samples:
                continue
            missing_hm = [f for f in samples if not _npz_has_key(f, "heightmap32")]
            if missing_hm:
                print("ERROR: Octree NPZ files missing heightmap32:")
                for f in missing_hm:
                    print(f"  - {f.name}")
                print("  Run: python data-cli.py dataprep --from-step column-heights-octree ...")
                return False
            sample_found = True
            break
        if not sample_found:
            print(f"ERROR: No octree NPZ files found in {data_dir}")
            return False
        print(f"  {_safe_unicode('✓', '[OK]')} Octree NPZ files have heightmap32")

    return True


# --- step runners (local, subprocess-based) --------------------


def _step_extract_octree(args: argparse.Namespace) -> bool:
    """Extract multi-LOD octree data from Voxy RocksDB databases."""
    print()
    print("=" * 70)
    print("  STEP 4/6: Extract octree data (all LOD levels)")
    print("=" * 70)
    print()

    voxy_dir: Path = getattr(args, "voxy_dir", None) or DEFAULT_VOXY_DIR
    dbs = _find_voxy_databases(voxy_dir)
    if not dbs:
        print(f"ERROR: No Voxy databases found under {voxy_dir}")
        return False

    data_dir: Path = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    print(f"Found {len(dbs)} Voxy database(s), output \u2192 {data_dir}")

    from VoxelTree.scripts.extract_octree_data import main as _extract_main

    extract_args = [
        *[str(d) for d in dbs],
        "--output-dir",
        str(data_dir),
        "--vocab",
        str(VOXY_VOCAB_PATH),
        "--min-solid",
        str(getattr(args, "min_solid", 0.02)),
    ]
    max_sections = getattr(args, "max_sections", None)
    if max_sections is not None:
        extract_args.extend(["--max-sections", str(max_sections)])

    try:
        _extract_main(extract_args)
        return True
    except SystemExit as exc:
        return exc.code == 0


def _step_column_heights_octree(args: argparse.Namespace) -> bool:
    """Merge vanilla heightmaps into octree NPZ files (32×32, 5-plane)."""
    print()
    print("=" * 70)
    print("  STEP 5/6: Merge 5-plane heightmaps into octree NPZs")
    print("=" * 70)
    print()

    data_dir: Path = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    noise_dump_dir: Path = getattr(args, "noise_dump_dir", DEFAULT_NOISE_DUMP_DIR)

    from VoxelTree.scripts.add_column_heights import main as _heights_main

    heights_args = [str(data_dir), "--noise-dump-dir", str(noise_dump_dir)]
    try:
        _heights_main(heights_args)
        return True
    except SystemExit as exc:
        return exc.code == 0


def _step_build_octree_pairs(args: argparse.Namespace) -> bool:
    """Build octree parent/child training pair caches."""
    print()
    print("=" * 70)
    print("  STEP 6/6: Build octree training pairs")
    print("=" * 70)
    print()

    data_dir: Path = getattr(args, "data_dir", DEFAULT_DATA_DIR)

    from VoxelTree.scripts.octree.build_pairs import main as _build_main

    pairs_args = [
        "--data-dir",
        str(data_dir),
        "--val-split",
        str(getattr(args, "val_split", 0.1)),
    ]
    if getattr(args, "clean", False):
        pairs_args.append("--clean")

    try:
        _build_main(pairs_args)
        return True
    except SystemExit as exc:
        return exc.code == 0


# --- dataprep orchestrator ------------------------------------------------


def cmd_dataprep(args: argparse.Namespace) -> None:
    """Run the dataprep pipeline from ``--from-step`` through the final step.

    Each step runs in order; if starting at an intermediate step the
    orchestrator first verifies that all earlier steps produced valid output.
    """
    from_step: str = args.from_step

    # Default --data-dir
    if args.data_dir is None:
        args.data_dir = DEFAULT_DATA_DIR

    step_list = DATAPREP_STEPS

    if from_step not in step_list:
        print(f"ERROR: '{from_step}' is not a valid step.")
        print(f"  Valid steps: {', '.join(step_list)}")
        sys.exit(1)

    idx = step_list.index(from_step)
    steps_to_run = step_list[idx:]

    print()
    print("=" * 70)
    print(f"  DATAPREP PIPELINE:  {' -> '.join(steps_to_run)}")
    print("=" * 70)

    # ---- prerequisite gate ----
    if idx > 0:
        print("\nChecking prerequisites for starting at '%s'..." % from_step)
        if not _check_prerequisites(from_step, args):
            print("\nPrerequisite check FAILED — aborting.")
            sys.exit(1)
        print()

    # ---- validate RCON args early if RCON steps are in the plan ----
    # Note: voxy-import is a MANUAL step (Voxy is client-only) — no RCON needed.
    rcon_needed = any(s in steps_to_run for s in ("pregen", "dumpnoise"))
    dry_run = getattr(args, "dry_run", False)
    if rcon_needed and not dry_run:
        password = getattr(args, "password", "")
        if not password:
            print("ERROR: --password is required for RCON steps (pregen, dumpnoise).")
            print("  Use --dry-run to preview commands without a server.")
            sys.exit(1)

    # ---- build PipelineConfig for RCON steps ----
    cfg = PipelineConfig(
        host=getattr(args, "host", "localhost"),
        port=getattr(args, "port", 25575),
        password=getattr(args, "password", ""),
        dry_run=dry_run,
        world=getattr(args, "world", "minecraft:overworld"),
        center_x=getattr(args, "center", [0, 0])[0],
        center_z=getattr(args, "center", [0, 0])[1],
        radius=getattr(args, "radius", 2048),
        verbose=getattr(args, "verbose", False),
        voxy_import_world=getattr(args, "world_name", "") or "New World/",
        voxy_import_timeout=getattr(args, "timeout", 300),
    )

    # ---- execute steps ----
    t0 = time.time()

    step_runners = {
        "voxy-import": lambda: cmd_voxy_import(cfg),
        "dumpnoise": lambda: cmd_dumpnoise(cfg),
        "extract-octree": lambda: _step_extract_octree(args),
        "column-heights-octree": lambda: _step_column_heights_octree(args),
        "build-octree-pairs": lambda: _step_build_octree_pairs(args),
    }

    for step in steps_to_run:
        runner = step_runners[step]
        result = runner()
        # Local steps return bool; RCON steps return None (they sys.exit on failure)
        if result is False:
            print(f"\nERROR: Step '{step}' failed — aborting pipeline.")
            sys.exit(1)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("  DATAPREP COMPLETE  (%d step(s) in %.1fs)" % (len(steps_to_run), elapsed))
    print("=" * 70)
    print()
    print("Next: python pipeline.py train")


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
    # Note: shape is no longer configurable — the pipeline always uses
    # /chunky corners to produce an exact square grid.

    parser = argparse.ArgumentParser(
        prog="data-cli",
        description="LODiffusion freeze + pregen pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Quickstart (development / dry-run):
              python data-cli.py info
              python data-cli.py pregen --radius 512 --dry-run

            Quickstart (live server with RCON):
              python data-cli.py pregen --password secret --radius 2048
        """),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("freeze", parents=[shared], help="Apply world-freeze gamerules")
    sub.add_parser("unfreeze", parents=[shared], help="Restore default gamerules")
    sub.add_parser(
        "pregen",
        parents=[shared, pregen_args],
        help="Freeze world + run Chunky pregeneration (with progress polling)",
    )
    p_voxy = sub.add_parser(
        "voxy-import",
        parents=[shared],
        help="Send /voxy import world <name> via RCON",
    )
    p_voxy.add_argument(
        "--world-name",
        required=True,
        metavar="NAME",
        help="Minecraft world save folder name (e.g. 'New World')",
    )
    p_voxy.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECS",
        help="Max seconds to wait for import completion (default: 300)",
    )
    p_purge = sub.add_parser(
        "purge",
        help="Delete stale world + training data for a clean start (see --seed to change seed)",
    )
    p_purge.add_argument(
        "--server-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Fabric server runtime directory (auto-detected from repo layout by default)",
    )
    p_purge.add_argument(
        "--seed",
        default=None,
        metavar="SEED",
        help="New level-seed value to write to server.properties",
    )
    p_purge.add_argument(
        "--clean-data",
        action="store_true",
        help="Also delete the NPZ octree training data (data/voxy_octree/)",
    )
    p_purge.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help="NPZ data directory to clean when --clean-data is set",
    )
    p_purge.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    p_idp = sub.add_parser(
        "install-datapack",
        help="Install the freeze-on-load datapack into the server world (re-run after manual world wipe)",
    )
    p_idp.add_argument(
        "--server-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Fabric server runtime directory (auto-detected from repo layout by default)",
    )
    sub.add_parser("status", parents=[shared], help="Query Chunky pregeneration progress")

    # ---- harvest (automated voxy-import via DataHarvester mod) ----
    p_hv = sub.add_parser(
        "harvest",
        parents=[shared, pregen_args],
        help="Automated Voxy import: pregen + RCON teleport spiral + DB monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Automates the voxy-import step by:
              1. (Optional) Running Chunky pregeneration
              2. Waiting for a player/bot to connect
              3. Teleporting the player in a spiral pattern via RCON
              4. Monitoring the Voxy RocksDB until it stabilises

            Requires the DataHarvester mod installed on the client, or a
            manually-connected player. See: VoxelTree/tools/data-harvester/
        """),
    )
    p_hv.add_argument(
        "--step",
        type=int,
        default=256,
        metavar="BLOCKS",
        help="Teleport step size in blocks (default: 256)",
    )
    p_hv.add_argument(
        "--dwell",
        type=float,
        default=8.0,
        metavar="SECS",
        help="Seconds to dwell at each teleport position (default: 8)",
    )
    p_hv.add_argument(
        "--stable-seconds",
        type=int,
        default=60,
        metavar="SECS",
        help="DB must be stable for this many seconds (default: 60)",
    )
    p_hv.add_argument(
        "--skip-pregen",
        action="store_true",
        help="Skip Chunky pregeneration (chunks already exist)",
    )
    p_hv.add_argument(
        "--spiral-only",
        action="store_true",
        help="Skip pregen + player wait; just run spiral + monitor",
    )
    p_hv.add_argument(
        "--player-timeout",
        type=int,
        default=600,
        metavar="SECS",
        help="Max seconds to wait for player connection (default: 600)",
    )
    p_hv.add_argument(
        "--voxy-dir",
        type=Path,
        default=DEFAULT_VOXY_DIR,
        metavar="DIR",
        help="Voxy saves directory to monitor",
    )

    p_dn = sub.add_parser(
        "dumpnoise",
        parents=[shared, pregen_args],
        help="Consolidate Stage1 + SparseRoot noise dumps via RCON (both formats needed for training)",
    )
    p_dn.add_argument(
        "--timeout",
        type=int,
        default=3600,
        metavar="SECS",
        help="Max seconds to wait for each dump to complete (default: 3600)",
    )
    sub.add_parser(
        "info",
        parents=[pregen_args],
        help="Print pipeline plan (no server connection needed)",
    )

    # ---- unified dataprep command -----------------------------------------
    p_dp = sub.add_parser(
        "dataprep",
        parents=[shared, pregen_args],
        help="Run the full data-preparation pipeline (pregen → build-octree-pairs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Pipeline steps:
              pregen                 RCON — freeze world + Chunky pregeneration
              voxy-import            RCON — /voxy import world <name>
              dumpnoise              RCON — /dumpnoise stage1 + /dumpnoise sparse_root → all formats
              extract-octree         Voxy RocksDB → data/voxy_octree/level_N/*.npz (all LOD levels)
              column-heights-octree  Merge 5-plane heightmaps (32×32) into octree NPZs
              build-octree-pairs     Build parent/child octree training pairs

            Examples:
              # Most common: local extraction steps only:
              python data-cli.py dataprep --from-step extract-octree \\
                     --voxy-dir LODiffusion/run/saves

              # Full pipeline including RCON:
              python data-cli.py dataprep --from-step pregen \\
                     --password secret --world-name "New World" \\
                     --voxy-dir LODiffusion/run/saves

              # Just rebuild octree pairs:
              python data-cli.py dataprep --from-step build-octree-pairs \\
                     --data-dir data/voxy_octree
            """),
    )
    p_dp.add_argument(
        "--from-step",
        choices=DATAPREP_STEPS,
        default="extract-octree",
        metavar="STEP",
        help="Start the pipeline from this step (default: extract-octree).  "
        "Steps: " + ", ".join(DATAPREP_STEPS),
    )
    # Voxy-import args
    p_dp.add_argument(
        "--world-name",
        default="",
        metavar="NAME",
        help="Minecraft world save folder name for voxy-import (e.g. 'New World')",
    )
    p_dp.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECS",
        help="Max seconds to wait for voxy import completion (default: 300)",
    )
    # Extract args
    p_dp.add_argument(
        "--voxy-dir",
        type=Path,
        default=DEFAULT_VOXY_DIR,
        metavar="DIR",
        help="Minecraft saves directory containing Voxy databases",
    )
    p_dp.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output / working directory for NPZ data (default: data/voxy_octree)",
    )
    p_dp.add_argument(
        "--min-solid",
        type=float,
        default=0.02,
        metavar="F",
        help="Min fraction of non-air blocks to keep a chunk (default: 0.02)",
    )
    p_dp.add_argument(
        "--max-sections",
        type=int,
        default=None,
        metavar="N",
        help="Limit extraction to N chunk sections (optional)",
    )
    # Column-heights / dumpnoise args
    p_dp.add_argument(
        "--noise-dump-dir",
        type=Path,
        default=DEFAULT_NOISE_DUMP_DIR,
        metavar="DIR",
        help="Directory containing /dumpnoise JSON files (default: LODiffusion/run/noise_dumps)",
    )
    # Build-pairs args
    p_dp.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        metavar="F",
        help="Fraction of data reserved for validation (default: 0.1)",
    )
    p_dp.add_argument(
        "--clean",
        action="store_true",
        help="Delete old pair caches before building new ones",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # dataprep uses args directly (it builds its own PipelineConfig for RCON)
    if args.subcommand == "dataprep":
        cmd_dataprep(args)
        return

    cfg = PipelineConfig(
        host=getattr(args, "host", "localhost"),
        port=getattr(args, "port", 25575),
        password=getattr(args, "password", ""),
        dry_run=getattr(args, "dry_run", False),
        world=getattr(args, "world", "minecraft:overworld"),
        center_x=getattr(args, "center", [0, 0])[0],
        center_z=getattr(args, "center", [0, 0])[1],
        radius=getattr(args, "radius", 2048),
        verbose=getattr(args, "verbose", False),
        voxy_import_world=getattr(args, "world_name", ""),
        voxy_import_timeout=getattr(args, "timeout", 300),
    )

    dispatch = {
        "voxy-import": cmd_voxy_import,
        "dumpnoise": cmd_dumpnoise,
    }
    dispatch[args.subcommand](cfg)


if __name__ == "__main__":
    main()
