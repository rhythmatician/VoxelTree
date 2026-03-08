#!/usr/bin/env python3
"""pipeline.py — Data-prep + training pipeline for VoxelTree.

Stage 1 — Data preparation (idempotent, world-locked):
  Phase 1a  extract:         Voxy RocksDB  →  data/voxy/*.npz (raw LOD0 chunks)
  Phase 1a½ column-heights:  Enrich NPZs with column-level heightmap_surface
  Phase 1b  build-pairs:     data/voxy/    →  data/voxy/*_pairs_v1.npz
                             Runs the Mipper once; 4 LOD transitions per chunk.

Stage 2 — Training:
  Phase 2   train:           *_pairs_v1.npz  →  model weights
  Phase 3   export:          checkpoint      →  production/model.onnx

The pipeline supports:
- Single-shot: extract once, build-pairs once, train once
- Iterative: extract → build-pairs → train → delete → new seed → repeat (future)

Usage::

    # Full pipeline: extract + build-pairs + train + (optional) export
    python pipeline.py run \\
        --voxy-dir "C:/path/to/LODiffusion/run/saves" \\
        --epochs 20

    # Data-prep only (stages 1a + 1b)
    python pipeline.py extract  --voxy-dir "C:/path/to/LODiffusion/run/saves"
    python pipeline.py build-pairs

    # Train only (stage 2, requires data/voxy/*_pairs_v1.npz)
    python pipeline.py train --epochs 20

    # Export ONNX after training
    python pipeline.py export --checkpoint models/voxy/best_model.pt

    # Full pipeline including ONNX export
    python pipeline.py run --voxy-dir "..." --epochs 20 --export
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

VOXY_VOCAB_PATH = Path("config/voxy_vocab.json")
DEFAULT_DATA_DIR = Path("data/voxy")
DEFAULT_MODEL_DIR = Path("models/voxy")
DEFAULT_EXPORT_DIR = Path("production")

# Pair-cache files written by build-pairs and consumed by MultiLODDataset
PAIRS_CACHE_GLOB = "*_pairs_v1.npz"


def find_voxy_databases(saves_dir: Path) -> list[Path]:
    """Discover all Voxy storage directories under a Minecraft saves folder."""
    dbs = sorted(saves_dir.glob("*/voxy/*/storage"))
    return [d for d in dbs if d.is_dir()]


def phase1_extract(
    voxy_dir: Path,
    data_dir: Path,
    *,
    min_solid: float = 0.02,
    max_sections: int | None = None,
    clean_first: bool = False,
) -> int:
    """Phase 1: Extract training data from Voxy RocksDB databases.

    Returns the number of training chunks extracted.
    """
    print()
    print("=" * 70)
    print("  PHASE 1: Extract training data from Voxy")
    print("=" * 70)
    print()

    if clean_first and data_dir.exists():
        print("Cleaning previous extraction: %s" % data_dir)
        shutil.rmtree(data_dir)

    # Find all Voxy databases
    dbs = find_voxy_databases(voxy_dir)
    if not dbs:
        print("ERROR: No Voxy databases found under %s" % voxy_dir)
        print("Expected structure: <saves>/<world>/voxy/<hash>/storage/")
        return 0

    print("Found %d Voxy database(s):" % len(dbs))
    for db in dbs:
        # Extract world name from path
        world_name = db.parts[-4] if len(db.parts) >= 4 else str(db)
        print("  - %s" % world_name)
    print()

    # Build CLI args for extract script
    cmd = [
        sys.executable,
        "scripts/extract_voxy_training_data.py",
        *[str(d) for d in dbs],
        "--output-dir",
        str(data_dir),
        "--vocab",
        str(VOXY_VOCAB_PATH),
        "--min-solid",
        str(min_solid),
    ]
    if max_sections is not None:
        cmd.extend(["--max-sections", str(max_sections)])

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("ERROR: Extraction failed with exit code %d" % result.returncode)
        return 0

    # Count output files
    n_files = len(list(data_dir.glob("*.npz"))) if data_dir.exists() else 0
    print()
    print(
        "Phase 1 complete: %d chunks in %.1fs (%.0f chunks/s)"
        % (n_files, elapsed, n_files / max(elapsed, 0.01))
    )
    return n_files


def phase1a_column_heights(data_dir: Path) -> bool:
    """Phase 1a½: Compute column-level surface heights for all extracted NPZs.

    This enriches each NPZ with ``heightmap_surface`` and ``heightmap_ocean_floor``
    (world-Y coordinates) by scanning all Y-slabs at each (x, z) column.  Must
    run after extraction and before build-pairs.

    Returns True on success.
    """
    print()
    print("=" * 70)
    print("  PHASE 1a½: Compute column-level surface heights")
    print("=" * 70)
    print()

    cmd = [
        sys.executable,
        "scripts/add_column_heights.py",
        str(data_dir),
    ]

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print("ERROR: add_column_heights failed with exit code %d" % result.returncode)
        return False
    return True


def phase1b_build_pairs(
    data_dir: Path,
    *,
    val_split: float = 0.1,
    min_solid: float = 0.02,
    clean_first: bool = False,
) -> int:
    """Phase 1b: Pre-compute LOD pair caches from extracted NPZ chunks.

    Returns the total number of training pairs built, or 0 on failure.
    The Mipper (3-D volume downsampling) runs exactly once per chunk here
    instead of being repeated on every training epoch.
    """
    print()
    print("=" * 70)
    print("  PHASE 1b: Build LOD training pairs")
    print("=" * 70)
    print()

    n_source = len(list(data_dir.glob("*.npz")))
    # Exclude cache files from count
    n_source -= len(list(data_dir.glob(PAIRS_CACHE_GLOB)))
    if n_source <= 0:
        print("ERROR: No source NPZ files in %s — run extraction first" % data_dir)
        return 0
    print("Source chunks: %d  in %s" % (n_source, data_dir))

    cmd = [
        sys.executable,
        "scripts/build_pairs.py",
        "--data-dir",
        str(data_dir),
        "--val-split",
        str(val_split),
        "--min-solid",
        str(min_solid),
    ]
    if clean_first:
        cmd.append("--clean")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("ERROR: build-pairs failed with exit code %d" % result.returncode)
        return 0

    # Count pairs from written cache files
    cache_files = list(data_dir.glob(PAIRS_CACHE_GLOB))
    if not cache_files:
        return 0
    print()
    print("Phase 1b complete in %.1fs" % elapsed)
    return sum(1 for _ in cache_files)  # success indicator


def phase2_train(
    data_dir: Path,
    model_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    base_channels: int = 32,
    device: str = "auto",
    surface_loss_weight: float = 0.1,
) -> Path | None:
    """Phase 2: Train the model on extracted data.

    Returns the path to the best checkpoint, or None on failure.
    """
    print()
    print("=" * 70)
    print("  PHASE 2: Train model (%d epochs)" % epochs)
    print("=" * 70)
    print()

    n_files = len(list(data_dir.glob("*.npz")))
    if n_files == 0:
        print("ERROR: No training data in %s — run extraction first" % data_dir)
        return None
    print("Training data: %d chunks in %s" % (n_files, data_dir))

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(model_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--base-channels",
        str(base_channels),
        "--device",
        device,
        "--surface-loss-weight",
        str(surface_loss_weight),
        "--save-every",
        "5",
        "--validate-every",
        "5",
        "--vocab",
        str(VOXY_VOCAB_PATH),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("ERROR: Training failed with exit code %d" % result.returncode)
        return None

    best = model_dir / "best_model.pt"
    if not best.exists():
        # Fall back to final
        best = model_dir / "final_model.pt"

    print()
    print("Phase 2 complete: trained for %d epochs in %.1f min" % (epochs, elapsed / 60))
    if best.exists():
        print("Best checkpoint: %s" % best)
    return best if best.exists() else None


def phase3_export(
    checkpoint: Path,
    export_dir: Path,
) -> Path | None:
    """Phase 3: Export ONNX model for LODiffusion.

    Config is read from the checkpoint — no external YAML needed.
    Returns the path to the exported ONNX file, or None on failure.
    """
    print()
    print("=" * 70)
    print("  PHASE 3: Export ONNX model")
    print("=" * 70)
    print()

    if not checkpoint.exists():
        print("ERROR: Checkpoint not found: %s" % checkpoint)
        return None

    cmd = [
        sys.executable,
        "scripts/export_lod.py",
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(export_dir),
    ]

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print("ERROR: ONNX export failed with exit code %d" % result.returncode)
        return None

    onnx_path = export_dir / "model.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print("Exported: %s (%.1f MB)" % (onnx_path, size_mb))
        return onnx_path
    return None


def deploy_onnx(onnx_dir: Path, lodiffusion_config: Path) -> bool:
    """Copy exported ONNX model + config to LODiffusion's config directory."""
    print()
    print("Deploying to LODiffusion...")
    src_onnx = onnx_dir / "model.onnx"
    src_config = onnx_dir / "model_config.json"

    if not src_onnx.exists():
        print("ERROR: No model.onnx in %s" % onnx_dir)
        return False

    lodiffusion_config.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_onnx, lodiffusion_config / "model.onnx")
    if src_config.exists():
        shutil.copy2(src_config, lodiffusion_config / "model_config.json")

    print("Deployed to %s" % lodiffusion_config)
    return True


# ------------------------------------------------------------------
# Iterative training (future expansion)
# ------------------------------------------------------------------


def run_iterative(
    voxy_dir: Path,
    data_dir: Path,
    model_dir: Path,
    *,
    cycles: int = 1,
    epochs_per_cycle: int = 20,
    clean_between: bool = True,
    **train_kwargs,
) -> None:
    """Run the extract→train loop for multiple cycles.

    For now, this just runs one cycle. In the future, each cycle could:
    1. Generate a new Minecraft world with a fresh seed
    2. Run Voxy import
    3. Extract training data
    4. Continue training from previous checkpoint
    5. Delete training data to free disk space
    6. Repeat
    """
    for cycle in range(cycles):
        print()
        print("#" * 70)
        print("  CYCLE %d / %d" % (cycle + 1, cycles))
        print("#" * 70)

        n = phase1_extract(
            voxy_dir,
            data_dir,
            clean_first=clean_between and cycle > 0,
        )
        if n == 0:
            print("No data extracted — stopping")
            break

        # On subsequent cycles, resume from previous checkpoint
        # TODO: pass resume_from to phase2_train when checkpoint resumption is implemented
        # resume_from = model_dir / "best_model.pt" if cycle > 0 else None

        best = phase2_train(
            data_dir,
            model_dir,
            epochs=epochs_per_cycle,
            **train_kwargs,
        )
        if best is None:
            print("Training failed — stopping")
            break

        if clean_between:
            print("Cleaning training data to free disk space...")
            shutil.rmtree(data_dir, ignore_errors=True)

    print()
    print("All cycles complete.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxelTree training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract ---
    p_ext = sub.add_parser("extract", help="Phase 1a: Extract data from Voxy")
    p_ext.add_argument(
        "--voxy-dir",
        type=Path,
        required=True,
        help="Minecraft saves directory (e.g. LODiffusion/run/saves)",
    )
    p_ext.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_ext.add_argument("--min-solid", type=float, default=0.02)
    p_ext.add_argument("--max-sections", type=int, default=None)
    p_ext.add_argument(
        "--clean", action="store_true", help="Delete existing data before extraction"
    )

    # -- column-heights ---
    p_ch = sub.add_parser(
        "column-heights",
        help="Phase 1a½: Compute column-level surface heights in extracted NPZs",
    )
    p_ch.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    # -- build-pairs ---
    p_bp = sub.add_parser(
        "build-pairs",
        help="Phase 1b: Pre-compute LOD pair caches from extracted NPZ chunks",
    )
    p_bp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_bp.add_argument("--val-split", type=float, default=0.1)
    p_bp.add_argument("--min-solid", type=float, default=0.02)
    p_bp.add_argument(
        "--clean", action="store_true", help="Delete existing pair caches before rebuilding"
    )

    # -- train ---
    p_train = sub.add_parser("train", help="Phase 2: Train model")
    p_train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--base-channels", type=int, default=32)
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--surface-loss-weight", type=float, default=0.1)

    # -- export ---
    p_exp = sub.add_parser("export", help="Phase 3: Export ONNX")
    p_exp.add_argument("--checkpoint", type=Path, required=True)
    p_exp.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)

    # -- deploy ---
    p_dep = sub.add_parser("deploy", help="Copy ONNX to LODiffusion config")
    p_dep.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_dep.add_argument(
        "--lodiffusion-config", type=Path, default=Path("../LODiffusion/config/lodiffusion")
    )

    # -- run (full pipeline) ---
    p_run = sub.add_parser("run", help="Full pipeline: extract + train [+ export]")
    p_run.add_argument("--voxy-dir", type=Path, required=True)
    p_run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_run.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_run.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_run.add_argument("--epochs", type=int, default=20)
    p_run.add_argument("--batch-size", type=int, default=4)
    p_run.add_argument("--lr", type=float, default=1e-4)
    p_run.add_argument("--base-channels", type=int, default=32)
    p_run.add_argument("--device", type=str, default="auto")
    p_run.add_argument("--surface-loss-weight", type=float, default=0.1)
    p_run.add_argument("--min-solid", type=float, default=0.02)
    p_run.add_argument("--max-sections", type=int, default=None)
    p_run.add_argument("--clean", action="store_true")
    p_run.add_argument("--val-split", type=float, default=0.1)
    p_run.add_argument("--export", action="store_true", help="Also export ONNX after training")
    p_run.add_argument(
        "--deploy", action="store_true", help="Also deploy to LODiffusion after export"
    )
    p_run.add_argument(
        "--lodiffusion-config", type=Path, default=Path("../LODiffusion/config/lodiffusion")
    )

    args = parser.parse_args()

    if args.command == "extract":
        phase1_extract(
            args.voxy_dir,
            args.data_dir,
            min_solid=args.min_solid,
            max_sections=args.max_sections,
            clean_first=args.clean,
        )

    elif args.command == "column-heights":
        if not phase1a_column_heights(args.data_dir):
            sys.exit(1)

    elif args.command == "build-pairs":
        phase1b_build_pairs(
            args.data_dir,
            val_split=args.val_split,
            min_solid=args.min_solid,
            clean_first=args.clean,
        )

    elif args.command == "train":
        phase2_train(
            args.data_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            device=args.device,
            surface_loss_weight=args.surface_loss_weight,
        )

    elif args.command == "export":
        phase3_export(args.checkpoint, args.export_dir)

    elif args.command == "deploy":
        deploy_onnx(args.export_dir, args.lodiffusion_config)

    elif args.command == "run":
        # Stage 1a: extract
        n = phase1_extract(
            args.voxy_dir,
            args.data_dir,
            min_solid=args.min_solid,
            max_sections=args.max_sections,
            clean_first=args.clean,
        )
        if n == 0:
            print("No data extracted — aborting")
            sys.exit(1)

        # Stage 1a½: column-level heights
        if not phase1a_column_heights(args.data_dir):
            print("Column heights failed — aborting")
            sys.exit(1)

        # Stage 1b: build pairs
        result = phase1b_build_pairs(
            args.data_dir,
            val_split=args.val_split,
            min_solid=args.min_solid,
        )
        if result == 0:
            print("Pair build failed — aborting")
            sys.exit(1)

        # Stage 2: train
        best = phase2_train(
            args.data_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            device=args.device,
            surface_loss_weight=args.surface_loss_weight,
        )
        if best is None:
            print("Training failed — aborting")
            sys.exit(1)

        if args.export:
            onnx = phase3_export(best, args.export_dir)
            if onnx and args.deploy:
                deploy_onnx(args.export_dir, args.lodiffusion_config)

    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
