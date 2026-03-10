#!/usr/bin/env python3
"""pipeline.py — Training + deployment pipeline for VoxelTree.

Data preparation (extract → column-heights → build-pairs, optionally
preceded by RCON pregen/voxy-import) now lives in **data-cli.py**.
Use ``python data-cli.py dataprep --from-step <step>`` for data prep.

This file handles the MODEL TRAINING & DEPLOYMENT pipeline:

  Phase 2: TRAINING
    └─ Input:  data/voxy/*_pairs_v2.npz (from data-cli.py)
    └─ Output: models/<dir>/best_model.pt
    └─ Command: python pipeline.py train --epochs <N>

  Phase 3: EXPORT (ONNX conversion)
    └─ Input:  checkpoint (best_model.pt)
    └─ Output: production/model.onnx + model_config.json
    └─ Command: python pipeline.py export --checkpoint <path>

  Phase 4: DEPLOY
    └─ Input:  production/model.onnx + model_config.json
    └─ Output: ../LODiffusion/config/lodiffusion/(model.onnx + config)
    └─ Command: python pipeline.py deploy

Typical workflow (end-to-end):
  1. Prepare data:  python data-cli.py dataprep --from-step extract ...
  2. Train model:   python pipeline.py run --epochs 20 --export --deploy
     (This chains: train → export → deploy)

The ``run`` meta-command delegates data preparation to *data-cli.py*
then runs train [+ export + deploy].

Usage::

    # Full pipeline: data-prep + train + (optional) export
    python pipeline.py run \\
        --voxy-dir "C:/path/to/LODiffusion/run/saves" \\
        --epochs 20

    # Train only (requires data/voxy/*_pairs_v2.npz)
    python pipeline.py train --epochs 20

    # Export ONNX after training
    python pipeline.py export --checkpoint models/voxy/best_model.pt

    # Data prep (use data-cli.py instead):
    python data-cli.py dataprep --from-step extract \\
        --voxy-dir LODiffusion/run/saves --data-dir data/voxy
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


# ------------------------------------------------------------------
# Data-preparation functions have moved to data-cli.py.
# Use:  python data-cli.py dataprep --from-step <step>
# ------------------------------------------------------------------

# ==================================================================
# PHASE 2: TRAINING — *_pairs_v2.npz → model checkpoint
# ==================================================================


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
    steps: list[str] | None = None,
) -> Path | None:
    """Phase 2: Train the model on extracted data.

    Args:
        steps: If set, only train these LOD steps (e.g. ['init_to_lod4', 'lod4to3']).
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
    if steps is not None:
        cmd.extend(["--steps"] + steps)

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


# ==================================================================
# PHASES 3–4: EXPORT & DEPLOY (always run together)
# —  checkpoint → ONNX → LODiffusion config
# ==================================================================


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
# CLI
# ------------------------------------------------------------------


def _run_dataprep(args: argparse.Namespace) -> None:
    """Delegate data preparation to data-cli.py dataprep."""
    from_step = "voxy-import" if args.voxy_import_world else "extract"

    cmd = [
        sys.executable,
        "data-cli.py",
        "dataprep",
        "--from-step",
        from_step,
        "--voxy-dir",
        str(args.voxy_dir),
        "--data-dir",
        str(args.data_dir),
        "--min-solid",
        str(args.min_solid),
        "--val-split",
        str(args.val_split),
    ]
    if args.max_sections is not None:
        cmd.extend(["--max-sections", str(args.max_sections)])
    if args.clean:
        cmd.append("--clean")
    # RCON args (needed for voxy-import / pregen steps)
    if args.voxy_import_world:
        cmd.extend(
            [
                "--world-name",
                args.voxy_import_world,
                "--password",
                args.rcon_password,
                "--host",
                args.rcon_host,
                "--port",
                str(args.rcon_port),
                "--timeout",
                str(args.rcon_timeout),
            ]
        )

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print("Data preparation failed — aborting")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxelTree training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train ---
    p_train = sub.add_parser("train", help="Train model (requires pair caches)")
    p_train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--base-channels", type=int, default=32)
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--surface-loss-weight", type=float, default=0.1)
    p_train.add_argument(
        "--steps",
        nargs="*",
        default=None,
        metavar="STEP",
        help="Train specific LOD steps only (e.g. --steps init_to_lod4 lod4to3)",
    )

    # -- export ---
    p_exp = sub.add_parser("export", help="Export ONNX model")
    p_exp.add_argument("--checkpoint", type=Path, required=True)
    p_exp.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)

    # -- deploy ---
    p_dep = sub.add_parser("deploy", help="Copy ONNX to LODiffusion config")
    p_dep.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_dep.add_argument(
        "--lodiffusion-config", type=Path, default=Path("../LODiffusion/config/lodiffusion")
    )

    # -- run (full pipeline: dataprep + train [+ export + deploy]) ---
    p_run = sub.add_parser(
        "run",
        help="Full pipeline: dataprep → train [→ export → deploy]",
        epilog="Data prep is delegated to data-cli.py dataprep.",
    )
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
    p_run.add_argument(
        "--steps",
        nargs="*",
        default=None,
        metavar="STEP",
        help="Train specific LOD steps only (e.g. --steps init_to_lod4 lod4to3)",
    )
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
    # RCON args for optional voxy-import in the full pipeline
    p_run.add_argument(
        "--voxy-import-world",
        metavar="NAME",
        default=None,
        help="If set, start dataprep from voxy-import (requires RCON)",
    )
    p_run.add_argument("--rcon-host", default="localhost", metavar="HOST")
    p_run.add_argument("--rcon-port", type=int, default=25575, metavar="PORT")
    p_run.add_argument("--rcon-password", default="", metavar="PASS")
    p_run.add_argument("--rcon-timeout", type=int, default=300, metavar="SECS")

    args = parser.parse_args()

    if args.command == "train":
        phase2_train(
            args.data_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            device=args.device,
            surface_loss_weight=args.surface_loss_weight,
            steps=args.steps,
        )

    elif args.command == "export":
        phase3_export(args.checkpoint, args.export_dir)

    elif args.command == "deploy":
        deploy_onnx(args.export_dir, args.lodiffusion_config)

    elif args.command == "run":
        # Data preparation → data-cli.py dataprep
        _run_dataprep(args)

        # Training
        best = phase2_train(
            args.data_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            device=args.device,
            surface_loss_weight=args.surface_loss_weight,
            steps=args.steps,
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
