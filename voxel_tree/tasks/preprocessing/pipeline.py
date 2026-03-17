#!/usr/bin/env python3
"""pipeline.py — Training + deployment pipeline for VoxelTree.

Data preparation (extract-octree → column-heights-octree → build-octree-pairs,
optionally preceded by RCON pregen/voxy-import) lives in **data-cli.py**.
Use ``python data-cli.py dataprep --from-step <step>`` for data prep.

This file handles the MODEL TRAINING & DEPLOYMENT pipeline:

  Phase 2: TRAINING
    └─ Input:  data/voxy_octree/{train,val}_octree_pairs.npz  (from data-cli.py)
    └─ Output: models/<dir>/best_model.pt
    └─ Command: python pipeline.py train --epochs <N>

  Phase 3: EXPORT (ONNX conversion)
    └─ Input:  checkpoint (best_model.pt)
    └─ Output: production/{octree_init,octree_refine,octree_leaf}.onnx + config
    └─ Command: python pipeline.py export --checkpoint <path>

  Phase 4: DEPLOY
    └─ Input:  production/pipeline_manifest.json + ONNX files
    └─ Output: ../LODiffusion/config/lodiffusion/ (3 ONNX models + config)
    └─ Command: python pipeline.py deploy

Typical workflow (end-to-end):
  1. Prepare data:  python data-cli.py dataprep --from-step extract-octree ...
  2. Train model:   python pipeline.py run --epochs 20 --export --deploy
     (This chains: train → export → deploy)

The ``run`` meta-command delegates data preparation to *data-cli.py*
then runs train [+ export + deploy].

Usage::

    # Full pipeline: data-prep + train + (optional) export
    python pipeline.py run \\
        --voxy-dir "C:/path/to/LODiffusion/run/saves" \\
        --epochs 20

    # Train only (requires data/voxy_octree/*_octree_pairs.npz)
    python pipeline.py train --epochs 20

    # Export ONNX after training
    python pipeline.py export --checkpoint models/voxy_octree/best_model.pt

    # Deploy to LODiffusion
    python pipeline.py deploy --export-dir production/latest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------
# Path anchors — pipeline.py lives at VoxelTree/VoxelTree/preprocessing/
# ------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent.parent  # VoxelTree/VoxelTree/ (the package)
_REPO_ROOT = _PKG_DIR.parent  # VoxelTree/ (repo root)

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

VOXY_VOCAB_PATH = _PKG_DIR / "config" / "voxy_vocab.json"
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "voxy_octree"
DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "voxy_octree"
DEFAULT_EXPORT_DIR = _REPO_ROOT / "production"


# ==================================================================
# PHASE 2: TRAINING — *_octree_pairs.npz → model checkpoint
# ==================================================================


def phase2_train(
    data_dir: Path,
    model_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "auto",
) -> Path | None:
    """Phase 2: Train the octree model on extracted data.

    Returns the path to the best checkpoint, or None on failure.
    """
    print()
    print("=" * 70)
    print("  PHASE 2: Train octree model (%d epochs)" % epochs)
    print("=" * 70)
    print()

    # Check for octree pair caches
    train_cache = data_dir / "train_octree_pairs.npz"
    if not train_cache.exists():
        print("ERROR: No training pair cache found at %s" % train_cache)
        print("Run: voxel-tree dataprep --from-step extract-octree ...")
        return None
    print("Training data: %s" % data_dir)

    from voxel_tree.tasks.octree.train import main as _train_main

    train_args = [
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
        "--device",
        device,
        "--save-every",
        "5",
        "--validate-every",
        "5",
    ]

    t0 = time.time()
    try:
        _train_main(train_args)
        success = True
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Training raised exception: %s" % exc)
        success = False
    elapsed = time.time() - t0

    if not success:
        print("ERROR: Training failed")
        return None

    best = model_dir / "best_model.pt"
    if not best.exists():
        best = model_dir / "final_model.pt"

    print()
    print("Phase 2 complete: trained for %d epochs in %.1f min" % (epochs, elapsed / 60))
    if best.exists():
        print("Best checkpoint: %s" % best)
    return best if best.exists() else None


# ==================================================================
# PHASE 3: EXPORT — checkpoint → 3 ONNX models
# ==================================================================


def phase3_export(
    checkpoint: Path | None,
    export_dir: Path,
    checkpoint_dir: Path | None = None,
    models: list[str] | None = None,
) -> bool:
    """Phase 3: Export 3 ONNX models (init, refine, leaf) for LODiffusion.

    Returns True on success.
    """
    print()
    print("=" * 70)
    print("  PHASE 3: Export ONNX models")
    print("=" * 70)
    print()

    if checkpoint is None and checkpoint_dir is None:
        print("ERROR: either --checkpoint or --checkpoint-dir must be provided")
        return False
    if checkpoint is not None and not checkpoint.exists():
        print("ERROR: Checkpoint not found: %s" % checkpoint)
        return False
    if checkpoint_dir is not None and not checkpoint_dir.exists():
        print("ERROR: Checkpoint directory not found: %s" % checkpoint_dir)
        return False

    from voxel_tree.tasks.octree.export import main as _export_main

    try:
        args_list: list[str] = ["--out-dir", str(export_dir)]
        if checkpoint is not None:
            args_list += ["--checkpoint", str(checkpoint)]
        if checkpoint_dir is not None:
            args_list += ["--checkpoint-dir", str(checkpoint_dir)]
        if models:
            args_list += ["--models"] + list(models)
        _export_main(args_list)
    except Exception as exc:  # noqa: BLE001
        print("ERROR: ONNX export raised exception: %s" % exc)
        return False

    # Verify expected outputs
    expected = ["octree_init.onnx", "octree_refine.onnx", "octree_leaf.onnx"]
    missing = [f for f in expected if not (export_dir / f).exists()]
    if missing:
        print("WARNING: Expected ONNX files not found: %s" % missing)
    else:
        for f in expected:
            size_mb = (export_dir / f).stat().st_size / (1024 * 1024)
            print("Exported: %s (%.1f MB)" % (export_dir / f, size_mb))
    return True


# ==================================================================
# PHASE 4: DEPLOY — production dir → LODiffusion config dir
# ==================================================================


def phase4_deploy(
    export_dir: Path, dest: Path | None = None, models: list[str] | None = None
) -> bool:
    """Phase 4: Deploy ONNX models to LODiffusion.

    Deploy is split into per-model scripts (init/refine/leaf). Each model has a
    dedicated deploy entry point so the GUI can run them independently.

    If ``models`` is None, all submodels will be deployed.
    """
    print()
    print("=" * 70)
    print("  PHASE 4: Deploy to LODiffusion")
    print("=" * 70)
    print()

    if models is None:
        models = ["init", "refine", "leaf"]

    def _deploy_model(model_name: str) -> None:
        module_name = f"voxel_tree.tasks.octree.deploy_{model_name}"
        module = __import__(module_name, fromlist=["main"])
        deploy_main = getattr(module, "main")

        args: list[str] = [str(export_dir)]
        if dest is not None:
            args += ["--dest", str(dest)]
        deploy_main(args)

    try:
        for model in models:
            _deploy_model(model)
        return True
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Deploy raised exception: %s" % exc)
        return False


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _run_dataprep(args: argparse.Namespace) -> None:
    """Delegate data preparation to VoxelTree.preprocessing.cli."""
    from voxel_tree.preprocessing.cli import main as _cli_main

    from_step = "voxy-import" if args.voxy_import_world else "extract-octree"

    dataprep_args = [
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
        dataprep_args.extend(["--max-sections", str(args.max_sections)])
    if args.clean:
        dataprep_args.append("--clean")
    if args.voxy_import_world:
        dataprep_args.extend(
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

    try:
        _cli_main(dataprep_args)
    except SystemExit as exc:
        if exc.code != 0:
            print("Data preparation failed — aborting")
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="VoxelTree octree training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train ---
    p_train = sub.add_parser("train", help="Train octree model (requires pair caches)")
    p_train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--device", type=str, default="auto")

    # -- export ---
    p_exp = sub.add_parser("export", help="Export 3 ONNX models (init, refine, leaf)")
    ckpt_group = p_exp.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", type=Path, help="Unified checkpoint file (.pt)")
    ckpt_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing per-model checkpoints (init_/refine_/leaf_*.pt)",
    )
    p_exp.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_exp.add_argument(
        "--models",
        nargs="+",
        choices=["init", "refine", "leaf"],
        help="Only export the specified submodels (default: all)",
    )

    # -- deploy ---
    p_dep = sub.add_parser("deploy", help="Copy ONNX to LODiffusion config (via deploy_models.py)")
    p_dep.add_argument(
        "export_dir", type=Path, help="Production directory with pipeline_manifest.json"
    )
    p_dep.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination (default: ../LODiffusion/run/config/lodiffusion)",
    )
    p_dep.add_argument(
        "--models",
        nargs="+",
        choices=["init", "refine", "leaf"],
        help="Only deploy the specified submodels (default: all)",
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
    p_run.add_argument("--device", type=str, default="auto")
    p_run.add_argument("--min-solid", type=float, default=0.02)
    p_run.add_argument("--max-sections", type=int, default=None)
    p_run.add_argument("--clean", action="store_true")
    p_run.add_argument("--val-split", type=float, default=0.1)
    p_run.add_argument("--export", action="store_true", help="Also export ONNX after training")
    p_run.add_argument(
        "--deploy", action="store_true", help="Also deploy to LODiffusion after export"
    )
    p_run.add_argument(
        "--models",
        nargs="+",
        choices=["init", "refine", "leaf"],
        help="Restrict export/deploy to the specified submodels when used with --export or --deploy",
    )
    p_run.add_argument(
        "--lodiffusion-config",
        type=Path,
        default=None,
        help="Destination for deploy step (default: ../LODiffusion/run/config/lodiffusion)",
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

    args = parser.parse_args(argv)

    if args.command == "train":
        phase2_train(
            args.data_dir,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

    elif args.command == "export":
        phase3_export(
            args.checkpoint,
            args.export_dir,
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
            models=args.models,
        )

    elif args.command == "deploy":
        phase4_deploy(args.export_dir, args.dest, models=args.models)

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
            device=args.device,
        )
        if best is None:
            print("Training failed — aborting")
            sys.exit(1)

        if args.export:
            ok = phase3_export(best, args.export_dir, models=args.models)
            if ok and args.deploy:
                phase4_deploy(args.export_dir, args.lodiffusion_config, models=args.models)

    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
