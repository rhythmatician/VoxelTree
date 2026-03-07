"""Train + export the first usable VoxelTree ONNX in one shot.

Usage
-----
  python make_first_model.py                      # 20 epochs, out → production/
  python make_first_model.py --epochs 50          # more training
  python make_first_model.py --checkpoint <path>  # skip training, just export
  python make_first_model.py --out-dir deploy/    # choose output directory

After completion you will find:
  <out-dir>/model.onnx
  <out-dir>/model_config.json
  <out-dir>/test_vectors.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Allow importing from train.* and scripts.* when run from the repo root
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_lod import LODiffusionAdapter  # noqa: E402
from scripts.export_lod import build_model, export_contract, load_checkpoint  # noqa: E402
from train.trainer import VoxelTrainer  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_best_checkpoint(runs_dir: Path) -> Path:
    """Return the best_model.pt from the most recently created run subdirectory."""
    run_dirs = sorted(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")
    best = run_dirs[0] / "best_model.pt"
    if not best.exists():
        # Fall back to final_checkpoint
        best = run_dirs[0] / "final_checkpoint.pt"
    if not best.exists():
        # Any checkpoint in that dir
        cps = sorted(run_dirs[0].glob("checkpoint_epoch_*.pt"))
        if not cps:
            raise FileNotFoundError(f"No checkpoint found in {run_dirs[0]}")
        best = cps[-1]
    return best


def do_train(cfg: dict, epochs: int, resume: Path | None) -> None:
    """Train directly via VoxelTrainer + MultiLODDataset, save to runs/run_<ts>/."""
    from torch.utils.data import DataLoader

    from train.multi_lod_dataset import MultiLODDataset

    logger = logging.getLogger("make_first_model.train")

    cfg["training"]["epochs"] = epochs

    data_cfg = cfg["data"]
    processed_dir = Path(data_cfg["processed_data_dir"])
    batch_size = cfg["training"]["batch_size"]

    train_ds = MultiLODDataset(processed_dir / "train")
    val_ds = MultiLODDataset(processed_dir / "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    run_dir = Path("runs") / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    trainer = VoxelTrainer(cfg)
    if resume:
        logger.info("Resuming from %s", resume)
        trainer.resume_from_checkpoint(resume)

    save_every = cfg["training"].get("save_every", 10)

    for epoch in range(trainer.current_epoch, epochs):
        trainer.model.train()
        train_metrics = trainer.train_one_epoch(train_loader)
        trainer.model.eval()
        val_metrics = trainer.validate_one_epoch(val_loader)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch + 1,
            epochs,
            train_metrics["loss"],
            val_metrics.get("val_loss", val_metrics.get("loss", float("nan"))),
        )

        if (epoch + 1) % save_every == 0:
            cp = run_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(cp, epoch + 1, train_metrics["loss"])
            logger.info("Saved periodic checkpoint: %s", cp)

        if train_metrics["loss"] < trainer.best_loss:
            trainer.best_loss = train_metrics["loss"]
            best = run_dir / "best_model.pt"
            trainer.save_checkpoint(best, epoch + 1, train_metrics["loss"])
            logger.info("New best checkpoint: %s (loss=%.4f)", best, train_metrics["loss"])

    final = run_dir / "final_checkpoint.pt"
    trainer.save_checkpoint(final, epochs, trainer.best_loss)
    logger.info("Final checkpoint: %s", final)


def do_export(checkpoint: Path, cfg: dict, out_dir: Path) -> Path:
    model = build_model(cfg)
    load_checkpoint(model, checkpoint)
    model.eval()
    biome_vocab = cfg["model"].get("biome_vocab_size", 256)
    adapter = LODiffusionAdapter(model, biome_vocab)
    return export_contract(adapter, cfg, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and export first VoxelTree ONNX model")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (ignored if --checkpoint supplied)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Skip training and export this checkpoint directly",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from this checkpoint before continuing",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("production"),
        help="Output directory for ONNX + config + test vectors",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("make_first_model")

    cfg = load_config(args.config)

    # ── Step 1: train (unless checkpoint supplied) ───────────────────────────
    if args.checkpoint is None:
        logger.info("=== STEP 1 / 2 — training for %d epochs ===", args.epochs)
        t0 = time.time()
        do_train(cfg, args.epochs, args.resume)
        logger.info("Training finished in %.1f s", time.time() - t0)

        checkpoint = find_best_checkpoint(Path("runs"))
        logger.info("Using checkpoint: %s", checkpoint)
    else:
        logger.info("=== Skipping training — using checkpoint %s ===", args.checkpoint)
        checkpoint = args.checkpoint

    # ── Step 2: export ───────────────────────────────────────────────────────
    logger.info("=== STEP 2 / 2 — exporting ONNX contract ===")
    onnx_path = do_export(checkpoint, cfg, args.out_dir)
    logger.info("ONNX model ready: %s", onnx_path)
    logger.info("Config sidecar:   %s", args.out_dir / "model_config.json")
    logger.info("Test vectors:     %s", args.out_dir / "test_vectors.npz")

    # ── Quick sanity check via onnxruntime ───────────────────────────────────
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        vecs = np.load(args.out_dir / "test_vectors.npz")
        outs = sess.run(
            None,
            {
                "x_parent": vecs["x_parent"],
                "x_biome": vecs["x_biome"],
                "x_height": vecs["x_height"],
                "x_lod": vecs["x_lod"],
            },
        )
        block_logits: np.ndarray = np.asarray(outs[0])
        air_mask: np.ndarray = np.asarray(outs[1])
        air_frac = float((air_mask > 0).mean())
        logger.info(
            "ORT sanity: block_logits %s  air_mask %s  air_frac=%.3f",
            block_logits.shape,
            air_mask.shape,
            air_frac,
        )
        if air_frac in (0.0, 1.0):
            logger.warning(
                "air_mask is all-%s — model may need more training.",
                "air" if air_frac == 1.0 else "solid",
            )
        else:
            logger.info("Model outputs non-trivial air/solid structure. Ready for LODiffusion!")
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ORT sanity check.")
    except Exception as e:
        logger.warning("ORT sanity check failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
