#!/usr/bin/env python3
"""Biome Classifier MLP — predict biome class from 6 climate RouterFields.

Architecture
------------
  Input  : 6 climate RouterField channels (temperature, vegetation, continents,
           erosion, depth, ridges) — one value per quart cell.
  Hidden : 64 → ReLU → 64 → ReLU
  Output : 54 logits (one per overworld biome), trained with CrossEntropyLoss

The model operates **per quart cell** — each sample is one (field_0 … field_5)
→ biome_class pair extracted from the v7 training data.

Training data
-------------
  Reads ``sparse_octree_pairs_v7.npz`` produced by ``build_sparse_octree_pairs.py``.
    noise_3d   : (N, 15, 4, 4, 4) float32
    biome_ids  : (N, 4, 4, 4)     int32

  We extract:
    - Input:   climate channels 0–5 from noise_3d → flatten to (N*64, 6)
    - Target:  biome_ids → flatten to (N*64,) long, remapped via biome_mapping

Model output
------------
  biome_classifier.pt    — PyTorch checkpoint
  biome_classifier.onnx  — ONNX export (opset 18, float32)

Usage
-----
  python -m voxel_tree.tasks.biome.train_biome_classifier \\
      --data noise_training_data/sparse_octree_pairs_v7.npz \\
      --epochs 200 --batch-size 4096 --lr 1e-3 \\
      --out-dir models/biome
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from voxel_tree.utils.router_field import CLIMATE_FIELDS
    from voxel_tree.utils.biome_mapping import OVERWORLD_BIOMES, UNKNOWN_BIOME_ID
except ImportError:
    CLIMATE_FIELDS = frozenset(range(6))
    UNKNOWN_BIOME_ID = 255
    OVERWORLD_BIOMES = [f"biome_{i}" for i in range(54)]

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

INPUT_SIZE = 6        # 6 climate RouterField channels
HIDDEN_SIZE = 64
NUM_BIOMES = len(OVERWORLD_BIOMES)  # 54

CLIMATE_INDICES = sorted(CLIMATE_FIELDS)  # [0, 1, 2, 3, 4, 5]


class BiomeClassifier(nn.Module):
    """6 → 64 → 64 → 54 MLP, ReLU hidden, logits output (no softmax)."""

    def __init__(self, num_classes: int = NUM_BIOMES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6) → (B, num_classes) logits"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _remap_biome_ids(raw_ids: np.ndarray) -> np.ndarray:
    """Remap raw Minecraft registry biome IDs (0–255) to contiguous 0–53, clamping unknowns.

    The data-harvester writes raw registry IDs.  We assume the registry
    mapping puts the 54 overworld biomes at IDs 0–53 (alphabetical order,
    matching biome_mapping.py).  IDs >= 54 or UNKNOWN_BIOME_ID (255) are
    mapped to a special 'drop' sentinel (-1) so they can be filtered out.
    """
    result = raw_ids.copy().astype(np.int64)
    mask = (result < 0) | (result >= NUM_BIOMES) | (result == UNKNOWN_BIOME_ID)
    result[mask] = -1  # sentinel for filtering
    return result


def load_data(npz_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load v7 training pairs and extract climate→biome samples.

    Returns (inputs, targets) where:
        inputs:  (M, 6) float32 — climate fields per quart cell
        targets: (M,)   int64   — biome class index [0, 53]
    Cells with unknown biomes (ID >= 54 or 255) are excluded.
    """
    print(f"  Loading {npz_path} ...")
    with np.load(npz_path) as data:
        noise_3d = data["noise_3d"]      # (N, 15, 4, 4, 4) float32
        biome_ids = data["biome_ids"]    # (N, 4, 4, 4) int32

    n = noise_3d.shape[0]
    assert noise_3d.shape[1] == 15, f"Expected 15 channels, got {noise_3d.shape[1]}"

    # Extract climate channels → (N, 6, 4, 4, 4) → (N*64, 6)
    clim = noise_3d[:, CLIMATE_INDICES, :, :, :]
    clim_flat = clim.reshape(n, INPUT_SIZE, -1).transpose(0, 2, 1).reshape(-1, INPUT_SIZE)

    # Biome targets → (N*64,) int64
    biome_flat = _remap_biome_ids(biome_ids.reshape(-1))

    # Filter out unknown biomes
    valid_mask = biome_flat >= 0
    clim_flat = clim_flat[valid_mask]
    biome_flat = biome_flat[valid_mask]

    n_total = n * 64
    n_valid = len(biome_flat)
    n_dropped = n_total - n_valid
    print(f"  Extracted {n_valid:,} valid samples ({n_dropped:,} dropped as unknown) "
          f"from {n:,} sections")

    return torch.from_numpy(clim_flat), torch.from_numpy(biome_flat)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 1e-3,
    out_dir: Path = Path("models/biome"),
    device: str | None = None,
) -> BiomeClassifier:
    """Train the BiomeClassifier and save checkpoint + ONNX."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"  Device: {dev}")

    # Train/val split (90/10)
    n = len(inputs)
    perm = torch.randperm(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(inputs[train_idx], targets[train_idx])
    val_ds = TensorDataset(inputs[val_idx], targets[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=(dev.type == "cuda"), num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                        pin_memory=(dev.type == "cuda"), num_workers=0)

    model = BiomeClassifier().to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_val = float("inf")
    best_acc = 0.0
    best_epoch = -1
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "biome_classifier.pt"
    onnx_path = out_dir / "biome_classifier.onnx"

    print(f"\n  Training BiomeClassifier: {INPUT_SIZE}→{HIDDEN_SIZE}→{HIDDEN_SIZE}→{NUM_BIOMES}")
    print(f"  Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")
    print(f"  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}\n")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                val_correct += (pred.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)
                val_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "val_accuracy": val_acc,
                "num_classes": NUM_BIOMES,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "climate_indices": CLIMATE_INDICES,
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train_ce={avg_train:.4f}  val_ce={avg_val:.4f}  "
                  f"val_acc={val_acc:.3f}  best={best_val:.4f}@{best_epoch}  "
                  f"lr={cur_lr:.1e}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s — "
          f"best val_ce={best_val:.4f} acc={best_acc:.3f} @ epoch {best_epoch}")

    # --- export ONNX ---
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    model.eval().cpu()
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["climate_input"],
        output_names=["biome_logits"],
        dynamic_axes={"climate_input": {0: "batch"}, "biome_logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"  ONNX export → {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train BiomeClassifier: 6 climate → 54-class MLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path,
                        default=Path("noise_training_data/sparse_octree_pairs_v7.npz"),
                        help="v7 training data NPZ file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("models/biome"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 62)
    print("  BiomeClassifier Training — 6 climate → 54 biomes")
    print("=" * 62)

    inputs, targets = load_data(args.data)
    train(inputs, targets,
          epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, out_dir=args.out_dir, device=args.device)

    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
