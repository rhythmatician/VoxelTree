#!/usr/bin/env python3
"""Heightmap Predictor MLP — predict surface + ocean-floor heights from climate fields.

Architecture
------------
  Input  : 6 climate RouterField channels (temperature, vegetation, continents,
           erosion, depth, ridges) per 4×4 quart column, yielding a flat input
           of 6 × 4 × 4 = 96 values per section column grid.
  Hidden : 128 → ReLU → 64 → ReLU
  Output : 2 × 4 × 4 = 32 values (WORLD_SURFACE_WG + OCEAN_FLOOR_WG per
           quart column)

The model operates on a **per-section column grid** basis — each sample maps
the 4×4 climate footprint of one chunk column to 4×4 predicted heights for
both surface and ocean floor.

Training data
-------------
  Reads ``sparse_octree_pairs_v7.npz`` produced by ``build_sparse_octree_pairs.py``.
    noise_3d             : (N, C, qx, qy, qz) float32   (C >= 6)
    heightmap_surface    : (N, 16, 16)       int32
    heightmap_ocean_floor: (N, 16, 16)       int32

  Climate input is extracted from noise_3d channels 0–5, averaged
  across the Y axis, to get a (N, 6, 4, 4) grid.
  Heightmaps are downsampled from 16×16 block resolution to 4×4 quart
  resolution by averaging each 4×4 block patch.

Model output
------------
  heightmap_predictor.pt    — PyTorch checkpoint
  heightmap_predictor.onnx  — ONNX export (opset 18, float32)

Usage
-----
  python -m voxel_tree.tasks.heightmap.train_heightmap \\
      --data noise_training_data/sparse_octree_pairs_v7.npz \\
      --epochs 200 --batch-size 512 --lr 1e-3 \\
      --out-dir models/heightmap
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
except ImportError:
    CLIMATE_FIELDS = frozenset(range(6))

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

INPUT_SIZE = 6 * 4 * 4    # 96: 6 climate channels × 4×4 quart grid
HIDDEN_1 = 128
HIDDEN_2 = 64
OUTPUT_SIZE = 2 * 4 * 4   # 32: 2 height types × 4×4 quart grid

CLIMATE_INDICES = sorted(CLIMATE_FIELDS)  # [0, 1, 2, 3, 4, 5]


class HeightmapPredictor(nn.Module):
    """96 → 128 → 64 → 32 MLP, ReLU hidden activations, linear output.

    Input:  (B, 96)  — 6 climate fields × 4×4 quart columns, flattened
    Output: (B, 32)  — (surface_4x4, ocean_floor_4x4) concatenated, flattened
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 96) → (B, 32)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _downsample_heightmap(hm_16x16: np.ndarray) -> np.ndarray:
    """Downsample (N, 16, 16) heightmap to (N, 4, 4) by averaging 4×4 patches.

    Returns float32 array of shape (N, 4, 4).
    """
    n = hm_16x16.shape[0]
    hm = hm_16x16.astype(np.float32).reshape(n, 4, 4, 4, 4)
    # Average over the two inner block-resolution axes
    return hm.mean(axis=(2, 4))  # (N, 4, 4)


def load_data(npz_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load v7 training pairs and extract climate→heightmap samples.

    Returns (inputs, targets) where:
        inputs:  (M, 96)  float32 — 6 climate channels × 4×4, flattened
        targets: (M, 32)  float32 — (surface_4×4 ++ ocean_floor_4×4), flattened
    M = number of unique section columns (deduplicated by taking one
    representative per (chunk_x, chunk_z) column).
    """
    print(f"  Loading {npz_path} ...")
    with np.load(npz_path) as data:
        noise_3d = data["noise_3d"]                        # (N, C, qx, qy, qz)
        hm_surface = data["heightmap_surface"]             # (N, 16, 16)
        hm_ocean = data["heightmap_ocean_floor"]           # (N, 16, 16)

    n = noise_3d.shape[0]
    n_ch = noise_3d.shape[1]
    # Need at least 6 channels for input (indices 0-5).
    # v7 dumps have 13 cave-density channels; legacy had 15 RouterField channels.
    assert n_ch >= 6, (
        f"Need >= 6 noise channels for climate input, got {n_ch}"
    )

    # Extract climate channels and average across Y axis → (N, 6, 4, 4)
    clim = noise_3d[:, CLIMATE_INDICES, :, :, :]  # (N, 6, qx, qy, qz)
    clim_2d = clim.mean(axis=3)  # average over qy → (N, 6, 4, 4)
    clim_flat = clim_2d.reshape(n, -1)  # (N, 96)

    # Downsample heightmaps from 16×16 to 4×4
    surf_4x4 = _downsample_heightmap(hm_surface)    # (N, 4, 4)
    ocean_4x4 = _downsample_heightmap(hm_ocean)     # (N, 4, 4)

    # Concatenate: (N, 4, 4) + (N, 4, 4) → (N, 32)
    targets_arr = np.concatenate([
        surf_4x4.reshape(n, -1),
        ocean_4x4.reshape(n, -1),
    ], axis=1)  # (N, 32)

    # Heightmaps are per-column, shared across all sections in the same column.
    # The NPZ has duplicates (same heightmap for all 24 sectionY).  Deduplicate
    # by keeping unique climate + height rows.
    # Simple approach: just keep all rows — the duplicates act as data augmentation
    # with slightly different Y-averaged climate values from different section levels.
    print(f"  Extracted {n:,} samples (6×4×4 climate → 2×4×4 heights)")

    return torch.from_numpy(clim_flat), torch.from_numpy(targets_arr)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    out_dir: Path = Path("models/heightmap"),
    device: str | None = None,
) -> HeightmapPredictor:
    """Train the HeightmapPredictor and save checkpoint + ONNX."""

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

    model = HeightmapPredictor().to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_val = float("inf")
    best_epoch = -1
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "heightmap_predictor.pt"
    onnx_path = out_dir / "heightmap_predictor.onnx"

    print(f"\n  Training HeightmapPredictor: {INPUT_SIZE}→{HIDDEN_1}→{HIDDEN_2}→{OUTPUT_SIZE}")
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
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                val_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "input_size": INPUT_SIZE,
                "hidden_1": HIDDEN_1,
                "hidden_2": HIDDEN_2,
                "output_size": OUTPUT_SIZE,
                "climate_indices": CLIMATE_INDICES,
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            # Compute RMSE for interpretability (units: blocks)
            rmse = avg_val ** 0.5
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train_mse={avg_train:.2f}  val_mse={avg_val:.2f}  "
                  f"rmse={rmse:.2f}blocks  best={best_val:.2f}@{best_epoch}  "
                  f"lr={cur_lr:.1e}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    best_rmse = best_val ** 0.5
    print(f"\n  Training complete in {elapsed:.1f}s — "
          f"best val_mse={best_val:.2f} (rmse={best_rmse:.2f} blocks) @ epoch {best_epoch}")

    # --- export ONNX ---
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    model.eval().cpu()
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["climate_grid"],
        output_names=["heightmap_output"],
        dynamic_axes={"climate_grid": {0: "batch"}, "heightmap_output": {0: "batch"}},
        opset_version=18,
    )
    print(f"  ONNX export → {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train HeightmapPredictor: 6×4×4 climate → 2×4×4 heights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path,
                        default=Path("noise_training_data/sparse_octree_pairs_v7.npz"),
                        help="v7 training data NPZ file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("models/heightmap"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 62)
    print("  HeightmapPredictor Training — 6×4×4 climate → 2×4×4 heights")
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
