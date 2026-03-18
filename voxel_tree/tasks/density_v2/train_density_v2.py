#!/usr/bin/env python3
"""Density V2 MLP — predict preliminary_surface_level + final_density from 6 climate fields.

Architecture
------------
  Input  : 6 climate RouterField channels (temperature, vegetation, continents,
           erosion, depth, ridges) — one value per quart cell.
  Hidden : 128 → ReLU → 128 → ReLU
  Output : 2 (preliminary_surface_level, final_density)

The model operates **per quart cell** — each sample is one (field_0 … field_5)
→ (psl, fd) pair extracted from the v7 training data.

Training data
-------------
  Reads ``sparse_octree_pairs_v7.npz`` produced by ``build_sparse_octree_pairs.py``.
  ``noise_3d`` has shape (N, 15, 4, 4, 4).  We extract:
    - Input  channels 0–5  (climate)           → flatten to (N*64, 6)
    - Target channels 6, 7 (psl, final_density) → flatten to (N*64, 2)

Model output
------------
  density_v2.pt    — PyTorch checkpoint
  density_v2.onnx  — ONNX export (opset 18, float32)

Usage
-----
  python -m voxel_tree.tasks.density_v2.train_density_v2 \\
      --data noise_training_data/sparse_octree_pairs_v7.npz \\
      --epochs 200 --batch-size 4096 --lr 1e-3 \\
      --out-dir models/density_v2
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
    from voxel_tree.utils.router_field import CLIMATE_FIELDS, DENSITY_FIELDS
except ImportError:
    # Standalone fallback
    CLIMATE_FIELDS = frozenset(range(6))
    DENSITY_FIELDS = frozenset({6, 7})

try:
    from voxel_tree.contracts import get_contract

    CONTRACT = get_contract("density", revision=1)
except Exception:  # standalone fallback
    CONTRACT = None

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

INPUT_SIZE = 6  # 6 climate fields
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2  # preliminary_surface_level, final_density

# Channel indices in the 15-channel noise_3d tensor
CLIMATE_INDICES = sorted(CLIMATE_FIELDS)  # [0, 1, 2, 3, 4, 5]
TARGET_INDICES = sorted(DENSITY_FIELDS)  # [6, 7]


class DensityV2(nn.Module):
    """6 → 128 → 128 → 2 MLP, ReLU hidden activations, linear output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6) → (B, 2)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(npz_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load v7 training pairs and extract climate→density samples.

    Returns (inputs, targets) where:
        inputs:  (M, 6) float32  — climate fields per quart cell
        targets: (M, 2) float32  — (preliminary_surface_level, final_density)
    M = N_sections * 64 (4×4×4 quart cells per section).
    """
    print(f"  Loading {npz_path} ...")
    with np.load(npz_path) as data:
        noise_3d = data["noise_3d"]  # (N, 15, 4, 4, 4) float32

    n = noise_3d.shape[0]
    assert noise_3d.shape[1] == 15, f"Expected 15 channels, got {noise_3d.shape[1]}"

    # Extract climate (input) and density (target) channels
    clim = noise_3d[:, CLIMATE_INDICES, :, :, :]  # (N, 6, 4, 4, 4)
    dens = noise_3d[:, TARGET_INDICES, :, :, :]  # (N, 2, 4, 4, 4)

    # Flatten spatial dims: (N, C, 4, 4, 4) → (N, C, 64) → (N*64, C)
    clim_flat = clim.reshape(n, INPUT_SIZE, -1).transpose(0, 2, 1).reshape(-1, INPUT_SIZE)
    dens_flat = dens.reshape(n, OUTPUT_SIZE, -1).transpose(0, 2, 1).reshape(-1, OUTPUT_SIZE)

    print(f"  Extracted {clim_flat.shape[0]:,} per-cell samples from {n:,} sections")
    return torch.from_numpy(clim_flat), torch.from_numpy(dens_flat)


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
    out_dir: Path = Path("models/density_v2"),
    device: str | None = None,
) -> DensityV2:
    """Train the DensityV2 MLP and save checkpoint + ONNX."""

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
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(dev.type == "cuda"),
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=(dev.type == "cuda"),
        num_workers=0,
    )

    model = DensityV2().to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val = float("inf")
    best_epoch = -1
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "density_v2.pt"
    onnx_path = out_dir / "density_v2.onnx"

    print(f"\n  Training DensityV2: {INPUT_SIZE}→{HIDDEN_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}")
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
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "climate_indices": CLIMATE_INDICES,
                "target_indices": TARGET_INDICES,
            }
            if CONTRACT is not None:
                ckpt_dict["contract_meta"] = CONTRACT.to_checkpoint_meta()
            torch.save(ckpt_dict, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train_mse={avg_train:.6f}  val_mse={avg_val:.6f}  "
                f"best={best_val:.6f}@{best_epoch}  "
                f"lr={cur_lr:.1e}  [{elapsed:.0f}s]"
            )

    elapsed = time.time() - t0
    print(
        f"\n  Training complete in {elapsed:.1f}s — best val_mse={best_val:.6f} @ epoch {best_epoch}"
    )

    # --- export ONNX ---
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)["model_state_dict"])
    model.eval().cpu()
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["climate_input"],
        output_names=["density_output"],
        dynamic_axes={"climate_input": {0: "batch"}, "density_output": {0: "batch"}},
        opset_version=18,
    )
    print(f"  ONNX export → {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train DensityV2: 6 climate → 2 density MLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("noise_training_data/sparse_octree_pairs_v7.npz"),
        help="v7 training data NPZ file",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("models/density_v2"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 62)
    print("  DensityV2 Training — 6 climate → 2 density")
    print("=" * 62)

    inputs, targets = load_data(args.data)
    train(
        inputs,
        targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        device=args.device,
    )

    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
