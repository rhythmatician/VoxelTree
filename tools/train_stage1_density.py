#!/usr/bin/env python3
"""
Stage 1 Density MLP Training  (WS-4.2)

Trains a small 12 → 64 → 64 → 1 MLP to predict Minecraft's finalDensity field
from the 12 hand-crafted noise features dumped by /dumpnoise stage1.

Input features (12 columns, in order):
  0  offset             – TerrainShaper output, horizontal shape
  1  factor             – TerrainShaper output, vertical scale
  2  jaggedness         – TerrainShaper output, surface roughness
  3  depth              – router.depth() (YClampedGradient + offset)
  4  sloped_cheese      – overworld/sloped_cheese NormalNoise
  5  y                  – cell-centre block Y (−64 … 316)
  6  entrances          – overworld/caves/entrances
  7  cheese_caves       – overworld/caves/pillars
  8  spaghetti_2d       – overworld/caves/spaghetti_2d
  9  roughness          – overworld/caves/spaghetti_roughness_function
 10  noodle             – overworld/caves/noodle
 11  base_3d_noise      – overworld/base_3d_noise

Output:
  final_density         – the raw GPU / CPU density value (clamped −64 … +64 in GPU,
                          but the Java dumper gives unclamped values)

Training data: run/stage1_dumps/chunk_*.json  (produced by /dumpnoise stage1)
Model output : stage1_mlp.pt   (PyTorch checkpoint)
               stage1_mlp.onnx (ONNX export for inference)
               stage1_mlp_weights.bin (flat float32 weights for SSBO upload)

Usage:
  python train_stage1_density.py [--data-dir PATH] [--epochs N] [--batch-size N]
                                  [--lr LR] [--out-dir PATH]
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

INPUT_FEATURES = 12
HIDDEN_SIZE    = 64
OUTPUT_SIZE    = 1

FEATURE_NAMES = [
    "offset", "factor", "jaggedness", "depth", "sloped_cheese",
    "y", "entrances", "cheese_caves", "spaghetti_2d",
    "roughness", "noodle", "base_3d_noise",
]


class Stage1DensityMLP(nn.Module):
    """
    12 → 64 → 64 → 1, ReLU hidden activations, linear output.

    Flat SSBO weight layout (used by extract_stage1_weights.py):
      W1[64,12] row-major  (768 floats)
      b1[64]               (64  floats)
      W2[64,64] row-major  (4096 floats)
      b2[64]               (64  floats)
      W3[1,64]  row-major  (64  floats)
      b3[1]                (1   float)
      total: 5057 floats
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chunk_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse one chunk_<cx>_<cz>.json and return
    (inputs[768, 12], labels[768]).
    """
    with path.open() as f:
        data = json.load(f)

    # Each entry in `inputs` is a 12-element list.
    inputs_raw = data["inputs"]          # list[list[float]], len=768
    label_raw  = data["final_density"]   # list[float], len=768

    inputs = np.array(inputs_raw, dtype=np.float32)   # (768, 12)
    labels = np.array(label_raw,  dtype=np.float32)   # (768,)

    if inputs.shape != (768, 12):
        raise ValueError(f"{path}: expected inputs shape (768,12), got {inputs.shape}")
    if labels.shape != (768,):
        raise ValueError(f"{path}: expected labels shape (768,), got {labels.shape}")

    return inputs, labels


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all chunk_*.json files from `data_dir`.
    Returns (X[N,12], y[N]) combined across all chunks.
    """
    files = sorted(data_dir.glob("chunk_*.json"))
    if not files:
        print(f"[ERROR] No chunk_*.json files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Data] Loading {len(files)} chunk files from {data_dir} …")
    all_inputs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for i, fp in enumerate(files):
        try:
            inp, lbl = load_chunk_file(fp)
            all_inputs.append(inp)
            all_labels.append(lbl)
        except Exception as e:
            print(f"  [WARN] Skipping {fp.name}: {e}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            print(f"  … {i + 1}/{len(files)} loaded")

    if not all_inputs:
        print("[ERROR] No valid data loaded — aborting.", file=sys.stderr)
        sys.exit(1)

    X = np.concatenate(all_inputs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"[Data] Total samples: {X.shape[0]:,}  (features: {X.shape[1]})")
    return X, y


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_normstats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean[12], std[12]) computed over the training data."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    # Avoid divide-by-zero for constant features (e.g. y on flat test data)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # --- Load data -----------------------------------------------------------
    X_np, y_np = load_dataset(data_dir)

    # Normalise inputs (z-score)
    mean_np, std_np = compute_normstats(X_np)
    X_norm = (X_np - mean_np) / std_np

    # Save normalisation stats (needed at inference time)
    np.save(out_dir / "stage1_norm_mean.npy", mean_np)
    np.save(out_dir / "stage1_norm_std.npy",  std_np)
    print(f"[Train] Normalisation stats saved to {out_dir}")

    # Train / validation split (90 % / 10 %)
    n_total = X_norm.shape[0]
    n_val   = max(1, int(n_total * 0.1))
    perm    = np.random.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train = torch.from_numpy(X_norm[train_idx]).to(device)
    y_train = torch.from_numpy(y_np[train_idx]).to(device)
    X_val   = torch.from_numpy(X_norm[val_idx]).to(device)
    y_val   = torch.from_numpy(y_np[val_idx]).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)

    print(f"[Train] Train samples: {len(train_idx):,}  Val samples: {n_val:,}")

    # --- Model ---------------------------------------------------------------
    model = Stage1DensityMLP().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {INPUT_FEATURES}→{HIDDEN_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}  "
          f"params: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch   = 0
    best_ckpt    = out_dir / "stage1_mlp_best.pt"

    print(f"\n[Train] Training for {args.epochs} epochs …\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train step ----
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation step ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse  = loss_fn(val_pred, y_val).item()

        # Save best
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch   = epoch
            torch.save({
                "epoch":     epoch,
                "state":     model.state_dict(),
                "val_mse":   val_mse,
                "norm_mean": mean_np,
                "norm_std":  std_np,
            }, best_ckpt)

        # Logging
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train_mse={avg_train_loss:.6f}  "
                  f"val_mse={val_mse:.6f}  "
                  f"lr={lr_now:.2e}  "
                  f"[{elapsed:.1f}s]")

        # Early stop if target met
        if val_mse < args.target_mse:
            print(f"\n[Train] Target MSE {args.target_mse} reached at epoch {epoch} "
                  f"(val_mse={val_mse:.6f}) — stopping early.")
            break

    # ---- Final summary ----
    total_time = time.time() - t0
    print(f"\n[Train] Done. Best val_mse={best_val_mse:.6f} at epoch {best_epoch}  "
          f"({total_time:.1f}s total)")
    target_met = "✓ TARGET MET" if best_val_mse < args.target_mse else "✗ target not met"
    print(f"[Train] {target_met}  (target={args.target_mse})")

    # ---- Load best weights for export ----
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state"])
    model.eval()
    model.cpu()

    # -- Save final PyTorch checkpoint --
    final_pt = out_dir / "stage1_mlp.pt"
    torch.save(ckpt, final_pt)
    print(f"[Export] PyTorch checkpoint → {final_pt}")

    # -- ONNX export --
    _export_onnx(model, mean_np, std_np, out_dir)

    # -- Flat binary weights --
    _export_flat_weights(model, out_dir)

    return best_val_mse


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(model: Stage1DensityMLP, mean_np: np.ndarray, std_np: np.ndarray,
                 out_dir: Path) -> None:
    """Export with normalisation baked in as a preprocessing Div/Sub."""
    import torch.onnx

    # Wrap model to include normalisation (subtract + divide in-graph)
    class NormalisedModel(nn.Module):
        def __init__(self, base: nn.Module, mean, std):
            super().__init__()
            self.base = base
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))

        def forward(self, x):
            return self.base((x - self.mean) / self.std)

    wrapped = NormalisedModel(model, mean_np, std_np)
    wrapped.eval()

    dummy = torch.zeros(1, INPUT_FEATURES, dtype=torch.float32)
    onnx_path = out_dir / "stage1_mlp.onnx"
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["features"],
        output_names=["final_density"],
        dynamic_axes={"features": {0: "batch"}, "final_density": {0: "batch"}},
        opset_version=17,
    )
    print(f"[Export] ONNX model → {onnx_path}")


# ---------------------------------------------------------------------------
# Flat binary weight export (for SSBO upload in GLSL)
# ---------------------------------------------------------------------------

def _export_flat_weights(model: Stage1DensityMLP, out_dir: Path) -> None:
    """
    Write all weights as a flat float32 binary in SSBO-compatible order:
      W1[64,12], b1[64], W2[64,64], b2[64], W3[1,64], b3[1]
    Total: 768 + 64 + 4096 + 64 + 64 + 1 = 5057 floats = 20228 bytes
    """
    layers = list(model.net.children())
    linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
    assert len(linear_layers) == 3, f"Expected 3 Linear layers, got {len(linear_layers)}"

    flat_weights: list[np.ndarray] = []
    for lin in linear_layers:
        W = lin.weight.detach().numpy().astype(np.float32)  # [out, in]
        b = lin.bias.detach().numpy().astype(np.float32)    # [out]
        flat_weights.append(W.flatten())
        flat_weights.append(b)

    blob = np.concatenate(flat_weights).astype(np.float32)
    expected = (64 * 12 + 64) + (64 * 64 + 64) + (1 * 64 + 1)  # 5057
    assert len(blob) == expected, f"Weight count mismatch: {len(blob)} vs {expected}"

    bin_path = out_dir / "stage1_mlp_weights.bin"
    blob.tofile(str(bin_path))
    print(f"[Export] Flat weights ({len(blob)} floats, {len(blob)*4} bytes) → {bin_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage 1 density MLP from /dumpnoise stage1 data."
    )
    p.add_argument("--data-dir",   default="stage1_dumps",
                   help="Directory containing chunk_*.json files  (default: stage1_dumps)")
    p.add_argument("--out-dir",    default="stage1_model",
                   help="Where to write trained weights/checkpoints  (default: stage1_model)")
    p.add_argument("--epochs",     type=int,   default=200,
                   help="Maximum training epochs  (default: 200)")
    p.add_argument("--batch-size", type=int,   default=4096,
                   help="SGD mini-batch size  (default: 4096)")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Initial AdamW learning rate  (default: 1e-3)")
    p.add_argument("--target-mse", type=float, default=0.001,
                   help="Early-stop when val MSE < this threshold  (default: 0.001)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed  (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("  Stage 1 Density MLP  —  WS-4.2")
    print("=" * 64)
    print(f"  data_dir   = {args.data_dir}")
    print(f"  out_dir    = {args.out_dir}")
    print(f"  epochs     = {args.epochs}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  lr         = {args.lr}")
    print(f"  target_mse = {args.target_mse}")
    print()

    val_mse = train(args)
    sys.exit(0 if val_mse < args.target_mse else 1)


if __name__ == "__main__":
    main()
