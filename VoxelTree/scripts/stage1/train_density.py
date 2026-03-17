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

# ---------------------------------------------------------------------------
# Feature layout
#
# Stage 1 format (12 direct density inputs from /dumpnoise stage1):
#   All 12 features are sampled at 4×48×4 cell resolution (768 per chunk).
#   This matches vanilla Minecraft's finalDensity input signals exactly.
# ---------------------------------------------------------------------------
INPUT_FEATURES = 12
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

FEATURE_NAMES = [
    "offset",  # 0: TerrainShaper output
    "factor",  # 1: TerrainShaper output
    "jaggedness",  # 2: TerrainShaper output
    "depth",  # 3: router.depth() (YClampedGradient + offset)
    "sloped_cheese",  # 4: overworld/sloped_cheese NormalNoise
    "y",  # 5: cell-centre block Y (−64 … 316)
    "entrances",  # 6: overworld/caves/entrances
    "cheese_caves",  # 7: overworld/caves/pillars
    "spaghetti_2d",  # 8: overworld/caves/spaghetti_2d
    "roughness",  # 9: overworld/caves/spaghetti_roughness_function
    "noodle",  # 10: overworld/caves/noodle
    "base_3d_noise",  # 11: overworld/base_3d_noise
]

# Cell grid dimensions
_CX = 4  # noise cells per chunk in X
_CY = 48  # noise cells per chunk in Y  (world height 384 / cell height 8)
_CZ = 4  # noise cells per chunk in Z


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


def _load_profile(path: str) -> dict:
    """Load a YAML profile and return its contents as a dict."""
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "[WARN] PyYAML not installed — ignoring --profile. " "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_profile(args: argparse.Namespace, profile: dict) -> None:
    """
    Override argparse defaults with values from the profile's `train` and
    `data` sections.  Explicit CLI flags always win over profile values.
    """
    mapping = {
        # profile key path             argparse attr    cast
        ("data", "stage1_dump_dir"): ("data_dir", str),
        ("data", "val_split"): ("val_split", float),
        ("train", "output_dir"): ("out_dir", str),
        ("train", "epochs"): ("epochs", int),
        ("train", "batch_size"): ("batch_size", int),
        ("train", "lr"): ("lr", float),
        ("train", "target_mse"): ("target_mse", float),
    }
    for (section, key), (attr, cast) in mapping.items():
        value = profile.get(section, {}).get(key)
        if value is not None and getattr(args, attr) is None:
            setattr(args, attr, cast(value))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_chunk_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse one chunk_<cx>_<cz>.json and return (inputs[768, 12], labels[768]).

    Expects the stage1 JSON format (produced by /dumpnoise stage1):
      All 12 input features (offset, factor, jaggedness, depth, sloped_cheese, y,
      entrances, cheese_caves, spaghetti_2d, roughness, noodle, base_3d_noise)
      are present as flat arrays of 768 floats (4×48×4 cell grid).
      final_density — 768 floats (training label)

    All inputs and labels are 4×48×4 cell resolution (768 samples per chunk).

    Raises KeyError / ValueError if required fields are missing or malformed.
    """
    with path.open() as f:
        data = json.load(f)

    # ── Load final_density label ────────────────────────────────────────────────
    fd = data.get("final_density")
    if fd is None:
        raise KeyError("final_density")

    n_cells = _CX * _CY * _CZ  # 768
    if len(fd) != n_cells:
        raise ValueError(f"final_density length {len(fd)} != {n_cells}")
    labels = np.array(fd, dtype=np.float32)  # (768,)

    # ── Load all 12 input features from stage1 dump ─────────────────────────────
    # Each feature is a flat array of 768 floats (4×48×4 cell grid)
    input_channels = []
    for feat_name in FEATURE_NAMES:
        vals = data.get(feat_name)
        if vals is None:
            raise KeyError(f"Missing feature: {feat_name}")
        if len(vals) != n_cells:
            raise ValueError(f"{feat_name} length {len(vals)} != {n_cells}")
        input_channels.append(np.array(vals, dtype=np.float32))  # (768,)

    # Stack all 12 channels: (768, 12)
    inputs = np.stack(input_channels, axis=1)  # (768, 12)
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

    print(f"[Data] Loading {len(files)} chunk files from {data_dir} ...")
    all_inputs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n_skipped = 0
    skip_reason: dict[str, int] = {}

    for i, fp in enumerate(files):
        try:
            inp, lbl = load_chunk_file(fp)
            all_inputs.append(inp)
            all_labels.append(lbl)
        except Exception as e:
            n_skipped += 1
            key = type(e).__name__ + ":" + str(e)
            skip_reason[key] = skip_reason.get(key, 0) + 1
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(files)} scanned  ({len(all_inputs)} loaded)")

    if n_skipped:
        print(f"  [WARN] Skipped {n_skipped:,} files:", file=sys.stderr)
        for reason, count in sorted(skip_reason.items(), key=lambda x: -x[1])[:5]:
            print(f"    {count:>6,}× {reason}", file=sys.stderr)

    if not all_inputs:
        print("[ERROR] No valid data loaded — aborting.", file=sys.stderr)
        sys.exit(1)

    X = np.concatenate(all_inputs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(
        f"[Data] Loaded {len(all_inputs):,} chunks -> {X.shape[0]:,} samples  (features: {X.shape[1]})"
    )
    return X, y


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def compute_normstats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean[12], std[12]) computed over the training data."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid divide-by-zero for constant features (e.g. y on flat test data)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
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
    np.save(out_dir / "stage1_norm_std.npy", std_np)
    print(f"[Train] Normalisation stats saved to {out_dir}")

    # Train / validation split (90 % / 10 %)
    n_total = X_norm.shape[0]
    n_val = max(1, int(n_total * 0.1))
    perm = np.random.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train = torch.from_numpy(X_norm[train_idx]).to(device)
    y_train = torch.from_numpy(y_np[train_idx]).to(device)
    X_val = torch.from_numpy(X_norm[val_idx]).to(device)
    y_val = torch.from_numpy(y_np[val_idx]).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    print(f"[Train] Train samples: {len(train_idx):,}  Val samples: {n_val:,}")

    # --- Model ---------------------------------------------------------------
    model = Stage1DensityMLP().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"[Train] Model: {INPUT_FEATURES}->{HIDDEN_SIZE}->{HIDDEN_SIZE}->{OUTPUT_SIZE}  "
        f"params: {param_count:,}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_ckpt = out_dir / "stage1_mlp_best.pt"

    print(f"\n[Train] Training for {args.epochs} epochs...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train step ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation step ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = loss_fn(val_pred, y_val).item()

        # Save best
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state": model.state_dict(),
                    "val_mse": val_mse,
                    "norm_mean": mean_np,
                    "norm_std": std_np,
                },
                best_ckpt,
            )

        # Logging
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            pct = epoch / args.epochs * 100.0
            # Emit a percentage so GUI progress bars can track training.
            print(
                f"  Epoch {epoch:4d}/{args.epochs}  "
                f"train_mse={avg_train_loss:.6f}  "
                f"val_mse={val_mse:.6f}  "
                f"lr={lr_now:.2e}  "
                f"[{elapsed:.1f}s]  "
                f"{pct:.1f}%"
            )

        # Early stop if target met
        if val_mse < args.target_mse:
            print(
                f"\n[Train] Target MSE {args.target_mse} reached at epoch {epoch} "
                f"(val_mse={val_mse:.6f}) — stopping early."
            )
            break

    # ---- Final summary ----
    total_time = time.time() - t0
    print(
        f"\n[Train] Done. Best val_mse={best_val_mse:.6f} at epoch {best_epoch}  "
        f"({total_time:.1f}s total)"
    )
    target_met = "[OK] TARGET MET" if best_val_mse < args.target_mse else "[FAIL] target not met"
    print(f"[Train] {target_met}  (target={args.target_mse})")

    # ---- Load best weights for export ----
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    model.cpu()

    # -- Save final PyTorch checkpoint --
    final_pt = out_dir / "stage1_mlp.pt"
    torch.save(ckpt, final_pt)
    print(f"[Export] PyTorch checkpoint -> {final_pt}")

    # -- ONNX export --
    _export_onnx(model, mean_np, std_np, out_dir)

    # -- Flat binary weights --
    _export_flat_weights(model, out_dir)

    return best_val_mse


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _export_onnx(
    model: Stage1DensityMLP, mean_np: np.ndarray, std_np: np.ndarray, out_dir: Path
) -> None:
    """Export with normalisation baked in as a preprocessing Div/Sub."""
    import torch.onnx

    # Wrap model to include normalisation (subtract + divide in-graph)
    class NormalisedModel(nn.Module):
        def __init__(self, base: nn.Module, mean, std):
            super().__init__()
            self.base = base
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        def forward(self, x):
            return self.base((x - self.mean) / self.std)

    wrapped = NormalisedModel(model, mean_np, std_np)
    wrapped.eval()

    dummy = torch.zeros(1, INPUT_FEATURES, dtype=torch.float32)
    onnx_path = out_dir / "stage1_mlp.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["features"],
        output_names=["final_density"],
        dynamic_axes={"features": {0: "batch"}, "final_density": {0: "batch"}},
        opset_version=17,
    )
    print(f"[Export] ONNX model -> {onnx_path}")


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
    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert len(linear_layers) == 3, f"Expected 3 Linear layers, got {len(linear_layers)}"

    flat_weights: list[np.ndarray] = []
    for lin in linear_layers:
        W = lin.weight.detach().numpy().astype(np.float32)  # [out, in]
        b = lin.bias.detach().numpy().astype(np.float32)  # [out]
        flat_weights.append(W.flatten())
        flat_weights.append(b)

    blob = np.concatenate(flat_weights).astype(np.float32)
    expected = (64 * 12 + 64) + (64 * 64 + 64) + (1 * 64 + 1)  # 5057
    assert len(blob) == expected, f"Weight count mismatch: {len(blob)} vs {expected}"

    bin_path = out_dir / "stage1_mlp_weights.bin"
    blob.tofile(str(bin_path))
    print(f"[Export] Flat weights ({len(blob)} floats, {len(blob)*4} bytes) -> {bin_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage 1 density MLP from /dumpnoise stage1 data."
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Path to a YAML profile (e.g. profiles/stage1_density.yaml). "
        "Profile values are used as defaults; explicit flags override them.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing chunk_*.json files  (default: stage1_dumps)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write trained weights/checkpoints  (default: stage1_model)",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="Maximum training epochs  (default: 200)"
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="SGD mini-batch size  (default: 4096)"
    )
    p.add_argument(
        "--lr", type=float, default=None, help="Initial AdamW learning rate  (default: 1e-3)"
    )
    p.add_argument(
        "--target-mse",
        type=float,
        default=None,
        help="Early-stop when val MSE < this threshold  (default: 0.001)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed  (default: 42)")
    args = p.parse_args(argv)

    # Apply profile defaults (before hard-coded fallbacks below)
    if args.profile:
        profile = _load_profile(args.profile)
        _apply_profile(args, profile)
        print(f"[Profile] Loaded: {args.profile}")
        if "name" in profile:
            print(f"[Profile] Name: {profile['name']}")
        if "description" in profile:
            print(f"[Profile] {profile['description']}")

    # Hard-coded fallbacks (after profile, before returning)
    if args.data_dir is None:
        args.data_dir = "stage1_dumps"
    if args.out_dir is None:
        args.out_dir = "stage1_model"
    if args.epochs is None:
        args.epochs = 200
    if args.batch_size is None:
        args.batch_size = 4096
    if args.lr is None:
        args.lr = 1e-3
    if args.target_mse is None:
        args.target_mse = 0.001

    return args


def main(
    argv: list[str] | None = None,
):  # Fix Windows terminal encoding for UTF-8 characters (e.g., emoji from torch.onnx)
    import io

    args = parse_args(argv)
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  Stage 1 Density MLP: 12 Direct Minecraft Density Inputs  ---  WS-4.2")
    print("=" * 70)
    print(f"  data_dir   = {args.data_dir}  (from /dumpnoise stage1)")
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
