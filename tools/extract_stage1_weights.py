#!/usr/bin/env python3
"""
Extract and deploy Stage 1 Density MLP weights  (WS-4.2a)

Loads the trained Stage 1 model (`stage1_model/stage1_mlp.pt`), re-exports:
  - stage1_mlp_weights.bin  — flat float32 SSBO blob
  - stage1_norm_mean.bin    — float32[12] input normalisation mean
  - stage1_norm_std.bin     — float32[12] input normalisation std

and copies them to:
  LODiffusion/src/main/resources/assets/lodiffusion/models/

so they are bundled into the mod JAR and available at runtime via
`Stage1DensityMlpSsbo.java`.

Usage:
  python extract_stage1_weights.py [--model-dir PATH] [--out-dir PATH]

Layout of stage1_mlp_weights.bin:
  W1[64,12] row-major   768 floats
  b1[64]                 64 floats
  W2[64,64] row-major  4096 floats
  b2[64]                 64 floats
  W3[1,64]  row-major    64 floats
  b3[1]                   1 float
  --------------------------------
  total                5057 floats  (20228 bytes)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mirror the model architecture from train_stage1_density.py
# ---------------------------------------------------------------------------

INPUT_FEATURES = 12
HIDDEN_SIZE    = 64
OUTPUT_SIZE    = 1

EXPECTED_WEIGHT_COUNT = (
    HIDDEN_SIZE * INPUT_FEATURES + HIDDEN_SIZE +   # W1 + b1
    HIDDEN_SIZE * HIDDEN_SIZE    + HIDDEN_SIZE +   # W2 + b2
    OUTPUT_SIZE * HIDDEN_SIZE    + OUTPUT_SIZE      # W3 + b3
)  # = 5057


class Stage1DensityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model_dir: Path) -> dict:
    ckpt_path = model_dir / "stage1_mlp.pt"
    if not ckpt_path.exists():
        # Fall back to best checkpoint
        ckpt_path = model_dir / "stage1_mlp_best.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint found in {model_dir}  —  train first with "
              f"train_stage1_density.py", file=sys.stderr)
        sys.exit(1)

    print(f"[Load] Checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def export_flat_weights(model: Stage1DensityMLP, out_path: Path) -> None:
    """Write weights as a flat float32 binary (SSBO-compatible)."""
    layers = [l for l in model.net.children() if isinstance(l, nn.Linear)]
    assert len(layers) == 3, f"Expected 3 Linear layers, got {len(layers)}"

    blobs: list[np.ndarray] = []
    for lin in layers:
        blobs.append(lin.weight.detach().numpy().astype(np.float32).flatten())
        blobs.append(lin.bias.detach().numpy().astype(np.float32))

    flat = np.concatenate(blobs)
    assert len(flat) == EXPECTED_WEIGHT_COUNT, \
        f"Weight count: {len(flat)} vs expected {EXPECTED_WEIGHT_COUNT}"

    flat.tofile(str(out_path))
    print(f"[Export] Weights ({len(flat)} floats, {len(flat)*4} bytes) → {out_path}")


def export_norm_stats(ckpt: dict, out_dir: Path) -> None:
    """Write input normalisation mean and std as separate float32 binary files."""
    mean_np = ckpt.get("norm_mean")
    std_np  = ckpt.get("norm_std")

    if mean_np is None or std_np is None:
        print("[WARN] Checkpoint does not contain norm_mean/norm_std — "
              "writing identity normalisation (all zeros / ones)")
        mean_np = np.zeros(INPUT_FEATURES, dtype=np.float32)
        std_np  = np.ones(INPUT_FEATURES,  dtype=np.float32)

    mean_np = np.asarray(mean_np, dtype=np.float32)
    std_np  = np.asarray(std_np,  dtype=np.float32)

    mean_path = out_dir / "stage1_norm_mean.bin"
    std_path  = out_dir / "stage1_norm_std.bin"
    mean_np.tofile(str(mean_path))
    std_np.tofile(str(std_path))
    print(f"[Export] Norm mean ({len(mean_np)*4} bytes) → {mean_path}")
    print(f"[Export] Norm std  ({len(std_np)*4}  bytes) → {std_path}")

    # Human-readable summary
    feature_names = [
        "offset", "factor", "jaggedness", "depth", "sloped_cheese",
        "y", "entrances", "cheese_caves", "spaghetti_2d",
        "roughness", "noodle", "base_3d_noise",
    ]
    print("\n  Feature normalisation:")
    print(f"  {'Feature':22s}  {'Mean':>10s}  {'Std':>10s}")
    print("  " + "-" * 46)
    for name, m, s in zip(feature_names, mean_np, std_np):
        print(f"  {name:22s}  {m:10.4f}  {s:10.4f}")


def verify_model(model: Stage1DensityMLP, ckpt: dict) -> None:
    """Report model quality metrics from the checkpoint."""
    val_mse = ckpt.get("val_mse")
    epoch   = ckpt.get("epoch", "?")

    print(f"\n[Model] Trained for {epoch} epochs")
    if val_mse is not None:
        target = 0.001
        status = "✓ TARGET MET" if val_mse < target else "✗ target not met"
        print(f"[Model] val_mse = {val_mse:.6f}  (target < {target})  {status}")

    # Spot-check: all-zero input
    with torch.no_grad():
        zero_in  = torch.zeros(1, INPUT_FEATURES)
        zero_out = model(zero_in).item()
        print(f"[Model] Forward check — zeros → {zero_out:.6f}")


def copy_to_lodiffusion(files: list[tuple[Path, str]], lodiffusion_models: Path) -> None:
    """Copy exported files to LODiffusion mod resources."""
    lodiffusion_models.mkdir(parents=True, exist_ok=True)
    print(f"\n[Deploy] Copying to LODiffusion resources: {lodiffusion_models}")
    for src_path, filename in files:
        if not src_path.exists():
            print(f"  [SKIP] {filename} — not found at {src_path}")
            continue
        dest = lodiffusion_models / filename
        shutil.copy2(src_path, dest)
        print(f"  Copied {src_path.name} → {dest}")
    print("[Deploy] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Stage 1 MLP weights and deploy to LODiffusion."
    )
    p.add_argument("--model-dir", default="stage1_model",
                   help="Directory containing stage1_mlp.pt  (default: stage1_model)")
    p.add_argument("--out-dir",   default=None,
                   help="Where to write .bin files  (default: same as --model-dir)")
    p.add_argument("--no-deploy", action="store_true",
                   help="Skip copying files to LODiffusion resources")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else model_dir

    print("=" * 64)
    print("  Stage 1 Density MLP — weight extraction  (WS-4.2a)")
    print("=" * 64)

    # Load checkpoint
    ckpt = load_checkpoint(model_dir)

    # Reconstruct model
    model = Stage1DensityMLP()
    model.load_state_dict(ckpt["state"])
    model.eval()
    model.cpu()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify quality
    verify_model(model, ckpt)

    # Export weights + norm stats
    weights_bin = out_dir / "stage1_mlp_weights.bin"
    export_flat_weights(model, weights_bin)
    export_norm_stats(ckpt, out_dir)

    # Deploy to LODiffusion resources
    if not args.no_deploy:
        lodiffusion_models = (
            Path(__file__).resolve().parent.parent.parent
            / "LODiffusion"
            / "src" / "main" / "resources"
            / "assets" / "lodiffusion" / "models"
        )
        to_copy = [
            (out_dir / "stage1_mlp_weights.bin",  "stage1_mlp_weights.bin"),
            (out_dir / "stage1_norm_mean.bin",     "stage1_norm_mean.bin"),
            (out_dir / "stage1_norm_std.bin",      "stage1_norm_std.bin"),
        ]
        # Also copy ONNX if present
        onnx_src = model_dir / "stage1_mlp.onnx"
        if onnx_src.exists():
            to_copy.append((onnx_src, "stage1_mlp.onnx"))

        copy_to_lodiffusion(to_copy, lodiffusion_models)
    else:
        print("\n[Deploy] Skipped (--no-deploy).")

    print("\n[Done] Stage 1 weight extraction complete.")


if __name__ == "__main__":
    main()
