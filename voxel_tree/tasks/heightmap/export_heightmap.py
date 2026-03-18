#!/usr/bin/env python3
"""Export HeightmapPredictor to ONNX with sidecar configuration.

Usage
-----
  python -m voxel_tree.tasks.heightmap.export_heightmap \\
      --checkpoint models/heightmap/heightmap_predictor.pt \\
      --out-dir exports/heightmap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from voxel_tree.utils.router_field import RouterField, CLIMATE_FIELDS
except ImportError:
    CLIMATE_FIELDS = frozenset(range(6))

from voxel_tree.tasks.heightmap.train_heightmap import (
    HeightmapPredictor,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CLIMATE_INDICES,
)

MODEL_CONTRACT = "lodiffusion.v7.heightmap_predictor"
MODEL_VERSION = "7.0.0"


def export(checkpoint_path: Path, out_dir: Path) -> None:
    """Export HeightmapPredictor checkpoint to ONNX + sidecar JSON."""

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "heightmap_predictor.onnx"
    sidecar_path = out_dir / "heightmap_predictor.json"

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HeightmapPredictor()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1.0)
    print(f"  Checkpoint epoch {epoch}, val_mse={val_loss:.2f}")

    # ONNX export
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["climate_grid"],
        output_names=["heightmap_output"],
        dynamic_axes={
            "climate_grid": {0: "batch"},
            "heightmap_output": {0: "batch"},
        },
        opset_version=18,
    )
    onnx_kb = onnx_path.stat().st_size / 1024
    print(f"  ONNX export → {onnx_path} ({onnx_kb:.1f} KB)")

    # Resolve field names
    try:
        input_names = [RouterField.by_index(i).lower_name for i in CLIMATE_INDICES]
    except Exception:
        input_names = ["temperature", "vegetation", "continents", "erosion", "depth", "ridges"]

    sidecar = {
        "contract": MODEL_CONTRACT,
        "version": MODEL_VERSION,
        "description": "HeightmapPredictor: 6×4×4 climate grid → 2×4×4 heights (surface + ocean floor)",
        "onnx_file": onnx_path.name,
        "opset": 18,
        "input": {
            "name": "climate_grid",
            "shape": ["batch", INPUT_SIZE],
            "channels": input_names,
            "channel_indices": CLIMATE_INDICES,
            "spatial": "4x4 quart grid, Y-averaged climate fields",
            "note": "Flat 96 = 6 channels × 4 qx × 4 qz, channel-outermost",
        },
        "output": {
            "name": "heightmap_output",
            "shape": ["batch", OUTPUT_SIZE],
            "layout": "first 16 = surface_4x4, next 16 = ocean_floor_4x4",
            "spatial": "4x4 quart grid, qx-outer qz-inner",
            "note": "Heights in blocks (Y coordinate), can be negative",
        },
        "training": {
            "epoch": epoch,
            "val_mse": float(val_loss),
        },
    }

    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"  Sidecar → {sidecar_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export HeightmapPredictor to ONNX + sidecar",
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/heightmap/heightmap_predictor.pt"),
                        help="Trained HeightmapPredictor checkpoint")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("exports/heightmap"),
                        help="Output directory for ONNX + sidecar")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 62)
    print("  HeightmapPredictor ONNX Export")
    print("=" * 62)

    export(args.checkpoint, args.out_dir)

    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
