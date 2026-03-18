#!/usr/bin/env python3
"""Export BiomeClassifier to ONNX with sidecar configuration.

Loads a trained BiomeClassifier checkpoint and exports:
  - biome_classifier.onnx  — ONNX model (opset 18, float32)
  - biome_classifier.json  — sidecar config for Java runtime

Usage
-----
  python -m voxel_tree.tasks.biome.export_biome \\
      --checkpoint models/biome/biome_classifier.pt \\
      --out-dir exports/biome
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from voxel_tree.utils.router_field import RouterField, CLIMATE_FIELDS
    from voxel_tree.utils.biome_mapping import OVERWORLD_BIOMES
except ImportError:
    CLIMATE_FIELDS = frozenset(range(6))
    OVERWORLD_BIOMES = [f"biome_{i}" for i in range(54)]

from voxel_tree.tasks.biome.train_biome_classifier import (
    BiomeClassifier,
    INPUT_SIZE,
    NUM_BIOMES,
    CLIMATE_INDICES,
)

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

MODEL_CONTRACT = "lodiffusion.v7.biome_classifier"
MODEL_VERSION = "7.0.0"


def export(checkpoint_path: Path, out_dir: Path) -> None:
    """Export BiomeClassifier checkpoint to ONNX + sidecar JSON."""

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "biome_classifier.onnx"
    sidecar_path = out_dir / "biome_classifier.json"

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BiomeClassifier(num_classes=ckpt.get("num_classes", NUM_BIOMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1.0)
    val_acc = ckpt.get("val_accuracy", -1.0)
    print(f"  Checkpoint epoch {epoch}, val_ce={val_loss:.4f}, val_acc={val_acc:.3f}")

    # ONNX export
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["climate_input"],
        output_names=["biome_logits"],
        dynamic_axes={
            "climate_input": {0: "batch"},
            "biome_logits": {0: "batch"},
        },
        opset_version=18,
    )
    onnx_kb = onnx_path.stat().st_size / 1024
    print(f"  ONNX export → {onnx_path} ({onnx_kb:.1f} KB)")

    # Resolve field names for sidecar
    try:
        input_names = [RouterField.by_index(i).lower_name for i in CLIMATE_INDICES]
    except Exception:
        input_names = ["temperature", "vegetation", "continents", "erosion", "depth", "ridges"]

    biome_names = [b.replace("minecraft:", "") for b in OVERWORLD_BIOMES]

    # Sidecar JSON
    sidecar = {
        "contract": MODEL_CONTRACT,
        "version": MODEL_VERSION,
        "description": "BiomeClassifier: 6 climate RouterFields → 54-class biome logits",
        "onnx_file": onnx_path.name,
        "opset": 18,
        "input": {
            "name": "climate_input",
            "shape": ["batch", INPUT_SIZE],
            "channels": input_names,
            "channel_indices": CLIMATE_INDICES,
            "note": "6 climate fields per quart cell, sampled at 4×4×4 resolution",
        },
        "output": {
            "name": "biome_logits",
            "shape": ["batch", NUM_BIOMES],
            "num_classes": NUM_BIOMES,
            "class_names": biome_names,
            "note": "Raw logits — apply argmax or softmax at inference time",
        },
        "training": {
            "epoch": epoch,
            "val_cross_entropy": float(val_loss),
            "val_accuracy": float(val_acc),
        },
    }

    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"  Sidecar → {sidecar_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export BiomeClassifier to ONNX + sidecar",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/biome/biome_classifier.pt"),
        help="Trained BiomeClassifier checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "model",
        help="Output directory for ONNX + sidecar",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 62)
    print("  BiomeClassifier ONNX Export")
    print("=" * 62)

    export(args.checkpoint, args.out_dir)

    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
