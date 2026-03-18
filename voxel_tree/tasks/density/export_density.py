#!/usr/bin/env python3
"""Export Density model to ONNX with sidecar configuration.

Loads a trained Density checkpoint and exports:
  - density.onnx      — ONNX model (opset 18, float32)
  - density.json      — sidecar config for Java runtime

The Java runtime (LODiffusion) loads the sidecar to discover input/output
shapes, channel mappings, and model version.

Usage
-----
  python -m voxel_tree.tasks.density.export_density \\
      --checkpoint models/density/density_best.pt \\
      --out-dir exports/density
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from voxel_tree.utils.router_field import RouterField, CLIMATE_FIELDS, DENSITY_FIELDS
except ImportError:
    CLIMATE_FIELDS = frozenset(range(6))
    DENSITY_FIELDS = frozenset({6, 7})

from voxel_tree.tasks.density.train_density import (
    DensityMLP,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CLIMATE_INDICES,
    TARGET_INDICES,
)

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

try:
    from voxel_tree.contracts import get_contract, validate_checkpoint_contract

    CONTRACT = get_contract("density", revision=1)
except Exception:
    CONTRACT = None
    validate_checkpoint_contract = None  # type: ignore[assignment]

# Legacy constants kept for backward compat; new code should use CONTRACT.
MODEL_CONTRACT = CONTRACT.contract_id if CONTRACT else "lodiffusion.v7.density"
MODEL_VERSION = f"r{CONTRACT.revision}" if CONTRACT else "7.0.0"


def export(checkpoint_path: Path, out_dir: Path) -> None:
    """Export Density checkpoint to ONNX + sidecar JSON."""

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "density.onnx"
    sidecar_path = out_dir / "density.json"

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Validate checkpoint contract if available
    if CONTRACT is not None and validate_checkpoint_contract is not None:
        validate_checkpoint_contract(ckpt, CONTRACT)

    model = DensityMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", -1)
    val_loss = ckpt.get("val_loss", -1.0)
    print(f"  Checkpoint epoch {epoch}, val_loss={val_loss:.6f}")

    # ONNX export
    dummy = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["climate_input"],
        output_names=["density_output"],
        dynamic_axes={
            "climate_input": {0: "batch"},
            "density_output": {0: "batch"},
        },
        opset_version=18,
    )
    onnx_kb = onnx_path.stat().st_size / 1024
    print(f"  ONNX export → {onnx_path} ({onnx_kb:.1f} KB)")

    # Build sidecar JSON from contract (or fall back to manual construction)
    if CONTRACT is not None:
        sidecar = CONTRACT.to_sidecar(
            epoch=epoch,
            val_mse=float(val_loss),
        )
        sidecar["onnx_file"] = onnx_path.name
    else:
        # Legacy fallback — identical to the original hand-rolled dict
        try:
            input_names = [RouterField.by_index(i).lower_name for i in CLIMATE_INDICES]
            output_names = [RouterField.by_index(i).lower_name for i in TARGET_INDICES]
        except Exception:
            input_names = ["temperature", "vegetation", "continents", "erosion", "depth", "ridges"]
            output_names = ["preliminary_surface_level", "final_density"]
        sidecar = {
            "contract": MODEL_CONTRACT,
            "version": MODEL_VERSION,
            "description": "Density MLP: 6 climate RouterFields → 2 density outputs",
            "onnx_file": onnx_path.name,
            "opset": 18,
            "input": {
                "name": "climate_input",
                "shape": ["batch", INPUT_SIZE],
                "channels": input_names,
                "channel_indices": CLIMATE_INDICES,
            },
            "output": {
                "name": "density_output",
                "shape": ["batch", OUTPUT_SIZE],
                "channels": output_names,
                "channel_indices": TARGET_INDICES,
            },
            "training": {"epoch": epoch, "val_mse": float(val_loss)},
        }

    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"  Sidecar → {sidecar_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export Density MLP to ONNX + sidecar",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/density/density_best.pt"),
        help="Trained Density checkpoint",
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
    print("  Density ONNX Export")
    print("=" * 62)

    export(args.checkpoint, args.out_dir)

    print("=" * 62)
    print("  DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
