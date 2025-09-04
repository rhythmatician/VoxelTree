"""
Export Multi-LOD Flexi    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) Model to ONNX

This script exports the trained flexible multi-LOD model to ONNX format
for integration with the LODiffusion mod.
"""

import argparse
import sys
from pathlib import Path

import onnx
import torch

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from train.unet3d import SimpleFlexibleUNet3D


def export_multi_lod_model(checkpoint_path: str, output_dir: str):
    """Export the multi-LOD model to ONNX format."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    model = SimpleFlexibleUNet3D(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Export separate ONNX models for each LOD transition
    lod_transitions = [
        ("lod4to3", 1, 2, 4),  # 1³ → 2³, LOD level 4
        ("lod3to2", 2, 4, 3),  # 2³ → 4³, LOD level 3
        ("lod2to1", 4, 8, 2),  # 4³ → 8³, LOD level 2
        ("lod1to0", 8, 16, 1),  # 8³ → 16³, LOD level 1
    ]

    for transition_name, input_size, output_size, lod_level in lod_transitions:
        print(f"\nExporting {transition_name}: {input_size}³ → {output_size}³")

        # Create dummy inputs for this transition
        dummy_parent_voxel = torch.randn(1, 1, input_size, input_size, input_size)
        dummy_biome_patch = torch.randn(1, 256, 16, 16)  # One-hot biome
        dummy_heightmap_patch = torch.randn(1, 1, 16, 16, 1)
        dummy_river_patch = torch.randn(1, 1, 16, 16, 1)
        dummy_y_index = torch.tensor([64], dtype=torch.int64)  # Middle Y level
        dummy_lod = torch.tensor([lod_level], dtype=torch.int64)

        # Test forward pass
        with torch.no_grad():
            outputs = model(
                dummy_parent_voxel,
                dummy_biome_patch,
                dummy_heightmap_patch,
                dummy_river_patch,
                dummy_y_index,
                dummy_lod,
            )

        air_mask_logits = outputs["air_mask_logits"]
        block_type_logits = outputs["block_type_logits"]

        print(f"  Input shape: {dummy_parent_voxel.shape}")
        print(f"  Air mask output: {air_mask_logits.shape}")
        print(f"  Block logits output: {block_type_logits.shape}")

        # Define input/output names for ONNX
        input_names = [
            "parent_voxel",
            "biome_patch",
            "heightmap_patch",
            "river_patch",
            "y_index",
            "lod",
        ]

        output_names = ["air_mask_logits", "block_type_logits"]

        # Export to ONNX
        onnx_path = output_dir / f"flexible_unet3d_{transition_name}.onnx"

        torch.onnx.export(
            model,
            (
                dummy_parent_voxel,
                dummy_biome_patch,
                dummy_heightmap_patch,
                dummy_river_patch,
                dummy_y_index,
                dummy_lod,
            ),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        print(f"  ✅ ONNX model saved: {onnx_path}")
        print("  ✅ ONNX model verified")

    # Create model configuration file
    model_config = {
        "model_type": "flexible_multi_lod",
        "architecture": "SimpleFlexibleUNet3D",
        "parameters": sum(p.numel() for p in model.parameters()),
        "base_channels": config.base_channels,
        "block_vocab_size": config.block_vocab_size,
        "biome_vocab_size": config.biome_vocab_size,
        "transitions": {
            "lod4to3": {"input_size": 1, "output_size": 2, "lod_level": 4},
            "lod3to2": {"input_size": 2, "output_size": 4, "lod_level": 3},
            "lod2to1": {"input_size": 4, "output_size": 8, "lod_level": 2},
            "lod1to0": {"input_size": 8, "output_size": 16, "lod_level": 1},
        },
        "inputs": {
            "parent_voxel": {"shape": "[1, 1, S, S, S]", "dtype": "float32"},
            "biome_patch": {"shape": "[1, 256, 16, 16]", "dtype": "float32"},
            "heightmap_patch": {"shape": "[1, 1, 16, 16, 1]", "dtype": "float32"},
            "river_patch": {"shape": "[1, 1, 16, 16, 1]", "dtype": "float32"},
            "y_index": {"shape": "[1]", "dtype": "int64"},
            "lod": {"shape": "[1]", "dtype": "int64"},
        },
        "outputs": {
            "air_mask_logits": {"shape": "[1, 1, T, T, T]", "dtype": "float32"},
            "block_type_logits": {"shape": "[1, 1104, T, T, T]", "dtype": "float32"},
        },
    }

    import json

    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✅ Model configuration saved: {config_path}")
    print("\n🎉 Multi-LOD model export complete!")
    print("   - 4 ONNX models exported for all LOD transitions")
    print("   - Model configuration saved")
    print("   - Ready for LODiffusion integration")


def main():
    parser = argparse.ArgumentParser(description="Export Multi-LOD Model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./onnx_export", help="Output directory")

    args = parser.parse_args()

    export_multi_lod_model(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
