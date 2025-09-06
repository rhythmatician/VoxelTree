#!/usr/bin/env python3
"""
ONNX Export Test for Progressive LOD Models

Tests ONNX export functionality and validates the exported models.
"""

import gc
import sys
import time
from pathlib import Path

import torch

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from train.progressive_lod_models import (  # noqa: E402
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
)
from train.unet3d import SimpleFlexibleConfig  # noqa: E402


def create_test_inputs_onnx(output_size=16):
    """Create test inputs for ONNX export (batch_size=1 only)."""
    return {
        "x_height_planes": torch.randn(1, 5, 1, 16, 16),
        "x_biome_quart": torch.randn(1, 6, 4, 4, 4),
        "x_router6": torch.randn(1, 6, 1, 16, 16),
        "x_chunk_pos": torch.randn(1, 2),
        "x_lod": torch.tensor([[0]], dtype=torch.long),
        "x_parent_prev": torch.randn(1, 1, output_size // 2, output_size // 2, output_size // 2),
    }


def export_model_to_onnx(model, model_name, output_dir):
    """Export a PyTorch model to ONNX format."""
    print(f"\nExporting {model_name} to ONNX...")

    model.eval()

    # Determine output size and inputs from model name
    if "0_Initial" in model_name:
        output_size = 1
        has_parent = False
    elif "1_LOD4to3" in model_name:
        output_size = 2
        has_parent = True
    elif "2_LOD3to2" in model_name:
        output_size = 4
        has_parent = True
    elif "3_LOD2to1" in model_name:
        output_size = 8
        has_parent = True
    elif "4_LOD1to0" in model_name:
        output_size = 16
        has_parent = True

    # Create test inputs
    inputs = create_test_inputs_onnx(output_size)

    # Export path
    onnx_path = output_dir / f"{model_name.lower().replace('_', '')}.onnx"

    try:
        # Define input/output names based on model type
        if has_parent:
            input_names = [
                "x_height_planes",
                "x_biome_quart",
                "x_router6",
                "x_chunk_pos",
                "x_lod",
                "x_parent_prev",
            ]
            example_inputs = (
                inputs["x_height_planes"],
                inputs["x_biome_quart"],
                inputs["x_router6"],
                inputs["x_chunk_pos"],
                inputs["x_lod"],
                inputs["x_parent_prev"],
            )
        else:
            input_names = ["x_height_planes", "x_biome_quart", "x_router6", "x_chunk_pos", "x_lod"]
            example_inputs = (
                inputs["x_height_planes"],
                inputs["x_biome_quart"],
                inputs["x_router6"],
                inputs["x_chunk_pos"],
                inputs["x_lod"],
            )

        output_names = ["air_mask", "block_logits"]

        # Export to ONNX
        start_time = time.perf_counter()
        torch.onnx.export(
            model,
            example_inputs,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # Static shapes only
        )
        export_time = time.perf_counter() - start_time

        # Get file size
        file_size_mb = onnx_path.stat().st_size / 1024 / 1024

        print(f"  Export time: {export_time:.2f} seconds")
        print(f"  ONNX file size: {file_size_mb:.2f} MB")
        print(f"  Saved to: {onnx_path}")

        return {
            "model_name": model_name,
            "onnx_path": str(onnx_path),
            "export_time_s": export_time,
            "file_size_mb": file_size_mb,
            "success": True,
        }

    except Exception as e:
        print(f"  Export failed: {e}")
        return {
            "model_name": model_name,
            "onnx_path": str(onnx_path),
            "export_time_s": 0,
            "file_size_mb": 0,
            "success": False,
            "error": str(e),
        }


def load_model(model_path, device="cpu"):
    """Load a trained model from disk."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint.get("model_name", "Unknown")

        # Create appropriate model based on name
        config = SimpleFlexibleConfig()

        if "0_Initial" in model_name:
            model = ProgressiveLODModel0_Initial(config, output_size=1)
        elif "1_LOD4to3" in model_name:
            model = ProgressiveLODModel(config, output_size=2)
        elif "2_LOD3to2" in model_name:
            model = ProgressiveLODModel(config, output_size=4)
        elif "3_LOD2to1" in model_name:
            model = ProgressiveLODModel(config, output_size=8)
        elif "4_LOD1to0" in model_name:
            model = ProgressiveLODModel(config, output_size=16)
        else:
            raise Exception(f"Unknown model type: {model_name}")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, model_name

    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None, "Unknown"


def main():
    """Main ONNX export test script."""
    print("Progressive LOD Models - ONNX Export Test")
    print("=" * 50)

    # Configuration
    device = "cpu"  # Use CPU for ONNX export
    print(f"Using device: {device}")

    models_dir = Path("models")
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    # Create ONNX output directory
    onnx_dir = Path("onnx_export_test")
    onnx_dir.mkdir(exist_ok=True)
    print(f"ONNX export directory: {onnx_dir}")

    # Find all model files
    model_files = list(models_dir.glob("quick_*.pt"))
    if not model_files:
        print(f"No model files found in {models_dir}")
        return

    print(f"Found {len(model_files)} models to export")

    export_results = []

    # Export each model
    for model_file in sorted(model_files):
        print(f"\nLoading {model_file.name}...")

        # Clear memory
        gc.collect()

        model, model_name = load_model(model_file, device)
        if model is not None:
            result = export_model_to_onnx(model, model_name, onnx_dir)
            export_results.append(result)

        # Clean up
        del model
        gc.collect()

    # Summary report
    print(f"\n{'='*50}")
    print("ONNX EXPORT SUMMARY")
    print(f"{'='*50}")

    if export_results:
        print(f"{'Model':<20} {'Status':<10} {'Export(s)':<12} {'Size(MB)':<10}")
        print("-" * 50)

        total_export_time = 0
        total_size = 0
        successful_exports = 0

        for result in export_results:
            name = result["model_name"].replace("Model_", "").replace("_", "->")
            status = "SUCCESS" if result["success"] else "FAILED"
            export_time = f"{result['export_time_s']:.2f}" if result["success"] else "N/A"
            size = f"{result['file_size_mb']:.1f}" if result["success"] else "N/A"

            print(f"{name:<20} {status:<10} {export_time:<12} {size:<10}")

            if result["success"]:
                total_export_time += result["export_time_s"]
                total_size += result["file_size_mb"]
                successful_exports += 1
            else:
                print(f"  Error: {result.get('error', 'Unknown')}")

        print("-" * 50)
        print(
            f"{'TOTAL':<20} {successful_exports}/{len(export_results)}{'':<5} {total_export_time:.2f}{'':<7} {total_size:.1f}{'':<6}"  # noqa: E501
        )

        if successful_exports > 0:
            print("\nExport Performance:")
            print(f"  Average export time: {total_export_time/successful_exports:.2f} seconds")
            print(f"  Total ONNX size: {total_size:.1f} MB")
            print(
                f"  Success rate: {successful_exports}/{len(export_results)} ({100*successful_exports/len(export_results):.0f}%)"  # noqa: E501
            )

        # List exported files
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if onnx_files:
            print("\nExported ONNX files:")
            for onnx_file in sorted(onnx_files):
                size_mb = onnx_file.stat().st_size / 1024 / 1024
                print(f"  {onnx_file.name} ({size_mb:.1f} MB)")

    else:
        print("No export attempts completed")


if __name__ == "__main__":
    main()
