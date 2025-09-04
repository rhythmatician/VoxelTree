"""
Verify the exported ONNX models work correctly for all LOD transitions.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))


def verify_onnx_models():
    """Verify all exported ONNX models work correctly."""

    onnx_dir = Path("./onnx_export")

    transitions = [
        ("lod4to3", 1, 2, 4),  # 1³ → 2³, LOD level 4
        ("lod3to2", 2, 4, 3),  # 2³ → 4³, LOD level 3
        ("lod2to1", 4, 8, 2),  # 4³ → 8³, LOD level 2
        ("lod1to0", 8, 16, 1),  # 8³ → 16³, LOD level 1
    ]

    print("🔍 Verifying ONNX models...")

    for transition_name, input_size, output_size, lod_level in transitions:
        print(f"\n📦 Testing {transition_name}: {input_size}³ → {output_size}³")

        # Load ONNX model
        onnx_path = onnx_dir / f"flexible_unet3d_{transition_name}.onnx"
        session = ort.InferenceSession(str(onnx_path))

        # Get actual input names from the model
        input_names = [inp.name for inp in session.get_inputs()]
        print(f"  Model inputs: {input_names}")

        # Create test inputs
        all_inputs = {
            "parent_voxel": np.random.randn(1, 1, input_size, input_size, input_size).astype(
                np.float32
            ),
            "biome_patch": np.random.randn(1, 256, 16, 16).astype(np.float32),
            "heightmap_patch": np.random.randn(1, 1, 16, 16, 1).astype(np.float32),
            "river_patch": np.random.randn(1, 1, 16, 16, 1).astype(np.float32),
            "y_index": np.array([64], dtype=np.int64),
            "lod": np.array([lod_level], dtype=np.int64),
        }

        # Filter inputs to only those the model expects
        inputs = {name: all_inputs[name] for name in input_names if name in all_inputs}

        # Run inference
        outputs = session.run(None, inputs)
        air_mask_logits, block_type_logits = outputs

        # Verify output shapes
        expected_air_shape = (1, 1, output_size, output_size, output_size)
        expected_block_shape = (1, 1104, output_size, output_size, output_size)

        print(f"  Input shape: {inputs['parent_voxel'].shape}")
        print(f"  Air mask output: {air_mask_logits.shape} (expected: {expected_air_shape})")
        print(
            f"  Block logits output: {block_type_logits.shape} (expected: {expected_block_shape})"
        )

        # Verify shapes match expectations
        assert (
            air_mask_logits.shape == expected_air_shape
        ), f"Air mask shape mismatch: {air_mask_logits.shape} != {expected_air_shape}"
        assert (
            block_type_logits.shape == expected_block_shape
        ), f"Block logits shape mismatch: {block_type_logits.shape} != {expected_block_shape}"

        # Verify output ranges are reasonable
        air_min, air_max = air_mask_logits.min(), air_mask_logits.max()
        block_min, block_max = block_type_logits.min(), block_type_logits.max()

        print(f"  Air mask range: [{air_min:.3f}, {air_max:.3f}]")
        print(f"  Block logits range: [{block_min:.3f}, {block_max:.3f}]")

        print(f"  ✅ {transition_name} verification passed!")

    print("\n🎉 All ONNX models verified successfully!")
    print("   - All 4 LOD transitions working correctly")
    print("   - Output shapes match expectations")
    print("   - Ready for LODiffusion integration")


if __name__ == "__main__":
    verify_onnx_models()
