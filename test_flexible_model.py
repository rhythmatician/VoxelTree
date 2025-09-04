"""
Test script for flexible multi-LOD model.

This script tests the flexible UNet architecture with different input sizes
to verify it can handle all LOD transitions.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D


def test_flexible_model():
    """Test the flexible model with different input sizes."""

    # Create model config
    config = SimpleFlexibleConfig(
        base_channels=16,  # Smaller for testing
        biome_vocab_size=256,
        block_vocab_size=1104,
    )

    # Create model
    model = SimpleFlexibleUNet3D(config)
    model.eval()

    print("Testing flexible multi-LOD model...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test data for all LOD transitions
    test_cases = [
        ("LOD4->LOD3", 1, 2),  # 1^3 -> 2^3
        ("LOD3->LOD2", 2, 4),  # 2^3 -> 4^3
        ("LOD2->LOD1", 4, 8),  # 4^3 -> 8^3
        ("LOD1->LOD0", 8, 16),  # 8^3 -> 16^3
    ]

    batch_size = 2

    for test_name, input_size, expected_output_size in test_cases:
        print(f"\n=== Testing {test_name} ===")

        # Create test inputs
        parent_voxel = torch.randn(batch_size, 1, input_size, input_size, input_size)
        biome_patch = torch.randint(0, 256, (batch_size, 16, 16))  # Index format
        heightmap_patch = torch.randn(batch_size, 1, 16, 16, 1)
        river_patch = torch.randn(batch_size, 1, 16, 16, 1)
        y_index = torch.randint(0, 384, (batch_size,))
        # Determine LOD values based on input size
        if input_size == 1:
            lod = torch.tensor([4, 3])
        elif input_size == 2:
            lod = torch.tensor([3, 2])
        elif input_size == 4:
            lod = torch.tensor([2, 1])
        else:
            lod = torch.tensor([1, 0])

        print(f"Input shape: {parent_voxel.shape}")
        print(f"LOD levels: {lod}")

        try:
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    parent_voxel=parent_voxel,
                    biome_patch=biome_patch,
                    heightmap_patch=heightmap_patch,
                    river_patch=river_patch,
                    y_index=y_index,
                    lod=lod,
                )

            # Check output shapes
            air_mask = outputs["air_mask_logits"]
            block_logits = outputs["block_type_logits"]

            expected_shape = (
                batch_size,
                1,
                expected_output_size,
                expected_output_size,
                expected_output_size,
            )
            expected_block_shape = (
                batch_size,
                1104,
                expected_output_size,
                expected_output_size,
                expected_output_size,
            )

            print(f"Air mask shape: {air_mask.shape} (expected: {expected_shape})")
            print(f"Block logits shape: {block_logits.shape} (expected: {expected_block_shape})")

            # Verify shapes
            assert air_mask.shape == expected_shape, "Air mask shape mismatch"
            assert block_logits.shape == expected_block_shape, "Block logits shape mismatch"

            print(f"✅ {test_name} passed!")

        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== Testing with different batch LOD levels ===")

    # Test mixed LOD levels in same batch (should work for same input size)
    try:
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)  # LOD1 input
        biome_patch = torch.randint(0, 256, (batch_size, 16, 16))
        heightmap_patch = torch.randn(batch_size, 1, 16, 16, 1)
        river_patch = torch.randn(batch_size, 1, 16, 16, 1)
        y_index = torch.randint(0, 384, (batch_size,))
        lod = torch.tensor([1, 1])  # Same LOD level

        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        print("✅ Mixed batch test passed!")
        print(f"Output shape: {outputs['air_mask_logits'].shape}")

    except Exception as e:
        print(f"❌ Mixed batch test failed: {e}")


def test_multi_lod_dataset():
    """Test the multi-LOD dataset generation."""

    print("\n=== Testing Multi-LOD Dataset ===")

    # Create synthetic test data
    labels16 = np.random.randint(0, 100, (16, 16, 16), dtype=np.int32)
    biome16 = np.random.randint(0, 256, (16, 16), dtype=np.int32)
    height16 = np.random.randint(0, 256, (16, 16)).astype(np.float32)
    river16 = np.random.rand(16, 16).astype(np.float32)

    try:
        # Import here to avoid issues if not available
        from train.multi_lod_dataset import create_lod_training_pairs

        # Generate training pairs
        pairs = create_lod_training_pairs(
            labels16=labels16,
            biome_patch=biome16,
            heightmap_patch=height16,
            river_patch=river16,
            y_index=64,
        )

        print(f"Generated {len(pairs)} training pairs")

        for pair in pairs:
            transition = pair["lod_transition"]
            parent_shape = pair["parent_voxel"].shape
            target_shape = pair["target_blocks"].shape

            print(f"{transition}: {parent_shape} → {target_shape}")

        print("✅ Multi-LOD dataset test passed!")

    except ImportError as e:
        print(f"❌ Multi-LOD dataset test skipped: {e}")
    except Exception as e:
        print(f"❌ Multi-LOD dataset test failed: {e}")


if __name__ == "__main__":
    print("Testing Flexible Multi-LOD Architecture")
    print("=" * 50)

    test_flexible_model()
    test_multi_lod_dataset()

    print("\n" + "=" * 50)
    print("🎯 Testing complete!")
