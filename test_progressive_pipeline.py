"""
Test script for all progressive LOD models including the initial model.
"""

import torch

from train.progressive_lod_models import (
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
)
from train.unet3d import SimpleFlexibleConfig


def test_all_progressive_lod_models():
    """Test the complete progressive LOD pipeline."""
    config = SimpleFlexibleConfig()

    # Create shared inputs (all models use the same conditioning)
    x_height_planes = torch.randn(1, 5, 1, 16, 16)  # [1,5,1,16,16]
    x_biome_quart = torch.randn(1, 6, 4, 4, 4)  # [1,6,4,4,4]
    x_router6 = torch.randn(1, 6, 1, 16, 16)  # [1,6,1,16,16]
    x_chunk_pos = torch.randn(1, 2)  # [1,2]
    x_lod = torch.tensor([[0]], dtype=torch.long)  # [1,1]

    # Optional inputs (can be None)
    x_barrier = torch.randn(1, 1, 1, 16, 16)  # [1,1,1,16,16]
    x_aquifer3 = torch.randn(1, 3, 1, 16, 16)  # [1,3,1,16,16]
    x_cave_prior4 = torch.randn(1, 1, 4, 4, 4)  # [1,1,4,4,4]

    print("=== Testing Progressive LOD Models ===\n")

    # Test Model 0: Initial (no parent) → 2×2×2
    print("Testing Model 0: Initial → 2×2×2")
    model0 = ProgressiveLODModel0_Initial(config, output_size=2)

    with torch.no_grad():
        outputs0 = model0(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )

    print("✅ Model 0 successful!")
    print(f"   Air mask: {outputs0['air_mask_logits'].shape}")
    print(f"   Block logits: {outputs0['block_type_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model0.parameters()):,}")
    print()

    # Test Model 1: LOD4→LOD3 (2×2×2 → 4×4×4)
    print("Testing Model 1: LOD4→LOD3 (2×2×2 → 4×4×4)")
    model1 = ProgressiveLODModel(config, output_size=4)

    x_parent_prev = outputs0["air_mask_logits"]  # Use output from model 0

    with torch.no_grad():
        outputs1 = model1(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            x_parent_prev,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )

    print("✅ Model 1 successful!")
    print(f"   Air mask: {outputs1['air_mask'].shape}")
    print(f"   Block logits: {outputs1['block_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    print()

    # Test Model 2: LOD3→LOD2 (4×4×4 → 8×8×8)
    print("Testing Model 2: LOD3→LOD2 (4×4×4 → 8×8×8)")
    model2 = ProgressiveLODModel(config, output_size=8)

    x_parent_prev = outputs1["air_mask"]  # Use output from model 1

    with torch.no_grad():
        outputs2 = model2(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            x_parent_prev,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )

    print("✅ Model 2 successful!")
    print(f"   Air mask: {outputs2['air_mask'].shape}")
    print(f"   Block logits: {outputs2['block_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print()

    # Test Model 3: LOD2→LOD1 (8×8×8 → 16×16×16)
    print("Testing Model 3: LOD2→LOD1 (8×8×8 → 16×16×16)")
    model3 = ProgressiveLODModel(config, output_size=16)

    x_parent_prev = outputs2["air_mask"]  # Use output from model 2

    with torch.no_grad():
        outputs3 = model3(
            x_height_planes,
            x_biome_quart,
            x_router6,
            x_chunk_pos,
            x_lod,
            x_parent_prev,
            x_barrier,
            x_aquifer3,
            x_cave_prior4,
        )

    print("✅ Model 3 successful!")
    print(f"   Air mask: {outputs3['air_mask'].shape}")
    print(f"   Block logits: {outputs3['block_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    print()

    print("=== Full Progressive Pipeline Test ===")
    print("Initial → 2×2×2 → 4×4×4 → 8×8×8 → 16×16×16")
    print("✅ All models work correctly!")

    # Total parameter count
    total_params = (
        sum(p.numel() for p in model0.parameters())
        + sum(p.numel() for p in model1.parameters())
        + sum(p.numel() for p in model2.parameters())
        + sum(p.numel() for p in model3.parameters())
    )
    print(f"Total parameters across all 4 models: {total_params:,}")


if __name__ == "__main__":
    test_all_progressive_lod_models()
