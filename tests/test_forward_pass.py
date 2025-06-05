# tests\test_forward_pass.py
import pytest
import torch

from train.unet3d import UNet3DConfig, VoxelUNet3D


class TestForwardPass:
    """Phase 4.2 - Extended forward pass test coverage (8³ → 16³).

    NOTE: TDD Violation Acknowledged - forward() method was already implemented
    in Phase 4.1. This phase expands test coverage with scenarios not covered
    in test_unet3d.py to provide genuine value beyond basic functionality.
    """

    @pytest.fixture
    def model(self):
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            base_channels=32,
            depth=3,
            biome_vocab_size=50,
            biome_embed_dim=16,
            heightmap_channels=1,
            river_channels=1,
            y_embed_dim=8,
            lod_embed_dim=8,
            block_type_channels=10,  # Correct: 10 channels for block types
        )
        return VoxelUNet3D(config)

    def test_forward_pass_shape(self, model):
        """Should return correct output shape for 8³ → 16³ upscaling."""
        batch_size = 1
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)
        out = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            river_patch=river_patch,
            y_index=y_index,
            lod=lod,
        )
        assert "air_mask_logits" in out
        assert "block_type_logits" in out
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_variable_batch_sizes(self, model, batch_size):
        """Should handle different batch sizes correctly."""
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_edge_case_all_zeros(self, model):
        """Should handle all-zero input gracefully."""
        batch_size = 2
        parent_voxel = torch.zeros(batch_size, 1, 8, 8, 8)
        biome_patch = torch.zeros((batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.zeros(batch_size, 1, 16, 16)
        river_patch = torch.zeros(batch_size, 1, 16, 16)
        y_index = torch.zeros((batch_size,), dtype=torch.long)
        lod = torch.ones((batch_size,), dtype=torch.long)  # LOD=1

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Outputs should be finite (not NaN/Inf)
        assert torch.isfinite(out["air_mask_logits"]).all()
        assert torch.isfinite(out["block_type_logits"]).all()

        # Outputs should have correct shapes
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_gradient_flow(self, model):
        """Should allow gradients to flow through the model."""
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=True)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        out = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            river_patch=river_patch,
            y_index=y_index,
            lod=lod,
        )

        # Compute loss and backpropagate
        loss = out["air_mask_logits"].sum() + out["block_type_logits"].sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert parent_voxel.grad is not None
        assert parent_voxel.grad.abs().sum() > 0

        # Check model parameters have gradients
        param_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert param_with_grad > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self, model):
        """Should work correctly on GPU when available."""
        batch_size = 2
        device = torch.device("cuda")

        # Move model to GPU
        model = model.to(device)

        # Create inputs on GPU
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, device=device)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long, device=device)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16, device=device)
        river_patch = torch.randn(batch_size, 1, 16, 16, device=device)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long, device=device)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Outputs should be on GPU
        assert out["air_mask_logits"].device.type == "cuda"
        assert out["block_type_logits"].device.type == "cuda"  # Check correct shapes
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    @pytest.mark.parametrize(
        "scenario_name,biome_id,y_val,lod_val",
        [
            ("plains_surface", 1, 10, 1),
            ("desert_underground", 15, 5, 3),
            ("ocean_deep", 0, 2, 2),
            ("mountain_high", 3, 18, 4),
        ],
    )
    def test_individual_conditioning_scenarios(
        self, model, scenario_name, biome_id, y_val, lod_val
    ):
        """Should handle individual conditioning scenarios without errors."""
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.full((batch_size, 16, 16), biome_id, dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.full((batch_size,), y_val, dtype=torch.long)
        lod = torch.full((batch_size,), lod_val, dtype=torch.long)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                river_patch=river_patch,
                y_index=y_index,
                lod=lod,
            )

        # Should produce finite outputs with correct shapes
        assert torch.isfinite(out["air_mask_logits"]).all()
        assert torch.isfinite(out["block_type_logits"]).all()
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_conditioning_sensitivity(self, model):
        """Should produce different outputs for different conditioning."""
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)

        # Two contrasting scenarios for comparison
        scenarios = [
            # Plains biome, surface level, LOD 1
            {
                "biome_patch": torch.full((batch_size, 16, 16), 1, dtype=torch.long),
                "y_index": torch.full((batch_size,), 10, dtype=torch.long),
                "lod": torch.full((batch_size,), 1, dtype=torch.long),
            },
            # Desert biome, underground, LOD 3
            {
                "biome_patch": torch.full((batch_size, 16, 16), 15, dtype=torch.long),
                "y_index": torch.full((batch_size,), 5, dtype=torch.long),
                "lod": torch.full((batch_size,), 3, dtype=torch.long),
            },
        ]

        outputs = []
        for scenario in scenarios:
            with torch.no_grad():
                out = model(
                    parent_voxel=parent_voxel,
                    biome_patch=scenario["biome_patch"],
                    heightmap_patch=heightmap_patch,
                    river_patch=river_patch,
                    y_index=scenario["y_index"],
                    lod=scenario["lod"],
                )
            outputs.append(out)

        # Outputs should be different for different conditioning
        air_diff = torch.abs(outputs[0]["air_mask_logits"] - outputs[1]["air_mask_logits"]).mean()
        block_diff = torch.abs(
            outputs[0]["block_type_logits"] - outputs[1]["block_type_logits"]
        ).mean()

        assert air_diff > 0.01, f"Air mask should vary with conditioning, got diff={air_diff}"
        assert (
            block_diff > 0.01
        ), f"Block types should vary with conditioning, got diff={block_diff}"
