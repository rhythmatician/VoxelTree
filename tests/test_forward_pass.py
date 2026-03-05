import pytest
import torch

from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D


class TestForwardPass:
    """Extended forward pass test coverage (8³ → 16³)."""

    @pytest.fixture
    def model(self):
        config = SimpleFlexibleConfig(
            base_channels=32,
            max_channels=128,
            biome_vocab_size=50,
            biome_embed_dim=16,
            lod_embed_dim=8,
            block_vocab_size=10,
        )
        return SimpleFlexibleUNet3D(config)

    def test_forward_pass_shape(self, model):
        batch_size = 1
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)
        out = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            y_index=y_index,
            lod=lod,
        )
        assert "air_mask_logits" in out
        assert "block_type_logits" in out
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_variable_batch_sizes(self, model, batch_size):
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                y_index=y_index,
                lod=lod,
            )

        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_edge_case_all_zeros(self, model):
        batch_size = 2
        parent_voxel = torch.zeros(batch_size, 1, 8, 8, 8)
        biome_patch = torch.zeros((batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.zeros(batch_size, 1, 16, 16)
        y_index = torch.zeros((batch_size,), dtype=torch.long)
        lod = torch.ones((batch_size,), dtype=torch.long)  # LOD=1

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                y_index=y_index,
                lod=lod,
            )

        assert torch.isfinite(out["air_mask_logits"]).all()
        assert torch.isfinite(out["block_type_logits"]).all()
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_gradient_flow(self, model):
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=True)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        out = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            y_index=y_index,
            lod=lod,
        )

        loss = out["air_mask_logits"].sum() + out["block_type_logits"].sum()
        loss.backward()

        assert parent_voxel.grad is not None
        assert parent_voxel.grad.abs().sum() > 0
        param_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert param_with_grad > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self, model):
        batch_size = 2
        device = torch.device("cuda")
        model = model.to(device)

        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, device=device)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long, device=device)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16, device=device)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long, device=device)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                y_index=y_index,
                lod=lod,
            )

        assert out["air_mask_logits"].device.type == "cuda"
        assert out["block_type_logits"].device.type == "cuda"
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
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.full((batch_size, 16, 16), biome_id, dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.full((batch_size,), y_val, dtype=torch.long)
        lod = torch.full((batch_size,), lod_val, dtype=torch.long)

        with torch.no_grad():
            out = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                y_index=y_index,
                lod=lod,
            )

        assert torch.isfinite(out["air_mask_logits"]).all()
        assert torch.isfinite(out["block_type_logits"]).all()
        assert out["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        assert out["block_type_logits"].shape == (batch_size, 10, 16, 16, 16)

    def test_conditioning_sensitivity(self, model):
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        scenarios = [
            {
                "biome_patch": torch.full((batch_size, 16, 16), 1, dtype=torch.long),
                "y_index": torch.full((batch_size,), 10, dtype=torch.long),
                "lod": torch.full((batch_size,), 1, dtype=torch.long),
            },
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
                    y_index=scenario["y_index"],
                    lod=scenario["lod"],
                )
            outputs.append(out)

        air_diff = torch.abs(outputs[0]["air_mask_logits"] - outputs[1]["air_mask_logits"]).mean()
        block_diff = torch.abs(
            outputs[0]["block_type_logits"] - outputs[1]["block_type_logits"]
        ).mean()

        assert air_diff > 0.01, f"Air mask should vary with conditioning, got diff={air_diff}"
        assert block_diff > 0.01, (
            f"Block types should vary with conditioning, got diff={block_diff}"
        )
