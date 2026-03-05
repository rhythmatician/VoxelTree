"""Clean unit tests for SimpleFlexibleUNet3D."""

import pytest
import torch

from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D


class TestSimpleFlexibleUNet3D:
    @pytest.fixture
    def basic_config(self):
        return SimpleFlexibleConfig(
            base_channels=32,
            max_channels=128,
            biome_vocab_size=50,
            biome_embed_dim=16,
            lod_embed_dim=8,
            block_vocab_size=100,
        )

    def test_model_instantiation(self, basic_config):
        model = SimpleFlexibleUNet3D(basic_config)
        assert isinstance(model, SimpleFlexibleUNet3D)
        assert model.config == basic_config

    def test_model_forward_pass_shapes(self, basic_config):
        model = SimpleFlexibleUNet3D(basic_config)
        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                parent_voxel=parent_voxel,
                biome_patch=biome_patch,
                heightmap_patch=heightmap_patch,
                y_index=y_index,
                lod=lod,
            )

        assert "air_mask_logits" in outputs
        assert "block_type_logits" in outputs
        assert outputs["air_mask_logits"].shape == (batch_size, 1, 16, 16, 16)
        expected_shape = (batch_size, basic_config.block_vocab_size, 16, 16, 16)
        assert outputs["block_type_logits"].shape == expected_shape

    def test_model_gradient_flow(self, basic_config):
        model = SimpleFlexibleUNet3D(basic_config)
        batch_size = 1
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=True)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)

        outputs = model(
            parent_voxel=parent_voxel,
            biome_patch=biome_patch,
            heightmap_patch=heightmap_patch,
            y_index=y_index,
            lod=lod,
        )

        loss = outputs["air_mask_logits"].sum() + outputs["block_type_logits"].sum()
        loss.backward()

        assert parent_voxel.grad is not None
        assert torch.isfinite(parent_voxel.grad).all()


class TestSimpleFlexibleConfig:
    def test_config_instantiation(self):
        config = SimpleFlexibleConfig()
        assert config.base_channels > 0
        assert config.max_channels >= config.base_channels
        assert config.biome_vocab_size > 0
        assert config.biome_embed_dim > 0
        assert config.lod_embed_dim > 0
        assert config.block_vocab_size > 0

    def test_config_custom_values(self):
        config = SimpleFlexibleConfig(
            base_channels=64,
            max_channels=256,
            biome_vocab_size=100,
            biome_embed_dim=32,
            lod_embed_dim=16,
            block_vocab_size=200,
        )
        assert config.base_channels == 64
        assert config.max_channels == 256
        assert config.biome_vocab_size == 100
        assert config.biome_embed_dim == 32
        assert config.lod_embed_dim == 16
        assert config.block_vocab_size == 200
