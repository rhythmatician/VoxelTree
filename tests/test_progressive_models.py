"""Tests for train/progressive_lod_models.py — 4-model family forward passes."""

import pytest
import torch

from train.progressive_lod_models import (
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
    create_init_model,
    create_lod2_to_lod1_model,
    create_lod3_to_lod2_model,
    create_lod4_to_lod3_model,
)
from train.unet3d import SimpleFlexibleConfig


@pytest.fixture
def config():
    return SimpleFlexibleConfig(
        base_channels=16,
        biome_vocab_size=50,
        biome_embed_dim=8,
        block_vocab_size=100,
    )


def _dummy_conditioning(batch_size: int = 2):
    """Shared conditioning inputs for all 4 models."""
    return {
        "height_planes": torch.randn(batch_size, 5, 16, 16),
        "biome_indices": torch.randint(0, 50, (batch_size, 16, 16)),
        "y_index": torch.randint(0, 24, (batch_size,)),
    }


# ===========================================================================
# Init model (Nothing → LOD4, 1×1×1)
# ===========================================================================


class TestInitModel:
    def test_factory(self, config):
        model = create_init_model(config)
        assert isinstance(model, ProgressiveLODModel0_Initial)
        assert model.output_size == 1

    def test_forward_shape(self, config):
        model = create_init_model(config)
        cond = _dummy_conditioning(batch_size=3)
        with torch.no_grad():
            out = model(**cond)
        assert out["air_mask_logits"].shape == (3, 1, 1, 1, 1)
        assert out["block_type_logits"].shape == (3, 100, 1, 1, 1)

    def test_gradient_flow(self, config):
        model = create_init_model(config)
        cond = _dummy_conditioning(batch_size=1)
        cond["height_planes"].requires_grad_(True)
        out = model(**cond)
        loss = out["air_mask_logits"].sum() + out["block_type_logits"].sum()
        loss.backward()
        # height_planes feeds into the network, so gradients should flow
        assert cond["height_planes"].grad is not None


# ===========================================================================
# Refinement models (LOD4→3, LOD3→2, LOD2→1)
# ===========================================================================


class TestRefinementModels:
    @pytest.mark.parametrize(
        "factory,output_size,parent_size",
        [
            (create_lod4_to_lod3_model, 2, 1),
            (create_lod3_to_lod2_model, 4, 2),
            (create_lod2_to_lod1_model, 8, 4),
        ],
    )
    def test_forward_shape(self, config, factory, output_size, parent_size):
        model = factory(config)
        B = 2
        cond = _dummy_conditioning(batch_size=B)
        cond["x_parent"] = torch.randn(B, 1, parent_size, parent_size, parent_size)

        with torch.no_grad():
            out = model(**cond)

        assert out["air_mask_logits"].shape == (B, 1, output_size, output_size, output_size)
        assert out["block_type_logits"].shape == (
            B,
            100,
            output_size,
            output_size,
            output_size,
        )

    @pytest.mark.parametrize(
        "factory,parent_size",
        [
            (create_lod4_to_lod3_model, 1),
            (create_lod3_to_lod2_model, 2),
            (create_lod2_to_lod1_model, 4),
        ],
    )
    def test_gradient_flow(self, config, factory, parent_size):
        model = factory(config)
        cond = _dummy_conditioning(batch_size=1)
        parent = torch.randn(1, 1, parent_size, parent_size, parent_size, requires_grad=True)
        cond["x_parent"] = parent
        out = model(**cond)
        loss = out["block_type_logits"].sum()
        loss.backward()
        assert parent.grad is not None
        assert torch.isfinite(parent.grad).all()

    def test_invalid_output_size_raises(self, config):
        with pytest.raises(AssertionError, match="output_size must be 2, 4, or 8"):
            ProgressiveLODModel(config, output_size=16)

    def test_all_zeros_input(self, config):
        """Models should produce finite outputs even with all-zero inputs."""
        model = create_lod2_to_lod1_model(config)
        B = 1
        cond = {
            "height_planes": torch.zeros(B, 5, 16, 16),
            "biome_indices": torch.zeros(B, 16, 16, dtype=torch.long),
            "y_index": torch.zeros(B, dtype=torch.long),
            "x_parent": torch.zeros(B, 1, 4, 4, 4),
        }
        with torch.no_grad():
            out = model(**cond)
        assert torch.isfinite(out["air_mask_logits"]).all()
        assert torch.isfinite(out["block_type_logits"]).all()


# ===========================================================================
# Model family consistency
# ===========================================================================


class TestModelFamily:
    def test_four_factories_produce_distinct_models(self, config):
        m0 = create_init_model(config)
        m1 = create_lod4_to_lod3_model(config)
        m2 = create_lod3_to_lod2_model(config)
        m3 = create_lod2_to_lod1_model(config)

        assert type(m0) is ProgressiveLODModel0_Initial
        assert type(m1) is ProgressiveLODModel
        assert type(m2) is ProgressiveLODModel
        assert type(m3) is ProgressiveLODModel

        # Different output sizes
        assert m1.output_size == 2
        assert m2.output_size == 4
        assert m3.output_size == 8

    def test_param_counts_increase_with_resolution(self, config):
        """Larger output models should have more parameters."""
        m1 = create_lod4_to_lod3_model(config)
        m2 = create_lod3_to_lod2_model(config)
        m3 = create_lod2_to_lod1_model(config)

        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        p3 = sum(p.numel() for p in m3.parameters())

        assert p1 < p2 < p3
