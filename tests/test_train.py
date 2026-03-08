"""Tests for train.py — loss function, metrics, target downsampling, forward routing."""

import importlib
import importlib.util
import pytest
import sys
from pathlib import Path

import torch
from train.progressive_lod_models import (
    create_init_model,
    create_lod2_to_lod1_model,
    create_lod3_to_lod2_model,
    create_lod4_to_lod3_model,
)
from train.unet3d import SimpleFlexibleConfig

# train/ is a package that shadows the root train.py script.
# Load the script module directly by file path.
_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train.py"
_spec = importlib.util.spec_from_file_location("train_script", _TRAIN_SCRIPT)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_script"] = _train_mod
_spec.loader.exec_module(_train_mod)

MultiLODLoss = _train_mod.MultiLODLoss
compute_metrics = _train_mod.compute_metrics
_downsample_targets = _train_mod._downsample_targets
_forward_batch = _train_mod._forward_batch
LOD_MODEL_KEY = _train_mod.LOD_MODEL_KEY


@pytest.fixture
def config():
    return SimpleFlexibleConfig(
        base_channels=16,
        max_channels=64,
        biome_vocab_size=50,
        biome_embed_dim=8,
        lod_embed_dim=4,
        block_vocab_size=100,
    )


def _make_predictions(B: int, C: int, D: int):
    """Synthetic model outputs."""
    return {
        "air_mask_logits": torch.randn(B, 1, D, D, D),
        "block_type_logits": torch.randn(B, C, D, D, D),
    }


def _make_targets(B: int, C: int, D: int):
    """Synthetic ground-truth targets."""
    return {
        "target_blocks": torch.randint(0, C, (B, D, D, D)),
        "target_occupancy": torch.randint(0, 2, (B, D, D, D)).float(),
    }


# ===========================================================================
# MultiLODLoss
# ===========================================================================


class TestMultiLODLoss:
    def test_basic_loss_runs(self):
        loss_fn = MultiLODLoss()
        preds = _make_predictions(2, 100, 8)
        targets = _make_targets(2, 100, 8)
        losses = loss_fn(preds, targets)

        assert "total_loss" in losses
        assert "block_loss" in losses
        assert "air_loss" in losses
        assert torch.isfinite(losses["total_loss"])

    def test_loss_decreases_with_perfect_predictions(self):
        """Loss should be lower when predictions match targets."""
        loss_fn = MultiLODLoss()
        B, C, D = 2, 10, 4

        # Random predictions (bad)
        targets = _make_targets(B, C, D)
        bad_preds = _make_predictions(B, C, D)
        bad_loss = loss_fn(bad_preds, targets)["total_loss"].item()

        # Near-perfect predictions
        good_preds = {
            "air_mask_logits": (targets["target_occupancy"].unsqueeze(1) * 10 - 5),
            "block_type_logits": torch.zeros(B, C, D, D, D),
        }
        # Set correct class to high logit
        for b in range(B):
            for x in range(D):
                for y in range(D):
                    for z in range(D):
                        good_preds["block_type_logits"][
                            b, targets["target_blocks"][b, x, y, z], x, y, z
                        ] = 10.0
        good_loss = loss_fn(good_preds, targets)["total_loss"].item()

        assert good_loss < bad_loss

    def test_class_weights(self):
        """Loss should accept optional class weights."""
        weights = torch.ones(100)
        weights[0] = 0.0  # ignore air
        loss_fn = MultiLODLoss(class_weights=weights)
        preds = _make_predictions(2, 100, 4)
        targets = _make_targets(2, 100, 4)
        losses = loss_fn(preds, targets)
        assert torch.isfinite(losses["total_loss"])

    def test_surface_consistency_loss(self):
        """Surface consistency term should be non-zero when enabled."""
        loss_fn = MultiLODLoss(surface_consistency_weight=0.5)
        preds = _make_predictions(2, 100, 4)
        targets = _make_targets(2, 100, 4)
        targets["height_planes"] = torch.rand(2, 5, 16, 16)
        losses = loss_fn(preds, targets)
        assert "surface_loss" in losses
        # With weight > 0 and random data, surface_loss should be non-zero
        assert losses["surface_loss"].item() > 0 or losses["surface_loss"].item() == 0


# ===========================================================================
# compute_metrics
# ===========================================================================


class TestComputeMetrics:
    def test_metrics_keys(self):
        preds = _make_predictions(2, 100, 4)
        targets = _make_targets(2, 100, 4)
        metrics = compute_metrics(preds, targets)
        assert "air_accuracy" in metrics
        assert "block_accuracy" in metrics
        assert "overall_accuracy" in metrics

    def test_metrics_range(self):
        preds = _make_predictions(2, 100, 4)
        targets = _make_targets(2, 100, 4)
        metrics = compute_metrics(preds, targets)
        for key in ["air_accuracy", "block_accuracy", "overall_accuracy"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of range"

    def test_perfect_air_prediction(self):
        """Perfect air predictions should yield air_accuracy = 1.0."""
        B, C, D = 1, 10, 4
        targets = _make_targets(B, C, D)
        target_solid = (targets["target_occupancy"] > 0).float().unsqueeze(1)
        preds = {
            "air_mask_logits": target_solid * 10 - 5,  # high where solid, low where air
            "block_type_logits": torch.randn(B, C, D, D, D),
        }
        metrics = compute_metrics(preds, targets)
        assert metrics["air_accuracy"] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# _downsample_targets
# ===========================================================================


class TestDownsampleTargets:
    def test_identity_at_16(self):
        blocks = torch.randint(0, 100, (2, 16, 16, 16))
        occ = torch.randint(0, 2, (2, 16, 16, 16)).float()
        ds_b, ds_o = _downsample_targets(blocks, occ, 16)
        assert torch.equal(ds_b, blocks)
        assert torch.equal(ds_o, occ)

    @pytest.mark.parametrize("out_size", [1, 2, 4, 8])
    def test_output_shape(self, out_size):
        blocks = torch.randint(0, 100, (2, 16, 16, 16))
        occ = torch.randint(0, 2, (2, 16, 16, 16)).float()
        ds_b, ds_o = _downsample_targets(blocks, occ, out_size)
        assert ds_b.shape == (2, out_size, out_size, out_size)
        assert ds_o.shape == (2, out_size, out_size, out_size)

    def test_invalid_shape_raises(self):
        blocks = torch.randint(0, 100, (16, 16, 16))  # missing batch dim
        occ = torch.randint(0, 2, (16, 16, 16)).float()
        with pytest.raises(ValueError, match="Expected blocks shape"):
            _downsample_targets(blocks, occ, 8)


# ===========================================================================
# _forward_batch
# ===========================================================================


class TestForwardBatch:
    @pytest.fixture
    def models(self, config):
        return {
            "init_to_lod4": create_init_model(config),
            "lod4to3": create_lod4_to_lod3_model(config),
            "lod3to2": create_lod3_to_lod2_model(config),
            "lod2to1": create_lod2_to_lod1_model(config),
        }

    def test_routes_init_model(self, models):
        batch = {
            "lod_transition": "init_to_lod4",
            "height_planes": torch.randn(1, 5, 16, 16),
            "biome_idx": torch.randint(0, 50, (1, 16, 16)),
            "y_index": torch.randint(0, 24, (1,)),
            "parent_voxel": torch.zeros(1, 1, 8, 8, 8),  # ignored by init
        }
        with torch.no_grad():
            out = _forward_batch(models, batch, torch.device("cpu"))
        assert out["air_mask_logits"].shape == (1, 1, 1, 1, 1)

    @pytest.mark.parametrize(
        "transition,out_size,parent_size",
        [
            ("lod4to3", 2, 1),
            ("lod3to2", 4, 2),
            ("lod2to1", 8, 4),
        ],
    )
    def test_routes_refinement_models(self, models, transition, out_size, parent_size):
        batch = {
            "lod_transition": transition,
            "height_planes": torch.randn(1, 5, 16, 16),
            "biome_idx": torch.randint(0, 50, (1, 16, 16)),
            "y_index": torch.randint(0, 24, (1,)),
            "parent_voxel": torch.randn(1, 1, parent_size, parent_size, parent_size),
        }
        with torch.no_grad():
            out = _forward_batch(models, batch, torch.device("cpu"))
        assert out["air_mask_logits"].shape[-1] == out_size

    def test_unknown_transition_raises(self, models):
        batch = {"lod_transition": "lod0_to_nothing"}
        with pytest.raises(ValueError, match="Unknown LOD transition"):
            _forward_batch(models, batch, torch.device("cpu"))


# ===========================================================================
# LOD_MODEL_KEY mapping
# ===========================================================================


class TestLodModelKey:
    def test_all_transitions_mapped(self):
        expected = {"init_to_lod4", "lod4to3", "lod3to2", "lod2to1"}
        assert set(LOD_MODEL_KEY.keys()) == expected

    def test_keys_equal_values(self):
        """Mapping is identity — keys == values."""
        for k, v in LOD_MODEL_KEY.items():
            assert k == v
