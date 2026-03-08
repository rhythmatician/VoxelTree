"""Tests for scripts/export_lod.py — ONNX export adapters and checkpoint loading."""

import json
import sys
from pathlib import Path

import pytest
import torch

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_lod import (  # noqa: E402
    MODEL_STEPS,
    InitModelAdapter,
    RefinementModelAdapter,
    export_step,
    load_progressive_checkpoint,
)
from train.progressive_lod_models import (  # noqa: E402
    create_init_model,
    create_lod2_to_lod1_model,
    create_lod3_to_lod2_model,
    create_lod4_to_lod3_model,
)
from train.unet3d import SimpleFlexibleConfig  # noqa: E402


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


# ===========================================================================
# MODEL_STEPS
# ===========================================================================


class TestModelSteps:
    def test_four_steps(self):
        assert len(MODEL_STEPS) == 4

    def test_step_keys(self):
        keys = [s[0] for s in MODEL_STEPS]
        assert keys == ["init_to_lod4", "lod4to3", "lod3to2", "lod2to1"]

    def test_output_sizes_increase(self):
        output_sizes = [s[2] for s in MODEL_STEPS]
        assert output_sizes == [1, 2, 4, 8]

    def test_onnx_filenames(self):
        filenames = [s[4] for s in MODEL_STEPS]
        assert all(f.endswith(".onnx") for f in filenames)


# ===========================================================================
# InitModelAdapter
# ===========================================================================


class TestInitModelAdapter:
    def test_forward_shape(self, config):
        model = create_init_model(config)
        adapter = InitModelAdapter(model)

        hp = torch.randn(1, 5, 16, 16)
        biome = torch.randint(0, 50, (1, 16, 16))
        yi = torch.tensor([5], dtype=torch.long)

        block_logits, air_mask = adapter(hp, biome, yi)

        assert block_logits.shape == (1, 100, 1, 1, 1)
        assert air_mask.shape == (1, 1, 1, 1, 1)

    def test_output_is_finite(self, config):
        model = create_init_model(config)
        adapter = InitModelAdapter(model)

        hp = torch.randn(1, 5, 16, 16)
        biome = torch.randint(0, 50, (1, 16, 16))
        yi = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            bl, am = adapter(hp, biome, yi)
        assert torch.isfinite(bl).all()
        assert torch.isfinite(am).all()


# ===========================================================================
# RefinementModelAdapter
# ===========================================================================


class TestRefinementModelAdapter:
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
        adapter = RefinementModelAdapter(model)

        hp = torch.randn(1, 5, 16, 16)
        biome = torch.randint(0, 50, (1, 16, 16))
        yi = torch.tensor([10], dtype=torch.long)
        parent = torch.randn(1, 1, parent_size, parent_size, parent_size)

        block_logits, air_mask = adapter(hp, biome, yi, parent)

        assert block_logits.shape == (1, 100, output_size, output_size, output_size)
        assert air_mask.shape == (1, 1, output_size, output_size, output_size)


# ===========================================================================
# load_progressive_checkpoint
# ===========================================================================


class TestLoadProgressiveCheckpoint:
    def test_loads_valid_checkpoint(self, config, tmp_path):
        """Round-trip: save checkpoint → load_progressive_checkpoint."""
        models = {
            "init_to_lod4": create_init_model(config),
            "lod4to3": create_lod4_to_lod3_model(config),
            "lod3to2": create_lod3_to_lod2_model(config),
            "lod2to1": create_lod2_to_lod1_model(config),
        }
        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save(
            {
                "config": config,
                "model_state_dicts": {k: m.state_dict() for k, m in models.items()},
                "epoch": 5,
            },
            ckpt_path,
        )

        loaded_config, loaded_models = load_progressive_checkpoint(ckpt_path)

        assert loaded_config.block_vocab_size == config.block_vocab_size
        assert set(loaded_models.keys()) == set(models.keys())

    def test_missing_config_raises(self, tmp_path):
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"model_state_dicts": {}}, ckpt_path)
        with pytest.raises(ValueError, match="no 'config' key"):
            load_progressive_checkpoint(ckpt_path)

    def test_missing_state_dicts_raises(self, config, tmp_path):
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"config": config}, ckpt_path)
        with pytest.raises(ValueError, match="no 'model_state_dicts' key"):
            load_progressive_checkpoint(ckpt_path)


# ===========================================================================
# export_step (integration, skips ONNX export if onnx not available)
# ===========================================================================


class TestExportStep:
    @pytest.mark.skipif(
        not hasattr(torch, "onnx"),
        reason="torch.onnx not available",
    )
    def test_init_model_export(self, config, tmp_path):
        model = create_init_model(config)
        adapter = InitModelAdapter(model)

        onnx_path = export_step(
            adapter=adapter,
            step_name="init_to_lod4",
            output_size=1,
            parent_size=0,
            onnx_filename="init_to_lod4.onnx",
            config=config,
            out_dir=tmp_path,
        )

        assert onnx_path.exists()
        assert (tmp_path / "init_to_lod4_config.json").exists()
        assert (tmp_path / "init_to_lod4_test_vectors.npz").exists()

        # Validate sidecar config
        with open(tmp_path / "init_to_lod4_config.json") as f:
            cfg = json.load(f)
        assert cfg["step"] == "init_to_lod4"
        assert cfg["output_resolution"] == 1
        assert cfg["block_vocab_size"] == 100

    @pytest.mark.skipif(
        not hasattr(torch, "onnx"),
        reason="torch.onnx not available",
    )
    def test_refinement_model_export(self, config, tmp_path):
        model = create_lod2_to_lod1_model(config)
        adapter = RefinementModelAdapter(model)

        onnx_path = export_step(
            adapter=adapter,
            step_name="lod2to1",
            output_size=8,
            parent_size=4,
            onnx_filename="refine_lod2_to_lod1.onnx",
            config=config,
            out_dir=tmp_path,
        )

        assert onnx_path.exists()
        sidecar = tmp_path / "refine_lod2_to_lod1_config.json"
        assert sidecar.exists()

        with open(sidecar) as f:
            cfg = json.load(f)
        assert cfg["output_resolution"] == 8
        assert "x_parent" in cfg["inputs"]
