"""
E2E smoke tests: synthetic data → training step → ONNX export & round-trip.

No real Minecraft world is required; all data is synthesised in-memory or in a
temp-directory.  The tests exercise the full critical path:

  1. Synthetic batch  →  VoxelTrainer.training_step()  →  finite loss
  2. VoxelTrainer.train_one_epoch()  (uses internal dummy batch)
  3. Synthetic NPZ patch  →  MultiLODDataset  →  training_step()
  4. ONNX export (static, opset 17)  →  onnxruntime forward  →  correct shapes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_CFG: Dict[str, Any] = {
    "model": {
        "block_vocab_size": 16,  # tiny vocab – fast tests
        "base_channels": 8,
        "biome_vocab_size": 50,
    },
    "training": {
        "learning_rate": 1e-3,
        "batch_size": 2,
    },
    "loss": {
        "mask_weight": 1.0,
        "type_weight": 1.0,
    },
}

B = 2  # batch size used in manual batches


def _make_batch(device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """Return a synthetic batch dict matching the VoxelTrainer contract."""
    return {
        "parent_voxel": torch.randn(B, 1, 8, 8, 8, device=device),
        "biome_patch": torch.randint(0, 50, (B, 16, 16), device=device),
        "heightmap_patch": torch.randint(50, 100, (B, 1, 16, 16), device=device).float(),
        "y_index": torch.randint(0, 24, (B,), device=device),
        "lod": torch.randint(0, 5, (B,), device=device),
        "target_mask": torch.randint(0, 2, (B, 1, 16, 16, 16), device=device).float(),
        "target_types": torch.randint(0, 16, (B, 16, 16, 16), device=device).long(),
    }


def _make_trainer():
    """Instantiate a VoxelTrainer with a tiny config (no TensorBoard, no multi-LOD)."""
    from train.trainer import VoxelTrainer

    return VoxelTrainer(_TINY_CFG)


# ---------------------------------------------------------------------------
# Test 1 – synthetic batch → training_step → finite loss
# ---------------------------------------------------------------------------


class TestSyntheticTrainingStep:
    def test_loss_is_finite_scalar(self):
        trainer = _make_trainer()
        batch = _make_batch(trainer.device)
        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor), "loss must be a Tensor"
        assert loss.ndim == 0, "loss must be scalar (0-dim)"
        assert torch.isfinite(loss), f"expected finite loss, got {loss.item()}"

    def test_loss_is_positive(self):
        """Cross-entropy on random logits should be > 0."""
        trainer = _make_trainer()
        batch = _make_batch(trainer.device)
        loss = trainer.training_step(batch)
        assert loss.item() > 0.0, f"expected positive loss, got {loss.item()}"

    def test_global_step_increments(self):
        trainer = _make_trainer()
        batch = _make_batch(trainer.device)
        trainer.training_step(batch)
        assert trainer.global_step == 1

    def test_second_step_still_finite(self):
        """Two back-to-back steps must both yield finite loss."""
        trainer = _make_trainer()
        for _ in range(2):
            loss = trainer.training_step(_make_batch(trainer.device))
            assert torch.isfinite(loss), "loss became non-finite on second step"


# ---------------------------------------------------------------------------
# Test 2 – train_one_epoch (no dataloader)
# ---------------------------------------------------------------------------


class TestTrainOneEpoch:
    def test_returns_loss_dict(self):
        trainer = _make_trainer()
        metrics = trainer.train_one_epoch()
        assert "loss" in metrics, "expected 'loss' key in returned dict"

    def test_loss_is_finite(self):
        trainer = _make_trainer()
        metrics = trainer.train_one_epoch()
        assert np.isfinite(metrics["loss"]), f"epoch loss not finite: {metrics['loss']}"

    def test_epoch_counter_increments(self):
        trainer = _make_trainer()
        trainer.train_one_epoch()
        assert trainer.current_epoch == 1


# ---------------------------------------------------------------------------
# Test 3 – synthetic NPZ → MultiLODDataset → training_step
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_npz_dir(tmp_path: Path) -> Path:
    """Write a single synthetic NPZ file and return the containing directory."""
    rng = np.random.default_rng(42)
    labels16 = rng.integers(0, 16, size=(16, 16, 16), dtype=np.int32)
    biome16 = rng.integers(0, 50, size=(16, 16), dtype=np.int32)
    height16 = rng.random(size=(1, 16, 16)).astype(np.float32)

    np.savez_compressed(
        tmp_path / "chunk_train_0000.npz",
        labels16=labels16,
        biome16=biome16,
        height16=height16,
    )
    return tmp_path


class TestNpzDatasetPipeline:
    def test_dataset_loads_npz(self, tmp_npz_dir: Path):
        from train.multi_lod_dataset import MultiLODDataset

        ds = MultiLODDataset(tmp_npz_dir, split="train")
        assert len(ds) > 0, "dataset reported zero items from NPZ"

    def test_dataset_item_shapes(self, tmp_npz_dir: Path):
        from train.multi_lod_dataset import MultiLODDataset

        ds = MultiLODDataset(tmp_npz_dir, split="train")
        item = ds[0]
        # parent_voxel must be (1, 1, S, S, S) for some S
        pv = item["parent_voxel"]
        assert pv.ndim == 5, "parent_voxel must be 5-D"
        assert pv.shape[1] == 1, "parent_voxel channel must be 1"

    def test_dataset_item_to_training_step(self, tmp_npz_dir: Path):
        """Adapt a MultiLODDataset item to the batch format and run a training step."""
        from train.multi_lod_dataset import MultiLODDataset

        ds = MultiLODDataset(tmp_npz_dir, split="train")
        item = ds[0]

        pv = torch.from_numpy(np.array(item["parent_voxel"], dtype=np.float32))  # (1,1,S,S,S)
        # Interpolate parent to canonical 8³ if needed
        if pv.shape[2:] != (8, 8, 8):
            pv = torch.nn.functional.interpolate(pv, size=(8, 8, 8), mode="nearest")

        batch = {
            "parent_voxel": pv,  # (1,1,8,8,8)
            "biome_patch": torch.randint(0, 50, (1, 16, 16)),
            "heightmap_patch": torch.randn(1, 1, 16, 16),
            "y_index": torch.tensor([0], dtype=torch.long),
            "lod": torch.tensor([1], dtype=torch.long),
            "target_mask": torch.from_numpy(
                np.array(item["target_occupancy"], dtype=np.float32)  # (1, S, S, S)
            ).unsqueeze(1)[
                :, :, :16, :16, :16
            ],  # -> (1, 1, S, S, S)
            "target_types": torch.randint(0, 16, (1, 16, 16, 16)).long(),
        }
        # Pad/crop target_mask to (1,1,16,16,16)
        if batch["target_mask"].shape[2] != 16:
            batch["target_mask"] = torch.nn.functional.interpolate(
                batch["target_mask"].float(), size=(16, 16, 16), mode="nearest"
            )

        trainer = _make_trainer()
        loss = trainer.training_step(batch)
        assert torch.isfinite(loss), "training step on NPZ-derived batch gave non-finite loss"


# ---------------------------------------------------------------------------
# Test 4 – ONNX export (static, opset 17) → onnxruntime round-trip
# ---------------------------------------------------------------------------


class TestOnnxExportRoundtrip:
    @pytest.fixture(autouse=True)
    def _skip_if_no_ort(self):
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        pytest.importorskip("onnx", reason="onnx not installed")

    def _export(self, onnx_path: Path):
        """Export a tiny model to ONNX and return dummy_input tuple."""
        import torch

        from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D

        model = SimpleFlexibleUNet3D(SimpleFlexibleConfig(**_TINY_CFG["model"]))
        model.eval()

        # Wrap to expose ordered positional outputs (block_logits, air_mask)
        class _ExportWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, parent_voxel, biome_patch, heightmap_patch, y_index, lod):
                out = self.m(
                    parent_voxel=parent_voxel,
                    biome_patch=biome_patch,
                    heightmap_patch=heightmap_patch,
                    y_index=y_index,
                    lod=lod,
                )
                return out["block_type_logits"], out["air_mask_logits"]

        wrapper = _ExportWrapper(model)
        wrapper.eval()

        dummy_input = (
            torch.randn(1, 1, 8, 8, 8),
            torch.randint(0, 50, (1, 16, 16)),
            torch.randn(1, 1, 16, 16),
            torch.randint(0, 24, (1,)),
            torch.randint(1, 5, (1,)),
        )

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["parent_voxel", "biome_patch", "heightmap_patch", "y_index", "lod"],
            output_names=["block_logits", "air_mask"],
            dynamic_axes=None,
        )
        return dummy_input

    def test_onnx_model_is_valid(self, tmp_path: Path):
        import onnx

        onnx_path = tmp_path / "model.onnx"
        self._export(onnx_path)
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)  # raises if invalid

    def test_onnx_opset_at_least_17(self, tmp_path: Path):
        import onnx

        onnx_path = tmp_path / "model.onnx"
        self._export(onnx_path)
        m = onnx.load(str(onnx_path))
        opset = max(o.version for o in m.opset_import)
        assert opset >= 17, f"expected opset ≥ 17, got {opset}"

    def test_onnx_output_names(self, tmp_path: Path):
        import onnx

        onnx_path = tmp_path / "model.onnx"
        self._export(onnx_path)
        m = onnx.load(str(onnx_path))
        out_names = [o.name for o in m.graph.output]
        assert "block_logits" in out_names, f"'block_logits' missing from {out_names}"
        assert "air_mask" in out_names, f"'air_mask' missing from {out_names}"

    def test_onnx_no_dynamic_axes(self, tmp_path: Path):
        """All output dimensions must be static (> 0)."""
        import onnx

        onnx_path = tmp_path / "model.onnx"
        self._export(onnx_path)
        m = onnx.load(str(onnx_path))
        for out in m.graph.output:
            shape = out.type.tensor_type.shape
            for i, dim in enumerate(shape.dim):
                assert dim.dim_value > 0, f"output '{out.name}' dim[{i}] is dynamic or zero: {dim}"

    def test_onnxruntime_forward_output_shapes(self, tmp_path: Path):
        """OnnxRuntime forward pass produces the expected output shapes."""
        import onnxruntime as ort

        onnx_path = tmp_path / "model.onnx"
        dummy_input = self._export(onnx_path)

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        # Build feed dict from only the inputs present in the exported graph
        # (y_index may be pruned if unused by the model)
        full_feed = {
            "parent_voxel": dummy_input[0].numpy(),
            "biome_patch": dummy_input[1].numpy(),
            "heightmap_patch": dummy_input[2].numpy(),
            "y_index": dummy_input[3].numpy(),
            "lod": dummy_input[4].numpy(),
        }
        graph_inputs = {inp.name for inp in sess.get_inputs()}
        feed = {k: v for k, v in full_feed.items() if k in graph_inputs}
        outputs = sess.run(None, feed)

        block_logits: np.ndarray = outputs[0]  # type: ignore[assignment]
        air_mask: np.ndarray = outputs[1]  # type: ignore[assignment]
        vocab = _TINY_CFG["model"]["block_vocab_size"]
        assert block_logits.shape == (
            1,
            vocab,
            16,
            16,
            16,
        ), f"block_logits shape mismatch: {block_logits.shape}"
        assert air_mask.shape == (1, 1, 16, 16, 16), f"air_mask shape mismatch: {air_mask.shape}"

    def test_onnxruntime_outputs_finite(self, tmp_path: Path):
        import onnxruntime as ort

        onnx_path = tmp_path / "model.onnx"
        dummy_input = self._export(onnx_path)

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        full_feed = {
            "parent_voxel": dummy_input[0].numpy(),
            "biome_patch": dummy_input[1].numpy(),
            "heightmap_patch": dummy_input[2].numpy(),
            "y_index": dummy_input[3].numpy(),
            "lod": dummy_input[4].numpy(),
        }
        graph_inputs = {inp.name for inp in sess.get_inputs()}
        feed = {k: v for k, v in full_feed.items() if k in graph_inputs}
        for arr in sess.run(None, feed):
            arr_np: np.ndarray = arr  # type: ignore[assignment]
            assert np.isfinite(arr_np).all(), "ONNX output contains non-finite values"
