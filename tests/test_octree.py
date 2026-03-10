"""Tests for the octree pipeline.

Covers:
  1.  Coordinate math          — child/parent round-trips, octant bit layout
  2.  Octant extraction        — extract_octant_and_upsample correctness
  3.  Section index builder    — build_section_index filename parsing
  4.  Level → model-type map   — _model_type_for_level routing
  5.  OctreeInitModel          — shape, keys, gradient flow
  6.  OctreeRefineModel        — shape, level embedding, missing-parent error
  7.  OctreeLeafModel          — shape, no occ head, missing-parent error
  8.  OctreeDataset            — cache loading, sample structure
  9.  collate_octree_batch     — majority-type selection, empty handling
  10. _bitmask_to_binary       — bit extraction correctness
  11. OctreeLoss               — all model types, occ_weight scaling, backward
  12. compute_octree_metrics   — accuracy ranges, perfect prediction, occ F1
  13. _prepare_targets         — tensor extraction from batch dict
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import scripts/build_octree_pairs.py (not a package, import by file)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_octree_pairs import (  # noqa: E402
    build_section_index,
    child_coords_from_parent,
    extract_octant_and_upsample,
    parent_coords_and_octant,
)

# ---------------------------------------------------------------------------
# Import from train/ package
# ---------------------------------------------------------------------------

from train.octree_dataset import (
    OctreeDataset,
    _model_type_for_level,
    collate_octree_batch,
)
from train.octree_models import (
    OctreeConfig,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)

# ---------------------------------------------------------------------------
# Import from train_octree.py (root script — not a package member)
# ---------------------------------------------------------------------------

_TRAIN_OCTREE_PATH = Path(__file__).resolve().parent.parent / "train_octree.py"
_spec = importlib.util.spec_from_file_location("train_octree_mod", _TRAIN_OCTREE_PATH)
_train_octree_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_octree_mod"] = _train_octree_mod
_spec.loader.exec_module(_train_octree_mod)

OctreeLoss = _train_octree_mod.OctreeLoss
_bitmask_to_binary = _train_octree_mod._bitmask_to_binary
compute_octree_metrics = _train_octree_mod.compute_octree_metrics
_prepare_targets = _train_octree_mod._prepare_targets


# ===========================================================================
# Shared helpers & fixtures
# ===========================================================================


@pytest.fixture
def tiny_config() -> OctreeConfig:
    """Tiny channel widths for fast CPU tests."""
    return OctreeConfig(
        block_vocab_size=32,
        biome_vocab_size=16,
        y_vocab_size=24,
        level_vocab_size=5,
        init_channels=(8, 16, 32),
        refine_channels=(8, 16, 32),
        leaf_channels=(8, 16, 32),
        parent_embed_dim=4,
        biome_embed_dim=4,
        y_embed_dim=4,
        level_embed_dim=4,
        height_channels=5,
    )


def _make_batch(B: int, level: int = 2, vocab: int = 32) -> Dict[str, Any]:
    """Return a fully populated batch dict matching collate_octree_batch output."""
    return {
        "labels32": torch.randint(0, vocab, (B, 32, 32, 32)),
        "parent_labels32": torch.randint(0, vocab, (B, 32, 32, 32)),
        "heightmap32": torch.randn(B, 5, 32, 32),
        "biome32": torch.randint(0, 16, (B, 32, 32)),
        "y_position": torch.randint(0, 24, (B,)),
        "level": torch.full((B,), level, dtype=torch.long),
        "non_empty_children": torch.randint(0, 256, (B,), dtype=torch.long),
        "model_type": _model_type_for_level(level),
    }


def _write_pair_cache(path: Path, n: int = 20, vocab: int = 32) -> None:
    """Write a synthetic ``*_octree_pairs.npz`` to *path*."""
    rng = np.random.RandomState(0)
    # Four samples at each level: L4 (init), L3/L2/L1 (refine), L0 (leaf)
    per_level = n // 5
    levels = np.array(
        [4] * per_level + [3] * per_level + [2] * per_level + [1] * per_level + [0] * per_level,
        dtype=np.int64,
    )[:n]
    np.savez_compressed(
        path,
        labels32=rng.randint(0, vocab, (n, 32, 32, 32), dtype=np.int32),
        parent_labels32=rng.randint(0, vocab, (n, 32, 32, 32), dtype=np.int32),
        heightmap32=rng.randn(n, 5, 32, 32).astype(np.float32),
        biome32=rng.randint(0, 16, (n, 32, 32), dtype=np.int32),
        y_position=rng.randint(0, 24, n).astype(np.int64),
        level=levels,
        non_empty_children=rng.randint(0, 256, n, dtype=np.uint8),
    )


# ===========================================================================
# 1. Coordinate Math
# ===========================================================================


class TestChildParentRoundTrip:
    @pytest.mark.parametrize("octant", range(8))
    def test_round_trip_positive_coords(self, octant: int) -> None:
        px, py, pz = 3, 5, 7
        cx, cy, cz = child_coords_from_parent(px, py, pz, octant)
        rpx, rpy, rpz, roct = parent_coords_and_octant(cx, cy, cz)
        assert (rpx, rpy, rpz) == (px, py, pz), f"Parent mismatch for octant {octant}"
        assert roct == octant, f"Octant mismatch: expected {octant}, got {roct}"

    @pytest.mark.parametrize("octant", range(8))
    def test_round_trip_negative_coords(self, octant: int) -> None:
        px, py, pz = -4, -2, -8
        cx, cy, cz = child_coords_from_parent(px, py, pz, octant)
        rpx, rpy, rpz, roct = parent_coords_and_octant(cx, cy, cz)
        assert (rpx, rpy, rpz) == (px, py, pz)
        assert roct == octant

    def test_bit0_is_x_offset(self) -> None:
        cx0, _, _ = child_coords_from_parent(0, 0, 0, 0b000)
        cx1, _, _ = child_coords_from_parent(0, 0, 0, 0b001)
        assert cx1 == cx0 + 1

    def test_bit1_is_z_offset(self) -> None:
        _, _, cz0 = child_coords_from_parent(0, 0, 0, 0b000)
        _, _, cz2 = child_coords_from_parent(0, 0, 0, 0b010)
        assert cz2 == cz0 + 1

    def test_bit2_is_y_offset(self) -> None:
        _, cy0, _ = child_coords_from_parent(0, 0, 0, 0b000)
        _, cy4, _ = child_coords_from_parent(0, 0, 0, 0b100)
        assert cy4 == cy0 + 1

    def test_all_eight_children_distinct(self) -> None:
        children = [child_coords_from_parent(0, 0, 0, oct) for oct in range(8)]
        assert len(set(children)) == 8, "All 8 child coords must be distinct"


# ===========================================================================
# 2. Octant Extraction & Upsampling
# ===========================================================================


class TestExtractOctantAndUpsample:
    def _checkerboard(self) -> np.ndarray:
        """32³ array where value equals the octant index for that voxel."""
        a = np.zeros((32, 32, 32), dtype=np.int32)
        for y in range(32):
            for z in range(32):
                for x in range(32):
                    a[y, z, x] = (x // 16) | ((z // 16) << 1) | ((y // 16) << 2)
        return a

    @pytest.mark.parametrize("octant", range(8))
    def test_extracts_correct_region(self, octant: int) -> None:
        """Each octant pulls from the matching 16³ sub-volume."""
        a = self._checkerboard()
        out = extract_octant_and_upsample(a, octant)
        assert out.shape == (32, 32, 32)
        assert (out == octant).all(), (
            f"Octant {octant} (bin {octant:03b}) contains unexpected values: {np.unique(out)}"
        )

    def test_output_shape_and_dtype(self) -> None:
        a = np.zeros((32, 32, 32), dtype=np.int32)
        out = extract_octant_and_upsample(a, 0)
        assert out.shape == (32, 32, 32)
        assert out.dtype == np.int32

    def test_nearest_neighbor_2x_upsample(self) -> None:
        """A single set voxel in the 16³ sub-block should map to a 2³ block of 8."""
        labels = np.zeros((32, 32, 32), dtype=np.int32)
        labels[1, 1, 1] = 7  # first voxel inside octant 0 region, offset (1,1,1)
        out = extract_octant_and_upsample(labels, 0)
        # [1,1,1] upsampls to [2:4, 2:4, 2:4]
        assert out[2, 2, 2] == 7
        assert out[3, 3, 3] == 7
        assert out[4, 4, 4] == 0  # next block is empty

    def test_upsample_doubles_volume(self) -> None:
        """Non-zero count after 2× upsample = 8× original non-zero count."""
        labels = np.zeros((32, 32, 32), dtype=np.int32)
        labels[0, 0, 0] = 1  # single non-zero in octant 0
        out = extract_octant_and_upsample(labels, 0)
        assert (out > 0).sum() == 8  # 2³ = 8


# ===========================================================================
# 3. Section Index Builder
# ===========================================================================


class TestBuildSectionIndex:
    def test_parses_valid_filenames(self, tmp_path: Path) -> None:
        level_dir = tmp_path / "level_2"
        level_dir.mkdir()
        (level_dir / "voxy_L2_x0_y-1_z3.npz").touch()
        (level_dir / "voxy_L2_x-2_y0_z0.npz").touch()
        idx = build_section_index(tmp_path, level=2)
        assert (0, -1, 3) in idx
        assert (-2, 0, 0) in idx
        assert len(idx) == 2

    def test_missing_level_dir_returns_empty(self, tmp_path: Path) -> None:
        assert build_section_index(tmp_path, level=99) == {}

    def test_ignores_non_npz_files(self, tmp_path: Path) -> None:
        level_dir = tmp_path / "level_0"
        level_dir.mkdir()
        (level_dir / "voxy_L0_x1_y0_z0.npz").touch()
        (level_dir / "voxy_L0_x2_y0_z0.json").touch()
        (level_dir / "not_a_section.npz").touch()
        idx = build_section_index(tmp_path, level=0)
        assert len(idx) == 1
        assert (1, 0, 0) in idx


# ===========================================================================
# 4. Level → Model-Type Routing
# ===========================================================================


class TestModelTypeForLevel:
    def test_level4_is_init(self) -> None:
        assert _model_type_for_level(4) == "init"

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_refine_levels(self, level: int) -> None:
        assert _model_type_for_level(level) == "refine"

    def test_level0_is_leaf(self) -> None:
        assert _model_type_for_level(0) == "leaf"

    def test_invalid_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _model_type_for_level(5)


# ===========================================================================
# 5. OctreeInitModel
# ===========================================================================


class TestOctreeInitModel:
    def test_output_keys(self, tiny_config: OctreeConfig) -> None:
        model = create_init_model(tiny_config)
        out = model(
            heightmap=torch.randn(2, 5, 32, 32),
            biome=torch.randint(0, 16, (2, 32, 32)),
            y_position=torch.randint(0, 24, (2,)),
        )
        assert "block_type_logits" in out
        assert "occ_logits" in out

    def test_output_shapes(self, tiny_config: OctreeConfig) -> None:
        model = create_init_model(tiny_config)
        B, V = 3, tiny_config.block_vocab_size
        out = model(
            heightmap=torch.randn(B, 5, 32, 32),
            biome=torch.randint(0, 16, (B, 32, 32)),
            y_position=torch.randint(0, 24, (B,)),
        )
        assert out["block_type_logits"].shape == (B, V, 32, 32, 32)
        assert out["occ_logits"].shape == (B, 8)

    def test_batch_size_one(self, tiny_config: OctreeConfig) -> None:
        model = create_init_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
        )
        assert out["block_type_logits"].shape[0] == 1

    def test_gradient_flows_through_heightmap(self, tiny_config: OctreeConfig) -> None:
        model = create_init_model(tiny_config)
        hm = torch.randn(1, 5, 32, 32, requires_grad=True)
        out = model(
            heightmap=hm,
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([5]),
        )
        out["block_type_logits"].sum().backward()
        assert hm.grad is not None

    def test_no_parent_context_argument(self, tiny_config: OctreeConfig) -> None:
        """Init model takes no parent_blocks argument at all."""
        model = create_init_model(tiny_config)
        import inspect
        sig = inspect.signature(model.forward)
        assert "parent_blocks" not in sig.parameters
        assert "parent_context" not in sig.parameters


# ===========================================================================
# 6. OctreeRefineModel
# ===========================================================================


class TestOctreeRefineModel:
    def _forward(
        self,
        model,
        B: int = 2,
        level: int = 2,
        vocab: int = 32,
    ) -> Dict[str, torch.Tensor]:
        return model(
            heightmap=torch.randn(B, 5, 32, 32),
            biome=torch.randint(0, 16, (B, 32, 32)),
            y_position=torch.randint(0, 24, (B,)),
            level=torch.full((B,), level, dtype=torch.long),
            parent_blocks=torch.randint(0, vocab, (B, 32, 32, 32)),
        )

    def test_output_keys(self, tiny_config: OctreeConfig) -> None:
        model = create_refine_model(tiny_config)
        out = self._forward(model)
        assert "block_type_logits" in out
        assert "occ_logits" in out

    def test_output_shapes(self, tiny_config: OctreeConfig) -> None:
        model = create_refine_model(tiny_config)
        B, V = 2, tiny_config.block_vocab_size
        out = self._forward(model, B=B, vocab=V)
        assert out["block_type_logits"].shape == (B, V, 32, 32, 32)
        assert out["occ_logits"].shape == (B, 8)

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_all_three_levels_forward(self, tiny_config: OctreeConfig, level: int) -> None:
        model = create_refine_model(tiny_config)
        out = self._forward(model, B=1, level=level)
        assert out["block_type_logits"].shape[0] == 1

    def test_different_levels_produce_different_logits(self, tiny_config: OctreeConfig) -> None:
        """Level embedding must change the output — not a no-op."""
        model = create_refine_model(tiny_config)
        torch.manual_seed(0)
        hm = torch.randn(1, 5, 32, 32)
        biome = torch.randint(0, 16, (1, 32, 32))
        y = torch.tensor([5])
        parent = torch.randint(0, tiny_config.block_vocab_size, (1, 32, 32, 32))

        out1 = model(heightmap=hm, biome=biome, y_position=y, level=torch.tensor([1]), parent_blocks=parent)
        out3 = model(heightmap=hm, biome=biome, y_position=y, level=torch.tensor([3]), parent_blocks=parent)

        assert not torch.allclose(out1["block_type_logits"], out3["block_type_logits"]), (
            "Level 1 and level 3 should produce different logits"
        )

    def test_gradient_flows_through_heightmap(self, tiny_config: OctreeConfig) -> None:
        model = create_refine_model(tiny_config)
        hm = torch.randn(1, 5, 32, 32, requires_grad=True)
        out = model(
            heightmap=hm,
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([5]),
            level=torch.tensor([2]),
            parent_blocks=torch.randint(0, tiny_config.block_vocab_size, (1, 32, 32, 32)),
        )
        out["block_type_logits"].sum().backward()
        assert hm.grad is not None

    def test_missing_parent_raises_value_error(self, tiny_config: OctreeConfig) -> None:
        model = create_refine_model(tiny_config)
        with pytest.raises(ValueError, match="parent"):
            model(
                heightmap=torch.randn(1, 5, 32, 32),
                biome=torch.randint(0, 16, (1, 32, 32)),
                y_position=torch.tensor([5]),
                level=torch.tensor([2]),
            )


# ===========================================================================
# 7. OctreeLeafModel
# ===========================================================================


class TestOctreeLeafModel:
    def _forward(self, model, B: int = 2, vocab: int = 32) -> Dict[str, torch.Tensor]:
        return model(
            heightmap=torch.randn(B, 5, 32, 32),
            biome=torch.randint(0, 16, (B, 32, 32)),
            y_position=torch.randint(0, 24, (B,)),
            parent_blocks=torch.randint(0, vocab, (B, 32, 32, 32)),
        )

    def test_output_keys(self, tiny_config: OctreeConfig) -> None:
        model = create_leaf_model(tiny_config)
        out = self._forward(model)
        assert "block_type_logits" in out
        assert "occ_logits" not in out, "Leaf model must not produce occ_logits"

    def test_output_shape(self, tiny_config: OctreeConfig) -> None:
        model = create_leaf_model(tiny_config)
        B, V = 2, tiny_config.block_vocab_size
        out = self._forward(model, B=B, vocab=V)
        assert out["block_type_logits"].shape == (B, V, 32, 32, 32)

    def test_gradient_flows_through_heightmap(self, tiny_config: OctreeConfig) -> None:
        model = create_leaf_model(tiny_config)
        hm = torch.randn(1, 5, 32, 32, requires_grad=True)
        out = model(
            heightmap=hm,
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([5]),
            parent_blocks=torch.randint(0, tiny_config.block_vocab_size, (1, 32, 32, 32)),
        )
        out["block_type_logits"].sum().backward()
        assert hm.grad is not None

    def test_missing_parent_raises_value_error(self, tiny_config: OctreeConfig) -> None:
        model = create_leaf_model(tiny_config)
        with pytest.raises(ValueError, match="parent"):
            model(
                heightmap=torch.randn(1, 5, 32, 32),
                biome=torch.randint(0, 16, (1, 32, 32)),
                y_position=torch.tensor([5]),
            )

    def test_batch_size_one(self, tiny_config: OctreeConfig) -> None:
        model = create_leaf_model(tiny_config)
        out = self._forward(model, B=1)
        assert out["block_type_logits"].shape[0] == 1


# ===========================================================================
# 8. OctreeDataset
# ===========================================================================


class TestOctreeDataset:
    def test_loads_correct_length(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz", n=20)
        ds = OctreeDataset(tmp_path, split="train")
        assert len(ds) == 20

    def test_missing_cache_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OctreeDataset(tmp_path, split="train")

    def test_sample_has_required_keys(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz")
        ds = OctreeDataset(tmp_path, split="train")
        sample = ds[0]
        expected = {
            "labels32", "parent_labels32", "heightmap32", "biome32",
            "y_position", "level", "non_empty_children", "model_type",
        }
        assert set(sample.keys()) == expected

    def test_sample_shapes(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz")
        ds = OctreeDataset(tmp_path, split="train")
        sample = ds[0]
        assert sample["labels32"].shape == (32, 32, 32)
        assert sample["parent_labels32"].shape == (32, 32, 32)
        assert sample["heightmap32"].shape == (5, 32, 32)
        assert sample["biome32"].shape == (32, 32)

    def test_sample_dtypes(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz")
        ds = OctreeDataset(tmp_path, split="train")
        sample = ds[0]
        assert sample["labels32"].dtype == torch.int64
        assert sample["heightmap32"].dtype == torch.float32
        assert sample["biome32"].dtype == torch.int64

    def test_init_samples_have_model_type_init(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz")
        ds = OctreeDataset(tmp_path, split="train")
        # Find an L4 sample via raw arrays
        for i, lvl in enumerate(ds._levels):
            if int(lvl) == 4:
                mt = _model_type_for_level(4)
                assert mt == "init"
                break

    def test_level_indices_cover_all_levels(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz", n=20)
        ds = OctreeDataset(tmp_path, split="train")
        for lvl in range(5):
            assert lvl in ds._level_indices, f"Level {lvl} missing from _level_indices"

    def test_level_indices_total_matches_dataset_size(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz", n=20)
        ds = OctreeDataset(tmp_path, split="train")
        total = sum(len(v) for v in ds._level_indices.values())
        assert total == 20

    def test_val_split_loaded_separately(self, tmp_path: Path) -> None:
        _write_pair_cache(tmp_path / "train_octree_pairs.npz", n=20)
        _write_pair_cache(tmp_path / "val_octree_pairs.npz", n=5)
        train_ds = OctreeDataset(tmp_path, split="train")
        val_ds = OctreeDataset(tmp_path, split="val")
        assert len(train_ds) == 20
        assert len(val_ds) == 5


# ===========================================================================
# 9. collate_octree_batch
# ===========================================================================


class TestCollateOctreeBatch:
    def _make_sample(self, level: int, vocab: int = 32) -> Dict[str, Any]:
        mt = _model_type_for_level(level)
        return {
            "labels32": torch.randint(0, vocab, (32, 32, 32)),
            "parent_labels32": torch.randint(0, vocab, (32, 32, 32)),
            "heightmap32": torch.randn(5, 32, 32),
            "biome32": torch.randint(0, 16, (32, 32)),
            "y_position": torch.tensor(5, dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.long),
            "non_empty_children": torch.tensor(0, dtype=torch.long),
            "model_type": mt,
        }

    def test_homogeneous_batch_shape(self) -> None:
        samples = [self._make_sample(2) for _ in range(4)]
        batch = collate_octree_batch(samples)
        assert batch["model_type"] == "refine"
        assert batch["labels32"].shape == (4, 32, 32, 32)

    def test_majority_type_selected(self) -> None:
        """3 refine + 1 leaf → batch contains 3 refine samples."""
        samples = [self._make_sample(2)] * 3 + [self._make_sample(0)] * 1
        batch = collate_octree_batch(samples)
        assert batch["model_type"] == "refine"
        assert batch["labels32"].shape[0] == 3

    def test_empty_input_returns_empty_sentinel(self) -> None:
        batch = collate_octree_batch([])
        assert batch["model_type"] == "empty"

    def test_single_sample_batch(self) -> None:
        batch = collate_octree_batch([self._make_sample(4)])
        assert batch["model_type"] == "init"
        assert batch["labels32"].shape == (1, 32, 32, 32)

    def test_all_tensor_fields_are_stacked(self) -> None:
        samples = [self._make_sample(1) for _ in range(3)]
        batch = collate_octree_batch(samples)
        tensor_keys = [
            "labels32", "parent_labels32", "heightmap32", "biome32",
            "y_position", "level", "non_empty_children",
        ]
        for key in tensor_keys:
            assert isinstance(batch[key], torch.Tensor), f"'{key}' must be a tensor"

    def test_leaf_samples_batch_correctly(self) -> None:
        samples = [self._make_sample(0) for _ in range(2)]
        batch = collate_octree_batch(samples)
        assert batch["model_type"] == "leaf"
        assert batch["labels32"].shape == (2, 32, 32, 32)


# ===========================================================================
# 10. _bitmask_to_binary
# ===========================================================================


class TestBitmaskToBinary:
    def test_zero_bitmask_all_zeros(self) -> None:
        out = _bitmask_to_binary(torch.tensor([0], dtype=torch.long))
        assert out.shape == (1, 8)
        assert (out == 0).all()

    def test_255_bitmask_all_ones(self) -> None:
        out = _bitmask_to_binary(torch.tensor([255], dtype=torch.long))
        assert out.shape == (1, 8)
        assert (out == 1).all()

    @pytest.mark.parametrize("bit", range(8))
    def test_single_bit_extraction(self, bit: int) -> None:
        out = _bitmask_to_binary(torch.tensor([1 << bit], dtype=torch.long))
        assert out[0, bit] == 1.0, f"bit {bit} should be set"
        assert out[0].sum().item() == pytest.approx(1.0), f"only bit {bit} should be set"

    def test_batch_dimension(self) -> None:
        bm = torch.tensor([0b00000001, 0b11111111, 0b10101010], dtype=torch.long)
        out = _bitmask_to_binary(bm)
        assert out.shape == (3, 8)
        assert out[0, 0] == 1.0   # bit 0 of first value set
        assert out[0, 1] == 0.0   # bit 1 of first value not set
        assert (out[1] == 1).all()  # all bits of 255 set

    def test_output_dtype_is_float(self) -> None:
        out = _bitmask_to_binary(torch.tensor([42], dtype=torch.long))
        assert out.dtype == torch.float32


# ===========================================================================
# 11. OctreeLoss
# ===========================================================================


class TestOctreeLoss:
    def _preds(self, B: int = 2, V: int = 32, with_occ: bool = True) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "block_type_logits": torch.randn(B, V, 32, 32, 32),
        }
        if with_occ:
            out["occ_logits"] = torch.randn(B, 8)
        return out

    def _targets(self, B: int = 2, V: int = 32) -> Dict[str, torch.Tensor]:
        return {
            "target_blocks": torch.randint(0, V, (B, 32, 32, 32)),
            "occ_targets": torch.randint(0, 2, (B, 8)).float(),
        }

    def test_returns_expected_keys(self) -> None:
        ld = OctreeLoss()(self._preds(), self._targets(), "refine")
        assert "total_loss" in ld
        assert "block_loss" in ld
        assert "occ_loss" in ld

    @pytest.mark.parametrize(
        "model_type,with_occ",
        [("init", True), ("refine", True), ("leaf", False)],
    )
    def test_all_losses_finite(self, model_type: str, with_occ: bool) -> None:
        ld = OctreeLoss()(self._preds(with_occ=with_occ), self._targets(), model_type)
        assert torch.isfinite(ld["total_loss"])
        assert torch.isfinite(ld["block_loss"])

    def test_leaf_occ_loss_is_zero(self) -> None:
        preds = {"block_type_logits": torch.randn(2, 32, 32, 32, 32)}
        ld = OctreeLoss()(preds, self._targets(), "leaf")
        assert ld["occ_loss"].item() == pytest.approx(0.0)

    def test_occ_weight_zero_removes_occ_from_total(self) -> None:
        preds = self._preds()
        targets = self._targets()
        ld = OctreeLoss(occ_weight=0.0)(preds, targets, "refine")
        assert ld["total_loss"].item() == pytest.approx(ld["block_loss"].item(), rel=1e-4)

    def test_occ_weight_scales_occ_contribution(self) -> None:
        torch.manual_seed(0)
        preds = self._preds()
        targets = self._targets()
        ld1 = OctreeLoss(occ_weight=1.0)(preds, targets, "refine")
        ld5 = OctreeLoss(occ_weight=5.0)(preds, targets, "refine")
        # Higher weight → higher total loss (occ_loss > 0 in practice)
        occ = ld1["occ_loss"].item()
        if occ > 0:
            assert ld5["total_loss"].item() > ld1["total_loss"].item()

    def test_backward_computes_gradients(self) -> None:
        block_logits = torch.randn(2, 32, 32, 32, 32, requires_grad=True)
        occ_logits = torch.randn(2, 8, requires_grad=True)
        preds = {"block_type_logits": block_logits, "occ_logits": occ_logits}
        ld = OctreeLoss()(preds, self._targets(), "init")
        ld["total_loss"].backward()
        assert block_logits.grad is not None
        assert occ_logits.grad is not None

    def test_confident_correct_predictions_lower_block_loss(self) -> None:
        """Logits that strongly predict the correct class → lower block_loss."""
        V = 32
        targets = self._targets(B=1, V=V)
        target_ids = targets["target_blocks"]  # (1, 32, 32, 32)

        # Build near-perfect logits: correct class gets +100, others get -100
        good_logits = torch.full((1, V, 32, 32, 32), -100.0)
        good_logits.scatter_(1, target_ids.unsqueeze(1), 100.0)

        bad_logits = torch.randn(1, V, 32, 32, 32)

        loss_fn = OctreeLoss(occ_weight=0.0)
        targets_leaf = {"target_blocks": target_ids, "occ_targets": torch.zeros(1, 8)}
        good_loss = loss_fn({"block_type_logits": good_logits}, targets_leaf, "leaf")
        bad_loss = loss_fn({"block_type_logits": bad_logits}, targets_leaf, "leaf")

        assert good_loss["block_loss"].item() < bad_loss["block_loss"].item()


# ===========================================================================
# 12. compute_octree_metrics
# ===========================================================================


class TestComputeOctreeMetrics:
    def _make_inputs(
        self, B: int = 2, V: int = 32, with_occ: bool = True
    ) -> tuple[Dict, Dict]:
        preds: Dict[str, torch.Tensor] = {
            "block_type_logits": torch.randn(B, V, 32, 32, 32),
        }
        if with_occ:
            preds["occ_logits"] = torch.randn(B, 8)
        targets: Dict[str, torch.Tensor] = {
            "target_blocks": torch.randint(0, V, (B, 32, 32, 32)),
            "occ_targets": torch.randint(0, 2, (B, 8)).float(),
        }
        return preds, targets

    def test_refine_returns_occ_metrics(self) -> None:
        preds, targets = self._make_inputs()
        m = compute_octree_metrics(preds, targets, "refine")
        assert "occ_f1" in m
        assert "occ_precision" in m
        assert "occ_recall" in m

    def test_leaf_omits_occ_metrics(self) -> None:
        preds = {"block_type_logits": torch.randn(2, 32, 32, 32, 32)}
        targets = {"target_blocks": torch.randint(0, 32, (2, 32, 32, 32))}
        m = compute_octree_metrics(preds, targets, "leaf")
        assert "occ_f1" not in m

    def test_all_accuracy_metrics_present(self) -> None:
        preds, targets = self._make_inputs(with_occ=False)
        m = compute_octree_metrics(preds, targets, "leaf")
        assert "overall_accuracy" in m
        assert "air_accuracy" in m
        assert "block_accuracy" in m

    @pytest.mark.parametrize("key", ["overall_accuracy", "air_accuracy", "block_accuracy"])
    def test_accuracy_in_unit_interval(self, key: str) -> None:
        preds, targets = self._make_inputs()
        m = compute_octree_metrics(preds, targets, "init")
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0, 1]"

    def test_occ_f1_in_unit_interval(self) -> None:
        preds, targets = self._make_inputs()
        m = compute_octree_metrics(preds, targets, "init")
        assert 0.0 <= m["occ_f1"] <= 1.0

    def test_perfect_block_predictions_give_accuracy_one(self) -> None:
        B, V = 1, 16
        targets = torch.randint(0, V, (B, 32, 32, 32))
        # Build logits that argmax to the correct class everywhere
        logits = torch.full((B, V, 32, 32, 32), -100.0)
        logits.scatter_(1, targets.unsqueeze(1), 100.0)
        preds = {"block_type_logits": logits}
        m = compute_octree_metrics(preds, {"target_blocks": targets}, "leaf")
        assert m["overall_accuracy"] == pytest.approx(1.0)

    def test_all_air_target(self) -> None:
        """All-air target grid: air_accuracy should be valid."""
        B, V = 1, 8
        targets = torch.zeros(B, 32, 32, 32, dtype=torch.long)
        preds = {"block_type_logits": torch.randn(B, V, 32, 32, 32)}
        m = compute_octree_metrics(preds, {"target_blocks": targets}, "leaf")
        assert 0.0 <= m["air_accuracy"] <= 1.0

    def test_no_solid_blocks_target(self) -> None:
        """All-air target → block_accuracy defaults to 1.0 (no solid blocks to get wrong)."""
        targets = torch.zeros(1, 32, 32, 32, dtype=torch.long)
        preds = {"block_type_logits": torch.randn(1, 8, 32, 32, 32)}
        m = compute_octree_metrics(preds, {"target_blocks": targets}, "leaf")
        assert m["block_accuracy"] == pytest.approx(1.0)


# ===========================================================================
# 13. _prepare_targets
# ===========================================================================


class TestPrepareTargets:
    def test_target_blocks_equals_labels32(self) -> None:
        batch = _make_batch(B=2, level=2)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert "target_blocks" in targets
        assert torch.equal(targets["target_blocks"], batch["labels32"])

    def test_target_blocks_shape(self) -> None:
        batch = _make_batch(B=3, level=1)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert targets["target_blocks"].shape == (3, 32, 32, 32)

    def test_occ_targets_shape(self) -> None:
        batch = _make_batch(B=2, level=3)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert "occ_targets" in targets
        assert targets["occ_targets"].shape == (2, 8)

    def test_all_ones_bitmask_gives_all_one_occ_targets(self) -> None:
        batch = _make_batch(B=2, level=2)
        batch["non_empty_children"] = torch.tensor([0xFF, 0xFF], dtype=torch.long)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert (targets["occ_targets"] == 1.0).all()

    def test_zero_bitmask_gives_all_zero_occ_targets(self) -> None:
        batch = _make_batch(B=2, level=2)
        batch["non_empty_children"] = torch.tensor([0, 0], dtype=torch.long)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert (targets["occ_targets"] == 0.0).all()

    def test_occ_targets_float_dtype(self) -> None:
        batch = _make_batch(B=2, level=2)
        targets = _prepare_targets(batch, torch.device("cpu"))
        assert targets["occ_targets"].dtype == torch.float32
