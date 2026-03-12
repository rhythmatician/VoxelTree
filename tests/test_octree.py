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

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from VoxelTree.scripts.build_octree_pairs import (
    build_section_index,
    child_coords_from_parent,
    extract_octant_and_upsample,
    parent_coords_and_octant,
)
from VoxelTree.train.octree_dataset import (
    OctreeDataset,
    _model_type_for_level,
    collate_octree_batch,
)
from VoxelTree.train.octree_models import (
    OccGateModule,
    OccupancyHead,
    OctreeConfig,
    UNet3D32,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)
from VoxelTree.train.train_octree import (
    OctreeLoss,
    _bitmask_to_binary,
    _prepare_targets,
    compute_octree_metrics,
)

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
        assert (
            out == octant
        ).all(), f"Octant {octant} (bin {octant:03b}) contains unexpected values: {np.unique(out)}"

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
    def test_default_uses_init_shootout_winner(self, tiny_config: OctreeConfig) -> None:
        model = create_init_model(tiny_config)
        assert model.init_architecture == "encoder2d_decoder3d"
        assert hasattr(model, "backbone_2d3d")

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

    def test_legacy_full3d_init_architecture_still_works(self, tiny_config: OctreeConfig) -> None:
        tiny_config.init_architecture = "full_3d_unet"
        model = create_init_model(tiny_config)
        out = model(
            heightmap=torch.randn(2, 5, 32, 32),
            biome=torch.randint(0, 16, (2, 32, 32)),
            y_position=torch.randint(0, 24, (2,)),
        )
        assert out["block_type_logits"].shape == (2, tiny_config.block_vocab_size, 32, 32, 32)
        assert out["occ_logits"].shape == (2, 8)


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

        out1 = model(
            heightmap=hm, biome=biome, y_position=y, level=torch.tensor([1]), parent_blocks=parent
        )
        out3 = model(
            heightmap=hm, biome=biome, y_position=y, level=torch.tensor([3]), parent_blocks=parent
        )

        assert not torch.allclose(
            out1["block_type_logits"], out3["block_type_logits"]
        ), "Level 1 and level 3 should produce different logits"

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

    def test_default_leaf_extra_bottleneck_depth_is_enabled(
        self, tiny_config: OctreeConfig
    ) -> None:
        model = create_leaf_model(tiny_config)
        assert model.unet.bottleneck_extra is not None
        assert len(model.unet.bottleneck_extra) == 1

    def test_leaf_extra_bottleneck_depth_can_be_disabled(self, tiny_config: OctreeConfig) -> None:
        tiny_config.leaf_bottleneck_extra_depth = 0
        model = create_leaf_model(tiny_config)
        assert model.unet.bottleneck_extra is None


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
            "labels32",
            "parent_labels32",
            "heightmap32",
            "biome32",
            "y_position",
            "level",
            "non_empty_children",
            "model_type",
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
            "labels32",
            "parent_labels32",
            "heightmap32",
            "biome32",
            "y_position",
            "level",
            "non_empty_children",
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
        assert out[0, 0] == 1.0  # bit 0 of first value set
        assert out[0, 1] == 0.0  # bit 1 of first value not set
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
    def _make_inputs(self, B: int = 2, V: int = 32, with_occ: bool = True) -> tuple[Dict, Dict]:
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


# ===========================================================================
# 14. OctreeLoss — focal loss for occupancy (OGN-inspired)
# ===========================================================================


class TestOctreeLossFocal:
    """Tests for focal-loss occupancy and warmup features added in Step 3."""

    def _preds(self, B: int = 2, V: int = 32) -> Dict[str, torch.Tensor]:
        return {
            "block_type_logits": torch.randn(B, V, 32, 32, 32),
            "occ_logits": torch.randn(B, 8),
        }

    def _targets(self, B: int = 2, V: int = 32) -> Dict[str, torch.Tensor]:
        return {
            "target_blocks": torch.randint(0, V, (B, 32, 32, 32)),
            "occ_targets": torch.randint(0, 2, (B, 8)).float(),
        }

    def test_focal_gamma_zero_matches_bce(self) -> None:
        """focal_gamma=0 should produce identical occ_loss as plain BCE."""
        torch.manual_seed(42)
        preds = self._preds()
        targets = self._targets()
        bce_loss = OctreeLoss(focal_gamma=0.0)(preds, targets, "refine")

        torch.manual_seed(42)
        preds2 = self._preds()
        targets2 = self._targets()
        bce_loss2 = OctreeLoss(focal_gamma=0.0)(preds2, targets2, "refine")

        assert bce_loss["occ_loss"].item() == pytest.approx(bce_loss2["occ_loss"].item(), rel=1e-5)

    def test_focal_gamma_positive_produces_finite_loss(self) -> None:
        ld = OctreeLoss(focal_gamma=2.0, focal_alpha=0.75)(self._preds(), self._targets(), "refine")
        assert torch.isfinite(ld["occ_loss"])
        assert torch.isfinite(ld["total_loss"])

    def test_focal_loss_backward(self) -> None:
        """Focal loss should allow gradient computation."""
        occ_logits = torch.randn(2, 8, requires_grad=True)
        preds = {
            "block_type_logits": torch.randn(2, 32, 32, 32, 32, requires_grad=True),
            "occ_logits": occ_logits,
        }
        ld = OctreeLoss(focal_gamma=2.0)(preds, self._targets(), "init")
        ld["total_loss"].backward()
        assert occ_logits.grad is not None

    def test_focal_reduces_easy_example_loss(self) -> None:
        """Focal loss on easy examples (high confidence) should be lower than BCE."""
        # Easy example: logits strongly match targets
        occ_targets = torch.ones(1, 8)  # all occupied
        easy_logits = torch.full((1, 8), 5.0)  # sigmoid(5) ≈ 0.993

        preds_easy = {
            "block_type_logits": torch.randn(1, 32, 32, 32, 32),
            "occ_logits": easy_logits,
        }
        targets = {
            "target_blocks": torch.randint(0, 32, (1, 32, 32, 32)),
            "occ_targets": occ_targets,
        }

        bce_occ = OctreeLoss(focal_gamma=0.0, occ_weight=1.0)(preds_easy, targets, "refine")[
            "occ_loss"
        ]
        focal_occ = OctreeLoss(focal_gamma=2.0, occ_weight=1.0)(preds_easy, targets, "refine")[
            "occ_loss"
        ]
        # Focal loss should be lower on easy examples
        assert focal_occ.item() < bce_occ.item()

    def test_focal_leaf_still_zero(self) -> None:
        """Leaf model_type should still have zero occ_loss even with focal params."""
        preds = {"block_type_logits": torch.randn(2, 32, 32, 32, 32)}
        ld = OctreeLoss(focal_gamma=2.0, focal_alpha=0.75)(preds, self._targets(), "leaf")
        assert ld["occ_loss"].item() == pytest.approx(0.0)

    def test_focal_alpha_affects_loss(self) -> None:
        """Different alpha values should produce different occ_loss."""
        torch.manual_seed(99)
        preds = self._preds()
        targets = self._targets()
        la = OctreeLoss(focal_gamma=2.0, focal_alpha=0.25)(preds, targets, "refine")
        lb = OctreeLoss(focal_gamma=2.0, focal_alpha=0.75)(preds, targets, "refine")
        # With different alpha, the occ_loss values should differ
        assert la["occ_loss"].item() != pytest.approx(lb["occ_loss"].item(), abs=1e-6)


# ===========================================================================
# 15. OccupancyHead — spatial octant pooling (OGN-inspired)
# ===========================================================================


class TestOccupancyHeadSpatial:
    """Tests for the spatially-aware OccupancyHead."""

    def test_output_shape(self) -> None:
        """Output should be [B, 8] for any valid bottleneck input."""
        head = OccupancyHead(in_channels=96)
        bottleneck = torch.randn(4, 96, 8, 8, 8)
        out = head(bottleneck)
        assert out.shape == (4, 8)

    def test_gradient_flows(self) -> None:
        head = OccupancyHead(in_channels=64)
        bottleneck = torch.randn(2, 64, 8, 8, 8, requires_grad=True)
        out = head(bottleneck)
        out.sum().backward()
        assert bottleneck.grad is not None

    def test_octant_sensitivity(self) -> None:
        """Each octant logit should depend on features in its spatial quadrant."""
        head = OccupancyHead(in_channels=16)
        head.eval()

        # Start with zeros everywhere
        bottleneck = torch.zeros(1, 16, 8, 8, 8)
        baseline = head(bottleneck).detach()

        # Perturb only octant 0 region (Y=0:4, Z=0:4, X=0:4)
        perturbed = bottleneck.clone()
        perturbed[:, :, 0:4, 0:4, 0:4] = 10.0
        after = head(perturbed).detach()

        # Octant 0 logit should change the most
        diffs = (after - baseline).abs().squeeze()
        assert diffs[0] > 0, "Octant 0 logit should change when its region is perturbed"
        # Octant 0 should be more affected than distant octant 7 (Y=4:8, Z=4:8, X=4:8)
        assert diffs[0] > diffs[7], "Octant 0 should be more affected than octant 7"

    def test_all_octants_responsive(self) -> None:
        """All 8 octant logits should respond to perturbations in their regions."""
        head = OccupancyHead(in_channels=16)
        head.eval()

        baseline_bn = torch.zeros(1, 16, 8, 8, 8)
        baseline = head(baseline_bn).detach()

        for octant in range(8):
            ox = octant & 1
            oz = (octant >> 1) & 1
            oy = (octant >> 2) & 1

            perturbed = baseline_bn.clone()
            perturbed[:, :, oy * 4 : (oy + 1) * 4, oz * 4 : (oz + 1) * 4, ox * 4 : (ox + 1) * 4] = (
                10.0
            )
            after = head(perturbed).detach()
            diff = (after - baseline).abs().squeeze()[octant]
            assert diff > 0, f"Octant {octant} logit should respond to its region"

    def test_batch_size_one(self) -> None:
        head = OccupancyHead(in_channels=128)
        out = head(torch.randn(1, 128, 8, 8, 8))
        assert out.shape == (1, 8)

    def test_no_gap_attribute(self) -> None:
        """Verify the old GAP-based design is gone."""
        head = OccupancyHead(in_channels=96)
        assert not hasattr(head, "gap"), "OccupancyHead should not have a 'gap' attribute"


# ── Step 4: Refine-path experiments ──────────────────────────────────


class TestParentContextAblation:
    """Test parent_context_mode='embed'/'zeros'/'disabled' on refine + leaf."""

    def _refine_fwd(self, mode: str) -> Dict[str, torch.Tensor]:
        from VoxelTree.train.octree_models import OctreeConfig, create_refine_model

        cfg = OctreeConfig(
            block_vocab_size=32,
            refine_channels=(8, 16, 32),
            parent_embed_dim=4,
            biome_embed_dim=4,
            y_embed_dim=4,
            level_embed_dim=4,
            parent_context_mode=mode,
        )
        model = create_refine_model(cfg)
        model.eval()
        B = 2
        hm = torch.randn(B, 5, 32, 32)
        biome = torch.randint(0, 16, (B, 32, 32))
        y = torch.randint(0, 24, (B,))
        level = torch.tensor([2, 2])
        parent = torch.randint(0, 32, (B, 32, 32, 32))
        with torch.no_grad():
            return model(
                heightmap=hm,
                biome=biome,
                y_position=y,
                level=level,
                parent_blocks=parent,
            )

    def _leaf_fwd(self, mode: str) -> Dict[str, torch.Tensor]:
        from VoxelTree.train.octree_models import OctreeConfig, create_leaf_model

        cfg = OctreeConfig(
            block_vocab_size=32,
            leaf_channels=(8, 16, 32),
            parent_embed_dim=4,
            biome_embed_dim=4,
            y_embed_dim=4,
            level_embed_dim=4,
            parent_context_mode=mode,
        )
        model = create_leaf_model(cfg)
        model.eval()
        B = 2
        hm = torch.randn(B, 5, 32, 32)
        biome = torch.randint(0, 16, (B, 32, 32))
        y = torch.randint(0, 24, (B,))
        parent = torch.randint(0, 32, (B, 32, 32, 32))
        with torch.no_grad():
            return model(
                heightmap=hm,
                biome=biome,
                y_position=y,
                parent_blocks=parent,
            )

    def test_refine_embed_mode(self) -> None:
        out = self._refine_fwd("embed")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (2, 8)

    def test_refine_zeros_mode(self) -> None:
        out = self._refine_fwd("zeros")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (2, 8)

    def test_refine_disabled_mode(self) -> None:
        out = self._refine_fwd("disabled")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (2, 8)

    def test_leaf_embed_mode(self) -> None:
        out = self._leaf_fwd("embed")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)
        assert "occ_logits" not in out

    def test_leaf_zeros_mode(self) -> None:
        out = self._leaf_fwd("zeros")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)

    def test_leaf_disabled_mode(self) -> None:
        out = self._leaf_fwd("disabled")
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)

    def test_disabled_has_fewer_params(self) -> None:
        """Disabled mode should have fewer parameters than embed mode."""
        from VoxelTree.train.octree_models import OctreeConfig, create_refine_model

        def _count(mode: str) -> int:
            cfg = OctreeConfig(
                block_vocab_size=32,
                refine_channels=(8, 16, 32),
                parent_embed_dim=8,
                biome_embed_dim=4,
                y_embed_dim=4,
                level_embed_dim=4,
                parent_context_mode=mode,
            )
            m = create_refine_model(cfg)
            return sum(p.numel() for p in m.parameters())

        embed_params = _count("embed")
        disabled_params = _count("disabled")
        assert (
            disabled_params < embed_params
        ), f"Disabled ({disabled_params}) should have fewer params than embed ({embed_params})"

    def test_zeros_has_same_arch_as_embed(self) -> None:
        """Zeros mode uses same U-Net input channels as embed, just zero-filled."""
        from VoxelTree.train.octree_models import OctreeConfig, create_refine_model

        cfg_embed = OctreeConfig(
            block_vocab_size=32,
            refine_channels=(8, 16, 32),
            parent_embed_dim=4,
            biome_embed_dim=4,
            y_embed_dim=4,
            level_embed_dim=4,
            parent_context_mode="embed",
        )
        cfg_zeros = OctreeConfig(
            block_vocab_size=32,
            refine_channels=(8, 16, 32),
            parent_embed_dim=4,
            biome_embed_dim=4,
            y_embed_dim=4,
            level_embed_dim=4,
            parent_context_mode="zeros",
        )
        m_embed = create_refine_model(cfg_embed)
        m_zeros = create_refine_model(cfg_zeros)
        # Same U-Net input channels (DoubleConv3d.block[0] → Conv3dBlock)
        in_ch_embed = m_embed.unet.enc1.block[0].conv.in_channels
        in_ch_zeros = m_zeros.unet.enc1.block[0].conv.in_channels
        assert in_ch_embed == in_ch_zeros

    def test_embed_raises_without_parent(self) -> None:
        """In embed mode, calling without parent should raise ValueError."""
        from VoxelTree.train.octree_models import OctreeConfig, create_refine_model

        cfg = OctreeConfig(
            block_vocab_size=32,
            refine_channels=(8, 16, 32),
            parent_embed_dim=4,
            biome_embed_dim=4,
            y_embed_dim=4,
            level_embed_dim=4,
            parent_context_mode="embed",
        )
        model = create_refine_model(cfg)
        with pytest.raises(ValueError, match="parent"):
            model(
                heightmap=torch.randn(1, 5, 32, 32),
                biome=torch.randint(0, 16, (1, 32, 32)),
                y_position=torch.randint(0, 24, (1,)),
                level=torch.tensor([2]),
            )


class TestPerLevelOccWeight:
    """Test per-level occupancy weight overrides in OctreeLoss."""

    def _make_loss_inputs(self) -> tuple:
        """Return (predictions, targets) for a refine-type batch."""
        preds = {
            "block_type_logits": torch.randn(2, 32, 32, 32, 32),
            "occ_logits": torch.randn(2, 8),
        }
        targets = {
            "target_blocks": torch.randint(0, 32, (2, 32, 32, 32)),
            "occ_targets": torch.rand(2, 8).round(),
        }
        return preds, targets

    def test_no_level_occ_weights_uses_global(self) -> None:
        """Without level_occ_weights, loss uses global occ_weight."""
        loss_fn = OctreeLoss(occ_weight=2.0)
        preds, targets = self._make_loss_inputs()
        result_a = loss_fn(preds, targets, "refine", level=3)
        result_b = loss_fn(preds, targets, "refine", level=None)
        # Both should produce identical total_loss
        assert torch.allclose(result_a["total_loss"], result_b["total_loss"])

    def test_level_occ_weights_override(self) -> None:
        """Per-level weights should override global occ_weight."""
        loss_fn = OctreeLoss(occ_weight=1.0, level_occ_weights={3: 5.0, 2: 0.1})
        preds, targets = self._make_loss_inputs()

        result_3 = loss_fn(preds, targets, "refine", level=3)
        result_2 = loss_fn(preds, targets, "refine", level=2)

        # Same block_loss, different total due to different occ weighting
        assert torch.allclose(result_3["block_loss"], result_2["block_loss"])
        # L3 should have higher total_loss because occ_weight=5.0 vs 0.1
        assert result_3["total_loss"] > result_2["total_loss"]

    def test_unspecified_level_falls_back(self) -> None:
        """Level not in dict should fall back to global occ_weight."""
        loss_fn = OctreeLoss(occ_weight=1.0, level_occ_weights={3: 10.0})
        preds, targets = self._make_loss_inputs()

        result_1 = loss_fn(preds, targets, "refine", level=1)
        result_no = loss_fn(preds, targets, "refine", level=None)
        # Both use global occ_weight=1.0
        assert torch.allclose(result_1["total_loss"], result_no["total_loss"])

    def test_level_occ_weights_init(self) -> None:
        """Init model (L4) should respect level_occ_weights."""
        loss_fn = OctreeLoss(occ_weight=1.0, level_occ_weights={4: 0.0})
        preds, targets = self._make_loss_inputs()

        result = loss_fn(preds, targets, "init", level=4)
        # With occ_weight=0 for L4, total_loss == block_loss
        assert torch.allclose(result["total_loss"], result["block_loss"])


class TestParentEmbedDimCLI:
    """Test that parent_embed_dim is properly wired through config."""

    def test_config_stores_parent_embed_dim(self) -> None:
        from VoxelTree.train.octree_models import OctreeConfig

        cfg = OctreeConfig(parent_embed_dim=64)
        assert cfg.parent_embed_dim == 64

    def test_config_stores_parent_context_mode(self) -> None:
        from VoxelTree.train.octree_models import OctreeConfig

        cfg = OctreeConfig(parent_context_mode="zeros")
        assert cfg.parent_context_mode == "zeros"

    def test_refine_model_uses_config_dim(self) -> None:
        """Refine model U-Net input channels should reflect parent_embed_dim."""
        from VoxelTree.train.octree_models import OctreeConfig, create_refine_model

        for dim in [4, 8, 32]:
            cfg = OctreeConfig(
                block_vocab_size=32,
                refine_channels=(8, 16, 32),
                parent_embed_dim=dim,
                biome_embed_dim=4,
                y_embed_dim=4,
                level_embed_dim=4,
                parent_context_mode="embed",
            )
            model = create_refine_model(cfg)
            expected_in = dim + 5 + 4 + 4 + 4  # parent + height + biome + y + level
            actual_in = model.unet.enc1.block[0].conv.in_channels
            assert (
                actual_in == expected_in
            ), f"dim={dim}: expected {expected_in} input channels, got {actual_in}"


# ===========================================================================
# Step 8: OGN-inspired targeted model improvements
# ===========================================================================


class TestBottleneckExtraDepth:
    """Tests for the bottleneck_extra_depth config option."""

    def test_default_has_no_extra(self) -> None:
        """Default config should have no extra bottleneck blocks."""
        unet = UNet3D32(in_channels=5, channels=(8, 16, 32))
        assert unet.bottleneck_extra is None

    def test_extra_depth_adds_blocks(self) -> None:
        """Setting extra depth > 0 should add DoubleConv3d blocks."""
        unet = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=2)
        assert unet.bottleneck_extra is not None
        assert len(unet.bottleneck_extra) == 2

    def test_extra_depth_same_output_shape(self) -> None:
        """Extra bottleneck blocks should not change output shapes."""
        x = torch.randn(2, 5, 32, 32, 32)
        unet_0 = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=0)
        unet_2 = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=2)

        f0, b0 = unet_0(x)
        f2, b2 = unet_2(x)
        assert f0.shape == f2.shape == (2, 8, 32, 32, 32)
        assert b0.shape == b2.shape == (2, 32, 8, 8, 8)

    def test_extra_depth_increases_params(self) -> None:
        """More bottleneck depth should mean more parameters."""
        unet_0 = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=0)
        unet_1 = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=1)
        n0 = sum(p.numel() for p in unet_0.parameters())
        n1 = sum(p.numel() for p in unet_1.parameters())
        assert n1 > n0

    def test_gradient_flows_through_extra(self) -> None:
        """Gradients should flow through extra bottleneck blocks."""
        unet = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=1)
        x = torch.randn(1, 5, 32, 32, 32)
        f, b = unet(x)
        (f.sum() + b.sum()).backward()
        # Check that extra block got gradients
        for p in unet.bottleneck_extra.parameters():
            assert p.grad is not None, "Extra bottleneck params should receive gradients"

    def test_encode_decode_matches_forward(self) -> None:
        """encode+decode should produce same result as forward."""
        unet = UNet3D32(in_channels=5, channels=(8, 16, 32), bottleneck_extra_depth=1)
        unet.eval()
        x = torch.randn(1, 5, 32, 32, 32)

        with torch.no_grad():
            f_fwd, b_fwd = unet(x)
            e1, e2, bn = unet.encode(x)
            f_split = unet.decode(e1, e2, bn)

        assert torch.allclose(f_fwd, f_split, atol=1e-6)
        assert torch.allclose(b_fwd, bn, atol=1e-6)

    def test_init_model_with_extra_depth(self, tiny_config: OctreeConfig) -> None:
        """Init model should work with bottleneck_extra_depth."""
        tiny_config.bottleneck_extra_depth = 1
        model = create_init_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
        )
        assert "block_type_logits" in out
        assert "occ_logits" in out
        assert out["block_type_logits"].shape == (1, 32, 32, 32, 32)

    def test_refine_model_with_extra_depth(self, tiny_config: OctreeConfig) -> None:
        """Refine model should work with bottleneck_extra_depth."""
        tiny_config.bottleneck_extra_depth = 2
        model = create_refine_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
            level=torch.tensor([2]),
            parent_blocks=torch.randint(0, 32, (1, 32, 32, 32)),
        )
        assert out["block_type_logits"].shape == (1, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (1, 8)


class TestParentRefineConv:
    """Tests for the parent_refine_conv config option."""

    def test_default_no_refine_conv(self, tiny_config: OctreeConfig) -> None:
        """Default config should not have parent refine conv."""
        assert not tiny_config.parent_refine_conv
        model = create_refine_model(tiny_config)
        assert model.parent_refine_conv_layer is None

    def test_refine_conv_adds_layer(self, tiny_config: OctreeConfig) -> None:
        """Setting parent_refine_conv=True should add a Conv3dBlock."""
        tiny_config.parent_refine_conv = True
        model = create_refine_model(tiny_config)
        assert model.parent_refine_conv_layer is not None
        # Should be a Conv3dBlock with parent_embed_dim in/out
        assert model.parent_refine_conv_layer.conv.in_channels == tiny_config.parent_embed_dim
        assert model.parent_refine_conv_layer.conv.out_channels == tiny_config.parent_embed_dim

    def test_refine_conv_forward(self, tiny_config: OctreeConfig) -> None:
        """Model should produce valid outputs with parent_refine_conv=True."""
        tiny_config.parent_refine_conv = True
        model = create_refine_model(tiny_config)
        out = model(
            heightmap=torch.randn(2, 5, 32, 32),
            biome=torch.randint(0, 16, (2, 32, 32)),
            y_position=torch.tensor([0, 1]),
            level=torch.tensor([2, 3]),
            parent_blocks=torch.randint(0, 32, (2, 32, 32, 32)),
        )
        assert out["block_type_logits"].shape == (2, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (2, 8)

    def test_refine_conv_gradient(self, tiny_config: OctreeConfig) -> None:
        """Gradients should flow through the parent refine conv."""
        tiny_config.parent_refine_conv = True
        model = create_refine_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
            level=torch.tensor([2]),
            parent_blocks=torch.randint(0, 32, (1, 32, 32, 32)),
        )
        out["block_type_logits"].sum().backward()
        for p in model.parent_refine_conv_layer.parameters():
            assert p.grad is not None

    def test_leaf_model_refine_conv(self, tiny_config: OctreeConfig) -> None:
        """Leaf model should also support parent_refine_conv."""
        tiny_config.parent_refine_conv = True
        model = create_leaf_model(tiny_config)
        assert model.parent_refine_conv_layer is not None
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
            parent_blocks=torch.randint(0, 32, (1, 32, 32, 32)),
        )
        assert "block_type_logits" in out

    def test_refine_conv_disabled_mode_no_layer(self, tiny_config: OctreeConfig) -> None:
        """parent_refine_conv with disabled parent mode should not add layer."""
        tiny_config.parent_refine_conv = True
        tiny_config.parent_context_mode = "disabled"
        model = create_refine_model(tiny_config)
        assert model.parent_refine_conv_layer is None


class TestOccGateModule:
    """Tests for the OccGateModule occupancy-gated bottleneck modulation."""

    def test_output_shapes(self) -> None:
        """OccGateModule should return (gated_bottleneck, occ_logits)."""
        gate = OccGateModule(in_channels=32)
        bn = torch.randn(2, 32, 8, 8, 8)
        gated, occ = gate(bn)
        assert gated.shape == (2, 32, 8, 8, 8)
        assert occ.shape == (2, 8)

    def test_occ_logits_match_standalone_head(self) -> None:
        """OccGateModule.occ_head should produce same logits as standalone."""
        gate = OccGateModule(in_channels=16)
        gate.eval()
        bn = torch.randn(1, 16, 8, 8, 8)
        with torch.no_grad():
            _, occ_from_gate = gate(bn)
            occ_standalone = gate.occ_head(bn)
        assert torch.allclose(occ_from_gate, occ_standalone)

    def test_gating_attenuates_empty_octants(self) -> None:
        """Octants predicted as empty should have attenuated features."""
        gate = OccGateModule(in_channels=16)
        gate.eval()

        # Force occ_head to predict specific pattern by using extreme values
        bn = torch.ones(1, 16, 8, 8, 8)
        with torch.no_grad():
            gated, occ = gate(bn)
            # Sigmoid < 0.5 means octant predicted empty → features attenuated
            for i in range(8):
                y_idx = (i >> 2) & 1
                z_idx = (i >> 1) & 1
                x_idx = i & 1
                octant_gate = torch.sigmoid(occ[:, i]).item()
                octant_features = gated[
                    :,
                    :,
                    y_idx * 4 : (y_idx + 1) * 4,
                    z_idx * 4 : (z_idx + 1) * 4,
                    x_idx * 4 : (x_idx + 1) * 4,
                ]
                orig_features = bn[
                    :,
                    :,
                    y_idx * 4 : (y_idx + 1) * 4,
                    z_idx * 4 : (z_idx + 1) * 4,
                    x_idx * 4 : (x_idx + 1) * 4,
                ]
                # Gated features should be original * gate value
                expected = orig_features * octant_gate
                assert torch.allclose(octant_features, expected, atol=1e-5)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through both the gate and bottleneck."""
        gate = OccGateModule(in_channels=16)
        bn = torch.randn(1, 16, 8, 8, 8, requires_grad=True)
        gated, occ = gate(bn)
        (gated.sum() + occ.sum()).backward()
        assert bn.grad is not None

    def test_init_model_with_occ_gate(self, tiny_config: OctreeConfig) -> None:
        """Init model should work with use_occ_gate=True."""
        tiny_config.use_occ_gate = True
        model = create_init_model(tiny_config)
        assert hasattr(model, "occ_gate")
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
        )
        assert out["block_type_logits"].shape == (1, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (1, 8)

    def test_refine_model_with_occ_gate(self, tiny_config: OctreeConfig) -> None:
        """Refine model should work with use_occ_gate=True."""
        tiny_config.use_occ_gate = True
        model = create_refine_model(tiny_config)
        assert hasattr(model, "occ_gate")
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
            level=torch.tensor([2]),
            parent_blocks=torch.randint(0, 32, (1, 32, 32, 32)),
        )
        assert out["block_type_logits"].shape == (1, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (1, 8)

    def test_occ_gate_backward(self, tiny_config: OctreeConfig) -> None:
        """Full model backward should work with occ_gate enabled."""
        tiny_config.use_occ_gate = True
        model = create_init_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
        )
        loss = out["block_type_logits"].sum() + out["occ_logits"].sum()
        loss.backward()
        # occ_gate params should have gradients
        for p in model.occ_gate.parameters():
            assert p.grad is not None


class TestStep8Combined:
    """Tests for combinations of Step 8 features."""

    def test_all_features_together(self, tiny_config: OctreeConfig) -> None:
        """All Step 8 features enabled simultaneously should work."""
        tiny_config.bottleneck_extra_depth = 1
        tiny_config.parent_refine_conv = True
        tiny_config.use_occ_gate = True
        model = create_refine_model(tiny_config)
        out = model(
            heightmap=torch.randn(1, 5, 32, 32),
            biome=torch.randint(0, 16, (1, 32, 32)),
            y_position=torch.tensor([0]),
            level=torch.tensor([2]),
            parent_blocks=torch.randint(0, 32, (1, 32, 32, 32)),
        )
        assert out["block_type_logits"].shape == (1, 32, 32, 32, 32)
        assert out["occ_logits"].shape == (1, 8)
        # Backward should work
        (out["block_type_logits"].sum() + out["occ_logits"].sum()).backward()

    def test_defaults_unchanged(self) -> None:
        """Default OctreeConfig should preserve legacy Step 8 defaults."""
        cfg = OctreeConfig()
        assert cfg.bottleneck_extra_depth == 0
        assert cfg.parent_refine_conv is False
        assert cfg.use_occ_gate is False
        assert cfg.init_architecture == "encoder2d_decoder3d"
        assert cfg.leaf_bottleneck_extra_depth == 1

    def test_param_count_increases(self, tiny_config: OctreeConfig) -> None:
        """Enabling Step 8 features should increase parameter count."""
        base = create_refine_model(tiny_config)
        n_base = sum(p.numel() for p in base.parameters())

        tiny_config.bottleneck_extra_depth = 1
        tiny_config.parent_refine_conv = True
        tiny_config.use_occ_gate = True
        enhanced = create_refine_model(tiny_config)
        n_enhanced = sum(p.numel() for p in enhanced.parameters())

        assert (
            n_enhanced > n_base
        ), f"Enhanced ({n_enhanced}) should have more params than base ({n_base})"

    def test_unet_encode_decode_api(self) -> None:
        """UNet3D32.encode() and decode() should be usable independently."""
        unet = UNet3D32(in_channels=5, channels=(8, 16, 32))
        x = torch.randn(1, 5, 32, 32, 32)
        e1, e2, bn = unet.encode(x)
        assert e1.shape == (1, 8, 32, 32, 32)
        assert e2.shape == (1, 16, 16, 16, 16)
        assert bn.shape == (1, 32, 8, 8, 8)
        features = unet.decode(e1, e2, bn)
        assert features.shape == (1, 8, 32, 32, 32)
