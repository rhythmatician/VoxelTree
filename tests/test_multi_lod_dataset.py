"""Tests for train/multi_lod_dataset.py — pair generation, caching, dataset loading."""

from pathlib import Path

import numpy as np
import pytest
import torch

from train.multi_lod_dataset import (
    MultiLODDataset,
    collate_multi_lod_batch,
    create_lod_training_pairs,
    create_occupancy_from_blocks,
    generate_pairs_from_npz_files,
    save_pairs_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labels16(seed: int = 42) -> np.ndarray:
    """Random 16³ block IDs with a mix of air (0) and solid (1-99)."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 100, size=(16, 16, 16), dtype=np.int32)
    # Ensure ~30% air so min-solid filters pass
    labels[rng.rand(16, 16, 16) < 0.3] = 0
    return labels


def _make_chunk_kwargs(seed: int = 42) -> dict:
    """Return kwargs suitable for create_lod_training_pairs()."""
    rng = np.random.RandomState(seed)
    return {
        "labels16": _make_labels16(seed),
        "biome_patch": rng.randint(0, 50, (16, 16), dtype=np.int32),
        "heightmap_patch": rng.rand(16, 16).astype(np.float32) * 100,
        "y_index": 5,
        "heightmap_surface": rng.rand(16, 16).astype(np.float32) * 200,
        "heightmap_ocean_floor": rng.rand(16, 16).astype(np.float32) * 50,
    }


def _write_fake_npz(path: Path, seed: int = 42) -> None:
    """Write a single fake chunk NPZ with all required keys."""
    kw = _make_chunk_kwargs(seed)
    np.savez(
        path,
        labels16=kw["labels16"],
        biome_patch=kw["biome_patch"],
        heightmap_patch=kw["heightmap_patch"],
        y_index=np.int64(kw["y_index"]),
        heightmap_surface=kw["heightmap_surface"],
        heightmap_ocean_floor=kw["heightmap_ocean_floor"],
    )


# ===========================================================================
# create_occupancy_from_blocks
# ===========================================================================


class TestCreateOccupancy:
    def test_air_is_zero(self):
        blocks = np.array([0, 1, 0, 5, 0])
        occ = create_occupancy_from_blocks(blocks)
        np.testing.assert_array_equal(occ, [0, 1, 0, 1, 0])

    def test_custom_air_id(self):
        blocks = np.array([3, 3, 1, 0])
        occ = create_occupancy_from_blocks(blocks, air_id=3)
        np.testing.assert_array_equal(occ, [0, 0, 1, 1])

    def test_all_air(self):
        blocks = np.zeros((4, 4, 4), dtype=np.int32)
        occ = create_occupancy_from_blocks(blocks)
        assert occ.sum() == 0

    def test_all_solid(self):
        blocks = np.ones((4, 4, 4), dtype=np.int32)
        occ = create_occupancy_from_blocks(blocks)
        assert occ.sum() == 64


# ===========================================================================
# create_lod_training_pairs
# ===========================================================================


class TestCreateLodTrainingPairs:
    @pytest.fixture
    def pairs(self):
        return create_lod_training_pairs(**_make_chunk_kwargs())

    def test_returns_four_pairs(self, pairs):
        assert len(pairs) == 4

    def test_transitions_present(self, pairs):
        transitions = {p["lod_transition"] for p in pairs}
        assert transitions == {"init_to_lod4", "lod4to3", "lod3to2", "lod2to1"}

    def test_init_pair_has_zero_parent(self, pairs):
        init_pair = [p for p in pairs if p["lod_transition"] == "init_to_lod4"][0]
        np.testing.assert_array_equal(init_pair["parent_voxel"], 0.0)
        assert init_pair["parent_size"] == 0

    def test_target_shapes_are_16_cubed(self, pairs):
        for p in pairs:
            assert p["target_mask"].shape == (16, 16, 16), p["lod_transition"]
            assert p["target_types"].shape == (16, 16, 16), p["lod_transition"]

    def test_parent_voxel_shape(self, pairs):
        for p in pairs:
            assert p["parent_voxel"].shape == (1, 8, 8, 8), p["lod_transition"]

    def test_height_planes_shape(self, pairs):
        for p in pairs:
            assert p["height_planes"].shape == (5, 16, 16), p["lod_transition"]

    def test_biome_idx_shape(self, pairs):
        for p in pairs:
            assert p["biome_idx"].shape == (16, 16), p["lod_transition"]

    def test_lod_values(self, pairs):
        expected_lods = {"init_to_lod4": 4, "lod4to3": 4, "lod3to2": 3, "lod2to1": 2}
        for p in pairs:
            assert int(p["lod"]) == expected_lods[p["lod_transition"]]

    def test_target_mask_is_binary(self, pairs):
        for p in pairs:
            vals = np.unique(p["target_mask"])
            assert all(
                v in (0.0, 1.0) for v in vals
            ), f"{p['lod_transition']} has non-binary mask values: {vals}"


# ===========================================================================
# generate_pairs_from_npz_files
# ===========================================================================


class TestGeneratePairsFromNpzFiles:
    def test_generates_pairs_from_valid_npz(self, tmp_path):
        for i in range(3):
            _write_fake_npz(tmp_path / f"chunk_{i}.npz", seed=i)
        files = sorted(tmp_path.glob("*.npz"))
        pairs = generate_pairs_from_npz_files(files)
        # 3 chunks × 4 transitions = 12
        assert len(pairs) == 12

    def test_skips_npz_missing_heightmap_surface(self, tmp_path):
        path = tmp_path / "bad.npz"
        kw = _make_chunk_kwargs()
        np.savez(
            path,
            labels16=kw["labels16"],
            biome_patch=kw["biome_patch"],
            heightmap_patch=kw["heightmap_patch"],
            y_index=np.int64(kw["y_index"]),
            # no heightmap_surface or heightmap_ocean_floor
        )
        pairs = generate_pairs_from_npz_files([path])
        assert len(pairs) == 0

    def test_skips_mostly_air_chunks(self, tmp_path):
        path = tmp_path / "air.npz"
        kw = _make_chunk_kwargs()
        kw["labels16"] = np.zeros((16, 16, 16), dtype=np.int32)  # all air
        np.savez(
            path,
            labels16=kw["labels16"],
            biome_patch=kw["biome_patch"],
            heightmap_patch=kw["heightmap_patch"],
            y_index=np.int64(kw["y_index"]),
            heightmap_surface=kw["heightmap_surface"],
            heightmap_ocean_floor=kw["heightmap_ocean_floor"],
        )
        pairs = generate_pairs_from_npz_files([path], min_solid_fraction=0.01)
        assert len(pairs) == 0


# ===========================================================================
# save / load round-trip via MultiLODDataset
# ===========================================================================


class TestPairCacheRoundTrip:
    def test_save_and_load(self, tmp_path):
        """save_pairs_cache → MultiLODDataset should round-trip cleanly."""
        pairs = create_lod_training_pairs(**_make_chunk_kwargs(seed=0))
        # Duplicate to simulate multiple chunks
        all_pairs = pairs * 4  # 16 pairs total

        cache_path = tmp_path / "train_pairs_v2.npz"
        save_pairs_cache(all_pairs, cache_path)

        assert cache_path.exists()

        ds = MultiLODDataset(data_dir=tmp_path, split="train")
        assert len(ds) == 16

    def test_dataset_getitem_returns_tensors(self, tmp_path):
        pairs = create_lod_training_pairs(**_make_chunk_kwargs(seed=1))
        save_pairs_cache(pairs * 2, tmp_path / "train_pairs_v2.npz")

        ds = MultiLODDataset(data_dir=tmp_path, split="train")
        sample = ds[0]

        # Check types
        assert isinstance(sample["parent_voxel"], torch.Tensor)
        assert isinstance(sample["biome_patch"], torch.Tensor)
        assert isinstance(sample["height_planes"], torch.Tensor)
        assert isinstance(sample["target_mask"], torch.Tensor)
        assert isinstance(sample["target_types"], torch.Tensor)
        assert isinstance(sample["lod_transition"], str)

    def test_missing_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Pair cache not found"):
            MultiLODDataset(data_dir=tmp_path, split="train")

    def test_use_pair_cache_false_raises(self, tmp_path):
        with pytest.raises(ValueError, match="use_pair_cache=False is no longer supported"):
            MultiLODDataset(data_dir=tmp_path, split="train", use_pair_cache=False)


# ===========================================================================
# collate_multi_lod_batch
# ===========================================================================


class TestCollateBatch:
    def test_collate_stacks_tensors(self, tmp_path):
        pairs = create_lod_training_pairs(**_make_chunk_kwargs())
        save_pairs_cache(pairs * 2, tmp_path / "train_pairs_v2.npz")
        ds = MultiLODDataset(data_dir=tmp_path, split="train")

        samples = [ds[i] for i in range(4)]
        batch = collate_multi_lod_batch(samples)

        assert batch["parent_voxel"].shape[0] == 4
        assert batch["target_types"].shape[0] == 4
        assert isinstance(batch["lod_transition"], str)

    def test_collate_preserves_shapes(self, tmp_path):
        pairs = create_lod_training_pairs(**_make_chunk_kwargs())
        save_pairs_cache(pairs * 2, tmp_path / "train_pairs_v2.npz")
        ds = MultiLODDataset(data_dir=tmp_path, split="train")

        samples = [ds[0], ds[1]]
        batch = collate_multi_lod_batch(samples)

        assert batch["parent_voxel"].shape == (2, 1, 8, 8, 8)
        assert batch["height_planes"].shape == (2, 5, 16, 16)
        assert batch["biome_patch"].shape == (2, 16, 16)
