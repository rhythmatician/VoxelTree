"""Tests for multi-level Voxy ground truth extraction and target building.

Validates:
  - extract_section_subcube coordinate math at all 5 LOD levels
  - build_multilevel_voxy_targets is_leaf / child_mask / label logic
  - SparseOctreeDataset loads the multi-level NPZ format
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (
    extract_section_subcube,
)
from voxel_tree.tasks.sparse_octree.sparse_octree_targets import (
    build_multilevel_voxy_targets,
    build_sparse_octree_targets,
)
from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
    SparseOctreeDataset,
    sparse_octree_collate,
)
from voxel_tree.utils.coords import section_to_world_section


# ---------------------------------------------------------------------------
# extract_section_subcube tests
# ---------------------------------------------------------------------------


def _make_labeled_grid() -> np.ndarray:
    """Create a 32³ grid where each voxel stores a unique hash of (y, z, x).

    This lets us verify that the correct sub-region is extracted.
    """
    grid = np.zeros((32, 32, 32), dtype=np.int32)
    for y in range(32):
        for z in range(32):
            for x in range(32):
                grid[y, z, x] = y * 10000 + z * 100 + x
    return grid


class TestExtractSectionSubcube:
    """Validate coordinate mapping from section coords to Voxy sub-cubes."""

    def test_level0_even_section(self) -> None:
        """At L0, ws = s >> 1.  Even s → first half (voxels 0..15)."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=4, sy=4, sz=4, level=0)
        assert sub.shape == (16, 16, 16)
        # sx=4, ws_x = 4>>1 = 2, ls = 4 - (2<<1) = 0, voxel_start = 0
        # vy, vz, vx all start at 0
        np.testing.assert_array_equal(sub, grid[0:16, 0:16, 0:16])

    def test_level0_odd_section(self) -> None:
        """At L0, odd s → second half (voxels 16..31)."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=5, sy=5, sz=5, level=0)
        assert sub.shape == (16, 16, 16)
        # sx=5, ws_x = 5>>1 = 2, ls = 5 - 4 = 1, voxel_start = 16
        np.testing.assert_array_equal(sub, grid[16:32, 16:32, 16:32])

    def test_level4_single_voxel(self) -> None:
        """At L4, each section maps to exactly 1 voxel (16>>4 = 1)."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=5, sy=3, sz=7, level=4)
        assert sub.shape == (1, 1, 1)
        # sx=5: ws=5>>5=0, ls=5, vx=5*1=5
        # sy=3: ws=3>>5=0, ls=3, vy=3
        # sz=7: ws=7>>5=0, ls=7, vz=7
        assert sub[0, 0, 0] == grid[3, 7, 5]

    def test_level3_two_cubed(self) -> None:
        """At L3, each section maps to 2³ voxels (16>>3 = 2)."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=2, sy=1, sz=3, level=3)
        assert sub.shape == (2, 2, 2)
        # sx=2: ws=2>>4=0, ls=2, n=2, vx=2*2=4
        # sy=1: ws=1>>4=0, ls=1, vy=1*2=2
        # sz=3: ws=3>>4=0, ls=3, vz=3*2=6
        np.testing.assert_array_equal(sub, grid[2:4, 6:8, 4:6])

    def test_level2_four_cubed(self) -> None:
        """At L2, each section maps to 4³ voxels."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=1, sy=0, sz=2, level=2)
        assert sub.shape == (4, 4, 4)
        # sx=1: ws=1>>3=0, ls=1, n=4, vx=1*4=4
        # sy=0: ws=0>>3=0, ls=0, vy=0
        # sz=2: ws=2>>3=0, ls=2, vz=2*4=8
        np.testing.assert_array_equal(sub, grid[0:4, 8:12, 4:8])

    def test_level1_eight_cubed(self) -> None:
        """At L1, each section maps to 8³ voxels."""
        grid = _make_labeled_grid()
        sub = extract_section_subcube(grid, sx=0, sy=1, sz=0, level=1)
        assert sub.shape == (8, 8, 8)
        # sx=0: ws=0>>2=0, ls=0, n=8, vx=0
        # sy=1: ws=1>>2=0, ls=1, vy=1*8=8
        # sz=0: ws=0>>2=0, ls=0, vz=0
        np.testing.assert_array_equal(sub, grid[8:16, 0:8, 0:8])

    def test_negative_section_coords(self) -> None:
        """Negative section coords should work via arithmetic right-shift."""
        grid = _make_labeled_grid()
        # sx=-1: ws = -1>>1 = -1, ls = -1 - (-1<<1) = -1 - (-2) = 1, vx = 16
        sub = extract_section_subcube(grid, sx=-1, sy=-1, sz=-1, level=0)
        assert sub.shape == (16, 16, 16)
        np.testing.assert_array_equal(sub, grid[16:32, 16:32, 16:32])

    def test_section_to_ws_to_subcube_roundtrip(self) -> None:
        """Verify that section_to_world_section + subcube extraction is consistent."""
        grid = _make_labeled_grid()
        sx, sy, sz = 10, -3, 15
        for level in range(5):
            ws_x = section_to_world_section(sx, level)
            ws_y = section_to_world_section(sy, level)
            ws_z = section_to_world_section(sz, level)
            sub = extract_section_subcube(grid, sx, sy, sz, level)
            n = 16 >> level
            assert sub.shape == (n, n, n), f"L{level}: expected ({n},{n},{n})"


# ---------------------------------------------------------------------------
# build_multilevel_voxy_targets tests
# ---------------------------------------------------------------------------


class TestBuildMultilevelVoxyTargets:
    """Validate the multi-level target builder."""

    def test_all_air_yields_all_leaf(self) -> None:
        """Homogeneous air at all levels → every node is a leaf."""
        grids = {
            4: np.zeros((1, 1, 1), dtype=np.int32),
            3: np.zeros((2, 2, 2), dtype=np.int32),
            2: np.zeros((4, 4, 4), dtype=np.int32),
            1: np.zeros((8, 8, 8), dtype=np.int32),
            0: np.zeros((16, 16, 16), dtype=np.int32),
        }
        result = build_multilevel_voxy_targets(grids)
        for level in range(5):
            assert level in result
            assert result[level].is_leaf.all(), f"L{level} should be all-leaf"
            assert (result[level].child_mask == 0).all(), f"L{level} child_mask should be 0"
            assert (result[level].labels == 0).all(), f"L{level} labels should be 0 (air)"

    def test_uniform_stone_yields_all_leaf(self) -> None:
        """All levels filled with same stone ID → every node is leaf."""
        stone_id = 42
        grids = {}
        for lv in range(5):
            n = 16 >> lv
            grids[lv] = np.full((n, n, n), stone_id, dtype=np.int32)
        result = build_multilevel_voxy_targets(grids)
        for level in range(5):
            assert result[level].is_leaf.all()
            assert (result[level].labels == stone_id).all()

    def test_split_at_root_when_children_differ(self) -> None:
        """If L3 children differ, L4 node should be a split."""
        grids = {
            4: np.array([[[1]]], dtype=np.int32),  # root = stone
            3: np.zeros((2, 2, 2), dtype=np.int32),  # all air
        }
        # Set one child to non-air
        grids[3][0, 0, 0] = 5

        result = build_multilevel_voxy_targets(grids)
        # L4 root: children differ → split
        assert not result[4].is_leaf[0, 0, 0]
        assert result[4].labels[0, 0, 0] == -1  # split_label
        # child_mask bit 0 (dx=0,dz=0,dy=0) should be set
        assert result[4].child_mask[0, 0, 0] & 1 == 1

        # L3: finest level → all leaf
        assert result[3].is_leaf.all()

    def test_leaf_when_all_children_same(self) -> None:
        """If all 8 L3 children have the same label, L4 is a leaf."""
        grids = {
            4: np.array([[[7]]], dtype=np.int32),
            3: np.full((2, 2, 2), 7, dtype=np.int32),
        }
        result = build_multilevel_voxy_targets(grids)
        assert result[4].is_leaf[0, 0, 0]
        assert result[4].labels[0, 0, 0] == 7
        assert result[4].child_mask[0, 0, 0] == 0

    def test_child_mask_bits_correct(self) -> None:
        """Verify child_mask bit encoding: bit = dx | (dz<<1) | (dy<<2)."""
        grids = {
            4: np.array([[[1]]], dtype=np.int32),
            3: np.zeros((2, 2, 2), dtype=np.int32),  # all air
        }
        # Set specific children to non-air:
        # (dy=0, dz=0, dx=1) → bit 1 (=dx) → mask bit 1
        grids[3][0, 0, 1] = 10
        # (dy=1, dz=1, dx=0) → bit = 0 | (1<<1) | (1<<2) = 0+2+4 = 6 → mask bit 6
        grids[3][1, 1, 0] = 20

        result = build_multilevel_voxy_targets(grids)
        mask = int(result[4].child_mask[0, 0, 0])
        assert mask & (1 << 1), "bit 1 should be set (dx=1, dz=0, dy=0)"
        assert mask & (1 << 6), "bit 6 should be set (dx=0, dz=1, dy=1)"
        # Other bits should NOT be set (air children)
        assert not (mask & (1 << 0)), "bit 0 should be clear"
        assert not (mask & (1 << 2)), "bit 2 should be clear"

    def test_matches_recursive_for_uniform_fine_data(self) -> None:
        """When all levels are derived from a uniform L0 grid, multi-level
        targets should match the recursive single-grid builder."""
        rng = np.random.default_rng(42)
        L0 = rng.integers(0, 10, size=(16, 16, 16), dtype=np.int32)

        # Build multi-level grids by subsampling L0 (majority vote).
        # This simulates what Voxy would store.
        grids = {0: L0}
        current = L0
        for lv in range(1, 5):
            S = current.shape[0] // 2
            coarser = np.zeros((S, S, S), dtype=np.int32)
            for y in range(S):
                for z in range(S):
                    for x in range(S):
                        block = current[y * 2 : y * 2 + 2, z * 2 : z * 2 + 2, x * 2 : x * 2 + 2]
                        vals, counts = np.unique(block, return_counts=True)
                        coarser[y, z, x] = vals[counts.argmax()]
            grids[lv] = coarser
            current = coarser

        multi = build_multilevel_voxy_targets(grids, air_id=0, split_label=-1)
        single = build_sparse_octree_targets(L0, air_id=0, split_label=-1)

        # At L0, both should be identical (all leaves).
        np.testing.assert_array_equal(multi[0].labels, single[0].labels)
        np.testing.assert_array_equal(multi[0].is_leaf, single[0].is_leaf)

        # At coarser levels, is_leaf should match when the downsampling
        # is consistent (majority vote = recursive subdivision check).
        for lv in range(1, 5):
            np.testing.assert_array_equal(
                multi[lv].is_leaf, single[lv].is_leaf, err_msg=f"is_leaf mismatch at L{lv}"
            )

    def test_partial_levels_only_include_available(self) -> None:
        """If only L4 and L3 are provided, result should only have those levels."""
        grids = {
            4: np.array([[[5]]], dtype=np.int32),
            3: np.full((2, 2, 2), 5, dtype=np.int32),
        }
        result = build_multilevel_voxy_targets(grids)
        assert set(result.keys()) == {4, 3}

    def test_finest_level_nodes_are_always_leaf(self) -> None:
        """Nodes at the finest available level should always be leaves."""
        # Finest = L2, skip L1 and L0
        grids = {
            4: np.array([[[1]]], dtype=np.int32),
            3: np.full((2, 2, 2), 1, dtype=np.int32),
            2: np.full((4, 4, 4), 1, dtype=np.int32),
        }
        # Make L2 non-uniform so L3 has splits
        grids[2][0, 0, 0] = 99

        result = build_multilevel_voxy_targets(grids)
        # L2 is finest → all leaf
        assert result[2].is_leaf.all()
        # L3 should detect the mix from L2 and split at least one node
        assert not result[3].is_leaf.all()


# ---------------------------------------------------------------------------
# Multi-level NPZ dataset tests
# ---------------------------------------------------------------------------


class TestMultilevelDataset:
    """Verify SparseOctreeDataset loads multi-level NPZ format."""

    @staticmethod
    def _make_multilevel_npz(tmpdir: str, n: int = 4) -> Path:
        """Create a minimal multi-level NPZ file."""
        npz_path = Path(tmpdir) / "multilevel.npz"
        rng = np.random.default_rng(123)
        save_dict = {
            "noise_3d": rng.standard_normal((n, 15, 4, 2, 4)).astype(np.float32),
            "biome_ids": rng.integers(0, 10, (n, 4, 2, 4)).astype(np.int32),
            "heightmap5": rng.standard_normal((n, 5, 4, 4)).astype(np.float32),
            "block_y_min": np.arange(n, dtype=np.int32) * 16,
            "finest_level": np.zeros(n, dtype=np.int32),  # all levels available
        }
        for lv in range(5):
            side = 16 >> lv
            save_dict[f"labels_L{lv}"] = rng.integers(0, 100, (n, side, side, side)).astype(
                np.int32
            )
        np.savez_compressed(npz_path, **save_dict)
        return npz_path

    def test_detects_multilevel_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir)
            ds = SparseOctreeDataset(npz_path, cache_targets=False)
            assert len(ds.level_grids) == 5
            assert ds.finest_level is not None

    def test_sample_count_correct(self) -> None:
        n = 6
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir, n=n)
            ds = SparseOctreeDataset(npz_path, cache_targets=False)
            assert len(ds) == n

    def test_level_grid_shapes(self) -> None:
        n = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir, n=n)
            ds = SparseOctreeDataset(npz_path, cache_targets=False)
            for lv in range(5):
                side = 16 >> lv
                assert ds.level_grids[lv].shape == (n, side, side, side)

    def test_getitem_returns_targets_at_all_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir)
            ds = SparseOctreeDataset(npz_path, cache_targets=False)
            item = ds[0]
            targets = item["targets"]
            assert set(targets.keys()) == {0, 1, 2, 3, 4}
            for lv in range(5):
                side = 2 ** (4 - lv)
                assert targets[lv]["label"].shape == (side**3,)
                assert targets[lv]["is_leaf"].shape == (side**3,)

    def test_collate_multilevel(self) -> None:
        n = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir, n=n)
            ds = SparseOctreeDataset(npz_path, cache_targets=True)
            batch = sparse_octree_collate([ds[i] for i in range(n)])
            assert batch["noise_3d"].shape == (n, 15, 4, 2, 4)
            for lv in range(5):
                side = 2 ** (4 - lv)
                assert batch["targets"][lv]["label"].shape == (n, side**3)

    def test_max_samples_limiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = self._make_multilevel_npz(tmpdir, n=10)
            ds = SparseOctreeDataset(npz_path, cache_targets=False, max_samples=3)
            assert len(ds) == 3
