"""Tests for scripts/mipper.py — Voxy Mipper algorithm."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import numpy.typing as npt

# Ensure scripts/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from scripts.mipper import build_opacity_table  # noqa: E402
from scripts.mipper import build_opacity_table_from_blocklist  # noqa: E402
from scripts.mipper import mip_once_numpy, mip_volume_numpy  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AIR = 0
STONE = 1  # opaque
WATER = 2  # transparent (tier 1) — only when vocab is provided
DIRT = 3  # opaque


def _simple_table(n: int = 16) -> npt.NDArray[np.int64]:
    """Opacity table: 0=air, 1..n-1=opaque (tier 15)."""
    t = build_opacity_table(n)
    assert t[0] == 0
    assert t[1] == 15
    return t


def _water_table(n: int = 16) -> npt.NDArray[np.int64]:
    """Opacity table: WATER (id=2) gets tier 1."""
    vocab = {
        "minecraft:air": 0,
        "minecraft:stone": STONE,
        "minecraft:water": WATER,
        "minecraft:dirt": DIRT,
    }
    return build_opacity_table(n, vocab=vocab)


def _all_block(block_id: int, shape=(16, 16, 16)) -> npt.NDArray[np.int64]:
    return np.full(shape, block_id, dtype=np.int64)


# ---------------------------------------------------------------------------
# build_opacity_table
# ---------------------------------------------------------------------------


class TestBuildOpacityTable:
    def test_air_always_zero(self):
        t = build_opacity_table(5)
        assert t[0] == 0

    def test_default_all_opaque(self):
        t = build_opacity_table(10)
        assert all(t[1:] == 15)

    def test_water_transparent_with_vocab(self):
        t = _water_table()
        assert t[WATER] == 1
        assert t[STONE] == 15
        assert t[DIRT] == 15
        assert t[AIR] == 0

    def test_length(self):
        n = 1105
        t = build_opacity_table(n)
        assert len(t) == n


# ---------------------------------------------------------------------------
# build_opacity_table_from_blocklist
# ---------------------------------------------------------------------------


def _make_blocklist_json(records: list[dict]) -> Path:
    """Write a temporary blocklist JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(records, f)
    f.close()
    return Path(f.name)


_MINI_BLOCKLIST = [
    {"block": "Air", "opaque": "No", "opacity": 0},
    {"block": "Stone", "opaque": "Yes", "opacity": 15},
    {"block": "Glass", "opaque": "No", "opacity": 0},
    {"block": "Water", "opaque": "No", "opacity": 1},
    {"block": "Grass Block", "opaque": "Yes", "opacity": 15},
]


class TestBuildOpacityTableFromBlocklist:
    def test_opaque_block_gets_tier_15(self):
        bl = _make_blocklist_json(_MINI_BLOCKLIST)
        vocab = {"minecraft:air": 0, "minecraft:stone": 1, "minecraft:glass": 2}
        t = build_opacity_table_from_blocklist(bl, vocab, n_blocks=10)
        assert t[1] == 15, "Stone should be tier 15"

    def test_transparent_block_gets_tier_1(self):
        bl = _make_blocklist_json(_MINI_BLOCKLIST)
        vocab = {"minecraft:air": 0, "minecraft:glass": 1, "minecraft:water": 2}
        t = build_opacity_table_from_blocklist(bl, vocab, n_blocks=10)
        assert t[1] == 1, "Glass should be tier 1"
        assert t[2] == 1, "Water should be tier 1"

    def test_air_id_always_0(self):
        bl = _make_blocklist_json(_MINI_BLOCKLIST)
        vocab = {"minecraft:air": 0, "minecraft:stone": 1}
        t = build_opacity_table_from_blocklist(bl, vocab, n_blocks=5)
        assert t[0] == 0, "ID 0 must always be tier 0 (air)"

    def test_multiword_name_normalisation(self):
        bl = _make_blocklist_json(_MINI_BLOCKLIST)
        vocab = {"minecraft:air": 0, "minecraft:grass_block": 1}
        t = build_opacity_table_from_blocklist(bl, vocab, n_blocks=5)
        assert t[1] == 15, "'Grass Block' → 'grass_block' should normalise correctly"

    def test_unmatched_block_falls_back_to_heuristic(self):
        # "minecraft:oak_leaves" isn't in _MINI_BLOCKLIST; fallback uses _TRANSPARENT_FRAGMENTS
        bl = _make_blocklist_json(_MINI_BLOCKLIST)
        vocab = {"minecraft:air": 0, "minecraft:oak_leaves": 1}
        t = build_opacity_table_from_blocklist(bl, vocab, n_blocks=5)
        assert t[1] == 1, "Leaves should fall back to transparent via heuristic"

    def test_real_blocklist_if_present(self):
        """Smoke-test against the actual project blocklist.json when available."""
        bl_path = Path(__file__).parent.parent / "blocklist.json"
        if not bl_path.exists():
            pytest.skip("blocklist.json not present in workspace root")
        vocab = {
            "minecraft:air": 0,
            "minecraft:stone": 1,
            "minecraft:glass": 2,
            "minecraft:water": 3,
            "minecraft:oak_log": 4,
        }
        t = build_opacity_table_from_blocklist(bl_path, vocab, n_blocks=16)
        assert t[0] == 0, "air tier"
        assert t[1] == 15, "stone should be opaque"
        assert t[2] == 1, "glass should be transparent"
        assert t[3] == 1, "water should be transparent"
        assert len(t) == 16


# ---------------------------------------------------------------------------
# mip_once_numpy
# ---------------------------------------------------------------------------


class TestMipOnceNumpy:
    def test_all_air_returns_air(self):
        labels = _all_block(AIR)
        tbl = _simple_table()
        coarse, occ = mip_once_numpy(labels, tbl)
        assert coarse.shape == (8, 8, 8)
        assert np.all(coarse == AIR)
        assert np.all(occ == 0)

    def test_all_stone_returns_stone(self):
        labels = _all_block(STONE)
        tbl = _simple_table()
        coarse, occ = mip_once_numpy(labels, tbl)
        assert np.all(coarse == STONE)
        assert np.all(occ == 1)

    def test_output_shape(self):
        labels = np.random.randint(0, 4, size=(16, 16, 16), dtype=np.int64)
        tbl = _simple_table(8)
        coarse, occ = mip_once_numpy(labels, tbl)
        assert coarse.shape == (8, 8, 8)
        assert occ.shape == (8, 8, 8)

    def test_opaque_wins_over_air(self):
        """A single stone voxel in an otherwise air window must win."""
        labels = _all_block(AIR)
        # Place stone at the (0,0,0) position of each 2×2×2 window — lowest priority corner
        labels[0::2, 0::2, 0::2] = STONE
        tbl = _simple_table()
        coarse, occ = mip_once_numpy(labels, tbl)
        assert np.all(coarse == STONE), "Opaque corner 0 should still win vs. all-air"
        assert np.all(occ == 1)

    def test_opaque_beats_transparent(self):
        """Opaque block (tier 15) beats transparent non-air (tier 1) regardless of corner."""
        labels = _all_block(WATER)
        # Place stone at the lowest-priority corner (0,0,0)
        labels[0::2, 0::2, 0::2] = STONE
        tbl = _water_table()
        coarse, occ = mip_once_numpy(labels, tbl)
        assert np.all(
            coarse == STONE
        ), "Opaque at priority-0 corner must beat transparent at priority-7 corner"

    def test_equal_opacity_favours_i111(self):
        """Among equal-opacity blocks, the highest-priority corner (I111) wins."""
        labels = np.zeros((16, 16, 16), dtype=np.int64)
        # Fill all windows with DIRT (opaque, tier 15), but use different IDs so we can
        # track which corner was selected.
        for c_idx, (da0, da1, da2) in enumerate(
            [
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                (1, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        ):
            labels[da0::2, da1::2, da2::2] = c_idx + 1  # IDs 1..8, all opaque

        tbl = _simple_table(9)
        coarse, _ = mip_once_numpy(labels, tbl)
        # I111 corresponds to corner 7 (id=8) which is labels[1::2,1::2,1::2]
        assert np.all(coarse == 8), f"Expected I111 winner (id=8), got unique={np.unique(coarse)}"

    def test_occupancy_is_nonzero_when_any_solid(self):
        """If at least one voxel in a window is solid, coarse_occ must be 1."""
        labels = _all_block(AIR)
        labels[0, 0, 0] = STONE  # Only corner 0 of the first window
        tbl = _simple_table()
        _, occ = mip_once_numpy(labels, tbl)
        assert occ[0, 0, 0] == 1

    def test_occupancy_dtype(self):
        labels = np.zeros((8, 8, 8), dtype=np.int64)
        tbl = _simple_table()
        _, occ = mip_once_numpy(labels, tbl)
        assert occ.dtype == np.uint8


# ---------------------------------------------------------------------------
# mip_volume_numpy
# ---------------------------------------------------------------------------


class TestMipVolumeNumpy:
    def test_factor1_is_identity(self):
        labels = np.random.randint(0, 4, (16, 16, 16), dtype=np.int64)
        tbl = _simple_table()
        coarse, occ = mip_volume_numpy(labels, factor=1, opacity_table=tbl)
        np.testing.assert_array_equal(coarse, labels)
        np.testing.assert_array_equal(occ, (labels != 0).astype(np.uint8))

    @pytest.mark.parametrize("factor", [2, 4, 8, 16])
    def test_output_shape(self, factor: int):
        labels = np.random.randint(0, 4, (16, 16, 16), dtype=np.int64)
        tbl = _simple_table()
        coarse, occ = mip_volume_numpy(labels, factor, tbl)
        expected = 16 // factor
        assert coarse.shape == (expected, expected, expected)
        assert occ.shape == (expected, expected, expected)

    def test_factor2_equals_single_mip(self):
        """mip_volume(factor=2) must match one call to mip_once."""
        labels = np.random.randint(0, 4, (16, 16, 16), dtype=np.int64)
        tbl = _simple_table()
        vol2, _ = mip_volume_numpy(labels, 2, tbl)
        once, _ = mip_once_numpy(labels, tbl)
        np.testing.assert_array_equal(vol2, once)

    def test_factor4_equals_two_mips(self):
        labels = np.random.randint(0, 4, (16, 16, 16), dtype=np.int64)
        tbl = _simple_table()
        vol4, _ = mip_volume_numpy(labels, 4, tbl)
        step1, _ = mip_once_numpy(labels, tbl)
        step2, _ = mip_once_numpy(step1, tbl)
        np.testing.assert_array_equal(vol4, step2)

    def test_occupancy_monotonicity(self):
        """Mipper occ ≤ OR-pool: if Mipper says solid, OR-pool must also say solid."""
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, (16, 16, 16), dtype=np.int64)
        tbl = _simple_table()
        _, mipper_occ = mip_volume_numpy(labels, 2, tbl)

        # Reference OR-pool
        occ16 = (labels != 0).astype(np.uint8)
        or_occ = (
            occ16[0::2, 0::2, 0::2]
            | occ16[1::2, 0::2, 0::2]
            | occ16[0::2, 1::2, 0::2]
            | occ16[1::2, 1::2, 0::2]
            | occ16[0::2, 0::2, 1::2]
            | occ16[1::2, 0::2, 1::2]
            | occ16[0::2, 1::2, 1::2]
            | occ16[1::2, 1::2, 1::2]
        )
        # Mipper occ must be 0 wherever OR-pool is 0 (no false solids)
        assert np.all(
            mipper_occ[or_occ == 0] == 0
        ), "Mipper created solid where OR-pool sees only air"

    def test_all_stone_preserved_at_every_factor(self):
        labels = _all_block(STONE)
        tbl = _simple_table()
        for factor in [1, 2, 4, 8, 16]:
            coarse, occ = mip_volume_numpy(labels, factor, tbl)
            assert np.all(coarse == STONE)
            assert np.all(occ == 1)

    def test_invalid_factor_raises(self):
        labels = np.zeros((16, 16, 16), dtype=np.int64)
        with pytest.raises(ValueError):
            mip_volume_numpy(labels, factor=3)

    def test_lazy_opacity_table(self):
        """Should work without explicitly passing an opacity table."""
        labels = np.random.randint(0, 4, (16, 16, 16), dtype=np.int64)
        coarse, occ = mip_volume_numpy(labels, factor=2)  # no table arg
        assert coarse.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# PyTorch equivalence check
# ---------------------------------------------------------------------------


class TestMipTorchMatchesNumpy:
    @pytest.fixture
    def labels_np(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 6, (16, 16, 16), dtype=np.int64)

    def _torch_available(self):
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.mark.parametrize("factor", [2, 4])
    def test_torch_matches_numpy(self, labels_np, factor):
        if not self._torch_available():
            pytest.skip("torch not installed")
        import torch

        from scripts.mipper import build_opacity_table, mip_volume_numpy, mip_volume_torch

        tbl_np = build_opacity_table(8)
        tbl_t = torch.from_numpy(tbl_np).long()

        expected_labels, expected_occ = mip_volume_numpy(labels_np, factor, tbl_np)

        labels_t = torch.from_numpy(labels_np).unsqueeze(0).long()  # (1, 16, 16, 16)
        coarse_t, occ_t = mip_volume_torch(labels_t, factor, tbl_t)

        np.testing.assert_array_equal(
            coarse_t.squeeze(0).numpy(),
            expected_labels,
            err_msg="PyTorch and NumPy Mipper disagree on block IDs",
        )
        np.testing.assert_array_equal(
            occ_t.squeeze(0).numpy().astype(np.uint8),
            expected_occ,
            err_msg="PyTorch and NumPy Mipper disagree on occupancy",
        )
