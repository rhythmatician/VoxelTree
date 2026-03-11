"""
tests/test_voxy_extraction.py — Schema and integrity tests for the octree
Voxy extraction pipeline.

All tests are skipped automatically when the data directory is absent
(CI / fresh clone).  They only run when a developer has already executed
the extraction steps locally:

    python data-cli.py dataprep --from-step extract-octree \
        --voxy-dir LODiffusion/run/saves
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent  # VoxelTree root
DATA_DIR = _HERE / "data" / "voxy_octree"

REQUIRED_LEVELS = [f"level_{i}" for i in range(5)]  # level_0 … level_4
PAIR_CACHE_NAMES = ["train_octree_pairs.npz", "val_octree_pairs.npz"]

# Required keys and expected shapes for individual section NPZs.
# (N=32 for all octree levels.)
SECTION_REQUIRED_KEYS = {
    "labels32": {"ndim": 3, "shape": (32, 32, 32)},
    "biome32": {"ndim": 2, "shape": (32, 32)},
    "y_section": {"ndim": 0},  # scalar
    "heightmap32": {"ndim": 3, "shape": (5, 32, 32)},
}

# Required keys for pair-cache NPZs (N samples per cache).
PAIR_CACHE_REQUIRED_KEYS = {
    "labels32": {"ndim": 4},           # (N, 32, 32, 32)
    "parent_labels32": {"ndim": 4},    # (N, 32, 32, 32)
    "heightmap32": {"ndim": 4},        # (N, 5, 32, 32)
    "biome32": {"ndim": 3},            # (N, 32, 32)
    "y_position": {"ndim": 1},         # (N,)
    "level": {"ndim": 1},              # (N,)
    "non_empty_children": {"ndim": 1},  # (N,)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_data(path: pathlib.Path) -> None:
    """Skip the test if *path* does not exist."""
    if not path.exists():
        pytest.skip(f"Data not present: {path}")


def _first_npz(level_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the first .npz file in *level_dir*, or None."""
    files = sorted(level_dir.glob("*.npz"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLevelDirectories:
    """Verify that all expected level directories exist and are non-empty."""

    def test_data_root_exists(self) -> None:
        _skip_if_no_data(DATA_DIR)
        assert DATA_DIR.is_dir()

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_level_directory_exists(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        assert level_dir.is_dir(), f"Missing level directory: {level_dir}"

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_level_directory_non_empty(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_files = list(level_dir.glob("*.npz"))
        assert len(npz_files) > 0, (
            f"No .npz files found in {level_dir}. "
            "Run: python data-cli.py dataprep --from-step extract-octree"
        )


class TestOctreeNpzSchema:
    """Validate the schema of individual section NPZ files at each level."""

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_required_keys_present(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        missing = set(SECTION_REQUIRED_KEYS) - set(data.files)
        assert not missing, (
            f"Missing keys in {npz_path.name}: {missing}"
        )

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_array_shapes(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        for key, spec in SECTION_REQUIRED_KEYS.items():
            if key not in data:
                continue
            arr = data[key]
            assert arr.ndim == spec["ndim"], (
                f"{npz_path.name}/{key}: expected ndim={spec['ndim']}, "
                f"got ndim={arr.ndim}"
            )
            if "shape" in spec:
                assert arr.shape == spec["shape"], (
                    f"{npz_path.name}/{key}: expected shape={spec['shape']}, "
                    f"got shape={arr.shape}"
                )

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_labels32_dtype(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        if "labels32" not in data:
            return
        assert np.issubdtype(data["labels32"].dtype, np.integer), (
            f"labels32 must be integer dtype, got {data['labels32'].dtype}"
        )

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_heightmap32_dtype(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        if "heightmap32" not in data:
            return
        assert data["heightmap32"].dtype == np.float32, (
            f"heightmap32 must be float32, got {data['heightmap32'].dtype}"
        )

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_no_nan_in_heightmap32(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        if "heightmap32" not in data:
            return
        assert not np.any(np.isnan(data["heightmap32"])), (
            f"NaN values found in heightmap32 of {npz_path.name}"
        )


class TestPairCacheSchema:
    """Validate the schema of the octree pair cache NPZs."""

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_pair_cache_exists(self, cache_name: str) -> None:
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        assert cache_path.exists(), (
            f"Pair cache not found: {cache_path}. "
            "Run: python data-cli.py dataprep --from-step build-octree-pairs"
        )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_required_keys_present(self, cache_name: str) -> None:
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        missing = set(PAIR_CACHE_REQUIRED_KEYS) - set(data.files)
        assert not missing, (
            f"Missing keys in {cache_name}: {missing}"
        )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_consistent_batch_size(self, cache_name: str) -> None:
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        # All arrays must have the same leading (batch) dimension.
        n_samples: int | None = None
        for key in PAIR_CACHE_REQUIRED_KEYS:
            if key not in data:
                continue
            arr = data[key]
            if n_samples is None:
                n_samples = arr.shape[0]
            else:
                assert arr.shape[0] == n_samples, (
                    f"{cache_name}/{key}: inconsistent batch size "
                    f"(expected {n_samples}, got {arr.shape[0]})"
                )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_array_ndims(self, cache_name: str) -> None:
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        for key, spec in PAIR_CACHE_REQUIRED_KEYS.items():
            if key not in data:
                continue
            arr = data[key]
            assert arr.ndim == spec["ndim"], (
                f"{cache_name}/{key}: expected ndim={spec['ndim']}, "
                f"got ndim={arr.ndim}"
            )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_level_values_in_range(self, cache_name: str) -> None:
        """Octree levels must be in [0, 4]."""
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "level" not in data:
            return
        levels = data["level"]
        assert np.all(levels >= 0) and np.all(levels <= 4), (
            f"{cache_name}/level: values out of range [0,4]: "
            f"min={levels.min()}, max={levels.max()}"
        )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_non_empty_children_bitmask(self, cache_name: str) -> None:
        """non_empty_children must be uint8 values in [0, 255]."""
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "non_empty_children" not in data:
            return
        nec = data["non_empty_children"]
        assert nec.dtype == np.uint8, (
            f"{cache_name}/non_empty_children: expected uint8, got {nec.dtype}"
        )
