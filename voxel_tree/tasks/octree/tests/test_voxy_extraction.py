"""
tests/test_voxy_extraction.py — Schema and integrity tests for the octree
Voxy extraction pipeline.

All tests are skipped automatically when the data directory is absent
(CI / fresh clone).  They only run when a developer has already executed
the extraction steps locally:

    python data-cli.py dataprep --from-step extract-octree \
        --voxy-dir LODiffusion/run/saves

Section NPZ schema (produced by extract_octree_data.py)::

    labels32             (32, 32, 32) int32  — canonical Voxy vocabulary IDs
    biome32              (32, 32)     int32  — biome IDs (column-wise majority)
    level                scalar       int64  — LOD level (0–4)
    section_x            scalar       int64  — section X coordinate
    section_y            scalar       int64  — section Y coordinate
    section_z            scalar       int64  — section Z coordinate
    non_empty_children   scalar       uint8  — 8-bit child occupancy mask

    NOTE: ``heightmap32`` is NOT present in section NPZs.  It is produced by a
    separate column-heights step and optionally merged during pair-building.

Pair cache NPZ schema (produced by build_pairs.py)::

    labels32             (N, 32, 32, 32) int32
    parent_labels32      (N, 32, 32, 32) int32   (zeros for L4 init model)
    parent_octant16      (N, 16, 16, 16) int32   (native 16³ Voxy octant)
    heightmap32          (N, 5, 32, 32)  float32
    biome32              (N, 32, 32)     int32
    y_position           (N,)            int64   (raw_y + 4 offset, range [0, 23])
    level                (N,)            int64
    non_empty_children   (N,)            uint8
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent  # VoxelTree root
DATA_DIR = _HERE / "data" / "voxy_octree"

REQUIRED_LEVELS = [f"level_{i}" for i in range(5)]  # level_0 … level_4

# Unified caches (model_type="all") and per-model caches are both valid.
# Only the unified ones are required for basic pipeline validation.
PAIR_CACHE_NAMES = ["train_octree_pairs.npz", "val_octree_pairs.npz"]

# Required keys and expected shapes for individual section NPZs.
# heightmap32 is NOT in section NPZs — it is added by the column-heights step
# and lives in the pair-cache only.
SECTION_REQUIRED_KEYS: dict[str, dict[str, Any]] = {
    "labels32": {"ndim": 3, "shape": (32, 32, 32)},
    "biome32": {"ndim": 2, "shape": (32, 32)},
    "level": {"ndim": 0},  # scalar int64
    "section_x": {"ndim": 0},  # scalar int64
    "section_y": {"ndim": 0},  # scalar int64
    "section_z": {"ndim": 0},  # scalar int64
    "non_empty_children": {"ndim": 0},  # scalar uint8
}

# Required keys for pair-cache NPZs (N samples per cache).
PAIR_CACHE_REQUIRED_KEYS: dict[str, dict[str, Any]] = {
    "labels32": {"ndim": 4},  # (N, 32, 32, 32)
    "parent_labels32": {"ndim": 4},  # (N, 32, 32, 32)
    "parent_octant16": {"ndim": 4},  # (N, 16, 16, 16) — native 16³ Voxy octant
    "heightmap32": {"ndim": 4},  # (N, 5, 32, 32)
    "biome32": {"ndim": 3},  # (N, 32, 32)
    "y_position": {"ndim": 1},  # (N,) — shifted: raw_y + 4
    "level": {"ndim": 1},  # (N,)
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
    """Validate the schema of individual section NPZ files at each level.

    Section NPZs are produced by ``extract_octree_data.py``.  They do NOT
    contain ``heightmap32`` — that field is produced by the separate
    column-heights step and lives in the pair cache only.
    """

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_required_keys_present(self, level: str) -> None:
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        missing = set(SECTION_REQUIRED_KEYS) - set(data.files)
        assert not missing, f"Missing keys in {npz_path.name}: {missing}"

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
            assert (
                arr.ndim == spec["ndim"]
            ), f"{npz_path.name}/{key}: expected ndim={spec['ndim']}, got ndim={arr.ndim}"
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
        assert np.issubdtype(
            data["labels32"].dtype, np.integer
        ), f"labels32 must be integer dtype, got {data['labels32'].dtype}"

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_scalar_fields_dtypes(self, level: str) -> None:
        """level/section_x/section_y/section_z must be int64; non_empty_children uint8."""
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        for key in ("level", "section_x", "section_y", "section_z"):
            if key not in data:
                continue
            assert (
                data[key].dtype == np.int64
            ), f"{npz_path.name}/{key}: expected int64, got {data[key].dtype}"
        if "non_empty_children" in data:
            assert data["non_empty_children"].dtype == np.uint8, (
                f"{npz_path.name}/non_empty_children: expected uint8, "
                f"got {data['non_empty_children'].dtype}"
            )

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_level_scalar_in_range(self, level: str) -> None:
        """The ``level`` scalar in each section NPZ must match the directory level (0–4)."""
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        expected_level = int(level.split("_")[1])
        data = np.load(npz_path)
        if "level" not in data:
            return
        actual = int(data["level"])
        assert (
            actual == expected_level
        ), f"{npz_path.name}: expected level={expected_level}, got level={actual}"

    @pytest.mark.parametrize("level", REQUIRED_LEVELS)
    def test_no_heightmap_in_section_npz(self, level: str) -> None:
        """heightmap32 must NOT be present in section NPZs.

        The column-heights step produces heightmaps and they are added only
        in the pair cache.  Finding heightmap32 here indicates the extraction
        script has regressed to an old schema.
        """
        _skip_if_no_data(DATA_DIR)
        level_dir = DATA_DIR / level
        npz_path = _first_npz(level_dir)
        if npz_path is None:
            pytest.skip(f"No NPZ files in {level_dir}")

        data = np.load(npz_path)
        assert "heightmap32" not in data.files, (
            f"{npz_path.name} unexpectedly contains 'heightmap32'. "
            "Section NPZs should not have heightmaps; check extract_octree_data.py."
        )


class TestPairCacheSchema:
    """Validate the schema of the octree pair cache NPZs.

    Pair caches are produced by ``build_pairs.py``.  The unified cache files
    (``{split}_octree_pairs.npz``) cover all 5 LOD levels.  Per-model caches
    (e.g. ``init_train_octree_pairs.npz``) follow the same schema.
    """

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
        assert not missing, f"Missing keys in {cache_name}: {missing}"

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
            assert (
                arr.ndim == spec["ndim"]
            ), f"{cache_name}/{key}: expected ndim={spec['ndim']}, got ndim={arr.ndim}"

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_parent_octant16_shape(self, cache_name: str) -> None:
        """parent_octant16 must be (N, 16, 16, 16) — the native Voxy 16³ octant."""
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "parent_octant16" not in data:
            return
        arr = data["parent_octant16"]
        assert arr.ndim == 4, f"{cache_name}/parent_octant16: expected ndim=4, got ndim={arr.ndim}"
        assert arr.shape[1:] == (16, 16, 16), (
            f"{cache_name}/parent_octant16: expected spatial shape (16,16,16), "
            f"got {arr.shape[1:]}"
        )

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_heightmap32_dtype(self, cache_name: str) -> None:
        """heightmap32 in the pair cache must be float32."""
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "heightmap32" not in data:
            return
        assert (
            data["heightmap32"].dtype == np.float32
        ), f"{cache_name}/heightmap32: expected float32, got {data['heightmap32'].dtype}"

    @pytest.mark.parametrize("cache_name", PAIR_CACHE_NAMES)
    def test_no_nan_in_heightmap32(self, cache_name: str) -> None:
        """heightmap32 must not contain NaN values."""
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "heightmap32" not in data:
            return
        assert not np.any(
            np.isnan(data["heightmap32"])
        ), f"NaN values found in heightmap32 of {cache_name}"

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
    def test_y_position_range(self, cache_name: str) -> None:
        """y_position must be in [0, 23].

        Values are stored as ``raw_section_y + 4`` to shift the typical
        Minecraft range (−4 … +19) into the embedding index range [0, 23].
        """
        _skip_if_no_data(DATA_DIR)
        cache_path = DATA_DIR / cache_name
        if not cache_path.exists():
            pytest.skip(f"Pair cache not found: {cache_path}")

        data = np.load(cache_path)
        if "y_position" not in data:
            return
        yp = data["y_position"]
        assert np.all(yp >= 0) and np.all(yp <= 23), (
            f"{cache_name}/y_position: values out of range [0, 23]: "
            f"min={yp.min()}, max={yp.max()}"
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
        assert (
            nec.dtype == np.uint8
        ), f"{cache_name}/non_empty_children: expected uint8, got {nec.dtype}"
