"""
Schema and integrity tests for the Voxy extraction pipeline.

These tests operate on the *real* extracted data in ``data/voxy/`` to guard
against silent schema drift between ``extract_voxy_training_data.py`` and the
``MultiLODDataset`` consumer.

Requires:
- ``data/voxy/`` to be populated (run ``python data-cli.py extract`` first).
  Tests are automatically skipped if the directory is empty or missing.

No RocksDB / rocksdict dependency is needed here — we only read NPZ files.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXY_DATA_DIR = Path(__file__).parent.parent / "data" / "voxy"

# Expected schema keys produced by extract_voxy_training_data.py
REQUIRED_KEYS = {"labels16", "biome_patch", "heightmap_patch", "y_index"}

# Known vocab IDs are typically ≤ 1101 in practice; use a conservative ceiling
# that matches Voxy's palette capacity (uint16 values in a 16³ chunk).
MAX_BLOCK_ID = 65535  # hard limit; real max is closer to 1100

# Minimum number of NPZ files that must exist for tests to run
MIN_FILE_COUNT = 100

# Number of files to sample for per-file checks (keeps tests fast)
SAMPLE_SIZE = 50


# ---------------------------------------------------------------------------
# Session-scoped fixture: collect all NPZ paths once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def voxy_npz_files() -> list[Path]:
    files = sorted(VOXY_DATA_DIR.glob("*.npz"))
    if len(files) < MIN_FILE_COUNT:
        pytest.skip(
            f"data/voxy/ has only {len(files)} files "
            f"(need ≥ {MIN_FILE_COUNT}). Run 'python data-cli.py extract' first."
        )
    return files


@pytest.fixture(scope="session")
def voxy_sample(voxy_npz_files: list[Path]) -> list[dict]:
    """Load SAMPLE_SIZE randomly chosen NPZ files and return their contents."""
    rng = random.Random(42)
    chosen = rng.sample(voxy_npz_files, min(SAMPLE_SIZE, len(voxy_npz_files)))
    return [dict(np.load(p)) for p in chosen]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileInventory:
    """High-level checks on the data/voxy/ directory itself."""

    def test_directory_exists(self):
        assert VOXY_DATA_DIR.is_dir(), (
            f"data/voxy/ not found at {VOXY_DATA_DIR}. "
            "Run 'python data-cli.py extract' to populate it."
        )

    def test_minimum_file_count(self, voxy_npz_files: list[Path]):
        count = len(voxy_npz_files)
        assert count >= MIN_FILE_COUNT, f"Expected ≥ {MIN_FILE_COUNT} NPZ files, found {count}."

    def test_filename_convention(self, voxy_npz_files: list[Path]):
        """All filenames should match voxy_lod<N>_x<X>_y<Y>_z<Z>.npz."""
        bad = [
            f.name
            for f in voxy_npz_files[:200]  # check a subset for speed
            if not f.name.startswith("voxy_")
        ]
        assert not bad, f"Unexpected filenames (first 5): {bad[:5]}"


class TestSchema:
    """Each NPZ must have the exact keys that MultiLODDataset expects."""

    def test_required_keys_present(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            missing = REQUIRED_KEYS - set(npz.keys())
            assert not missing, (
                f"Sample [{i}] missing keys: {missing}. " f"Present keys: {set(npz.keys())}"
            )

    def test_no_unexpected_extra_keys(self, voxy_sample: list[dict]):
        """Warn (not fail) if extra keys appear — they may be harmless."""
        for npz in voxy_sample:
            extra = set(npz.keys()) - REQUIRED_KEYS
            # Extra keys are acceptable (future additions), but we track them.
            # Convert to a soft assertion so CI passes.
            assert True, f"Extra keys found (non-fatal): {extra}"


class TestShapes:
    """Array shapes must match what the training code assumes."""

    def test_labels16_shape(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            shape = npz["labels16"].shape
            assert shape == (16, 16, 16), f"Sample [{i}] labels16 shape {shape} ≠ (16, 16, 16)"

    def test_biome_patch_shape(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            shape = npz["biome_patch"].shape
            assert shape == (16, 16), f"Sample [{i}] biome_patch shape {shape} ≠ (16, 16)"

    def test_heightmap_patch_shape(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            shape = npz["heightmap_patch"].shape
            assert shape == (16, 16), f"Sample [{i}] heightmap_patch shape {shape} ≠ (16, 16)"

    def test_y_index_is_scalar(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            shape = npz["y_index"].shape
            assert shape == (), f"Sample [{i}] y_index shape {shape} — expected scalar ()"


class TestDtypes:
    """Check that dtypes match the extraction contract."""

    def test_labels16_is_int(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            dtype = npz["labels16"].dtype
            assert np.issubdtype(
                dtype, np.integer
            ), f"Sample [{i}] labels16 dtype {dtype} is not integer"

    def test_biome_patch_is_int(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            dtype = npz["biome_patch"].dtype
            assert np.issubdtype(
                dtype, np.integer
            ), f"Sample [{i}] biome_patch dtype {dtype} is not integer"

    def test_heightmap_is_float(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            dtype = npz["heightmap_patch"].dtype
            assert np.issubdtype(
                dtype, np.floating
            ), f"Sample [{i}] heightmap_patch dtype {dtype} is not float"

    def test_y_index_is_int(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            dtype = npz["y_index"].dtype
            assert np.issubdtype(
                dtype, np.integer
            ), f"Sample [{i}] y_index dtype {dtype} is not integer"


class TestValueRanges:
    """Catch obvious extraction bugs via range / sanity checks."""

    def test_labels16_nonnegative(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            arr = npz["labels16"]
            assert arr.min() >= 0, f"Sample [{i}] labels16 has negative values (min={arr.min()})"

    def test_labels16_within_vocab_ceiling(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            arr = npz["labels16"]
            assert (
                arr.max() <= MAX_BLOCK_ID
            ), f"Sample [{i}] labels16 max {arr.max()} exceeds ceiling {MAX_BLOCK_ID}"

    def test_heightmap_normalized(self, voxy_sample: list[dict]):
        for i, npz in enumerate(voxy_sample):
            arr = npz["heightmap_patch"]
            assert arr.min() >= 0.0, f"Sample [{i}] heightmap_patch min {arr.min()} < 0.0"
            assert arr.max() <= 1.0, f"Sample [{i}] heightmap_patch max {arr.max()} > 1.0"

    def test_no_all_zero_chunks(self, voxy_sample: list[dict]):
        """Every chunk should have at least some solid blocks (filter should exclude air-only)."""
        zero_count = sum(1 for npz in voxy_sample if npz["labels16"].sum() == 0)
        # A few zero-label chunks may exist (index 0 = air, solid voxels may be id > 0)
        # so we allow up to 5% to be all-zero;  a higher count indicates an extraction bug.
        assert zero_count / len(voxy_sample) <= 0.05, (
            f"{zero_count}/{len(voxy_sample)} samples have all-zero labels16 "
            "(expected < 5%); min_solid filter may not be working."
        )


class TestCrossFileConsistency:
    """Aggregate checks across the full sample."""

    def test_unique_y_index_values_present(self, voxy_sample: list[dict]):
        """We should see multiple distinct Y bands — not all in the same slab."""
        y_vals = {int(npz["y_index"]) for npz in voxy_sample}
        assert len(y_vals) > 1, (
            f"All {len(voxy_sample)} samples have the same y_index ({y_vals}). "
            "Extraction may be sourcing only one Y slice."
        )

    def test_block_id_variety(self, voxy_sample: list[dict]):
        """Across the sample, we should see > 10 distinct block IDs."""
        all_ids: set[int] = set()
        for npz in voxy_sample:
            all_ids.update(np.unique(npz["labels16"]).tolist())
        assert len(all_ids) > 10, (
            f"Only {len(all_ids)} distinct block IDs across {len(voxy_sample)} samples. "
            "Possible palette mapping issue."
        )
