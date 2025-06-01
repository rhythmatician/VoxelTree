"""
Test suite for Phase 2.1: Assembling parent-child LOD pairs

RED phase tests - these should all fail initially as we haven't implemented
the PatchPairer class yet. Each test validates a specific aspect of LOD
patch pairing functionality.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.pairing.lod_validator import LODValidator
# Import the classes we'll be testing (these don't exist yet - will fail)
from scripts.pairing.patch_pairer import PatchPairer


class TestPatchPairerInitialization:
    """Test PatchPairer class initialization and configuration."""

    def test_patch_pairer_init_with_config(self):
        """RED: Fails because PatchPairer doesn't exist yet."""
        config_path = Path("config.yaml")
        pairer = PatchPairer(config_path=config_path)

        assert pairer.config_path == config_path
        assert hasattr(pairer, "extracted_data_dir")
        assert hasattr(pairer, "seed_inputs_dir")
        assert hasattr(pairer, "output_dir")
        assert hasattr(pairer, "lod_levels")

    def test_patch_pairer_init_default_config(self):
        """RED: Fails because PatchPairer doesn't exist yet."""
        pairer = PatchPairer()

        assert pairer.config_path == Path("config.yaml")
        assert hasattr(pairer, "lod_levels")
        assert pairer.lod_levels > 0


class TestParentChildPairAssembly:
    """Test core functionality of assembling parent-child LOD pairs."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with mock extracted chunk data."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create mock extracted chunk data
        chunk_data = {
            "block_types": np.random.randint(0, 5, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(0),
            "chunk_z": np.int32(0),
            "region_file": "test_region.mca",
        }

        chunks_dir = temp_dir / "chunks"
        chunks_dir.mkdir()
        np.savez_compressed(chunks_dir / "chunk_0_0.npz", **chunk_data)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_extract_parent_child_pairs_from_chunk(self, temp_data_dir):
        """RED: Fails if parent-child pair extraction is incorrect."""
        pairer = PatchPairer()
        chunk_file = temp_data_dir / "chunks" / "chunk_0_0.npz"

        pairs = pairer.extract_parent_child_pairs(chunk_file)

        # Should generate multiple 16x16x16 subchunks with 8x8x8 parents
        assert len(pairs) > 0
        assert len(pairs) == 24  # 384 height / 16 = 24 subchunks

        for pair in pairs:
            # Validate parent voxel dimensions (downsampled)
            assert pair["parent_voxel"].shape == (8, 8, 8)
            # Validate target dimensions (full resolution)
            assert pair["target_mask"].shape == (16, 16, 16)
            assert pair["target_types"].shape == (16, 16, 16)
            # Validate metadata
            assert "y_index" in pair
            assert "chunk_x" in pair
            assert "chunk_z" in pair

    def test_downsample_to_parent_voxel(self, temp_data_dir):
        """RED: Fails if downsampling logic is incorrect."""
        pairer = PatchPairer()

        # Create a 16x16x16 target subchunk
        target_data = np.random.choice([True, False], size=(16, 16, 16))

        parent_voxel = pairer.downsample_to_parent(target_data)

        # Should be downsampled to 8x8x8
        assert parent_voxel.shape == (8, 8, 8)
        # Should use max pooling (most common block in 2x2x2 region)
        assert parent_voxel.dtype == target_data.dtype

    def test_slice_chunk_into_subchunks(self, temp_data_dir):
        """RED: Fails if chunk slicing is misaligned."""
        pairer = PatchPairer()
        chunk_file = temp_data_dir / "chunks" / "chunk_0_0.npz"

        subchunks = pairer.slice_chunk_into_subchunks(chunk_file)

        # Should create 24 subchunks (384 / 16 = 24 vertical slices)
        assert len(subchunks) == 24

        for i, subchunk in enumerate(subchunks):
            assert subchunk["target_mask"].shape == (16, 16, 16)
            assert subchunk["target_types"].shape == (16, 16, 16)
            assert subchunk["y_index"] == i

    def test_lod_level_assignment(self, temp_data_dir):
        """RED: Fails if LOD levels aren't properly assigned."""
        pairer = PatchPairer()
        chunk_file = temp_data_dir / "chunks" / "chunk_0_0.npz"

        pairs = pairer.extract_parent_child_pairs(chunk_file, lod_level=2)

        for pair in pairs:
            assert pair["lod"] == 2
            # Parent should be more aggressively downsampled at higher LOD
            if pair["lod"] > 1:
                assert pair["parent_voxel"].shape[0] <= 8


class TestLODAlignment:
    """Test LOD alignment and validation logic."""

    def test_validate_lod_alignment(self):
        """RED: Fails if LOD alignment validation is missing."""
        validator = LODValidator()

        # Valid LOD pair
        parent = np.ones((8, 8, 8), dtype=bool)
        target = np.ones((16, 16, 16), dtype=bool)

        is_valid = validator.validate_lod_alignment(parent, target, lod_level=1)
        assert is_valid is True

        # Invalid LOD pair (wrong dimensions)
        invalid_parent = np.ones((4, 4, 4), dtype=bool)
        is_invalid = validator.validate_lod_alignment(invalid_parent, target, lod_level=1)
        assert is_invalid is False

    def test_detect_lod_mismatch(self):
        """RED: Fails if LOD mismatch detection is incorrect."""
        validator = LODValidator()

        # Create mismatched parent-child pair
        parent = np.ones((8, 8, 8), dtype=bool)
        target = np.ones((32, 32, 32), dtype=bool)  # Wrong target size

        mismatch = validator.detect_lod_mismatch(parent, target)
        assert mismatch is True

        # Create correctly matched pair
        correct_target = np.ones((16, 16, 16), dtype=bool)
        no_mismatch = validator.detect_lod_mismatch(parent, correct_target)
        assert no_mismatch is False


class TestBatchPairProcessing:
    """Test batch processing of multiple chunks into LOD pairs."""

    @pytest.fixture
    def temp_batch_dir(self):
        """Create temporary directory with multiple mock chunk files."""
        temp_dir = Path(tempfile.mkdtemp())
        chunks_dir = temp_dir / "chunks"
        chunks_dir.mkdir()

        # Create multiple mock chunks
        for x in range(3):
            for z in range(3):
                chunk_data = {
                    "block_types": np.random.randint(0, 5, size=(16, 16, 384), dtype=np.uint8),
                    "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
                    "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                    "heightmap": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                    "chunk_x": np.int32(x),
                    "chunk_z": np.int32(z),
                    "region_file": f"test_region_{x}_{z}.mca",
                }
                np.savez_compressed(chunks_dir / f"chunk_{x}_{z}.npz", **chunk_data)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_process_batch_chunks(self, temp_batch_dir):
        """RED: Fails if batch processing doesn't handle multiple chunks."""
        pairer = PatchPairer()
        chunks_dir = temp_batch_dir / "chunks"
        output_dir = temp_batch_dir / "pairs"

        pair_count = pairer.process_batch(chunks_dir, output_dir)

        # Should process 9 chunks (3x3) × 24 subchunks each = 216 pairs
        assert pair_count == 216
        assert output_dir.exists()

        # Check output files exist
        pair_files = list(output_dir.glob("*.npz"))
        assert len(pair_files) == 216

    def test_parallel_pair_processing(self, temp_batch_dir):
        """RED: Fails if parallel processing doesn't work correctly."""
        pairer = PatchPairer()
        chunks_dir = temp_batch_dir / "chunks"
        output_dir = temp_batch_dir / "pairs_parallel"

        pair_count = pairer.process_batch_parallel(chunks_dir, output_dir, num_workers=2)

        # Should produce same result as sequential processing
        assert pair_count == 216

        # Validate all pairs were created correctly
        pair_files = list(output_dir.glob("*.npz"))
        assert len(pair_files) == 216


class TestPairOutput:
    """Test the output format and validation of LOD pairs."""

    def test_pair_output_format(self):
        """RED: Fails if pair output format doesn't match specification."""
        pairer = PatchPairer()

        # Mock input data
        mock_subchunk = {
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
        }

        pair = pairer.create_training_pair(mock_subchunk, lod_level=1)

        # Validate required keys
        required_keys = [
            "parent_voxel",
            "target_mask",
            "target_types",
            "y_index",
            "chunk_x",
            "chunk_z",
            "lod",
        ]
        for key in required_keys:
            assert key in pair

        # Validate data types and shapes
        assert pair["parent_voxel"].shape == (8, 8, 8)
        assert pair["target_mask"].shape == (16, 16, 16)
        assert pair["target_types"].shape == (16, 16, 16)
        assert isinstance(pair["y_index"], (int, np.integer))
        assert isinstance(pair["lod"], (int, np.integer))

    def test_save_pair_npz(self, tmp_path):
        """RED: Fails if NPZ saving doesn't preserve data integrity."""
        pairer = PatchPairer()

        # Create mock pair data
        pair_data = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }

        output_path = tmp_path / "test_pair.npz"
        saved_path = pairer.save_pair_npz(pair_data, output_path)

        # Verify file was created
        assert saved_path.exists()

        # Verify data integrity
        loaded_data = np.load(saved_path)
        for key, value in pair_data.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(loaded_data[key], value)
            else:
                assert loaded_data[key] == value


class TestErrorHandling:
    """Test error handling and edge cases in pair assembly."""

    def test_handle_corrupted_chunk_file(self):
        """RED: Fails if corrupted chunk handling is missing."""
        pairer = PatchPairer()

        # Create an invalid/corrupted NPZ file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"corrupted data")
            corrupted_file = Path(f.name)

        try:
            with pytest.raises(Exception):  # Should raise appropriate exception
                pairer.extract_parent_child_pairs(corrupted_file)
        finally:
            corrupted_file.unlink()

    def test_handle_missing_chunk_data(self):
        """RED: Fails if missing data keys aren't handled properly."""
        pairer = PatchPairer()

        # Create chunk with missing required keys
        incomplete_data = {
            "block_types": np.ones((16, 16, 384), dtype=np.uint8),
            # Missing air_mask, biomes, heightmap, etc.
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **incomplete_data)
            incomplete_file = Path(f.name)

        try:
            with pytest.raises(KeyError):  # Should raise KeyError for missing keys
                pairer.extract_parent_child_pairs(incomplete_file)
        finally:
            incomplete_file.unlink()

    def test_handle_wrong_chunk_dimensions(self):
        """RED: Fails if wrong dimension handling is missing."""
        pairer = PatchPairer()

        # Create chunk with wrong dimensions
        wrong_dims_data = {
            "block_types": np.ones((10, 10, 200), dtype=np.uint8),  # Wrong dims
            "air_mask": np.ones((10, 10, 200), dtype=bool),
            "biomes": np.ones((10, 10), dtype=np.uint8),
            "heightmap": np.ones((10, 10), dtype=np.uint16),
            "chunk_x": np.int32(0),
            "chunk_z": np.int32(0),
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **wrong_dims_data)
            wrong_dims_file = Path(f.name)

        try:
            with pytest.raises(ValueError):  # Should raise ValueError for wrong dims
                pairer.extract_parent_child_pairs(wrong_dims_file)
        finally:
            wrong_dims_file.unlink()
