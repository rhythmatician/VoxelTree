"""
Test suite for Phase 2.2: Link with seed-derived input

RED phase tests for linking LOD pairs with seed-derived conditioning variables
(biomes, heightmaps, river noise) to create complete training examples.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the classes we'll be testing (these will be extended)
from scripts.pairing.seed_input_linker import \
    SeedInputLinker  # This doesn't exist yet


class TestSeedInputLinker:
    """Test linking LOD pairs with seed-derived conditioning variables."""

    def test_seed_input_linker_init(self):
        """RED: Fails because SeedInputLinker doesn't exist yet."""
        config_path = Path("config.yaml")
        linker = SeedInputLinker(config_path=config_path)

        assert linker.config_path == config_path
        assert hasattr(linker, "seed_inputs_dir")
        assert hasattr(linker, "pairs_dir")
        assert hasattr(linker, "output_dir")
        assert hasattr(linker, "seed")

    def test_load_seed_input_patch(self):
        """RED: Fails if seed input loading is incorrect."""
        linker = SeedInputLinker()

        # Should load biome and heightmap data for given coordinates
        chunk_x, chunk_z = 10, 15
        seed_inputs = linker.load_seed_input_patch(chunk_x, chunk_z)

        assert "biomes" in seed_inputs
        assert "heightmap" in seed_inputs
        assert "river_noise" in seed_inputs
        assert seed_inputs["biomes"].shape == (16, 16)
        assert seed_inputs["heightmap"].shape == (16, 16)
        assert seed_inputs["river_noise"].shape == (16, 16)


class TestBiomeHeightmapPairing:
    """Test correct pairing of biome and heightmap data with LOD pairs."""

    @pytest.fixture
    def temp_paired_data(self):
        """Create temporary directory with mock paired data."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create mock LOD pairs
        pairs_dir = temp_dir / "pairs"
        pairs_dir.mkdir()

        pair_data = {
            "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
            "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
            "target_types": np.random.randint(0, 5, size=(16, 16, 16), dtype=np.uint8),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }
        np.savez_compressed(pairs_dir / "pair_chunk_10_15_y5.npz", **pair_data)

        # Create mock seed inputs
        seed_inputs_dir = temp_dir / "seed_inputs"
        seed_inputs_dir.mkdir()

        seed_data = {
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
            "river_noise": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
            "chunk_x": 10,
            "chunk_z": 15,
        }
        np.savez_compressed(seed_inputs_dir / "seed_patch_10_15.npz", **seed_data)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_link_pair_with_seed_inputs(self, temp_paired_data):
        """RED: Fails if biome/heightmap pairing is incorrect."""
        linker = SeedInputLinker()

        pairs_dir = temp_paired_data / "pairs"
        seed_inputs_dir = temp_paired_data / "seed_inputs"
        output_dir = temp_paired_data / "linked"

        pair_file = pairs_dir / "pair_chunk_10_15_y5.npz"

        linked_example = linker.link_pair_with_seed_inputs(pair_file, seed_inputs_dir, output_dir)

        # Should contain all original pair data plus seed inputs
        assert "parent_voxel" in linked_example
        assert "target_mask" in linked_example
        assert "target_types" in linked_example
        assert "biomes" in linked_example
        assert "heightmap" in linked_example
        assert "river_noise" in linked_example
        assert "y_index" in linked_example
        assert "lod" in linked_example

        # Validate coordinate consistency
        assert linked_example["chunk_x"] == 10
        assert linked_example["chunk_z"] == 15

    def test_coordinate_matching(self, temp_paired_data):
        """RED: Fails if coordinate matching between pairs and seed inputs is wrong."""
        linker = SeedInputLinker()

        pair_data = {"chunk_x": 10, "chunk_z": 15, "y_index": 5}

        seed_data = {
            "chunk_x": 10,
            "chunk_z": 15,
            "biomes": np.ones((16, 16), dtype=np.uint8),
            "heightmap": np.ones((16, 16), dtype=np.uint16),
        }

        # Should return True for matching coordinates
        assert linker.validate_coordinate_match(pair_data, seed_data) is True

        # Should return False for mismatched coordinates
        mismatched_seed = {**seed_data, "chunk_x": 20}
        assert linker.validate_coordinate_match(pair_data, mismatched_seed) is False

    def test_biome_conditioning_extraction(self):
        """RED: Fails if biome conditioning extraction is incorrect."""
        linker = SeedInputLinker()

        # Mock biome data for a 16x16 patch
        biomes = np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        )  # 4x4 for simplicity
        y_index = 5

        biome_conditioning = linker.extract_biome_conditioning(biomes, y_index)

        # Should extract relevant biome features for the y-level
        assert biome_conditioning.shape == biomes.shape  # Same spatial dimensions
        assert biome_conditioning.dtype in [np.uint8, np.int32, np.float32]

    def test_heightmap_conditioning_extraction(self):
        """RED: Fails if heightmap conditioning is incorrect."""
        linker = SeedInputLinker()

        heightmap = np.random.randint(50, 150, size=(16, 16), dtype=np.uint16)
        y_index = 5

        height_conditioning = linker.extract_height_conditioning(heightmap, y_index)

        # Should provide height-relative conditioning
        assert height_conditioning.shape == (16, 16)
        assert height_conditioning.dtype in [np.float32, np.int16]

    def test_river_noise_conditioning(self):
        """RED: Fails if river noise conditioning is incorrect."""
        linker = SeedInputLinker()

        river_noise = np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32)
        y_index = 5

        river_conditioning = linker.extract_river_conditioning(river_noise, y_index)

        # Should apply y-level specific river effects
        assert river_conditioning.shape == (16, 16)
        assert river_conditioning.dtype == np.float32
        assert np.all(river_conditioning >= -1) and np.all(river_conditioning <= 1)


class TestBatchLinking:
    """Test batch processing of linking pairs with seed inputs."""

    @pytest.fixture
    def temp_batch_data(self):
        """Create temporary directory with multiple pairs and seed inputs."""
        temp_dir = Path(tempfile.mkdtemp())

        pairs_dir = temp_dir / "pairs"
        seed_inputs_dir = temp_dir / "seed_inputs"
        pairs_dir.mkdir()
        seed_inputs_dir.mkdir()

        # Create multiple LOD pairs for different chunks and y-levels
        for chunk_x in range(2):
            for chunk_z in range(2):
                # Create seed input for this chunk
                seed_data = {
                    "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                    "heightmap": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                    "river_noise": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                    "chunk_x": chunk_x,
                    "chunk_z": chunk_z,
                }
                np.savez_compressed(
                    seed_inputs_dir / f"seed_patch_{chunk_x}_{chunk_z}.npz", **seed_data
                )

                # Create multiple y-level pairs for this chunk
                for y_index in range(3):  # Just 3 y-levels for testing
                    pair_data = {
                        "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                        "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                        "target_types": np.random.randint(0, 5, size=(16, 16, 16), dtype=np.uint8),
                        "y_index": y_index,
                        "chunk_x": chunk_x,
                        "chunk_z": chunk_z,
                        "lod": 1,
                    }
                    np.savez_compressed(
                        pairs_dir / f"pair_chunk_{chunk_x}_{chunk_z}_y{y_index}.npz",
                        **pair_data,
                    )

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_process_batch_linking(self, temp_batch_data):
        """RED: Fails if batch linking doesn't handle multiple files correctly."""
        linker = SeedInputLinker()

        pairs_dir = temp_batch_data / "pairs"
        seed_inputs_dir = temp_batch_data / "seed_inputs"
        output_dir = temp_batch_data / "linked"

        linked_count = linker.process_batch_linking(pairs_dir, seed_inputs_dir, output_dir)

        # Should process 2x2 chunks × 3 y-levels = 12 linked examples
        assert linked_count == 12
        assert output_dir.exists()

        linked_files = list(output_dir.glob("*.npz"))
        assert len(linked_files) == 12

    def test_parallel_batch_linking(self, temp_batch_data):
        """RED: Fails if parallel linking doesn't work correctly."""
        linker = SeedInputLinker()

        pairs_dir = temp_batch_data / "pairs"
        seed_inputs_dir = temp_batch_data / "seed_inputs"
        output_dir = temp_batch_data / "linked_parallel"

        linked_count = linker.process_batch_linking_parallel(
            pairs_dir, seed_inputs_dir, output_dir, num_workers=2
        )

        # Should produce same result as sequential processing
        assert linked_count == 12

        linked_files = list(output_dir.glob("*.npz"))
        assert len(linked_files) == 12


class TestLinkedOutputFormat:
    """Test the output format of linked training examples."""

    def test_linked_example_format(self):
        """RED: Fails if linked example format doesn't match specification."""
        linker = SeedInputLinker()

        # Mock input data
        pair_data = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }

        seed_data = {
            "biomes": np.ones((16, 16), dtype=np.uint8),
            "heightmap": np.ones((16, 16), dtype=np.uint16),
            "river_noise": np.ones((16, 16), dtype=np.float32),
            "chunk_x": 10,
            "chunk_z": 15,
        }

        linked_example = linker.create_linked_example(pair_data, seed_data)

        # Validate all required keys are present
        required_keys = [
            "parent_voxel",
            "target_mask",
            "target_types",
            "biome_patch",
            "heightmap_patch",
            "river_patch",
            "y_index",
            "chunk_x",
            "chunk_z",
            "lod",
        ]
        for key in required_keys:
            assert key in linked_example

        # Validate data types and shapes
        assert linked_example["parent_voxel"].shape == (8, 8, 8)
        assert linked_example["target_mask"].shape == (16, 16, 16)
        assert linked_example["target_types"].shape == (16, 16, 16)
        assert linked_example["biome_patch"].shape == (16, 16)
        assert linked_example["heightmap_patch"].shape == (16, 16)
        assert linked_example["river_patch"].shape == (16, 16)

    def test_save_linked_example_npz(self, tmp_path):
        """RED: Fails if NPZ saving of linked examples loses data integrity."""
        linker = SeedInputLinker()

        # Create mock linked example
        linked_example = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }

        output_path = tmp_path / "test_linked_example.npz"
        saved_path = linker.save_linked_example_npz(linked_example, output_path)

        # Verify file was created and data is intact
        assert saved_path.exists()

        loaded_data = np.load(saved_path)
        for key, value in linked_example.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(loaded_data[key], value)
            else:
                assert loaded_data[key] == value


class TestErrorHandlingLinked:
    """Test error handling in seed input linking."""

    def test_handle_missing_seed_input_file(self):
        """RED: Fails if missing seed input handling is incorrect."""
        linker = SeedInputLinker()

        # Try to link with non-existent seed input file
        with pytest.raises(FileNotFoundError):
            linker.load_seed_input_patch(999, 999)  # Non-existent coordinates

    def test_handle_coordinate_mismatch(self):
        """RED: Fails if coordinate mismatch handling is missing."""
        linker = SeedInputLinker()

        pair_data = {"chunk_x": 10, "chunk_z": 15}
        mismatched_seed = {"chunk_x": 20, "chunk_z": 25}

        with pytest.raises(ValueError):
            linker.validate_coordinate_match(pair_data, mismatched_seed, strict=True)

    def test_handle_malformed_seed_data(self):
        """RED: Fails if malformed seed data handling is missing."""
        linker = SeedInputLinker()

        # Create malformed seed data (missing required keys)
        malformed_seed = {
            "biomes": np.ones((16, 16), dtype=np.uint8),
            # Missing heightmap and river_noise
        }

        with pytest.raises(KeyError):
            linker.validate_seed_data_format(malformed_seed)

    def test_handle_shape_mismatch_in_seed_data(self):
        """RED: Fails if shape validation is missing."""
        linker = SeedInputLinker()

        # Create seed data with wrong shapes
        wrong_shape_seed = {
            "biomes": np.ones((10, 10), dtype=np.uint8),  # Wrong shape
            "heightmap": np.ones((16, 16), dtype=np.uint16),
            "river_noise": np.ones((16, 16), dtype=np.float32),
        }

        with pytest.raises(ValueError):
            linker.validate_seed_data_format(wrong_shape_seed)
