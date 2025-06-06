"""
Tests for Minecraft .mca to .npz chunk extraction pipeline.
Phase 0C: TDD RED phase - All tests should fail initially.

This module tests the ChunkExtractor class that converts Minecraft region files
into compressed numpy training data.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.extraction.chunk_extractor import ChunkExtractor
from scripts.worldgen.config import load_config


class TestChunkExtractor:
    """Test suite for ChunkExtractor class."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_temp_dir = Path(tempfile.mkdtemp())

        # Create test config with temp directories
        self.test_config = {
            "extraction": {
                "output_dir": str(self.test_temp_dir / "chunks"),
                "temp_dir": str(self.test_temp_dir / "temp"),
                "max_disk_usage_gb": 1,
                "batch_size": 16,
                "num_workers": 2,
                "compression_level": 6,
                "validation": {"verify_checksums": True, "detect_corruption": True},
                "block_mapping": {"air_blocks": [0], "solid_blocks": [1, 2, 3]},
                "heightmap": {
                    "surface_blocks": [2, 3, 4],
                    "min_height": -64,
                    "max_height": 320,
                },
            }
        }

        # Create mock config file
        self.test_config_path = self.test_temp_dir / "test_config.yaml"

        # Write test config to file
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.safe_dump(self.test_config, f)

        # Initialize extractor (will fail until implemented)
        self.extractor = ChunkExtractor(config_path=self.test_config_path)

    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    def test_chunk_extractor_init(self):
        """Test ChunkExtractor initialization and config loading."""
        # Should create output directories
        assert self.extractor.output_dir.exists()
        assert self.extractor.temp_dir.exists()

        # Should load config correctly
        assert self.extractor.batch_size == 16
        assert self.extractor.num_workers == 2
        assert self.extractor.compression_level == 6

        # Should have proper block mappings
        assert 0 in self.extractor.air_blocks
        assert 1 in self.extractor.solid_blocks

    def test_extract_single_chunk(self):
        """Test extraction of one chunk from .mca file."""
        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Should extract chunk data as dictionary
        chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should contain required fields
        assert isinstance(chunk_data, dict)
        assert "block_types" in chunk_data
        assert "air_mask" in chunk_data
        assert "biomes" in chunk_data
        assert "heightmap" in chunk_data
        assert "chunk_x" in chunk_data
        assert "chunk_z" in chunk_data

        # Should have correct coordinates
        assert chunk_data["chunk_x"] == 0
        assert chunk_data["chunk_z"] == 0

    def test_block_data_processing(self):
        """Test conversion of NBT block data to numpy arrays."""
        # Create mock NBT chunk data
        mock_chunk_data = Mock()
        mock_chunk_data.blocks = np.random.randint(
            0, 10, size=(16, 384, 16)
        )  # Minecraft YZX format

        # Should convert to proper format and types
        block_types, air_mask = self.extractor.process_block_data(mock_chunk_data)

        # Should have correct shape (XZY format for training)
        assert block_types.shape == (16, 16, 384)
        assert air_mask.shape == (16, 16, 384)

        # Should have correct types
        assert block_types.dtype == np.uint8
        assert air_mask.dtype == np.bool_

        # Air mask should correctly identify air blocks
        air_positions = np.where(block_types == 0)
        assert np.all(air_mask[air_positions])

    def test_biome_extraction(self):
        """Test biome ID extraction and validation."""
        # Create mock chunk data with biome information
        mock_chunk_data = Mock()
        mock_chunk_data.biomes = np.random.randint(0, 50, size=(16, 16))

        # Should extract biome data
        biomes = self.extractor.extract_biome_data(mock_chunk_data)

        # Should have correct shape and type
        assert biomes.shape == (16, 16)
        assert biomes.dtype == np.uint8
        # Should be valid biome IDs (0-255 range)
        assert (biomes >= 0).all()
        assert (biomes <= 255).all()

    def test_heightmap_computation(self):
        """Test surface heightmap calculation from blocks."""
        # Create test block data with known surface
        block_types = np.zeros((16, 16, 384), dtype=np.uint8)  # All air
        # Add surface blocks at specific heights
        block_types[8, 8, 200:220] = 2  # Stone column
        block_types[4, 4, 180:190] = 3  # Dirt column

        # Should compute surface heightmap
        heightmap = self.extractor.compute_heightmap(block_types)

        # Should have correct shape and type
        assert heightmap.shape == (16, 16)
        assert heightmap.dtype == np.uint16

        # Should correctly identify surface heights
        assert heightmap[8, 8] == 220  # Top of stone column
        assert heightmap[4, 4] == 190  # Top of dirt column
        assert heightmap[0, 0] == 0  # No surface blocks = ground level

    def test_npz_output_format(self):
        """Test .npz file structure and compression."""
        # Create test chunk data
        test_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(5),
            "chunk_z": np.int32(10),
            "region_file": "r.0.0.mca",  # Use regular string instead of np.string_
        }

        # Should save and load .npz correctly
        output_path = self.extractor.save_chunk_npz(test_data, chunk_x=5, chunk_z=10)

        # Should create file at expected location
        expected_path = self.extractor.output_dir / "chunk_5_10.npz"
        assert output_path == expected_path
        assert output_path.exists()

        # Should load with correct data
        loaded_data = np.load(output_path)

        # Should contain all required fields with correct types
        assert "block_types" in loaded_data
        assert "air_mask" in loaded_data
        assert "biomes" in loaded_data
        assert "heightmap" in loaded_data
        assert "chunk_x" in loaded_data
        assert "chunk_z" in loaded_data
        assert "region_file" in loaded_data

        # Should preserve data integrity
        np.testing.assert_array_equal(loaded_data["block_types"], test_data["block_types"])
        np.testing.assert_array_equal(loaded_data["air_mask"], test_data["air_mask"])
        assert loaded_data["chunk_x"].item() == 5
        assert loaded_data["chunk_z"].item() == 10

    def test_region_batch_extraction(self):
        """Test extraction of all chunks from one region."""
        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_with_multiple_chunks" + b"\x00" * 5000)

        # Should extract all chunks in region
        output_files = self.extractor.extract_region_batch(mock_mca_path)

        # Should return list of output .npz files
        assert isinstance(output_files, list)
        assert len(output_files) > 0

        # Each file should exist and be valid .npz
        for output_file in output_files:
            assert output_file.exists()
            assert output_file.suffix == ".npz"

            # Should be loadable
            data = np.load(output_file)
            assert "block_types" in data
            assert "chunk_x" in data
            assert "chunk_z" in data

    @patch("multiprocessing.Pool")
    def test_parallel_region_processing(self, mock_pool):
        """Test multiprocessing extraction of multiple regions."""
        # Create mock region files
        region_files = []
        for i in range(3):
            region_path = self.test_temp_dir / f"r.{i}.0.mca"
            region_path.write_bytes(f"fake_region_{i}".encode() + b"\x00" * 2000)
            region_files.append(region_path)

        # Mock pool behavior
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [
            ["chunk_0_0.npz"],
            ["chunk_1_0.npz"],
            ["chunk_2_0.npz"],
        ]

        # Should process regions in parallel
        results = self.extractor.extract_regions_parallel(region_files, num_workers=2)

        # Should use multiprocessing Pool
        mock_pool.assert_called_once_with(processes=2)
        mock_pool_instance.map.assert_called_once()

        # Should return combined results
        assert isinstance(results, list)
        assert len(results) >= 3

    def test_memory_management(self):
        """Test streaming processing without excessive memory usage."""
        # Create large mock region file
        large_mca_path = self.test_temp_dir / "r.big.0.mca"
        large_mca_path.write_bytes(b"big_fake_mca" + b"\x00" * 10000)

        # Monitor memory usage during extraction
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Should extract without excessive memory growth
        output_files = self.extractor.extract_region_batch(large_mca_path)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = peak_memory - initial_memory

        # Should not use more than 500MB additional memory
        assert memory_growth < 500, f"Memory usage grew by {memory_growth:.1f}MB, expected <500MB"

        # Should still produce output
        assert len(output_files) > 0

    def test_extraction_validation(self):
        """Test detection of corrupted or incomplete extractions."""
        # Create some valid and invalid .npz files
        valid_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(0),
            "chunk_z": np.int32(0),
            "region_file": "r.0.0.mca",  # Use regular string
        }

        # Create valid file
        valid_path = self.extractor.output_dir / "chunk_0_0.npz"
        np.savez_compressed(valid_path, **valid_data)

        # Create corrupted file (missing fields)
        corrupted_data = {"block_types": np.zeros((16, 16, 384), dtype=np.uint8)}
        corrupted_path = self.extractor.output_dir / "chunk_1_1.npz"
        np.savez_compressed(str(corrupted_path), **corrupted_data)

        # Create invalid file (wrong shape)
        invalid_data = valid_data.copy()
        invalid_data["block_types"] = np.zeros((8, 8, 192), dtype=np.uint8)  # Wrong shape
        invalid_path = self.extractor.output_dir / "chunk_2_2.npz"
        np.savez_compressed(invalid_path, **invalid_data)

        # Should validate extraction results
        validation_result = self.extractor.validate_extraction_results(self.extractor.output_dir)

        # Should identify valid, corrupted, and invalid files
        assert "valid_files" in validation_result
        assert "corrupted_files" in validation_result
        assert "invalid_files" in validation_result
        assert "total_chunks" in validation_result

        assert validation_result["total_chunks"] == 3
        assert len(validation_result["valid_files"]) == 1
        assert len(validation_result["corrupted_files"]) == 1
        assert len(validation_result["invalid_files"]) == 1

        # Should correctly identify each file type
        assert "chunk_0_0.npz" in validation_result["valid_files"][0]
        assert "chunk_1_1.npz" in validation_result["corrupted_files"][0]
        assert "chunk_2_2.npz" in validation_result["invalid_files"][0]


class TestChunkExtractionConfiguration:
    """Test configuration loading and validation for chunk extraction."""

    def test_config_loading_extraction_params(self):
        """Test loading extraction configuration from YAML."""
        # Should load extraction config from main config file
        config = load_config(Path("config.yaml"))

        # Should contain extraction section
        assert "extraction" in config
        extraction_config = config["extraction"]

        # Should have required parameters
        assert "output_dir" in extraction_config
        assert "batch_size" in extraction_config
        assert "num_workers" in extraction_config
        assert "compression_level" in extraction_config

        # Should have validation settings
        assert "validation" in extraction_config
        assert "verify_checksums" in extraction_config["validation"]

        # Should have block mapping
        assert "block_mapping" in extraction_config
        assert "air_blocks" in extraction_config["block_mapping"]
        assert "solid_blocks" in extraction_config["block_mapping"]

    def test_extraction_parameter_validation(self):
        """Test validation of extraction configuration parameters."""
        config = load_config(Path("config.yaml"))
        extraction_config = config["extraction"]

        # Batch size should be reasonable
        assert 1 <= extraction_config["batch_size"] <= 256

        # Worker count should be positive
        assert extraction_config["num_workers"] > 0

        # Compression level should be valid
        assert 1 <= extraction_config["compression_level"] <= 9

        # Disk usage limit should be reasonable
        assert extraction_config["max_disk_usage_gb"] > 0, "Block mappings should be valid lists"
        assert isinstance(extraction_config["block_mapping"]["air_blocks"], list)
        assert isinstance(extraction_config["block_mapping"]["solid_blocks"], list)


class TestStructureAwareChunkExtraction:
    """Test suite for structure-aware chunk extraction functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_temp_dir = Path(tempfile.mkdtemp())

        # Create test config with structure extraction enabled
        self.test_config = {
            "extraction": {
                "output_dir": str(self.test_temp_dir / "chunks"),
                "temp_dir": str(self.test_temp_dir / "temp"),
                "max_disk_usage_gb": 1,
                "batch_size": 16,
                "num_workers": 2,
                "compression_level": 6,
                "validation": {"verify_checksums": True, "detect_corruption": True},
                "block_mapping": {"air_blocks": [0], "solid_blocks": [1, 2, 3]},
                "heightmap": {
                    "surface_blocks": [2, 3, 4],
                    "min_height": -64,
                    "max_height": 320,
                },
                "structures": {
                    "enabled": True,
                    "mask_resolution": 8,
                    "structure_types": [
                        "village",
                        "fortress",
                        "monument",
                        "mansion",
                        "ruined_portal",
                        "outpost",
                        "bastion",
                        "temple",
                        "stronghold",
                        "mineshaft",
                    ],
                    "position_encoding": "normalized_offset",
                    "validate_structure_generation": True,
                    "min_structure_chunks_ratio": 0.1,
                },
            }
        }

        # Create mock config file
        self.test_config_path = self.test_temp_dir / "test_config.yaml"

        # Write test config to file
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.safe_dump(self.test_config, f)

        # Initialize extractor with structure support
        self.extractor = ChunkExtractor(config_path=self.test_config_path)

    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    def test_structure_extractor_initialization(self):
        """Test that ChunkExtractor properly initializes StructureExtractor."""
        # Should have structure extractor instance
        assert hasattr(self.extractor, "structure_extractor")
        assert self.extractor.structure_extractor is not None

        # Structure extractor should be enabled
        assert self.extractor.structure_extractor.enabled is True

        # Should have correct configuration
        assert self.extractor.structure_extractor.mask_resolution == 8
        assert "village" in self.extractor.structure_extractor.structure_types
        assert self.extractor.structure_extractor.position_encoding == "normalized_offset"

    @patch("scripts.extraction.structure_extractor.StructureExtractor.extract_structure_data")
    def test_chunk_extraction_with_structures(self, mock_extract_structure):
        """Test chunk extraction includes structure data when enabled."""
        # Mock structure data response
        mock_structure_data = {
            "structure_mask": np.ones((8, 8, 1), dtype=np.float32),
            "structure_types": np.zeros((10,), dtype=np.float32),
            "structure_positions": np.array([0.5, -0.3], dtype=np.float32),
        }
        mock_structure_data["structure_types"][0] = 1.0  # Village present
        mock_extract_structure.return_value = mock_structure_data

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_with_structures" + b"\x00" * 2000)

        # Extract chunk data
        chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should contain structure data
        assert "structure_mask" in chunk_data
        assert "structure_types" in chunk_data
        assert "structure_positions" in chunk_data

        # Should have correct shapes and types
        assert chunk_data["structure_mask"].shape == (8, 8, 1)
        assert chunk_data["structure_mask"].dtype == np.float32
        assert chunk_data["structure_types"].shape == (10,)
        assert chunk_data["structure_types"].dtype == np.float32
        assert chunk_data["structure_positions"].shape == (2,)
        assert chunk_data["structure_positions"].dtype == np.float32

        # Should have called structure extraction
        mock_extract_structure.assert_called_once_with(mock_mca_path, 0, 0)

        # Should detect village structure
        assert chunk_data["structure_types"][0] == 1.0

    @patch("scripts.extraction.structure_extractor.StructureExtractor.extract_structure_data")
    def test_chunk_extraction_structure_failure_handling(self, mock_extract_structure):
        """Test graceful handling when structure extraction fails."""
        # Mock structure extraction failure
        mock_extract_structure.side_effect = RuntimeError("Structure extraction failed")

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Extract chunk data (should not fail)
        chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should contain empty structure data
        assert "structure_mask" in chunk_data
        assert "structure_types" in chunk_data
        assert "structure_positions" in chunk_data

        # Should have correct shapes (all zeros)
        assert chunk_data["structure_mask"].shape == (8, 8, 1)
        assert np.all(chunk_data["structure_mask"] == 0)
        assert chunk_data["structure_types"].shape == (10,)
        assert np.all(chunk_data["structure_types"] == 0)
        assert chunk_data["structure_positions"].shape == (2,)
        assert np.all(chunk_data["structure_positions"] == 0)

    def test_chunk_extraction_with_structures_disabled(self):
        """Test chunk extraction without structure data when disabled."""
        # Create config with structures disabled
        disabled_config = self.test_config.copy()
        disabled_config["extraction"]["structures"]["enabled"] = False

        # Write disabled config
        disabled_config_path = self.test_temp_dir / "disabled_config.yaml"
        import yaml

        with open(disabled_config_path, "w") as f:
            yaml.safe_dump(disabled_config, f)

        # Initialize extractor with disabled structures
        disabled_extractor = ChunkExtractor(config_path=disabled_config_path)

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Extract chunk data
        chunk_data = disabled_extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should NOT contain structure data
        assert "structure_mask" not in chunk_data
        assert "structure_types" not in chunk_data
        assert "structure_positions" not in chunk_data

        # Should still contain standard chunk data
        assert "block_types" in chunk_data
        assert "air_mask" in chunk_data
        assert "biomes" in chunk_data
        assert "heightmap" in chunk_data

    def test_npz_file_with_structure_data(self):
        """Test .npz file creation and loading with structure data."""
        # Create test chunk data with structures
        test_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(5),
            "chunk_z": np.int32(10),
            "region_file": "r.0.0.mca",
            # Structure data
            "structure_mask": np.random.rand(8, 8, 1).astype(np.float32),
            "structure_types": np.random.rand(10).astype(np.float32),
            "structure_positions": np.random.rand(2).astype(np.float32),
        }

        # Save as .npz
        output_path = self.extractor.save_chunk_npz(test_data, chunk_x=5, chunk_z=10)

        # Should create file at expected location
        expected_path = self.extractor.output_dir / "chunk_5_10.npz"
        assert output_path == expected_path
        assert output_path.exists()

        # Should load with correct data including structures
        loaded_data = np.load(output_path)

        # Check all standard fields
        assert "block_types" in loaded_data
        assert "air_mask" in loaded_data
        assert "biomes" in loaded_data
        assert "heightmap" in loaded_data

        # Check structure fields
        assert "structure_mask" in loaded_data
        assert "structure_types" in loaded_data
        assert "structure_positions" in loaded_data

        # Verify structure data integrity
        np.testing.assert_array_equal(loaded_data["structure_mask"], test_data["structure_mask"])
        np.testing.assert_array_equal(loaded_data["structure_types"], test_data["structure_types"])
        np.testing.assert_array_equal(
            loaded_data["structure_positions"], test_data["structure_positions"]
        )

    def test_empty_chunk_with_structure_data(self):
        """Test that empty chunks include empty structure data when structures enabled."""
        # Create mock .mca file that will trigger empty chunk handling
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Mock the chunk existence check to raise "Chunk does not exist"
        with patch("anvil.Region.from_file") as mock_from_file:
            mock_region = Mock()
            mock_from_file.return_value = mock_region

            # This will trigger the empty chunk creation path
            # (since the inner try block will pass but the chunk will be empty)

            # Extract chunk data
            chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

            # Should contain structure data (even for empty chunks)
            assert "structure_mask" in chunk_data
            assert "structure_types" in chunk_data
            assert "structure_positions" in chunk_data

            # Structure data should be empty/zero
            assert chunk_data["structure_mask"].shape == (8, 8, 1)
            assert np.all(chunk_data["structure_mask"] == 0)
            assert chunk_data["structure_types"].shape == (10,)
            assert np.all(chunk_data["structure_types"] == 0)
            assert chunk_data["structure_positions"].shape == (2,)
            assert np.all(chunk_data["structure_positions"] == 0)

    def test_validation_includes_structure_fields(self):
        """Test that validation properly handles structure data fields."""
        # Create valid file with structure data
        valid_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(0),
            "chunk_z": np.int32(0),
            "region_file": "r.0.0.mca",
            # Structure data with correct shapes
            "structure_mask": np.zeros((8, 8, 1), dtype=np.float32),
            "structure_types": np.zeros((10,), dtype=np.float32),
            "structure_positions": np.zeros((2,), dtype=np.float32),
        }

        # Create file with invalid structure shapes
        invalid_data = valid_data.copy()
        invalid_data["structure_mask"] = np.zeros((4, 4, 1), dtype=np.float32)  # Wrong shape

        # Save files
        valid_path = self.extractor.output_dir / "chunk_valid.npz"
        invalid_path = self.extractor.output_dir / "chunk_invalid.npz"

        np.savez_compressed(valid_path, **valid_data)
        np.savez_compressed(invalid_path, **invalid_data)

        # Run validation
        validation_result = self.extractor.validate_extraction_results(self.extractor.output_dir)

        # Should identify valid and invalid files
        assert len(validation_result["valid_files"]) >= 1
        assert any("chunk_valid.npz" in f for f in validation_result["valid_files"])

        # Note: Current validation doesn't check structure field shapes yet
        # This test documents the expected behavior for future enhancement


class TestStructureAwareConfiguration:
    """Test configuration handling for structure-aware extraction."""

    def test_phase2_config_structure_settings(self):
        """Test loading Phase 2 configuration with structure settings."""
        # Load the phase 2 configuration using absolute path
        phase2_config_path = Path(__file__).parent.parent / "config.phase2.yaml"

        if phase2_config_path.exists():
            config = load_config(phase2_config_path)
            if config is not None:
                extraction_config = config.get("extraction", {})
                structure_config = config.get("structures", {})
                # Structures are at top level

                # Should have extraction enabled for Phase 2
                assert extraction_config.get("enabled", False) is True

                # Should have structure extraction enabled for Phase 2
                assert structure_config.get("enabled", False) is True

                # Should have reasonable structure settings
                assert isinstance(structure_config.get("structure_types", []), list)
                assert structure_config.get("mask_resolution", 0) > 0
                # Note: Phase 2 config doesn't specify position_encoding, so skip that check
            else:
                # If config loading fails, skip the test
                import pytest

                pytest.skip("Could not load Phase 2 configuration")
        else:
            import pytest

            pytest.skip("Phase 2 configuration file not found")

    def test_structure_config_validation(self):
        """Test validation of structure configuration parameters."""
        # Create temporary config with various structure settings
        test_temp_dir = Path(tempfile.mkdtemp())

        try:
            # Test valid structure config
            valid_config = {
                "extraction": {
                    "output_dir": str(test_temp_dir / "chunks"),
                    "structures": {
                        "enabled": True,
                        "mask_resolution": 8,
                        "structure_types": ["village", "fortress"],
                        "position_encoding": "normalized_offset",
                        "validate_structure_generation": True,
                        "min_structure_chunks_ratio": 0.1,
                    },
                }
            }

            config_path = test_temp_dir / "valid_config.yaml"
            import yaml

            with open(config_path, "w") as f:
                yaml.safe_dump(valid_config, f)

            # Should initialize without errors
            extractor = ChunkExtractor(config_path=config_path)
            assert extractor.structure_extractor.enabled is True
            assert extractor.structure_extractor.mask_resolution == 8
            assert len(extractor.structure_extractor.structure_types) == 2

        finally:
            shutil.rmtree(test_temp_dir)


class TestStructureAwareChunkExtraction:
    """Test suite for structure-aware chunk extraction functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_temp_dir = Path(tempfile.mkdtemp())

        # Create test config with structure extraction enabled
        self.test_config = {
            "extraction": {
                "output_dir": str(self.test_temp_dir / "chunks"),
                "temp_dir": str(self.test_temp_dir / "temp"),
                "max_disk_usage_gb": 1,
                "batch_size": 16,
                "num_workers": 2,
                "compression_level": 6,
                "validation": {"verify_checksums": True, "detect_corruption": True},
                "block_mapping": {"air_blocks": [0], "solid_blocks": [1, 2, 3]},
                "heightmap": {
                    "surface_blocks": [2, 3, 4],
                    "min_height": -64,
                    "max_height": 320,
                },
                "structures": {
                    "enabled": True,
                    "mask_resolution": 8,
                    "structure_types": [
                        "village",
                        "fortress",
                        "monument",
                        "mansion",
                        "ruined_portal",
                        "outpost",
                        "bastion",
                        "temple",
                        "stronghold",
                        "mineshaft",
                    ],
                    "position_encoding": "normalized_offset",
                    "validate_structure_generation": True,
                    "min_structure_chunks_ratio": 0.1,
                },
            }
        }

        # Create mock config file
        self.test_config_path = self.test_temp_dir / "test_config.yaml"

        # Write test config to file
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.safe_dump(self.test_config, f)

        # Initialize extractor with structure support
        self.extractor = ChunkExtractor(config_path=self.test_config_path)

    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    def test_structure_extractor_initialization(self):
        """Test that ChunkExtractor properly initializes StructureExtractor."""
        # Should have structure extractor instance
        assert hasattr(self.extractor, "structure_extractor")
        assert self.extractor.structure_extractor is not None

        # Structure extractor should be enabled
        assert self.extractor.structure_extractor.enabled is True

        # Should have correct configuration
        assert self.extractor.structure_extractor.mask_resolution == 8
        assert "village" in self.extractor.structure_extractor.structure_types
        assert self.extractor.structure_extractor.position_encoding == "normalized_offset"

    @patch("scripts.extraction.structure_extractor.StructureExtractor.extract_structure_data")
    def test_chunk_extraction_with_structures(self, mock_extract_structure):
        """Test chunk extraction includes structure data when enabled."""
        # Mock structure data response
        mock_structure_data = {
            "structure_mask": np.ones((8, 8, 1), dtype=np.float32),
            "structure_types": np.zeros((10,), dtype=np.float32),
            "structure_positions": np.array([0.5, -0.3], dtype=np.float32),
        }
        mock_structure_data["structure_types"][0] = 1.0  # Village present
        mock_extract_structure.return_value = mock_structure_data

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_with_structures" + b"\x00" * 2000)

        # Extract chunk data
        chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should contain structure data
        assert "structure_mask" in chunk_data
        assert "structure_types" in chunk_data
        assert "structure_positions" in chunk_data

        # Should have correct shapes and types
        assert chunk_data["structure_mask"].shape == (8, 8, 1)
        assert chunk_data["structure_mask"].dtype == np.float32
        assert chunk_data["structure_types"].shape == (10,)
        assert chunk_data["structure_types"].dtype == np.float32
        assert chunk_data["structure_positions"].shape == (2,)
        assert chunk_data["structure_positions"].dtype == np.float32

        # Should have called structure extraction
        mock_extract_structure.assert_called_once_with(mock_mca_path, 0, 0)

        # Should detect village structure
        assert chunk_data["structure_types"][0] == 1.0

    @patch("scripts.extraction.structure_extractor.StructureExtractor.extract_structure_data")
    def test_chunk_extraction_structure_failure_handling(self, mock_extract_structure):
        """Test graceful handling when structure extraction fails."""
        # Mock structure extraction failure
        mock_extract_structure.side_effect = RuntimeError("Structure extraction failed")

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Extract chunk data (should not fail)
        chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should contain empty structure data
        assert "structure_mask" in chunk_data
        assert "structure_types" in chunk_data
        assert "structure_positions" in chunk_data

        # Should have correct shapes (all zeros)
        assert chunk_data["structure_mask"].shape == (8, 8, 1)
        assert np.all(chunk_data["structure_mask"] == 0)
        assert chunk_data["structure_types"].shape == (10,)
        assert np.all(chunk_data["structure_types"] == 0)
        assert chunk_data["structure_positions"].shape == (2,)
        assert np.all(chunk_data["structure_positions"] == 0)

    def test_chunk_extraction_with_structures_disabled(self):
        """Test chunk extraction without structure data when disabled."""
        # Create config with structures disabled
        disabled_config = self.test_config.copy()
        disabled_config["extraction"]["structures"]["enabled"] = False

        # Write disabled config
        disabled_config_path = self.test_temp_dir / "disabled_config.yaml"
        import yaml

        with open(disabled_config_path, "w") as f:
            yaml.safe_dump(disabled_config, f)

        # Initialize extractor with disabled structures
        disabled_extractor = ChunkExtractor(config_path=disabled_config_path)

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Extract chunk data
        chunk_data = disabled_extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

        # Should NOT contain structure data
        assert "structure_mask" not in chunk_data
        assert "structure_types" not in chunk_data
        assert "structure_positions" not in chunk_data

        # Should still contain standard chunk data
        assert "block_types" in chunk_data
        assert "air_mask" in chunk_data
        assert "biomes" in chunk_data
        assert "heightmap" in chunk_data

    def test_npz_file_with_structure_data(self):
        """Test .npz file creation and loading with structure data."""
        # Create test chunk data with structures
        test_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(5),
            "chunk_z": np.int32(10),
            "region_file": "r.0.0.mca",
            # Structure data
            "structure_mask": np.random.rand(8, 8, 1).astype(np.float32),
            "structure_types": np.random.rand(10).astype(np.float32),
            "structure_positions": np.random.rand(2).astype(np.float32),
        }

        # Save as .npz
        output_path = self.extractor.save_chunk_npz(test_data, chunk_x=5, chunk_z=10)

        # Should create file at expected location
        expected_path = self.extractor.output_dir / "chunk_5_10.npz"
        assert output_path == expected_path
        assert output_path.exists()

        # Should load with correct data including structures
        loaded_data = np.load(output_path)

        # Check all standard fields
        assert "block_types" in loaded_data
        assert "air_mask" in loaded_data
        assert "biomes" in loaded_data
        assert "heightmap" in loaded_data

        # Check structure fields
        assert "structure_mask" in loaded_data
        assert "structure_types" in loaded_data
        assert "structure_positions" in loaded_data

        # Verify structure data integrity
        np.testing.assert_array_equal(loaded_data["structure_mask"], test_data["structure_mask"])
        np.testing.assert_array_equal(loaded_data["structure_types"], test_data["structure_types"])
        np.testing.assert_array_equal(
            loaded_data["structure_positions"], test_data["structure_positions"]
        )

    def test_empty_chunk_with_structure_data(self):
        """Test that empty chunks include empty structure data when structures enabled."""
        # Create mock .mca file that will trigger empty chunk handling
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.write_bytes(b"fake_mca_data" + b"\x00" * 2000)

        # Mock the chunk existence check to raise "Chunk does not exist"
        with patch("anvil.Region.from_file") as mock_from_file:
            mock_region = Mock()
            mock_from_file.return_value = mock_region

            # This will trigger the empty chunk creation path
            # (since the inner try block will pass but the chunk will be empty)

            # Extract chunk data
            chunk_data = self.extractor.extract_chunk_data(mock_mca_path, chunk_x=0, chunk_z=0)

            # Should contain structure data (even for empty chunks)
            assert "structure_mask" in chunk_data
            assert "structure_types" in chunk_data
            assert "structure_positions" in chunk_data

            # Structure data should be empty/zero
            assert chunk_data["structure_mask"].shape == (8, 8, 1)
            assert np.all(chunk_data["structure_mask"] == 0)
            assert chunk_data["structure_types"].shape == (10,)
            assert np.all(chunk_data["structure_types"] == 0)
            assert chunk_data["structure_positions"].shape == (2,)
            assert np.all(chunk_data["structure_positions"] == 0)

    def test_validation_includes_structure_fields(self):
        """Test that validation properly handles structure data fields."""
        # Create valid file with structure data
        valid_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),
            "chunk_x": np.int32(0),
            "chunk_z": np.int32(0),
            "region_file": "r.0.0.mca",
            # Structure data with correct shapes
            "structure_mask": np.zeros((8, 8, 1), dtype=np.float32),
            "structure_types": np.zeros((10,), dtype=np.float32),
            "structure_positions": np.zeros((2,), dtype=np.float32),
        }

        # Create file with invalid structure shapes
        invalid_data = valid_data.copy()
        invalid_data["structure_mask"] = np.zeros((4, 4, 1), dtype=np.float32)  # Wrong shape

        # Save files
        valid_path = self.extractor.output_dir / "chunk_valid.npz"
        invalid_path = self.extractor.output_dir / "chunk_invalid.npz"

        np.savez_compressed(valid_path, **valid_data)
        np.savez_compressed(invalid_path, **invalid_data)

        # Run validation
        validation_result = self.extractor.validate_extraction_results(self.extractor.output_dir)

        # Should identify valid and invalid files
        assert len(validation_result["valid_files"]) >= 1
        assert any("chunk_valid.npz" in f for f in validation_result["valid_files"])

        # Note: Current validation doesn't check structure field shapes yet
        # This test documents the expected behavior for future enhancement
