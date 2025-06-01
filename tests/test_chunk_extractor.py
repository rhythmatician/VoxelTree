"""
Tests for Minecraft .mca to .npz chunk extraction pipeline.
Phase 0C: TDD RED phase - All tests should fail initially.

This module tests the ChunkExtractor class that converts Minecraft region files
into compressed numpy training data.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from worldgen.config import load_config
from extraction.chunk_extractor import ChunkExtractor


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
                "validation": {
                    "verify_checksums": True,
                    "detect_corruption": True
                },
                "block_mapping": {
                    "air_blocks": [0],
                    "solid_blocks": [1, 2, 3]
                },
                "heightmap": {
                    "surface_blocks": [2, 3, 4],
                    "min_height": -64,
                    "max_height": 320
                }
            }
        }
          # Create mock config file
        self.test_config_path = self.test_temp_dir / "test_config.yaml"
        
        # Write test config to file
        import yaml
        with open(self.test_config_path, 'w') as f:
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
        mock_chunk_data.blocks = np.random.randint(0, 10, size=(16, 384, 16))  # Minecraft YZX format
        
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
        assert np.all(biomes >= 0)
        assert np.all(biomes <= 255)
    
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
        assert heightmap[8, 8] == 219  # Top of stone column
        assert heightmap[4, 4] == 189  # Top of dirt column
        assert heightmap[0, 0] == 0    # No surface blocks = ground level
    
    def test_npz_output_format(self):
        """Test .npz file structure and compression."""
        # Create test chunk data
        test_data = {
            "block_types": np.random.randint(0, 10, size=(16, 16, 384), dtype=np.uint8),
            "air_mask": np.random.choice([True, False], size=(16, 16, 384)),
            "biomes": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
            "heightmap": np.random.randint(0, 320, size=(16, 16), dtype=np.uint16),            "chunk_x": np.int32(5),
            "chunk_z": np.int32(10),
            "region_file": "r.0.0.mca"  # Use regular string instead of np.string_
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
    
    @patch('multiprocessing.Pool')
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
        mock_pool_instance.map.return_value = [["chunk_0_0.npz"], ["chunk_1_0.npz"], ["chunk_2_0.npz"]]
        
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
            "region_file": "r.0.0.mca"  # Use regular string
        }
          # Create valid file
        valid_path = self.extractor.output_dir / "chunk_0_0.npz"
        np.savez_compressed(valid_path, **valid_data)
        
        # Create corrupted file (missing fields)
        corrupted_data = {"block_types": np.zeros((16, 16, 384), dtype=np.uint8)}
        corrupted_path = self.extractor.output_dir / "chunk_1_1.npz"
        np.savez_compressed(corrupted_path, **corrupted_data)
        
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
        assert extraction_config["max_disk_usage_gb"] > 0
        
        # Block mappings should be valid lists
        assert isinstance(extraction_config["block_mapping"]["air_blocks"], list)
        assert isinstance(extraction_config["block_mapping"]["solid_blocks"], list)
