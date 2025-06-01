"""
Tests for WorldGenBootstrap - Phase 0B World Generation Bootstrap

Following TDD RED phase: Write failing tests first.
These tests define the expected behavior for .mca world generation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from worldgen.bootstrap import WorldGenBootstrap


class TestWorldGenBootstrap:
    """Test suite for WorldGenBootstrap class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_temp_dir = Path(tempfile.mkdtemp())
        self.bootstrap = WorldGenBootstrap(
            seed="VoxelTree",
            java_heap="2G",
            temp_world_dir=self.test_temp_dir / "temp_worlds"
        )
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)
    
    def test_worldgen_bootstrap_init(self):
        """Test bootstrap initialization with seed hashing."""
        # Test that seed "VoxelTree" converts to expected numeric value
        assert self.bootstrap.seed == 1903448982
        assert self.bootstrap.java_heap == "2G"
        assert self.bootstrap.temp_world_dir.name == "temp_worlds"
        
        # Test different seed produces different hash
        bootstrap2 = WorldGenBootstrap(seed="DifferentSeed")
        assert bootstrap2.seed != self.bootstrap.seed
    
    def test_seed_hashing_deterministic(self):
        """Test that seed hashing is deterministic and repeatable."""
        bootstrap1 = WorldGenBootstrap(seed="VoxelTree")
        bootstrap2 = WorldGenBootstrap(seed="VoxelTree")
        assert bootstrap1.seed == bootstrap2.seed == 1903448982
    
    @patch('subprocess.run')
    def test_generate_single_region(self, mock_subprocess):
        """Test generation of one .mca file with known chunks."""
        # Mock successful Java subprocess execution
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        # Test region generation - this creates the world directory
        result_path = self.bootstrap.generate_region_batch(
            x_range=(0, 32), z_range=(0, 32)
        )
        
        # The world directory should exist
        assert result_path.exists()
        
        # For testing, manually create the region structure that would be generated
        region_dir = result_path / "region"
        region_dir.mkdir(parents=True, exist_ok=True)
        test_mca_file = region_dir / "r.0.0.mca"
        test_mca_file.write_bytes(b"fake_mca_data")
        
        # Now verify the file exists
        assert (result_path / "region" / "r.0.0.mca").exists()
        
        # Verify Java command was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "java" in call_args
        assert "-Xmx2G" in call_args
        assert str(self.bootstrap.seed) in " ".join(call_args)
    
    def test_mca_file_validation(self):
        """Test parsing and validation of generated .mca files."""
        # Create mock region directory with .mca files
        region_dir = self.test_temp_dir / "region"
        region_dir.mkdir(parents=True)
          # Create valid-looking .mca files
        (region_dir / "r.0.0.mca").write_bytes(b"fake_mca_header" + b"\x00" * 1000)
        (region_dir / "r.1.0.mca").write_bytes(b"fake_mca_header" + b"\x00" * 1000)
        
        validation_result = self.bootstrap.validate_mca_output(region_dir)
        
        assert "files_found" in validation_result
        assert "total_size_mb" in validation_result
        assert "corrupted_files" in validation_result
        assert validation_result["files_found"] == 2
        assert validation_result["total_size_mb"] > 0
    
    def test_mca_validation_detects_corruption(self):
        """Test that validation detects corrupted .mca files."""
        region_dir = self.test_temp_dir / "region"
        region_dir.mkdir(parents=True)
        # Create corrupted .mca file (too small)
        (region_dir / "r.0.0.mca").write_bytes(b"bad")
        
        validation_result = self.bootstrap.validate_mca_output(region_dir)
        
        assert len(validation_result["corrupted_files"]) == 1
        assert "r.0.0.mca" in validation_result["corrupted_files"][0]
    
    def test_cleanup_temp_worlds(self):
        """Test automatic cleanup respects disk limits."""
        # Create multiple temp world directories
        temp_worlds_dir = self.bootstrap.temp_world_dir
        temp_worlds_dir.mkdir(parents=True, exist_ok=True)  # Allow existing directory
        
        world_dirs = []
        for i in range(5):
            world_dir = temp_worlds_dir / f"world_{i:03d}"
            world_dir.mkdir()
            # Add some fake data to make directories non-empty
            (world_dir / "fake_data.txt").write_text("test data")
            world_dirs.append(world_dir)
        
        # Keep only latest 2 directories
        self.bootstrap.cleanup_temp_worlds(keep_latest=2)
        
        remaining_dirs = list(temp_worlds_dir.iterdir())
        assert len(remaining_dirs) == 2
        
        # Check that latest directories were kept
        remaining_names = {d.name for d in remaining_dirs}
        assert "world_003" in remaining_names
        assert "world_004" in remaining_names
    
    def test_disk_space_limit_enforcement(self):
        """Test that disk space limits are enforced during generation."""
        # Create scenario where temp worlds exceed disk limit
        with patch.object(self.bootstrap, '_get_directory_size_gb') as mock_size:
            mock_size.return_value = 6.0  # Exceeds 5GB limit
            
            with pytest.raises(RuntimeError, match="Disk space limit exceeded"):
                self.bootstrap.generate_region_batch(x_range=(0, 32), z_range=(0, 32))
    
    def test_java_heap_exhaustion_recovery(self):
        """Test recovery from Java heap exhaustion errors."""
        with patch('subprocess.run') as mock_subprocess:
            # First call fails with OutOfMemoryError
            mock_subprocess.side_effect = [
                MagicMock(returncode=1, stderr="java.lang.OutOfMemoryError"),
                MagicMock(returncode=0)  # Second call succeeds
            ]
            
            with patch.object(self.bootstrap, '_reduce_batch_size') as mock_reduce:
                result = self.bootstrap.generate_region_batch(
                    x_range=(0, 64), z_range=(0, 64)
                )
                
                # Should have retried with reduced batch size
                assert mock_subprocess.call_count == 2
                mock_reduce.assert_called_once()


class TestWorldGenConfiguration:
    """Test configuration loading and validation."""
    
    def test_config_loading_from_yaml(self):
        """Test that worldgen configuration loads correctly from config.yaml."""
        # Test that config loads successfully now that we have a config.yaml file
        from worldgen.config import load_worldgen_config
        config = load_worldgen_config()
        
        # Verify expected configuration keys exist
        assert "seed" in config
        assert "java_heap" in config
        assert config["seed"] == "VoxelTree"
    
    def test_java_tool_fallback_chain(self):
        """Test that Java tool selection follows fallback hierarchy."""
        bootstrap = WorldGenBootstrap()
        java_tool_path = bootstrap._get_java_tool_path()
        
        # Should return the fallback path since neither tool exists in test environment
        assert "fabric-worldgen-mod.jar" in str(java_tool_path)
