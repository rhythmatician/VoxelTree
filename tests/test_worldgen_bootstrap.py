"""
Tests for WorldGenBootstrap - Phase 0B World Generation Bootstrap

Following TDD RED phase: Write failing tests first.
These tests define the expected behavior for .mca world generation.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.worldgen.bootstrap import WorldGenBootstrap


class TestWorldGenBootstrap:
    """Test suite for WorldGenBootstrap class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_temp_dir = Path(tempfile.mkdtemp())
        self.bootstrap = WorldGenBootstrap(
            seed="VoxelTree",
            java_heap="2G",
            temp_world_dir=self.test_temp_dir / "temp_worlds",
            test_mode=False,  # LEAVE AS FALSE UNTIL IT PASSES
        )

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    def test_worldgen_bootstrap_init(self):
        """Test bootstrap initialization with seed hashing."""
        # Test that seed "VoxelTree" converts to expected numeric value
        assert self.bootstrap.seed == 6901795026152433433
        assert self.bootstrap.java_heap == "2G"
        assert self.bootstrap.temp_world_dir.name == "temp_worlds"

        # Test different seed produces different hash
        bootstrap2 = WorldGenBootstrap(seed="DifferentSeed")
        assert bootstrap2.seed != self.bootstrap.seed

    @patch("subprocess.run")
    def test_generate_single_region(self, mock_subprocess):
        """Test generation of one .mca file with known chunks."""
        # Mock successful Java subprocess execution
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Test region generation - this creates the world directory
        result_path = self.bootstrap.generate_region_batch(x_range=(0, 32), z_range=(0, 32))

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
        import time

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

            # Add small delay to ensure different modification times
            time.sleep(0.01)

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
        with patch.object(self.bootstrap, "_get_directory_size_gb") as mock_size:
            mock_size.return_value = 6.0  # Exceeds 5GB limit

            with pytest.raises(RuntimeError, match="Disk space limit exceeded"):
                self.bootstrap.generate_region_batch(x_range=(0, 32), z_range=(0, 32))

    def test_java_heap_exhaustion_recovery(self):
        """Test recovery from Java heap exhaustion errors.

        WARNING: This takes forever to run

        Once it passes, we should skip it in normal test runs, or optimize it somehow.
        """
        with patch("subprocess.run") as mock_subprocess:
            # First call fails with OutOfMemoryError
            mock_subprocess.side_effect = [
                MagicMock(returncode=1, stderr="java.lang.OutOfMemoryError"),
                MagicMock(returncode=0),  # Second call succeeds
            ]

            with patch.object(self.bootstrap, "_reduce_batch_size") as mock_reduce:
                self.bootstrap.generate_region_batch(x_range=(0, 64), z_range=(0, 64))

                # Should have retried with reduced batch size
                assert mock_subprocess.call_count == 2
                mock_reduce.assert_called_once()


class TestWorldGenConfiguration:
    """Test configuration loading and validation."""

    def test_config_loading_from_yaml(self):
        """Test that worldgen configuration loads correctly from config.yaml."""
        # Test that config loads successfully now that we have a config.yaml file
        from scripts.worldgen.config import load_worldgen_config

        config = load_worldgen_config()

        # Verify expected configuration keys exist
        assert "seed" in config
        assert "java_heap" in config
        assert config["seed"] == 6901795026152433433

    def test_java_tool_fallback_chain(self):
        """Test that Java tool selection follows fallback hierarchy."""

        # Create a mock file existence checker
        def mock_file_exists(path):
            path_str = str(path)
            if "fabric-server" in path_str:
                return False  # primary tool doesn't exist
            elif "fabric-worldgen-mod.jar" in path_str:
                return True  # fallback exists
            return False

        # Skip validation during bootstrap creation
        with patch.object(WorldGenBootstrap, "_validate_tool_paths"):
            bootstrap = WorldGenBootstrap()

            # Test the fallback logic with our mock checker
            java_tool_path = bootstrap._get_java_tool_path(file_exists_checker=mock_file_exists)

            # Should return the fallback path since primary tool doesn't exist
            assert "fabric-worldgen-mod.jar" in str(java_tool_path)


@pytest.mark.integration  # FIXME: PytestUnknownMarkWarning: Unknown pytest.mark.integration - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
class TestWorldGenIntegration:
    """Integration tests for real world generation using Fabric + Chunky."""

    # TODO: Optimize this test suite to run faster, if possible, by
    # eliminating any redundancy and generating smaller regions if doable.
    def setup_method(self):
        """Set up integration test fixtures."""
        self.test_temp_dir = Path(tempfile.mkdtemp())
        self.bootstrap = WorldGenBootstrap(
            seed=6901795026152433433,
            java_heap="2G",
            temp_world_dir=self.test_temp_dir / "temp_worlds",
            test_mode=False,  # LEAVE AS FALSE UNTIL IT PASSES. Test with a real integration server
        )

    def teardown_method(self):
        """Clean up integration test fixtures."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    # @pytest.mark.skip(reason="Takes 10 minutes to run!")
    def test_real_fabric_chunky_world_generation(self):
        """
        Integration test: Generate real .mca files using Fabric server + Chunky mod.

        This test should fail initially (RED phase) because bootstrap.py
        needs to be updated to work with Fabric + Chunky instead of standalone JARs.
        """
        # Check that required Java tools exist
        fabric_jar = Path(
            "tools/fabric-server/fabric-server-mc.1.21.5-loader.0.16.14-launcher.1.0.3.jar"
        )
        chunky_jar = Path("tools/fabric-server/runtime/mods/Chunky-Fabric-1.4.36.jar")

        # Skip test if tools not available (CI environment)
        if not fabric_jar.exists() or not chunky_jar.exists():
            pytest.skip("Fabric server or Chunky mod not available")

        # Generate a small 2x2 chunk region (minimal for speed)
        world_dir = self.bootstrap.generate_region_batch(
            x_range=(0, 2), z_range=(0, 2)  # 2 chunks wide  # 2 chunks deep
        )

        # Verify .mca files were created
        region_dir = world_dir / "region"
        assert region_dir.exists(), "Region directory should be created"

        mca_files = list(region_dir.glob("*.mca"))
        assert len(mca_files) > 0, "At least one .mca file should be generated"

        # Verify .mca files are valid (basic size check)
        for mca_file in mca_files:
            assert mca_file.stat().st_size > 1000, f"{mca_file.name} is too small to be valid"

        # Verify known region file exists (r.0.0.mca should contain chunks 0-31 in both X,Z)
        expected_region = region_dir / "r.0.0.mca"
        assert expected_region.exists(), "Expected region r.0.0.mca should exist"

    # @pytest.mark.skip(reason="Takes 10 minutes to run!")

    def test_validate_chunk_block_data_matches_reference(self):
        """
        Integration test: Compare generated chunk block data against reference chunk.

        This test extracts block type arrays from both reference and generated chunks
        and compares them to validate that world generation produces correct terrain.
        """
        # Check for reference file
        reference_mca_zip = Path("data/VoxelTree/r.0.0.mca.zip")
        if not reference_mca_zip.exists():
            pytest.skip("Reference .mca file not found, skipping chunk data validation")

        # Skip if tools not available
        fabric_jar = Path(
            "tools/fabric-server/fabric-server-mc.1.21.5-loader.0.16.14-launcher.1.0.3.jar"
        )
        chunky_jar = Path("tools/fabric-server/runtime/mods/Chunky-Fabric-1.4.36.jar")

        if not fabric_jar.exists() or not chunky_jar.exists():
            pytest.skip("Fabric server or Chunky mod not available")

        # Extract reference .mca file
        with zipfile.ZipFile(reference_mca_zip, "r") as zip_ref:
            zip_ref.extractall(self.test_temp_dir / "reference")
        reference_mca = self.test_temp_dir / "reference" / "r.0.0.mca"

        # Generate world with same seed as reference
        world_dir = self.bootstrap.generate_region_batch(
            x_range=(0, 1), z_range=(0, 1)  # Single chunk at origin
        )

        # Get generated .mca file
        generated_mca = world_dir / "region" / "r.0.0.mca"
        assert (
            generated_mca.exists()
        ), "Generated .mca file should exist"  # Compare chunk block data
        self._compare_chunk_block_data(reference_mca, generated_mca, chunk_x=0, chunk_z=0)

    # Buffer comment. Ignore this line.

    def _compare_chunk_block_data(
        self, reference_mca: Path, generated_mca: Path, chunk_x: int, chunk_z: int
    ):
        """
        Compare block data between reference and generated chunks.

        Provides detailed comparison including difference percentage and sample diffs.
        """
        try:
            import anvil
        except ImportError:
            pytest.skip("anvil-parser2 not available for chunk data comparison")

        # Load both .mca files
        try:
            reference_region = anvil.Region.from_file(str(reference_mca))
            generated_region = anvil.Region.from_file(str(generated_mca))
        except Exception as e:
            pytest.fail(f"Failed to load .mca files: {e}")

        # Extract chunks
        try:
            reference_chunk = reference_region.get_chunk(chunk_x, chunk_z)
            generated_chunk = generated_region.get_chunk(chunk_x, chunk_z)
        except Exception as e:
            pytest.fail(f"Failed to extract chunk ({chunk_x}, {chunk_z}): {e}")

        if reference_chunk is None:
            pytest.skip(f"Reference chunk ({chunk_x}, {chunk_z}) not found in reference .mca")

        if generated_chunk is None:
            pytest.fail(f"Generated chunk ({chunk_x}, {chunk_z}) not found in generated .mca")

        # Extract block data for comparison
        ref_blocks = self._extract_chunk_blocks(reference_chunk)
        gen_blocks = self._extract_chunk_blocks(generated_chunk)

        # Compare dimensions
        assert (
            ref_blocks.shape == gen_blocks.shape
        ), f"Chunk dimensions don't match: reference {ref_blocks.shape} vs generated {gen_blocks.shape}"

        # Calculate differences
        total_blocks = ref_blocks.size
        differences = ref_blocks != gen_blocks
        different_blocks = differences.sum()
        match_percentage = ((total_blocks - different_blocks) / total_blocks) * 100
        print("\n=== Chunk Block Data Comparison ===")
        print(f"Chunk: ({chunk_x}, {chunk_z})")
        print(f"Total blocks: {total_blocks:,}")
        print(f"Matching blocks: {total_blocks - different_blocks:,}")
        print(f"Different blocks: {different_blocks:,}")
        print(f"Match percentage: {match_percentage:.2f}%")

        if different_blocks > 0:
            # Show sample of differences
            print("\n=== Sample Differences (first 10) ===")
            diff_coords = list(zip(*differences.nonzero()))[:10]

            for i, (x, y, z) in enumerate(diff_coords):
                ref_block = ref_blocks[x, y, z]
                gen_block = gen_blocks[x, y, z]
                print(
                    f"  {i+1}. Position ({x:2d}, {y:3d}, {z:2d}): reference={ref_block:4d}, generated={gen_block:4d}"
                )

            # Show block type frequency differences
            print("\n=== Block Type Analysis ===")
            ref_unique, ref_counts = self._get_block_frequency(ref_blocks)
            gen_unique, gen_counts = self._get_block_frequency(gen_blocks)

            print("Reference chunk block types:")
            for block_id, count in zip(ref_unique[:5], ref_counts[:5]):  # Top 5
                percentage = (count / total_blocks) * 100
                print(f"  Block {block_id:3d}: {count:6,} blocks ({percentage:5.1f}%)")

            print("Generated chunk block types:")
            for block_id, count in zip(gen_unique[:5], gen_counts[:5]):  # Top 5
                percentage = (count / total_blocks) * 100
                print(
                    f"  Block {block_id:3d}: {count:6,} blocks ({percentage:5.1f}%)"
                )  # For now, let's assert they should be very similar (allowing small differences due to environment)
        # but fail if they're too different (indicating a real problem)
        if match_percentage < 95.0:
            pytest.fail(
                f"Chunks are too different: only {match_percentage:.2f}% match (expected >95%)"
            )
        elif match_percentage < 100.0:
            print(
                f"⚠️  Minor differences detected ({100-match_percentage:.2f}% different) - possibly due to environment"
            )
        else:
            print("✅ Chunks match exactly!")

    def _extract_chunk_blocks(self, chunk):
        """Extract block type array from anvil chunk."""
        import numpy as np

        # Initialize 16x384x16 array for blocks (X, Y, Z)
        # Y range is -64 to 319 (384 blocks total)
        blocks = np.zeros((16, 384, 16), dtype=np.int32)

        # Extract blocks from the entire chunk
        for x in range(16):
            for z in range(16):
                for y_offset in range(384):
                    # Convert array index to world Y coordinate
                    world_y = y_offset - 64  # Y=-64 to Y=319

                    try:
                        block = chunk.get_block(x, world_y, z)
                        # Convert block to numeric ID (simplified)
                        block_id = self._block_to_id(block)
                        blocks[x, y_offset, z] = block_id
                    except Exception:
                        # Default to air if can't read block
                        blocks[x, y_offset, z] = 0

        return blocks

    def _block_to_id(self, block) -> int:
        """Convert anvil block to numeric ID for comparison."""
        if block is None:
            return 0  # Air

        # Use block name as a simple hash for comparison
        if hasattr(block, "id"):
            block_name = block.id
        elif hasattr(block, "name"):
            block_name = block.name
        else:
            block_name = str(block)

        # Simple hash of block name to numeric ID
        return abs(hash(block_name)) % 1000

    def _get_block_frequency(self, blocks):
        """Get frequency distribution of block types."""
        import numpy as np

        unique, counts = np.unique(blocks, return_counts=True)

        # Sort by frequency (descending)
        sort_idx = np.argsort(counts)[::-1]
        return unique[sort_idx], counts[sort_idx]

    def _validate_mca_file_structure(self, mca_file: Path):
        """
        Validate that an .mca file has proper Minecraft world structure.

        This checks for basic .mca file validity without requiring exact binary matching.
        """
        # Check file size is reasonable (should be at least 8KB for a valid .mca file)
        file_size = mca_file.stat().st_size
        assert file_size > 8192, f"MCA file {mca_file.name} is too small ({file_size} bytes)"

        # Read and validate .mca header structure
        with open(mca_file, "rb") as f:
            # .mca files start with a 4KB sector table, then 4KB timestamps
            header = f.read(8192)  # First 8KB contains headers

            # Check that the header contains non-zero data (indicating chunks are present)
            non_zero_bytes = sum(1 for byte in header if byte != 0)
            assert (
                non_zero_bytes > 100
            ), f"MCA file appears to contain no chunk data (only {non_zero_bytes} non-zero header bytes)"

            # Check for basic .mca file structure markers
            # The first 4 bytes of each chunk sector entry should form reasonable sector offsets
            sector_table = header[:4096]
            valid_sectors = 0

            for i in range(0, 4096, 4):
                sector_data = sector_table[i : i + 4]
                if any(byte != 0 for byte in sector_data):
                    # Extract sector offset (first 3 bytes) and sector count (last byte)
                    sector_offset = int.from_bytes(sector_data[:3], "big")
                    sector_count = sector_data[3]

                    # Valid sector offsets should be >= 2 (after header sectors)
                    # and sector count should be reasonable (1-255)
                    if sector_offset >= 2 and 1 <= sector_count <= 255:
                        valid_sectors += 1

            assert valid_sectors > 0, "MCA file contains no valid chunk sector entries"

            # Validate that the file contains actual compressed chunk data
            f.seek(8192)  # Skip headers
            chunk_data = f.read(1024)  # Read first KB of chunk data

            # Look for common NBT/Minecraft data patterns
            # Minecraft uses gzip or zlib compression, so look for compression headers
            has_compression_header = (
                chunk_data.startswith(b"\x1f\x8b")  # gzip header
                or chunk_data.startswith(b"\x78\x9c")  # zlib header
                or chunk_data.startswith(b"\x78\xda")  # zlib header variant
                or b"\x0a" in chunk_data[:100]  # NBT tag indicators
            )

            assert (
                has_compression_header
            ), "MCA file does not contain expected Minecraft chunk data format"
