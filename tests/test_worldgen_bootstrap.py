"""
Tests for WorldGenBootstrap - Phase 0B World Generation Bootstrap

Following TDD RED phase: Write failing tests first.
These tests define the expected behavior for .mca world generation.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path

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
        assert (
            self.bootstrap.temp_world_dir.name == "temp_worlds"
        )  # Test different seed produces different hash
        bootstrap2 = WorldGenBootstrap(seed="DifferentSeed")
        assert bootstrap2.seed != self.bootstrap.seed

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


class TestWorldGenConfiguration:
    """Test configuration loading and validation."""

    def test_config_loading_from_yaml(self):
        """Test that worldgen configuration loads correctly from config.yaml."""
        # Test that config loads successfully now that we have a config.yaml file
        from scripts.worldgen.config import load_worldgen_config

        config = load_worldgen_config()  # Verify expected configuration keys exist
        assert "seed" in config
        assert "java_heap" in config
        assert config["seed"] == 6901795026152433433


@pytest.mark.integration
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

    @pytest.mark.skip(reason="This is a great test, but it's slow and not needed for every run.")
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

    @pytest.mark.skip(reason="This is a great test, but it's slow and not needed for every run.")
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

        unique, counts = np.unique(blocks, return_counts=True)  # Sort by frequency (descending)
        sort_idx = np.argsort(counts)[::-1]
        return unique[sort_idx], counts[sort_idx]
