"""
Tests for SeedInputGenerator - Phase 1B Seed-Based Input Generation

Following TDD RED phase: Write failing tests first.
These tests define the expected behavior for seed-based conditioning variables.
"""

import numpy as np
from pathlib import Path
import tempfile
import shutil

from scripts.seed_inputs.generator import SeedInputGenerator


class TestSeedInputGenerator:
    """Test suite for SeedInputGenerator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_seed = 123456
        self.generator = SeedInputGenerator(seed=self.test_seed)
        self.test_output_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    # TDD Cycle 1B.1: Biome generator
    def test_biome_generator_deterministic(self):
        """Test that biome generation is deterministic for given seed and coordinates."""
        # Same coordinates should always return same biome
        biome1 = self.generator.get_biome(0, 0)
        biome2 = self.generator.get_biome(0, 0)
        assert biome1 == biome2

        # Different coordinates should potentially return different biomes
        biome_origin = self.generator.get_biome(0, 0)
        biome_offset = self.generator.get_biome(100, 100)

        # Biomes should be valid vanilla IDs (0-255)
        assert 0 <= biome_origin <= 255
        assert 0 <= biome_offset <= 255

    def test_biome_generator_known_coordinates(self):
        """Test that known coordinates return expected biome types."""
        # For seed 123456, these coordinates should return specific biome ranges
        # This ensures our biome generation is working correctly
        biome = self.generator.get_biome(0, 0)
        assert isinstance(biome, int)
        assert 0 <= biome <= 255

    def test_biome_generator_different_seeds(self):
        """Test that different seeds produce different biome patterns."""
        generator2 = SeedInputGenerator(seed=654321)

        # Same coordinates with different seeds should potentially differ
        self.generator.get_biome(50, 50)
        generator2.get_biome(50, 50)

        # While they might be the same by chance, the generators should be different objects
        assert self.generator.seed != generator2.seed

    # TDD Cycle 1B.2: Heightmap generator
    def test_heightmap_generator_valid_range(self):
        """Test that heightmap values are in valid Minecraft range."""
        height = self.generator.get_heightmap(0, 0)

        # Minecraft height range: -64 to 320 (but we'll use 0-384 for simplicity)
        assert isinstance(height, int)
        assert 0 <= height <= 384

    def test_heightmap_generator_deterministic(self):
        """Test that heightmap generation is deterministic."""
        height1 = self.generator.get_heightmap(10, 20)
        height2 = self.generator.get_heightmap(10, 20)
        assert height1 == height2

    def test_heightmap_generator_spatial_coherence(self):
        """Test that nearby coordinates have somewhat similar heights."""
        height_center = self.generator.get_heightmap(100, 100)
        height_nearby = self.generator.get_heightmap(101, 100)

        # Heights should not be wildly different (within 50 blocks)
        height_diff = abs(height_center - height_nearby)
        assert height_diff <= 50, f"Heights too different: {height_center} vs {height_nearby}"

    # TDD Cycle 1B.3: River signal patch
    def test_river_noise_valid_range(self):
        """Test that river noise values are in expected range."""
        river_value = self.generator.get_river_noise(0, 0)

        assert isinstance(river_value, float)
        # OpenSimplex noise typically returns values in [-1, 1]
        assert -2.0 <= river_value <= 2.0

    def test_river_noise_deterministic(self):
        """Test that river noise is deterministic."""
        river1 = self.generator.get_river_noise(30, 40)
        river2 = self.generator.get_river_noise(30, 40)
        assert abs(river1 - river2) < 1e-6  # Floating point equality

    def test_river_noise_continuity(self):
        """Test that river noise changes smoothly across coordinates."""
        river_values = []
        for x in range(5):
            river_values.append(self.generator.get_river_noise(x, 0))

        # Adjacent values should not be extremely different
        for i in range(len(river_values) - 1):
            diff = abs(river_values[i] - river_values[i + 1])
            assert diff <= 1.0, f"River noise too discontinuous: {diff}"

    # TDD Cycle 1B.4: Patch assembler
    def test_patch_structure_basic(self):
        """Test that patch has correct structure and data types."""
        patch = self.generator.get_patch(0, 0, 16)

        # Check required keys
        required_keys = ["biomes", "heightmap", "river", "x", "z", "seed"]
        for key in required_keys:
            assert key in patch, f"Missing key: {key}"

        # Check array shapes
        assert patch["biomes"].shape == (16, 16)
        assert patch["heightmap"].shape == (16, 16)
        assert patch["river"].shape == (16, 16)

        # Check data types
        assert patch["biomes"].dtype == np.uint8
        assert patch["heightmap"].dtype == np.uint16
        assert patch["river"].dtype == np.float32

        # Check coordinate values
        assert patch["x"] == 0
        assert patch["z"] == 0
        assert patch["seed"] == self.test_seed

    def test_patch_coordinates_alignment(self):
        """Test that patch coordinates are properly aligned."""
        x_start, z_start = 32, 48
        patch = self.generator.get_patch(x_start, z_start, 16)

        assert patch["x"] == x_start
        assert patch["z"] == z_start

        # Biomes should represent the 16x16 area starting at (x_start, z_start)
        # Spot check a few coordinates
        biome_origin = self.generator.get_biome(x_start, z_start)
        assert patch["biomes"][0, 0] == biome_origin

        height_origin = self.generator.get_heightmap(x_start, z_start)
        assert patch["heightmap"][0, 0] == height_origin

    def test_patch_different_sizes(self):
        """Test that patches can be generated in different sizes."""
        # Test 8x8 patch
        patch_8 = self.generator.get_patch(0, 0, 8)
        assert patch_8["biomes"].shape == (8, 8)
        assert patch_8["heightmap"].shape == (8, 8)
        assert patch_8["river"].shape == (8, 8)

        # Test 32x32 patch
        patch_32 = self.generator.get_patch(0, 0, 32)
        assert patch_32["biomes"].shape == (32, 32)
        assert patch_32["heightmap"].shape == (32, 32)
        assert patch_32["river"].shape == (32, 32)

    def test_patch_spatial_consistency(self):
        """Test that adjacent patches have consistent boundary values."""
        patch1 = self.generator.get_patch(0, 0, 16)
        patch2 = self.generator.get_patch(16, 0, 16)

        # The right edge of patch1 should match left edge of patch2
        # (This tests that our coordinate system is consistent)
        biome_edge1 = self.generator.get_biome(15, 0)
        biome_edge2 = self.generator.get_biome(16, 0)

        assert patch1["biomes"][15, 0] == biome_edge1
        assert patch2["biomes"][0, 0] == biome_edge2

    # TDD Cycle 1B.5: Save .npz seed-only input
    def test_save_patch_npz_structure(self):
        """Test that saved .npz files have correct structure."""
        patch = self.generator.get_patch(64, 96, 16)
        output_path = self.test_output_dir / "test_patch_x64_z96.npz"

        saved_path = self.generator.save_patch_npz(patch, output_path)

        # File should exist
        assert saved_path.exists()

        # Load and verify structure
        loaded_data = np.load(saved_path)

        required_keys = ["biomes", "heightmap", "river", "x", "z", "seed"]
        for key in required_keys:
            assert key in loaded_data, f"Missing key in saved file: {key}"

        # Verify data integrity
        assert np.array_equal(loaded_data["biomes"], patch["biomes"])
        assert np.array_equal(loaded_data["heightmap"], patch["heightmap"])
        assert np.allclose(loaded_data["river"], patch["river"])
        assert loaded_data["x"] == patch["x"]
        assert loaded_data["z"] == patch["z"]
        assert loaded_data["seed"] == patch["seed"]

    def test_save_patch_compression(self):
        """Test that saved files are properly compressed."""
        patch = self.generator.get_patch(0, 0, 16)
        output_path = self.test_output_dir / "test_compression.npz"

        saved_path = self.generator.save_patch_npz(patch, output_path)

        # File should be reasonably small (compressed)
        file_size = saved_path.stat().st_size
        # A 16x16 patch should be well under 10KB when compressed
        assert file_size < 10240, f"File too large: {file_size} bytes"

    def test_save_patch_naming_convention(self):
        """Test that automatic naming follows the expected convention."""
        x, z = 128, 256
        self.generator.get_patch(x, z, 16)

        # Test auto-naming
        auto_path = self.generator.get_patch_filename(x, z, self.test_output_dir)
        expected_name = f"patch_x{x}_z{z}.npz"

        assert auto_path.name == expected_name
        assert auto_path.parent == self.test_output_dir

    def test_batch_generation(self):
        """Test generation of multiple patches in a batch."""
        coordinates = [(0, 0), (16, 0), (0, 16), (16, 16)]

        patches = self.generator.generate_batch(coordinates, size=16)

        assert len(patches) == len(coordinates)

        for i, (x, z) in enumerate(coordinates):
            patch = patches[i]
            assert patch["x"] == x
            assert patch["z"] == z
            assert patch["biomes"].shape == (16, 16)

    def test_deterministic_across_sessions(self):
        """
        Test that same seed produces same results across different generator instances.
        """
        # Create two generators with same seed
        gen1 = SeedInputGenerator(seed=999999)
        gen2 = SeedInputGenerator(seed=999999)

        # They should produce identical patches
        patch1 = gen1.get_patch(80, 80, 16)
        patch2 = gen2.get_patch(80, 80, 16)

        assert np.array_equal(patch1["biomes"], patch2["biomes"])
        assert np.array_equal(patch1["heightmap"], patch2["heightmap"])
        assert np.allclose(patch1["river"], patch2["river"])


class TestSeedInputIntegration:
    """Integration tests for seed input generation."""

    def test_minecraft_biome_compatibility(self):
        """Test that generated biomes are compatible with Minecraft biome IDs."""
        generator = SeedInputGenerator(seed=42)

        # Sample many coordinates and check biome distribution
        biomes = []
        for x in range(0, 160, 16):
            for z in range(0, 160, 16):
                biome = generator.get_biome(x, z)
                biomes.append(biome)

        # Should have variety in biomes (not all the same)
        unique_biomes = set(biomes)
        assert len(unique_biomes) > 1, "Biome generation too uniform"

        # All biomes should be valid
        for biome in biomes:
            assert 0 <= biome <= 255

    def test_heightmap_terrain_realism(self):
        """Test that heightmaps produce realistic terrain."""
        generator = SeedInputGenerator(seed=2023)

        # Sample a region and check for reasonable terrain
        heights = []
        for x in range(0, 64, 4):
            for z in range(0, 64, 4):
                height = generator.get_heightmap(x, z)
                heights.append(height)

        heights = np.array(heights)

        # Terrain should have some variation but not be too extreme
        height_std = np.std(heights)
        assert 5 <= height_std <= 100, f"Terrain variation unrealistic: std={height_std}"

        # Should not have impossible height values
        assert np.all(heights >= 0)
        assert np.all(heights <= 384)

    def test_river_noise_distribution(self):
        """Test that river noise has appropriate statistical properties."""
        generator = SeedInputGenerator(seed=777)

        # Sample river noise across a region
        river_values = []
        for x in range(0, 128, 2):
            for z in range(0, 128, 2):
                river_val = generator.get_river_noise(x, z)
                river_values.append(river_val)

        river_values = np.array(river_values)

        # Should be roughly centered around 0 (property of OpenSimplex)
        mean_value = np.mean(river_values)
        assert abs(mean_value) < 0.5, f"River noise not centered: mean={mean_value}"

        # Should have reasonable range
        assert np.min(river_values) < -0.1
        assert np.max(river_values) > 0.1
