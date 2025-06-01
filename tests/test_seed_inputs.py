"""
RED Phase Tests for SeedInputGenerator

This test suite follows TDD methodology for Phase 1B - Seed-Based Input Generation.
These tests are designed to FAIL initially and guide implementation development.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from scripts.seed_inputs.generator import SeedInputGenerator


class TestSeedInputGeneratorBounds:
    """Test that SeedInputGenerator produces values within expected bounds."""

    def test_biome_id_in_expected_range(self):
        """Test that biome IDs are within valid Minecraft range [0, 255]."""
        generator = SeedInputGenerator(seed=12345)

        # Test multiple coordinates to ensure consistency
        test_coords = [(0, 0), (100, 200), (-50, -75), (1000, 1000)]

        for x, z in test_coords:
            biome_id = generator.get_biome(x, z)
            assert isinstance(biome_id, int), f"Biome ID should be int, got {type(biome_id)}"
            assert 0 <= biome_id <= 255, f"Biome ID {biome_id} out of range [0, 255] at ({x}, {z})"

    def test_heightmap_bounds_are_valid(self):
        """Test that heightmap values are within valid Minecraft range [0, 384]."""
        generator = SeedInputGenerator(seed=54321)

        # Test multiple coordinates including extremes
        test_coords = [(0, 0), (500, 500), (-1000, 1000), (10000, -5000)]

        for x, z in test_coords:
            height = generator.get_heightmap(x, z)
            assert isinstance(height, int), f"Height should be int, got {type(height)}"
            assert 0 <= height <= 384, f"Height {height} out of range [0, 384] at ({x}, {z})"

    def test_river_noise_is_float(self):
        """Test that river noise returns float values."""
        generator = SeedInputGenerator(seed=98765)

        test_coords = [(0, 0), (123, 456), (-789, 321)]

        for x, z in test_coords:
            river_value = generator.get_river_noise(x, z)
            assert isinstance(
                river_value, float
            ), f"River noise should be float, got {type(river_value)}"
            # River noise should be reasonable but no strict bounds enforced
            assert (
                -10.0 <= river_value <= 10.0
            ), f"River noise {river_value} seems unreasonable at ({x}, {z})"


class TestPatchGeneration:
    """Test patch generation functionality."""

    def test_patch_shapes_match_requested_size(self):
        """Test that generated patches have correct array shapes."""
        generator = SeedInputGenerator(seed=11111)

        test_sizes = [8, 16, 32, 64]
        test_coords = [(0, 0), (100, 200), (-50, 75)]

        for size in test_sizes:
            for x, z in test_coords:
                patch = generator.get_patch(x, z, size)

                # Check that patch is a dictionary
                assert isinstance(patch, dict), f"Patch should be dict, got {type(patch)}"

                # Check array shapes
                assert patch["biomes"].shape == (
                    size,
                    size,
                ), f"Biomes shape mismatch: expected ({size}, {size}), got {patch['biomes'].shape}"
                assert patch["heightmap"].shape == (
                    size,
                    size,
                ), f"Heightmap shape mismatch: expected ({size}, {size}), got {patch['heightmap'].shape}"
                assert patch["river"].shape == (
                    size,
                    size,
                ), f"River shape mismatch: expected ({size}, {size}), got {patch['river'].shape}"

    def test_patch_contains_required_keys(self):
        """Test that patches contain all required metadata keys."""
        generator = SeedInputGenerator(seed=22222)
        patch = generator.get_patch(0, 0, 16)

        required_keys = {"biomes", "heightmap", "river", "x", "z", "seed"}
        patch_keys = set(patch.keys())

        missing_keys = required_keys - patch_keys
        assert not missing_keys, f"Patch missing required keys: {missing_keys}"

        # Check metadata types
        assert isinstance(patch["x"], int), "Patch x coordinate should be int"
        assert isinstance(patch["z"], int), "Patch z coordinate should be int"
        assert isinstance(patch["seed"], int), "Patch seed should be int"

    def test_patch_array_dtypes(self):
        """Test that patch arrays have correct data types."""
        generator = SeedInputGenerator(seed=33333)
        patch = generator.get_patch(0, 0, 16)

        assert (
            patch["biomes"].dtype == np.uint8
        ), f"Biomes should be uint8, got {patch['biomes'].dtype}"
        assert (
            patch["heightmap"].dtype == np.uint16
        ), f"Heightmap should be uint16, got {patch['heightmap'].dtype}"
        assert (
            patch["river"].dtype == np.float32
        ), f"River should be float32, got {patch['river'].dtype}"


class TestFileOperations:
    """Test file I/O operations."""

    def test_npz_file_contains_required_keys(self):
        """Test that saved .npz files contain all required data keys."""
        generator = SeedInputGenerator(seed=44444)

        # Generate a test patch
        patch = generator.get_patch(0, 0, 16)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_file = temp_path / "test_patch.npz"

            saved_path = generator.save_patch_npz(patch, output_file)
            assert saved_path.exists(), "NPZ file should exist after saving"

            # Load and verify contents
            loaded_data = np.load(saved_path)

            try:
                required_keys = {"biomes", "heightmap", "river", "x", "z", "seed"}
                loaded_keys = set(loaded_data.keys())

                missing_keys = required_keys - loaded_keys
                assert not missing_keys, f"NPZ file missing required keys: {missing_keys}"

                # Verify data integrity
                np.testing.assert_array_equal(loaded_data["biomes"], patch["biomes"])
                np.testing.assert_array_equal(loaded_data["heightmap"], patch["heightmap"])
                np.testing.assert_array_equal(loaded_data["river"], patch["river"])
                assert loaded_data["x"] == patch["x"]
                assert loaded_data["z"] == patch["z"]
                assert loaded_data["seed"] == patch["seed"]
            finally:
                # Properly close the numpy file
                loaded_data.close()

    def test_patch_filename_generation(self):
        """Test standardized filename generation."""
        generator = SeedInputGenerator(seed=55555)
        output_dir = Path("test_output")

        test_coords = [(0, 0), (100, 200), (-50, -75)]

        for x, z in test_coords:
            filename_path = generator.get_patch_filename(x, z, output_dir)
            expected_filename = f"patch_x{x}_z{z}.npz"

            assert (
                filename_path.name == expected_filename
            ), f"Filename mismatch: expected {expected_filename}, got {filename_path.name}"
            assert (
                filename_path.parent == output_dir
            ), f"Output directory mismatch: expected {output_dir}, got {filename_path.parent}"


class TestBatchOperations:
    """Test batch generation and saving functionality."""

    def test_batch_generation_length(self):
        """Test that batch generation produces correct number of patches."""
        generator = SeedInputGenerator(seed=66666)

        coordinates = [(0, 0), (16, 16), (32, 32), (48, 48)]
        size = 16

        patches = generator.generate_batch(coordinates, size)

        assert len(patches) == len(
            coordinates
        ), f"Expected {len(coordinates)} patches, got {len(patches)}"

        # Verify each patch corresponds to correct coordinates
        for i, (expected_x, expected_z) in enumerate(coordinates):
            patch = patches[i]
            assert (
                patch["x"] == expected_x
            ), f"Patch {i} x coordinate mismatch: expected {expected_x}, got {patch['x']}"
            assert (
                patch["z"] == expected_z
            ), f"Patch {i} z coordinate mismatch: expected {expected_z}, got {patch['z']}"

    def test_batch_saving_creates_all_files(self):
        """Test that batch saving creates correct number of files."""
        generator = SeedInputGenerator(seed=77777)

        # Generate test patches
        coordinates = [(0, 0), (16, 16), (32, 32)]
        patches = generator.generate_batch(coordinates, 16)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            saved_paths = generator.save_batch(patches, temp_path)

            assert len(saved_paths) == len(
                patches
            ), f"Expected {len(patches)} saved files, got {len(saved_paths)}"

            # Verify all files exist
            for path in saved_paths:
                assert path.exists(), f"Saved file {path} does not exist"
                assert path.suffix == ".npz", f"Saved file {path} should have .npz extension"

            # Verify filenames match coordinates
            for i, (x, z) in enumerate(coordinates):
                expected_filename = f"patch_x{x}_z{z}.npz"
                actual_filename = saved_paths[i].name
                assert (
                    actual_filename == expected_filename
                ), f"Filename mismatch: expected {expected_filename}, got {actual_filename}"


class TestDeterminism:
    """Test that generation is deterministic and reproducible."""

    def test_same_seed_same_output(self):
        """Test that same seed produces identical output."""
        seed = 88888
        x, z = 100, 200
        size = 16

        # Generate with first instance
        generator1 = SeedInputGenerator(seed=seed)
        patch1 = generator1.get_patch(x, z, size)

        # Generate with second instance (same seed)
        generator2 = SeedInputGenerator(seed=seed)
        patch2 = generator2.get_patch(x, z, size)

        # Should be identical
        np.testing.assert_array_equal(patch1["biomes"], patch2["biomes"])
        np.testing.assert_array_equal(patch1["heightmap"], patch2["heightmap"])
        np.testing.assert_array_equal(patch1["river"], patch2["river"])
        assert patch1["x"] == patch2["x"]
        assert patch1["z"] == patch2["z"]
        assert patch1["seed"] == patch2["seed"]

    def test_different_seeds_different_output(self):
        """Test that different seeds produce different output."""
        x, z = 100, 200
        size = 16

        # Generate with different seeds
        generator1 = SeedInputGenerator(seed=11111)
        generator2 = SeedInputGenerator(seed=22222)

        patch1 = generator1.get_patch(x, z, size)
        patch2 = generator2.get_patch(x, z, size)

        # Should be different (at least some values)
        # Using a tolerance check since arrays might occasionally match by chance
        biomes_different = not np.array_equal(patch1["biomes"], patch2["biomes"])
        heights_different = not np.array_equal(patch1["heightmap"], patch2["heightmap"])
        river_different = not np.array_equal(patch1["river"], patch2["river"])

        # At least one should be different
        assert (
            biomes_different or heights_different or river_different
        ), "Different seeds should produce different terrain"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_size_patch(self):
        """Test handling of zero-size patch request."""
        generator = SeedInputGenerator(seed=99999)

        with pytest.raises((ValueError, AssertionError)):
            generator.get_patch(0, 0, 0)

    def test_negative_coordinates(self):
        """Test that negative coordinates are handled correctly."""
        generator = SeedInputGenerator(seed=12121)

        # Should not raise exceptions
        patch = generator.get_patch(-1000, -2000, 16)

        assert patch["x"] == -1000
        assert patch["z"] == -2000
        assert patch["biomes"].shape == (16, 16)
        assert patch["heightmap"].shape == (16, 16)
        assert patch["river"].shape == (16, 16)

    def test_large_coordinates(self):
        """Test that very large coordinates are handled correctly."""
        generator = SeedInputGenerator(seed=23232)

        # Should not raise exceptions or produce invalid values
        large_x, large_z = 1000000, -500000
        patch = generator.get_patch(large_x, large_z, 8)

        assert patch["x"] == large_x
        assert patch["z"] == large_z

        # All values should still be in valid ranges
        assert np.all((patch["biomes"] >= 0) & (patch["biomes"] <= 255))
        assert np.all((patch["heightmap"] >= 0) & (patch["heightmap"] <= 384))
