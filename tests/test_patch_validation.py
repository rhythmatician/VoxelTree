"""
Test suite for Phase 2.3: Validate patch format

RED phase tests for training example validation. These tests ensure that
all linked training examples (from Phase 2.2) have correct format and are
not malformed before being fed to the dataset loader.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import the classes we'll be testing (this doesn't exist yet - will fail)
from scripts.pairing.patch_validator import PatchValidator


class TestPatchValidator:
    """Test PatchValidator class initialization and configuration."""

    def test_patch_validator_init(self):
        """RED: Fails because PatchValidator doesn't exist yet."""
        config_path = Path("config.yaml")
        validator = PatchValidator(config_path=config_path)

        assert validator.config_path == config_path
        assert hasattr(validator, "required_keys")
        assert hasattr(validator, "shape_specs")
        assert hasattr(validator, "dtype_specs")

    def test_validator_load_format_specs(self):
        """RED: Fails if format specifications aren't loaded correctly."""
        validator = PatchValidator()

        # Should have format specifications for training examples
        assert hasattr(validator, "required_keys")
        assert len(validator.required_keys) > 0

        # Should include all expected keys from linked examples
        expected_keys = {
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
        }
        assert expected_keys.issubset(set(validator.required_keys))


class TestFormatValidation:
    """Test core format validation functionality."""

    @pytest.fixture
    def valid_training_example(self):
        """Create a valid training example for testing."""
        return {
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

    def test_validate_complete_example(self, valid_training_example):
        """RED: Fails if complete example validation is incorrect."""
        validator = PatchValidator()

        is_valid = validator.validate_training_example(valid_training_example)
        assert is_valid is True

        # Should return no errors for valid example
        errors = validator.get_validation_errors(valid_training_example)
        assert len(errors) == 0

    def test_detect_missing_keys(self, valid_training_example):
        """RED: Fails if missing key detection is incorrect."""
        validator = PatchValidator()

        # Remove required key
        incomplete_example = valid_training_example.copy()
        del incomplete_example["parent_voxel"]

        is_valid = validator.validate_training_example(incomplete_example)
        assert is_valid is False

        errors = validator.get_validation_errors(incomplete_example)
        assert len(errors) > 0
        assert any("missing" in error.lower() for error in errors)
        assert any("parent_voxel" in error for error in errors)

    def test_detect_wrong_shapes(self, valid_training_example):
        """RED: Fails if shape validation is incorrect."""
        validator = PatchValidator()

        # Wrong shape for parent_voxel
        wrong_shape_example = valid_training_example.copy()
        wrong_shape_example["parent_voxel"] = np.ones((4, 4, 4), dtype=bool)

        is_valid = validator.validate_training_example(wrong_shape_example)
        assert is_valid is False

        errors = validator.get_validation_errors(wrong_shape_example)
        assert len(errors) > 0
        assert any("shape" in error.lower() for error in errors)

    def test_detect_wrong_dtypes(self, valid_training_example):
        """RED: Fails if dtype validation is incorrect."""
        validator = PatchValidator()

        # Wrong dtype for target_types
        wrong_dtype_example = valid_training_example.copy()
        wrong_dtype_example["target_types"] = np.ones((16, 16, 16), dtype=np.float32)

        is_valid = validator.validate_training_example(wrong_dtype_example)
        assert is_valid is False

        errors = validator.get_validation_errors(wrong_dtype_example)
        assert len(errors) > 0
        assert any("dtype" in error.lower() or "type" in error.lower() for error in errors)

    def test_validate_coordinate_consistency(self, valid_training_example):
        """RED: Fails if coordinate consistency validation is missing."""
        validator = PatchValidator()

        # Inconsistent coordinates
        inconsistent_example = valid_training_example.copy()
        inconsistent_example["chunk_x"] = 10
        inconsistent_example["chunk_z"] = 15
        # y_index should be in valid range (0-23 for 384 block height)
        inconsistent_example["y_index"] = 50  # Invalid y_index

        is_valid = validator.validate_training_example(inconsistent_example)
        assert is_valid is False

        errors = validator.get_validation_errors(inconsistent_example)
        assert len(errors) > 0
        assert any("y_index" in error or "coordinate" in error.lower() for error in errors)


class TestBatchValidation:
    """Test batch validation of multiple training examples."""

    @pytest.fixture
    def temp_linked_data(self):
        """Create temporary directory with mixed valid/invalid linked examples."""
        temp_dir = Path(tempfile.mkdtemp())
        linked_dir = temp_dir / "linked"
        linked_dir.mkdir()

        # Create valid examples
        for i in range(3):
            valid_example = {
                "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                "target_types": np.random.randint(0, 5, size=(16, 16, 16), dtype=np.uint8),
                "biome_patch": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                "heightmap_patch": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                "river_patch": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                "y_index": i,
                "chunk_x": 10,
                "chunk_z": 15,
                "lod": 1,
            }
            np.savez_compressed(linked_dir / f"valid_example_{i}.npz", **valid_example)

        # Create invalid examples
        invalid_example1 = {
            "parent_voxel": np.ones((4, 4, 4), dtype=bool),  # Wrong shape
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
        np.savez_compressed(linked_dir / "invalid_shape.npz", **invalid_example1)

        invalid_example2 = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            # Missing target_types
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }
        np.savez_compressed(linked_dir / "missing_key.npz", **invalid_example2)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_validate_batch_directory(self, temp_linked_data):
        """RED: Fails if batch directory validation is incorrect."""
        validator = PatchValidator()

        linked_dir = temp_linked_data / "linked"

        validation_report = validator.validate_batch_directory(linked_dir)

        # Should detect 3 valid and 2 invalid files
        assert validation_report["total_files"] == 5
        assert validation_report["valid_files"] == 3
        assert validation_report["invalid_files"] == 2
        assert len(validation_report["errors"]) == 2

    def test_filter_valid_examples(self, temp_linked_data):
        """RED: Fails if filtering valid examples doesn't work correctly."""
        validator = PatchValidator()

        linked_dir = temp_linked_data / "linked"

        valid_files = validator.get_valid_training_files(linked_dir)

        # Should return only the 3 valid files
        assert len(valid_files) == 3
        assert all("valid_example" in str(f) for f in valid_files)

    def test_generate_validation_report(self, temp_linked_data):
        """RED: Fails if validation report generation is incorrect."""
        validator = PatchValidator()

        linked_dir = temp_linked_data / "linked"

        report = validator.generate_validation_report(linked_dir)

        # Should contain summary statistics
        assert "summary" in report
        assert "file_details" in report
        assert "common_errors" in report

        # Summary should have correct counts
        assert report["summary"]["total_files"] == 5
        assert report["summary"]["valid_files"] == 3
        assert report["summary"]["invalid_files"] == 2


class TestSpecificValidations:
    """Test specific validation rules for training examples."""

    def test_validate_lod_consistency(self):
        """RED: Fails if LOD level validation is incorrect."""
        validator = PatchValidator()

        example = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32),
            "y_index": 5,
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": -1,  # Invalid LOD level
        }

        is_valid = validator.validate_training_example(example)
        assert is_valid is False

        errors = validator.get_validation_errors(example)
        assert any("lod" in error.lower() for error in errors)

    def test_validate_y_index_range(self):
        """RED: Fails if y_index range validation is incorrect."""
        validator = PatchValidator()

        example = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8),
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32),
            "y_index": 30,  # Invalid - should be 0-23 for 384 blocks
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }

        is_valid = validator.validate_training_example(example)
        assert is_valid is False

    def test_validate_data_ranges(self):
        """RED: Fails if data range validation is missing."""
        validator = PatchValidator()

        example = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_mask": np.ones((16, 16, 16), dtype=bool),
            "target_types": np.ones((16, 16, 16), dtype=np.uint8) * 255,  # Valid but at boundary
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32) * 10,  # Out of range [0,1]
            "y_index": 30,  # Out of range [0,23]
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 5,  # Out of range [1,4]
        }

        is_valid = validator.validate_training_example(example)
        assert is_valid is False

        errors = validator.get_validation_errors(example)
        assert len(errors) > 0


class TestValidationReporting:
    """Test validation error reporting and statistics."""

    def test_error_categorization(self):
        """RED: Fails if error categorization is missing."""
        validator = PatchValidator()

        # Create example with multiple types of errors
        bad_example = {
            "parent_voxel": np.ones((4, 4, 4), dtype=bool),  # Wrong shape
            "target_mask": np.ones((16, 16, 16), dtype=np.float32),  # Wrong dtype
            # Missing target_types
            "biome_patch": np.ones((16, 16), dtype=np.uint8),
            "heightmap_patch": np.ones((16, 16), dtype=np.uint16),
            "river_patch": np.ones((16, 16), dtype=np.float32),
            "y_index": 50,  # Out of range
            "chunk_x": 10,
            "chunk_z": 15,
            "lod": 1,
        }

        errors = validator.get_validation_errors(bad_example)
        categorized = validator.categorize_errors(errors)

        assert "shape_errors" in categorized
        assert "dtype_errors" in categorized
        assert "missing_key_errors" in categorized
        assert "range_errors" in categorized

    def test_validation_statistics(self):
        """RED: Fails if validation statistics generation is missing."""
        validator = PatchValidator()

        # Mock some validation results
        validation_results = [
            {"valid": True, "errors": []},
            {"valid": False, "errors": ["Shape error", "Missing key"]},
            {"valid": False, "errors": ["Dtype error"]},
            {"valid": True, "errors": []},
        ]

        stats = validator.compute_validation_statistics(validation_results)

        assert stats["total"] == 4
        assert stats["valid"] == 2
        assert stats["invalid"] == 2
        assert stats["success_rate"] == 0.5


class TestErrorHandling:
    """Test error handling in patch validation."""

    def test_handle_corrupted_npz_file(self):
        """RED: Fails if corrupted file handling is missing."""
        validator = PatchValidator()

        # Create corrupted NPZ file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"corrupted data")
            corrupted_file = Path(f.name)

        try:
            is_valid = validator.validate_file(corrupted_file)
            assert is_valid is False

            errors = validator.get_file_errors(corrupted_file)
            assert len(errors) > 0
            assert any(
                "corrupted" in error.lower() or "invalid" in error.lower() for error in errors
            )
        finally:
            corrupted_file.unlink()

    def test_handle_missing_file(self):
        """RED: Fails if missing file handling is incorrect."""
        validator = PatchValidator()

        non_existent_file = Path("does_not_exist.npz")

        is_valid = validator.validate_file(non_existent_file)
        assert is_valid is False

        errors = validator.get_file_errors(non_existent_file)
        assert len(errors) > 0
        assert any("not found" in error.lower() or "missing" in error.lower() for error in errors)

    def test_handle_empty_directory(self):
        """RED: Fails if empty directory handling is missing."""
        validator = PatchValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            report = validator.validate_batch_directory(empty_dir)

            assert report["total_files"] == 0
            assert report["valid_files"] == 0
            assert report["invalid_files"] == 0
