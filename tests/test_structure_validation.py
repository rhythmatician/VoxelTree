"""
Tests for structure data validation.

This module tests the validation logic that ensures structure generation
was enabled during world generation and that structure data is present
for structure-aware fine-tuning.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.extraction.structure_extractor import StructureExtractor, StructureValidationError
from train.fine_tuning import validate_structure_data_for_training


class TestStructureValidation:
    """Test structure data validation functionality."""

    def test_structure_extractor_validation_settings(self):
        """Test that structure extractor loads validation settings correctly."""
        extractor = StructureExtractor()

        # Check validation settings are loaded
        assert hasattr(extractor, "validate_structure_generation")
        assert hasattr(extractor, "min_structure_chunks_ratio")
        assert hasattr(extractor, "_chunks_processed")
        assert hasattr(extractor, "_chunks_with_structures")

        # Check default values
        assert extractor.validate_structure_generation is True
        assert extractor.min_structure_chunks_ratio == 0.1

    def test_structure_statistics_tracking(self):
        """Test that structure statistics are tracked correctly."""
        extractor = StructureExtractor()

        # Initial state
        stats = extractor.get_structure_statistics()
        assert stats["chunks_processed"] == 0
        assert stats["chunks_with_structures"] == 0
        assert stats["structure_ratio"] == 0.0

        # Simulate processing chunks
        extractor._chunks_processed = 10
        extractor._chunks_with_structures = 3

        stats = extractor.get_structure_statistics()
        assert stats["chunks_processed"] == 10
        assert stats["chunks_with_structures"] == 3
        assert stats["structure_ratio"] == 0.3

        # Reset tracking
        extractor.reset_validation_tracking()
        stats = extractor.get_structure_statistics()
        assert stats["chunks_processed"] == 0
        assert stats["chunks_with_structures"] == 0

    def test_validate_structure_data_no_files(self):
        """Test validation fails when no structure files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Mock the StructureExtractor to be enabled
            with patch("train.fine_tuning.StructureExtractor") as mock_extractor_class:
                mock_extractor = MagicMock()
                mock_extractor.enabled = True
                mock_extractor_class.return_value = mock_extractor

                with pytest.raises(StructureValidationError) as exc_info:
                    validate_structure_data_for_training(data_path)

                assert "No structure data files" in str(exc_info.value)

    def test_validate_structure_data_disabled_extraction(self):
        """Test validation fails when structure extraction is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Mock the StructureExtractor to be disabled
            with patch("train.fine_tuning.StructureExtractor") as mock_extractor_class:
                mock_extractor = MagicMock()
                mock_extractor.enabled = False
                mock_extractor_class.return_value = mock_extractor

                with pytest.raises(StructureValidationError) as exc_info:
                    validate_structure_data_for_training(data_path)

                assert "Structure extraction is disabled" in str(exc_info.value)

    def test_validate_structure_data_insufficient_structures(self):
        """Test validation fails when structure files have no actual structure data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create structure files with empty structure data
            for i in range(5):
                file_path = data_path / f"chunk_{i}_structure.npz"
                # Create file with empty structure mask
                structure_mask = np.zeros((8, 8, 1), dtype=np.float32)
                structure_types = np.zeros(10, dtype=np.float32)
                structure_positions = np.zeros(2, dtype=np.float32)

                np.savez_compressed(
                    file_path,
                    structure_mask=structure_mask,
                    structure_types=structure_types,
                    structure_positions=structure_positions,
                )

            # Mock the StructureExtractor to be enabled
            with patch("train.fine_tuning.StructureExtractor") as mock_extractor_class:
                mock_extractor = MagicMock()
                mock_extractor.enabled = True
                mock_extractor_class.return_value = mock_extractor

                with pytest.raises(StructureValidationError) as exc_info:
                    validate_structure_data_for_training(data_path)

                assert "contain actual structure data" in str(exc_info.value)

    def test_validate_structure_data_sufficient_structures(self):
        """Test validation passes when structure files have sufficient structure data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Create structure files with some actual structure data
            for i in range(5):
                file_path = data_path / f"chunk_{i}_structure.npz"

                # Create structure mask with some structures (at least for 2 files)
                structure_mask = np.zeros((8, 8, 1), dtype=np.float32)
                if i < 2:  # 40% of files have structures (above 10% threshold)
                    structure_mask[2:4, 2:4, 0] = 1.0  # Add some structure data

                structure_types = np.zeros(10, dtype=np.float32)
                structure_positions = np.zeros(2, dtype=np.float32)

                np.savez_compressed(
                    file_path,
                    structure_mask=structure_mask,
                    structure_types=structure_types,
                    structure_positions=structure_positions,
                )

            # Mock the StructureExtractor to be enabled
            with patch("train.fine_tuning.StructureExtractor") as mock_extractor_class:
                mock_extractor = MagicMock()
                mock_extractor.enabled = True
                mock_extractor_class.return_value = mock_extractor

                # Should not raise exception
                validate_structure_data_for_training(data_path)

    def test_world_structure_generation_validation(self):
        """Test world structure generation validation."""
        extractor = StructureExtractor()

        with tempfile.TemporaryDirectory() as temp_dir:
            world_path = Path(temp_dir)

            # Test missing level.dat
            with pytest.raises(StructureValidationError) as exc_info:
                extractor.validate_world_structure_generation(world_path)
            assert "level.dat not found" in str(exc_info.value)

    def test_structure_validation_error(self):
        """Test that StructureValidationError can be raised and caught."""
        with pytest.raises(StructureValidationError) as exc_info:
            raise StructureValidationError("Test error message")

        assert "Test error message" in str(exc_info.value)
