"""
Tests for structure data extraction from Minecraft .mca files.
Phase 4: Structure-Aware Fine-Tuning - RED phase

This module tests the extraction of structure data from Minecraft region files
for use in structure-aware fine-tuning.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts.extraction.chunk_extractor import ChunkExtractor
from scripts.extraction.structure_extractor import StructureExtractor


class TestStructureExtractor:
    """Test suite for structure extraction functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_temp_dir = Path(tempfile.mkdtemp())

        # Create test config with temp directories
        self.test_config = {
            "extraction": {
                "output_dir": str(self.test_temp_dir / "chunks"),
                "temp_dir": str(self.test_temp_dir / "temp"),
                "structures": {
                    "enabled": True,
                    "mask_resolution": 8,  # Resolution of structure masks (8x8)
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
                    "position_encoding": "normalized_offset",  # or "absolute"
                },
            }
        }

        # Create mock config file
        self.test_config_path = self.test_temp_dir / "test_config.yaml"

        # Write test config to file
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.safe_dump(self.test_config, f)

        # Initialize extractors
        self.extractor = ChunkExtractor(config_path=self.test_config_path)
        self.structure_extractor = StructureExtractor(config_path=self.test_config_path)

    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)

    def test_structure_extractor_init(self):
        """Test StructureExtractor initialization and config loading."""
        assert self.structure_extractor.enabled is True
        assert self.structure_extractor.mask_resolution == 8
        assert "village" in self.structure_extractor.structure_types
        assert self.structure_extractor.position_encoding == "normalized_offset"

    @patch("anvil.Region.from_file")
    def test_extract_structure_data_from_chunk(self, mock_from_file):
        """Test extraction of structure data from a chunk with structures."""
        # Mock Region and chunk objects
        mock_region = Mock()
        mock_chunk = Mock()

        # Set up from_file to return our mock region
        mock_from_file.return_value = mock_region

        # Mock structure NBT data
        mock_structure_data = {
            "References": {
                "minecraft:village_plains": {"References": [{"Pos": [4, 0, 12]}]},
                "minecraft:fortress": {"References": [{"Pos": [-32, 70, 64]}]},
            }
        }

        # Set up mock returns
        mock_chunk.get_nbt.return_value = {"Structures": mock_structure_data}
        mock_region.get_chunk.return_value = mock_chunk

        # Create mock .mca file
        mock_mca_path = self.test_temp_dir / "r.0.0.mca"
        mock_mca_path.touch()

        # Extract structure data
        structure_data = self.structure_extractor.extract_structure_data(
            mock_mca_path, chunk_x=0, chunk_z=0
        )

        # Check results
        assert "structure_mask" in structure_data
        assert "structure_types" in structure_data
        assert "structure_positions" in structure_data

        # Verify correct shape
        assert structure_data["structure_mask"].shape == (
            self.structure_extractor.mask_resolution,
            self.structure_extractor.mask_resolution,
            1,
        )

        # Verify structure type encoding
        assert structure_data["structure_types"].shape == (
            len(self.structure_extractor.structure_types),
        )

        # Verify a village type was detected
        village_idx = self.structure_extractor.structure_types.index("village")
        assert structure_data["structure_types"][village_idx] == 1

        # Verify position encoding
        assert structure_data["structure_positions"].shape == (2,)  # [x, z] offset

    def test_create_structure_mask(self):
        """Test creation of structure mask from structure positions."""
        # Create test structure position (in chunk coordinates)
        structure_pos = np.array([4, 0, 12])  # x, y, z

        # Create structure mask
        mask = self.structure_extractor.create_structure_mask(
            structure_pos, resolution=8, chunk_size=16
        )

        # Check mask shape
        assert mask.shape == (8, 8, 1)

        # Check that mask has 1s near the structure position
        x_idx = int(structure_pos[0] / 16 * 8)
        z_idx = int(structure_pos[2] / 16 * 8)
        assert mask[x_idx, z_idx, 0] == 1

    def test_encode_structure_types(self):
        """Test one-hot encoding of structure types."""
        # Test with a village and fortress
        structures = ["village", "fortress"]

        # Encode structure types
        encoded_types = self.structure_extractor.encode_structure_types(structures)

        # Verify shape
        assert encoded_types.shape == (len(self.structure_extractor.structure_types),)

        # Check that village and fortress are marked as 1
        village_idx = self.structure_extractor.structure_types.index("village")
        fortress_idx = self.structure_extractor.structure_types.index("fortress")
        assert encoded_types[village_idx] == 1
        assert encoded_types[fortress_idx] == 1

        # Other types should be 0
        assert encoded_types.sum() == 2

    def test_normalize_structure_positions(self):
        """Test normalization of structure positions relative to patch center."""
        # Structure at chunk coordinates [4, 0, 12]
        structure_pos = np.array([[4, 0, 12]])

        # Chunk center is at [8, 0, 8]
        chunk_center = np.array([8, 0, 8])

        # Normalize positions
        normalized_pos = self.structure_extractor.normalize_structure_positions(
            structure_pos, chunk_center
        )

        # Check shape
        assert normalized_pos.shape == (2,)  # [x, z] offset

        # Check normalization (-1 to 1 range)
        assert -1 <= normalized_pos[0] <= 1  # x offset
        assert -1 <= normalized_pos[1] <= 1  # z offset

        # Check specific values (should be negative x, positive z)
        assert normalized_pos[0] < 0  # Structure is left of center
        assert normalized_pos[1] > 0  # Structure is below center
