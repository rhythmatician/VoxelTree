"""
Structure extraction module for extracting structure information from Minecraft .mca files.

This module provides the StructureExtractor class that handles parsing Minecraft
structure data from region files and prepares it for machine learning input.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import anvil  # type: ignore
import numpy as np

from scripts.worldgen.config import load_config

logger = logging.getLogger(__name__)


class StructureValidationError(Exception):
    """Raised when structure generation validation fails."""

    pass


class StructureExtractor:
    """
    Extracts structure data from Minecraft region files for structure-aware fine-tuning.

    This class handles the extraction of structures (like villages, fortresses, etc.)
    from Minecraft region files and converts them into formats suitable for model inputs.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize StructureExtractor with configuration.

        Args:
            config_path: Path to config.yaml file, defaults to "config.yaml"
        """
        # Load configuration
        if config_path is None:
            config_path = Path("config.yaml")

        self.config = load_config(config_path)
        extraction_config = self.config.get("extraction", {})
        structure_config = extraction_config.get("structures", {})

        # Structure extraction settings
        self.enabled = structure_config.get("enabled", False)
        self.mask_resolution = structure_config.get("mask_resolution", 8)
        self.structure_types = structure_config.get(
            "structure_types",
            [
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
        )
        self.position_encoding = structure_config.get("position_encoding", "normalized_offset")

        # Validation settings
        self.validate_structure_generation = structure_config.get(
            "validate_structure_generation", True
        )
        self.min_structure_chunks_ratio = structure_config.get(
            "min_structure_chunks_ratio", 0.1
        )  # At least 10% of chunks should have structures

        # Tracking for validation
        self._chunks_processed = 0
        self._chunks_with_structures = 0

        logger.info(f"StructureExtractor initialized with enabled={self.enabled}")

    def extract_structure_data(
        self, region_file: Path, chunk_x: int, chunk_z: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract structure data from a specific chunk in a region file.

        Args:
            region_file: Path to .mca region file
            chunk_x: Chunk X coordinate (within the region)
            chunk_z: Chunk Z coordinate (within the region)

        Returns:
            Dictionary containing structure data:
            - structure_mask: Binary mask of structure presence (resolution×resolution×1)
            - structure_types: One-hot encoded structure types
            - structure_positions: Normalized position offsets [x, z]
        """
        if not self.enabled:
            # Return empty structure data if disabled
            return {
                "structure_mask": np.zeros(
                    (self.mask_resolution, self.mask_resolution, 1), dtype=np.float32
                ),
                "structure_types": np.zeros((len(self.structure_types),), dtype=np.float32),
                "structure_positions": np.zeros((2,), dtype=np.float32),
            }

        try:
            # Open region file with anvil-parser
            region = anvil.Region.from_file(region_file)

            # Get chunk data
            chunk = region.get_chunk(chunk_x, chunk_z)

            # Get chunk NBT data
            chunk_nbt = chunk.get_nbt()
            # Extract structure references from NBT data
            structures_found = []
            structure_positions = []

            # Validate that structure generation was enabled when the world was created
            has_structure_data = False

            if "Structures" in chunk_nbt and "References" in chunk_nbt["Structures"]:
                structure_refs = chunk_nbt["Structures"]["References"]
                if structure_refs:  # Check if References dict is not empty
                    has_structure_data = True

                # Process each structure type in the chunk
                for structure_key, structure_data in structure_refs.items():
                    # Remove minecraft: prefix if present
                    structure_type = structure_key.replace("minecraft:", "")

                    # Debug log the structure type we found
                    logger.debug(f"Found structure: {structure_type}")
                    # Convert to standardized type (e.g., village_plains -> village)
                    matched = False
                    for base_type in self.structure_types:
                        # Check if base_type is a substring of structure_type
                        if base_type in structure_type:
                            structures_found.append(base_type)
                            matched = True
                            logger.debug(f"Matched '{structure_type}' to base type '{base_type}'")

                            # Extract positions
                            if "References" in structure_data and structure_data["References"]:
                                for reference in structure_data["References"]:
                                    if "Pos" in reference:
                                        structure_positions.append(np.array(reference["Pos"]))

                            # Once matched, break out of the loop to avoid multiple matches
                            break
                    if not matched:
                        logger.warning(
                            f"Couldn't match structure type: {structure_type} to any base type"
                        )  # Log warning if no structure data was found (may indicate generate-structures=false)
            if not has_structure_data:
                logger.debug(
                    f"No structure data found in chunk ({chunk_x}, {chunk_z}). "
                    "This may be normal for some chunks, or could indicate that "
                    "the world was generated with generate-structures=false."
                )

            # Update validation tracking
            self._chunks_processed += 1
            if has_structure_data:
                self._chunks_with_structures += 1

            # Check if we should validate structure generation
            if (
                self.validate_structure_generation
                and self._chunks_processed > 0
                and self._chunks_processed % 100 == 0
            ):  # Check every 100 chunks
                self._check_structure_generation_ratio()

            # Create structure mask (8×8×1 resolution)
            structure_mask = np.zeros(
                (self.mask_resolution, self.mask_resolution, 1), dtype=np.float32
            )

            # Populate structure mask if structures were found
            if structure_positions:
                for pos in structure_positions:
                    mask = self.create_structure_mask(pos, self.mask_resolution, chunk_size=16)
                    # Combine masks with maximum value
                    structure_mask = np.maximum(structure_mask, mask)

            # Encode structure types as one-hot vector
            structure_types = self.encode_structure_types(structures_found)

            # Calculate normalized position offsets
            chunk_center = np.array([8, 0, 8])  # Center of a 16×16 chunk
            normalized_positions = np.zeros((2,), dtype=np.float32)

            if structure_positions:
                normalized_positions = self.normalize_structure_positions(
                    np.array(structure_positions), chunk_center
                )

            return {
                "structure_mask": structure_mask,
                "structure_types": structure_types,
                "structure_positions": normalized_positions,
            }

        except Exception as e:
            logger.warning(
                f"Failed to extract structure data for chunk ({chunk_x}, {chunk_z}): {e}"
            )
            # Return empty structure data on error
            return {
                "structure_mask": np.zeros(
                    (self.mask_resolution, self.mask_resolution, 1), dtype=np.float32
                ),
                "structure_types": np.zeros((len(self.structure_types),), dtype=np.float32),
                "structure_positions": np.zeros((2,), dtype=np.float32),
            }

    def create_structure_mask(
        self, position: np.ndarray, resolution: int, chunk_size: int = 16
    ) -> np.ndarray:
        """
        Create a spatial binary mask showing structure presence.

        Args:
            position: [x, y, z] position of the structure
            resolution: Size of the mask grid (typically 8×8)
            chunk_size: Size of a Minecraft chunk (typically 16×16)

        Returns:
            Binary mask with 1s near structure positions
        """
        # Create empty mask
        mask = np.zeros((resolution, resolution, 1), dtype=np.float32)

        # Scale structure position to mask resolution
        scale_factor = resolution / chunk_size
        x_idx = int(position[0] * scale_factor) % resolution
        z_idx = int(position[2] * scale_factor) % resolution

        # Handle out-of-bounds indices
        x_idx = max(0, min(x_idx, resolution - 1))
        z_idx = max(0, min(z_idx, resolution - 1))

        # Mark structure position in mask
        mask[x_idx, z_idx, 0] = 1.0

        return mask

    def encode_structure_types(self, structures: List[str]) -> np.ndarray:
        """
        Create one-hot encoding of structure types.

        Args:
            structures: List of structure types found in the chunk

        Returns:
            One-hot encoded vector of structure types
        """
        # Initialize one-hot vector
        one_hot = np.zeros((len(self.structure_types),), dtype=np.float32)

        # Set 1s for each structure type found
        for structure in structures:
            if structure in self.structure_types:
                idx = self.structure_types.index(structure)
                one_hot[idx] = 1.0

        return one_hot

    def normalize_structure_positions(
        self, positions: np.ndarray, chunk_center: np.ndarray
    ) -> np.ndarray:
        """
        Normalize structure positions relative to chunk center.

        Args:
            positions: Array of structure positions [[x, y, z], ...]
            chunk_center: Center coordinates of the chunk [x, y, z]

        Returns:
            Normalized [x, z] coordinates in range [-1, 1]
        """
        if len(positions) == 0:
            return np.zeros((2,), dtype=np.float32)

        # Calculate center position if multiple structures
        if len(positions) > 1:
            mean_pos = positions.mean(axis=0)
        else:
            mean_pos = positions[0]

        # Calculate relative offset from chunk center
        offset_x = (mean_pos[0] - chunk_center[0]) / (chunk_center[0])
        offset_z = (mean_pos[2] - chunk_center[2]) / (chunk_center[2])

        # Clip to [-1, 1] range
        offset_x = np.clip(offset_x, -1.0, 1.0)
        offset_z = np.clip(offset_z, -1.0, 1.0)

        return np.array([offset_x, offset_z], dtype=np.float32)

    def _check_structure_generation_ratio(self) -> None:
        """
        Check if the ratio of chunks with structures meets the minimum threshold.

        Raises:
            StructureValidationError: If structure ratio is below threshold
        """
        if self._chunks_processed == 0:
            return

        structure_ratio = self._chunks_with_structures / self._chunks_processed

        logger.info(
            f"Structure validation: {self._chunks_with_structures}/{self._chunks_processed} "
            f"chunks have structures ({structure_ratio:.3f} ratio)"
        )

        if structure_ratio < self.min_structure_chunks_ratio:
            raise StructureValidationError(
                f"Only {structure_ratio:.3f} of chunks contain structure data, "
                f"which is below the minimum threshold of {self.min_structure_chunks_ratio}. "
                f"This likely indicates that the world was generated with "
                f"'generate-structures=false' in server.properties. "
                f"Structure-aware fine-tuning requires worlds with structure generation enabled."
            )

    def validate_world_structure_generation(self, world_path: Path) -> None:
        """
        Validate that a world was generated with structure generation enabled.

        Args:
            world_path: Path to the world directory containing level.dat

        Raises:
            StructureValidationError: If validation fails
        """
        level_dat_path = world_path / "level.dat"

        if not level_dat_path.exists():
            raise StructureValidationError(
                f"level.dat not found at {level_dat_path}. "
                f"Cannot validate structure generation settings."
            )

        try:
            # Try to read level.dat with anvil-parser
            import anvil

            level_nbt = anvil.read_level_dat(level_dat_path)
            # Check if structure generation was enabled
            data = level_nbt.get("Data", {})

            # Check if generateStructures flag is disabled
            generate_structures = data.get("generateStructures", True)

            if not generate_structures:
                raise StructureValidationError(
                    f"World at {world_path} was generated with generateStructures=false. "
                    f"Structure-aware fine-tuning requires worlds with structure generation enabled."
                )

            logger.info("World validation passed: structure generation is enabled")

        except Exception as e:
            logger.warning(
                f"Could not validate structure generation settings for world at {world_path}: {e}. "
                f"Proceeding with extraction but structure validation may fail later."
            )

    def get_structure_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about structure extraction progress.

        Returns:
            Dictionary with structure extraction statistics
        """
        if self._chunks_processed == 0:
            structure_ratio = 0.0
        else:
            structure_ratio = self._chunks_with_structures / self._chunks_processed

        return {
            "chunks_processed": self._chunks_processed,
            "chunks_with_structures": self._chunks_with_structures,
            "structure_ratio": structure_ratio,
            "min_required_ratio": self.min_structure_chunks_ratio,
            "validation_enabled": self.validate_structure_generation,
        }

    def reset_validation_tracking(self) -> None:
        """Reset validation tracking counters."""
        self._chunks_processed = 0
        self._chunks_with_structures = 0
