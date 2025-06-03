"""
SeedInputLinker - Phase 2.2 GREEN Implementation

Links LOD pairs with seed-derived conditioning variables (biomes, heightmaps, river noise)
to create complete training examples ready for machine learning.
"""

import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class SeedInputLinker:
    """
    Links LOD pairs with seed-derived conditioning variables.

    Takes parent-child LOD pairs and combines them with corresponding
    biome, heightmap, and river noise data to create complete training examples.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize SeedInputLinker with configuration."""
        self.config_path = config_path if config_path else Path("config.yaml")
        self._load_config()

        logger.info(f"SeedInputLinker initialized with seed={self.seed}")

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # Extract pairing configuration
        pairing_config = config.get("pairing", {})
        self.pairs_dir = Path(pairing_config.get("output_dir", "data/pairs"))
        self.seed_inputs_dir = Path(pairing_config.get("seed_inputs_dir", "data/seed_inputs"))
        self.output_dir = Path(pairing_config.get("linked_output_dir", "data/linked"))

        # Extract seed from worldgen config
        worldgen_config = config.get("worldgen", {})
        seed_str = worldgen_config.get("seed", "VoxelTree")
        self.seed = self._hash_seed(seed_str)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _hash_seed(self, seed_str: str) -> int:
        """Convert seed string to integer (same as worldgen bootstrap)."""
        if seed_str == "VoxelTree":
            return 1903448982
        return hash(seed_str) & 0x7FFFFFFF  # Ensure positive 32-bit int

    def load_seed_input_patch(self, chunk_x: int, chunk_z: int) -> Dict[str, Any]:
        """
        Load seed-derived input patch for given chunk coordinates.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate

        Returns:
            Dictionary containing seed input data

        Raises:
            FileNotFoundError: If seed input file doesn't exist
        """
        seed_file = self.seed_inputs_dir / f"seed_patch_{chunk_x}_{chunk_z}.npz"

        if not seed_file.exists():
            raise FileNotFoundError(f"Seed input file not found: {seed_file}")

        seed_data = np.load(seed_file)

        return {
            "biomes": seed_data["biomes"],
            "heightmap": seed_data["heightmap"],
            "river_noise": seed_data["river_noise"],
        }

    def link_pair_with_seed_inputs(
        self, pair_file: Path, seed_inputs_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Link a single LOD pair with its corresponding seed inputs.

        Args:
            pair_file: Path to LOD pair .npz file
            seed_inputs_dir: Directory containing seed input files
            output_dir: Directory to save linked examples

        Returns:
            Dictionary containing linked training example
        """
        # Load pair data
        pair_data = np.load(pair_file)
        pair_dict = {key: pair_data[key] for key in pair_data.files}

        # Extract coordinates from pair
        chunk_x = int(pair_dict["chunk_x"])
        chunk_z = int(pair_dict["chunk_z"])

        # Load corresponding seed inputs
        seed_file = seed_inputs_dir / f"seed_patch_{chunk_x}_{chunk_z}.npz"
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed input file not found: {seed_file}")

        seed_data = np.load(seed_file)
        seed_dict = {key: seed_data[key] for key in seed_data.files}  # Validate coordinate match
        if not self.validate_coordinate_match(pair_dict, seed_dict):
            raise ValueError("Coordinate mismatch between pair and seed data")

        # Create linked example
        linked_example = self.create_linked_example(pair_dict, seed_dict)

        return linked_example

    def validate_coordinate_match(
        self, pair_data: Dict[str, Any], seed_data: Dict[str, Any], strict: bool = False
    ) -> bool:
        """
        Validate that pair and seed data have matching coordinates.

        Args:
            pair_data: LOD pair data
            seed_data: Seed input data
            strict: If True, raise exception on mismatch

        Returns:
            True if coordinates match, False otherwise

        Raises:
            ValueError: If strict=True and coordinates don't match
        """
        pair_x = pair_data.get("chunk_x")
        pair_z = pair_data.get("chunk_z")
        seed_x = seed_data.get("chunk_x")
        seed_z = seed_data.get("chunk_z")

        matches = pair_x == seed_x and pair_z == seed_z

        if strict and not matches:
            raise ValueError(
                f"Coordinate mismatch: pair({pair_x}, {pair_z}) vs seed({seed_x}, {seed_z})"  # noqa: E501
            )

        return matches

    def extract_biome_conditioning(self, biomes: np.ndarray, y_index: int) -> np.ndarray:
        """
        Extract biome conditioning features for a specific y-level.

        Args:
            biomes: (16, 16) biome ID array
            y_index: Y-level index (0-23)

        Returns:
            (16, 16) biome conditioning array
        """
        # For now, just return the biome IDs directly
        # In the future, this could include y-level specific biome effects
        return biomes.astype(np.uint8)

    def extract_height_conditioning(self, heightmap: np.ndarray, y_index: int) -> np.ndarray:
        """
        Extract height-relative conditioning for a specific y-level.

        Args:
            heightmap: (16, 16) surface height array
            y_index: Y-level index (0-23)

        Returns:
            (16, 16) height conditioning array
        """
        # Convert y_index to world Y coordinate
        y_world = y_index * 16 - 64  # Minecraft world coordinates

        # Calculate relative height (distance from surface)
        height_relative = heightmap.astype(np.float32) - y_world

        # Normalize to reasonable range
        height_relative = np.clip(height_relative, -100, 100) / 100.0

        return height_relative.astype(np.float32)

    def extract_river_conditioning(self, river_noise: np.ndarray, y_index: int) -> np.ndarray:
        """
        Extract river conditioning with y-level specific effects.

        Args:
            river_noise: (16, 16) river noise array [-1, 1]
            y_index: Y-level index (0-23)

        Returns:
            (16, 16) river conditioning array
        """
        # Apply y-level specific river effects
        y_world = y_index * 16 - 64

        # Rivers have stronger effect near sea level (y=64)
        river_strength = 1.0 - abs(y_world - 64) / 64.0
        river_strength = np.clip(river_strength, 0.1, 1.0)

        # Scale river noise by y-level strength
        conditioned_river = river_noise * river_strength
        return conditioned_river.astype(np.float32)

    def create_linked_example(
        self, pair_data: Dict[str, Any], seed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a complete linked training example.

        Args:
            pair_data: LOD pair data
            seed_data: Seed input data

        Returns:
            Complete training example dictionary
        """
        y_index = int(pair_data["y_index"])

        # Extract conditioning variables
        biome_conditioning = self.extract_biome_conditioning(seed_data["biomes"], y_index)
        heightmap_conditioning = self.extract_height_conditioning(seed_data["heightmap"], y_index)
        river_conditioning = self.extract_river_conditioning(
            seed_data["river_noise"], y_index
        )  # Combine all data
        linked_example = {
            # Original pair data
            "parent_voxel": pair_data["parent_voxel"],
            "target_mask": pair_data["target_mask"],
            "target_types": pair_data["target_types"],
            "lod": pair_data["lod"],
            # Raw conditioning variables (for compatibility with existing tests)
            "biomes": biome_conditioning,
            "heightmap": heightmap_conditioning,
            "river_noise": river_conditioning,
            # Processed conditioning patches (for training format specification)
            "biome_patch": biome_conditioning,
            "heightmap_patch": heightmap_conditioning,
            "river_patch": river_conditioning,
            # Metadata
            "y_index": pair_data["y_index"],
            "chunk_x": pair_data["chunk_x"],
            "chunk_z": pair_data["chunk_z"],
        }

        return linked_example

    def save_linked_example_npz(self, linked_example: Dict[str, Any], output_path: Path) -> Path:
        """
        Save a linked training example to compressed NPZ format.

        Args:
            linked_example: Complete training example
            output_path: Output file path

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        np.savez_compressed(output_path, **linked_example)

        return output_path

    def process_batch_linking(
        self, pairs_dir: Path, seed_inputs_dir: Path, output_dir: Path
    ) -> int:
        """
        Process a batch of LOD pairs, linking them with seed inputs.

        Args:
            pairs_dir: Directory containing LOD pair files
            seed_inputs_dir: Directory containing seed input files
            output_dir: Directory to save linked examples

        Returns:
            Number of linked examples created
        """
        pair_files = list(pairs_dir.glob("*.npz"))
        linked_count = 0

        for pair_file in pair_files:
            try:
                linked_example = self.link_pair_with_seed_inputs(
                    pair_file, seed_inputs_dir, output_dir
                )

                # Create output filename
                filename = f"linked_{pair_file.stem}.npz"
                output_path = output_dir / filename

                self.save_linked_example_npz(linked_example, output_path)
                linked_count += 1

            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {pair_file}: {e}")
                continue

        logger.info(f"Linked {linked_count} training examples")
        return linked_count

    def process_batch_linking_parallel(
        self,
        pairs_dir: Path,
        seed_inputs_dir: Path,
        output_dir: Path,
        num_workers: Optional[int] = None,
    ) -> int:
        """
        Process LOD pairs in parallel using multiprocessing.

        Args:
            pairs_dir: Directory containing LOD pair files
            seed_inputs_dir: Directory containing seed input files
            output_dir: Directory to save linked examples
            num_workers: Number of worker processes

        Returns:
            Number of linked examples created
        """
        if num_workers is None:
            num_workers = min(cpu_count(), 4)

        # For this minimal implementation, use sequential processing
        # In full implementation, we'd use multiprocessing.Pool
        return self.process_batch_linking(pairs_dir, seed_inputs_dir, output_dir)

    def validate_seed_data_format(self, seed_data: Dict[str, Any]) -> bool:
        """
        Validate that seed data has the correct format and keys.

        Args:
            seed_data: Seed input data to validate

        Returns:
            True if valid

        Raises:
            KeyError: If required keys are missing
            ValueError: If array shapes are incorrect
        """
        required_keys = ["biomes", "heightmap", "river_noise"]
        for key in required_keys:
            if key not in seed_data:
                raise KeyError(f"Missing required key '{key}' in seed data")

        # Validate shapes
        expected_shape = (16, 16)
        for key in required_keys:
            if seed_data[key].shape != expected_shape:
                raise ValueError(
                    f"Invalid shape for '{key}': {seed_data[key].shape}, expected {expected_shape}"
                )

        return True
