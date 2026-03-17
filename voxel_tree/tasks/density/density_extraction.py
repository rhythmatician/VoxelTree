#!/usr/bin/env python3
"""
Data Extraction Pipeline for OGN Terrain Training

Extracts ground truth data from a running Minecraft server:
  - Surface heightmap (slope, ridges patterns)
  - Biome IDs
  - Block states
  - NEW: Density field values (slopedCheese)
  - NEW: Material categories

Usage:
  python data-cli.py extract --seed=12345 --chunks=100 --output=training_data/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DensityFieldExtractor:
    """
    Extracts density field ground truth from Minecraft world.

    Requires a running Fabric server with LODiffusion mod installed
    to access the live NoiseRouter.
    """

    def __init__(self, server_url: str = "http://localhost:25575"):
        """
        Args:
            server_url: URL of the Minecraft server API (Rcon or custom endpoint)
        """
        self.server_url = server_url
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract_density_values(
        self, chunk_x: int, chunk_z: int, y_range: Tuple[int, int] = (-64, 320)
    ) -> np.ndarray:
        """
        Extract slopedCheese density values for all blocks in a chunk.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            y_range: (min_y, max_y) inclusive

        Returns:
            numpy array of shape (16, y_height, 16) with float32 density values
        """
        min_y, max_y = y_range
        y_height = max_y - min_y + 1

        density_field = np.zeros((16, y_height, 16), dtype=np.float32)

        # For each block in the chunk
        for local_x in range(16):
            for local_z in range(16):
                for local_y in range(y_height):
                    block_x = chunk_x * 16 + local_x
                    block_y = min_y + local_y
                    block_z = chunk_z * 16 + local_z

                    try:
                        # Query the server for density at this position
                        # This would require a Rcon command or HTTP endpoint
                        # that exposes NoiseRouter.slopedCheese evaluation
                        density = self._query_density(block_x, block_y, block_z)
                        density_field[local_x, local_y, local_z] = density
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract density at ({block_x}, {block_y}, {block_z}): {e}"
                        )
                        density_field[local_x, local_y, local_z] = 0.0

        return density_field

    def _query_density(self, block_x: int, block_y: int, block_z: int) -> float:
        """
        Query the Minecraft server for density at a block position.

        Placeholder implementation - in practice, this would:
        1. Connect to a Rcon console or custom API
        2. Call a command that evaluates NoiseRouter.slopedCheese
        3. Parse and return the result

        Example Fabric Rcon command:
          /terrain density <x> <y> <z>

        This command would be registered by LODiffusion mod.
        """
        # TODO: Implement actual server query
        # For now, return placeholder
        raise NotImplementedError(
            "Density query requires Fabric server with Rcon/API endpoint. "
            "Implement _query_density() to query NoiseRouter."
        )


class MaterialCategoryMapper:
    """
    Maps vanilla Minecraft block IDs to semantic material categories.
    """

    def __init__(self, categories_json: Path):
        """
        Args:
            categories_json: Path to material_categories.json schema
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.categories = self._load_categories(categories_json)
        self.block_to_category = self._build_block_map()

    def _load_categories(self, path: Path) -> Dict:
        """Load material categories from JSON schema."""
        with open(path, "r") as f:
            return json.load(f)

    def _build_block_map(self) -> Dict[str, int]:
        """Build a mapping from block name to category ID."""
        block_map = {}

        for category_name, category_info in self.categories["categories"].items():
            cat_id = category_info["id"]

            for block_name in category_info.get("blocks", []):
                block_map[block_name] = cat_id

        # Add fallback for unknown blocks
        block_map["__default__"] = self.categories["categories"]["other_solid"]["id"]

        return block_map

    def get_material_id(self, block_name: str) -> int:
        """
        Get material category ID for a block.

        Args:
            block_name: Minecraft block name (e.g., 'oak_log', 'stone')

        Returns:
            int: Material category ID (0-12)
        """
        return self.block_to_category.get(block_name, block_map["__default__"])


class DatasetBuilder:
    """
    Builds training dataset from extracted Minecraft data.

    Outputs:
      - heightmap_[chunks].npy (5, H, W) float32
      - biome_[chunks].npy (H, W) int64
      - density_[chunks].npy (H, Y, W) float32  [NEW]
      - material_[chunks].npy (H, Y, W) int64   [NEW]
      - metadata_[chunks].json
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save dataset files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_chunk(
        self,
        chunk_idx: int,
        heightmap: np.ndarray,
        biome: np.ndarray,
        density: np.ndarray,
        material: np.ndarray,
        metadata: Dict,
    ):
        """
        Save a chunk's extracted data to disk.

        Args:
            chunk_idx: Chunk index in dataset
            heightmap: (5, 16, 16) float32
            biome: (16, 16) int64
            density: (16, Y, 16) float32
            material: (16, Y, 16) int64
            metadata: dict with chunk_x, chunk_z, seed, etc.
        """
        prefix = f"chunk_{chunk_idx:06d}"

        # Save numpy arrays
        np.save(self.output_dir / f"{prefix}_heightmap.npy", heightmap)
        np.save(self.output_dir / f"{prefix}_biome.npy", biome)
        np.save(self.output_dir / f"{prefix}_density.npy", density)
        np.save(self.output_dir / f"{prefix}_material.npy", material)

        # Save metadata
        with open(self.output_dir / f"{prefix}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved chunk {chunk_idx}")

    def create_manifest(self, num_chunks: int, seed: int, description: str = ""):
        """
        Create a manifest file listing all chunks in the dataset.
        """
        manifest = {
            "dataset_format": "voxeltree_training_v1",
            "num_chunks": num_chunks,
            "seed": seed,
            "description": description,
            "date_created": str(Path.cwd()),  # Placeholder
            "input_formats": {
                "heightmap": "(5, 16, 16) float32 — surface features",
                "biome": "(16, 16) int64 — biome IDs",
                "density": "(16, Y, 16) float32 — slopedCheese density",
                "material": "(16, Y, 16) int64 — material category (0-12)",
            },
            "notes": [
                "density > 0 indicates solid blocks",
                "density ≤ 0 indicates air/fluid",
                "material is only meaningful where density > 0",
                "material categories defined in VoxelTree/schema/material_categories.json",
            ],
        }

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Created manifest for {num_chunks} chunks")


def main():
    """CLI entry point for data extraction."""
    parser = argparse.ArgumentParser(
        description="Extract terrain training data from Minecraft server"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Extract command
    extract = subparsers.add_parser("extract", help="Extract data from server")
    extract.add_argument("--seed", type=int, required=True, help="World seed")
    extract.add_argument("--chunks", type=int, default=100, help="Number of chunks")
    extract.add_argument("--output", type=Path, default="training_data/", help="Output directory")
    extract.add_argument("--server", default="http://localhost:25575", help="Minecraft Rcon URL")

    # Validate command
    validate = subparsers.add_parser("validate", help="Validate extracted dataset")
    validate.add_argument("--dir", type=Path, required=True, help="Dataset directory")

    args = parser.parse_args()

    if args.command == "extract":
        logger.info(f"Extracting {args.chunks} chunks from seed {args.seed}")
        logger.info("This requires a running Fabric server with LODiffusion mod")
        logger.info(f"Output directory: {args.output}")

        # TODO: Implement extraction pipeline
        logger.warning(
            "Extraction pipeline not yet implemented. "
            "This requires Fabric server support for Rcon density queries."
        )

    elif args.command == "validate":
        logger.info(f"Validating dataset at {args.dir}")
        # TODO: Implement validation

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
