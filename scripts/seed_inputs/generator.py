"""
SeedInputGenerator - Generate conditioning variables from seed and coordinates

This module implements seed-based generation of biomes, heightmaps, and river noise
without reading .mca files. Uses deterministic vanilla-accurate biome generation
and procedural noise for heightmaps and rivers.
"""

import logging
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import opensimplex
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SeedInputGenerator:
    """
    Generate conditioning variables from seed and coordinates alone.

    Uses deterministic vanilla-accurate biome generation and procedural noise
    for heightmaps and river features for ML model conditioning.
    """

    seed: int
    config_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize generators with seed and load configuration."""
        # Load configuration
        self._load_config()

        # Initialize biome generation system
        self._init_biome_generator()

        # Create noise generators for non-biome features
        self.height_noise = opensimplex.OpenSimplex(seed=self.seed + 1)
        self.river_noise = opensimplex.OpenSimplex(seed=self.seed + 2)

        logger.info(
            f"SeedInputGenerator initialized with seed={self.seed}, biome_source={self.biome_source}"
        )

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            # Default to config.yaml in project root
            self.config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                self.config = config.get("seed_inputs", {})
        else:
            logger.warning(
                f"Config file not found at {self.config_path}, using defaults"
            )
            self.config = {}

        # Set biome generation parameters
        self.biome_source = self.config.get("biome_source", "vanilla")
        self.resolution = self.config.get("resolution", 4)

        # Set noise parameters (only for height and river)
        noise_params = self.config.get("noise_parameters", {})
        self.height_scale = noise_params.get("height_scale", 0.01)
        self.river_scale = noise_params.get("river_scale", 0.005)

        height_params = self.config.get("height_parameters", {})
        self.base_height = height_params.get("base_height", 64)
        self.height_variation = height_params.get("height_variation", 60)
        self.min_height = height_params.get("min_height", 0)
        self.max_height = height_params.get("max_height", 384)

        # Vanilla biome generation settings
        vanilla_config = self.config.get("vanilla_biome", {})
        self.java_tool = vanilla_config.get("java_tool", "tools/amidst-cli.jar")
        self.fallback_tool = vanilla_config.get("fallback_tool", "tools/cubeseed.jar")
        self.cache_dir = Path(vanilla_config.get("cache_dir", "data/biome_cache"))
        self.chunk_batch_size = vanilla_config.get("chunk_batch_size", 64)

    def _init_biome_generator(self):
        """Initialize the biome generation system."""
        if self.biome_source == "vanilla":
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Check if Java tools are available
            self.java_tool_path = Path(self.java_tool)
            self.fallback_tool_path = Path(self.fallback_tool)

            if (
                not self.java_tool_path.exists()
                and not self.fallback_tool_path.exists()
            ):
                logger.warning(
                    "No vanilla biome tools found, falling back to noise-based biomes"
                )
                self.biome_source = "noise"
                self._init_noise_biomes()
        else:
            self._init_noise_biomes()

    def _init_noise_biomes(self):
        """Initialize noise-based biome generation as fallback."""
        self.biome_noise = opensimplex.OpenSimplex(seed=self.seed)
        self.biome_scale = 0.001  # Large scale for biome regions

        # Simple biome mapping for fallback
        self.biome_ranges = [
            ([0.0, 0.1], 16),  # Beach
            ([0.1, 0.2], 7),  # River
            ([0.2, 0.4], 1),  # Plains
            ([0.4, 0.6], 4),  # Forest
            ([0.6, 0.75], 18),  # Forest Hills
            ([0.75, 0.9], 3),  # Extreme Hills
            ([0.9, 1.0], 12),  # Ice Plains
        ]

    def get_biome(self, x: int, z: int) -> int:
        """
        Get biome ID for given world coordinates.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Biome ID (0-255, compatible with Minecraft)
        """
        # Use low-frequency noise for large biome regions
        noise_value = self.biome_noise.noise2(
            x * self.biome_scale, z * self.biome_scale
        )  # Map noise [-1, 1] to biome IDs using configured ranges
        normalized = (noise_value + 1) / 2  # Map to [0, 1]

        for (range_min, range_max), biome_id in self.biome_ranges:
            if range_min <= normalized < range_max:
                return biome_id

        # Fallback to last biome if no range matches (should not happen)
        return self.biome_ranges[-1][1]

    def get_heightmap(self, x: int, z: int) -> int:
        """
        Get surface height for given world coordinates.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Surface height (0-384, compatible with Minecraft)
        """
        # Use medium-frequency noise for realistic terrain
        noise_value = self.height_noise.noise2(
            x * self.height_scale, z * self.height_scale
        )

        # Add some octaves for more realistic terrain
        octave1 = (
            self.height_noise.noise2(
                x * self.height_scale * 2, z * self.height_scale * 2
            )
            * 0.5
        )
        octave2 = (
            self.height_noise.noise2(
                x * self.height_scale * 4, z * self.height_scale * 4
            )
            * 0.25
        )

        combined_noise = noise_value + octave1 + octave2

        # Map to height range [0, 384]
        height = self.base_height + (combined_noise * self.height_variation)

        # Clamp to valid range
        height = max(0, min(384, int(height)))

        return height

    def get_river_noise(self, x: int, z: int) -> float:
        """
        Get river noise value for given world coordinates.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            River noise value (typically in [-1, 1])
        """
        # Use river-specific noise for water features
        river_value = self.river_noise.noise2(
            x * self.river_scale, z * self.river_scale
        )

        # Add some directionality for river-like features
        directional = (
            self.river_noise.noise2(
                x * self.river_scale * 0.5, z * self.river_scale * 2.0
            )
            * 0.3
        )

        return float(river_value + directional)

    def get_patch(self, x_start: int, z_start: int, size: int) -> Dict[str, Any]:
        """
        Generate a patch of conditioning variables.

        Args:
            x_start: Starting X coordinate (world coordinates)
            z_start: Starting Z coordinate (world coordinates)
            size: Patch size (size x size)

        Returns:
            Dictionary containing patch arrays and metadata
        """
        # Validate size parameter
        if size <= 0:
            raise ValueError(f"Patch size must be positive, got {size}")

        # Initialize output arrays
        biomes = np.zeros((size, size), dtype=np.uint8)
        heightmap = np.zeros((size, size), dtype=np.uint16)
        river = np.zeros((size, size), dtype=np.float32)

        # Fill arrays by sampling each coordinate
        for i in range(size):
            for j in range(size):
                world_x = x_start + i
                world_z = z_start + j

                biomes[i, j] = self.get_biome(world_x, world_z)
                heightmap[i, j] = self.get_heightmap(world_x, world_z)
                river[i, j] = self.get_river_noise(world_x, world_z)

        # Create patch dictionary
        patch = {
            "biomes": biomes,
            "heightmap": heightmap,
            "river": river,
            "x": x_start,
            "z": z_start,
            "seed": self.seed,
        }

        logger.debug(f"Generated patch at ({x_start}, {z_start}) with size {size}")
        return patch

    def save_patch_npz(self, patch: Dict[str, Any], output_path: Path) -> Path:
        """
        Save patch data as compressed .npz file.

        Args:
            patch: Patch dictionary from get_patch()
            output_path: Path to save .npz file

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        np.savez_compressed(output_path, **patch)

        logger.debug(f"Saved patch to {output_path}")
        return output_path

    def get_patch_filename(self, x: int, z: int, output_dir: Path) -> Path:
        """
        Generate standardized filename for patch.

        Args:
            x: Patch X coordinate
            z: Patch Z coordinate
            output_dir: Output directory

        Returns:
            Path with standardized filename
        """
        filename = f"patch_x{x}_z{z}.npz"
        return output_dir / filename

    def generate_batch(
        self, coordinates: List[Tuple[int, int]], size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple patches in a batch.

        Args:
            coordinates: List of (x, z) coordinate tuples
            size: Patch size for all patches

        Returns:
            List of patch dictionaries
        """
        patches = []

        for x, z in coordinates:
            patch = self.get_patch(x, z, size)
            patches.append(patch)

        logger.info(f"Generated batch of {len(patches)} patches with size {size}")
        return patches

    def save_batch(self, patches: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
        """
        Save multiple patches to .npz files.

        Args:
            patches: List of patch dictionaries
            output_dir: Output directory

        Returns:
            List of paths to saved files
        """
        saved_paths = []

        for patch in patches:
            x, z = patch["x"], patch["z"]
            output_path = self.get_patch_filename(x, z, output_dir)
            saved_path = self.save_patch_npz(patch, output_path)
            saved_paths.append(saved_path)

        logger.info(f"Saved batch of {len(saved_paths)} patches to {output_dir}")
        return saved_paths
