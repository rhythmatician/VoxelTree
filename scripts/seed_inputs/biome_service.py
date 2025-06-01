"""
BiomeLookupService - Deterministic vanilla-accurate biome generation

This module provides a service for looking up biome IDs using vanilla Minecraft
algorithms via external Java tools or cached results.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BiomeLookupService:
    """
    Service for deterministic biome lookup using vanilla Minecraft algorithms.

    Supports batch generation, caching, and fallback strategies for robust
    biome generation without .mca file dependencies.
    """

    def __init__(self, seed: int, cache_dir: Path, java_tool: str, fallback_tool: str):
        """
        Initialize the biome lookup service.

        Args:
            seed: World seed for biome generation
            cache_dir: Directory for caching biome results
            java_tool: Path to primary Java biome tool
            fallback_tool: Path to fallback Java biome tool
        """
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.java_tool = Path(java_tool)
        self.fallback_tool = Path(fallback_tool)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check tool availability
        self.tools_available = self._check_tools()

        # Cache for biome data (region-based caching)
        self._biome_cache: Dict[Tuple[int, int], np.ndarray] = {}

        logger.info(
            f"BiomeLookupService initialized for seed {seed}, tools_available={self.tools_available}"
        )

    def _check_tools(self) -> bool:
        """Check if Java biome tools are available."""
        return self.java_tool.exists() or self.fallback_tool.exists()

    def get_biome(self, x: int, z: int) -> int:
        """
        Get biome ID for a single coordinate.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Biome ID (0-255, compatible with Minecraft)
        """
        if not self.tools_available:
            logger.warning("No biome tools available, returning fallback biome")
            return 1  # Plains as fallback

        # Calculate region coordinates (512x512 block regions for caching)
        region_x = x // 512
        region_z = z // 512
        local_x = x % 512
        local_z = z % 512

        # Get or generate region biome data
        region_biomes = self._get_region_biomes(region_x, region_z)

        # Return biome for specific coordinate
        return int(region_biomes[local_x, local_z])

    def get_biome_patch(self, x_start: int, z_start: int, size: int) -> np.ndarray:
        """
        Get biome IDs for a patch of coordinates.

        Args:
            x_start: Starting X coordinate
            z_start: Starting Z coordinate
            size: Patch size (size x size)

        Returns:
            2D array of biome IDs with shape (size, size)
        """
        biomes = np.zeros((size, size), dtype=np.uint8)

        for i in range(size):
            for j in range(size):
                biomes[i, j] = self.get_biome(x_start + i, z_start + j)

        return biomes

    def _get_region_biomes(self, region_x: int, region_z: int) -> np.ndarray:
        """
        Get or generate biome data for a 512x512 region.

        Args:
            region_x: Region X coordinate
            region_z: Region Z coordinate

        Returns:
            2D array of biome IDs with shape (512, 512)
        """
        region_key = (region_x, region_z)

        # Check memory cache first
        if region_key in self._biome_cache:
            return self._biome_cache[region_key]

        # Check disk cache
        cache_file = self._get_cache_file(region_x, region_z)
        if cache_file.exists():
            try:
                region_biomes = np.load(cache_file)
                self._biome_cache[region_key] = region_biomes
                return region_biomes
            except Exception as e:
                logger.warning(f"Failed to load cached biomes from {cache_file}: {e}")

        # Generate biomes using Java tool
        region_biomes = self._generate_region_biomes(region_x, region_z)

        # Cache the result
        self._cache_region_biomes(region_x, region_z, region_biomes)
        self._biome_cache[region_key] = region_biomes

        return region_biomes

    def _get_cache_file(self, region_x: int, region_z: int) -> Path:
        """Get cache file path for a region."""
        # Include seed in filename to avoid conflicts
        filename = f"biomes_seed{self.seed}_r{region_x}_{region_z}.npy"
        return self.cache_dir / filename

    def _generate_region_biomes(self, region_x: int, region_z: int) -> np.ndarray:
        """
        Generate biome data for a region using Java tools.

        Args:
            region_x: Region X coordinate
            region_z: Region Z coordinate

        Returns:
            2D array of biome IDs with shape (512, 512)
        """
        # Calculate world coordinates for this region
        world_x_start = region_x * 512
        world_z_start = region_z * 512

        # Try each available tool
        for tool_path in [self.java_tool, self.fallback_tool]:
            if not tool_path.exists():
                continue

            try:
                result = self._call_java_tool(tool_path, world_x_start, world_z_start, 512, 512)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Failed to generate biomes with {tool_path}: {e}")
                continue

        # If all tools fail, generate fallback pattern
        logger.warning(
            f"All biome tools failed for region ({region_x}, {region_z}), using fallback"
        )
        return self._generate_fallback_biomes(world_x_start, world_z_start, 512, 512)

    def _call_java_tool(
        self, tool_path: Path, x: int, z: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        """
        Call Java biome tool to generate biome data.

        Args:
            tool_path: Path to Java tool
            x: Starting X coordinate
            z: Starting Z coordinate
            width: Width of region
            height: Height of region

        Returns:
            2D array of biome IDs or None if failed
        """
        # Create temporary output file
        temp_output = self.cache_dir / f"temp_biomes_{x}_{z}.json"

        try:
            # Example command structure (adjust based on actual tool)
            cmd = [
                "java",
                "-jar",
                str(tool_path),
                "--seed",
                str(self.seed),
                "--x",
                str(x),
                "--z",
                str(z),
                "--width",
                str(width),
                "--height",
                str(height),
                "--output",
                str(temp_output),
                "--format",
                "json",
            ]

            # Run the command with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and temp_output.exists():
                # Parse the output
                with open(temp_output, "r") as f:
                    data = json.load(f)

                # Convert to numpy array
                biomes = np.array(data["biomes"], dtype=np.uint8)

                # Clean up temp file
                temp_output.unlink()

                return biomes
            else:
                logger.warning(f"Java tool failed: {result.stderr}")
                return None

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error calling Java tool {tool_path}: {e}")
            return None
        finally:
            # Clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()

    def _generate_fallback_biomes(self, x: int, z: int, width: int, height: int) -> np.ndarray:
        """
        Generate fallback biome pattern when Java tools are unavailable.

        Args:
            x: Starting X coordinate
            z: Starting Z coordinate
            width: Width of region
            height: Height of region

        Returns:
            2D array of biome IDs with simple pattern
        """
        biomes = np.ones((width, height), dtype=np.uint8)  # Default to Plains

        # Add some simple variation based on coordinates
        for i in range(width):
            for j in range(height):
                world_x = x + i
                world_z = z + j

                # Simple hash-based biome assignment
                coord_hash = hash((world_x, world_z, self.seed)) % 7

                if coord_hash == 0:
                    biomes[i, j] = 4  # Forest
                elif coord_hash == 1:
                    biomes[i, j] = 16  # Beach
                elif coord_hash == 2:
                    biomes[i, j] = 3  # Extreme Hills
                else:
                    biomes[i, j] = 1  # Plains

        return biomes

    def _cache_region_biomes(self, region_x: int, region_z: int, biomes: np.ndarray):
        """Cache biome data to disk."""
        cache_file = self._get_cache_file(region_x, region_z)
        try:
            np.save(cache_file, biomes)
            logger.debug(f"Cached biomes for region ({region_x}, {region_z}) to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache biomes: {e}")

    def clear_cache(self):
        """Clear all cached biome data."""
        self._biome_cache.clear()

        # Remove cache files
        for cache_file in self.cache_dir.glob(f"biomes_seed{self.seed}_*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info("Cleared biome cache")
