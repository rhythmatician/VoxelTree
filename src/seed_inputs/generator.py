"""
SeedInputGenerator - Generate conditioning variables from seed and coordinates

This module implements seed-based generation of biomes, heightmaps, and river noise
without reading .mca files. Uses procedural noise to approximate vanilla Minecraft
terrain features for diffusion model conditioning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import opensimplex

logger = logging.getLogger(__name__)


@dataclass
class SeedInputGenerator:
    """
    Generate conditioning variables from seed and coordinates alone.
    
    Uses OpenSimplex noise to create vanilla-compatible biome, heightmap,
    and river noise patterns for ML model conditioning.
    """
    
    seed: int
    
    def __post_init__(self):
        """Initialize noise generators with seed."""
        # Create separate noise generators for different features
        self.biome_noise = opensimplex.OpenSimplex(seed=self.seed)
        self.height_noise = opensimplex.OpenSimplex(seed=self.seed + 1)
        self.river_noise = opensimplex.OpenSimplex(seed=self.seed + 2)
        
        # Terrain generation parameters
        self.biome_scale = 0.001  # Large scale for biome regions
        self.height_scale = 0.01  # Medium scale for height variation
        self.river_scale = 0.005  # Medium scale for river features
        
        # Height parameters
        self.base_height = 64  # Sea level equivalent
        self.height_variation = 60  # Max height variation
        
        logger.info(f"SeedInputGenerator initialized with seed={self.seed}")
    
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
        noise_value = self.biome_noise.noise2(x * self.biome_scale, z * self.biome_scale)
        
        # Map noise [-1, 1] to biome IDs [0, 255]
        # Group into major biome categories for realism
        normalized = (noise_value + 1) / 2  # Map to [0, 1]
        
        if normalized < 0.1:
            return 16  # Beach
        elif normalized < 0.2:
            return 7   # River
        elif normalized < 0.4:
            return 1   # Plains
        elif normalized < 0.6:
            return 4   # Forest
        elif normalized < 0.75:
            return 18  # Forest Hills
        elif normalized < 0.9:
            return 3   # Extreme Hills
        else:
            return 12  # Ice Plains
    
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
        noise_value = self.height_noise.noise2(x * self.height_scale, z * self.height_scale)
        
        # Add some octaves for more realistic terrain
        octave1 = self.height_noise.noise2(x * self.height_scale * 2, z * self.height_scale * 2) * 0.5
        octave2 = self.height_noise.noise2(x * self.height_scale * 4, z * self.height_scale * 4) * 0.25
        
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
        river_value = self.river_noise.noise2(x * self.river_scale, z * self.river_scale)
        
        # Add some directionality for river-like features
        directional = self.river_noise.noise2(
            x * self.river_scale * 0.5, 
            z * self.river_scale * 2.0
        ) * 0.3
        
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
            "seed": self.seed
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
    
    def generate_batch(self, coordinates: List[Tuple[int, int]], size: int) -> List[Dict[str, Any]]:
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
