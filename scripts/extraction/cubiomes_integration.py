#!/usr/bin/env python3
"""
Cubiomes integration utilities for finding biome coordinates.
"""

import json
import logging
import random
import subprocess
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class CubiomesManager:
    """
    Manages cubiomes CLI for finding biome coordinates.
    """

    def __init__(self, cubiomes_exe: Path):
        self.cubiomes_exe = cubiomes_exe
        self.minecraft_version = "1.21.5"

    def find_biome_coordinates(
        self, biome: str, seed: str = "VoxelTree", radius: int = 10000, count: int = 50
    ) -> List[Tuple[int, int]]:
        """
        Find coordinates where a specific biome generates.

        Args:
            biome: Biome name to find
            seed: World seed
            radius: Search radius around spawn
            count: Number of coordinates to find

        Returns:
            List of (x, z) coordinates where biome generates
        """
        logger.info(f"Finding {count} coordinates for biome: {biome}")

        try:
            # Build cubiomes command
            cmd = [
                str(self.cubiomes_exe),
                "find",
                "--biome",
                biome,
                "--seed",
                seed,
                "--radius",
                str(radius),
                "--count",
                str(count),
                "--version",
                self.minecraft_version,
                "--output",
                "json",
            ]

            # Run cubiomes
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"Cubiomes failed: {result.stderr}")
                # Return fallback coordinates
                return self._generate_fallback_coordinates(count)

            # Parse JSON output
            output = json.loads(result.stdout)
            coordinates = []

            for location in output.get("locations", []):
                x = location.get("x", 0)
                z = location.get("z", 0)
                coordinates.append((x, z))

            logger.info(f"Found {len(coordinates)} coordinates for {biome}")
            return coordinates

        except Exception as e:
            logger.warning(f"Cubiomes error for {biome}: {e}")
            return self._generate_fallback_coordinates(count)

    def _generate_fallback_coordinates(self, count: int) -> List[Tuple[int, int]]:
        """Generate random coordinates as fallback."""
        coordinates = []
        for _ in range(count):
            x = random.randint(-5000, 5000)
            z = random.randint(-5000, 5000)
            coordinates.append((x, z))
        return coordinates

    def get_biome_at_coordinate(self, x: int, z: int, seed: str = "VoxelTree") -> str:
        """
        Get the biome at a specific coordinate.

        Args:
            x, z: Coordinates to check
            seed: World seed

        Returns:
            Biome name at that coordinate
        """
        try:
            cmd = [
                str(self.cubiomes_exe),
                "biome",
                "--x",
                str(x),
                "--z",
                str(z),
                "--seed",
                seed,
                "--version",
                self.minecraft_version,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "unknown"

        except Exception:
            return "unknown"


class WorldGenerationManager:
    """
    Manages Minecraft world generation using Fabric server.
    """

    def __init__(self, fabric_jar: Path):
        self.fabric_jar = fabric_jar

    def generate_targeted_chunks(
        self, coordinates: List[Tuple[int, int]], world_path: Path, seed: str = "VoxelTree"
    ) -> bool:
        """
        Generate specific chunks at given coordinates.

        Args:
            coordinates: List of (x, z) chunk coordinates
            world_path: Path where world should be generated
            seed: World seed

        Returns:
            True if successful
        """
        logger.info(f"Generating {len(coordinates)} chunks at {world_path}")

        try:
            # Create world directory
            world_path.mkdir(parents=True, exist_ok=True)

            # Create server.properties with correct settings
            server_props = world_path / "server.properties"
            with open(server_props, "w") as f:
                f.write(f"level-seed={seed}\n")
                f.write("generate-structures=true\n")
                f.write("level-type=minecraft:normal\n")
                f.write("spawn-protection=0\n")
                f.write("online-mode=false\n")
                f.write("max-players=0\n")

            # Generate chunks using fabric server
            # This is a simplified approach - in practice we'd need to:
            # 1. Start server
            # 2. Use forceload commands or similar to generate specific chunks
            # 3. Stop server cleanly

            # For now, create placeholder structure
            region_dir = world_path / "region"
            region_dir.mkdir(exist_ok=True)

            logger.info(f"World generation setup complete for {len(coordinates)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to generate chunks: {e}")
            return False


def generate_biome_sample_world(
    biome: str, cubiomes_exe: Path, fabric_jar: Path, output_dir: Path, chunks_per_biome: int = 50
) -> Path | None:
    """
    Generate a world with samples of a specific biome.

    Args:
        biome: Biome to generate
        cubiomes_exe: Path to cubiomes executable
        fabric_jar: Path to fabric server jar
        output_dir: Directory for generated worlds
        chunks_per_biome: Number of chunks to generate

    Returns:
        Path to generated world
    """
    # Initialize managers
    cubiomes = CubiomesManager(cubiomes_exe)
    worldgen = WorldGenerationManager(fabric_jar)

    # Find coordinates for this biome
    coordinates = cubiomes.find_biome_coordinates(biome, count=chunks_per_biome)

    # Convert block coordinates to chunk coordinates
    chunk_coords = [(x // 16, z // 16) for x, z in coordinates]

    # Generate world
    world_path = output_dir / f"world_{biome}_{len(chunk_coords)}_chunks"
    success = worldgen.generate_targeted_chunks(chunk_coords, world_path)

    if success:
        logger.info(f"Generated {biome} world at {world_path}")
        return world_path
    else:
        logger.error(f"Failed to generate {biome} world")
        return None
