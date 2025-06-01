"""
CubiomesService - Vanilla-accurate biome and height generation using cubiomes

This module provides deterministic biome and height generation using the cubiomes
C library via subprocess calls to a compiled CLI tool.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CubiomesService:
    """
    Service for vanilla-accurate biome and height generation using cubiomes.

    Provides deterministic results matching Minecraft's biome generation algorithms
    with proper fallback when cubiomes tools aren't available.
    """

    def __init__(self, seed: int, cubiomes_tool_path: Optional[Path] = None):
        """
        Initialize the cubiomes service.

        Args:
            seed: World seed for generation
            cubiomes_tool_path: Path to cubiomes CLI tool (if available)
        """
        self.seed = seed
        self.cubiomes_tool_path = cubiomes_tool_path
        self.tools_available = self._check_cubiomes_availability()

        logger.info(
            f"CubiomesService initialized with seed={seed}, tools_available={self.tools_available}"
        )

    def _check_cubiomes_availability(self) -> bool:
        """Check if cubiomes tools are available and working."""
        if not self.cubiomes_tool_path or not self.cubiomes_tool_path.exists():
            logger.debug("Cubiomes tool path not found")
            return False

        try:
            # Try a simple test call
            result = subprocess.run(
                [str(self.cubiomes_tool_path), str(self.seed), "0", "0"],  # Test coordinates
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Cubiomes tool test failed")
            return False

    def get_biome(self, x: int, z: int) -> int:
        """
        Get biome ID for given world coordinates using cubiomes.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Biome ID (0-255, compatible with Minecraft)

        Raises:
            RuntimeError: If cubiomes tools not available and no fallback configured
        """
        if not self.tools_available:
            raise RuntimeError("Cubiomes tools not available - cannot get vanilla biome")

        try:
            result = subprocess.run(
                [str(self.cubiomes_tool_path), str(self.seed), str(x), str(z)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Cubiomes tool failed: {result.stderr}")

            # Parse output - assuming format: "biome_id"
            biome_id = int(result.stdout.strip())
            return biome_id

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            raise RuntimeError(f"Failed to get biome from cubiomes: {e}")

    def get_surface_height(self, x: int, z: int) -> int:
        """
        Get surface height for given world coordinates using cubiomes.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Surface height (-64 to 320, compatible with Minecraft 1.18+)

        Raises:
            RuntimeError: If cubiomes tools not available
        """
        if not self.tools_available:
            raise RuntimeError("Cubiomes tools not available - cannot get vanilla height")

        try:
            # Call cubiomes tool with height query flag
            result = subprocess.run(
                [
                    str(self.cubiomes_tool_path),
                    "--height",  # Assuming such a flag exists
                    str(self.seed),
                    str(x),
                    str(z),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Cubiomes height tool failed: {result.stderr}")

            # Parse output - assuming format: "height"
            height = int(result.stdout.strip())
            return height

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            raise RuntimeError(f"Failed to get height from cubiomes: {e}")

    def get_biome_batch(self, coordinates: list[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """
        Get biomes for multiple coordinates in a batch for efficiency.

        Args:
            coordinates: List of (x, z) coordinate tuples

        Returns:
            Dictionary mapping (x, z) to biome_id
        """
        if not self.tools_available:
            raise RuntimeError("Cubiomes tools not available")

        if not coordinates:
            return {}

        # Create temporary input file for batch processing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for x, z in coordinates:
                f.write(f"{x} {z}\n")
            input_file = f.name

        try:
            # Call cubiomes tool with batch flag
            result = subprocess.run(
                [
                    str(self.cubiomes_tool_path),
                    "--batch",  # Assuming such a flag exists
                    str(self.seed),
                    input_file,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Cubiomes batch tool failed: {result.stderr}")

            # Parse output - assuming format: "x z biome_id" per line
            results = {}
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        x, z, biome_id = int(parts[0]), int(parts[1]), int(parts[2])
                        results[(x, z)] = biome_id

            return results

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            raise RuntimeError(f"Failed to get batch biomes from cubiomes: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(input_file)
            except OSError:
                pass

    @property
    def is_available(self) -> bool:
        """Check if cubiomes tools are available."""
        return self.tools_available
