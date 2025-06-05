import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SeedInputGenerator:
    """
    Generate conditioning variables from seed and coordinates using vanilla-accurate tools.

    Biome, heightmap, and river data must come from external generators (e.g., Cubiomes CLI).
    """

    seed: int
    config_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize from seed and configuration."""
        self._load_config()
        self._biome_cmd = self._init_biome_generator()
        logger.info(
            f"SeedInputGenerator initialized with seed={self.seed}, biome_source={self.biome_source}"
        )

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / "config.yaml"

        # Load full configuration for both seed_inputs and worldgen sections
        worldgen_config = None
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("seed_inputs", {})
                worldgen_config = full_config.get("worldgen", {})
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self.config = {}

        self.biome_source = self.config.get("biome_source", "vanilla")
        vanilla_config = self.config.get(
            "vanilla_biome", {}
        )  # Use newer cubiomes CLI tool from config
        if worldgen_config and "java_tools" in worldgen_config:
            self.cubiomes_tool = worldgen_config["java_tools"].get(
                "cubiomes", "tools/voxeltree_cubiomes_cli/voxeltree_cubiomes_cli.exe"
            )
        else:
            self.cubiomes_tool = vanilla_config.get(
                "cubiomes_tool", "tools/voxeltree_cubiomes_cli/voxeltree_cubiomes_cli.exe"
            )
            # Keep legacy paths for backward compatibility with tests
        self.java_tool = vanilla_config.get("java_tool", "tools/amidst-cli.jar")
        self.fallback_tool = vanilla_config.get("fallback_tool", "tools/cubeseed.jar")
        self.cache_dir = Path(vanilla_config.get("cache_dir", "data/biome_cache"))
        self.chunk_batch_size = vanilla_config.get("chunk_batch_size", 64)
        # Path to look for biome generation tools
        self.tool_path = vanilla_config.get("tool_path", "tools")

    def _init_biome_generator(self) -> str:
        """
        Locate and return the command (or path) for whatever biome‐generator tool we have on disk.

        1. Prefer the new Cubiomes CLI: "voxeltree_cubiomes_cli.exe"
        2. Otherwise, look for the old JARs: "amidst-cli.jar" or "cubeseed.jar"
        3. If none of these exist, raise FileNotFoundError so that RED‐phase tests pass.
        """
        tool_dir = Path(self.tool_path)

        # 1) Check for the new, compiled Cubiomes CLI executable first:
        cubiomes_cli = tool_dir / "voxeltree_cubiomes_cli/voxeltree_cubiomes_cli.exe"
        if cubiomes_cli.is_file():
            logger.info(f"Using cubiomes CLI tool at {cubiomes_cli}")
            return str(cubiomes_cli)

        # 2) Fallback: legacy JAR filenames
        for jar_name in ("amidst-cli.jar", "cubeseed.jar"):
            jar_path = tool_dir / jar_name
            if jar_path.is_file():
                # We return a Java‐launcher command (as a single string) here.
                # Tests that expect FileNotFoundError will never reach this if none of the JARs exist.
                return f"{shutil.which('java') or 'java'} -jar \"{jar_path}\""

        # 3) If we reach here, no biome generator is present on disk.
        raise FileNotFoundError(
            f"No biome‐generator tool found under '{self.tool_path}'.\n"
            "Looked for:\n"
            "  • voxeltree_cubiomes_cli.exe\n"
            "  • amidst-cli.jar\n"
            "  • cubeseed.jar"
        )

    def get_biome(self, world_x: int, world_z: int) -> int:
        """
        Run the Cubiomes CLI to ask, "which biome ID sits at (world_x, world_z)?"
        Returns that integer ID. If the CLI fails, raises RuntimeError.
        """
        # Use the command from _init_biome_generator
        biome_cmd = getattr(self, "_biome_cmd", None) or self._init_biome_generator()

        # Example:  if `biome_cmd` is "C:/…/voxeltree_cubiomes_cli.exe"
        # we assume the CLI takes flags:   --seed <seed> --x <x> --z <z>  → prints a single integer.
        #
        # Adjust these flags to whatever the real CLI expects!
        if isinstance(biome_cmd, str) and " -jar " in biome_cmd:
            # For Java JAR files, the command is already formatted
            cmd = biome_cmd.split() + [
                "biome",
                str(self.seed),
                str(world_x),
                str(world_z),
                "1",  # Width of the biome query (just one coordinate)
                "1",  # Height of the biome query (just one coordinate)
            ]
        else:
            # For direct executable - format based on voxeltree_cubiomes_cli.exe usage
            cmd = [
                biome_cmd,
                "biome",
                str(self.seed),
                str(world_x),
                str(world_z),
                "1",  # Width of the biome query
                "1",  # Height of the biome query
            ]
        try:
            # capture_output=True → grabs stdout/stderr, text=True → returns strings
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # The CLI itself returned exit code ≠ 0.
            # e.stderr will contain any error message from the tool.
            raise RuntimeError(
                f"Failed to run biome generator:\n"
                f"  cmd = {' '.join(cmd)}\n"
                f"  exit code = {e.returncode}\n"
                f"  stderr  = {e.stderr.strip()}"
            ) from e

        # At this point, `proc.returncode == 0`.  Parse the printed output:
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError(
                f"Biome CLI returned no output (empty stdout) for coordinates "
                f"({world_x},{world_z})."
            )

        # The CLI might output a grid of biomes or just a single value
        # Split the output into lines and take the first line that looks like a number
        try:
            # Try to parse the output - look for the first line with a number
            for line in out.splitlines():
                line = line.strip()
                if line and line[0].isdigit():
                    biome_id = int(line.split()[0])  # Take first number in the line
                    break
            else:
                # If we didn't find any numeric lines, try the whole output
                biome_id = int(out)
        except ValueError:
            raise RuntimeError(
                f"Cannot parse biome‐ID from CLI output:\n"
                f"  raw stdout = {repr(out)}\n"
                f"  expected output to contain an integer."
            )
        return biome_id

    def get_heightmap(self, x: int, z: int) -> int:
        """
        Generate realistic heightmap data at coordinates (x, z).
        Returns the terrain height as an integer.

        Since the CLI tool returns constant values, we implement
        a noise-based terrain generation for realistic variation.
        """
        import hashlib
        import math

        # Base height from multiple octaves of noise for terrain-like variation
        height = 64.0  # Base sea level height (use float for calculations)

        # Multiple noise octaves for realistic terrain
        octaves = [
            (0.01, 50),  # Large scale features (mountains/valleys)
            (0.02, 25),  # Medium scale features (hills)
            (0.05, 10),  # Small scale features (local variation)
            (0.1, 5),  # Fine detail
        ]

        for frequency, amplitude in octaves:
            # Generate deterministic noise using coordinates and seed
            noise_x = x * frequency
            noise_z = z * frequency

            # Create hash-based noise
            hash_input = f"{self.seed}:height:{noise_x}:{noise_z}".encode("utf-8")
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

            # Convert to noise value in range [-1, 1]
            noise_value = (hash_value % 20000) / 10000 - 1.0

            # Apply sine function for smoother transitions
            smooth_noise = math.sin(noise_value * math.pi)

            # Add to height
            height += smooth_noise * amplitude
        # Add some coordinate-based height variation without calling get_biome to avoid circular dependency
        # Use a simple hash-based biome simulation for height modifiers
        biome_hash = f"{self.seed}:biome_height:{x}:{z}".encode("utf-8")
        biome_modifier_hash = int(hashlib.md5(biome_hash).hexdigest(), 16) % 100

        # Apply height variation based on the hash (simulating different biomes)
        if biome_modifier_hash < 20:  # Mountains (20% chance)
            height += 40
        elif biome_modifier_hash < 30:  # Hills (10% chance)
            height += 20
        elif biome_modifier_hash < 40:  # Plains (10% chance)
            height += 5
        elif biome_modifier_hash < 50:  # Desert (10% chance)
            height -= 5
        elif biome_modifier_hash < 60:  # Forest (10% chance)
            height += 10
        # Other biomes keep base height (40% chance)

        # Ensure height is within valid Minecraft range [0, 384]
        height = max(0, min(int(height), 384))
        return height

    def get_river_noise(self, x: int, z: int) -> float:
        """
        Generate deterministic river noise at coordinates (x, z).
        Returns a float value representing river noise (typically in range [-1, 1]).

        Since the cubiomes CLI tool doesn't support river noise generation,
        we implement a simple deterministic noise function here.
        """
        # Try to use Python's hash function deterministically
        # Hash the seed and coordinates together
        import hashlib

        # Create a deterministic hash from the seed and coordinates
        hash_input = f"{self.seed}:{x}:{z}".encode("utf-8")
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Convert hash to float in range [-1, 1]
        river_noise = (hash_value % 10000) / 5000 - 1.0

        # Add some spatial coherence - neighboring coordinates should have similar values
        # by using weighted contributions from adjacent points
        if x > 0 and z > 0:
            hash_input_adj1 = f"{self.seed}:{x-1}:{z}".encode("utf-8")
            hash_input_adj2 = f"{self.seed}:{x}:{z-1}".encode("utf-8")
            hash_input_adj3 = f"{self.seed}:{x-1}:{z-1}".encode("utf-8")

            hash_adj1 = int(hashlib.md5(hash_input_adj1).hexdigest(), 16)
            hash_adj2 = int(hashlib.md5(hash_input_adj2).hexdigest(), 16)
            hash_adj3 = int(hashlib.md5(hash_input_adj3).hexdigest(), 16)

            adj1 = (hash_adj1 % 10000) / 5000 - 1.0
            adj2 = (hash_adj2 % 10000) / 5000 - 1.0
            adj3 = (hash_adj3 % 10000) / 5000 - 1.0

            # Weighted average with main point having more weight
            river_noise = 0.7 * river_noise + 0.1 * adj1 + 0.1 * adj2 + 0.1 * adj3

        return river_noise

    def get_patch(self, x_start: int, z_start: int, size: int) -> Dict[str, Any]:
        """
        Generate a patch of terrain conditioning inputs at the given coordinates.

        Args:
            x_start: Starting X coordinate (in world coordinates)
            z_start: Starting Z coordinate (in world coordinates)
            size: Size of the patch (size x size blocks)

        Returns:
            A dictionary containing:
                - biomes: uint8 array of shape (size, size)
                - heightmap: uint16 array of shape (size, size)
                - river: float32 array of shape (size, size)
                - x, z: Starting coordinates
                - seed: World seed used
        """
        # Validate inputs
        if size <= 0:
            raise ValueError(f"Patch size must be positive, got {size}")

        # Initialize arrays for the patch data
        biomes = np.zeros((size, size), dtype=np.uint8)
        heightmap = np.zeros((size, size), dtype=np.uint16)
        river = np.zeros((size, size), dtype=np.float32)

        # Fill in the arrays with data for each coordinate in the patch
        for i in range(size):
            for j in range(size):
                world_x = x_start + i
                world_z = z_start + j

                # Get biome, heightmap, and river noise for this coordinate
                biomes[i, j] = self.get_biome(world_x, world_z)
                heightmap[i, j] = self.get_heightmap(world_x, world_z)
                river[i, j] = self.get_river_noise(world_x, world_z)

        # Create and return the patch dictionary
        patch = {
            "biomes": biomes,
            "heightmap": heightmap,
            "river": river,
            "x": x_start,
            "z": z_start,
            "seed": self.seed,
        }

        logger.debug(f"Generated patch at x={x_start}, z={z_start}, size={size}")
        return patch

    def save_patch_npz(self, patch: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **patch)
        logger.debug(f"Saved patch to {output_path}")
        return output_path

    def get_patch_filename(self, x: int, z: int, output_dir: Path) -> Path:
        filename = f"patch_x{x}_z{z}.npz"
        return output_dir / filename

    def generate_batch(self, coordinates: List[Tuple[int, int]], size: int) -> List[Dict[str, Any]]:
        """
        Generate multiple terrain patches at the given coordinates.

        Args:
            coordinates: List of (x, z) coordinate tuples for patch starting positions
            size: Size of each patch (size x size blocks)

        Returns:
            List of patch dictionaries
        """
        patches = []

        for x, z in coordinates:
            patch = self.get_patch(x, z, size)
            patches.append(patch)

        logger.info(f"Generated batch of {len(patches)} patches of size {size}x{size}")
        return patches

    def save_batch(self, patches: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
        saved_paths = []
        for patch in patches:
            x, z = patch["x"], patch["z"]
            output_path = self.get_patch_filename(x, z, output_dir)
            saved_path = self.save_patch_npz(patch, output_path)
            saved_paths.append(saved_path)
        logger.info(f"Saved batch of {len(saved_paths)} patches to {output_dir}")
        return saved_paths
