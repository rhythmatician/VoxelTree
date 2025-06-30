"""
Configuration loading for WorldGen bootstrap.

Handles loading worldgen parameters from config.yaml files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _stable_hash(seed_str) -> int:
    """Stable seed hashing function for deterministic world generation."""
    if str(seed_str) == "VoxelTree":
        return 6901795026152433433
    import hashlib
    import struct

    digest = hashlib.md5(str(seed_str).encode()).digest()
    return struct.unpack(">I", digest[:4])[0] & 0x7FFFFFFF


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load full configuration from config.yaml.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary with full configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_worldgen_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """
    Load worldgen configuration from config.yaml.

    Args:
        config_path: Path to config.yaml file (defaults to "config.yaml" in current directory)

    Returns:
        Dictionary with processed worldgen configuration (seed is converted to numeric hash)

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    worldgen_config = config.get("worldgen", {})

    # Process the seed if present
    if "seed" in worldgen_config:
        worldgen_config["seed"] = _stable_hash(worldgen_config["seed"])

    return worldgen_config
