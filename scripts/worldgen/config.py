"""
Configuration loading for WorldGen bootstrap.

Handles loading worldgen parameters from config.yaml files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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


def load_worldgen_config(config_path: Optional[Path] = Path("config.yaml")) -> Dict[str, Any]:
    """
    Load worldgen configuration from config.yaml.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary with worldgen configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("worldgen", {})
