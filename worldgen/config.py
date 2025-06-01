"""
Configuration loading for WorldGen bootstrap.

Handles loading worldgen parameters from config.yaml files.
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_worldgen_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load worldgen configuration from config.yaml.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary with worldgen configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        config_path = Path("config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('worldgen', {})
