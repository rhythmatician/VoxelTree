from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from scripts.worldgen.config import load_config, load_worldgen_config


def test_load_config_file_not_found():
    """Test that load_config raises FileNotFoundError if config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent_config.yaml"))


def test_load_config_success():
    """Test that load_config successfully loads a valid config file."""
    mock_yaml_content = """
    worldgen:
      seed: "VoxelTree"
      java_heap: "4G"
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            config = load_config(Path("config.yaml"))
            assert config["worldgen"]["seed"] == "VoxelTree"
            assert config["worldgen"]["java_heap"] == "4G"


def test_load_worldgen_config_file_not_found():
    """Test that load_worldgen_config raises FileNotFoundError if config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_worldgen_config(Path("nonexistent_config.yaml"))


def test_load_worldgen_config_success():
    """Test that load_worldgen_config successfully loads the worldgen section."""
    mock_yaml_content = """
    worldgen:
      seed: "VoxelTree"
      java_heap: "4G"
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            worldgen_config = load_worldgen_config(Path("config.yaml"))
            assert worldgen_config["seed"] == "VoxelTree"
            assert worldgen_config["java_heap"] == "4G"


def test_load_worldgen_config_missing_section():
    """Test that load_worldgen_config returns an empty dictionary if worldgen section is missing."""
    mock_yaml_content = """
    extraction:
      output_dir: "data/chunks"
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            worldgen_config = load_worldgen_config(Path("config.yaml"))
            assert worldgen_config == {}


def test_load_worldgen_config_default_path():
    """Test that load_worldgen_config uses default config.yaml path when none provided."""
    mock_yaml_content = """
    worldgen:
      seed: "VoxelTree"
      java_heap: "4G"
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            # Call without providing config_path to test default behavior
            worldgen_config = load_worldgen_config()
            assert worldgen_config["seed"] == "VoxelTree"
            assert worldgen_config["java_heap"] == "4G"
