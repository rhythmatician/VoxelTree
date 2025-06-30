"""
Tests for corpus generation workflow.
This ensures the dataset generation pipeline works end-to-end.
"""

import os
import shutil

# Append project root to path to ensure modules can be imported
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.generate_corpus import check_disk_space, load_config, parse_seed_range, split_dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_config():
    """Create a minimal configuration for testing."""
    config = {
        "worldgen": {
            "seed": "TestSeed",
            "chunk_region_bounds": {"x_min": 0, "x_max": 1, "z_min": 0, "z_max": 1},
            "max_temp_disk_gb": 1,
        },
        "extraction": {
            "output_dir": "data/test_chunks",
            "temp_dir": "data/test_temp",
            "max_disk_usage_gb": 1,
        },
        "data": {
            "processed_data_dir": "data/test_processed",
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
        },
        "pairing": {
            "extracted_data_dir": "data/test_chunks",
            "output_dir": "data/test_pairs",
            "lod_levels": 2,
        },
    }
    return config


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with dummy dataset files."""
    temp_dir = tempfile.mkdtemp()

    # Create some dummy npz files
    for i in range(10):
        # Create minimal valid training example
        example = {
            "parent_voxel": np.zeros((8, 8, 8), dtype=np.float32),
            "biome_patch": np.zeros((16, 16), dtype=np.int32),
            "heightmap_patch": np.zeros((1, 16, 16), dtype=np.float32),
            "river_patch": np.zeros((1, 16, 16), dtype=np.float32),
            "y_index": np.array([0], dtype=np.int32),
            "lod": np.array([1], dtype=np.int32),
            "target_mask": np.zeros((1, 16, 16, 16), dtype=np.float32),
            "target_types": np.zeros((16, 16, 16), dtype=np.int32),
        }
        np.savez_compressed(os.path.join(temp_dir, f"example_{i:02d}.npz"), **example)

    yield Path(temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir)


def test_load_config():
    """Test configuration loading."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "worldgen": {"seed": "TestSeed"},
                "extraction": {"output_dir": "test"},
                "training": {"batch_size": 32},
                "data": {"processed_data_dir": "data/test"},
                "pairing": {"extracted_data_dir": "test", "output_dir": "test_pairs"},
            },
            f,
        )
        config_path = Path(f.name)

    try:
        # Test loading
        config = load_config(config_path)
        assert config["worldgen"]["seed"] == "TestSeed"
        assert config["extraction"]["output_dir"] == "test"
        assert config["training"]["batch_size"] == 32
    finally:
        # Cleanup
        os.unlink(config_path)


def test_parse_seed_range():
    """Test seed range parsing."""
    # Test range format
    seeds = parse_seed_range("1000-1005")
    assert seeds == [1000, 1001, 1002, 1003, 1004, 1005]

    # Test comma-separated format
    seeds = parse_seed_range("1000,1002,1004,1006")
    assert seeds == [1000, 1002, 1004, 1006]


def test_disk_space_monitoring():
    """Test disk space monitoring functions."""
    # Check disk space - this should run without error
    available_gb, sufficient = check_disk_space(min_required_gb=1000000)  # Unrealistically high
    assert isinstance(available_gb, float)
    assert isinstance(sufficient, bool)
    assert not sufficient  # No one has a petabyte of free space


def test_split_dataset(temp_dataset_dir, temp_config):
    """Test dataset splitting functionality."""
    output_dir = Path(tempfile.mkdtemp())

    try:
        # Split the dataset
        train_dir, val_dir, test_dir = split_dataset(
            temp_dataset_dir, output_dir, train_ratio=0.6, val_ratio=0.3
        )

        # Check that the splits were created
        assert train_dir.exists()
        assert val_dir.exists()
        assert test_dir.exists()

        # Check file distribution (10 files total)
        # Train: 60% = 6 files
        # Val: 30% = 3 files
        # Test: 10% = 1 file
        train_files = list(train_dir.glob("*.npz"))
        val_files = list(val_dir.glob("*.npz"))
        test_files = list(test_dir.glob("*.npz"))

        assert len(train_files) == 6
        assert len(val_files) == 3
        assert len(test_files) == 1
    finally:
        # Cleanup
        shutil.rmtree(output_dir)


def test_manual_split_calculation():
    """Test that split calculations are correct and deterministic."""
    # Create a simulated list of 100 patch files
    total_files = 100

    # Calculate splits with standard ratios
    train_ratio = 0.7
    val_ratio = 0.2
    # test_ratio = 0.1

    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    # Verify calculations
    assert train_count == 70
    assert val_count == 20
    assert test_count == 10
    assert train_count + val_count + test_count == total_files


# Integration test for corpus generation is impractical in unit tests
# as it requires Java tools and significant disk operations.
# Instead, the CI smoke-test.yml will cover this.
# Instead, the CI smoke-test.yml will cover this.


# Integration test for corpus generation is impractical in unit tests
# as it requires Java tools and significant disk operations.
# Instead, the CI smoke-test.yml will cover this.
# Instead, the CI smoke-test.yml will cover this.

# Integration test for corpus generation is impractical in unit tests
# as it requires Java tools and significant disk operations.
# Instead, the CI smoke-test.yml will cover this.
# Instead, the CI smoke-test.yml will cover this.
