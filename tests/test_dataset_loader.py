"""
Test suite for Phase 3: Dataset Loader

RED phase tests for PyTorch dataset integration. These tests ensure that
training data can be loaded efficiently and formatted correctly for ML training.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the classes we'll be testing (these don't exist yet - will fail)
from train.dataset import VoxelTreeDataset, VoxelTreeDataLoader, TrainingDataCollator


class TestVoxelTreeDataset:
    """Test PyTorch Dataset implementation for loading .npz training patches."""

    @pytest.fixture
    def temp_training_data(self):
        """Create temporary directory with mock training data."""
        temp_dir = Path(tempfile.mkdtemp())
        training_dir = temp_dir / "training_data"
        training_dir.mkdir()

        # Create mock training examples (linked pairs from Phase 2)
        for i in range(10):
            training_example = {
                "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                "target_types": np.random.randint(0, 10, size=(16, 16, 16), dtype=np.uint8),
                "biome_patch": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                "heightmap_patch": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                "river_patch": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                "y_index": i % 24,  # 0-23 for vertical subchunks
                "chunk_x": i % 5,
                "chunk_z": i // 5,
                "lod": 1,
            }
            np.savez_compressed(training_dir / f"linked_pair_{i}.npz", **training_example)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_dataset_init(self, temp_training_data):
        """RED: Fails if dataset initialization doesn't work."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir)

        # Should find all training files
        assert len(dataset) == 10
        assert dataset.data_dir == training_dir

    def test_dataset_getitem_shapes(self, temp_training_data):
        """RED: Fails if __getitem__ returns bad shapes."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir)

        # Get first sample
        sample = dataset[0]

        # Validate all expected keys are present
        expected_keys = [
            "parent_voxel",
            "target_mask",
            "target_types",
            "biome_patch",
            "heightmap_patch",
            "river_patch",
            "y_index",
            "chunk_x",
            "chunk_z",
            "lod",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"

        # Validate shapes match expected format
        assert sample["parent_voxel"].shape == (8, 8, 8)
        assert sample["target_mask"].shape == (16, 16, 16)
        assert sample["target_types"].shape == (16, 16, 16)
        assert sample["biome_patch"].shape == (16, 16)
        assert sample["heightmap_patch"].shape == (16, 16)
        assert sample["river_patch"].shape == (16, 16)

        # Validate scalar types
        assert isinstance(sample["y_index"], (int, np.integer))
        assert isinstance(sample["chunk_x"], (int, np.integer))
        assert isinstance(sample["chunk_z"], (int, np.integer))
        assert isinstance(sample["lod"], (int, np.integer))

    def test_dataset_tensor_conversion(self, temp_training_data):
        """RED: Fails if tensor conversion doesn't work correctly."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir, return_tensors=True)

        sample = dataset[0]

        # Should return PyTorch tensors for array data
        assert isinstance(sample["parent_voxel"], torch.Tensor)
        assert isinstance(sample["target_mask"], torch.Tensor)
        assert isinstance(sample["target_types"], torch.Tensor)
        assert isinstance(sample["biome_patch"], torch.Tensor)
        assert isinstance(sample["heightmap_patch"], torch.Tensor)
        assert isinstance(sample["river_patch"], torch.Tensor)

        # Validate tensor dtypes
        assert sample["parent_voxel"].dtype == torch.bool
        assert sample["target_mask"].dtype == torch.bool
        assert sample["target_types"].dtype == torch.uint8
        assert sample["biome_patch"].dtype == torch.uint8
        assert sample["heightmap_patch"].dtype == torch.int16  # Promote from uint16
        assert sample["river_patch"].dtype == torch.float32

    def test_dataset_file_missing_handling(self, temp_training_data):
        """RED: Fails if file missing or data misaligned."""
        training_dir = temp_training_data / "training_data"

        # Remove one file to test missing file handling
        files = list(training_dir.glob("*.npz"))
        files[0].unlink()

        dataset = VoxelTreeDataset(training_dir)
        # Should only find 9 files now
        assert len(dataset) == 9

    def test_dataset_corrupted_file_handling(self, temp_training_data):
        """RED: Fails if corrupted files aren't handled properly."""
        training_dir = temp_training_data / "training_data"

        # Create a corrupted file
        with open(training_dir / "corrupted.npz", "w") as f:
            f.write("not a valid npz file")

        dataset = VoxelTreeDataset(training_dir)

        # Should skip corrupted files and only load valid ones
        assert len(dataset) == 10  # Still 10 valid files


class TestTrainingDataCollator:
    """Test batch collation for PyTorch DataLoader."""

    @pytest.fixture
    def mock_samples(self):
        """Create mock samples for collation testing."""
        samples = []
        for i in range(4):  # Batch size of 4
            sample = {
                "parent_voxel": torch.rand(8, 8, 8),
                "target_mask": torch.randint(0, 2, (16, 16, 16), dtype=torch.bool),
                "target_types": torch.randint(0, 10, (16, 16, 16), dtype=torch.uint8),
                "biome_patch": torch.randint(0, 50, (16, 16), dtype=torch.uint8),
                "heightmap_patch": torch.randint(60, 100, (16, 16), dtype=torch.int16),
                "river_patch": torch.rand(16, 16),
                "y_index": i,
                "chunk_x": i,
                "chunk_z": i,
                "lod": 1,
            }
            samples.append(sample)
        return samples

    def test_collator_batch_shapes(self, mock_samples):
        """RED: Fails if batch collation produces wrong shapes."""
        collator = TrainingDataCollator()
        batch = collator(mock_samples)

        # Validate batch dimensions (first dim should be batch size)
        batch_size = len(mock_samples)
        assert batch["parent_voxel"].shape == (batch_size, 8, 8, 8)
        assert batch["target_mask"].shape == (batch_size, 16, 16, 16)
        assert batch["target_types"].shape == (batch_size, 16, 16, 16)
        assert batch["biome_patch"].shape == (batch_size, 16, 16)
        assert batch["heightmap_patch"].shape == (batch_size, 16, 16)
        assert batch["river_patch"].shape == (batch_size, 16, 16)

        # Validate scalar batching
        assert batch["y_index"].shape == (batch_size,)
        assert batch["chunk_x"].shape == (batch_size,)
        assert batch["chunk_z"].shape == (batch_size,)
        assert batch["lod"].shape == (batch_size,)

    def test_collator_dtype_preservation(self, mock_samples):
        """RED: Fails if data types aren't preserved during collation."""
        collator = TrainingDataCollator()
        batch = collator(mock_samples)

        # Validate dtypes are preserved
        assert batch["parent_voxel"].dtype == torch.float32  # Converted from bool
        assert batch["target_mask"].dtype == torch.bool
        assert batch["target_types"].dtype == torch.uint8
        assert batch["biome_patch"].dtype == torch.uint8
        assert batch["heightmap_patch"].dtype == torch.int16
        assert batch["river_patch"].dtype == torch.float32


class TestVoxelTreeDataLoader:
    """Test PyTorch DataLoader integration with VoxelTree dataset."""

    @pytest.fixture
    def temp_training_data(self):
        """Create temporary directory with mock training data."""
        temp_dir = Path(tempfile.mkdtemp())
        training_dir = temp_dir / "training_data"
        training_dir.mkdir()

        # Create more training examples for batching tests
        for i in range(20):
            training_example = {
                "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                "target_types": np.random.randint(0, 10, size=(16, 16, 16), dtype=np.uint8),
                "biome_patch": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                "heightmap_patch": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                "river_patch": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                "y_index": i % 24,
                "chunk_x": i % 10,
                "chunk_z": i // 10,
                "lod": (i % 4) + 1,  # LOD levels 1-4
            }
            np.savez_compressed(training_dir / f"linked_pair_{i}.npz", **training_example)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_dataloader_batch_iteration(self, temp_training_data):
        """RED: Fails if PyTorch DataLoader returns wrong batch format."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir, return_tensors=True)

        dataloader = VoxelTreeDataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0  # No multiprocessing for tests
        )

        # Get first batch
        batch = next(iter(dataloader))

        # Validate batch format
        assert isinstance(batch, dict)
        assert "parent_voxel" in batch
        assert "target_mask" in batch
        assert "target_types" in batch

        # Validate batch dimensions
        assert batch["parent_voxel"].shape[0] == 4  # Batch size
        assert batch["target_mask"].shape[0] == 4
        assert batch["target_types"].shape[0] == 4

    def test_dataloader_full_epoch(self, temp_training_data):
        """RED: Fails if full epoch iteration doesn't work."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir, return_tensors=True)

        dataloader = VoxelTreeDataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        total_samples = 0
        for batch in dataloader:
            total_samples += batch["parent_voxel"].shape[0]

        # Should process all 20 samples
        assert total_samples == 20

    def test_dataloader_shuffling(self, temp_training_data):
        """RED: Fails if shuffling doesn't work correctly."""
        training_dir = temp_training_data / "training_data"
        dataset = VoxelTreeDataset(training_dir, return_tensors=True)

        # Create two dataloaders with different shuffle settings
        dataloader_no_shuffle = VoxelTreeDataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )
        dataloader_shuffle = VoxelTreeDataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        # Get first batches
        batch_no_shuffle = next(iter(dataloader_no_shuffle))
        batch_shuffle = next(iter(dataloader_shuffle))

        # Shapes should be the same
        assert batch_no_shuffle["parent_voxel"].shape == batch_shuffle["parent_voxel"].shape

        # With shuffle=True, order might be different (not guaranteed to be different in small dataset)
        # Just verify shuffle=True doesn't break anything
        assert batch_shuffle["parent_voxel"].shape[0] == 4


class TestDatasetConfiguration:
    """Test dataset configuration and filtering options."""

    def test_dataset_lod_filtering(self, tmp_path):
        """RED: Fails if LOD level filtering doesn't work."""
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()

        # Create examples with different LOD levels
        for lod in [1, 2, 3, 4]:
            for i in range(5):
                training_example = {
                    "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                    "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                    "target_types": np.random.randint(0, 10, size=(16, 16, 16), dtype=np.uint8),
                    "biome_patch": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                    "heightmap_patch": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                    "river_patch": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                    "y_index": i,
                    "chunk_x": i,
                    "chunk_z": i,
                    "lod": lod,
                }
                np.savez_compressed(
                    training_dir / f"linked_pair_lod{lod}_{i}.npz", **training_example
                )  # Test filtering by LOD level
        dataset_lod1 = VoxelTreeDataset(training_dir, lod_filter=[1])
        assert len(dataset_lod1) == 5

        dataset_lod12 = VoxelTreeDataset(training_dir, lod_filter=[1, 2])
        assert len(dataset_lod12) == 10

        dataset_all = VoxelTreeDataset(training_dir)  # No filter
        assert len(dataset_all) == 20

    def test_dataset_memory_management(self, tmp_path):
        """RED: Fails if memory management options don't work."""
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()

        # Create a few training examples
        for i in range(5):
            training_example = {
                "parent_voxel": np.random.choice([True, False], size=(8, 8, 8)),
                "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
                "target_types": np.random.randint(0, 10, size=(16, 16, 16), dtype=np.uint8),
                "biome_patch": np.random.randint(0, 50, size=(16, 16), dtype=np.uint8),
                "heightmap_patch": np.random.randint(60, 100, size=(16, 16), dtype=np.uint16),
                "river_patch": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float32),
                "y_index": i,
                "chunk_x": i,
                "chunk_z": i,
                "lod": 1,
            }
            np.savez_compressed(training_dir / f"linked_pair_{i}.npz", **training_example)

        # Test lazy loading (default)
        dataset_lazy = VoxelTreeDataset(training_dir, cache_in_memory=False)
        assert len(dataset_lazy) == 5

        # Test memory caching
        dataset_cached = VoxelTreeDataset(training_dir, cache_in_memory=True)
        assert len(dataset_cached) == 5

        # Both should return same data
        sample_lazy = dataset_lazy[0]
        sample_cached = dataset_cached[0]
        assert sample_lazy["y_index"] == sample_cached["y_index"]
