"""
VoxelTreeDataset - Phase 3.1 GREEN Implementation

PyTorch Dataset implementation for loading .npz training patches.
Minimal implementation to make the RED tests pass.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class VoxelTreeDataset(Dataset):
    """
    PyTorch Dataset for loading VoxelTree training examples.

    Loads .npz files containing linked LOD pairs with seed-derived conditioning
    variables, ready for training the voxel super-resolution model.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        return_tensors: bool = False,
        lod_filter: Optional[List[int]] = None,
        cache_in_memory: bool = False,
    ):
        """
        Initialize VoxelTreeDataset.

        Args:
            data_dir: Directory containing .npz training files
            return_tensors: If True, return PyTorch tensors instead of numpy arrays
            lod_filter: List of LOD levels to include (None = all)
            cache_in_memory: If True, cache all data in memory for faster access
        """
        self.data_dir = Path(data_dir)
        self.return_tensors = return_tensors
        self.lod_filter = lod_filter
        self.cache_in_memory = cache_in_memory

        # Find all valid training files
        self.file_paths = self._find_valid_files()

        # Memory cache for data if enabled
        self._cache = {} if cache_in_memory else None

        # Pre-load data if caching enabled
        if self.cache_in_memory:
            self._preload_cache()

        logger.info(f"VoxelTreeDataset initialized with {len(self.file_paths)} training examples")

    def _find_valid_files(self) -> List[Path]:
        """Find all valid .npz files in the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        npz_files = list(self.data_dir.glob("*.npz"))
        valid_files = []

        for file_path in npz_files:
            try:
                # Quick validation by attempting to load
                data = np.load(file_path)

                # Check for required keys
                required_keys = [
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

                missing_keys = [key for key in required_keys if key not in data.keys()]
                if missing_keys:
                    logger.warning(f"Skipping {file_path}: missing keys {missing_keys}")
                    continue

                # Apply LOD filter if specified
                if self.lod_filter is not None:
                    lod = int(data["lod"])
                    if lod not in self.lod_filter:
                        continue

                valid_files.append(file_path)

            except Exception as e:
                logger.warning(f"Skipping corrupted file {file_path}: {e}")
                continue

        return valid_files

    def _preload_cache(self):
        """Pre-load all data into memory cache."""
        logger.info("Pre-loading dataset into memory cache...")
        for i in range(len(self.file_paths)):
            self._cache[i] = self._load_sample(self.file_paths[i])
        logger.info(f"Cached {len(self._cache)} training examples")

    def _load_sample(self, file_path: Path) -> Dict[str, Any]:
        """Load a single training sample from file."""
        data = np.load(file_path)

        # Convert to dictionary with proper scalar conversion
        sample = {}
        for key in data.keys():
            value = data[key]
            # Convert numpy scalars to Python scalars
            if key in ["y_index", "chunk_x", "chunk_z", "lod"]:
                sample[key] = int(value)
            else:
                sample[key] = value

        # Convert tensors if requested
        if self.return_tensors:
            sample = self._convert_to_tensors(sample)

        return sample

    def _convert_to_tensors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays to PyTorch tensors with appropriate dtypes."""
        tensor_sample = {}

        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                # Convert to appropriate tensor dtype
                if key in ["parent_voxel", "target_mask"]:
                    # Boolean masks
                    tensor_sample[key] = torch.from_numpy(value.astype(bool))
                elif key == "target_types":
                    # Block type IDs
                    tensor_sample[key] = torch.from_numpy(value.astype(np.uint8))
                elif key == "biome_patch":
                    # Biome IDs
                    tensor_sample[key] = torch.from_numpy(value.astype(np.uint8))
                elif key == "heightmap_patch":
                    # Height values (promote uint16 to int16 for PyTorch compatibility)
                    tensor_sample[key] = torch.from_numpy(value.astype(np.int16))
                elif key == "river_patch":
                    # River noise values
                    tensor_sample[key] = torch.from_numpy(value.astype(np.float32))
                else:
                    # Default conversion
                    tensor_sample[key] = torch.from_numpy(value)
            else:
                # Keep scalar values as-is
                tensor_sample[key] = value

        return tensor_sample

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training example by index.

        Args:
            idx: Index of the training example

        Returns:
            Dictionary containing training example data
        """
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.file_paths)}")

        # Use cache if available
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        # Load from file
        file_path = self.file_paths[idx]
        return self._load_sample(file_path)


class TrainingDataCollator:
    """
    Collator for batching VoxelTree training examples.

    Handles proper tensor stacking and dtype conversion for PyTorch DataLoader.
    """

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Args:
            samples: List of training example dictionaries

        Returns:
            Batched dictionary with tensor arrays
        """
        if not samples:
            return {}

        # Get all keys from first sample
        keys = samples[0].keys()
        batch = {}

        for key in keys:
            values = [sample[key] for sample in samples]

            # Handle different data types
            if key in [
                "parent_voxel",
                "target_mask",
                "target_types",
                "biome_patch",
                "heightmap_patch",
                "river_patch",
            ]:
                # Stack tensor arrays
                if isinstance(values[0], torch.Tensor):
                    stacked = torch.stack(values)
                else:
                    # Convert numpy arrays to tensors first
                    tensors = []
                    for value in values:
                        if key in ["parent_voxel", "target_mask"]:
                            tensors.append(torch.from_numpy(value.astype(bool)))
                        elif key == "target_types":
                            tensors.append(torch.from_numpy(value.astype(np.uint8)))
                        elif key == "biome_patch":
                            tensors.append(torch.from_numpy(value.astype(np.uint8)))
                        elif key == "heightmap_patch":
                            tensors.append(torch.from_numpy(value.astype(np.int16)))
                        elif key == "river_patch":
                            tensors.append(torch.from_numpy(value.astype(np.float32)))
                        else:
                            tensors.append(torch.from_numpy(value))
                    stacked = torch.stack(tensors)

                # Special handling for parent_voxel (convert bool to float32 for training)
                if key == "parent_voxel":
                    batch[key] = stacked.float()
                else:
                    batch[key] = stacked

            elif key in ["y_index", "chunk_x", "chunk_z", "lod"]:
                # Stack scalar values
                batch[key] = torch.tensor(values, dtype=torch.long)

            else:
                # Handle other data types
                try:
                    batch[key] = torch.tensor(values)
                except (ValueError, TypeError):
                    # Keep as list if can't convert to tensor
                    batch[key] = values

        return batch


class VoxelTreeDataLoader(DataLoader):
    """
    Specialized DataLoader for VoxelTree training data.

    Wrapper around PyTorch DataLoader with VoxelTree-specific defaults
    and the custom collator.
    """

    def __init__(
        self,
        dataset: VoxelTreeDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Initialize VoxelTreeDataLoader.

        Args:
            dataset: VoxelTreeDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop incomplete final batch
            **kwargs: Additional arguments passed to DataLoader
        """
        # Use custom collator
        collate_fn = TrainingDataCollator()

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs,
        )

        logger.info(
            f"VoxelTreeDataLoader initialized: batch_size={batch_size}, "
            f"shuffle={shuffle}, num_workers={num_workers}"
        )
