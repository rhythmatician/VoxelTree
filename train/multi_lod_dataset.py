"""
Multi-LOD Training Data Pipeline

This module creates training data for the flexible multi-LOD model by:
1. Loading full-resolution 16³ chunks
2. Downsampling to create parent voxels at different LOD levels
3. Creating training pairs for all LOD transitions

The downsampling process simulates what would happen during progressive
refinement, allowing the model to learn to reverse the process.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from scripts.mipper import build_opacity_table, mip_volume_numpy

# Module-level opacity table (lazy)
_OPACITY_TABLE: np.ndarray | None = None


def _get_opacity_table(max_id: int = 4096) -> np.ndarray:
    global _OPACITY_TABLE
    if _OPACITY_TABLE is None or len(_OPACITY_TABLE) < max_id + 1:
        _OPACITY_TABLE = build_opacity_table(max(max_id + 1, 4096))
    return _OPACITY_TABLE


def create_occupancy_from_blocks(block_data: np.ndarray, air_id: int = 0) -> np.ndarray:
    """
    Create binary occupancy data from block IDs.

    Args:
        block_data: Array of block IDs
        air_id: ID of air blocks (default 0)

    Returns:
        Binary occupancy array (1 = solid, 0 = air)
    """
    return (block_data != air_id).astype(np.uint8)


def create_lod_training_pairs(
    labels16: np.ndarray,
    biome_patch: np.ndarray,
    heightmap_patch: np.ndarray,
    y_index: int = 0,
    air_id: int = 0,
) -> List[Dict]:
    """
    Create training pairs for all LOD transitions from a single 16³ chunk.

    Args:
        labels16: (16, 16, 16) array of block IDs
        biome_patch: (16, 16) array of biome IDs
        heightmap_patch: (16, 16) array of heights
        y_index: Y-level index for this chunk
        air_id: ID of air blocks

    Returns:
        List of training samples for different LOD transitions
    """
    # Create occupancy data
    occ16 = create_occupancy_from_blocks(labels16, air_id)

    # Generate all LOD levels by downsampling
    lod_data: Dict[int, Dict[str, np.ndarray | int]] = {}

    # LOD 0 (target resolution)
    lod_data[0] = {"blocks": labels16, "occupancy": occ16, "size": 16}

    # LOD 1, 2, 3, 4 (progressively coarser)
    for lod_level in [1, 2, 3, 4]:
        factor = 2**lod_level  # 2, 4, 8, 16
        size = 16 // factor  # 8, 4, 2, 1

        # Downsample blocks using Voxy Mipper (opacity-biased corner selection)
        tbl = _get_opacity_table(int(labels16.max()))
        blocks_down, occ_down = mip_volume_numpy(labels16, factor, tbl)

        lod_data[lod_level] = {"blocks": blocks_down, "occupancy": occ_down, "size": size}

    # Create training pairs for each LOD transition
    training_pairs: List[Dict] = []

    for lod_level in [4, 3, 2, 1]:  # Parent LOD levels
        target_lod = lod_level - 1  # Target LOD level

        parent_data = lod_data[lod_level]
        target_data = lod_data[target_lod]

        # Prepare input data
        parent_occupancy = np.asarray(parent_data["occupancy"]).astype(np.float32)
        target_blocks = np.asarray(target_data["blocks"])
        target_occupancy = np.asarray(target_data["occupancy"])

        # Add batch and channel dimensions
        parent_voxel = parent_occupancy[None, None, ...]  # (1, 1, S, S, S)

        # Add batch dimensions to targets
        target_blocks = target_blocks[None, ...]  # (1, H, W, D)
        target_occupancy = target_occupancy[None, ...]  # (1, H, W, D)

        # Prepare conditioning data
        biome_onehot = np.eye(256, dtype=np.float32)[biome_patch]  # (16, 16, 256)
        biome_tensor = biome_onehot.transpose(2, 0, 1)[None, ...]  # (1, 256, 16, 16)

        heightmap_norm = heightmap_patch.astype(np.float32) / 256.0  # Normalize
        heightmap_tensor = heightmap_norm[None, None, ..., None]  # (1, 1, 16, 16, 1)

        training_pair = {
            # Inputs
            "parent_voxel": parent_voxel,
            "biome_patch": biome_tensor,
            "heightmap_patch": heightmap_tensor,
            "y_index": np.array([y_index], dtype=np.int64),
            "lod": np.array([lod_level], dtype=np.int64),
            # Targets
            "target_blocks": target_blocks,
            "target_occupancy": target_occupancy,
            "target_size": target_data["size"],
            # Metadata
            "lod_transition": f"lod{lod_level}to{target_lod}",
            "parent_size": parent_data["size"],
        }

        training_pairs.append(training_pair)

    return training_pairs


class MultiLODDataset:
    """
    Dataset that provides multi-LOD training pairs from NPZ chunk data.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        lod_sampling_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-LOD dataset.

        Args:
            data_dir: Directory containing NPZ files
            split: "train" or "val"
            lod_sampling_weights: Weights for sampling different LOD transitions
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Default sampling weights (can emphasize certain LOD levels)
        if lod_sampling_weights is None:
            self.lod_sampling_weights = {
                "lod4to3": 0.2,  # Coarsest level
                "lod3to2": 0.25,
                "lod2to1": 0.25,
                "lod1to0": 0.3,  # Finest level (most important)
            }
        else:
            self.lod_sampling_weights = lod_sampling_weights

        # Load NPZ file paths
        self.npz_files = list(self.data_dir.glob(f"*_{split}_*.npz"))
        if not self.npz_files:
            self.npz_files = list(self.data_dir.glob("*.npz"))

        print(f"Found {len(self.npz_files)} NPZ files for {split} split")

        # Pre-generate all training pairs (memory permitting)
        self.training_pairs: List[Dict] = []
        self._generate_all_pairs()

    def _generate_all_pairs(self):
        """Generate all possible training pairs from NPZ files."""
        print("Generating multi-LOD training pairs...")

        for npz_file in self.npz_files:
            try:
                data = np.load(npz_file)

                # Extract required fields
                labels16 = data["labels16"]  # (16, 16, 16)
                biome16 = data.get("biome16", np.zeros((16, 16), dtype=np.int32))
                height16 = data.get("height16", np.zeros((1, 16, 16), dtype=np.float32))
                # river removed from contract

                # Handle different height formats
                if height16.ndim == 3:
                    height16 = height16[0]  # Take first channel

                # Generate Y-index (could be extracted from filename or data)
                y_index = 64  # Default Y-level, could be randomized or extracted

                # Create training pairs for all LOD transitions
                pairs = create_lod_training_pairs(
                    labels16=labels16,
                    biome_patch=biome16,
                    heightmap_patch=height16,
                    y_index=y_index,
                )

                self.training_pairs.extend(pairs)

            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                continue

        print(f"Generated {len(self.training_pairs)} total training pairs")

        # Print distribution by LOD transition
        lod_counts = {}
        for pair in self.training_pairs:
            transition = pair["lod_transition"]
            lod_counts[transition] = lod_counts.get(transition, 0) + 1

        for transition, count in lod_counts.items():
            print(f"  {transition}: {count} pairs")

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        """Get a training sample."""
        # Choose sample based on LOD sampling weights
        if random.random() < 0.1:  # 10% random sampling
            pair = self.training_pairs[idx % len(self.training_pairs)]
        else:
            # Weighted sampling by LOD transition
            transition_weights = []
            transitions = []
            for pair in self.training_pairs:
                transition = pair["lod_transition"]
                transitions.append(transition)
                transition_weights.append(self.lod_sampling_weights.get(transition, 1.0))

            # Sample based on weights
            chosen_idx = random.choices(
                range(len(self.training_pairs)), weights=transition_weights, k=1
            )[0]
            pair = self.training_pairs[chosen_idx]

        # Convert to tensors
        sample = {
            "parent_voxel": torch.from_numpy(pair["parent_voxel"]),
            "biome_patch": torch.from_numpy(pair["biome_patch"]),
            "heightmap_patch": torch.from_numpy(pair["heightmap_patch"]),
            "y_index": torch.from_numpy(pair["y_index"]),
            "lod": torch.from_numpy(pair["lod"]),
            "target_blocks": torch.from_numpy(pair["target_blocks"]).long(),  # Convert to int64
            "target_occupancy": torch.from_numpy(pair["target_occupancy"]),
            "lod_transition": pair["lod_transition"],
        }

        return sample


def collate_multi_lod_batch(samples: List[Dict]) -> Dict:
    """
    Collate function for multi-LOD batches.
    Groups samples by LOD transition type.
    """
    # Group samples by LOD transition
    grouped: Dict[str, List[Dict]] = {}
    for sample in samples:
        transition = sample["lod_transition"]
        if transition not in grouped:
            grouped[transition] = []
        grouped[transition].append(sample)

    # For now, just take the first transition type
    # In practice, you might want to process all types or sample one
    transition_type = list(grouped.keys())[0]
    transition_samples = grouped[transition_type]

    # Standard batching for samples of the same transition type
    # Use a general dict for mixed types (tensors + metadata string)
    batch: Dict[str, object] = {}
    for key in [
        "parent_voxel",
        "biome_patch",
        "heightmap_patch",
        "y_index",
        "lod",
        "target_blocks",
        "target_occupancy",
    ]:
        if key in transition_samples[0]:
            # Use cat instead of stack since our samples already have batch dimension
            batch[key] = torch.cat([s[key] for s in transition_samples], dim=0)

    batch["lod_transition"] = transition_type
    return batch
