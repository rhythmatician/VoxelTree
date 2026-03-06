"""
Pack S dataset that generates vanilla-parity inputs from existing training data.

This dataset transforms our current NPZ data into Pack S format that matches
vanilla's noise sampling strategy.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .multi_lod_dataset import create_occupancy_from_blocks


class PackSDataset:
    """
    Dataset that provides Pack S (surface-centric) training data.

    Transforms existing NPZ data into vanilla-parity inputs:
    - Height planes from heightmap data
    - Router-6 slices (simulated from biome/height data)
    - Biome data at quart resolution (4×4×4)
    - Chunk coordinates and LOD
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        lod_sampling_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Pack S dataset.

        Args:
            data_dir: Directory containing NPZ files
            split: "train" or "val"
            lod_sampling_weights: Weights for sampling different LOD transitions
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Default sampling weights (emphasize finer LODs)
        if lod_sampling_weights is None:
            self.lod_sampling_weights = {
                "lod4to3": 0.15,  # Coarsest level
                "lod3to2": 0.2,
                "lod2to1": 0.25,
                "lod1to0": 0.4,  # Finest level (most important)
            }
        else:
            self.lod_sampling_weights = lod_sampling_weights

        # Load NPZ file paths
        self.npz_files = list(self.data_dir.glob(f"*_{split}_*.npz"))
        if not self.npz_files:
            raise ValueError(f"No NPZ files found for split '{split}' in {self.data_dir}")

        print(f"Loaded {len(self.npz_files)} NPZ files for Pack S {split} dataset")

        # Generate Pack S training pairs
        self.training_pairs = self._generate_pack_s_pairs()
        print(f"Generated {len(self.training_pairs)} Pack S training pairs")

    def _generate_pack_s_pairs(self) -> List[Dict]:
        """Generate Pack S training pairs from NPZ data."""
        training_pairs = []

        for npz_file in self.npz_files:
            try:
                data = np.load(npz_file)

                # Extract base data
                chunk_x = int(data.get("chunk_x", 0))
                chunk_z = int(data.get("chunk_z", 0))
                y_index = int(data.get("y_index", 0))

                # Get height and biome data
                heightmap = data["heightmap_patch"]  # [16,16]
                biomes = data["biome_patch"]  # [16,16]

                # Get target data
                if "target_blocks" in data:
                    target_blocks = data["target_blocks"]
                else:
                    target_blocks = data["target_types"]
                target_occupancy = create_occupancy_from_blocks(target_blocks)

                # Generate LOD transitions
                for lod_level in [1, 2, 3, 4]:
                    parent_size = 16 // (2**lod_level)  # 8, 4, 2, 1

                    if parent_size < 1:
                        continue

                    # Create parent voxel via Voxy Mipper (opacity-biased block selection)
                    parent_voxel = self._create_parent_voxel(target_blocks, parent_size)

                    # Generate Pack S features
                    pack_s_features = self._generate_pack_s_features(
                        heightmap, biomes, chunk_x, chunk_z, y_index
                    )

                    training_pair = {
                        # Pack S inputs
                        "x_parent": parent_voxel,
                        "x_height_planes": pack_s_features["height_planes"],
                        "x_router6_slices": pack_s_features["router6_slices"],
                        "x_biome_quart": pack_s_features["biome_quart"],
                        "x_chunk_coords": np.array([chunk_x, chunk_z], dtype=np.float32),
                        "x_lod": np.array([lod_level], dtype=np.int64),
                        # Targets
                        "target_blocks": target_blocks.astype(np.int64),
                        "target_occupancy": target_occupancy.astype(np.float32),
                        # Metadata
                        "lod_transition": f"lod{lod_level}to{lod_level-1}",
                    }

                    training_pairs.append(training_pair)

            except Exception as e:
                print(f"Warning: Failed to process {npz_file}: {e}")
                continue

        return training_pairs

    def _create_parent_voxel(self, target_blocks: np.ndarray, parent_size: int) -> np.ndarray:
        """Create parent voxel by Mipper-downsampling block-type labels.

        Args:
            target_blocks: (16, 16, 16) integer array of block IDs.
            parent_size:   Target size (8, 4, 2, or 1).

        Returns:
            Float32 occupancy array of shape (1, S, S, S) where S = parent_size.
        """
        from scripts.mipper import build_opacity_table, mip_volume_numpy

        factor = 16 // parent_size
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(f"Invalid parent_size {parent_size} for 16³ input")

        tbl = build_opacity_table(max(int(target_blocks.max()) + 2, 16))
        _, coarse_occ = mip_volume_numpy(target_blocks.astype(np.int64), factor, tbl)
        return coarse_occ.astype(np.float32)[None, ...]  # Add channel dimension

    def _generate_pack_s_features(
        self, heightmap: np.ndarray, biomes: np.ndarray, chunk_x: int, chunk_z: int, y_index: int
    ) -> Dict[str, np.ndarray]:
        """Generate Pack S features from basic heightmap and biome data."""

        # 1. Height planes (WORLD_SURFACE_WG + OCEAN_FLOOR_WG)
        # For now, simulate OCEAN_FLOOR_WG as surface - 10 blocks
        surface_height = heightmap  # [16,16]
        ocean_floor = np.maximum(surface_height - 0.1, 0.0)  # Simulate ocean floor
        height_planes = np.stack([surface_height, ocean_floor], axis=0)  # [2,16,16]

        # 2. Router-6 slices at 3 Y levels
        # Simulate router fields based on height and biome patterns
        router6_slices = self._simulate_router6_slices(
            heightmap, biomes, y_index, chunk_x, chunk_z
        )  # [18,16,16]

        # 3. Biome data at quart resolution (4×4×4)
        biome_quart = self._create_biome_quart(biomes, y_index)  # [64,4,4,4]

        return {
            "height_planes": height_planes,
            "router6_slices": router6_slices,
            "biome_quart": biome_quart,
        }

    def _simulate_router6_slices(
        self, heightmap: np.ndarray, biomes: np.ndarray, y_index: int, chunk_x: int, chunk_z: int
    ) -> np.ndarray:
        """Simulate router-6 fields based on heightmap and biome data."""
        H, W = heightmap.shape

        # Define 3 Y levels for sampling (deep, mid, surface)
        y_levels = [0.2, 0.5, 0.8]  # Normalized Y positions

        router6_slices = np.zeros((18, H, W), dtype=np.float32)  # 6 fields × 3 levels

        for level_idx, y_level in enumerate(y_levels):
            base_idx = level_idx * 6

            # Simulate each router field based on heightmap and biome patterns
            # Temperature: varies with height and biome
            temperature = (1.0 - heightmap) * 0.8 + np.random.normal(0, 0.1, (H, W))
            router6_slices[base_idx + 0] = np.clip(temperature, 0, 1)

            # Vegetation: correlates with temperature and moisture
            vegetation = temperature * 0.7 + np.random.normal(0, 0.1, (H, W))
            router6_slices[base_idx + 1] = np.clip(vegetation, 0, 1)

            # Continents: large-scale pattern based on chunk coords
            continent_base = np.sin(chunk_x * 0.1) * np.cos(chunk_z * 0.1)
            continents = np.full((H, W), continent_base) + np.random.normal(0, 0.1, (H, W))
            router6_slices[base_idx + 2] = np.clip(continents, -1, 1)

            # Erosion: varies with height gradient
            erosion = np.gradient(heightmap)[0] ** 2 + np.gradient(heightmap)[1] ** 2
            router6_slices[base_idx + 3] = np.clip(erosion, 0, 1)

            # Depth: based on Y level and height
            depth = y_level - heightmap + np.random.normal(0, 0.1, (H, W))
            router6_slices[base_idx + 4] = np.clip(depth, -1, 1)

            # Ridges: based on heightmap curvature
            ridges = np.abs(np.gradient(np.gradient(heightmap)[0])[0]) + np.abs(
                np.gradient(np.gradient(heightmap)[1])[1]
            )
            router6_slices[base_idx + 5] = np.clip(ridges, 0, 1)

        return router6_slices

    def _create_biome_quart(self, biomes: np.ndarray, y_index: int) -> np.ndarray:
        """Create biome data at quart resolution (4×4×4)."""
        # Downsample biomes from 16×16 to 4×4 (spatial)
        biome_4x4 = biomes[::4, ::4]  # Simple downsampling

        # Create 4×4×4 biome volume (same biome for all Y in each quart)
        biome_quart_sparse = np.zeros((4, 4, 4), dtype=np.int64)
        for i in range(4):
            for j in range(4):
                biome_quart_sparse[i, j, :] = biome_4x4[i, j]

        # Convert to one-hot encoding (max 64 biomes)
        biome_quart_onehot = np.zeros((64, 4, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    biome_id = biome_quart_sparse[i, j, k]
                    if biome_id < 64:
                        biome_quart_onehot[biome_id, i, j, k] = 1.0

        return biome_quart_onehot

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        """Get a Pack S training sample."""
        # Choose sample based on LOD sampling weights
        if random.random() < 0.1:  # 10% random sampling
            pair = self.training_pairs[idx % len(self.training_pairs)]
        else:
            # Weighted sampling by LOD transition
            transition_weights = []
            for pair in self.training_pairs:
                transition = pair["lod_transition"]
                transition_weights.append(self.lod_sampling_weights.get(transition, 1.0))

            # Sample based on weights
            chosen_idx = random.choices(
                range(len(self.training_pairs)), weights=transition_weights, k=1
            )[0]
            pair = self.training_pairs[chosen_idx]

        # Convert to tensors
        sample = {
            # Pack S inputs
            "x_parent": torch.from_numpy(pair["x_parent"]),
            "x_height_planes": torch.from_numpy(pair["x_height_planes"]),
            "x_router6_slices": torch.from_numpy(pair["x_router6_slices"]),
            "x_biome_quart": torch.from_numpy(pair["x_biome_quart"]),
            "x_chunk_coords": torch.from_numpy(pair["x_chunk_coords"]),
            "x_lod": torch.from_numpy(pair["x_lod"]),
            # Targets
            "target_blocks": torch.from_numpy(pair["target_blocks"]).long(),
            "target_occupancy": torch.from_numpy(pair["target_occupancy"]),
            # Metadata
            "lod_transition": pair["lod_transition"],
        }

        return sample


def collate_pack_s_batch(samples: List[Dict]) -> Dict[str, torch.Tensor | list]:
    """
    Collate function for Pack S batches.
    Groups samples by LOD transition type.
    """
    if not samples:
        return {}

    # Group by LOD transition
    grouped_samples: Dict[str, List[Dict]] = {}
    for sample in samples:
        transition = sample["lod_transition"]
        if transition not in grouped_samples:
            grouped_samples[transition] = []
        grouped_samples[transition].append(sample)

    # For now, just use the first transition type
    # In full training, you'd want to handle multiple transitions
    first_transition = list(grouped_samples.keys())[0]
    transition_samples = grouped_samples[first_transition]

    # Stack tensors
    batch: Dict[str, torch.Tensor | list] = {}
    for key in transition_samples[0].keys():
        if isinstance(transition_samples[0][key], torch.Tensor):
            batch[key] = torch.stack([sample[key] for sample in transition_samples])
        else:
            batch[key] = [sample[key] for sample in transition_samples]

    return batch
