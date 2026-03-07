"""
Multi-LOD Training Data Pipeline

This module creates training data for the flexible multi-LOD model by:
1. Loading full-resolution 16³ chunks
2. Downsampling to create parent voxels at different LOD levels
3. Creating training pairs for all LOD transitions

The downsampling process simulates what would happen during progressive
refinement, allowing the model to learn to reverse the process.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.mipper import build_opacity_table, mip_volume_numpy
from train.anchor_conditioning import approximate_router6_from_biome

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
    router6: Optional[np.ndarray] = None,
    heightmap_surface: Optional[np.ndarray] = None,
    heightmap_ocean_floor: Optional[np.ndarray] = None,
    slope_x: Optional[np.ndarray] = None,
    slope_z: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Create training pairs for all LOD coarsening levels from a single 16³ chunk.

    The model*always* predicts 16³ from an 8³ parent.  The parent is the Voxy
    Mipper output of `labels16` downsampled by `f`, then nearest-upsampled to
    the canonical 8³ input size.  The target is always `labels16` (16³).

    Coarsening factors and resulting LOD token:
        f=2   → 8³ parent, lod=1  (native LOD1→LOD0 transition)
        f=4   → 4³→8³ parent, lod=2  (simulated LOD2→LOD0)
        f=8   → 2³→8³ parent, lod=3  (simulated LOD3→LOD0)
        f=16  → 1³→8³ parent, lod=4  (simulated LOD4→LOD0)

    Anchor channels (height_planes, router6) are passed through directly when
    available from NoiseDumper extraction.  When None, they are approximated
    from the existing heightmap and biome data.

    Args:
        labels16:            (16, 16, 16) int array of block IDs at LOD0
        biome_patch:         (16, 16) int array of biome IDs
        heightmap_patch:     (16, 16) float array of heights
        y_index:             Y-level slab index for this chunk
        air_id:              block ID that represents air (default 0)
        router6:             (6, 16, 16) float32 or None → approximate from biome
        heightmap_surface:   (16, 16) float32 or None → use heightmap_patch
        heightmap_ocean_floor:(16, 16) float32 or None → zeros if not available
        slope_x:             (16, 16) float32 or None → computed from surface height
        slope_z:             (16, 16) float32 or None → computed from surface height
        curvature:           (16, 16) float32 or None → computed from surface height

    Returns:
        List of training-sample dicts, one per coarsening factor.
    """
    tbl = _get_opacity_table(int(labels16.max()) + 1)

    # Ground-truth 16³ occupancy (used as target for all LOD levels)
    occ16 = create_occupancy_from_blocks(labels16, air_id).astype(np.float32)

    # ------------------------------------------------------------------
    # Build conditioning tensors
    # ------------------------------------------------------------------
    # Biome (legacy one-hot kept for backward compat; also keep index form)
    biome_onehot = np.eye(256, dtype=np.float32)[biome_patch]  # (16,16,256)
    biome_tensor = biome_onehot.transpose(2, 0, 1)  # (256,16,16)
    biome_idx = biome_patch.astype(np.int64)  # (16,16)  -- for anchor fusion

    # Heightmap: normalise patch to [0,1]
    heightmap_norm = (heightmap_patch.astype(np.float32) - heightmap_patch.min()) / (
        max(heightmap_patch.max() - heightmap_patch.min(), 1e-6)
    )
    heightmap_tensor = heightmap_norm[None, ...]  # (1,16,16)

    # ------------------------------------------------------------------
    # Compute height_planes tensor [5, 16, 16]
    # ------------------------------------------------------------------
    surf = heightmap_surface if heightmap_surface is not None else heightmap_norm
    ofloor = heightmap_ocean_floor if heightmap_ocean_floor is not None else np.zeros_like(surf)

    if slope_x is None or slope_z is None or curvature is None:
        # Compute via central differences
        _sx = np.gradient(surf, axis=1).astype(np.float32)
        _sz = np.gradient(surf, axis=0).astype(np.float32)
        _lap = np.gradient(_sx, axis=1) + np.gradient(_sz, axis=0)
        if slope_x is None:
            slope_x = _sx
        if slope_z is None:
            slope_z = _sz
        if curvature is None:
            curvature = _lap.astype(np.float32)

    height_planes = np.stack(
        [
            surf.astype(np.float32),
            ofloor.astype(np.float32),
            slope_x.astype(np.float32),
            slope_z.astype(np.float32),
            curvature.astype(np.float32),
        ],
        axis=0,
    )  # (5, 16, 16)

    # ------------------------------------------------------------------
    # Compute router6 [6, 16, 16] — use real data or approximate
    # ------------------------------------------------------------------
    if router6 is not None:
        router6_tensor = router6.astype(np.float32)  # (6, 16, 16)
    else:
        # Approximate from biome + heightmap using PyTorch helper
        with torch.no_grad():
            _bx = torch.from_numpy(biome_idx).unsqueeze(0).float()  # (1,16,16)
            _hx = torch.from_numpy(heightmap_norm[None, None, ...])  # (1,1,16,16)
            _r6 = approximate_router6_from_biome(_bx, _hx)  # (1,6,16,16)
        router6_tensor = _r6.squeeze(0).numpy()  # (6, 16, 16)

    training_pairs: List[Dict] = []

    for f in (2, 4, 8, 16):
        # Coarsen labels16 by factor f using the Voxy Mipper
        coarse_labels, coarse_occ = mip_volume_numpy(labels16, f, tbl)
        # shape: (16//f, 16//f, 16//f)

        # Nearest-neighbour upsample to canonical 8³
        coarse_size = coarse_labels.shape[0]  # 8, 4, 2, or 1
        if coarse_size != 8:
            scale = 8 // coarse_size
            coarse_occ_8 = np.repeat(
                np.repeat(np.repeat(coarse_occ.astype(np.float32), scale, axis=0), scale, axis=1),
                scale,
                axis=2,
            )
        else:
            coarse_occ_8 = coarse_occ.astype(np.float32)

        lod_token = int(math.log2(f))  # 1, 2, 3, 4

        training_pairs.append(
            {
                # --- model inputs (per-sample shapes, no batch dim) ---
                "parent_voxel": coarse_occ_8[None, ...],  # (1,8,8,8)  C=1
                "biome_patch": biome_tensor,  # (256,16,16) legacy one-hot
                "biome_idx": biome_idx,  # (16,16) integer indices
                "heightmap_patch": heightmap_tensor,  # (1,16,16)  C=1
                "height_planes": height_planes,  # (5,16,16)
                "router6": router6_tensor,  # (6,16,16)
                "y_index": np.int64(y_index),  # scalar
                "lod": np.int64(lod_token),  # scalar
                # --- targets (always 16³) ---
                "target_mask": occ16.astype(np.float32),  # (16,16,16)
                "target_types": labels16.astype(np.int64),  # (16,16,16)
                # --- metadata ---
                "lod_transition": f"lod{lod_token}to0",
                "parent_size": 8,
                "target_size": 16,
            }
        )

    return training_pairs


class MultiLODDataset(Dataset):
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
                "lod4to0": 0.2,  # Coarsest parent (simulated)
                "lod3to0": 0.25,
                "lod2to0": 0.25,
                "lod1to0": 0.3,  # Native LOD1→LOD0 (most important)
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

                # Extract required fields — support both canonical and legacy key names
                if "labels16" in data:
                    labels16 = data["labels16"]
                elif "target_types" in data:
                    labels16 = data["target_types"].astype(np.int32)
                else:
                    print(f"Skipping {npz_file}: no labels16 or target_types key")
                    continue

                biome16 = (
                    data["biome16"]
                    if "biome16" in data
                    else (
                        data["biome_patch"]
                        if "biome_patch" in data
                        else np.zeros((16, 16), dtype=np.int32)
                    )
                )

                height16 = (
                    data["height16"]
                    if "height16" in data
                    else (
                        data["heightmap_patch"]
                        if "heightmap_patch" in data
                        else np.zeros((16, 16), dtype=np.float32)
                    )
                )

                # river removed from contract

                # Handle different height formats
                if height16.ndim == 3:
                    height16 = height16[0]  # Take first channel

                # Generate Y-index — prefer the value stored in the NPZ
                y_index = int(data["y_index"]) if "y_index" in data else 64

                # Create training pairs for all LOD transitions
                pairs = create_lod_training_pairs(
                    labels16=labels16,
                    biome_patch=biome16,
                    heightmap_patch=height16,
                    y_index=y_index,
                    # Load real anchor data when available (from NoiseDumper extraction)
                    router6=data["router6"] if "router6" in data else None,
                    heightmap_surface=(
                        data["heightmap_surface"] if "heightmap_surface" in data else None
                    ),
                    heightmap_ocean_floor=(
                        data["heightmap_ocean_floor"] if "heightmap_ocean_floor" in data else None
                    ),
                    slope_x=data["slope_x"] if "slope_x" in data else None,
                    slope_z=data["slope_z"] if "slope_z" in data else None,
                    curvature=data["curvature"] if "curvature" in data else None,
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
            "parent_voxel": torch.from_numpy(np.asarray(pair["parent_voxel"])).float(),
            "biome_patch": torch.from_numpy(np.asarray(pair["biome_patch"])).float(),
            "biome_idx": torch.from_numpy(np.asarray(pair["biome_idx"])).long(),  # (16,16)
            "heightmap_patch": torch.from_numpy(np.asarray(pair["heightmap_patch"])).float(),
            "height_planes": torch.from_numpy(
                np.asarray(pair["height_planes"])
            ).float(),  # (5,16,16)
            "router6": torch.from_numpy(np.asarray(pair["router6"])).float(),  # (6,16,16)
            "y_index": torch.tensor(int(pair["y_index"]), dtype=torch.long),
            "lod": torch.tensor(int(pair["lod"]), dtype=torch.long),
            "target_mask": torch.from_numpy(np.asarray(pair["target_mask"])).float(),
            "target_types": torch.from_numpy(np.asarray(pair["target_types"])).long(),
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
        "biome_idx",
        "heightmap_patch",
        "height_planes",
        "router6",
        "y_index",
        "lod",
        "target_mask",
        "target_types",
    ]:
        if key in transition_samples[0]:
            batch[key] = torch.stack([s[key] for s in transition_samples], dim=0)

    batch["lod_transition"] = transition_type
    return batch
