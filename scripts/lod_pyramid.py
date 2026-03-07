#!/usr/bin/env python3
"""
VoxelTree LOD Pyramid Generator

Implements Item 2 from acceptance criteria: LOD pyramid generation with
2×2×2 pooling and coarsening factor support for multi-LOD training.

This utility can be integrated into the dataset/training pipeline to
generate parent voxels at multiple levels of detail from 16³ target volumes.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from scripts.mipper import build_opacity_table, mip_volume_torch

logger = logging.getLogger(__name__)

# Lazy opacity table for LOD pyramid (PyTorch)
_LOD_OPACITY_TABLE: "torch.Tensor | None" = None


def _get_lod_opacity_table() -> "torch.Tensor":
    global _LOD_OPACITY_TABLE
    if _LOD_OPACITY_TABLE is None:
        tbl = build_opacity_table(n_blocks=4096)
        _LOD_OPACITY_TABLE = torch.from_numpy(tbl).long()
    return _LOD_OPACITY_TABLE


class LODPyramidGenerator:
    """
    Generates LOD pyramids from target volumes using 2×2×2 max pooling.

    Supports coarsening factors [1, 2, 4, 8, 16] where factor f means
    the target 16³ volume is divided into f³ blocks and max-pooled.
    """

    def __init__(self, target_size: int = 16, parent_size: int = 8):
        """
        Initialize pyramid generator.

        Args:
            target_size: Size of target volumes (typically 16)
            parent_size: Size of parent volumes (typically 8)
        """
        self.target_size = target_size
        self.parent_size = parent_size
        self.valid_factors = [1, 2, 4, 8, 16]

        # Validate sizes
        if target_size not in [16, 32, 64, 128]:
            logger.warning(f"Unusual target size {target_size}, expected power of 2")
        if parent_size != 8:
            logger.warning(f"Parent size {parent_size} != 8 may not work with current model")

    def generate_parent_pyramid(self, target_mask: torch.Tensor, factor: int) -> torch.Tensor:
        """Generate parent voxel from target using Voxy Mipper coarsening.

        Args:
            target_mask: (B, C, D, H, W) target occupancy mask used *only* for shape validation.
                         Block labels from ``target_types`` (if available in caller) should be
                         passed via :meth:`generate_parent_pyramid_from_labels`.
            factor: Coarsening factor in [1, 2, 4, 8, 16]

        Returns:
            Parent voxel (B, C, parent_size, parent_size, parent_size) as float32 occupancy.

        .. note::
            This overload accepts an occupancy mask for backwards compatibility but
            falls back to OR-pooling (not Mipper) because block types are unavailable.
            Prefer :meth:`generate_parent_pyramid_from_labels` when block-type labels exist.
        """
        if factor not in self.valid_factors:
            raise ValueError(f"Factor {factor} not in valid factors {self.valid_factors}")

        B, C, D, H, W = target_mask.shape
        if D != self.target_size or H != self.target_size or W != self.target_size:
            raise ValueError(
                f"Expected target shape (..., {self.target_size}³), got {target_mask.shape}"
            )

        if factor == 1:
            coarse = target_mask
        else:
            k = factor
            reshaped = target_mask.view(B, C, D // k, k, H // k, k, W // k, k)
            coarse = reshaped.amax(dim=(3, 5, 7))  # OR-pool (fallback, no block labels)

        if coarse.shape[2:] != (self.parent_size, self.parent_size, self.parent_size):
            parent = F.interpolate(
                coarse, size=(self.parent_size, self.parent_size, self.parent_size), mode="nearest"
            )
        else:
            parent = coarse

        return parent.float()

    def generate_parent_pyramid_from_labels(
        self, target_labels: torch.Tensor, factor: int
    ) -> torch.Tensor:
        """Generate parent occupancy using Voxy Mipper from integer block-type labels.

        Args:
            target_labels: (B, D, H, W) long tensor of block IDs (0 = air).
            factor: Coarsening factor in [1, 2, 4, 8, 16]

        Returns:
            Parent occupancy (B, 1, parent_size, parent_size, parent_size) float32.
        """
        if factor not in self.valid_factors:
            raise ValueError(f"Factor {factor} not in valid factors {self.valid_factors}")

        tbl = _get_lod_opacity_table().to(target_labels.device)
        _, coarse_occ = mip_volume_torch(target_labels.long(), factor, tbl)
        coarse_occ = coarse_occ.unsqueeze(1)  # (B, 1, D//f, H//f, W//f)

        if coarse_occ.shape[2:] != (self.parent_size, self.parent_size, self.parent_size):
            parent = F.interpolate(
                coarse_occ,
                size=(self.parent_size, self.parent_size, self.parent_size),
                mode="nearest",
            )
        else:
            parent = coarse_occ

        return parent.float()

    def generate_pyramid_batch(
        self, target_masks: torch.Tensor, factors: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Generate parent pyramids for all specified factors.

        Args:
            target_masks: (B, C, D, H, W) batch of target masks
            factors: List of factors to generate (default: all valid factors)

        Returns:
            Dictionary mapping factor -> parent tensor
        """
        if factors is None:
            factors = self.valid_factors.copy()

        pyramid = {}
        for factor in factors:
            pyramid[factor] = self.generate_parent_pyramid(target_masks, factor)

        return pyramid

    def compute_lod_indices(self, factors: List[int]) -> Dict[int, int]:
        """
        Map coarsening factors to LOD indices for model conditioning.

        Args:
            factors: List of coarsening factors

        Returns:
            Dictionary mapping factor -> lod_index
        """
        # Standard mapping: 1->0, 2->1, 4->2, 8->3, 16->4
        factor_to_lod = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
        return {f: factor_to_lod[f] for f in factors if f in factor_to_lod}

    def analyze_pyramid_statistics(
        self, target_mask: torch.Tensor, pyramid: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyze occupancy statistics across pyramid levels.

        Args:
            target_mask: Original target mask
            pyramid: Generated pyramid dictionary

        Returns:
            Statistics dictionary
        """
        stats: Dict[str, Any] = {
            "target_occupancy": float(target_mask.mean()),
            "target_volume": int(target_mask.sum()),
            "pyramid_occupancies": {},
            "occupancy_preservation": {},
        }
        target_occupancy = float(target_mask.mean())

        for factor, parent in pyramid.items():
            parent_occupancy = float(parent.mean())
            stats["pyramid_occupancies"][factor] = parent_occupancy

            # Measure how well coarsening preserves occupancy
            preservation = parent_occupancy / max(target_occupancy, 1e-8)
            stats["occupancy_preservation"][factor] = preservation

        return stats


class MultiLODDatasetAugmenter:
    """
    Augments training datasets with dynamically generated multi-LOD parents.

    This can be integrated into the DataLoader/Dataset to provide
    random coarsening factor sampling during training.
    """

    def __init__(
        self, factors: List[int] = [1, 2, 4, 8, 16], factor_weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-LOD augmenter.

        Args:
            factors: Available coarsening factors
            factor_weights: Sampling weights for factors (default: uniform)
        """
        self.pyramid_gen = LODPyramidGenerator()
        self.factors = factors

        if factor_weights is None:
            self.factor_weights = [1.0] * len(factors)
        else:
            assert len(factor_weights) == len(factors)
            self.factor_weights = factor_weights

        # Normalize weights
        total_weight = sum(self.factor_weights)
        self.factor_weights = [w / total_weight for w in self.factor_weights]

    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Augment a training batch with random multi-LOD parents.

        Args:
            batch: Original training batch

        Returns:
            Augmented batch with new parent_voxel and lod values
        """
        if "target_mask" not in batch:
            logger.warning("No target_mask in batch, skipping multi-LOD augmentation")
            return batch

        # Sample random factor for this batch
        factor = np.random.choice(self.factors, p=self.factor_weights)

        # Generate new parent from target
        target_mask = batch["target_mask"]
        new_parent = self.pyramid_gen.generate_parent_pyramid(target_mask, factor)

        # Update batch
        augmented_batch = batch.copy()
        augmented_batch["parent_voxel"] = new_parent

        # Update LOD indices
        lod_mapping = self.pyramid_gen.compute_lod_indices([factor])
        lod_index = lod_mapping[factor]
        augmented_batch["lod"] = torch.full_like(batch["lod"], lod_index)

        return augmented_batch


def demo_pyramid_generation():
    """
    Demonstrate LOD pyramid generation with example data.
    """
    logging.basicConfig(level=logging.INFO)

    # Create example target volume (16³ with some structure)
    target = torch.zeros(1, 1, 16, 16, 16)

    # Add some structure: filled bottom half + some scattered blocks
    target[:, :, :8, :, :] = 1.0  # Bottom half filled
    target[:, :, 12:, 4:12, 4:12] = 1.0  # Top structure

    logger.info(f"Created target with occupancy: {target.mean():.3f}")

    # Generate pyramid
    pyramid_gen = LODPyramidGenerator()
    pyramid = pyramid_gen.generate_pyramid_batch(target)

    # Analyze results
    stats = pyramid_gen.analyze_pyramid_statistics(target, pyramid)

    logger.info("Pyramid statistics:")
    for factor in pyramid.keys():
        occ = stats["pyramid_occupancies"][factor]
        pres = stats["occupancy_preservation"][factor]
        logger.info(f"  Factor {factor:2d}: occupancy={occ:.3f}, preservation={pres:.3f}")

    # Demo augmentation
    logger.info("\nTesting dataset augmentation...")

    dummy_batch = {
        "target_mask": target,
        "parent_voxel": torch.zeros(1, 1, 8, 8, 8),  # Will be replaced
        "lod": torch.tensor([0]),  # Will be updated
        "biome_patch": torch.zeros(1, 16, 16),
    }

    augmenter = MultiLODDatasetAugmenter()

    for i in range(5):
        aug_batch = augmenter.augment_batch(dummy_batch)
        factor_used = None
        lod_used = aug_batch["lod"].item()

        # Reverse lookup factor from lod
        lod_to_factor = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}
        factor_used = (
            lod_to_factor.get(  # FIXME: No overloads for "get" match the provided arguments
                lod_used, "unknown"
            )
        )

        parent_occ = aug_batch["parent_voxel"].mean()
        logger.info(
            f"  Augmentation {i+1}: factor={factor_used}, lod={lod_used}, parent_occ={parent_occ:.3f}"  # noqa
        )


if __name__ == "__main__":
    demo_pyramid_generation()
