"""
Loss functions for VoxelTree training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def voxel_loss_fn(
    air_mask_logits: torch.Tensor,
    block_type_logits: torch.Tensor,
    target_mask: torch.Tensor,
    target_types: torch.Tensor,
    mask_weight: float = 1.0,
    type_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined loss for voxel prediction.

    Args:
        air_mask_logits: (B, 1, H, W, D) predicted air mask logits
        block_type_logits: (B, C, H, W, D) predicted block type logits
        target_mask: (B, 1, H, W, D) target air mask (0=air, 1=solid)
        target_types: (B, H, W, D) target block types (long tensor)
        mask_weight: Weight for mask loss
        type_weight: Weight for type loss

    Returns:
        Combined scalar loss
    """
    # Binary cross-entropy for air mask
    mask_loss = F.binary_cross_entropy_with_logits(air_mask_logits, target_mask, reduction="mean")

    # Cross-entropy for block types (only where mask indicates solid blocks)
    solid_mask = target_mask.squeeze(1) > 0.5  # (B, H, W, D)

    if solid_mask.sum() > 0:
        # Only compute type loss for solid voxels
        type_logits_flat = block_type_logits.permute(0, 2, 3, 4, 1)[solid_mask]  # (N, C)
        type_targets_flat = target_types[solid_mask]  # (N,)
        type_loss = F.cross_entropy(type_logits_flat, type_targets_flat, reduction="mean")
    else:
        type_loss = torch.tensor(0.0, device=air_mask_logits.device, requires_grad=True)

    total_loss = mask_weight * mask_loss + type_weight * type_loss
    return total_loss
