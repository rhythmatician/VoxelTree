"""
Loss functions for VoxelTree training.

Implements combined loss for air mask prediction and block type classification.
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
        air_mask_logits: Predicted air/solid logits, shape (B, 1, H, W, D)
        block_type_logits: Predicted block type logits, shape (B, C, H, W, D)
        target_mask: Target air/solid mask, shape (B, 1, H, W, D)
        target_types: Target block type indices, shape (B, H, W, D)
        mask_weight: Weight for air mask loss
        type_weight: Weight for block type loss

    Returns:
        Combined loss tensor (scalar)
    """
    # Air mask loss (binary classification)
    mask_loss = F.binary_cross_entropy_with_logits(
        air_mask_logits.squeeze(1), target_mask.squeeze(1)  # Remove channel dim for BCE
    )

    # Block type loss (multi-class classification)
    # Only compute loss where target is not air (mask == 0)
    solid_mask = target_mask.squeeze(1) == 0  # Solid blocks where mask is False

    if solid_mask.sum() > 0:
        # Flatten spatial dimensions for cross entropy
        type_logits_flat = block_type_logits.permute(0, 2, 3, 4, 1).contiguous()
        type_logits_flat = type_logits_flat.view(-1, block_type_logits.size(1))

        target_types_flat = target_types.view(-1)
        solid_mask_flat = solid_mask.view(-1)

        # Only compute loss on solid blocks
        type_loss = F.cross_entropy(
            type_logits_flat[solid_mask_flat], target_types_flat[solid_mask_flat]
        )
    else:
        # No solid blocks, zero loss
        type_loss = torch.zeros(1, device=air_mask_logits.device, requires_grad=True)

    # Combine losses
    total_loss = mask_weight * mask_loss + type_weight * type_loss

    return total_loss
