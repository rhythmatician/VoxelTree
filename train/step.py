"""
Training step implementation for VoxelTree Phase 5.2 (GREEN)

Implements a single training step that:
- Runs forward pass through model
- Computes loss using voxel_loss_fn
- Performs backpropagation and optimizer step
- Updates model parameters
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Callable


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable,
    device: str = "cpu",
) -> float:
    """
    Perform a single training step.

    Args:
        model: VoxelUNet3D model
        optimizer: PyTorch optimizer (e.g., Adam)
        batch: Dictionary containing input tensors and targets
        loss_fn: Loss function (e.g., voxel_loss_fn)
        device: Device to run on ("cpu" or "cuda")

    Returns:
        float: Loss value as a Python float
    """
    # Zero gradients before forward pass
    optimizer.zero_grad()

    # Move batch to device
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value

    # Move model to device
    model = model.to(device)
    # Forward pass through model
    outputs = model(
        parent_voxel=batch_device["parent_voxel"],
        biome_patch=batch_device["biome_patch"],
        heightmap_patch=batch_device["heightmap_patch"],
        river_patch=batch_device["river_patch"],
        y_index=batch_device["y_index"],
        lod=batch_device["lod"],
    )

    # Extract outputs
    air_mask_logits = outputs["air_mask_logits"]
    block_type_logits = outputs["block_type_logits"]

    # Compute loss
    loss = loss_fn(
        air_mask_logits=air_mask_logits,
        block_type_logits=block_type_logits,
        target_mask=batch_device["target_mask"],
        target_types=batch_device["target_types"],
    )

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Return loss as float
    return loss.item()
