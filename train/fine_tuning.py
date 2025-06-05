"""
Fine-tuning module for Structure-Aware training.

This module provides utilities for transfer learning from a baseline terrain model
to a structure-aware model. It handles weight loading, layer freezing, and
structure-aware training configuration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from train.unet3d import UNet3DConfig, VoxelUNet3D

logger = logging.getLogger(__name__)


def freeze_encoder_layers(model: VoxelUNet3D, unfreeze_structure_branches: bool = True) -> None:
    """
    Freeze encoder layers for fine-tuning while keeping structure branches unfrozen.

    Args:
        model: The VoxelUNet3D model to freeze
        unfreeze_structure_branches: Whether to keep structure branches unfrozen
    """
    # Freeze encoder layers
    for name, param in model.named_parameters():
        # Freeze encoder path
        if "encoder" in name:
            param.requires_grad = False

        # Freeze initial convolution
        if "initial_conv" in name:
            param.requires_grad = False

        # Freeze bottleneck (optional)
        if "bottleneck" in name and "film" not in name:
            param.requires_grad = False

        # Keep structure branches unfrozen if enabled
        if unfreeze_structure_branches and "structure" in name:
            param.requires_grad = True

    # Log trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]

    logger.info(f"Froze {len(frozen_params)} parameters for fine-tuning")
    logger.info(f"Keeping {len(trainable_params)} parameters trainable")
    logger.debug(f"Trainable parameters: {trainable_params}")


def load_baseline_weights(
    model: VoxelUNet3D, checkpoint_path: Union[str, Path], strict: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load weights from a baseline model for fine-tuning.

    Args:
        model: Target model to load weights into
        checkpoint_path: Path to baseline model checkpoint
        strict: Whether to require exact matching keys

    Returns:
        Dict containing loaded state and missing/unexpected keys
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading baseline weights from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state", checkpoint)

    # For structure-aware models, we need to filter out incompatible conditioning fusion weights
    # since the number of input channels will be different due to structure features
    if model.structure_enabled:
        # Remove conditioning fusion weights that will have size mismatches
        filtered_state = {}
        for key, value in model_state.items():
            if "conditioning_fusion.fusion_conv" in key:
                logger.info(f"Skipping incompatible layer: {key}")
                continue
            filtered_state[key] = value
        model_state = filtered_state

    # Load state dict with potential mismatched keys
    result = model.load_state_dict(model_state, strict=strict)

    # Log loading results
    if not strict and (result.missing_keys or result.unexpected_keys):
        logger.info(
            f"Loaded baseline weights with {len(result.missing_keys)} missing keys "
            f"and {len(result.unexpected_keys)} unexpected keys"
        )
        logger.debug(f"Missing keys: {result.missing_keys}")
        logger.debug(f"Unexpected keys: {result.unexpected_keys}")

    return {
        "checkpoint": checkpoint,
        "missing_keys": result.missing_keys if not strict else [],
        "unexpected_keys": result.unexpected_keys if not strict else [],
    }


def create_structure_aware_config(
    base_config: UNet3DConfig,
    structure_mask_channels: int = 1,
    structure_type_count: int = 10,
    structure_embed_dim: int = 16,
    structure_pos_channels: int = 2,
) -> UNet3DConfig:
    """
    Create a structure-aware configuration based on a baseline config.

    Args:
        base_config: Base configuration to modify
        structure_mask_channels: Number of structure mask channels
        structure_type_count: Number of structure types to encode
        structure_embed_dim: Embedding dimension for structure types
        structure_pos_channels: Number of position encoding channels

    Returns:
        Updated configuration with structure parameters enabled
    """
    # Clone the base config
    config_dict = base_config.__dict__.copy()

    # Update with structure parameters
    config_dict.update(
        {
            "structure_enabled": True,
            "structure_mask_channels": structure_mask_channels,
            "structure_type_count": structure_type_count,
            "structure_embed_dim": structure_embed_dim,
            "structure_pos_channels": structure_pos_channels,
        }
    )

    # Create new config
    return UNet3DConfig(**config_dict)
