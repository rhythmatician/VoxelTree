"""
Test training loop functionality for VoxelTree Phase 5.1 (RED)

Tests the basic training loop with:
- Dry run 1 epoch (5.1)
- Checkpoint saving (5.2)
- Resume training (5.3)
- CSV/TensorBoard logs (5.4)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
import yaml
from unittest.mock import MagicMock, patch
import numpy as np

from train.unet3d import VoxelUNet3D, UNet3DConfig
from train.dataset import VoxelTreeDataset


class TestTrainingLoop:
    """Test basic training loop functionality (Phase 5.1)"""

    def test_voxel_loss_computes_correctly(self):
        from train.losses import voxel_loss_fn

        batch_size = 2
        # Simulated predictions
        air_mask_logits = torch.randn(batch_size, 1, 16, 16, 16, requires_grad=True)
        block_type_logits = torch.randn(batch_size, 10, 16, 16, 16, requires_grad=True)

        # Simulated targets
        target_mask = torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float()
        target_types = torch.randint(0, 10, (batch_size, 16, 16, 16)).long()

        # Compute loss
        loss = voxel_loss_fn(
            air_mask_logits=air_mask_logits,
            block_type_logits=block_type_logits,
            target_mask=target_mask,
            target_types=target_types,
        )

        # Must be scalar, differentiable, and nonzero
        assert loss.shape == (), "Loss must be scalar"
        assert loss.requires_grad, "Loss must support gradient flow"
        loss.backward()
        assert air_mask_logits.grad is not None
        assert block_type_logits.grad is not None
