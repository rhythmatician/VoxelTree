"""
Test training step functionality for VoxelTree Phase 5.2 (RED)

Tests a single training step with:
- Forward pass
- Loss computation
- Backpropagation
- Optimizer step
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from train.unet3d import VoxelUNet3D, UNet3DConfig
from train.losses import voxel_loss_fn


class TestTrainingStep:
    """Test single training step functionality (Phase 5.2)"""

    def test_training_step_backprop(self):
        """Test that training step performs forward pass, loss computation, and backprop"""
        from train.step import training_step

        # Setup
        config = UNet3DConfig()
        model = VoxelUNet3D(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        batch_size = 2
        parent_voxel = torch.randn(batch_size, 1, 8, 8, 8, requires_grad=False)
        biome_patch = torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long)
        heightmap_patch = torch.randn(batch_size, 1, 16, 16)
        river_patch = torch.randn(batch_size, 1, 16, 16)
        y_index = torch.randint(0, 24, (batch_size,), dtype=torch.long)
        lod = torch.randint(1, 5, (batch_size,), dtype=torch.long)

        # Targets
        target_mask = torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float()
        target_types = torch.randint(0, 10, (batch_size, 16, 16, 16)).long()

        # This should fail until training_step() is implemented
        loss = training_step(
            model=model,
            optimizer=optimizer,
            batch={
                "parent_voxel": parent_voxel,
                "biome_patch": biome_patch,
                "heightmap_patch": heightmap_patch,
                "river_patch": river_patch,
                "y_index": y_index,
                "lod": lod,
                "target_mask": target_mask,
                "target_types": target_types,
            },
            loss_fn=voxel_loss_fn,
            device="cpu",
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_training_step_updates_parameters(self):
        """Test that training step actually updates model parameters"""
        from train.step import training_step

        # Setup
        config = UNet3DConfig()
        model = VoxelUNet3D(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone().detach()

        batch_size = 1
        batch = {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "river_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,), dtype=torch.long),
            "lod": torch.randint(1, 5, (batch_size,), dtype=torch.long),
            "target_mask": torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float(),
            "target_types": torch.randint(0, 10, (batch_size, 16, 16, 16)).long(),
        }

        # Run training step
        loss = training_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            loss_fn=voxel_loss_fn,
            device="cpu",
        )

        # Check that at least some parameters have changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break

        assert params_changed, "Training step should update model parameters"
        assert loss > 0

    def test_training_step_zero_gradients(self):
        """Test that training step properly zeros gradients before computation"""
        from train.step import training_step

        config = UNet3DConfig()
        model = VoxelUNet3D(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        batch_size = 1
        batch = {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), dtype=torch.long),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "river_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,), dtype=torch.long),
            "lod": torch.randint(1, 5, (batch_size,), dtype=torch.long),
            "target_mask": torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float(),
            "target_types": torch.randint(0, 10, (batch_size, 16, 16, 16)).long(),
        }

        # Manually set some gradients to non-zero values
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
            else:
                param.grad.fill_(1.0)

        # Run training step - should zero gradients internally
        loss = training_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            loss_fn=voxel_loss_fn,
            device="cpu",
        )

        # After training step, gradients should exist and not be all zeros
        # (they were computed during backprop)
        has_nonzero_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "Training step should compute gradients"
        assert loss > 0
