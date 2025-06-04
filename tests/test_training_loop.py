"""
Test training loop functionality for VoxelTree Phase 5.1 (RED)

Tests the basic training loop with:
- Dry run 1 epoch (5.1)
- Checkpoint saving (5.2)
- Resume training (5.3)
- CSV/TensorBoard logs (5.4)
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from train.dataset import VoxelTreeDataset
from train.unet3d import UNet3DConfig, VoxelUNet3D


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
        )  # Must be scalar, differentiable, and nonzero
        assert loss.shape == (), "Loss must be scalar"
        assert loss.requires_grad, "Loss must support gradient flow"
        loss.backward()
        assert air_mask_logits.grad is not None
        assert block_type_logits.grad is not None

    def test_dry_run_one_epoch_passes_with_trainer(self):
        """Test that training can complete one epoch (Phase 5.1)"""
        from train.trainer import VoxelTrainer

        config = {
            "model": {"base_channels": 32, "depth": 3},
            "training": {"batch_size": 2, "learning_rate": 1e-4},
        }

        trainer = VoxelTrainer(config)
        metrics = trainer.train_one_epoch()

        assert "loss" in metrics
        assert "epoch" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["epoch"] == 0

    def test_forward_pass_integration_passes_with_trainer(self):
        """Test full forward pass through model during training"""
        from train.trainer import VoxelTrainer

        # Create mock data batch
        batch = {
            "parent_voxel": torch.randn(2, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (2, 16, 16)),
            "heightmap_patch": torch.randint(50, 100, (2, 1, 16, 16)).float(),
            "river_patch": torch.randn(2, 1, 16, 16),
            "y_index": torch.randint(0, 24, (2,)),
            "lod": torch.randint(0, 5, (2,)),
            "target_mask": torch.randint(0, 2, (2, 1, 16, 16, 16)).float(),
            "target_types": torch.randint(0, 10, (2, 16, 16, 16)).long(),
        }

        trainer = VoxelTrainer({})
        loss = trainer.training_step(batch)
        assert loss.requires_grad


class TestCheckpointSaving:
    """Test checkpoint saving and loading functionality (Phase 5.2)"""

    def test_save_checkpoint_creates_file(self):
        """Test saving model checkpoint"""
        from train.trainer import VoxelTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = VoxelTrainer({})
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

            trainer.save_checkpoint(checkpoint_path, epoch=5, loss=0.123)

            assert checkpoint_path.exists()

            # Verify checkpoint contents
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert checkpoint["epoch"] == 5
            assert checkpoint["loss"] == 0.123
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint_restores_state(self):
        """Test loading model checkpoint"""
        from train.trainer import VoxelTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save checkpoint
            trainer1 = VoxelTrainer({})
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            trainer1.save_checkpoint(checkpoint_path, epoch=10, loss=0.456)

            # Create new trainer and load checkpoint
            trainer2 = VoxelTrainer({})
            epoch, loss = trainer2.load_checkpoint(checkpoint_path)

            assert epoch == 10
            assert loss == 0.456
            assert trainer2.current_epoch == 10


class TestResumeTraining:
    """Test training resumption functionality (Phase 5.3)"""

    def test_resume_training_loads_state(self):
        """Test that training can be resumed from checkpoint"""
        from train.trainer import VoxelTrainer

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save checkpoint
            trainer1 = VoxelTrainer({})
            trainer1.current_epoch = 15
            trainer1.global_step = 1000
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            trainer1.save_checkpoint(checkpoint_path, epoch=15, loss=0.789)

            # Resume training
            trainer2 = VoxelTrainer({})
            trainer2.resume_from_checkpoint(checkpoint_path)

            assert trainer2.current_epoch == 15
            assert trainer2.global_step == 1000

    def test_resume_training_fails_missing_checkpoint(self):
        """Test resume fails gracefully when checkpoint missing"""
        from train.trainer import VoxelTrainer

        trainer = VoxelTrainer({})
        with pytest.raises(FileNotFoundError):
            trainer.resume_from_checkpoint(Path("nonexistent_checkpoint.pt"))


class TestTrainingLogs:
    """Test training logging functionality (Phase 5.4)"""

    def test_csv_logging_creates_file(self):
        """Test CSV logging of training metrics"""
        from train.logger import TrainingLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(log_dir=Path(temp_dir))
            logger.log_metrics({"epoch": 1, "loss": 0.5, "lr": 1e-4})

            csv_path = Path(temp_dir) / "training_log.csv"
            assert csv_path.exists()

            # Verify CSV contents
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["epoch"] == "1"
                assert rows[0]["loss"] == "0.5"

    def test_tensorboard_logging_initializes(self):
        """Test TensorBoard logging initialization"""
        from train.logger import TrainingLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(log_dir=Path(temp_dir), use_tensorboard=True)
            logger.log_metrics({"epoch": 1, "loss": 0.5, "lr": 1e-4})

            # Should not raise errors
            logger.close()

    def test_jsonl_logging_creates_file(self):
        """Test JSONL logging of training metrics"""
        from train.logger import TrainingLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(log_dir=Path(temp_dir))
            logger.log_metrics({"epoch": 1, "loss": 0.5, "lr": 1e-4})

            jsonl_path = Path(temp_dir) / "training_log.jsonl"
            assert jsonl_path.exists()

            # Verify JSONL contents
            with open(jsonl_path, "r") as f:
                line = f.readline().strip()
                data = json.loads(line)
                assert data["epoch"] == 1
                assert data["loss"] == 0.5
