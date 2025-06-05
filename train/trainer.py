"""
VoxelTree training orchestration and checkpoint management.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim

from .losses import voxel_loss_fn
from .unet3d import UNet3DConfig, VoxelUNet3D


class VoxelTrainer:
    """Main trainer class for VoxelTree model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Initialize model
        model_config = UNet3DConfig(**config.get("model", {}))
        self.model = VoxelUNet3D(model_config).to(self.device)
        self.model.train()  # Ensure model is in training mode

        # Initialize optimizer
        training_config = config.get("training", {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-5),
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Loss configuration
        loss_config = config.get("loss", {})
        self.mask_weight = loss_config.get("mask_weight", 1.0)
        self.type_weight = loss_config.get("type_weight", 1.0)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute one training step."""
        self.model.train()
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Forward pass
        outputs = self.model(
            parent_voxel=batch["parent_voxel"],
            biome_patch=batch["biome_patch"],
            heightmap_patch=batch["heightmap_patch"],
            river_patch=batch["river_patch"],
            y_index=batch["y_index"],
            lod=batch["lod"],
        )
        air_mask_logits = outputs["air_mask_logits"]
        block_type_logits = outputs["block_type_logits"]

        # Compute loss
        loss = voxel_loss_fn(
            air_mask_logits=air_mask_logits,
            block_type_logits=block_type_logits,
            target_mask=batch["target_mask"],
            target_types=batch["target_types"],
            mask_weight=self.mask_weight,
            type_weight=self.type_weight,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        return loss

    def train_one_epoch(self, dataloader=None) -> Dict[str, float]:
        """Train for one epoch.

        Note: This is currently a stub implementation for TDD Phase 5.1 (RED).
        The dummy implementation is intentional to make initial tests pass.
        """
        if dataloader is None:
            # TODO: Phase 5.1 (RED) - This is a deliberate stub implementation
            # TODO: Phase 5.2 (GREEN) - Replace with proper dataloader integration
            # This dummy batch exists only to satisfy test requirements during TDD cycle
            dummy_batch = self._create_dummy_batch()
            loss = self.training_step(dummy_batch)
            self.current_epoch += 1
            return {"loss": loss.item(), "epoch": self.current_epoch - 1}

        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            loss = self.training_step(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time

        self.current_epoch += 1

        return {
            "loss": avg_loss,
            "epoch": self.current_epoch - 1,
            "epoch_time": epoch_time,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save_checkpoint(self, checkpoint_path: Path, epoch: int, loss: float, **kwargs) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "best_loss": self.best_loss,
            "config": self.config,
            "timestamp": time.time(),
            **kwargs,
        }

        torch.save(checkpoint, checkpoint_path)

        # Update best loss if this is better
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = checkpoint_path.parent / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[int, float]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        return checkpoint["epoch"], checkpoint["loss"]

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from checkpoint."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        epoch, loss = self.load_checkpoint(checkpoint_path)
        logging.info(f"Resumed training from epoch {epoch} with loss {loss:.6f}")

    def _create_dummy_batch(self) -> Dict[str, torch.Tensor]:
        """Create dummy batch for testing."""
        batch_size = self.config.get("training", {}).get("batch_size", 2)

        return {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8, device=self.device),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16), device=self.device),
            "heightmap_patch": torch.randint(
                50, 100, (batch_size, 1, 16, 16), device=self.device
            ).float(),
            "river_patch": torch.randn(batch_size, 1, 16, 16, device=self.device),
            "y_index": torch.randint(0, 24, (batch_size,), device=self.device),
            "lod": torch.randint(0, 5, (batch_size,), device=self.device),
            "target_mask": torch.randint(
                0, 2, (batch_size, 1, 16, 16, 16), device=self.device
            ).float(),
            "target_types": torch.randint(
                0, 10, (batch_size, 16, 16, 16), device=self.device
            ).long(),
        }
