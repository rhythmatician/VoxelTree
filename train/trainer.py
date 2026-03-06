"""
VoxelTree training orchestration and checkpoint management.
"""

import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Mipper for Voxy-compatible LOD coarsening
from scripts.mipper import build_opacity_table
from scripts.mipper import mip_volume_torch as _mip_volume_torch

from .confusion_analyzer import create_confusion_analyzer
from .losses import voxel_loss_fn
from .unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D
from .visualizer import TensorBoardLogger

_OPACITY_TABLE_TORCH: "torch.Tensor | None" = None  # noqa: F821


def _get_opacity_table_torch() -> "torch.Tensor":  # noqa: F821
    global _OPACITY_TABLE_TORCH
    if _OPACITY_TABLE_TORCH is None:
        import torch as _torch

        tbl = build_opacity_table(n_blocks=4096)
        _OPACITY_TABLE_TORCH = _torch.from_numpy(tbl).long()
    return _OPACITY_TABLE_TORCH


class VoxelTrainer:
    """Main trainer class for VoxelTree model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model
        model_config = SimpleFlexibleConfig(**config.get("model", {}))
        self.model = SimpleFlexibleUNet3D(model_config).to(self.device)
        self.model.train()  # Ensure model is in training mode

        # Initialize optimizer
        training_config = config.get("training", {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-5),
        )  # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Initialize TensorBoard logger
        tensorboard_config = training_config.get("tensorboard", {})
        if tensorboard_config.get("enabled", False):
            log_dir = tensorboard_config.get("log_dir", "runs/tensorboard")
            self.tb_logger: Optional[TensorBoardLogger] = TensorBoardLogger(log_dir, enabled=True)

            # Log model graph if enabled (will be called with first batch)
            if tensorboard_config.get("log_model_graph", False):
                self._log_model_graph_pending = True
            else:
                self._log_model_graph_pending = False
        else:
            self.tb_logger = None

        # Initialize confusion matrix analyzer for 99% accuracy tracking
        self.confusion_analyzer = create_confusion_analyzer(config)

        # Multi-LOD / scheduled sampling configuration
        mlod_cfg = training_config.get("multi_lod", {})
        self.multi_lod_enabled: bool = mlod_cfg.get("enabled", False)
        self.multi_lod_factors: List[int] = mlod_cfg.get("factors", [1, 2, 4, 8, 16])
        # Ensure valid powers of two up to 16
        self.multi_lod_factors = [f for f in self.multi_lod_factors if f in [1, 2, 4, 8, 16]]
        if self.multi_lod_enabled and not self.multi_lod_factors:
            raise ValueError("multi_lod enabled but no valid factors provided")

        sched_cfg = training_config.get("scheduled_sampling", {})
        self.sched_enabled: bool = sched_cfg.get("enabled", False)
        self.sched_start: float = float(sched_cfg.get("start_prob", 0.0))
        self.sched_end: float = float(sched_cfg.get("end_prob", 0.3))
        self.sched_warmup_epochs: int = int(sched_cfg.get("warmup_epochs", 1))
        self.sched_total_epochs: int = int(
            sched_cfg.get("total_epochs", max(1, training_config.get("epochs", 1)))
        )

        # Loss configuration
        loss_config = config.get("loss", {})
        self.mask_weight = loss_config.get("mask_weight", 1.0)
        self.type_weight = loss_config.get("type_weight", 1.0)
        # Cache for scheduled sampling (initialized lazily)
        self._last_air_mask_pred: Optional[torch.Tensor] = None

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute one training step."""
        self.model.train()
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Log model graph with first batch if pending
        if hasattr(self, "_log_model_graph_pending") and self._log_model_graph_pending:
            if self.tb_logger:
                try:
                    self.tb_logger.log_model_graph(self.model, batch)
                    self._log_model_graph_pending = False
                except Exception as e:
                    logging.warning(f"Failed to log model graph: {e}")
                    self._log_model_graph_pending = False

        # Optionally replace parent voxel with dynamically generated multi-LOD parent
        if self.multi_lod_enabled and "target_types" in batch:
            factor = random.choice(self.multi_lod_factors)
            # Map factor -> lod index (powers of two) 1->0,2->1,4->2,8->3,16->4
            lod_index = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[factor]
            target_types = batch["target_types"]  # (B, 16, 16, 16) integer block IDs
            if target_types.dim() == 5:
                target_types = target_types.squeeze(1)  # (B, 16, 16, 16)
            with torch.no_grad():
                # Coarsen using Voxy Mipper (opacity-biased, I111 tie-break)
                tbl = _get_opacity_table_torch().to(target_types.device)
                coarse_labels, coarse_occ = _mip_volume_torch(
                    target_types.long(), factor, tbl
                )  # (B, 16/f, 16/f, 16/f)
                coarse_occ = coarse_occ.unsqueeze(1)  # (B, 1, 16/f, 16/f, 16/f)
                # Upsample to canonical 8³ parent size if needed
                if coarse_occ.shape[2:] != (8, 8, 8):
                    coarse_up = F.interpolate(coarse_occ, size=(8, 8, 8), mode="nearest")
                else:
                    coarse_up = coarse_occ
                # Scheduled sampling: optionally blend with previous model prediction
                if self.sched_enabled and self.current_epoch >= self.sched_warmup_epochs:
                    progress = min(
                        1.0,
                        (self.current_epoch - self.sched_warmup_epochs)
                        / max(1, (self.sched_total_epochs - self.sched_warmup_epochs)),
                    )
                    p = self.sched_start + (self.sched_end - self.sched_start) * progress
                    if random.random() < p and self._last_air_mask_pred is not None:
                        # Threshold previous logits (<0 -> solid occupancy)
                        pred_parent = (self._last_air_mask_pred.detach() < 0).float()
                        if pred_parent.shape != coarse_up.shape:
                            pred_parent = F.interpolate(pred_parent, size=(8, 8, 8), mode="nearest")
                        # Blend: occupancy OR
                        coarse_up = torch.clamp(coarse_up + pred_parent, 0, 1)
                batch["parent_voxel"] = coarse_up.float()
                batch["lod"] = torch.full_like(batch["lod"], lod_index, dtype=torch.long)

        # Forward pass
        outputs = self.model(
            parent_voxel=batch["parent_voxel"],
            biome_patch=batch["biome_patch"],
            heightmap_patch=batch["heightmap_patch"],
            y_index=batch["y_index"],
            lod=batch["lod"],
        )
        air_mask_logits = outputs["air_mask_logits"]
        block_type_logits = outputs["block_type_logits"]

        # Cache air mask logits for scheduled sampling (detach to avoid grad retention)
        self._last_air_mask_pred = air_mask_logits.detach()

        # Compute loss
        loss = voxel_loss_fn(
            air_mask_logits=air_mask_logits,
            block_type_logits=block_type_logits,
            target_mask=batch["target_mask"],
            target_types=batch["target_types"],
            mask_weight=self.mask_weight,
            type_weight=self.type_weight,
        )  # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        # Log to TensorBoard
        if self.tb_logger and self.global_step % 10 == 0:  # Log every 10 steps
            self.tb_logger.log_metrics(
                {"train/loss": loss.item()}, step=self.global_step, prefix=""
            )

            # Log visualizations occasionally
            if self.global_step % 100 == 0:
                tensorboard_config = self.config.get("training", {}).get("tensorboard", {})
                if tensorboard_config.get("log_visualizations", False):
                    try:
                        self.tb_logger.log_voxel_batch(
                            inputs=batch["parent_voxel"],
                            predictions=air_mask_logits,
                            targets=batch["target_mask"],
                            step=self.global_step,
                            max_samples=tensorboard_config.get("visualization_samples", 2),
                        )
                    except Exception as e:
                        logging.warning(f"Failed to log visualizations: {e}")

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

    def validate_one_epoch(self, dataloader=None) -> Dict[str, float]:
        """Validate for one epoch."""
        if dataloader is None:
            # Dummy validation for testing
            dummy_batch = self._create_dummy_batch()
            with torch.no_grad():
                self.model.eval()
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in dummy_batch.items()
                }

                outputs = self.model(
                    parent_voxel=batch["parent_voxel"],
                    biome_patch=batch["biome_patch"],
                    heightmap_patch=batch["heightmap_patch"],
                    y_index=batch["y_index"],
                    lod=batch["lod"],
                )

                val_loss = voxel_loss_fn(
                    air_mask_logits=outputs["air_mask_logits"],
                    block_type_logits=outputs["block_type_logits"],
                    target_mask=batch["target_mask"],
                    target_types=batch["target_types"],
                    mask_weight=self.mask_weight,
                    type_weight=self.type_weight,
                )

                return {"loss": val_loss.item()}

        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                self.model.eval()
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    parent_voxel=batch["parent_voxel"],
                    biome_patch=batch["biome_patch"],
                    heightmap_patch=batch["heightmap_patch"],
                    y_index=batch["y_index"],
                    lod=batch["lod"],
                )

                val_loss = voxel_loss_fn(
                    air_mask_logits=outputs["air_mask_logits"],
                    block_type_logits=outputs["block_type_logits"],
                    target_mask=batch["target_mask"],
                    target_types=batch["target_types"],
                    mask_weight=self.mask_weight,
                    type_weight=self.type_weight,
                )

                total_loss += val_loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time

        return {
            "loss": avg_loss,
            "epoch_time": epoch_time,
        }

    def analyze_confusion_matrix(self, dataloader, epoch: int) -> Dict[str, Any]:
        """
        Run comprehensive confusion matrix analysis on validation data.

        Args:
            dataloader: Validation dataloader
            epoch: Current epoch number

        Returns:
            Dictionary with confusion matrix metrics
        """
        if dataloader is None:
            return {"overall_accuracy": 0.0, "goal_achieved": False}

        # Reset accumulated confusion matrix
        self.confusion_analyzer.reset()

        logger = logging.getLogger(__name__)
        logger.info(f"Running confusion matrix analysis for epoch {epoch}...")

        with torch.no_grad():
            self.model.eval()

            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
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

                # Update confusion matrix (only for solid blocks)
                self.confusion_analyzer.update(
                    predictions=outputs["block_type_logits"],
                    targets=batch["target_types"],
                    mask=batch["target_mask"],  # Only analyze solid blocks
                )

                if batch_idx % 50 == 0:
                    logger.info(f"Processed {batch_idx + 1} validation batches...")

        # Generate analysis
        overall_accuracy = self.confusion_analyzer.compute_overall_accuracy()
        goal_achieved, _ = self.confusion_analyzer.is_99_percent_achieved()

        # Save detailed analysis every few epochs
        if epoch % 5 == 0 or goal_achieved:
            self.confusion_analyzer.save_analysis(epoch)

        # Log summary to TensorBoard
        if self.tb_logger:
            per_class_acc = self.confusion_analyzer.compute_per_class_accuracy()
            valid_accuracies = [acc for acc in per_class_acc.values() if not np.isnan(acc)]

            self.tb_logger.log_metrics(
                {
                    "val/overall_accuracy": float(overall_accuracy),
                    "val/mean_class_accuracy": float(
                        np.mean(valid_accuracies) if valid_accuracies else 0.0
                    ),
                    "val/min_class_accuracy": float(
                        np.min(valid_accuracies) if valid_accuracies else 0.0
                    ),
                    "val/blocks_above_99pct": float(
                        sum(1 for acc in valid_accuracies if acc >= 0.99)
                    ),
                    "val/goal_progress": float(overall_accuracy / 0.99),
                },
                step=self.global_step,
            )

        logger.info(
            f"Confusion analysis complete - Overall accuracy: {overall_accuracy:.4f} "
            f"({overall_accuracy*100:.2f}%)"
        )
        if goal_achieved:
            logger.info("🎉 99% accuracy goal ACHIEVED! 🎉")
        else:
            gap = 99.0 - overall_accuracy * 100
            logger.info(f"Gap to 99% goal: {gap:.2f} percentage points")

        return {
            "overall_accuracy": overall_accuracy,
            "goal_achieved": goal_achieved,
            "mean_class_accuracy": np.mean(valid_accuracies) if valid_accuracies else 0.0,
            "blocks_evaluated": len(valid_accuracies),
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
            "provenance": self._collect_provenance(),
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
            # river removed
            "y_index": torch.randint(0, 24, (batch_size,), device=self.device),
            "lod": torch.randint(0, 5, (batch_size,), device=self.device),
            "target_mask": torch.randint(
                0, 2, (batch_size, 1, 16, 16, 16), device=self.device
            ).float(),
            "target_types": torch.randint(
                0, 10, (batch_size, 16, 16, 16), device=self.device
            ).long(),
        }

    # ------------------------- Provenance Utilities ------------------------- #
    def _collect_provenance(self) -> Dict[str, Any]:
        """Collect lightweight provenance info for checkpoints."""
        prov: Dict[str, Any] = {}
        # Git commit SHA
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            prov["git_commit"] = sha
        except Exception:  # pragma: no cover - environment may not have git
            pass
        # Model parameter count
        prov["param_count"] = sum(p.numel() for p in self.model.parameters())
        # Multi-LOD settings
        if self.multi_lod_enabled:
            prov["multi_lod_factors"] = self.multi_lod_factors
        if self.sched_enabled:
            prov["scheduled_sampling"] = {
                "start_prob": self.sched_start,
                "end_prob": self.sched_end,
                "warmup_epochs": self.sched_warmup_epochs,
            }
        return prov
        return prov
        return prov
