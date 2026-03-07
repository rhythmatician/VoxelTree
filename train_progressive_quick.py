#!/usr/bin/env python3
"""
Quick training script for progressive LOD models.
Trains just a few epochs to create models for runtime performance evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from scripts.mipper import build_opacity_table as _build_opacity_table  # noqa: E402
from scripts.mipper import mip_volume_torch as _mip_volume_torch  # noqa: E402
from train.multi_lod_dataset import MultiLODDataset  # noqa: E402
from train.multi_lod_dataset import collate_multi_lod_batch  # noqa: E402
from train.progressive_lod_models import ProgressiveLODModel  # noqa: E402
from train.progressive_lod_models import ProgressiveLODModel0_Initial  # noqa: E402
from train.unet3d import SimpleFlexibleConfig  # noqa: E402


class ProgressiveLODLoss(nn.Module):
    """Loss for progressive LOD models."""

    def __init__(
        self,
        air_loss_weight: float = 0.25,
        freq_top_k: int = 100,
        rare_weight: float = 0.2,
    ):
        super().__init__()
        self.air_loss_weight = air_loss_weight
        self.freq_top_k = freq_top_k
        self.rare_weight = rare_weight
        # We'll compute CE per-voxel (reduction=none) to apply frequency weights
        self.block_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.air_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for progressive LOD predictions."""

        # Handle different output key names
        if "air_mask_logits" in predictions:
            air_mask_logits = predictions["air_mask_logits"]
            block_type_logits = predictions["block_type_logits"]
        else:
            air_mask_logits = predictions["air_mask"]
            block_type_logits = predictions["block_logits"]

        target_blocks = targets["target_blocks"]
        target_occupancy = targets["target_occupancy"]

        # Block type loss
        B, C, H, W, D = block_type_logits.shape
        block_logits_flat = block_type_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_blocks_flat = target_blocks.reshape(-1)
        # Frequency-aware weighting: emphasize frequent blocks within the batch
        with torch.no_grad():
            # Count frequencies for this batch
            unique, counts = torch.unique(target_blocks_flat, return_counts=True)
            # Get top-k frequent labels
            k = min(self.freq_top_k, unique.numel())
            topk_idx = torch.topk(counts, k).indices
            frequent_labels = unique[topk_idx]
            # Build a mask: 1 for frequent, 0 for rare
            is_frequent = torch.isin(target_blocks_flat, frequent_labels)
            # Build weights vector
            ones = torch.ones_like(target_blocks_flat, dtype=torch.float32)
            rares = torch.full_like(target_blocks_flat, self.rare_weight, dtype=torch.float32)
            weights = torch.where(is_frequent, ones, rares)

        per_voxel_ce = self.block_loss_fn(block_logits_flat, target_blocks_flat)
        block_loss = (per_voxel_ce * weights).mean()

        # Air mask loss
        target_air = (target_occupancy == 0).float()
        if target_air.dim() == 4:
            target_air = target_air.unsqueeze(1)
        air_loss = self.air_loss_fn(air_mask_logits, target_air)

        # Combined loss
        total_loss = block_loss + self.air_loss_weight * air_loss

        return {
            "total_loss": total_loss,
            "block_loss": block_loss,
            "air_loss": air_loss,
        }


def _downsample_targets(blocks: torch.Tensor, occ: torch.Tensor, out_size: int):
    """Downsample 16³ targets to ``out_size``³ using Voxy Mipper (blocks) and OR (occ)."""
    if blocks.dim() == 4:
        B, H, W, D = blocks.shape
    else:
        raise ValueError("Expected blocks shape [B,16,16,16]")

    if out_size == 16:
        return blocks, occ

    factor = 16 // out_size
    if 16 % out_size != 0:
        raise ValueError(f"out_size {out_size} must divide 16")
    if (factor & (factor - 1)) != 0:
        raise ValueError(f"factor {factor} must be a power of 2")

    # Build opacity table lazily
    tbl = torch.from_numpy(_build_opacity_table(n_blocks=4096)).long().to(blocks.device)

    # Mipper: opacity-biased block selection
    coarse_blks, coarse_occ = _mip_volume_torch(blocks.long(), factor, tbl)
    return coarse_blks, coarse_occ


def _build_parent_logits_from_targets(blocks16: torch.Tensor, parent_size: int, num_classes: int):
    """Build parent block logits [B,C,P,P,P] from 16^3 integer block labels via window mode.

    We downsample labels to parent_size using window mode (like _downsample_targets) and then
    create one-hot logits scaled to a positive margin for the true class.
    """
    if parent_size is None:
        return None
    if 16 % parent_size != 0:
        raise ValueError("parent_size must divide 16")
    # Reuse downsample to get parent labels (ignore occupancy result)
    parent_labels, _ = _downsample_targets(blocks16, torch.ones_like(blocks16), parent_size)
    # One-hot -> logits
    B, P, _, _ = parent_labels.shape
    one_hot = torch.nn.functional.one_hot(
        parent_labels.long(), num_classes=num_classes
    )  # [B,P,P,P,C]
    one_hot = one_hot.permute(0, 4, 1, 2, 3).contiguous().float()  # [B,C,P,P,P]
    logits = (one_hot * 6.0) + ((1.0 - one_hot) * -6.0)  # margin logits
    return logits


def compute_metrics(
    predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    with torch.no_grad():
        # Handle different output key names
        if "air_mask_logits" in predictions:
            air_mask_logits = predictions["air_mask_logits"]
            block_type_logits = predictions["block_type_logits"]
        else:
            air_mask_logits = predictions["air_mask"]
            block_type_logits = predictions["block_logits"]

        target_blocks = targets["target_blocks"]
        target_occupancy = targets["target_occupancy"]

        # Air mask accuracy
        air_pred = (torch.sigmoid(air_mask_logits) > 0.5).float()
        target_air = (target_occupancy == 0).float().unsqueeze(1)
        air_acc = (air_pred == target_air).float().mean().item()

        # Block type accuracy (only for non-air voxels)
        block_pred = torch.argmax(block_type_logits, dim=1)
        non_air_mask = target_occupancy > 0
        if non_air_mask.sum() > 0:
            block_acc = (
                (block_pred[non_air_mask] == target_blocks[non_air_mask]).float().mean().item()
            )
        else:
            block_acc = 1.0

        # Frequent-set accuracy (top-k in this batch)
        flat_targets = target_blocks.view(-1)
        unique, counts = torch.unique(flat_targets, return_counts=True)
        k = min(100, unique.numel())
        topk_idx = torch.topk(counts, k).indices
        frequent_labels = unique[topk_idx]
        freq_mask = torch.isin(target_blocks, frequent_labels) & non_air_mask
        if freq_mask.sum() > 0:
            freq_acc = (block_pred[freq_mask] == target_blocks[freq_mask]).float().mean().item()
        else:
            freq_acc = block_acc

        # Overall accuracy (weighted equally)
        overall_acc = 0.5 * air_acc + 0.5 * freq_acc

        return {
            "air_accuracy": air_acc,
            "block_accuracy": block_acc,
            "overall_accuracy": overall_acc,
            "frequent_block_accuracy": freq_acc,
        }


def train_model(model, train_loader, val_loader, config, model_name, device="cuda"):
    """Train a single model for a few epochs."""
    print(f"\n=== Training {model_name} ===")

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_fn = ProgressiveLODLoss(air_loss_weight=config["air_loss_weight"])

    # Quick training - just a few epochs
    epochs = config["quick_epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config["max_batches_per_epoch"]:  # Limit batches for quick training
                break

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()

            try:
                # Create inputs for progressive model
                if "0_Initial" in model_name:
                    # Model 0 - no parent input
                    predictions = model(
                        batch["height_planes"],
                        batch["biome_quart"],
                        batch["router6"],
                        batch["chunk_pos"],
                        batch["lod"],
                    )
                else:
                    # Models 1-4 - with parent input
                    # Infer expected parent size from model name
                    if "1_LOD4to3" in model_name:
                        parent_size = 1
                    elif "2_LOD3to2" in model_name:
                        parent_size = 2
                    elif "3_LOD2to1" in model_name:
                        parent_size = 4
                    elif "4_LOD1to0" in model_name:
                        parent_size = 8
                    else:
                        parent_size = None

                    parent_input = batch.get("parent_logits")
                    # If parent logits missing or wrong shape, rebuild from targets
                    if parent_size is not None:
                        rebuild_parent = True
                        if isinstance(parent_input, torch.Tensor) and parent_input.dim() == 5:
                            if parent_input.shape[-1] == parent_size:
                                rebuild_parent = False
                        if rebuild_parent:
                            num_classes = model.config.block_vocab_size
                            parent_input = _build_parent_logits_from_targets(
                                batch["target_blocks"], parent_size, num_classes
                            ).to(device)

                    predictions = model(
                        batch["height_planes"],
                        batch["biome_quart"],
                        batch["router6"],
                        batch["chunk_pos"],
                        batch["lod"],
                        parent_input,  # parent input
                    )

                # Downsample targets to match model output size
                out_logits = (
                    predictions.get("block_type_logits")
                    if "0_Initial" in model_name
                    else predictions["block_logits"]
                )
                out_size = out_logits.shape[-1]
                blocks_ds, occ_ds = _downsample_targets(
                    batch["target_blocks"], batch["target_occupancy"], out_size
                )

                targets = {
                    "target_blocks": blocks_ds.to(device),
                    "target_occupancy": occ_ds.to(device),
                }

                losses = loss_fn(predictions, targets)
                loss = losses["total_loss"]

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.4f}")

        # Quick eval on a couple of val batches
        model.eval()
        with torch.no_grad():
            val_iters = 0
            agg = {"air_accuracy": 0.0, "block_accuracy": 0.0, "overall_accuracy": 0.0}
            for vb in val_loader:
                if val_iters >= 3:
                    break
                for k in vb:
                    if isinstance(vb[k], torch.Tensor):
                        vb[k] = vb[k].to(device)

                if "0_Initial" in model_name:
                    preds = model(
                        vb["height_planes"],
                        vb["biome_quart"],
                        vb["router6"],
                        vb["chunk_pos"],
                        vb["lod"],
                    )
                    out_logits = preds["block_type_logits"]
                else:
                    # infer parent size for eval
                    if "1_LOD4to3" in model_name:
                        ps = 1
                    elif "2_LOD3to2" in model_name:
                        ps = 2
                    elif "3_LOD2to1" in model_name:
                        ps = 4
                    else:
                        ps = 8
                    num_classes = model.config.block_vocab_size
                    parent_in = _build_parent_logits_from_targets(
                        vb["target_blocks"], ps, num_classes
                    ).to(device)
                    preds = model(
                        vb["height_planes"],
                        vb["biome_quart"],
                        vb["router6"],
                        vb["chunk_pos"],
                        vb["lod"],
                        parent_in,
                    )
                    out_logits = preds["block_logits"]

                s = out_logits.shape[-1]
                tb, to = _downsample_targets(vb["target_blocks"], vb["target_occupancy"], s)
                m = compute_metrics(
                    preds,
                    {"target_blocks": tb.to(device), "target_occupancy": to.to(device)},
                )
                for k in agg:
                    agg[k] += m[k]
                val_iters += 1

            if val_iters > 0:
                print(
                    f"  Val metrics: air_acc={agg['air_accuracy']/val_iters:.3f}, "
                    f"block_acc={agg['block_accuracy']/val_iters:.3f}, "
                    f"overall={agg['overall_accuracy']/val_iters:.3f}"
                )

    # Save model
    save_path = Path("models") / f"quick_{model_name.lower().replace(' ', '_')}.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "model_name": model_name,
        },
        save_path,
    )

    print(f"  Model saved to {save_path}")
    return model


def create_quick_data_inputs(batch_size=1):
    """Create dummy inputs for testing when no real data is available."""
    return {
        "height_planes": torch.randn(batch_size, 5, 1, 16, 16),
        "biome_quart": torch.randn(batch_size, 6, 4, 4, 4),
        "router6": torch.randn(batch_size, 6, 1, 16, 16),
        "chunk_pos": torch.randn(batch_size, 2),
        "lod": torch.randint(0, 4, (batch_size, 1)),
        # parent_voxel omitted in quick mode; we synthesize per model as needed
        "target_blocks": torch.randint(0, 1104, (batch_size, 16, 16, 16)),
        "target_occupancy": torch.randint(0, 2, (batch_size, 16, 16, 16)),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick training for progressive LOD models")
    parser.add_argument("--config", default="config_multi_lod.yaml", help="Config file")
    parser.add_argument("--data-dir", default="data/pairs", help="Training data directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--quick", action="store_true", help="Use dummy data for quick testing")
    args = parser.parse_args()

    # Quick training config
    quick_config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "air_loss_weight": 0.25,
        "quick_epochs": 3,  # Just a few epochs
        "max_batches_per_epoch": 20,  # Limit batches for speed
        "batch_size": 2,  # Small batch size for quick training
    }

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model config
    model_config = SimpleFlexibleConfig()

    # Create dummy data loader
    class DummyDataLoader:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def __iter__(self):
            for _ in range(20):  # 20 dummy batches
                yield create_quick_data_inputs(self.batch_size)

    if args.quick:
        print("Using dummy data for quick testing...")

        train_loader = DummyDataLoader(quick_config["batch_size"])
        val_loader = DummyDataLoader(quick_config["batch_size"])
    else:
        # Try to load real data
        try:
            print(f"Loading training data from {args.data_dir}...")
            dataset = MultiLODDataset(
                data_dir=Path(args.data_dir),
                split="train",
            )
            train_loader = DataLoader(
                dataset,
                batch_size=quick_config["batch_size"],
                shuffle=True,
                collate_fn=collate_multi_lod_batch,
                num_workers=0,  # Avoid multiprocessing issues
            )
            val_loader = train_loader  # Use same for validation in quick mode
            print(f"Loaded {len(dataset)} training samples")
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to dummy data...")
            train_loader = DummyDataLoader(quick_config["batch_size"])
            val_loader = DummyDataLoader(quick_config["batch_size"])

    # Create and train models
    models_to_train = [
        ("Model_0_Initial", ProgressiveLODModel0_Initial(model_config, output_size=1)),
        ("Model_1_LOD4to3", ProgressiveLODModel(model_config, output_size=2)),
        ("Model_2_LOD3to2", ProgressiveLODModel(model_config, output_size=4)),
        ("Model_3_LOD2to1", ProgressiveLODModel(model_config, output_size=8)),
        ("Model_4_LOD1to0", ProgressiveLODModel(model_config, output_size=16)),
    ]

    print(f"\n{'='*60}")
    print("QUICK PROGRESSIVE LOD TRAINING")
    print(f"{'='*60}")

    trained_models = {}
    for model_name, model in models_to_train:
        try:
            trained_model = train_model(
                model, train_loader, val_loader, quick_config, model_name, device
            )
            trained_models[model_name] = trained_model
            print(f"[OK] {model_name} training completed")
        except Exception as e:
            print(f"[FAIL] {model_name} training failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully trained: {len(trained_models)}/{len(models_to_train)} models")
    for model_name in trained_models:
        print(f"[OK] {model_name}")

    print("\nModels saved in: models/")
    print("Ready for runtime performance evaluation!")


if __name__ == "__main__":
    main()
