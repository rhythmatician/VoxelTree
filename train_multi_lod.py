"""
Multi-LOD Training Script

This script trains a single flexible model that can handle all LOD transitions:
- LOD4→LOD3: 1³ → 2³
- LOD3→LOD2: 2³ → 4³
- LOD2→LOD1: 4³ → 8³
- LOD1→LOD0: 8³ → 16³
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from train.multi_lod_dataset import MultiLODDataset, collate_multi_lod_batch
from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D

# Default Voxy vocabulary path
DEFAULT_VOCAB_PATH = Path("config/voxy_vocab.json")


class MultiLODLoss(nn.Module):
    """
    Combined loss for air mask and block type prediction across all LOD levels.
    Optionally includes a surface-consistency term that penalises the model
    when its predicted top-surface height deviates from the heightmap anchor.

    Args:
        class_weights: Optional float32 tensor of shape [block_vocab_size].
            Computed by scripts/compute_class_weights.py (median-frequency
            balancing). Weight 0 means ignore that class (used for air=0 and
            unseen blocks). Passed to F.cross_entropy at forward time so it
            moves to the correct device automatically.
    """

    def __init__(
        self,
        air_loss_weight: float = 0.25,
        surface_consistency_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.air_loss_weight = air_loss_weight
        self.surface_consistency_weight = surface_consistency_weight
        # Register as buffer so .to(device) / .cuda() move it automatically
        self.register_buffer("class_weights", class_weights)  # None is fine
        # pos_weight > 1 up-weights the minority class (solid ~25%)
        # This compensates for the 75/25 air/solid imbalance
        self.air_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-LOD loss.

        Args:
            predictions: Dict with 'air_mask_logits' and 'block_type_logits'.
            targets: Dict with 'target_blocks', 'target_occupancy', and
                     optionally 'height_planes' (B, 5, 16, 16).
        Returns:
            Dict with individual and total losses.
        """
        air_mask_logits = predictions["air_mask_logits"]
        block_type_logits = predictions["block_type_logits"]

        target_blocks = targets["target_blocks"]
        target_occupancy = targets["target_occupancy"]

        # Block type loss (cross-entropy)
        # Reshape for cross-entropy: (B*H*W*D, C) and (B*H*W*D,)
        B, C, H, W, D = block_type_logits.shape
        block_logits_flat = block_type_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_blocks_flat = target_blocks.reshape(-1)

        block_loss = torch.nn.functional.cross_entropy(
            block_logits_flat,
            target_blocks_flat,
            weight=self.class_weights if isinstance(self.class_weights, torch.Tensor) else None,
            ignore_index=0,
        )

        # Air mask loss (binary cross-entropy)
        # Polarity: positive logit = SOLID, matching Java runtime convention
        target_solid = (target_occupancy > 0).float()
        if target_solid.dim() == 4:  # (B, H, W, D) -> add channel dim
            target_solid = target_solid.unsqueeze(1)  # (B, 1, H, W, D)

        air_loss = self.air_loss_fn(air_mask_logits, target_solid)

        # Surface consistency loss (optional)
        # Penalise mismatch between the predicted top-occupancy surface and
        # the height_planes[0] (normalised WORLD_SURFACE_WG) anchor.
        surface_loss = block_type_logits.new_zeros(1).squeeze()
        if self.surface_consistency_weight > 0 and "height_planes" in targets:
            height_planes = targets["height_planes"]  # (B, 5, 16, 16)
            surface_anchor = height_planes[:, 0, :, :]  # (B, 16, 16) normalised 0..1

            # Predicted solid prob: sigmoid of air_mask_logits, shape (B,1,D,H,W)
            # (positive = solid after polarity fix)
            solid_prob_5d = torch.sigmoid(air_mask_logits)  # (B,1,D,H,W)
            # Top surface = first solid slab from above (dim=2, y-axis)
            # Approximate as the y-coordinate of maximum solid probability, normalised
            solid_prob = solid_prob_5d.squeeze(1)  # (B, D, H, W)
            y_weights = torch.linspace(0, 1, D, device=solid_prob.device)
            # Shape: (B, H, W)  — weighted average y of solid voxels per column
            predicted_surface = (solid_prob * y_weights[None, :, None, None]).sum(dim=1)  # (B,H,W)
            predicted_surface = predicted_surface / (solid_prob.sum(dim=1).clamp(min=1e-6))
            surface_loss = torch.nn.functional.l1_loss(predicted_surface, surface_anchor)

        # Combined loss
        total_loss = (
            block_loss
            + self.air_loss_weight * air_loss
            + self.surface_consistency_weight * surface_loss
        )

        return {
            "total_loss": total_loss,
            "block_loss": block_loss,
            "air_loss": air_loss,
            "surface_loss": surface_loss,
        }


def compute_metrics(
    predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute evaluation metrics for multi-LOD predictions."""

    with torch.no_grad():
        air_mask_logits = predictions["air_mask_logits"]
        block_type_logits = predictions["block_type_logits"]

        target_blocks = targets["target_blocks"]
        target_occupancy = targets["target_occupancy"]

        # Air mask accuracy (positive = solid, matching Java convention)
        air_pred = (torch.sigmoid(air_mask_logits) > 0.5).float()
        target_solid = (target_occupancy > 0).float().unsqueeze(1)
        air_acc = (air_pred == target_solid).float().mean().item()

        # Block type accuracy (only on solid voxels)
        block_pred = block_type_logits.argmax(dim=1)
        solid_mask = target_occupancy > 0

        if solid_mask.sum() > 0:
            block_acc = (block_pred[solid_mask] == target_blocks[solid_mask]).float().mean().item()
        else:
            block_acc = 1.0  # All air, trivially correct

        # Overall accuracy
        # Air voxels: correct if predicted as air (solid logit < 0.5)
        # Solid voxels: correct if block type matches
        air_mask = target_occupancy == 0
        solid_mask = target_occupancy > 0

        air_correct = (air_pred.squeeze(1)[air_mask] < 0.5).sum()
        solid_correct = (block_pred[solid_mask] == target_blocks[solid_mask]).sum()
        total_voxels = target_occupancy.numel()

        overall_acc = (air_correct + solid_correct).float() / total_voxels

        return {
            "air_accuracy": air_acc,
            "block_accuracy": block_acc,
            "overall_accuracy": overall_acc.item(),
        }


def train_epoch(
    model: SimpleFlexibleUNet3D,
    dataloader: DataLoader,
    loss_fn: MultiLODLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Train for one epoch."""

    model.train()

    total_loss = 0.0
    total_block_loss = 0.0
    total_air_loss = 0.0
    total_air_acc = 0.0
    total_block_acc = 0.0
    total_overall_acc = 0.0
    num_batches = 0

    lod_counts: dict[str, int] = {}

    total_batches = len(dataloader)

    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        total=total_batches,
        unit="batches",
        desc="Training",
        leave=False,
        dynamic_ncols=True,
    ):
        # Move to device — include anchor tensors when present
        inputs = {
            "parent_voxel": batch["parent_voxel"].to(device),
            "biome_patch": batch["biome_idx"].to(device),  # integer indices
            "heightmap_patch": batch["heightmap_patch"].to(device),
            "y_index": batch["y_index"].to(device),
            "lod": batch["lod"].to(device),
            "height_planes": batch["height_planes"].to(device),
            "router6": batch["router6"].to(device),
        }

        targets = {
            "target_blocks": batch["target_types"].to(device),
            "target_occupancy": batch["target_mask"].to(device),
            "height_planes": batch["height_planes"].to(device),
        }

        # Track LOD distribution
        lod_transition: str = batch["lod_transition"]
        lod_counts[lod_transition] = lod_counts.get(lod_transition, 0) + 1

        # Forward pass
        optimizer.zero_grad()
        predictions = model(**inputs)

        # Compute loss
        losses = loss_fn(predictions, targets)
        loss = losses["total_loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        metrics = compute_metrics(predictions, targets)

        # Accumulate stats
        total_loss += loss.item()
        total_block_loss += losses["block_loss"].item()
        total_air_loss += losses["air_loss"].item()
        total_air_acc += metrics["air_accuracy"]
        total_block_acc += metrics["block_accuracy"]
        total_overall_acc += metrics["overall_accuracy"]
        num_batches += 1

    # Average metrics
    return {
        "loss": total_loss / num_batches,
        "block_loss": total_block_loss / num_batches,
        "air_loss": total_air_loss / num_batches,
        "air_accuracy": total_air_acc / num_batches,
        "block_accuracy": total_block_acc / num_batches,
        "overall_accuracy": total_overall_acc / num_batches,
        "lod_distribution": lod_counts,
    }


def validate_epoch(
    model: SimpleFlexibleUNet3D,
    dataloader: DataLoader,
    loss_fn: MultiLODLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate for one epoch."""

    model.eval()

    total_loss = 0.0
    total_block_loss = 0.0
    total_air_loss = 0.0
    total_air_acc = 0.0
    total_block_acc = 0.0
    total_overall_acc = 0.0
    num_batches = 0

    lod_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            # Move to device — include anchor tensors when present
            inputs = {
                "parent_voxel": batch["parent_voxel"].to(device),
                "biome_patch": batch["biome_idx"].to(device),  # integer indices
                "heightmap_patch": batch["heightmap_patch"].to(device),
                "y_index": batch["y_index"].to(device),
                "lod": batch["lod"].to(device),
                "height_planes": batch["height_planes"].to(device),
                "router6": batch["router6"].to(device),
            }

            targets = {
                "target_blocks": batch["target_types"].to(device),
                "target_occupancy": batch["target_mask"].to(device),
                "height_planes": batch["height_planes"].to(device),
            }

            lod_transition = batch["lod_transition"]

            # Forward pass
            predictions = model(**inputs)

            # Compute loss
            losses = loss_fn(predictions, targets)

            # Compute metrics
            metrics = compute_metrics(predictions, targets)

            # Accumulate overall stats
            total_loss += losses["total_loss"].item()
            total_block_loss += losses["block_loss"].item()
            total_air_loss += losses["air_loss"].item()
            total_air_acc += metrics["air_accuracy"]
            total_block_acc += metrics["block_accuracy"]
            total_overall_acc += metrics["overall_accuracy"]
            num_batches += 1

            # Accumulate per-LOD stats
            if lod_transition not in lod_metrics:
                lod_metrics[lod_transition] = {
                    "count": 0,
                    "loss": 0.0,
                    "air_accuracy": 0.0,
                    "block_accuracy": 0.0,
                    "overall_accuracy": 0.0,
                }

            lod_stats = lod_metrics[lod_transition]
            lod_stats["count"] += 1
            lod_stats["loss"] += losses["total_loss"].item()
            lod_stats["air_accuracy"] += metrics["air_accuracy"]
            lod_stats["block_accuracy"] += metrics["block_accuracy"]
            lod_stats["overall_accuracy"] += metrics["overall_accuracy"]

    # Average overall metrics
    results = {
        "loss": total_loss / num_batches,
        "block_loss": total_block_loss / num_batches,
        "air_loss": total_air_loss / num_batches,
        "air_accuracy": total_air_acc / num_batches,
        "block_accuracy": total_block_acc / num_batches,
        "overall_accuracy": total_overall_acc / num_batches,
    }

    # Average per-LOD metrics
    for lod_transition, stats in lod_metrics.items():
        if stats["count"] > 0:
            results[f"{lod_transition}_loss"] = stats["loss"] / stats["count"]
            results[f"{lod_transition}_air_acc"] = stats["air_accuracy"] / stats["count"]
            results[f"{lod_transition}_block_acc"] = stats["block_accuracy"] / stats["count"]
            results[f"{lod_transition}_overall_acc"] = stats["overall_accuracy"] / stats["count"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Multi-LOD Flexible Model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to training data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./multi_lod_training", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels")
    parser.add_argument("--air-loss-weight", type=float, default=0.25, help="Weight for air loss")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--validate-every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument(
        "--surface-loss-weight",
        type=float,
        default=0.0,
        help="Weight for surface-consistency loss (0=disabled, 0.1 recommended with anchor data)",
    )
    parser.add_argument(
        "--height-channels",
        type=int,
        default=5,
        help="Height-plane channels for anchor conditioning",
    )
    parser.add_argument(
        "--router6-channels",
        type=int,
        default=6,
        help="Router6 channels for anchor conditioning",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=main process, safest on Windows CPU)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Path to Voxy vocabulary JSON (default: config/voxy_vocab.json)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint .pt to resume training from",
    )
    parser.add_argument(
        "--no-pair-cache",
        action="store_true",
        help="Force regeneration of training pairs (ignore cached pairs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit training to N randomly-sampled pairs per run (val = N//10). "
            "Use for quick smoke-tests, e.g. --max-samples 2000 finishes in ~2 min "
            "vs ~60 min for a full epoch."
        ),
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        metavar="PATH_OR_AUTO",
        help=(
            "Path to class_weights.npz (from scripts/compute_class_weights.py), "
            "or 'auto' to compute from the data dir on the fly. "
            "Applies median-frequency balancing to the block-type loss so rare "
            "blocks (granite, dirt, wood…) are trained as aggressively as stone."
        ),
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load Voxy vocabulary for block_vocab_size
    vocab_path: Path = args.vocab
    if vocab_path.exists():
        with open(vocab_path) as f:
            voxy_vocab = json.load(f)
        block_vocab_size = len(voxy_vocab)
        print(f"Voxy vocabulary: {block_vocab_size} block types from {vocab_path}")
    else:
        block_vocab_size = 1102  # fallback if no vocab file
        print(f"Warning: vocab file {vocab_path} not found, using default size {block_vocab_size}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model (include anchor-channel dimensions)
    config = SimpleFlexibleConfig(
        base_channels=args.base_channels,
        biome_vocab_size=256,
        block_vocab_size=block_vocab_size,
        height_channels=args.height_channels,
        router6_channels=args.router6_channels,
    )

    model = SimpleFlexibleUNet3D(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = MultiLODDataset(
        data_dir=args.data_dir,
        split="train",
        lod_sampling_weights={
            # Keys must match lod_transition names: "lod{N}to{N-1}"
            "lod4to3": 0.2,
            "lod3to2": 0.25,
            "lod2to1": 0.25,
            "lod1to0": 0.3,  # Emphasize finest level
        },
        use_pair_cache=not args.no_pair_cache,
    )

    val_dataset = MultiLODDataset(
        data_dir=args.data_dir,
        split="val",
        use_pair_cache=not args.no_pair_cache,
    )

    # ── Subset for quick test runs (--max-samples) ────────────────────────
    if args.max_samples is not None:
        n_train = min(args.max_samples, len(train_dataset))
        n_val = min(max(1, args.max_samples // 10), len(val_dataset))
        train_idx = random.sample(range(len(train_dataset)), n_train)
        val_idx = random.sample(range(len(val_dataset)), n_val)
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)
        print(
            f"--max-samples: using {n_train} train + {n_val} val samples "
            f"({n_train / 80000 * 100:.1f}% of a full epoch)"
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multi_lod_batch,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multi_lod_batch,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ── Class weights for block-type loss ──────────────────────────────────
    class_weights_tensor: Optional[torch.Tensor] = None
    if args.class_weights is not None:
        from scripts.compute_class_weights import compute_weights, load_weights  # noqa: E402

        cw_arg = args.class_weights.strip()
        if cw_arg.lower() == "auto":
            cw_path = Path(args.data_dir) / "class_weights.npz"
            if cw_path.exists():
                print(f"Loading cached class weights: {cw_path}")
                cw_arr = load_weights(cw_path)
            else:
                print("Computing class weights from training pairs (auto)…")
                cw_arr = compute_weights(
                    data_dir=Path(args.data_dir),
                    vocab_size=block_vocab_size,
                    verbose=True,
                )
                np.savez_compressed(cw_path, class_weights=cw_arr)
                print(f"  Cached → {cw_path}")
        else:
            cw_path = Path(cw_arg)
            if not cw_path.exists():
                raise FileNotFoundError(f"--class-weights file not found: {cw_path}")
            print(f"Loading class weights: {cw_path}")
            cw_arr = load_weights(cw_path)

        class_weights_tensor = torch.tensor(cw_arr, dtype=torch.float32)
        nonzero = int((class_weights_tensor > 0).sum())
        print(f"  Class weights loaded: {nonzero} / {len(cw_arr)} non-zero classes")
        print(
            f"  Weight range (non-zero): "
            f"{class_weights_tensor[class_weights_tensor > 0].min():.3f} … "
            f"{class_weights_tensor.max():.3f}"
        )

    # Create loss function and optimizer
    loss_fn = MultiLODLoss(
        air_loss_weight=args.air_loss_weight,
        surface_consistency_weight=args.surface_loss_weight,
        class_weights=class_weights_tensor,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        # Advance scheduler to match resumed epoch
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(
            f"Resumed from {args.resume} "
            f"(epoch {start_epoch - 1}, "
            f"best_val_loss={best_val_loss:.4f})"
        )

    start_time = time.time()
    print("Starting training...")

    for epoch in tqdm(
        range(start_epoch, args.epochs + 1),
        unit="epochs",
        dynamic_ncols=True,
        desc="Training  Epochs",
    ):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Update learning rate
        scheduler.step()

        # Print training stats
        print(f"Epoch {epoch}/{args.epochs}")
        print(
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(Block: {train_metrics['block_loss']:.4f}, Air: {train_metrics['air_loss']:.4f})"
        )
        print(
            f"  Train Acc: Overall {train_metrics['overall_accuracy']:.3f}, "
            f"Air {train_metrics['air_accuracy']:.3f}, Block {train_metrics['block_accuracy']:.3f}"
        )
        print(f"  LOD Distribution: {train_metrics['lod_distribution']}")

        # Validate
        if epoch % args.validate_every == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)

            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(
                f"  Val Acc: Overall {val_metrics['overall_accuracy']:.3f}, "
                f"Air {val_metrics['air_accuracy']:.3f}, Block {val_metrics['block_accuracy']:.3f}"
            )

            # Print per-LOD validation metrics
            for lod_transition in ["lod4to3", "lod3to2", "lod2to1", "lod1to0"]:
                if f"{lod_transition}_overall_acc" in val_metrics:
                    acc = val_metrics[f"{lod_transition}_overall_acc"]
                    print(f"  {lod_transition}: {acc:.3f}")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": val_metrics["loss"],
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best_model.pt",
                )
                print(f"  ** New best model saved (val_loss: {best_val_loss:.4f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")
        print()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours")

    # Save final model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_dir / "final_model.pt",
    )

    print(f"Final model saved to {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
