#!/usr/bin/env python3
"""Octree training script — 3 models on 32³ WorldSection grids.

Trains three separate 3D U-Net models for the octree generation pipeline:

  - **init**   (L4):        OctreeInitModel   — root, no parent
  - **refine** (L3/L2/L1):  OctreeRefineModel — shared with level embedding
  - **leaf**   (L0):        OctreeLeafModel   — block-level, no occ head

Each batch is routed to exactly one model based on the ``model_type`` field
produced by :func:`collate_octree_batch`.

Usage::

    python train_octree.py \\
        --data-dir data/octree_pairs \\
        --output-dir octree_training \\
        --epochs 100 --batch-size 8 --lr 3e-4

Resume from checkpoint::

    python train_octree.py --resume octree_training/best_model.pt ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.octree_dataset import OctreeDataset, collate_octree_batch  # noqa: E402
from train.octree_models import (  # noqa: E402
    OctreeConfig,
    OctreeInitModel,
    OctreeLeafModel,
    OctreeRefineModel,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)

# Default paths
DEFAULT_VOCAB_PATH = Path("config/voxy_vocab.json")


# ── Loss function ─────────────────────────────────────────────────────


class OctreeLoss(nn.Module):
    """Combined block-type cross-entropy + occupancy BCE loss.

    The occupancy loss is only applied to init (L4) and refine (L3-L1)
    models.  Leaf (L0) has no child octants to predict.

    Args:
        occ_weight: Weight for the occupancy BCE loss term.
        class_weights: Optional per-class weight tensor for CE loss,
            shape ``[block_vocab_size]``.
    """

    def __init__(
        self,
        occ_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.occ_weight = occ_weight
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model_type: str,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            predictions: Model output dict (``block_type_logits``, optionally ``occ_logits``).
            targets: Dict with ``target_blocks`` ``[B, 32, 32, 32]`` int64,
                     and ``occ_targets`` ``[B, 8]`` float (for init/refine).
            model_type: ``"init"``, ``"refine"``, or ``"leaf"``.

        Returns:
            Dict with ``total_loss``, ``block_loss``, and ``occ_loss`` tensors.
        """
        # Block classification loss
        block_logits = predictions["block_type_logits"]  # [B, V, 32, 32, 32]
        target_blocks = targets["target_blocks"]  # [B, 32, 32, 32]

        B, V, D, H, W = block_logits.shape
        logits_flat = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, V)
        targets_flat = target_blocks.reshape(-1)

        block_loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.class_weights if isinstance(self.class_weights, torch.Tensor) else None,
        )

        # Occupancy loss (init and refine only)
        device = block_logits.device
        occ_loss = torch.zeros(1, device=device).squeeze()

        if model_type in ("init", "refine") and "occ_logits" in predictions:
            occ_logits = predictions["occ_logits"]  # [B, 8]
            occ_targets = targets["occ_targets"]  # [B, 8] float
            occ_loss = nn.functional.binary_cross_entropy_with_logits(occ_logits, occ_targets)

        total_loss = block_loss + self.occ_weight * occ_loss

        return {
            "total_loss": total_loss,
            "block_loss": block_loss,
            "occ_loss": occ_loss,
        }


# ── Metrics ───────────────────────────────────────────────────────────


def _bitmask_to_binary(bitmask: torch.Tensor) -> torch.Tensor:
    """Convert uint8 bitmask ``[B]`` to ``[B, 8]`` float binary targets."""
    bits = torch.arange(8, device=bitmask.device).unsqueeze(0)
    return ((bitmask.unsqueeze(1).long() >> bits) & 1).float()


@torch.no_grad()
def compute_octree_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    model_type: str,
) -> Dict[str, float]:
    """Compute per-batch evaluation metrics.

    Args:
        predictions: Model output dict.
        targets: Target dict (``target_blocks``, ``occ_targets``).
        model_type: ``"init"``, ``"refine"``, or ``"leaf"``.

    Returns:
        Dict with accuracy metrics.
    """
    block_logits = predictions["block_type_logits"]  # [B, V, 32, 32, 32]
    target_blocks = targets["target_blocks"]  # [B, 32, 32, 32]

    block_pred = block_logits.argmax(dim=1)  # [B, 32, 32, 32]

    # Overall accuracy
    overall_acc = (block_pred == target_blocks).float().mean().item()

    # Air accuracy (class 0)
    air_mask = target_blocks == 0
    if air_mask.sum() > 0:
        air_acc = (block_pred[air_mask] == 0).float().mean().item()
    else:
        air_acc = 1.0

    # Solid block accuracy
    solid_mask = target_blocks > 0
    if solid_mask.sum() > 0:
        block_acc = (block_pred[solid_mask] == target_blocks[solid_mask]).float().mean().item()
    else:
        block_acc = 1.0

    metrics: Dict[str, float] = {
        "overall_accuracy": overall_acc,
        "air_accuracy": air_acc,
        "block_accuracy": block_acc,
    }

    # Occupancy F1 (init/refine only)
    if model_type in ("init", "refine") and "occ_logits" in predictions:
        occ_logits = predictions["occ_logits"]
        occ_targets = targets["occ_targets"]  # [B, 8]
        occ_pred = (occ_logits > 0).float()  # sigmoid threshold at 0.5

        tp = (occ_pred * occ_targets).sum().item()
        fp = (occ_pred * (1 - occ_targets)).sum().item()
        fn = ((1 - occ_pred) * occ_targets).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["occ_precision"] = precision
        metrics["occ_recall"] = recall
        metrics["occ_f1"] = f1

    return metrics


# ── Batch routing ─────────────────────────────────────────────────────


def _prepare_targets(
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract target tensors from a batch dict."""
    labels = batch["labels32"]
    assert isinstance(labels, torch.Tensor)
    targets: Dict[str, torch.Tensor] = {
        "target_blocks": labels.to(device),
    }

    # Occupancy targets from bitmask
    nec = batch["non_empty_children"]
    assert isinstance(nec, torch.Tensor)
    targets["occ_targets"] = _bitmask_to_binary(nec).to(device)

    return targets


def _forward_batch(
    models: Dict[str, nn.Module],
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Route batch to the correct model and return predictions."""
    model_type = batch["model_type"]
    assert isinstance(model_type, str)
    model = models[model_type]

    heightmap = batch["heightmap32"]
    biome = batch["biome32"]
    y_position = batch["y_position"]
    assert isinstance(heightmap, torch.Tensor)
    assert isinstance(biome, torch.Tensor)
    assert isinstance(y_position, torch.Tensor)

    heightmap = heightmap.to(device)
    biome = biome.to(device)
    y_position = y_position.to(device)

    if model_type == "init":
        assert isinstance(model, OctreeInitModel)
        return model(heightmap=heightmap, biome=biome, y_position=y_position)

    # Refine and leaf both take parent context
    parent_blocks = batch["parent_labels32"]
    assert isinstance(parent_blocks, torch.Tensor)
    parent_blocks = parent_blocks.to(device)

    if model_type == "refine":
        assert isinstance(model, OctreeRefineModel)
        level = batch["level"]
        assert isinstance(level, torch.Tensor)
        level = level.to(device)
        return model(
            heightmap=heightmap,
            biome=biome,
            y_position=y_position,
            level=level,
            parent_blocks=parent_blocks,
        )

    # model_type == "leaf"
    assert isinstance(model, OctreeLeafModel)
    return model(
        heightmap=heightmap,
        biome=biome,
        y_position=y_position,
        parent_blocks=parent_blocks,
    )


# ── Train / validate epochs ──────────────────────────────────────────


def train_octree_epoch(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,  # type: ignore[type-arg]
    loss_fn: OctreeLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    active_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Train for one epoch, routing each batch to the correct model.

    Args:
        models: Dict mapping ``"init"``/``"refine"``/``"leaf"`` to models.
        dataloader: Training DataLoader with :func:`collate_octree_batch`.
        loss_fn: :class:`OctreeLoss` instance.
        optimizer: Shared optimizer over all active model parameters.
        device: Torch device.
        active_types: If set, only these model types are trained (others frozen).

    Returns:
        Dict of averaged metrics across the epoch.
    """
    for key, m in models.items():
        if active_types is None or key in active_types:
            m.train()
        # else: stays in eval with frozen params

    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    num_batches = 0

    total_batches = len(dataloader)

    for batch in tqdm(
        dataloader,
        total=total_batches,
        unit="batch",
        desc="Train",
        leave=False,
        dynamic_ncols=True,
    ):
        model_type = batch.get("model_type", "empty")
        if model_type == "empty":
            continue
        assert isinstance(model_type, str)

        # Skip batches for inactive model types
        if active_types is not None and model_type not in active_types:
            continue

        targets = _prepare_targets(batch, device)

        optimizer.zero_grad()
        predictions = _forward_batch(models, batch, device)
        losses = loss_fn(predictions, targets, model_type)
        loss = losses["total_loss"]

        loss.backward()
        optimizer.step()

        metrics = compute_octree_metrics(predictions, targets, model_type)

        # Accumulate global metrics
        num_batches += 1
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            accum[k] = accum.get(k, 0.0) + val
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v

        # Accumulate per-type metrics
        type_key = f"{model_type}/"
        counts[model_type] = counts.get(model_type, 0) + 1
        for k, v in metrics.items():
            full_key = type_key + k
            accum[full_key] = accum.get(full_key, 0.0) + v

    # Average everything
    results: Dict[str, Any] = {}
    if num_batches > 0:
        for k, v in accum.items():
            if "/" in k:
                mt = k.split("/")[0]
                results[k] = v / max(counts.get(mt, 1), 1)
            else:
                results[k] = v / num_batches
    results["num_batches"] = num_batches
    results["type_counts"] = dict(counts)
    return results


def validate_octree_epoch(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,  # type: ignore[type-arg]
    loss_fn: OctreeLoss,
    device: torch.device,
) -> Dict[str, Any]:
    """Validate for one epoch with per-level metrics.

    Args:
        models: Dict mapping ``"init"``/``"refine"``/``"leaf"`` to models.
        dataloader: Validation DataLoader.
        loss_fn: :class:`OctreeLoss` instance.
        device: Torch device.

    Returns:
        Dict of averaged metrics (global + per model type + per level).
    """
    for m in models.values():
        m.eval()

    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            unit="batch",
            desc="Val",
            leave=False,
            dynamic_ncols=True,
        ):
            model_type = batch.get("model_type", "empty")
            if model_type == "empty":
                continue
            assert isinstance(model_type, str)

            targets = _prepare_targets(batch, device)
            predictions = _forward_batch(models, batch, device)
            losses = loss_fn(predictions, targets, model_type)
            metrics = compute_octree_metrics(predictions, targets, model_type)

            num_batches += 1
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                accum[k] = accum.get(k, 0.0) + val
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v

            # Per model type
            type_key = f"{model_type}/"
            counts[model_type] = counts.get(model_type, 0) + 1
            for k, v in metrics.items():
                accum[type_key + k] = accum.get(type_key + k, 0.0) + v
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                accum[type_key + k] = accum.get(type_key + k, 0.0) + val

            # Per level (refine batches may have mixed levels, but all share model)
            level_tensor = batch.get("level")
            if isinstance(level_tensor, torch.Tensor) and level_tensor.numel() > 0:
                avg_level = level_tensor.float().mean().item()
                level_key = f"L{int(round(avg_level))}/"
                counts[level_key] = counts.get(level_key, 0) + 1
                for k, v in metrics.items():
                    accum[level_key + k] = accum.get(level_key + k, 0.0) + v

    # Average
    results: Dict[str, Any] = {}
    if num_batches > 0:
        for k, v in accum.items():
            if "/" in k:
                group_key = k.split("/")[0]
                c = max(counts.get(group_key, 1), 1)
                results[k] = v / c
            else:
                results[k] = v / num_batches
    results["num_batches"] = num_batches
    results["type_counts"] = dict(counts)
    return results


# ── Main training loop ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3-model octree generation pipeline on 32³ WorldSections"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to octree pair caches (must contain train/val_octree_pairs.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./octree_training",
        help="Checkpoint output directory",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--init-channels",
        type=int,
        nargs=3,
        default=[24, 48, 96],
        metavar=("C0", "C1", "C2"),
        help="Channel widths for init model (default: 24 48 96)",
    )
    parser.add_argument(
        "--refine-channels",
        type=int,
        nargs=3,
        default=[32, 64, 128],
        metavar=("C0", "C1", "C2"),
        help="Channel widths for refine model (default: 32 64 128)",
    )
    parser.add_argument(
        "--leaf-channels",
        type=int,
        nargs=3,
        default=[48, 96, 192],
        metavar=("C0", "C1", "C2"),
        help="Channel widths for leaf model (default: 48 96 192)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Path to voxy_vocab.json (default: config/voxy_vocab.json)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint .pt to resume from",
    )
    parser.add_argument(
        "--occ-weight",
        type=float,
        default=1.0,
        help="Weight for occupancy BCE loss (default: 1.0)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--validate-every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=main process, safest on Windows)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to N randomly-sampled pairs (quick smoke test)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Limit PyTorch intra-op CPU threads",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="LR scheduler: cosine (CosineAnnealingLR) or plateau (ReduceLROnPlateau)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        metavar="PATH_OR_AUTO",
        help=(
            "Path to class_weights.npz or 'auto' to compute on the fly. "
            "Applies median-frequency balancing to block-type CE loss."
        ),
    )
    args = parser.parse_args()

    # CPU thread limit
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"CPU threads limited to: {args.num_threads}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Block vocabulary
    vocab_path: Path = args.vocab
    if vocab_path.exists():
        with open(vocab_path) as f:
            voxy_vocab = json.load(f)
        block_vocab_size = len(voxy_vocab)
        print(f"Voxy vocabulary: {block_vocab_size} block types from {vocab_path}")
    else:
        block_vocab_size = 1104
        print(f"Warning: {vocab_path} not found, using default vocab size {block_vocab_size}")

    # Check data dir for max block ID
    data_dir = Path(args.data_dir)
    for candidate in sorted(data_dir.glob("train_octree_pairs*.npz")):
        peek = np.load(candidate, mmap_mode="r")
        if "labels32" in peek:
            data_max = int(peek["labels32"].max())
            if data_max >= block_vocab_size:
                print(
                    f"Warning: data has block ID {data_max} >= vocab {block_vocab_size} "
                    f"— extending to {data_max + 1}"
                )
                block_vocab_size = data_max + 1
        break

    # When resuming, honour checkpoint's vocab size
    if args.resume is not None:
        ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt_peek.get("config")
        if ckpt_cfg is not None and hasattr(ckpt_cfg, "block_vocab_size"):
            ckpt_bvs = ckpt_cfg.block_vocab_size
            if ckpt_bvs != block_vocab_size:
                print(
                    f"Checkpoint vocab={ckpt_bvs} differs from "
                    f"detected ({block_vocab_size}) — using checkpoint value"
                )
                block_vocab_size = ckpt_bvs
        del ckpt_peek

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build config + models ────────────────────────────────────────
    config = OctreeConfig(
        block_vocab_size=block_vocab_size,
        init_channels=tuple(args.init_channels),
        refine_channels=tuple(args.refine_channels),
        leaf_channels=tuple(args.leaf_channels),
    )

    models: Dict[str, nn.Module] = {
        "init": create_init_model(config).to(device),
        "refine": create_refine_model(config).to(device),
        "leaf": create_leaf_model(config).to(device),
    }

    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models.values())
    print(f"\nTotal parameters across 3 models: {total_params:,}")
    for name, m in models.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n:,} params")

    # ── Datasets ─────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = OctreeDataset(data_dir=data_dir, split="train")
    val_dataset = OctreeDataset(data_dir=data_dir, split="val")

    # Subset for smoke tests
    if args.max_samples is not None:
        n_train = min(args.max_samples, len(train_dataset))
        n_val = min(max(1, args.max_samples // 10), len(val_dataset))
        train_idx = random.sample(range(len(train_dataset)), n_train)
        val_idx = random.sample(range(len(val_dataset)), n_val)
        train_dataset = Subset(train_dataset, train_idx)  # type: ignore[assignment]
        val_dataset = Subset(val_dataset, val_idx)  # type: ignore[assignment]
        print(f"--max-samples: {n_train} train + {n_val} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_octree_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_octree_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")

    # ── Class weights ────────────────────────────────────────────────
    class_weights_tensor: Optional[torch.Tensor] = None
    if args.class_weights is not None:
        cw_arg = args.class_weights.strip()
        if cw_arg.lower() == "auto":
            cw_path = data_dir / "class_weights.npz"
            if cw_path.exists():
                print(f"Loading cached class weights: {cw_path}")
                cw_data = np.load(cw_path)
                cw_arr = cw_data["class_weights"]
            else:
                print("Warning: auto class weights requested but no class_weights.npz found")
                cw_arr = None
        else:
            cw_path = Path(cw_arg)
            if not cw_path.exists():
                raise FileNotFoundError(f"--class-weights file not found: {cw_path}")
            cw_data = np.load(cw_path)
            cw_arr = cw_data["class_weights"]

        if cw_arr is not None:
            class_weights_tensor = torch.tensor(cw_arr, dtype=torch.float32)
            if len(class_weights_tensor) < block_vocab_size:
                pad = block_vocab_size - len(class_weights_tensor)
                class_weights_tensor = nn.functional.pad(class_weights_tensor, (0, pad), value=0.0)
            nonzero = int((class_weights_tensor > 0).sum())
            print(f"  Class weights: {nonzero}/{len(cw_arr)} non-zero classes")

    # ── Loss, optimizer, scheduler ───────────────────────────────────
    loss_fn = OctreeLoss(
        occ_weight=args.occ_weight,
        class_weights=class_weights_tensor,
    ).to(device)

    all_params = list(p for m in models.values() for p in m.parameters() if p.requires_grad)
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    if args.scheduler == "cosine":
        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        print(f"\nResuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        if "model_state_dicts" not in ckpt:
            raise ValueError(
                f"Checkpoint {args.resume} has no 'model_state_dicts'. "
                f"Keys: {list(ckpt.keys())}"
            )

        for name, m in models.items():
            if name in ckpt["model_state_dicts"]:
                result = m.load_state_dict(ckpt["model_state_dicts"][name], strict=False)
                if result.unexpected_keys:
                    print(f"  {name}: ignoring unexpected keys: {result.unexpected_keys}")
                if result.missing_keys:
                    print(f"  {name}: missing keys: {result.missing_keys}")
            else:
                print(f"  Warning: no saved state for '{name}', using random weights")

        try:
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            print(f"  Warning: could not restore optimizer ({e}), using fresh")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

        # Advance scheduler
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            for _ in range(start_epoch - 1):
                scheduler.step()

        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    print("\nStarting training...\n")
    start_time = time.time()

    for epoch in tqdm(
        range(start_epoch, args.epochs + 1),
        unit="epoch",
        dynamic_ncols=True,
        desc="Octree Training",
    ):
        epoch_start = time.time()

        # Train
        train_metrics = train_octree_epoch(models, train_loader, loss_fn, optimizer, device)

        # Step scheduler
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Print training stats
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(
            f"  Train — Loss: {train_metrics.get('total_loss', 0):.4f} "
            f"(Block: {train_metrics.get('block_loss', 0):.4f}, "
            f"Occ: {train_metrics.get('occ_loss', 0):.4f})"
        )
        print(
            f"  Train — Acc: {train_metrics.get('overall_accuracy', 0):.3f} "
            f"(Air: {train_metrics.get('air_accuracy', 0):.3f}, "
            f"Block: {train_metrics.get('block_accuracy', 0):.3f})"
        )
        if "occ_f1" in train_metrics:
            print(f"  Train — Occ F1: {train_metrics['occ_f1']:.3f}")

        # Per-type breakdown
        type_counts = train_metrics.get("type_counts", {})
        for mt in ("init", "refine", "leaf"):
            if mt in type_counts:
                acc = train_metrics.get(f"{mt}/overall_accuracy", 0)
                print(f"    {mt} ({type_counts[mt]} batches): acc={acc:.3f}")

        # Validate
        if epoch % args.validate_every == 0:
            val_metrics = validate_octree_epoch(models, val_loader, loss_fn, device)

            val_loss = val_metrics.get("total_loss", float("inf"))
            print(
                f"  Val   — Loss: {val_loss:.4f} "
                f"(Block: {val_metrics.get('block_loss', 0):.4f}, "
                f"Occ: {val_metrics.get('occ_loss', 0):.4f})"
            )
            print(
                f"  Val   — Acc: {val_metrics.get('overall_accuracy', 0):.3f} "
                f"(Air: {val_metrics.get('air_accuracy', 0):.3f}, "
                f"Block: {val_metrics.get('block_accuracy', 0):.3f})"
            )

            # Per-type/level breakdown
            for prefix in ("init", "refine", "leaf", "L0", "L1", "L2", "L3", "L4"):
                key = f"{prefix}/overall_accuracy"
                if key in val_metrics:
                    print(f"    {prefix}: acc={val_metrics[key]:.3f}")

            # ReduceLROnPlateau step
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dicts": {name: m.state_dict() for name, m in models.items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best_model.pt",
                )
                print(f"  ** New best model saved (val_loss: {best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dicts": {name: m.state_dict() for name, m in models.items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch time: {epoch_time:.1f}s | LR: {lr:.2e}")

    # ── Final save ───────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dicts": {name: m.state_dict() for name, m in models.items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_dir / "final_model.pt",
    )
    print(f"Final model saved to {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
