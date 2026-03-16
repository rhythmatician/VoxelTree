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
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from VoxelTree.train.octree_dataset import OctreeDataset, collate_octree_batch
from VoxelTree.train.octree_models import (
    OctreeConfig,
    OctreeInitModel,
    OctreeLeafModel,
    OctreeRefineModel,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)
from VoxelTree.train.prior_init import init_models_from_train_priors

# Default paths
_PKG_DIR = Path(__file__).resolve().parent.parent  # VoxelTree/
DEFAULT_VOCAB_PATH = _PKG_DIR / "config" / "voxy_vocab.json"


# ── Loss function ─────────────────────────────────────────────────────


class OctreeLoss(nn.Module):
    """Combined block-type cross-entropy + occupancy + hierarchical consistency loss.

    The occupancy loss is only applied to init (L4) and refine (L3-L1)
    models.  Leaf (L0) has no child octants to predict.

    When ``focal_gamma > 0``, occupancy uses sigmoid focal loss (inspired
    by OGN's emphasis on hard-to-classify mixed/occupied nodes).  When 0,
    falls back to plain BCE.

    RocNet insight: ``occ_pos_weight`` applies asymmetric weighting to
    penalize false negatives (missed occupied octants) more than false
    positives.  In recursive octrees, a false negative erases an entire
    subtree, so FN cost >> FP cost.  ``occ_pos_weight > 1.0`` biases
    toward recall.

    The hierarchical consistency loss (``consistency_weight > 0``) penalises
    LOD seams — the failure mode where a coarser level predicts *solid* but
    its finer children predict *air* (or vice-versa), causing visible
    popping when Voxy transitions between LODs.  It is applied to refine
    and leaf models only (the init model has no meaningful parent to be
    consistent with).

    Implementation: the child's ``block_type_logits [B, V, 32, 32, 32]`` are
    average-pooled 2× to ``[B, V, 16, 16, 16]``, then compared against
    ``parent_octant16 [B, 16, 16, 16]`` — the native 16³ Voxy octant stored
    directly in the NPZ cache.  Cross-entropy is used so the loss is
    compatible with the per-class ``class_weights``.

    Args:
        occ_weight: Weight for the occupancy loss term.
        class_weights: Optional per-class weight tensor for CE loss,
            shape ``[block_vocab_size]``.
        focal_gamma: Focal-loss focusing parameter for occupancy.  0 means
            plain BCE, >0 down-weights easy examples.  OGN-inspired.
        focal_alpha: Focal-loss alpha for the positive (occupied) class.
            Higher values up-weight the occupied minority class.
        occ_pos_weight: Positive class weight for BCE occupancy loss.
            Values > 1.0 penalize false negatives (missed subtrees) more
            heavily.  RocNet-inspired.  Ignored when focal_gamma > 0.
        consistency_weight: Weight for the hierarchical consistency loss
            (default: 0.0 = disabled).  Recommended starting value: 0.1.
    """

    def __init__(
        self,
        occ_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.75,
        level_occ_weights: Optional[Dict[int, float]] = None,
        occ_pos_weight: float = 1.0,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.occ_weight = occ_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.level_occ_weights = level_occ_weights  # e.g. {4: 2.0, 3: 1.5}
        self.occ_pos_weight = occ_pos_weight
        self.consistency_weight = consistency_weight
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model_type: str,
        level: Optional[int] = None,
        occ_weight_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            predictions: Model output dict (``block_type_logits``, optionally ``occ_logits``).
            targets: Dict with ``target_blocks`` ``[B, 32, 32, 32]`` int64,
                     ``occ_targets`` ``[B, 8]`` float (for init/refine), and
                     optionally ``parent_octant16`` ``[B, 16, 16, 16]`` int64
                     (native Voxy parent octant for the hierarchical consistency loss).
            model_type: ``"init"``, ``"refine"``, or ``"leaf"``.
            level: Optional LOD level (4, 3, 2, 1, 0) for per-level occ weight.
            occ_weight_override: If provided, overrides the default occ_weight for this call
                (e.g., for warmup scheduling). Takes precedence over level-based weights.

        Returns:
            Dict with ``total_loss``, ``block_loss``, ``occ_loss``, and
            ``consistency_loss`` tensors.
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

            if self.focal_gamma > 0:
                # Focal loss: down-weights easy negatives (air-only octants),
                # focuses on hard occupied/mixed predictions.
                occ_loss = sigmoid_focal_loss(
                    occ_logits,
                    occ_targets,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction="mean",
                )
            else:
                # Plain BCE with optional pos_weight for asymmetric FN penalty.
                # RocNet: false negatives erase subtrees so FN >> FP cost.
                pos_weight = (
                    torch.tensor(self.occ_pos_weight, device=device)
                    if self.occ_pos_weight != 1.0
                    else None
                )
                occ_loss = nn.functional.binary_cross_entropy_with_logits(
                    occ_logits, occ_targets, pos_weight=pos_weight
                )

        # Resolve effective occ weight (override > per-level > global)
        if occ_weight_override is not None:
            effective_occ_weight = occ_weight_override
        elif level is not None and self.level_occ_weights is not None:
            effective_occ_weight = self.level_occ_weights.get(level, self.occ_weight)
        else:
            effective_occ_weight = self.occ_weight

        # Hierarchical consistency loss (refine + leaf only; disabled when weight == 0)
        # Penalises LOD seams where coarser levels disagree with finer ones.
        # ``parent_octant16`` is the native 16³ Voxy octant stored directly
        # in the NPZ cache — no roundtrip through our own upsampling logic.
        consistency_loss = torch.zeros(1, device=device).squeeze()
        if self.consistency_weight > 0.0 and model_type in ("refine", "leaf"):
            parent_octant = targets.get("parent_octant16")  # [B, 16, 16, 16]
            if parent_octant is not None:
                # Coarsen child logits: [B, V, 32, 32, 32] → [B, V, 16, 16, 16]
                child_coarse = nn.functional.avg_pool3d(
                    block_logits.float(), kernel_size=2, stride=2
                )
                B2, V2, D2, H2, W2 = child_coarse.shape
                cons_flat = child_coarse.permute(0, 2, 3, 4, 1).reshape(-1, V2)
                par_flat = parent_octant.long().reshape(-1)
                consistency_loss = nn.functional.cross_entropy(
                    cons_flat,
                    par_flat,
                    weight=(
                        self.class_weights if isinstance(self.class_weights, torch.Tensor) else None
                    ),
                )

        total_loss = (
            block_loss
            + effective_occ_weight * occ_loss
            + self.consistency_weight * consistency_loss
        )

        return {
            "total_loss": total_loss,
            "block_loss": block_loss,
            "occ_loss": occ_loss,
            "consistency_loss": consistency_loss,
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

        # RocNet insight: false-negative rate is the critical metric for
        # recursive octrees.  A false negative erases an entire subtree,
        # so even 1% FNR can compound across levels to lose significant
        # geometry.  Track it separately for visibility.
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        # Also compute recall at the runtime threshold (default 0.3)
        # to match what actually happens at inference
        occ_pred_runtime = (occ_logits > -0.8473).float()  # sigmoid(x)>0.3 ⟺ x>ln(0.3/0.7)≈-0.8473
        tp_rt = (occ_pred_runtime * occ_targets).sum().item()
        fn_rt = ((1 - occ_pred_runtime) * occ_targets).sum().item()
        recall_rt = tp_rt / (tp_rt + fn_rt) if (tp_rt + fn_rt) > 0 else 1.0

        metrics["occ_precision"] = precision
        metrics["occ_recall"] = recall
        metrics["occ_f1"] = f1
        metrics["occ_fnr"] = fnr
        metrics["occ_recall_rt"] = recall_rt

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

    # Parent labels for hierarchical consistency loss (refine/leaf batches only;
    # init batches carry a zero placeholder that the loss skips automatically
    # because model_type=="init" never enters the consistency branch).
    parent_labels = batch.get("parent_labels32")
    if isinstance(parent_labels, torch.Tensor):
        targets["parent_labels32"] = parent_labels.to(device)
    parent_octant = batch.get("parent_octant16")
    if isinstance(parent_octant, torch.Tensor):
        targets["parent_octant16"] = parent_octant.to(device)

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
    scheduled_sampling_p: float = 0.0,
    occ_warmup_scale: float = 1.0,
) -> Dict[str, Any]:
    """Train for one epoch, routing each batch to the correct model.

    Args:
        models: Dict mapping ``"init"``/``"refine"``/``"leaf"`` to models.
        dataloader: Training DataLoader with :func:`collate_octree_batch`.
        loss_fn: :class:`OctreeLoss` instance.
        optimizer: Shared optimizer over all active model parameters.
        device: Torch device.
        active_types: If set, only these model types are trained (others frozen).
        scheduled_sampling_p: Probability of replacing GT parent context with
            zeros during training (default 0.0 = always use GT).  Inspired by
            OGN's PROP_KNOWN → PROP_PRED transition — trains robustness to
            imperfect parent predictions at inference time.
        occ_warmup_scale: Warmup scale for occupancy weight (0.0–1.0).
            Linearly interpolates from 0 to full weight during warmup epochs.
            Default 1.0 (no warmup).

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

        # Scheduled sampling: simulate imperfect parent predictions to
        # train robustness for recursive autoregressive rollout.
        #
        # RocNet insight: the parent→child handoff is the primary failure
        # surface in recursive octree systems.  Simply zeroing the parent
        # doesn't match the actual inference distribution (where the model
        # sees its own argmax predictions, not zeros).  We instead corrupt
        # the ground-truth parent with realistic noise:
        #   - With prob p/2: random block IDs in occupied voxels (simulates
        #     wrong block type predictions)
        #   - With prob p/2: zero out parent entirely (simulates absent
        #     parent context / worst-case mismatch)
        if scheduled_sampling_p > 0.0 and model_type in ("refine", "leaf"):
            parent = batch.get("parent_labels32")
            if isinstance(parent, torch.Tensor) and random.random() < scheduled_sampling_p:
                if random.random() < 0.5:
                    # Corrupt: replace non-air voxels with random block IDs
                    # to simulate argmax errors from the parent model
                    corrupted = parent.clone()
                    non_air = corrupted > 0
                    num_non_air = int(non_air.sum().item())
                    if num_non_air > 0:
                        # Replace ~30% of non-air voxels with random IDs
                        noise_mask = torch.rand_like(corrupted.float()) < 0.3
                        replace_mask = non_air & noise_mask
                        num_replace = int(replace_mask.sum().item())
                        if num_replace > 0:
                            random_ids = torch.randint(
                                1,
                                1104,
                                (num_replace,),
                                dtype=corrupted.dtype,
                                device=corrupted.device,
                            )
                            corrupted[replace_mask] = random_ids
                    batch["parent_labels32"] = corrupted
                else:
                    # Zero out entirely (worst-case mismatch)
                    batch["parent_labels32"] = torch.zeros_like(parent)

        optimizer.zero_grad()
        predictions = _forward_batch(models, batch, device)

        # Resolve level for per-level occ weight
        level_val: Optional[int] = None
        level_tensor = batch.get("level")
        if isinstance(level_tensor, torch.Tensor) and level_tensor.numel() > 0:
            level_val = int(level_tensor[0].item())
        elif model_type == "init":
            level_val = 4
        elif model_type == "leaf":
            level_val = 0

        # Compute effective warmup-scaled occ weight
        warmup_occ_weight = loss_fn.occ_weight * occ_warmup_scale

        losses = loss_fn(
            predictions,
            targets,
            model_type,
            level=level_val,
            occ_weight_override=warmup_occ_weight,
        )
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

            # Resolve level for per-level occ weight
            level_val: Optional[int] = None
            level_tensor = batch.get("level")
            if isinstance(level_tensor, torch.Tensor) and level_tensor.numel() > 0:
                level_val = int(level_tensor[0].item())
            elif model_type == "init":
                level_val = 4
            elif model_type == "leaf":
                level_val = 0

            losses = loss_fn(
                predictions,
                targets,
                model_type,
                level=level_val,
            )
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


def main(argv: list[str] | None = None) -> None:  # noqa: FA100
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
        "--init-architecture",
        type=str,
        default="encoder2d_decoder3d",
        choices=["encoder2d_decoder3d", "full_3d_unet"],
        help=(
            "Init-model backbone (default: encoder2d_decoder3d). "
            "The 2D→3D path is the current shootout winner; full_3d_unet is kept "
            "for backward compatibility and ablation runs."
        ),
    )
    parser.add_argument(
        "--refine-channels",
        type=int,
        nargs=3,
        default=[48, 96, 192],
        metavar=("C0", "C1", "C2"),
        help="Channel widths for refine model (default: 48 96 192)",
    )
    parser.add_argument(
        "--leaf-channels",
        type=int,
        nargs=3,
        default=[32, 64, 128],
        metavar=("C0", "C1", "C2"),
        help="Channel widths for leaf model (default: 32 64 128)",
    )
    parser.add_argument(
        "--leaf-bottleneck-extra-depth",
        type=int,
        default=1,
        help=(
            "Extra DoubleConv3d blocks at the leaf 8³ bottleneck (default: 1). "
            "This is the conservative leaf-model upgrade from the leaf shootout."
        ),
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
        help="Weight for occupancy loss (default: 1.0)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help=(
            "Focal-loss gamma for occupancy (default: 2.0). "
            "0 = plain BCE.  >0 down-weights easy examples (OGN-inspired)."
        ),
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help=(
            "Focal-loss alpha for occupied class (default: 0.75). "
            "Higher values up-weight the occupied minority class."
        ),
    )
    parser.add_argument(
        "--occ-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs to linearly ramp occupancy weight from 0 "
            "to --occ-weight (default: 0, no warmup)."
        ),
    )
    parser.add_argument(
        "--occ-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Positive class weight for occupancy BCE loss (default: 1.0). "
            "Values > 1.0 penalize false negatives (missed subtrees) more "
            "heavily.  RocNet-inspired: in recursive octrees, FN cost >> FP cost. "
            "Recommended: 2.0-3.0.  Ignored when --focal-gamma > 0."
        ),
    )
    parser.add_argument(
        "--parent-embed-dim",
        type=int,
        default=16,
        help="Parent context embedding dimension (default: 16)",
    )
    parser.add_argument(
        "--parent-context-mode",
        type=str,
        default="embed",
        choices=["embed", "zeros", "disabled"],
        help=(
            "Parent context ablation mode (default: embed). "
            "'embed' = learned embedding, 'zeros' = zero-filled channels, "
            "'disabled' = remove parent channels entirely."
        ),
    )
    parser.add_argument(
        "--level-occ-weights",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Per-level occupancy weight overrides as 'L:W,L:W,...'. "
            "Example: '4:2.0,3:1.5' sets L4=2.0, L3=1.5. "
            "Unspecified levels use --occ-weight."
        ),
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
    parser.add_argument(
        "--init-block-bias",
        action="store_true",
        help=(
            "Initialize init/refine/leaf block_head biases from training-split "
            "log-frequency block priors before training. Ignored when --resume is used."
        ),
    )
    # ── Step 8: OGN-inspired model options ───────────────────────────
    parser.add_argument(
        "--bottleneck-extra-depth",
        type=int,
        default=0,
        help=(
            "Legacy global fallback for extra DoubleConv3d blocks at the 8³ bottleneck "
            "(default: 0). Used by models without a per-model override. "
            "OGN-inspired: multiple convolutions per octree level."
        ),
    )
    parser.add_argument(
        "--parent-refine-conv",
        action="store_true",
        default=False,
        help=(
            "Add a Conv3dBlock after parent embedding lookup to learn "
            "spatial correlations before concatenation (OGN-inspired)."
        ),
    )
    parser.add_argument(
        "--use-occ-gate",
        action="store_true",
        default=False,
        help=(
            "Enable occupancy-gated bottleneck modulation (OGN-inspired). "
            "Sigmoid(occ_logits) gates bottleneck features before decoding, "
            "so occupancy predictions influence block-type generation."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling",
        type=float,
        default=0.0,
        metavar="P",
        help=(
            "Scheduled sampling probability (0.0-1.0, default: 0.0). "
            "During training, with probability P replace GT parent_labels32 "
            "with the parent model's own argmax predictions.  Inspired by "
            "OGN's PROP_KNOWN → PROP_PRED transition."
        ),
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        metavar="W",
        help=(
            "Weight for hierarchical consistency loss (default: 0.0 = disabled). "
            "Penalises LOD seams by enforcing that the refine/leaf model's "
            "predictions, when coarsened 2×, agree with the parent section's "
            "target labels.  Recommended starting value: 0.1.  "
            "Only affects refine and leaf models."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        metavar="TYPES",
        help=(
            "Comma-separated model types to train: init, refine, leaf, or 'all' (default). "
            "Examples: --models init  --models init,leaf  --models all. "
            "Non-'all' values save per-model checkpoints named '{types}_best.pt' etc., "
            "and load model-specific pair caches when available."
        ),
    )
    args = parser.parse_args(argv)

    # ── Parse active model types ──────────────────────────────────────────────
    _VALID_MODEL_TYPES = {"init", "refine", "leaf"}
    if args.models == "all":
        active_types: Optional[Set[str]] = None  # train all models jointly
        _ckpt_prefix = ""  # legacy unified naming: best_model.pt etc.
    else:
        active_types = set(args.models.split(","))
        invalid = active_types - _VALID_MODEL_TYPES
        if invalid:
            raise ValueError(f"Unknown model types: {invalid}. Valid: init, refine, leaf or 'all'")
        # Checkpoint filename prefix: e.g. 'init_' or 'init_leaf_'
        _ckpt_prefix = "_".join(sorted(active_types)) + "_"
    print(f"Active model types: {sorted(active_types) if active_types else 'all'}")

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
        init_architecture=args.init_architecture,
        parent_embed_dim=args.parent_embed_dim,
        parent_context_mode=args.parent_context_mode,
        bottleneck_extra_depth=args.bottleneck_extra_depth,
        leaf_bottleneck_extra_depth=args.leaf_bottleneck_extra_depth,
        parent_refine_conv=args.parent_refine_conv,
        use_occ_gate=args.use_occ_gate,
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
    print(f"  init architecture: {config.init_architecture}")
    print(
        "  leaf bottleneck extra depth: "
        f"{getattr(config, 'leaf_bottleneck_extra_depth', config.bottleneck_extra_depth)}"
    )

    if args.init_block_bias:
        if args.resume is not None:
            print("\nSkipping block-head bias init because --resume will load checkpoint weights.")
        else:
            print("\nInitializing block_head biases from training-split block priors...")
            init_models_from_train_priors(
                models,
                data_dir,
                block_vocab_size,
                split="train",
                verbose=True,
            )

    # Print Step 8 feature status
    step8_active = []
    if config.bottleneck_extra_depth > 0:
        step8_active.append(f"bottleneck_extra_depth={config.bottleneck_extra_depth}")
    if config.parent_refine_conv:
        step8_active.append("parent_refine_conv")
    if config.use_occ_gate:
        step8_active.append("occ_gate")
    if args.scheduled_sampling > 0:
        step8_active.append(f"scheduled_sampling={args.scheduled_sampling:.2f}")
    if step8_active:
        print(f"\nStep 8 (OGN-inspired): {', '.join(step8_active)}")

    # ── Datasets ─────────────────────────────────────────────────────
    # When training a single model type, prefer its dedicated pair cache so
    # the DataLoader only loops over the relevant levels.
    dataset_model_type: Optional[str] = None
    if active_types is not None and len(active_types) == 1:
        dataset_model_type = next(iter(active_types))

    print("\nLoading datasets...")
    train_dataset = OctreeDataset(data_dir=data_dir, split="train", model_type=dataset_model_type)
    val_dataset = OctreeDataset(data_dir=data_dir, split="val", model_type=dataset_model_type)

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
    # Parse per-level occ weights
    level_occ_weights: Optional[Dict[int, float]] = None
    if args.level_occ_weights is not None:
        level_occ_weights = {}
        for pair in args.level_occ_weights.split(","):
            parts = pair.strip().split(":")
            if len(parts) != 2:
                raise ValueError(f"Bad --level-occ-weights format: '{pair}'. Expected L:W")
            level_occ_weights[int(parts[0])] = float(parts[1])
        print(f"Per-level occ weights: {level_occ_weights}")

    loss_fn = OctreeLoss(
        occ_weight=args.occ_weight,
        class_weights=class_weights_tensor,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        level_occ_weights=level_occ_weights,
        occ_pos_weight=args.occ_pos_weight,
        consistency_weight=args.consistency_weight,
    ).to(device)

    if args.consistency_weight > 0.0:
        print(f"Hierarchical consistency loss enabled (weight={args.consistency_weight})")

    # Stash warmup configuration on the loss_fn for use in the training loop
    _occ_warmup_epochs: int = args.occ_warmup_epochs

    if active_types is not None:
        # Independent training: only optimise active model parameters
        active_params = [
            p
            for name, m in models.items()
            if name in active_types
            for p in m.parameters()
            if p.requires_grad
        ]
        optimizer = optim.AdamW(active_params, lr=args.lr, weight_decay=1e-4)
        print(f"Optimizer: {len(active_params)} tensors from {sorted(active_types)}")
    else:
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

        # Compute occupancy weight warmup scale (without mutating loss_fn)
        occ_warmup_scale = 1.0
        if _occ_warmup_epochs > 0 and epoch <= _occ_warmup_epochs:
            occ_warmup_scale = epoch / _occ_warmup_epochs

        # Train
        train_metrics = train_octree_epoch(
            models,
            train_loader,
            loss_fn,
            optimizer,
            device,
            active_types=active_types,
            scheduled_sampling_p=args.scheduled_sampling,
            occ_warmup_scale=occ_warmup_scale,
        )

        # Step scheduler
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Print training stats
        print(f"\nEpoch {epoch}/{args.epochs}")
        _cons = train_metrics.get("consistency_loss", 0.0)
        _cons_str = f", Cons: {_cons:.4f}" if _cons else ""
        print(
            f"  Train — Loss: {train_metrics.get('total_loss', 0):.4f} "
            f"(Block: {train_metrics.get('block_loss', 0):.4f}, "
            f"Occ: {train_metrics.get('occ_loss', 0):.4f}"
            f"{_cons_str})"
        )
        print(
            f"  Train — Acc: {train_metrics.get('overall_accuracy', 0):.3f} "
            f"(Air: {train_metrics.get('air_accuracy', 0):.3f}, "
            f"Block: {train_metrics.get('block_accuracy', 0):.3f})"
        )
        if "occ_f1" in train_metrics:
            print(
                f"  Train — Occ F1: {train_metrics['occ_f1']:.3f}"
                f"  Recall: {train_metrics.get('occ_recall', 0):.3f}"
                f"  FNR: {train_metrics.get('occ_fnr', 0):.3f}"
                f"  Recall@0.3: {train_metrics.get('occ_recall_rt', 0):.3f}"
            )

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
            _val_cons = val_metrics.get("consistency_loss", 0.0)
            _val_cons_str = f", Cons: {_val_cons:.4f}" if _val_cons else ""
            print(
                f"  Val   — Loss: {val_loss:.4f} "
                f"(Block: {val_metrics.get('block_loss', 0):.4f}, "
                f"Occ: {val_metrics.get('occ_loss', 0):.4f}"
                f"{_val_cons_str})"
            )
            print(
                f"  Val   — Acc: {val_metrics.get('overall_accuracy', 0):.3f} "
                f"(Air: {val_metrics.get('air_accuracy', 0):.3f}, "
                f"Block: {val_metrics.get('block_accuracy', 0):.3f})"
            )
            if "occ_f1" in val_metrics:
                print(
                    f"  Val   — Occ F1: {val_metrics['occ_f1']:.3f}"
                    f"  Recall: {val_metrics.get('occ_recall', 0):.3f}"
                    f"  FNR: {val_metrics.get('occ_fnr', 0):.3f}"
                    f"  Recall@0.3: {val_metrics.get('occ_recall_rt', 0):.3f}"
                )

            # Per-type/level breakdown
            for prefix in ("init", "refine", "leaf", "L0", "L1", "L2", "L3", "L4"):
                key = f"{prefix}/overall_accuracy"
                if key in val_metrics:
                    print(f"    {prefix}: acc={val_metrics[key]:.3f}")

            # ReduceLROnPlateau step
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Save best (unified or per-model depending on active_types)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_state_dicts = {
                    name: m.state_dict()
                    for name, m in models.items()
                    if active_types is None or name in active_types
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dicts": _save_state_dicts,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "active_types": list(active_types) if active_types else None,
                    },
                    output_dir / f"{_ckpt_prefix}best_model.pt",
                )
                print(f"  ** New best model saved (val_loss: {best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            _save_state_dicts = {
                name: m.state_dict()
                for name, m in models.items()
                if active_types is None or name in active_types
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dicts": _save_state_dicts,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "active_types": list(active_types) if active_types else None,
                },
                output_dir / f"{_ckpt_prefix}checkpoint_epoch_{epoch}.pt",
            )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch time: {epoch_time:.1f}s | LR: {lr:.2e}")

    # ── Final save ───────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")

    _save_state_dicts = {
        name: m.state_dict()
        for name, m in models.items()
        if active_types is None or name in active_types
    }
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dicts": _save_state_dicts,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "active_types": list(active_types) if active_types else None,
        },
        output_dir / f"{_ckpt_prefix}final_model.pt",
    )
    _final_name = f"{_ckpt_prefix}final_model.pt"
    print(f"Final model saved to {output_dir / _final_name}")


if __name__ == "__main__":
    main()
