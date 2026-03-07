#!/usr/bin/env python3
"""
VoxelTree Evaluation Metrics Framework

Implements Item 4 from acceptance criteria: comprehensive metrics & evaluation harness
including IoU, frequent-set accuracy, rollout evaluation, and confusion matrix analysis.

This provides the foundation for tracking 99% accuracy goals and model performance
across different LOD levels and block types.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class VoxelMetrics:
    """
    Comprehensive metrics calculator for voxel prediction tasks.

    Tracks IoU, accuracy, confusion matrices, and frequency-based metrics
    across different LOD levels and block types.
    """

    def __init__(
        self,
        block_vocab_size: int = 1104,
        frequent_blocks_k: int = 50,
        accuracy_threshold: float = 0.99,
    ):
        """
        Initialize metrics calculator.

        Args:
            block_vocab_size: Total number of block types
            frequent_blocks_k: Number of most frequent blocks to track separately
            accuracy_threshold: Accuracy threshold for goal tracking (e.g., 0.99)
        """
        self.block_vocab_size = block_vocab_size
        self.frequent_blocks_k = frequent_blocks_k
        self.accuracy_threshold = accuracy_threshold

        # Accumulated metrics
        self.reset()

        # Frequent blocks (to be determined from data)
        self.frequent_block_ids: Optional[List[int]] = None

    def reset(self):
        """Reset accumulated metrics."""
        self.total_samples = 0
        self.total_correct = 0
        self.total_air_correct = 0
        self.total_air_samples = 0
        self.total_solid_correct = 0
        self.total_solid_samples = 0

        # IoU components (for solid/air binary classification)
        self.intersection_solid = 0
        self.union_solid = 0
        self.intersection_air = 0
        self.union_air = 0

        # Per-LOD tracking
        self.lod_metrics = {}

        # Confusion matrix accumulation
        self.confusion_predictions = []
        self.confusion_targets = []

        # Frequent block tracking
        self.frequent_correct = 0
        self.frequent_total = 0

        logger.debug("Reset metrics accumulator")

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        air_mask_pred: torch.Tensor,
        air_mask_true: torch.Tensor,
        lod_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: (B, H, W, D) predicted block type logits
            targets: (B, H, W, D) true block type indices
            air_mask_pred: (B, 1, H, W, D) predicted air mask logits
            air_mask_true: (B, 1, H, W, D) true air mask (0=air, 1=solid)
            lod_indices: (B,) LOD level indices for per-LOD tracking
        """

        # Convert predictions to indices
        pred_indices = torch.argmax(predictions, dim=1)  # (B, H, W, D)
        air_pred_indices = (air_mask_pred.squeeze(1) > 0).long()  # (B, H, W, D)
        air_true_indices = air_mask_true.squeeze(1).long()  # (B, H, W, D)

        # Flatten for easier computation
        pred_flat = pred_indices.flatten()
        target_flat = targets.flatten()
        air_pred_flat = air_pred_indices.flatten()
        air_true_flat = air_true_indices.flatten()

        # Overall accuracy
        correct = (pred_flat == target_flat).sum().item()
        total = pred_flat.numel()
        self.total_correct += correct
        self.total_samples += total

        # Air/solid accuracy
        air_correct = (air_pred_flat == air_true_flat).sum().item()
        self.total_air_correct += air_correct
        self.total_air_samples += air_pred_flat.numel()

        # Solid block accuracy (only count where mask indicates solid)
        solid_mask = air_true_flat == 1
        if solid_mask.sum() > 0:
            solid_pred = pred_flat[solid_mask]
            solid_true = target_flat[solid_mask]
            solid_correct = (solid_pred == solid_true).sum().item()
            self.total_solid_correct += solid_correct
            self.total_solid_samples += solid_pred.numel()

        # IoU computation (binary solid/air)
        # Intersection = both predict and target are solid/air
        # Union = either predict or target is solid/air
        solid_pred_mask = air_pred_flat == 1
        solid_true_mask = air_true_flat == 1
        air_pred_mask = air_pred_flat == 0
        air_true_mask = air_true_flat == 0

        self.intersection_solid += (solid_pred_mask & solid_true_mask).sum().item()
        self.union_solid += (solid_pred_mask | solid_true_mask).sum().item()
        self.intersection_air += (air_pred_mask & air_true_mask).sum().item()
        self.union_air += (air_pred_mask | air_true_mask).sum().item()

        # Accumulate for confusion matrix (only solid blocks to keep manageable)
        if solid_mask.sum() > 0:
            solid_pred = pred_flat[solid_mask]
            solid_true = target_flat[solid_mask]
            self.confusion_predictions.extend(solid_pred.cpu().numpy().tolist())
            self.confusion_targets.extend(solid_true.cpu().numpy().tolist())

        # Per-LOD tracking
        if lod_indices is not None:
            for i, lod in enumerate(lod_indices):
                lod_val = lod.item()
                if lod_val not in self.lod_metrics:
                    self.lod_metrics[lod_val] = {
                        "correct": 0,
                        "total": 0,
                        "air_correct": 0,
                        "air_total": 0,
                        "solid_correct": 0,
                        "solid_total": 0,
                    }

                # Extract single sample metrics
                sample_pred = pred_indices[i].flatten()
                sample_target = targets[i].flatten()
                sample_air_pred = air_pred_indices[i].flatten()
                sample_air_true = air_true_indices[i].flatten()

                sample_correct = (sample_pred == sample_target).sum().item()
                sample_total = sample_pred.numel()
                sample_air_correct = (sample_air_pred == sample_air_true).sum().item()

                sample_solid_mask = sample_air_true == 1
                sample_solid_correct = 0
                sample_solid_total = 0
                if sample_solid_mask.sum() > 0:
                    sample_solid_pred = sample_pred[sample_solid_mask]
                    sample_solid_true = sample_target[sample_solid_mask]
                    sample_solid_correct = int((sample_solid_pred == sample_solid_true).sum().item())  # type: ignore[assignment]
                    sample_solid_total = sample_solid_pred.numel()

                self.lod_metrics[lod_val]["correct"] += sample_correct
                self.lod_metrics[lod_val]["total"] += sample_total
                self.lod_metrics[lod_val]["air_correct"] += sample_air_correct
                self.lod_metrics[lod_val]["air_total"] += sample_total
                self.lod_metrics[lod_val]["solid_correct"] += sample_solid_correct
                self.lod_metrics[lod_val]["solid_total"] += sample_solid_total

        # Frequent block accuracy (if frequent blocks are known)
        if self.frequent_block_ids is not None:
            # Ensure consistent dtypes for masking
            target_flat = target_flat.long()  # Ensure int64
            frequent_ids = torch.tensor(
                self.frequent_block_ids, device=target_flat.device, dtype=torch.long
            )
            frequent_mask = torch.isin(target_flat, frequent_ids)

            if frequent_mask.sum() > 0:
                frequent_pred = pred_flat[frequent_mask]
                frequent_true = target_flat[frequent_mask]
                frequent_correct = (frequent_pred == frequent_true).sum().item()
                self.frequent_correct += frequent_correct
                self.frequent_total += frequent_pred.numel()
            else:
                logger.debug(
                    f"No frequent blocks found in batch. Target range: "
                    f"{target_flat.min()}-{target_flat.max()}, "
                    f"Frequent IDs: {self.frequent_block_ids[:5]}..."
                )

    def compute_overall_accuracy(self) -> float:
        """Compute overall prediction accuracy."""
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    def compute_air_solid_accuracy(self) -> Tuple[float, float]:
        """Compute air/solid mask accuracy and solid block accuracy."""
        air_acc = self.total_air_correct / max(self.total_air_samples, 1)
        solid_acc = self.total_solid_correct / max(self.total_solid_samples, 1)
        return air_acc, solid_acc

    def compute_iou(self) -> Tuple[float, float]:
        """Compute IoU for solid and air classes."""
        solid_iou = self.intersection_solid / max(self.union_solid, 1)
        air_iou = self.intersection_air / max(self.union_air, 1)
        return solid_iou, air_iou

    def compute_per_lod_metrics(self) -> Dict[int, Dict[str, float]]:
        """Compute per-LOD level metrics."""
        results = {}
        for lod, metrics in self.lod_metrics.items():
            results[lod] = {
                "accuracy": float(metrics["correct"] / max(metrics["total"], 1)),
                "air_accuracy": float(metrics["air_correct"] / max(metrics["air_total"], 1)),
                "solid_accuracy": float(metrics["solid_correct"] / max(metrics["solid_total"], 1)),
            }
        return results

    def compute_frequent_block_accuracy(self) -> float:
        """Compute accuracy on frequent blocks only."""
        if self.frequent_total == 0:
            return 0.0
        return self.frequent_correct / self.frequent_total

    def analyze_confusion_matrix(self, top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze confusion matrix for most confused block types.

        Args:
            top_k: Number of most confused pairs to return

        Returns:
            Confusion analysis dictionary
        """
        if not self.confusion_predictions:
            return {"error": "No confusion data accumulated"}

        # Compute confusion matrix
        cm = confusion_matrix(
            self.confusion_targets,
            self.confusion_predictions,
            labels=list(range(self.block_vocab_size)),
        )

        # Overall accuracy from confusion matrix
        cm_accuracy = np.trace(cm) / max(np.sum(cm), 1)

        # Find most confused pairs (off-diagonal elements)
        confused_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))

        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_confused = confused_pairs[:top_k]

        # Per-class accuracy
        per_class_acc = {}
        for i in range(cm.shape[0]):
            class_total = cm[i, :].sum()
            if class_total > 0:
                per_class_acc[i] = cm[i, i] / class_total

        # Count classes above threshold
        classes_above_threshold = sum(
            1 for acc in per_class_acc.values() if acc >= self.accuracy_threshold
        )

        return {
            "overall_accuracy": cm_accuracy,
            "per_class_accuracy": per_class_acc,
            "top_confused_pairs": top_confused,
            "classes_above_threshold": classes_above_threshold,
            "total_classes_evaluated": len(per_class_acc),
            "threshold": self.accuracy_threshold,
            "goal_achieved": cm_accuracy >= self.accuracy_threshold,
        }

    def determine_frequent_blocks(self, min_samples: int = 100) -> List[int]:
        """
        Determine most frequent block types from accumulated data.

        Args:
            min_samples: Minimum samples required to consider a block frequent

        Returns:
            List of frequent block IDs
        """
        if not self.confusion_targets:
            return []

        # Count occurrences
        unique, counts = np.unique(self.confusion_targets, return_counts=True)
        counts_py: list[int] = counts.tolist()  # convert numpy ints → Python ints

        # Filter by minimum samples and take top K
        frequent_candidates = [
            (block_id, count) for block_id, count in zip(unique, counts_py) if count >= min_samples
        ]
        frequent_candidates.sort(key=lambda x: x[1], reverse=True)

        frequent_blocks = [
            block_id for block_id, _ in frequent_candidates[: self.frequent_blocks_k]
        ]

        logger.info(
            f"Determined {len(frequent_blocks)} frequent blocks from {len(unique)} total types"
        )

        return frequent_blocks

    def export_detailed_report(
        self, output_path: Path, epoch: int, metadata: Optional[Dict] = None
    ):
        """
        Export comprehensive metrics report to JSON.

        Args:
            output_path: Path to save the report
            epoch: Current epoch number
            metadata: Additional metadata to include
        """
        overall_acc = self.compute_overall_accuracy()
        air_acc, solid_acc = self.compute_air_solid_accuracy()
        solid_iou, air_iou = self.compute_iou()
        per_lod = self.compute_per_lod_metrics()
        frequent_acc = self.compute_frequent_block_accuracy()
        confusion_analysis = self.analyze_confusion_matrix()

        # Convert numpy types for JSON serialization
        if confusion_analysis.get("top_confused_pairs"):
            confusion_analysis["top_confused_pairs"] = [
                [int(a), int(b), int(c)] for a, b, c in confusion_analysis["top_confused_pairs"]
            ]

        if confusion_analysis.get("per_class_accuracy"):
            confusion_analysis["per_class_accuracy"] = {
                str(k): float(v) for k, v in confusion_analysis["per_class_accuracy"].items()
            }

        # Convert boolean and numeric types to JSON-safe types
        confusion_analysis["goal_achieved"] = bool(confusion_analysis.get("goal_achieved", False))
        confusion_analysis["overall_accuracy"] = float(
            confusion_analysis.get("overall_accuracy", 0.0)
        )
        confusion_analysis["threshold"] = float(confusion_analysis.get("threshold", 0.99))
        confusion_analysis["classes_above_threshold"] = int(
            confusion_analysis.get("classes_above_threshold", 0)
        )
        confusion_analysis["total_classes_evaluated"] = int(
            confusion_analysis.get("total_classes_evaluated", 0)
        )

        report = {
            "epoch": epoch,
            "timestamp": None,  # Could add timestamp
            "overall_metrics": {
                "accuracy": float(overall_acc),
                "air_accuracy": float(air_acc),
                "solid_accuracy": float(solid_acc),
                "solid_iou": float(solid_iou),
                "air_iou": float(air_iou),
                "frequent_block_accuracy": float(frequent_acc),
                "goal_achieved": bool(float(overall_acc) >= float(self.accuracy_threshold)),
                "accuracy_threshold": float(self.accuracy_threshold),
            },
            "per_lod_metrics": per_lod,
            "confusion_analysis": confusion_analysis,
            "sample_counts": {
                "total_samples": int(self.total_samples),
                "solid_samples": int(self.total_solid_samples),
                "frequent_samples": int(self.frequent_total),
                "confusion_entries": len(self.confusion_predictions),
            },
            "frequent_blocks": [int(x) for x in (self.frequent_block_ids or [])],
            "metadata": metadata or {},
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported metrics report to {output_path}")
        status = "GOAL MET" if overall_acc >= self.accuracy_threshold else "below goal"
        logger.info(f"Overall accuracy: {overall_acc:.4f} ({status})")


class RolloutEvaluator:
    """
    Evaluates model performance on multi-step rollout chains.

    Tests the model's ability to maintain consistency when generating
    multiple LOD levels in sequence (e.g., 8³ -> 16³ -> 32³ -> 64³).
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Initialize rollout evaluator.

        Args:
            model: Trained VoxelTree model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_rollout_chain(
        self,
        initial_parent: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        max_steps: int = 3,
        target_sizes: List[int] = [16, 32, 64],
    ) -> Dict[str, Any]:
        """
        Evaluate a multi-step rollout chain.

        Args:
            initial_parent: (1, 1, 8, 8, 8) initial parent voxel
            conditioning: Conditioning inputs (biome, height, etc.)
            max_steps: Maximum rollout steps
            target_sizes: Target volume sizes for each step

        Returns:
            Rollout evaluation results
        """
        results: Dict[str, Any] = {
            "steps": [],
            "occupancy_progression": [],
            "consistency_metrics": [],
        }

        current_parent = initial_parent.to(self.device)

        with torch.no_grad():
            for step in range(min(max_steps, len(target_sizes))):
                target_size = target_sizes[step]
                lod_index = step  # Could be more sophisticated

                # Prepare conditioning
                step_conditioning = {k: v.to(self.device) for k, v in conditioning.items()}
                step_conditioning["lod"] = torch.tensor([lod_index], device=self.device)

                # Forward pass
                outputs = self.model(parent_voxel=current_parent, **step_conditioning)

                # Extract predictions
                air_mask_logits = outputs["air_mask_logits"]
                block_type_logits = outputs["block_type_logits"]

                # Convert to predictions
                air_pred = (air_mask_logits > 0).float()
                block_pred = torch.argmax(block_type_logits, dim=1)

                # Compute step metrics
                occupancy = air_pred.mean().item()
                unique_blocks = torch.unique(block_pred).numel()

                step_result = {
                    "step": step,
                    "target_size": target_size,
                    "lod_index": lod_index,
                    "occupancy": occupancy,
                    "unique_blocks": unique_blocks,
                    "output_shape": list(air_pred.shape),
                }

                results["steps"].append(step_result)
                results["occupancy_progression"].append(occupancy)

                # Prepare next parent (downsample current prediction)
                if step < max_steps - 1:
                    # Rollout provides only air_pred probabilities, not block IDs, so we
                    # cannot call the Mipper algorithm here (Mipper requires integer block
                    # types for opacity-biased corner selection).  OR-pooling via max_pool3d
                    # is the correct fallback for a probability occupancy tensor.
                    # TODO(milestone-5): thread argmax(block_logits) through rollout so
                    # this step can use mip_volume_torch instead.
                    next_parent = F.max_pool3d(air_pred, kernel_size=2, stride=2)
                    # Resize to canonical 8³ if needed
                    if next_parent.shape[2:] != (8, 8, 8):
                        next_parent = F.interpolate(next_parent, size=(8, 8, 8), mode="nearest")
                    current_parent = next_parent

        # Compute consistency metrics
        if len(results["occupancy_progression"]) > 1:
            occupancy_variance = np.var(results["occupancy_progression"])
            occupancy_trend = np.polyfit(
                range(len(results["occupancy_progression"])), results["occupancy_progression"], 1
            )[0]
            results["consistency_metrics"] = {
                "occupancy_variance": occupancy_variance,
                "occupancy_trend": occupancy_trend,
                "stable_rollout": occupancy_variance < 0.1,  # Arbitrary threshold
            }

        return results


def demo_metrics_framework():
    """Demonstrate the metrics framework with synthetic data."""
    logging.basicConfig(level=logging.INFO)

    # Create synthetic predictions and targets
    batch_size = 4
    spatial_size = 16
    block_vocab_size = 1104

    # Create some structured synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    # Targets: mostly air (0) with some structure
    targets = torch.zeros(batch_size, spatial_size, spatial_size, spatial_size, dtype=torch.long)
    air_mask_true = torch.zeros(batch_size, 1, spatial_size, spatial_size, spatial_size)

    # Add some structure: bottom half has blocks
    targets[:, : spatial_size // 2, :, :] = torch.randint(
        1, 50, (batch_size, spatial_size // 2, spatial_size, spatial_size)
    )
    air_mask_true[:, :, : spatial_size // 2, :, :] = 1.0

    # Predictions: similar but with some errors
    predictions = torch.randn(
        batch_size, block_vocab_size, spatial_size, spatial_size, spatial_size
    )
    # Set correct predictions for most locations
    for b in range(batch_size):
        for i in range(spatial_size):
            for j in range(spatial_size):
                for k in range(spatial_size):
                    true_class = targets[b, i, j, k].item()
                    # 80% chance of correct prediction
                    if np.random.random() < 0.8:
                        predictions[b, true_class, i, j, k] = 5.0  # High logit for correct class

    air_mask_pred = torch.randn(batch_size, 1, spatial_size, spatial_size, spatial_size)
    # Make air predictions mostly correct
    air_mask_pred = air_mask_true + 0.2 * torch.randn_like(air_mask_true)

    lod_indices = torch.randint(0, 5, (batch_size,))

    logger.info("Created synthetic evaluation data")
    logger.info(f"Targets shape: {targets.shape}, Air mask shape: {air_mask_true.shape}")
    logger.info(f"True occupancy: {air_mask_true.mean():.3f}")

    # Initialize metrics
    metrics = VoxelMetrics(block_vocab_size=block_vocab_size, frequent_blocks_k=10)

    # Update with synthetic data
    metrics.update(
        predictions=predictions,
        targets=targets,
        air_mask_pred=air_mask_pred,
        air_mask_true=air_mask_true,
        lod_indices=lod_indices,
    )

    # Compute and display metrics
    overall_acc = metrics.compute_overall_accuracy()
    air_acc, solid_acc = metrics.compute_air_solid_accuracy()
    solid_iou, air_iou = metrics.compute_iou()
    per_lod = metrics.compute_per_lod_metrics()

    logger.info(f"Overall accuracy: {overall_acc:.3f}")
    logger.info(f"Air accuracy: {air_acc:.3f}, Solid accuracy: {solid_acc:.3f}")
    logger.info(f"Solid IoU: {solid_iou:.3f}, Air IoU: {air_iou:.3f}")
    logger.info(f"Per-LOD metrics: {per_lod}")

    # Analyze confusion matrix
    confusion_analysis = metrics.analyze_confusion_matrix(top_k=5)
    logger.info(f"Confusion analysis: {confusion_analysis['overall_accuracy']:.3f} accuracy")
    threshold = metrics.accuracy_threshold
    classes_above = confusion_analysis["classes_above_threshold"]
    logger.info(f"Classes above {threshold}: {classes_above}")

    # Determine frequent blocks
    frequent_blocks = metrics.determine_frequent_blocks(min_samples=5)
    metrics.frequent_block_ids = frequent_blocks
    logger.info(f"Frequent blocks: {frequent_blocks[:5]}...")

    # Export detailed report
    report_path = Path("test_metrics_report.json")
    metrics.export_detailed_report(report_path, epoch=42, metadata={"test": "synthetic_demo"})

    logger.info("Metrics framework demonstration complete!")


if __name__ == "__main__":
    demo_metrics_framework()
