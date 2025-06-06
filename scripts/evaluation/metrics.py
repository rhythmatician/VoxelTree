"""
VoxelTree Evaluation Metrics Module

This module provides comprehensive evaluation tools for the VoxelTree model,
including accuracy metrics, IoU/Dice scores, and structure-aware evaluation.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class AccuracyMetrics:
    """Base class for computing accuracy metrics."""

    def __init__(self):
        """Initialize accuracy metrics calculator."""
        pass

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute comprehensive accuracy metrics.

        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Compute mask accuracy if available
        if "air_mask_logits" in predictions and "air_mask" in targets:
            mask_calc = MaskAccuracyCalculator()
            mask_metrics = mask_calc.calculate_accuracy(
                predictions["air_mask_logits"], targets["air_mask"]
            )
            metrics.update(mask_metrics)

        # Compute block type accuracy if available
        if "block_type_logits" in predictions and "block_types" in targets:
            block_calc = BlockTypeAccuracyCalculator()
            block_metrics = block_calc.calculate_accuracy(
                predictions["block_type_logits"], targets["block_types"]
            )
            metrics.update(
                block_metrics
            )  # Compute structure accuracy if any structure data is available
        has_structure_mask = "structure_mask" in predictions and "structure_mask" in targets
        has_structure_types = "structure_types" in predictions and "structure_types" in targets

        if has_structure_mask or has_structure_types:
            struct_calc = StructureAccuracyCalculator()
            struct_metrics = struct_calc.calculate_accuracy(predictions, targets)
            metrics.update(struct_metrics)

        return metrics


class MaskAccuracyCalculator:
    """Calculator for air mask accuracy metrics."""

    def calculate_accuracy(
        self, mask_logits: torch.Tensor, target_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate mask prediction accuracy.

        Args:
            mask_logits: Predicted mask logits (B, 1, H, W, D)
            target_mask: Target mask (B, 1, H, W, D) or (B, H, W, D)

        Returns:
            Dictionary with mask accuracy metrics
        """
        # Convert logits to predictions
        mask_probs = torch.sigmoid(mask_logits)
        mask_preds = (mask_probs > 0.5).float()

        # Ensure target mask has same shape
        if target_mask.dim() == 4:
            target_mask = target_mask.unsqueeze(1)
        target_mask = target_mask.float()

        # Calculate accuracy
        correct = (mask_preds == target_mask).float()
        accuracy = correct.mean().item()

        # Calculate precision, recall, F1
        true_positive = (mask_preds * target_mask).sum().item()
        predicted_positive = mask_preds.sum().item()
        actual_positive = target_mask.sum().item()

        precision = true_positive / max(predicted_positive, 1e-8)
        recall = true_positive / max(actual_positive, 1e-8)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

        # Calculate IoU for air/solid classification
        intersection = (mask_preds * target_mask).sum().item()
        union = (mask_preds + target_mask - mask_preds * target_mask).sum().item()
        iou = intersection / max(union, 1e-8)

        return {
            "mask_accuracy": accuracy,
            "mask_precision": precision,
            "mask_recall": recall,
            "mask_f1": f1_score,
            "mask_iou": iou,
        }


class BlockTypeAccuracyCalculator:
    """Calculator for block type accuracy metrics."""

    def calculate_accuracy(
        self, type_logits: torch.Tensor, target_types: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate block type prediction accuracy.

        Args:
            type_logits: Predicted type logits (B, C, H, W, D)
            target_types: Target types (B, H, W, D)

        Returns:
            Dictionary with block type accuracy metrics
        """
        # Get predicted classes
        type_preds = torch.argmax(type_logits, dim=1)  # (B, H, W, D)

        # Calculate overall accuracy
        correct = (type_preds == target_types).float()
        accuracy = correct.mean().item()

        # Calculate per-class accuracy
        num_classes = type_logits.shape[1]
        per_class_acc = {}

        for class_idx in range(num_classes):
            class_mask = target_types == class_idx
            if class_mask.sum() > 0:
                class_correct = (type_preds == target_types)[class_mask]
                per_class_acc[f"class_{class_idx}_accuracy"] = class_correct.float().mean().item()

        # Calculate top-k accuracy (top-3)
        top3_preds = torch.topk(type_logits, k=3, dim=1)[1]  # (B, 3, H, W, D)
        target_expanded = target_types.unsqueeze(1).expand_as(top3_preds)
        top3_correct = (top3_preds == target_expanded).any(dim=1).float()
        top3_accuracy = top3_correct.mean().item()

        result = {
            "block_type_accuracy": accuracy,
            "block_type_top3_accuracy": top3_accuracy,
        }
        result.update(per_class_acc)

        return result


class StructureAccuracyCalculator:
    """Calculator for structure-aware accuracy metrics."""

    def calculate_accuracy(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Calculate structure prediction accuracy.

        Args:
            predictions: Dictionary with structure predictions
            targets: Dictionary with structure targets

        Returns:
            Dictionary with structure accuracy metrics
        """
        metrics = {}

        # Structure mask accuracy with enhanced metrics
        if "structure_mask" in predictions and "structure_mask" in targets:
            pred_mask = predictions["structure_mask"]
            target_mask = targets["structure_mask"]

            # Binarize predictions
            pred_binary = (pred_mask > 0.5).float()
            target_binary = target_mask.float()

            # Calculate accuracy
            correct = (pred_binary == target_binary).float()
            mask_accuracy = correct.mean().item()

            # Calculate IoU for structure regions
            intersection = (pred_binary * target_binary).sum()
            union = (pred_binary + target_binary - pred_binary * target_binary).sum()
            structure_iou = (intersection / torch.clamp(union, min=1e-8)).item()

            # Enhanced: Add precision, recall, F1 for structure mask
            true_positive = (pred_binary * target_binary).sum().item()
            predicted_positive = pred_binary.sum().item()
            actual_positive = target_binary.sum().item()

            precision = true_positive / max(predicted_positive, 1e-8)
            recall = true_positive / max(actual_positive, 1e-8)
            f1_score = 2 * precision * recall / max(precision + recall, 1e-8)

            metrics.update(
                {
                    "structure_mask_accuracy": mask_accuracy,
                    "structure_iou": structure_iou,
                    "structure_mask_precision": precision,
                    "structure_mask_recall": recall,
                    "structure_mask_f1": f1_score,
                }
            )

        # Structure type accuracy with top-k support
        if "structure_types" in predictions and "structure_types" in targets:
            pred_types = predictions["structure_types"]
            target_types = targets["structure_types"]

            # Handle both logits and one-hot cases
            if pred_types.dim() > 1 and pred_types.shape[-1] > 1:
                # Multi-class case with logits or probabilities
                if pred_types.dim() == 2 and target_types.dim() == 2:
                    # One-hot encoded case
                    pred_classes = torch.argmax(pred_types, dim=1)
                    target_classes = torch.argmax(target_types, dim=1)
                else:
                    # Logits case - assume last dim is classes
                    pred_classes = torch.argmax(pred_types, dim=-1)
                    if target_types.dim() == pred_types.dim():
                        target_classes = torch.argmax(target_types, dim=-1)
                    else:
                        target_classes = target_types

                # Calculate accuracy
                type_accuracy = (pred_classes == target_classes).float().mean().item()
                metrics["structure_type_accuracy"] = type_accuracy

                # Add top-k accuracy if we have logits
                if pred_types.shape[-1] >= 3:  # At least 3 classes for top-3
                    top3_preds = torch.topk(pred_types, k=3, dim=-1)[1]
                    target_expanded = target_classes.unsqueeze(-1).expand_as(top3_preds)
                    top3_correct = (top3_preds == target_expanded).any(dim=-1).float()
                    metrics["structure_type_top3_accuracy"] = top3_correct.mean().item()

        # Enhanced: Structure presence detection with FP/FN analysis
        if "structure_mask" in predictions and "structure_mask" in targets:
            pred_mask = predictions["structure_mask"]
            target_mask = targets["structure_mask"]

            # Check if any structure is present (per batch item)
            pred_present = (pred_mask.sum(dim=tuple(range(1, pred_mask.dim()))) > 0.5).float()
            target_present = (target_mask.sum(dim=tuple(range(1, target_mask.dim()))) > 0.5).float()

            # Overall presence accuracy
            presence_accuracy = (pred_present == target_present).float().mean().item()
            metrics["structure_presence_accuracy"] = presence_accuracy

            # False positive/negative analysis for presence detection
            false_positives = ((pred_present == 1) & (target_present == 0)).float().sum().item()
            false_negatives = ((pred_present == 0) & (target_present == 1)).float().sum().item()
            true_positives = ((pred_present == 1) & (target_present == 1)).float().sum().item()
            true_negatives = ((pred_present == 0) & (target_present == 0)).float().sum().item()

            total_samples = len(pred_present)

            metrics.update(
                {
                    "structure_presence_false_positive_rate": false_positives
                    / max(total_samples, 1),
                    "structure_presence_false_negative_rate": false_negatives
                    / max(total_samples, 1),
                    "structure_presence_true_positive_rate": true_positives / max(total_samples, 1),
                    "structure_presence_true_negative_rate": true_negatives / max(total_samples, 1),
                }
            )

        return metrics


class IoUCalculator:
    """Calculator for Intersection over Union (IoU) metrics."""

    def calculate_iou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate IoU metrics for semantic segmentation.

        Args:
            predictions: Predicted logits or probabilities
            targets: Target labels
            num_classes: Number of classes (inferred if None)

        Returns:
            Dictionary with IoU metrics
        """
        if predictions.dim() > targets.dim():
            # Convert logits to class predictions
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions

        if num_classes is None:
            num_classes = max(pred_classes.max().item(), targets.max().item()) + 1

        ious = []
        for class_idx in range(num_classes):
            pred_mask = pred_classes == class_idx
            target_mask = targets == class_idx

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union > 0:
                iou = intersection / union
                ious.append(iou.item())

        mean_iou = np.mean(ious) if ious else 0.0

        return {
            "mean_iou": mean_iou,
            "per_class_iou": {f"class_{i}_iou": iou for i, iou in enumerate(ious)},
        }


class DiceCalculator:
    """Calculator for Dice coefficient metrics."""

    def calculate_dice(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-8,
    ) -> Dict[str, float]:
        """
        Calculate Dice coefficient for binary or multi-class segmentation.

        Args:
            predictions: Predicted logits, probabilities, or binary masks
            targets: Target masks or class indices
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dictionary with Dice coefficient metrics
        """
        if predictions.dim() > targets.dim():
            # Multi-class case: convert logits to probabilities
            pred_probs = F.softmax(predictions, dim=1)
            num_classes = predictions.shape[1]

            dice_scores = []
            for class_idx in range(num_classes):
                pred_class = pred_probs[:, class_idx]
                target_class = (targets == class_idx).float()

                intersection = (pred_class * target_class).sum()
                dice = (2 * intersection + smooth) / (
                    pred_class.sum() + target_class.sum() + smooth
                )
                dice_scores.append(dice.item())

            mean_dice = np.mean(dice_scores)

            return {
                "mean_dice": mean_dice,
                "per_class_dice": {f"class_{i}_dice": dice for i, dice in enumerate(dice_scores)},
            }
        else:
            # Binary case
            pred_binary = (predictions > 0.5).float()
            target_binary = targets.float()

            intersection = (pred_binary * target_binary).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)

            return {"dice_coefficient": dice.item()}
