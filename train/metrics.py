"""
VoxelTree metrics for model evaluation.

Implements evaluation metrics for voxel-based models, including:
- IoU (Intersection over Union)
- Dice coefficient
- Per-class accuracy
- Confusion matrix analysis
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class VoxelMetrics:
    """Evaluation metrics for VoxelTree model predictions."""

    @staticmethod
    def binary_iou(
        pred_mask: Union[np.ndarray, Tensor],
        target_mask: Union[np.ndarray, Tensor],
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> float:
        """
        Calculate binary IoU (Intersection over Union) for air/solid masks.

        Args:
            pred_mask: Predicted mask logits or probabilities
            target_mask: Target binary mask
            threshold: Threshold to convert predictions to binary
            eps: Small value to avoid division by zero

        Returns:
            IoU score in range [0, 1]
        """
        # Convert to binary tensors
        if isinstance(pred_mask, np.ndarray):
            pred_mask = torch.from_numpy(pred_mask)
        if isinstance(target_mask, np.ndarray):
            target_mask = torch.from_numpy(target_mask)

        # Apply threshold to predictions
        pred_binary = (pred_mask > threshold).float()
        target_binary = target_mask.float()

        # Calculate intersection and union
        intersection = (pred_binary * target_binary).sum().float()
        union = (pred_binary + target_binary).sum().float() - intersection

        # IoU = intersection / union
        iou = intersection / (union + eps)

        return iou.item()

    @staticmethod
    def batch_binary_iou(
        pred_mask: Tensor,
        target_mask: Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> Tensor:
        """
        Calculate binary IoU for each example in batch.

        Args:
            pred_mask: Predicted mask logits or probabilities (B, 1, D, H, W)
            target_mask: Target binary mask (B, 1, D, H, W)
            threshold: Threshold to convert predictions to binary
            eps: Small value to avoid division by zero

        Returns:
            IoU scores for batch (B,)
        """
        # Apply threshold to predictions
        pred_binary = (pred_mask > threshold).float()
        target_binary = target_mask.float()

        # Calculate intersection and union for each item in batch
        intersection = (pred_binary * target_binary).flatten(1).sum(dim=1)
        union = (pred_binary + target_binary).flatten(1).sum(dim=1) - intersection

        # IoU = intersection / union
        iou = intersection / (union + eps)

        return iou

    @staticmethod
    def dice_coefficient(
        pred_mask: Union[np.ndarray, Tensor],
        target_mask: Union[np.ndarray, Tensor],
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> float:
        """
        Calculate Dice coefficient for mask prediction.

        Args:
            pred_mask: Predicted mask logits or probabilities
            target_mask: Target binary mask
            threshold: Threshold to convert predictions to binary
            eps: Small value to avoid division by zero

        Returns:
            Dice coefficient in range [0, 1]
        """
        # Convert to binary tensors
        if isinstance(pred_mask, np.ndarray):
            pred_mask = torch.from_numpy(pred_mask)
        if isinstance(target_mask, np.ndarray):
            target_mask = torch.from_numpy(target_mask)

        # Apply threshold to predictions
        pred_binary = (pred_mask > threshold).float()
        target_binary = target_mask.float()

        # Calculate intersection
        intersection = (pred_binary * target_binary).sum().float() * 2.0

        # Calculate sum of areas
        sum_areas = pred_binary.sum() + target_binary.sum()

        # Dice = (2 * intersection) / sum of areas
        dice = intersection / (sum_areas + eps)

        return dice.item()

    @staticmethod
    def batch_dice_coefficient(
        pred_mask: Tensor,
        target_mask: Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> Tensor:
        """
        Calculate Dice coefficient for each example in batch.

        Args:
            pred_mask: Predicted mask logits or probabilities (B, 1, D, H, W)
            target_mask: Target binary mask (B, 1, D, H, W)
            threshold: Threshold to convert predictions to binary
            eps: Small value to avoid division by zero

        Returns:
            Dice scores for batch (B,)
        """
        # Apply threshold to predictions
        pred_binary = (pred_mask > threshold).float()
        target_binary = target_mask.float()

        # Calculate intersection for each item in batch
        intersection = (pred_binary * target_binary).flatten(1).sum(dim=1) * 2.0

        # Calculate sum of areas for each item in batch
        sum_areas = pred_binary.flatten(1).sum(dim=1) + target_binary.flatten(1).sum(dim=1)

        # Dice = (2 * intersection) / sum of areas
        dice = intersection / (sum_areas + eps)

        return dice

    @staticmethod
    def per_class_accuracy(
        pred_input: Union[np.ndarray, Tensor, Dict[str, Any]],
        target_classes: Optional[Union[np.ndarray, Tensor]] = None,
        n_classes: Optional[int] = None,
        mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Union[Dict[int, float], Tensor]:
        """
        Calculate per-class accuracy for block type prediction.

        This method can be called in two ways:
        1. With confusion matrix: per_class_accuracy(confusion_matrix)
        2. With predictions and targets: per_class_accuracy(pred_logits, target_classes, n_classes, mask)

        Args:
            pred_input: Either confusion matrix or predicted class logits (C, D, H, W) or (B, C, D, H, W)
            target_classes: Target class indices (D, H, W) or (B, D, H, W) if not using confusion matrix
            n_classes: Number of classes if not using confusion matrix
            mask: Optional mask to only evaluate solid blocks

        Returns:
            Dictionary mapping class indices to accuracy scores, or tensor of per-class accuracies
        """
        # Case 1: Confusion matrix input
        if target_classes is None and isinstance(pred_input, (torch.Tensor, np.ndarray)):
            # We got a confusion matrix
            if isinstance(pred_input, np.ndarray):
                conf_matrix = torch.from_numpy(pred_input)
            else:
                conf_matrix = pred_input

            # Calculate per-class accuracy from confusion matrix
            # Each row in confusion matrix represents a true class
            # and the entries in that row are predictions to each class
            class_correct = torch.diag(conf_matrix)  # Number of correct predictions for each class
            class_total = conf_matrix.sum(dim=1)  # Total samples for each class

            # Handle zero division
            eps = torch.finfo(conf_matrix.dtype).eps
            per_class_acc = class_correct / torch.clamp(class_total, min=eps)

            return per_class_acc

        # Case 2: Predictions and targets input
        # Convert to tensors
        if isinstance(pred_input, np.ndarray):
            pred_logits = torch.from_numpy(pred_input)
        else:
            pred_logits = pred_input

        if isinstance(target_classes, np.ndarray):
            target_classes = torch.from_numpy(target_classes)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Get predicted classes
        if len(pred_logits.shape) >= 4:  # Batch mode (4D or 5D)
            pred_classes = torch.argmax(pred_logits, dim=1)
        else:  # Single example mode (3D)
            pred_classes = torch.argmax(pred_logits, dim=0)

        # Initialize result
        class_accuracy = {}

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) >= 4 and mask.shape[1] == 1:  # Batch mode with mask (B, 1, ...)
                mask = mask.squeeze(1).bool()
            elif len(mask.shape) >= 3:  # Single example mode with mask (1, ...)
                mask = mask.squeeze(0).bool()
            else:
                mask = mask.bool()

        # Calculate accuracy for each class
        for c in range(n_classes):
            # Create mask for current class
            class_mask = target_classes == c

            # Apply solid mask if provided
            if mask is not None:
                class_mask = class_mask & mask

            # Skip if no examples of this class
            if class_mask.sum() == 0:
                class_accuracy[int(c)] = float("nan")
                continue

            # Count correct predictions for this class
            correct = ((pred_classes == c) & class_mask).sum()
            total = class_mask.sum()

            # Calculate accuracy
            accuracy = correct.float() / total.float()
            class_accuracy[int(c)] = accuracy.item()

        return class_accuracy

    @staticmethod
    def confusion_matrix(
        pred_logits: Union[np.ndarray, Tensor],
        target_classes: Union[np.ndarray, Tensor],
        n_classes: int,
        mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> np.ndarray:
        """
        Calculate confusion matrix for block type prediction.

        Args:
            pred_logits: Predicted class logits (C, D, H, W) or (B, C, D, H, W)
            target_classes: Target class indices (D, H, W) or (B, D, H, W)
            n_classes: Number of classes
            mask: Optional mask to only evaluate solid blocks

        Returns:
            Confusion matrix as numpy array (n_classes, n_classes)
        """
        # Convert to tensors
        if isinstance(pred_logits, np.ndarray):
            pred_logits = torch.from_numpy(pred_logits)
        if isinstance(target_classes, np.ndarray):
            target_classes = torch.from_numpy(target_classes)
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Get predicted classes
        if len(pred_logits.shape) == 5:  # Batch mode
            pred_classes = torch.argmax(pred_logits, dim=1)
        else:  # Single example mode
            pred_classes = torch.argmax(pred_logits, dim=0)

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 5:  # Batch mode with mask (B, 1, D, H, W)
                mask = mask.squeeze(1).bool()
            else:  # Single example mode with mask (1, D, H, W)
                mask = mask.squeeze(0).bool()

            # Apply mask
            pred_classes_masked = pred_classes[mask]
            target_classes_masked = target_classes[mask]
        else:
            pred_classes_masked = pred_classes.flatten()
            target_classes_masked = target_classes.flatten()

        # Initialize confusion matrix
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

        # Convert to numpy for easier handling
        pred_np = pred_classes_masked.cpu().numpy()
        target_np = target_classes_masked.cpu().numpy()

        # Build confusion matrix
        for i in range(len(pred_np)):
            confusion[target_np[i], pred_np[i]] += 1

        return confusion

    @staticmethod
    def compute_confusion_matrix(
        pred_classes: Union[np.ndarray, Tensor],
        target_classes: Union[np.ndarray, Tensor],
        n_classes: int,
    ) -> Tensor:
        """
        Calculate confusion matrix from predicted and target class indices.

        Args:
            pred_classes: Predicted class indices (B, D, H, W) or (D, H, W)
            target_classes: Target class indices (B, D, H, W) or (D, H, W)
            n_classes: Number of classes

        Returns:
            Confusion matrix as tensor (n_classes, n_classes)
        """
        # Convert to tensors
        if isinstance(pred_classes, np.ndarray):
            pred_classes = torch.from_numpy(pred_classes)
        if isinstance(target_classes, np.ndarray):
            target_classes = torch.from_numpy(target_classes)

        # Flatten predictions and targets
        pred_flat = pred_classes.flatten()
        target_flat = target_classes.flatten()

        # Initialize confusion matrix
        confusion = torch.zeros(
            n_classes, n_classes, dtype=torch.float32, device=pred_classes.device
        )

        # Update confusion matrix
        for i in range(pred_flat.shape[0]):
            p_class = pred_flat[i].item()
            t_class = target_flat[i].item()
            confusion[t_class, p_class] += 1

        return confusion

    @staticmethod
    def compute_all_metrics(
        pred_mask_logits: Tensor,
        pred_type_logits: Tensor,
        target_mask: Tensor,
        target_types: Tensor,
        n_classes: int,
        mask_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all metrics for model evaluation.

        Args:
            pred_mask_logits: Predicted mask logits (B, 1, D, H, W)
            pred_type_logits: Predicted type logits (B, C, D, H, W)
            target_mask: Target binary mask (B, 1, D, H, W)
            target_types: Target class indices (B, D, H, W)
            n_classes: Number of classes
            mask_threshold: Threshold to convert mask predictions to binary

        Returns:
            Dictionary of all metrics
        """
        # Apply sigmoid to mask logits
        pred_mask_probs = torch.sigmoid(pred_mask_logits)

        # Calculate IoU and Dice for mask
        batch_iou = VoxelMetrics.batch_binary_iou(pred_mask_probs, target_mask, mask_threshold)
        batch_dice = VoxelMetrics.batch_dice_coefficient(
            pred_mask_probs, target_mask, mask_threshold
        )

        # Calculate class accuracy and confusion matrix
        class_acc = VoxelMetrics.per_class_accuracy(
            pred_type_logits, target_types, n_classes, target_mask
        )

        # Compute mean metrics
        mean_iou = batch_iou.mean().item()
        mean_dice = batch_dice.mean().item()

        # Get valid class accuracies (not NaN)
        valid_accs = [acc for acc in class_acc.values() if not np.isnan(acc)]
        mean_class_acc = np.mean(valid_accs) if valid_accs else 0.0

        # Convert class accuracies to tensor for per-class output
        per_class_tensor = torch.zeros(n_classes)
        for class_idx, acc in class_acc.items():
            if not np.isnan(acc):
                per_class_tensor[class_idx] = acc

        metrics = {
            "mask_iou": mean_iou,
            "mask_dice": mean_dice,
            "type_accuracy": mean_class_acc,
            "type_accuracy_per_class": per_class_tensor,
            "combined_score": (mean_iou + mean_dice + mean_class_acc) / 3,
        }

        return metrics
