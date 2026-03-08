#!/usr/bin/env python3
"""
VoxelTree Confusion Matrix Analyzer

This module provides comprehensive confusion matrix analysis for the full Minecraft
block vocabulary (1024 blocks), designed to help achieve the 99% accuracy goal
for Phase-1 terrain generation.

Features:
- Full vocabulary confusion matrix (1024x1024)
- Block group analysis (stone types, dirt types, ores, etc.)
- Top-K most confused blocks identification
- Terrain-specific accuracy metrics
- Visualization and reporting tools
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    pass  # keep TYPE_CHECKING import for future use

# Heavy visualisation deps are lazy-imported so that importing this module
# (and therefore trainer.py) works even when seaborn/sklearn/matplotlib
# are not installed — important for fast test suites.
_plt: Any = None
_sns: Any = None
_sk_confusion_matrix: Any = None


def _ensure_viz_deps() -> None:
    """Lazily import matplotlib, seaborn, sklearn on first use."""
    global _plt, _sns, _sk_confusion_matrix
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    if _sns is None:
        import seaborn as sns
        _sns = sns
    if _sk_confusion_matrix is None:
        from sklearn.metrics import confusion_matrix as sk_cm
        _sk_confusion_matrix = sk_cm


logger = logging.getLogger(__name__)


class MinecraftBlockGroups:
    """Defines block groups for analysis of the full Minecraft vocabulary."""

    # Common terrain block groups for analysis
    TERRAIN_GROUPS = {
        "air": [0],  # Air
        "stone_types": list(range(1, 20)),  # Stone, granite, diorite, andesite, etc.
        "dirt_types": list(range(20, 35)),  # Dirt, grass, podzol, etc.
        "sand_types": list(range(35, 45)),  # Sand, sandstone variants
        "water_types": list(range(45, 55)),  # Water, ice, etc.
        "wood_types": list(range(55, 120)),  # All wood types, logs, planks
        "ore_types": list(range(120, 160)),  # All ores
        "deepslate_types": list(range(160, 180)),  # Deepslate variants
        "misc_terrain": list(range(180, 300)),  # Gravel, clay, etc.
        "rare_blocks": list(range(300, 1024)),  # Less common terrain blocks
    }

    @classmethod
    def get_block_group(cls, block_id: int) -> str:
        """Get the group name for a block ID."""
        for group_name, block_ids in cls.TERRAIN_GROUPS.items():
            if block_id in block_ids:
                return group_name
        return "unknown"

    @classmethod
    def get_common_terrain_blocks(cls) -> List[int]:
        """Get the most common terrain blocks for detailed analysis."""
        return (
            cls.TERRAIN_GROUPS["air"]
            + cls.TERRAIN_GROUPS["stone_types"][:10]
            + cls.TERRAIN_GROUPS["dirt_types"][:10]
            + cls.TERRAIN_GROUPS["sand_types"][:5]
            + cls.TERRAIN_GROUPS["water_types"][:5]
        )


class ConfusionAnalyzer:
    """Comprehensive confusion matrix analysis for full vocabulary training."""

    def __init__(self, n_classes: int = 1024, save_dir: Optional[Path] = None):
        """
        Initialize confusion matrix analyzer.

        Args:
            n_classes: Number of block classes (default 1024 for full vocabulary)
            save_dir: Directory to save analysis results
        """
        self.n_classes = n_classes
        self.save_dir = Path(save_dir) if save_dir else Path("analysis")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track accumulated confusion matrix across batches
        self.accumulated_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.total_samples = 0

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update accumulated confusion matrix with new batch.

        Args:
            predictions: Predicted class logits (B, C, D, H, W)
            targets: Target class indices (B, D, H, W)
            mask: Optional mask for solid blocks only (B, 1, D, H, W)
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1)

        # Apply mask if provided (only evaluate solid blocks)
        if mask is not None:
            mask = mask.squeeze(1).bool()  # (B, D, H, W)
            pred_classes = pred_classes[mask]
            targets = targets[mask]
        else:
            pred_classes = pred_classes.flatten()
            targets = targets.flatten()

        # Convert to numpy
        pred_np = pred_classes.cpu().numpy()
        target_np = targets.cpu().numpy()

        # Update accumulated confusion matrix
        _ensure_viz_deps()
        batch_confusion = _sk_confusion_matrix(target_np, pred_np, labels=range(self.n_classes))
        self.accumulated_confusion += batch_confusion
        self.total_samples += len(pred_np)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.accumulated_confusion = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.total_samples = 0

    def compute_overall_accuracy(self) -> float:
        """Compute overall classification accuracy."""
        correct = np.trace(self.accumulated_confusion)
        total = np.sum(self.accumulated_confusion)
        return correct / total if total > 0 else 0.0

    def compute_per_class_accuracy(self) -> Dict[int, float]:
        """Compute per-class accuracy for all block types."""
        per_class_acc = {}

        for i in range(self.n_classes):
            class_total = np.sum(self.accumulated_confusion[i, :])
            class_correct = self.accumulated_confusion[i, i]

            if class_total > 0:
                per_class_acc[i] = class_correct / class_total
            else:
                per_class_acc[i] = float("nan")

        return per_class_acc

    def get_top_confused_pairs(self, k: int = 20) -> List[Tuple[int, int, int]]:
        """
        Get the top K most confused block pairs.

        Args:
            k: Number of confused pairs to return

        Returns:
            List of (true_class, pred_class, count) tuples
        """
        confused_pairs = []

        # Find all off-diagonal elements (misclassifications)
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and self.accumulated_confusion[i, j] > 0:
                    confused_pairs.append((i, j, self.accumulated_confusion[i, j]))

        # Sort by confusion count and return top K
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs[:k]

    def analyze_block_groups(
        self,
    ) -> dict[str, dict[str, Any | float | np.floating]]:
        """Analyze accuracy by block groups (stone, dirt, ores, etc.)."""
        group_stats = {}

        for group_name, block_ids in MinecraftBlockGroups.TERRAIN_GROUPS.items():
            # Get confusion submatrix for this group
            group_confusion = self.accumulated_confusion[np.ix_(block_ids, block_ids)]

            # Calculate group metrics
            group_correct = np.trace(group_confusion)
            group_total = np.sum(group_confusion)
            group_accuracy = group_correct / group_total if group_total > 0 else 0.0

            # Calculate group representation in dataset
            group_samples = np.sum(self.accumulated_confusion[block_ids, :])
            group_percentage = group_samples / self.total_samples if self.total_samples > 0 else 0.0

            group_stats[group_name] = {
                "accuracy": group_accuracy,
                "samples": int(group_samples),
                "percentage": group_percentage,
                "blocks_in_group": len(block_ids),
            }

        return group_stats

    def get_worst_performing_blocks(self, k: int = 50) -> list[tuple[int, float, np.int64]]:
        """
        Get the K worst performing block types.

        Args:
            k: Number of worst blocks to return

        Returns:
            List of (block_id, accuracy, sample_count) tuples
        """
        per_class_acc = self.compute_per_class_accuracy()

        # Filter out blocks with no samples and sort by accuracy
        valid_blocks = [
            (block_id, acc, np.sum(self.accumulated_confusion[block_id, :]))
            for block_id, acc in per_class_acc.items()
            if not np.isnan(acc) and np.sum(self.accumulated_confusion[block_id, :]) > 0
        ]

        valid_blocks.sort(key=lambda x: x[1])  # Sort by accuracy (ascending)
        return valid_blocks[:k]

    def visualize_confusion_matrix(
        self,
        blocks_to_show: Optional[Sequence[int]] = None,
        title: str = "Confusion Matrix",
    ) -> Any:
        """
        Create confusion matrix visualization.

        Args:
            blocks_to_show: Specific block IDs to show, defaults to common terrain blocks
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if blocks_to_show is None:
            blocks_to_show = MinecraftBlockGroups.get_common_terrain_blocks()

        # Extract submatrix for visualization
        confusion_subset = self.accumulated_confusion[np.ix_(blocks_to_show, blocks_to_show)]

        # Normalize by row (true class)
        row_sums = confusion_subset.sum(axis=1, keepdims=True)
        normalized_confusion = np.divide(
            confusion_subset,
            row_sums,
            out=np.zeros_like(confusion_subset, dtype=float),
            where=row_sums != 0,
        )

        # Create visualization
        _ensure_viz_deps()
        fig, ax = _plt.subplots(figsize=(12, 10))
        _sns.heatmap(
            normalized_confusion,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=blocks_to_show,
            yticklabels=blocks_to_show,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted Block ID")
        ax.set_ylabel("True Block ID")

        _plt.tight_layout()
        return fig

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        overall_acc = self.compute_overall_accuracy()
        per_class_acc = self.compute_per_class_accuracy()
        group_stats = self.analyze_block_groups()
        top_confused = self.get_top_confused_pairs(10)
        worst_blocks = self.get_worst_performing_blocks(20)

        # Calculate summary statistics
        valid_accuracies = [acc for acc in per_class_acc.values() if not np.isnan(acc)]
        mean_class_acc = np.mean(valid_accuracies) if valid_accuracies else 0.0
        min_class_acc = np.min(valid_accuracies) if valid_accuracies else 0.0
        blocks_above_99 = sum(1 for acc in valid_accuracies if acc >= 0.99)
        blocks_evaluated = len(valid_accuracies)

        report = f"""
VoxelTree Confusion Matrix Analysis Report
==========================================

Overall Performance:
- Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)
- Mean Per-Class Accuracy: {mean_class_acc:.4f} ({mean_class_acc*100:.2f}%)
- Minimum Class Accuracy: {min_class_acc:.4f} ({min_class_acc*100:.2f}%)
- Total Samples Evaluated: {self.total_samples:,}

Phase-1 Goal Progress:
- Target: 99% accuracy
- Current: {overall_acc*100:.2f}%
- Gap to Goal: {99.0 - overall_acc*100:.2f}%
- Blocks with >=99% Accuracy: {blocks_above_99}/{blocks_evaluated} ({blocks_above_99/blocks_evaluated*100:.1f}%)

Block Group Performance:
"""

        for group_name, stats in group_stats.items():
            report += f"- {group_name}: {stats['accuracy']*100:.2f}% ({stats['samples']:,} samples, {stats['percentage']*100:.1f}% of dataset)\n"

        report += """

Top 10 Most Confused Block Pairs:
"""
        for i, (true_id, pred_id, count) in enumerate(top_confused, 1):
            report += f"{i:2d}. True:{true_id:3d} → Pred:{pred_id:3d} ({count:,} errors)\n"

        report += """

Bottom 20 Worst Performing Blocks:
"""
        for i, (block_id, acc, samples) in enumerate(worst_blocks, 1):
            report += f"{i:2d}. Block {block_id:3d}: {acc*100:.2f}% ({samples:,} samples)\n"

        return report

    def save_analysis(self, epoch: int, prefix: str = "confusion_analysis") -> None:
        """
        Save complete analysis to files.

        Args:
            epoch: Current training epoch
            prefix: Filename prefix
        """
        timestamp = epoch

        # Save confusion matrix
        np.save(
            self.save_dir / f"{prefix}_confusion_matrix_epoch_{timestamp}.npy",
            self.accumulated_confusion,
        )

        # Save analysis report
        report = self.generate_report()
        with open(
            self.save_dir / f"{prefix}_report_epoch_{timestamp}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(report)

        # Save per-class accuracies as CSV
        per_class_acc = self.compute_per_class_accuracy()
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "block_id": bid,
                    "accuracy": acc,
                    "group": MinecraftBlockGroups.get_block_group(bid),
                }
                for bid, acc in per_class_acc.items()
                if not np.isnan(acc)
            ]
        )
        df.to_csv(self.save_dir / f"{prefix}_per_class_acc_epoch_{timestamp}.csv", index=False)

        # Save confusion matrix visualization
        fig = self.visualize_confusion_matrix(title=f"Confusion Matrix - Epoch {timestamp}")
        fig.savefig(
            self.save_dir / f"{prefix}_confusion_viz_epoch_{timestamp}.png",
            dpi=150,
            bbox_inches="tight",
        )
        _ensure_viz_deps()
        _plt.close(fig)

        logger.info(f"Confusion analysis saved to {self.save_dir}")

    def is_99_percent_achieved(self) -> Tuple[bool, float]:
        """
        Check if 99% accuracy goal is achieved.

        Returns:
            Tuple of (goal_achieved, current_accuracy)
        """
        overall_acc = self.compute_overall_accuracy()
        return overall_acc >= 0.99, overall_acc


def create_confusion_analyzer(config: Dict, save_dir: Optional[Path] = None) -> ConfusionAnalyzer:
    """
    Factory function to create confusion analyzer from config.

    Args:
        config: Training configuration
        save_dir: Directory to save analysis results

    Returns:
        Configured ConfusionAnalyzer
    """
    model_config = config.get("model", {})
    n_classes = model_config.get("block_type_channels", 1024)

    if save_dir is None:
        save_dir = (
            Path(config.get("training", {}).get("checkpoint_dir", "runs")) / "confusion_analysis"
        )

    return ConfusionAnalyzer(n_classes=n_classes, save_dir=save_dir)
