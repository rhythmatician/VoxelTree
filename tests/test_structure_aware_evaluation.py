"""
Comprehensive test suite for structure-aware evaluation metrics.

This test suite validates the enhanced evaluation metrics module with
mock structure-aware predictions for Phase 2 validation.
"""

import numpy as np
import pytest
import torch

from scripts.evaluation.metrics import (
    AccuracyMetrics,
    BlockTypeAccuracyCalculator,
    DiceCalculator,
    IoUCalculator,
    MaskAccuracyCalculator,
    StructureAccuracyCalculator,
)


class TestStructureAwareEvaluationMetrics:
    """Test suite for structure-aware evaluation metrics."""

    def setup_method(self):
        """Set up test environment."""
        self.batch_size = 4
        self.spatial_dims = (8, 8, 16)  # H, W, D
        self.num_classes = 10
        self.num_structure_types = 5

        # Initialize calculators
        self.accuracy_metrics = AccuracyMetrics()
        self.structure_calc = StructureAccuracyCalculator()
        self.iou_calc = IoUCalculator()
        self.dice_calc = DiceCalculator()

    def _create_mock_predictions(self) -> dict:
        """Create mock model predictions for testing."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        return {
            "air_mask_logits": torch.randn(B, 1, H, W, D),
            "block_type_logits": torch.randn(B, self.num_classes, H, W, D),
            "structure_mask": torch.sigmoid(torch.randn(B, 1, H, W, D)),
            "structure_types": torch.randn(B, self.num_structure_types),  # Logits
            "structure_positions": torch.randn(B, 3),  # x, y, z offsets
        }

    def _create_mock_targets(self) -> dict:
        """Create mock ground truth targets for testing."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        return {
            "air_mask": torch.randint(0, 2, (B, H, W, D)).float(),
            "block_types": torch.randint(0, self.num_classes, (B, H, W, D)),
            "structure_mask": torch.randint(0, 2, (B, 1, H, W, D)).float(),
            "structure_types": torch.eye(self.num_structure_types)[
                torch.randint(0, self.num_structure_types, (B,))
            ],  # One-hot encoded
        }

    def test_enhanced_structure_accuracy_precision_recall(self):
        """Test enhanced structure accuracy with precision/recall/F1."""
        predictions = self._create_mock_predictions()
        targets = self._create_mock_targets()

        metrics = self.structure_calc.calculate_accuracy(predictions, targets)

        # Check that enhanced metrics are present
        expected_keys = [
            "structure_mask_accuracy",
            "structure_iou",
            "structure_mask_precision",
            "structure_mask_recall",
            "structure_mask_f1",
            "structure_type_accuracy",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float), f"Metric {key} should be float"
            assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} should be in [0,1]"

    def test_structure_presence_false_positive_negative_analysis(self):
        """Test enhanced presence detection with FP/FN analysis."""
        predictions = self._create_mock_predictions()
        targets = self._create_mock_targets()

        # Create specific test case: some with structures, some without
        predictions["structure_mask"][0] = torch.ones_like(predictions["structure_mask"][0]) * 0.8  # Present
        predictions["structure_mask"][1] = torch.zeros_like(predictions["structure_mask"][1])  # Absent
        
        targets["structure_mask"][0] = torch.ones_like(targets["structure_mask"][0])  # True present
        targets["structure_mask"][1] = torch.ones_like(targets["structure_mask"][1])  # True present (FN case)

        metrics = self.structure_calc.calculate_accuracy(predictions, targets)

        # Check FP/FN metrics
        fp_fn_keys = [
            "structure_presence_false_positive_rate",
            "structure_presence_false_negative_rate", 
            "structure_presence_true_positive_rate",
            "structure_presence_true_negative_rate",
        ]

        for key in fp_fn_keys:
            assert key in metrics, f"Missing FP/FN metric: {key}"
            assert isinstance(metrics[key], float), f"Metric {key} should be float"
            assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} should be in [0,1]"

        # Rates should sum to 1.0 (accounting for floating point precision)
        total_rate = (
            metrics["structure_presence_false_positive_rate"] +
            metrics["structure_presence_false_negative_rate"] +
            metrics["structure_presence_true_positive_rate"] +
            metrics["structure_presence_true_negative_rate"]
        )
        assert abs(total_rate - 1.0) < 1e-6, f"Rates should sum to 1.0, got {total_rate}"

    def test_structure_top_k_accuracy(self):
        """Test top-k accuracy for structure types."""
        B = self.batch_size
        
        # Create logits where correct class is in top-3
        structure_logits = torch.randn(B, self.num_structure_types)
        target_classes = torch.randint(0, self.num_structure_types, (B,))
        
        # Ensure correct class has high probability for at least some samples
        for i in range(B // 2):
            structure_logits[i, target_classes[i]] = 10.0  # High score for correct class

        predictions = {"structure_types": structure_logits}
        targets = {"structure_types": torch.eye(self.num_structure_types)[target_classes]}

        metrics = self.structure_calc.calculate_accuracy(predictions, targets)

        assert "structure_type_accuracy" in metrics
        if self.num_structure_types >= 3:
            assert "structure_type_top3_accuracy" in metrics
            # Top-3 should be >= regular accuracy
            assert metrics["structure_type_top3_accuracy"] >= metrics["structure_type_accuracy"]

    def test_comprehensive_structure_aware_evaluation(self):
        """Test complete structure-aware evaluation pipeline."""
        predictions = self._create_mock_predictions()
        targets = self._create_mock_targets()

        # Test full AccuracyMetrics integration
        metrics = self.accuracy_metrics.compute_metrics(predictions, targets)

        # Should include all types of metrics
        metric_categories = {
            "mask": ["mask_accuracy", "mask_precision", "mask_recall", "mask_f1", "mask_iou"],
            "block": ["block_type_accuracy"],
            "structure": ["structure_mask_accuracy", "structure_iou", "structure_type_accuracy"],
        }

        for category, expected_metrics in metric_categories.items():
            for metric in expected_metrics:
                assert metric in metrics, f"Missing {category} metric: {metric}"

    def test_iou_calculator_with_structure_masks(self):
        """Test IoU calculator specifically for structure masks."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        # Create binary structure predictions
        predictions = torch.randint(0, 2, (B, H, W, D))
        targets = torch.randint(0, 2, (B, H, W, D))

        metrics = self.iou_calc.calculate_iou(predictions, targets, num_classes=2)

        assert "mean_iou" in metrics
        assert "per_class_iou" in metrics
        assert isinstance(metrics["per_class_iou"], dict)
        assert "class_0_iou" in metrics["per_class_iou"]
        assert "class_1_iou" in metrics["per_class_iou"]

    def test_dice_calculator_with_structure_masks(self):
        """Test Dice calculator for binary structure segmentation."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        # Binary predictions (structure present/absent)
        predictions = torch.rand(B, H, W, D)  # Probabilities
        targets = torch.randint(0, 2, (B, H, W, D)).float()

        metrics = self.dice_calc.calculate_dice(predictions, targets)

        assert "dice_coefficient" in metrics
        assert isinstance(metrics["dice_coefficient"], float)
        assert 0.0 <= metrics["dice_coefficient"] <= 1.0

    def test_partial_prediction_handling(self):
        """Test that metrics handle partial predictions gracefully."""
        # Only provide some prediction types
        partial_predictions = {
            "structure_mask": torch.sigmoid(torch.randn(self.batch_size, 1, *self.spatial_dims)),
            # Missing structure_types, air_mask_logits, block_type_logits
        }
        
        partial_targets = {
            "structure_mask": torch.randint(0, 2, (self.batch_size, 1, *self.spatial_dims)).float(),
            "air_mask": torch.randint(0, 2, (self.batch_size, *self.spatial_dims)).float(),
            # Missing structure_types, block_types
        }

        # Should not crash and should compute available metrics
        metrics = self.accuracy_metrics.compute_metrics(partial_predictions, partial_targets)

        # Should have structure mask metrics but not others
        assert "structure_mask_accuracy" in metrics
        assert "structure_iou" in metrics
        assert "mask_accuracy" not in metrics  # air mask not predicted
        assert "block_type_accuracy" not in metrics  # block types not predicted

    def test_edge_case_empty_structures(self):
        """Test handling of chunks with no structures."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        predictions = {
            "structure_mask": torch.zeros(B, 1, H, W, D),  # No structures predicted
        }
        targets = {
            "structure_mask": torch.zeros(B, 1, H, W, D),  # No structures in ground truth
        }

        metrics = self.structure_calc.calculate_accuracy(predictions, targets)

        # All should be perfect when both predict no structures
        assert metrics["structure_mask_accuracy"] == 1.0
        assert metrics["structure_presence_accuracy"] == 1.0
        assert metrics["structure_presence_true_negative_rate"] == 1.0
        assert metrics["structure_presence_false_positive_rate"] == 0.0

    def test_edge_case_all_structures(self):
        """Test handling of chunks completely filled with structures."""
        B, H, W, D = self.batch_size, *self.spatial_dims
        
        predictions = {
            "structure_mask": torch.ones(B, 1, H, W, D),  # All structures predicted
        }
        targets = {
            "structure_mask": torch.ones(B, 1, H, W, D),  # All structures in ground truth
        }

        metrics = self.structure_calc.calculate_accuracy(predictions, targets)

        # All should be perfect when both predict all structures
        assert metrics["structure_mask_accuracy"] == 1.0
        assert metrics["structure_iou"] == 1.0
        assert metrics["structure_presence_accuracy"] == 1.0
        assert metrics["structure_presence_true_positive_rate"] == 1.0

    def test_regression_validation_workflow(self):
        """Test complete workflow for Phase 2 regression validation."""
        # Simulate before/after fine-tuning comparison
        
        # "Before" structure-aware fine-tuning (poor structure prediction)
        before_predictions = self._create_mock_predictions()
        before_predictions["structure_mask"] = torch.rand_like(before_predictions["structure_mask"]) * 0.3  # Low confidence
        
        # "After" structure-aware fine-tuning (better structure prediction)  
        after_predictions = self._create_mock_predictions()
        targets = self._create_mock_targets()

        # Make "after" predictions more aligned with targets
        after_predictions["structure_mask"] = targets["structure_mask"] + torch.randn_like(targets["structure_mask"]) * 0.1

        before_metrics = self.accuracy_metrics.compute_metrics(before_predictions, targets)
        after_metrics = self.accuracy_metrics.compute_metrics(after_predictions, targets)

        # After fine-tuning should generally have better structure metrics
        # (This is a mock test, so we'll just verify the metrics exist)
        structure_metrics = [
            "structure_mask_accuracy",
            "structure_iou", 
            "structure_presence_accuracy"
        ]

        for metric in structure_metrics:
            assert metric in before_metrics, f"Before metrics missing {metric}"
            assert metric in after_metrics, f"After metrics missing {metric}"
            assert isinstance(before_metrics[metric], float)
            assert isinstance(after_metrics[metric], float)

        print(f"Phase 2 regression test completed:")
        print(f"  Structure IoU: {before_metrics['structure_iou']:.3f} → {after_metrics['structure_iou']:.3f}")
        print(f"  Structure Accuracy: {before_metrics['structure_mask_accuracy']:.3f} → {after_metrics['structure_mask_accuracy']:.3f}")


if __name__ == "__main__":
    # Run specific test for development
    test_suite = TestStructureAwareEvaluationMetrics()
    test_suite.setup_method()
    test_suite.test_regression_validation_workflow()
