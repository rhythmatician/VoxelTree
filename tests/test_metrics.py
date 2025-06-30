"""
Unit tests for VoxelTree metrics module.
Tests metrics calculat        # Expected confusion matrix:
        # [1, 0, 1]  # 1 correct class 0, 1 class 0 predicted as class 2
        # [1, 2, 0]  # 1 class 1 predicted as class 0, 2 correct class 1
        # [0, 0, 1]  # 1 correct class 2
        expected = torch.tensor([[1, 0, 1], [1, 2, 0], [0, 0, 1]], dtype=torch.float32)ccuracy with controlled inputs.
"""

import numpy as np
import torch

from train.metrics import VoxelMetrics


class TestVoxelMetrics:
    """Test suite for VoxelMetrics class."""

    def test_binary_iou(self):
        """Test IoU calculation with simple test cases."""
        # Perfect match case
        pred_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        target_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        iou = VoxelMetrics.binary_iou(pred_mask, target_mask)
        assert abs(iou - 1.0) < 1e-5, f"Expected perfect IoU to be 1.0, got {iou}"

        # No overlap case
        pred_mask = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        target_mask = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        iou = VoxelMetrics.binary_iou(pred_mask, target_mask)
        assert abs(iou - 0.0) < 1e-5, f"Expected zero overlap IoU to be 0.0, got {iou}"

        # 50% overlap case
        pred_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        target_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        iou = VoxelMetrics.binary_iou(pred_mask, target_mask)
        assert abs(iou - 0.333) < 0.01, f"Expected IoU to be ~0.333, got {iou}"

        # Test with logits (threshold 0.5)
        pred_logits = torch.tensor([[[[0.8, 0.9], [-0.5, -0.2]]]])
        target_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        iou = VoxelMetrics.binary_iou(pred_logits, target_mask)
        assert abs(iou - 1.0) < 1e-5, f"Expected perfect IoU with thresholded logits, got {iou}"

    def test_dice_coefficient(self):
        """Test Dice coefficient calculation."""
        # Perfect match case
        pred_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        target_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        dice = VoxelMetrics.dice_coefficient(pred_mask, target_mask)
        assert abs(dice - 1.0) < 1e-5, f"Expected perfect Dice to be 1.0, got {dice}"

        # No overlap case
        pred_mask = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        target_mask = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        dice = VoxelMetrics.dice_coefficient(pred_mask, target_mask)
        assert abs(dice - 0.0) < 1e-5, f"Expected zero overlap Dice to be 0.0, got {dice}"

        # 50% overlap case
        pred_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        target_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        dice = VoxelMetrics.dice_coefficient(pred_mask, target_mask)
        assert abs(dice - 0.5) < 0.01, f"Expected Dice to be ~0.5, got {dice}"

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        # Simple 3-class case
        pred_types = torch.tensor([[[0, 1, 2], [1, 0, 2]]])  # Shape: (1, 2, 3)
        target_types = torch.tensor([[[0, 1, 2], [1, 1, 0]]])  # Shape: (1, 2, 3)
        n_classes = 3

        conf_matrix = VoxelMetrics.compute_confusion_matrix(pred_types, target_types, n_classes)

        # Expected confusion matrix:
        # [1, 0, 1]  # 1 correct class 0, 1 class 0 predicted as class 2
        # [1, 2, 0]  # 1 class 1 predicted as class 0, 2 correct class 1
        # [0, 0, 1]  # 0 class 2 predicted incorrectly, 1 correct class 2
        expected = torch.tensor([[1, 0, 1], [1, 2, 0], [0, 0, 1]], dtype=torch.float32)

        assert torch.allclose(
            conf_matrix, expected
        ), f"Confusion matrix mismatch. Expected {expected}, got {conf_matrix}"

    def test_per_class_accuracy(self):
        """Test per-class accuracy calculation."""
        # Construct a simple confusion matrix
        conf_matrix = torch.tensor(
            [
                [5, 1, 2],  # 5 correct class 0, misclassified 1+2
                [2, 8, 0],  # 8 correct class 1, misclassified 2
                [1, 3, 6],  # 6 correct class 2, misclassified 1+3
            ],
            dtype=torch.float32,
        )

        accuracy = VoxelMetrics.per_class_accuracy(conf_matrix)

        # Expected accuracies (recall - diagonal / row sum):
        # Class 0: 5 / (5+1+2) = 5/8 = 0.625
        # Class 1: 8 / (2+8+0) = 8/10 = 0.8
        # Class 2: 6 / (1+3+6) = 6/10 = 0.6
        expected = torch.tensor([0.625, 0.8, 0.6])

        assert torch.allclose(
            accuracy, expected, atol=1e-3
        ), f"Per-class accuracy mismatch. Expected {expected}, got {accuracy}"

    def test_compute_all_metrics(self):
        """Test the combined metrics calculation."""
        # Create simple test data
        batch_size = 2
        n_classes = 3

        # Simple voxel grid with test patterns
        pred_mask_logits = torch.tensor(
            [
                [[[0.8, 0.9], [-0.5, -0.2]]],  # First batch item
                [[[0.7, -0.3], [0.6, 0.8]]],  # Second batch item
            ]
        )

        pred_type_logits = torch.zeros((batch_size, n_classes, 2, 2))
        # Set highest probability to specific classes
        # Batch 0: [[0, 1], [1, 2]]
        pred_type_logits[0, 0, 0, 0] = 2.0  # Class 0 for (0,0)
        pred_type_logits[0, 1, 0, 1] = 2.0  # Class 1 for (0,1)
        pred_type_logits[0, 1, 1, 0] = 2.0  # Class 1 for (1,0)
        pred_type_logits[0, 2, 1, 1] = 2.0  # Class 2 for (1,1)

        # Batch 1: [[2, 0], [0, 1]]
        pred_type_logits[1, 2, 0, 0] = 2.0  # Class 2 for (0,0)
        pred_type_logits[1, 0, 0, 1] = 2.0  # Class 0 for (0,1)
        pred_type_logits[1, 0, 1, 0] = 2.0  # Class 0 for (1,0)
        pred_type_logits[1, 1, 1, 1] = 2.0  # Class 1 for (1,1)

        target_mask = torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],  # First batch item
                [[[1.0, 0.0], [0.0, 1.0]]],  # Second batch item
            ]
        )

        target_types = torch.tensor(
            [[[0, 1], [1, 2]], [[2, 0], [0, 1]]]  # First batch item  # Second batch item
        )

        # Compute all metrics
        metrics = VoxelMetrics.compute_all_metrics(
            pred_mask_logits, pred_type_logits, target_mask, target_types, n_classes
        )

        # Check that all expected keys are present
        expected_keys = [
            "mask_iou",
            "mask_dice",
            "type_accuracy",
            "type_accuracy_per_class",
            "combined_score",
        ]
        for key in expected_keys:
            assert key in metrics, f"Expected '{key}' in metrics result"

        # For this test case, we expect perfect type prediction and near-perfect mask
        assert metrics["type_accuracy"] == 1.0, "Expected perfect type accuracy"
        assert metrics["mask_iou"] > 0.7, "Expected high mask IoU"
        assert metrics["mask_dice"] > 0.8, "Expected high mask Dice coefficient"


def test_batch_processing():
    """Test metrics work correctly with batched inputs."""
    # Create a batch of test inputs
    batch_size = 4
    n_classes = 5
    spatial_dims = (3, 3, 3)  # Small 3D grid

    # Random predictions and targets
    pred_mask_logits = torch.randn((batch_size, 1) + spatial_dims)
    target_mask = torch.randint(0, 2, (batch_size, 1) + spatial_dims).float()

    pred_type_logits = torch.randn((batch_size, n_classes) + spatial_dims)
    target_types = torch.randint(0, n_classes, (batch_size,) + spatial_dims)

    # Compute metrics for the batch
    metrics = VoxelMetrics.compute_all_metrics(
        pred_mask_logits, pred_type_logits, target_mask, target_types, n_classes
    )

    # Verify metrics are reasonable (specific values aren't important here)
    assert 0 <= metrics["mask_iou"] <= 1, "IoU should be between 0 and 1"
    assert 0 <= metrics["mask_dice"] <= 1, "Dice coefficient should be between 0 and 1"
    assert 0 <= metrics["type_accuracy"] <= 1, "Type accuracy should be between 0 and 1"
    assert metrics["type_accuracy_per_class"].shape == (
        n_classes,
    ), f"Per-class accuracy should have shape ({n_classes},)"


def test_numpy_input_handling():
    """Test that the metrics functions handle numpy inputs correctly."""
    # Create numpy arrays
    pred_mask = np.random.rand(2, 1, 4, 4)
    target_mask = np.random.randint(0, 2, (2, 1, 4, 4)).astype(np.float32)

    # Compute IoU
    iou = VoxelMetrics.binary_iou(pred_mask, target_mask)

    # Should return a scalar value between 0 and 1
    assert isinstance(iou, float), "IoU should be a float"
    assert 0 <= iou <= 1, "IoU should be between 0 and 1"


if __name__ == "__main__":
    # Run tests manually if needed
    test_instance = TestVoxelMetrics()
    test_instance.test_binary_iou()
    test_instance.test_dice_coefficient()
    test_instance.test_confusion_matrix()
    test_instance.test_per_class_accuracy()
    test_instance.test_compute_all_metrics()
    test_batch_processing()
    test_numpy_input_handling()
    print("All tests passed!")
