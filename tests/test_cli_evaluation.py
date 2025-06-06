"""
Test for CLI evaluation script functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

from scripts.evaluation.evaluate_model import ModelEvaluator


class TestCLIEvaluation:
    """Test CLI evaluation functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_temp_dir = Path(tempfile.mkdtemp())

        # Create test config
        self.test_config = {
            "model": {
                "in_channels": 1,
                "out_channels": 2,
                "base_channels": 32,
                "depth": 4,
                "structure_aware": True,
                "heightmap_conditioning": True,
                "biome_conditioning": True,
            },
            "dataset": {
                "chunk_size": [16, 16, 16],
                "parent_size": [8, 8, 8],
                "normalize_inputs": True,
                "augmentation": False,
                "structure_aware": True,
            },
            "eval_batch_size": 4,
            "num_workers": 2,
        }

        # Save test config
        self.config_path = self.test_temp_dir / "test_eval_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.test_config, f)

    def test_model_evaluator_initialization(self):
        """Test that ModelEvaluator can be initialized with config."""
        # Create a mock checkpoint
        checkpoint_path = self.test_temp_dir / "test_checkpoint.pth"

        # Create minimal checkpoint data
        mock_state_dict = {
            "encoder.0.weight": torch.randn(32, 1, 3, 3, 3),
            "encoder.0.bias": torch.randn(32),
        }
        torch.save({"model_state_dict": mock_state_dict}, checkpoint_path)

        # Test initialization (should not crash)
        with patch("scripts.evaluation.evaluate_model.VoxelUNet3D") as mock_model:
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance

            evaluator = ModelEvaluator(
                config_path=str(self.config_path),
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )

            # Should have loaded config
            assert evaluator.config == self.test_config
            assert evaluator.device.type == "cpu"

            # Should have initialized model
            mock_model.assert_called_once()
            mock_model_instance.load_state_dict.assert_called_once()

    def test_device_auto_selection(self):
        """Test automatic device selection."""
        checkpoint_path = self.test_temp_dir / "test_checkpoint.pth"
        torch.save({"model_state_dict": {}}, checkpoint_path)

        with patch("scripts.evaluation.evaluate_model.VoxelUNet3D"):
            evaluator = ModelEvaluator(
                config_path=str(self.config_path),
                checkpoint_path=str(checkpoint_path),
                device="auto",
            )

            # Should select a valid device
            assert evaluator.device.type in ["cpu", "cuda", "mps"]

    def test_metrics_aggregation_workflow(self):
        """Test the metrics aggregation workflow."""
        checkpoint_path = self.test_temp_dir / "test_checkpoint.pth"
        torch.save({"model_state_dict": {}}, checkpoint_path)

        with patch("scripts.evaluation.evaluate_model.VoxelUNet3D"):
            evaluator = ModelEvaluator(
                config_path=str(self.config_path),
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )

            # Test metrics aggregation
            mock_metrics = [
                {"accuracy": 0.8, "f1": 0.7, "structure_iou": 0.6},
                {"accuracy": 0.9, "f1": 0.8, "structure_iou": 0.7},
                {"accuracy": 0.85, "f1": 0.75, "structure_iou": 0.65},
            ]

            aggregated = evaluator._aggregate_metrics(mock_metrics)

            # Should average correctly
            assert abs(aggregated["accuracy"] - 0.85) < 1e-6
            assert abs(aggregated["f1"] - 0.75) < 1e-6
            assert abs(aggregated["structure_iou"] - 0.65) < 1e-6

    def test_forward_pass_structure_inputs(self):
        """Test that forward pass handles structure inputs correctly."""
        checkpoint_path = self.test_temp_dir / "test_checkpoint.pth"
        torch.save({"model_state_dict": {}}, checkpoint_path)

        with patch("scripts.evaluation.evaluate_model.VoxelUNet3D") as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_model.return_value = {"air_mask_logits": torch.randn(2, 1, 8, 8, 8)}

            evaluator = ModelEvaluator(
                config_path=str(self.config_path),
                checkpoint_path=str(checkpoint_path),
                device="cpu",
            )

            # Create mock batch with structure data
            batch = {
                "parent_voxel": torch.randn(2, 1, 8, 8, 8),
                "biome_patch": torch.randn(2, 10, 8, 8),
                "heightmap": torch.randn(2, 1, 8, 8),
                "structure_mask": torch.randn(2, 1, 8, 8, 8),
                "structure_types": torch.randn(2, 5),
                "structure_positions": torch.randn(2, 3),
            }

            # Run forward pass
            predictions = evaluator._forward_pass(batch)

            # Should call model with structure inputs
            mock_model.assert_called_once()
            call_args = mock_model.call_args

            # Check that structure arguments are passed
            assert "structure_mask" in call_args.kwargs
            assert "structure_types" in call_args.kwargs
            assert "structure_positions" in call_args.kwargs

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir)
