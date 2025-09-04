"""
Tests for voxel visualization and TensorBoard logging functionality.

This module tests the VoxelVisualizer and TensorBoardLogger classes
to ensure proper visualization generation and logging capabilities.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from train.visualizer import TensorBoardLogger, VoxelVisualizer


class TestTensorBoardLogger:
    """Test suite for TensorBoardLogger functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for TensorBoard logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_tensorboard(self):
        """Mock TensorBoard SummaryWriter to avoid actual file I/O."""
        with patch("train.visualizer.SummaryWriter") as mock_writer:
            yield mock_writer

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data for testing."""
        return {
            "parent_voxel": torch.rand(2, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 256, (2, 256, 16, 16)),
            "heightmap_patch": torch.rand(2, 1, 16, 16),
            "river_patch": torch.rand(2, 1, 16, 16),
            "y_index": torch.randint(0, 10, (2,)),
            "lod": torch.randint(1, 5, (2,)),
            "target_mask": torch.randint(0, 2, (2, 1, 16, 16, 16), dtype=torch.bool),
            "target_types": torch.randint(0, 100, (2, 16, 16, 16)),
        }

    @pytest.fixture
    def sample_model_outputs(self):
        """Create sample model outputs for testing."""
        return {
            "block_logits": torch.randn(2, 1104, 16, 16, 16),
            "air_mask": torch.randn(2, 1, 16, 16, 16),
        }

    def test_logger_initialization_enabled(self, temp_log_dir, mock_tensorboard):
        """Test TensorBoardLogger initialization when enabled."""
        logger = TensorBoardLogger(temp_log_dir, enabled=True)

        assert logger.enabled is True
        assert logger.log_dir == temp_log_dir
        mock_tensorboard.assert_called_once_with(log_dir=str(temp_log_dir))

    def test_logger_initialization_disabled(self, temp_log_dir):
        """Test TensorBoardLogger initialization when disabled."""
        logger = TensorBoardLogger(temp_log_dir, enabled=False)

        assert logger.enabled is False
        assert logger.writer is None

    @patch("train.visualizer.TENSORBOARD_AVAILABLE", False)
    def test_logger_tensorboard_unavailable(self, temp_log_dir):
        """Test TensorBoardLogger when TensorBoard is not available."""
        logger = TensorBoardLogger(temp_log_dir, enabled=True)

        assert logger.enabled is False
        assert logger.writer is None

    def test_log_metrics(self, temp_log_dir, mock_tensorboard):
        """Test logging scalar metrics."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        logger = TensorBoardLogger(temp_log_dir, enabled=True)

        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "lr": 0.001,
            "epoch": 1,  # Should be logged as it's numeric
        }

        logger.log_metrics(metrics, step=10, prefix="train")

        # Check that add_scalar was called for each metric
        expected_calls = [
            ("train/loss", 0.5, 10),
            ("train/accuracy", 0.95, 10),
            ("train/lr", 0.001, 10),
            ("train/epoch", 1, 10),
        ]

        assert mock_writer_instance.add_scalar.call_count == len(expected_calls)
        for expected_call in expected_calls:
            mock_writer_instance.add_scalar.assert_any_call(*expected_call)

    def test_log_metrics_disabled(self, temp_log_dir):
        """Test that metrics logging is skipped when disabled."""
        logger = TensorBoardLogger(temp_log_dir, enabled=False)

        # Should not raise an error
        logger.log_metrics({"loss": 0.5}, step=1)

    def test_log_model_graph_with_sample_batch(self, temp_log_dir, mock_tensorboard, sample_batch):
        """Test logging model graph with real batch data."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        # Create a simple mock model
        mock_model = Mock()
        # Mock parameter for device detection
        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])

        logger = TensorBoardLogger(temp_log_dir, enabled=True)
        logger.log_model_graph(mock_model, sample_batch)

        # Should call add_graph (even if with None input due to complexity)
        mock_writer_instance.add_graph.assert_called_once()

    def test_log_model_graph_without_batch(self, temp_log_dir, mock_tensorboard):
        """Test logging model graph without sample batch."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        # Create a mock model with GPU device
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        logger = TensorBoardLogger(temp_log_dir, enabled=True)
        logger.log_model_graph(mock_model)

        # Should call add_graph
        mock_writer_instance.add_graph.assert_called_once()

    def test_log_voxel_batch(
        self, temp_log_dir, mock_tensorboard, sample_batch, sample_model_outputs
    ):
        """Test logging voxel batch visualizations."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        logger = TensorBoardLogger(temp_log_dir, enabled=True)

        # Mock the VoxelVisualizer.visualize_comparison method
        with patch.object(VoxelVisualizer, "visualize_comparison") as mock_viz:
            mock_fig = Mock()
            mock_viz.return_value = mock_fig

            logger.log_voxel_batch(
                inputs=sample_batch["parent_voxel"],
                predictions=sample_model_outputs["air_mask"],
                targets=sample_batch["target_mask"],
                step=5,
                max_samples=2,
            )

            # Should call add_figure for each sample
            assert mock_writer_instance.add_figure.call_count == 2
            mock_writer_instance.add_figure.assert_any_call("voxel_sample_0", mock_fig, 5)
            mock_writer_instance.add_figure.assert_any_call("voxel_sample_1", mock_fig, 5)

    def test_log_embedding(self, temp_log_dir, mock_tensorboard):
        """Test logging embeddings."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        logger = TensorBoardLogger(temp_log_dir, enabled=True)

        features = torch.randn(10, 64)  # 10 samples, 64 dimensions
        metadata = [f"sample_{i}" for i in range(10)]

        logger.log_embedding(features, metadata, step=100, tag="lod_embeddings")

        mock_writer_instance.add_embedding.assert_called_once_with(
            features, metadata=metadata, tag="lod_embeddings", global_step=100
        )

    def test_close(self, temp_log_dir, mock_tensorboard):
        """Test closing the logger."""
        mock_writer_instance = Mock()
        mock_tensorboard.return_value = mock_writer_instance

        logger = TensorBoardLogger(temp_log_dir, enabled=True)
        logger.close()

        mock_writer_instance.close.assert_called_once()


class TestVoxelVisualizer:
    """Test suite for VoxelVisualizer functionality."""

    @pytest.fixture
    def sample_voxel_data(self):
        """Create sample 3D voxel data for testing."""
        # Create some structured test data
        parent_voxel = np.zeros((8, 8, 8), dtype=bool)
        parent_voxel[2:6, 2:6, 2:6] = True  # Cube in center

        pred_mask = np.random.rand(16, 16, 16) > 0.5
        pred_types = np.random.randint(0, 10, (16, 16, 16))

        target_mask = np.random.rand(16, 16, 16) > 0.4
        target_types = np.random.randint(0, 10, (16, 16, 16))

        return {
            "parent_voxel": parent_voxel,
            "pred_mask": pred_mask,
            "pred_types": pred_types,
            "target_mask": target_mask,
            "target_types": target_types,
        }

    def test_get_type_color_common_types(self):
        """Test color generation for common block types."""
        # Test known colors
        assert VoxelVisualizer._get_type_color(0) == (0.5, 0.5, 0.5, 1.0)  # Stone
        assert VoxelVisualizer._get_type_color(1) == (0.6, 0.4, 0.2, 1.0)  # Dirt
        assert VoxelVisualizer._get_type_color(2) == (0.0, 0.7, 0.0, 1.0)  # Grass

    def test_get_type_color_unknown_types(self):
        """Test color generation for unknown block types."""
        # Should generate consistent colors for same type
        color1 = VoxelVisualizer._get_type_color(999)
        color2 = VoxelVisualizer._get_type_color(999)
        assert color1 == color2

        # Different types should have different colors
        color3 = VoxelVisualizer._get_type_color(998)
        assert color1 != color3

        # Should be valid RGBA
        assert len(color1) == 4
        assert all(0 <= c <= 1 for c in color1)
        assert color1[3] == 1.0  # Alpha should be 1

    @patch("train.visualizer.plt")
    def test_visualize_prediction_with_targets(self, mock_plt, sample_voxel_data):
        """Test visualization with both predictions and targets."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots = Mock()

        # Mock subplot creation
        mock_axes = [Mock() for _ in range(4)]
        for i, ax in enumerate(mock_axes):
            mock_fig.add_subplot.return_value = ax if i == 0 else mock_axes[i]

        VoxelVisualizer.visualize_prediction(
            parent_voxel=sample_voxel_data["parent_voxel"],
            pred_mask=sample_voxel_data["pred_mask"],
            pred_types=sample_voxel_data["pred_types"],
            target_mask=sample_voxel_data["target_mask"],
            target_types=sample_voxel_data["target_types"],
            metadata={"epoch": 1, "loss": 0.5},
        )

        # Should create figure with 4 subplots (parent, pred_mask, pred_types, target)
        mock_plt.figure.assert_called_once()
        assert mock_fig.add_subplot.call_count == 4

        # Should show the plot
        mock_plt.show.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("train.visualizer.plt")
    def test_visualize_prediction_without_targets(self, mock_plt, sample_voxel_data):
        """Test visualization without targets."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        # Mock subplot creation
        mock_axes = [Mock() for _ in range(3)]
        for i, ax in enumerate(mock_axes):
            mock_fig.add_subplot.return_value = ax if i == 0 else mock_axes[i]

        VoxelVisualizer.visualize_prediction(
            parent_voxel=sample_voxel_data["parent_voxel"],
            pred_mask=sample_voxel_data["pred_mask"],
            pred_types=sample_voxel_data["pred_types"],
        )

        # Should create figure with 3 subplots (parent, pred_mask, pred_types)
        assert mock_fig.add_subplot.call_count == 3

    def test_visualize_prediction_tensor_inputs(self, sample_voxel_data):
        """Test visualization with PyTorch tensor inputs."""
        # Convert to tensors
        tensor_data = {
            "parent_voxel": torch.from_numpy(sample_voxel_data["parent_voxel"].astype(np.float32)),
            "pred_mask": torch.from_numpy(sample_voxel_data["pred_mask"].astype(np.float32)),
            "pred_types": torch.from_numpy(sample_voxel_data["pred_types"]),
        }

        with patch("train.visualizer.plt") as mock_plt:
            mock_fig = Mock()
            mock_plt.figure.return_value = mock_fig

            # Should not raise an error
            VoxelVisualizer.visualize_prediction(
                parent_voxel=tensor_data["parent_voxel"],
                pred_mask=tensor_data["pred_mask"],
                pred_types=tensor_data["pred_types"],
            )

            mock_plt.figure.assert_called_once()

    def test_visualize_prediction_with_output_path(self, sample_voxel_data, tmp_path):
        """Test visualization with file output."""
        output_path = tmp_path / "test_viz.png"

        with patch("train.visualizer.plt") as mock_plt:
            mock_fig = Mock()
            mock_plt.figure.return_value = mock_fig

            result = VoxelVisualizer.visualize_prediction(
                parent_voxel=sample_voxel_data["parent_voxel"],
                pred_mask=sample_voxel_data["pred_mask"],
                pred_types=sample_voxel_data["pred_types"],
                output_path=output_path,
            )

            # Should save instead of show
            mock_plt.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
            mock_plt.show.assert_not_called()

            # Should return the output path
            assert result == output_path

    @patch("train.visualizer.plt")
    def test_visualize_comparison(self, mock_plt, sample_voxel_data):
        """Test the visualize_comparison static method."""
        mock_fig = Mock()
        mock_plt.figure.return_value = mock_fig

        result = VoxelVisualizer.visualize_comparison(
            sample_voxel_data["parent_voxel"],
            sample_voxel_data["pred_mask"],
            sample_voxel_data["target_mask"],
            return_figure=True,
        )

        # Should return the figure when return_figure=True
        assert result == mock_fig

    def test_visualize_batch_integration(self, tmp_path):
        """Test batch visualization integration."""
        # Create sample batch data
        batch_data = {
            "parent_voxel": torch.rand(2, 1, 8, 8, 8),
            "target_mask": torch.randint(0, 2, (2, 1, 16, 16, 16), dtype=torch.bool),
            "target_types": torch.randint(0, 10, (2, 16, 16, 16)),
            "y_index": torch.tensor([5, 7]),
            "lod": torch.tensor([1, 2]),
        }

        model_outputs = {
            "air_mask_logits": torch.randn(2, 1, 16, 16, 16),
            "block_type_logits": torch.randn(2, 10, 16, 16, 16),
        }

        with patch("train.visualizer.VoxelVisualizer.visualize_prediction") as mock_viz:
            mock_viz.return_value = tmp_path / "test.png"

            paths = VoxelVisualizer.visualize_batch(
                batch_data=batch_data,
                model_outputs=model_outputs,
                output_dir=tmp_path,
                max_samples=2,
            )

            # Should create visualizations for both samples
            assert len(paths) == 2
            assert mock_viz.call_count == 2


class TestVoxelVisualizationIntegration:
    """Integration tests for voxel visualization pipeline."""

    def test_full_tensorboard_logging_pipeline(self, tmp_path):
        """Test the complete voxel logging pipeline."""
        # Create sample data
        inputs = torch.rand(1, 1, 8, 8, 8)
        predictions = torch.rand(1, 1, 16, 16, 16)
        targets = torch.randint(0, 2, (1, 1, 16, 16, 16), dtype=torch.float32)

        with patch("train.visualizer.SummaryWriter") as mock_writer_class:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(tmp_path, enabled=True)

            with patch.object(VoxelVisualizer, "visualize_comparison") as mock_viz:
                mock_fig = Mock()
                mock_viz.return_value = mock_fig

                logger.log_voxel_batch(inputs, predictions, targets, step=10)

                # Verify the pipeline worked
                mock_viz.assert_called_once()
                mock_writer.add_figure.assert_called_once_with("voxel_sample_0", mock_fig, 10)

    def test_error_handling_in_visualization(self, tmp_path):
        """Test error handling in visualization logging."""
        with patch("train.visualizer.SummaryWriter") as mock_writer_class:
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer

            logger = TensorBoardLogger(tmp_path, enabled=True)

            # Simulate an error in visualization
            with patch.object(
                VoxelVisualizer, "visualize_comparison", side_effect=Exception("Test error")
            ):
                # Should not raise an error, just log warning
                logger.log_voxel_batch(
                    inputs=torch.rand(1, 1, 8, 8, 8),
                    predictions=torch.rand(1, 1, 16, 16, 16),
                    targets=torch.rand(1, 1, 16, 16, 16),
                    step=1,
                )

                # Writer should not be called due to error
                mock_writer.add_figure.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
