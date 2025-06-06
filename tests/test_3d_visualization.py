"""
Tests for VoxelTree 3D visualization functionality.
Phase 6.3: TDD implementation for 3D voxel render previews.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from scripts.evaluation.visualization import VoxelRenderer, VoxelVisualizationSuite


class TestVoxelRenderer:
    """Test suite for VoxelRenderer class."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.renderer = VoxelRenderer()

    def test_renderer_initialization(self):
        """Test VoxelRenderer initializes with default colors."""
        assert self.renderer.block_colors is not None
        assert 0 in self.renderer.block_colors  # Air should be present
        assert len(self.renderer.block_colors) >= 10  # Should have multiple block types

    def test_custom_block_colors(self):
        """Test VoxelRenderer accepts custom block colors."""
        custom_colors = {0: "#FF0000", 1: "#00FF00", 2: "#0000FF"}
        renderer = VoxelRenderer(block_colors=custom_colors)

        assert renderer.block_colors == custom_colors
        assert renderer.block_colors[0] == "#FF0000"

    def test_render_simple_voxel_chunk(self):
        """Test rendering a simple 3D voxel chunk."""
        # Create test voxel data (4x4x4)
        voxel_data = np.zeros((4, 4, 4), dtype=np.uint8)
        voxel_data[1:3, 1:3, 1:3] = 1  # Place some solid blocks

        # Test rendering without saving
        fig = self.renderer.render_voxel_chunk(voxel_data, title="Test Chunk")

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Should have one 3D axis
        plt.close(fig)

    def test_render_with_air_mask(self):
        """Test rendering voxel chunk with air mask filtering."""
        # Create test data
        voxel_data = np.ones((3, 3, 3), dtype=np.uint8)
        air_mask = np.zeros((3, 3, 3), dtype=bool)
        air_mask[0, 0, 0] = True  # Mark one position as air

        # Render with air mask
        fig = self.renderer.render_voxel_chunk(
            voxel_data, air_mask=air_mask, title="Test with Air Mask"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_torch_tensors(self):
        """Test rendering with PyTorch tensors as input."""
        # Create torch tensor data
        voxel_data = torch.zeros(3, 3, 3, dtype=torch.uint8)
        voxel_data[1, 1, 1] = 2  # Place one block

        air_mask = torch.zeros(3, 3, 3, dtype=torch.bool)
        air_mask[0, 0, 0] = True

        # Should handle torch tensors correctly
        fig = self.renderer.render_voxel_chunk(
            voxel_data, air_mask=air_mask, title="Torch Tensor Test"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_batched_input(self):
        """Test rendering with batched input (4D array)."""
        # Create batched voxel data (batch_size=2, 3x3x3)
        voxel_data = np.zeros((2, 3, 3, 3), dtype=np.uint8)
        voxel_data[0, 1, 1, 1] = 1  # Place block in first batch item

        # Should take first batch item
        fig = self.renderer.render_voxel_chunk(voxel_data, title="Batched Input Test")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_render_with_save(self, mock_savefig):
        """Test rendering and saving to file."""
        voxel_data = np.ones((2, 2, 2), dtype=np.uint8)
        output_path = Path("test_output.png")

        fig = self.renderer.render_voxel_chunk(
            voxel_data, output_path=output_path, title="Save Test"
        )

        # Should call savefig with correct path
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(output_path) in str(args[0])
        plt.close(fig)

    def test_render_comparison(self):
        """Test side-by-side comparison rendering."""
        # Create ground truth and prediction data
        ground_truth = np.zeros((3, 3, 3), dtype=np.uint8)
        ground_truth[1, 1, 1] = 1

        prediction = np.zeros((3, 3, 3), dtype=np.uint8)
        prediction[1, 1, 0] = 1  # Slightly different prediction

        fig = self.renderer.render_comparison(
            ground_truth, prediction, title_prefix="Comparison Test"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Should have two 3D subplots
        plt.close(fig)

    def test_render_structure_overlay(self):
        """Test rendering with structure overlay highlighting."""
        # Create test data
        voxel_data = np.ones((4, 4, 4), dtype=np.uint8)
        structure_mask = np.zeros((4, 4, 4), dtype=bool)
        structure_mask[1:3, 1:3, 1:3] = True  # Mark central region as structure

        fig = self.renderer.render_structure_overlay(
            voxel_data, structure_mask, title="Structure Overlay Test"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_empty_voxel_data(self):
        """Test rendering with empty/all-air voxel data."""
        voxel_data = np.zeros((3, 3, 3), dtype=np.uint8)  # All air

        # Should handle empty data gracefully
        fig = self.renderer.render_voxel_chunk(voxel_data, title="Empty Data Test")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVoxelVisualizationSuite:
    """Test suite for VoxelVisualizationSuite class."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.viz_suite = VoxelVisualizationSuite(output_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_suite_initialization(self):
        """Test VoxelVisualizationSuite initializes correctly."""
        assert self.viz_suite.output_dir == self.temp_dir
        assert self.temp_dir.exists()
        assert isinstance(self.viz_suite.renderer, VoxelRenderer)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_visualize_training_sample(self, mock_close, mock_savefig):
        """Test visualization of a complete training sample."""
        # Create mock training sample
        sample = {
            "parent_voxel": np.ones((8, 8, 8), dtype=bool),
            "target_types": np.random.randint(0, 5, size=(16, 16, 16), dtype=np.uint8),
            "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
            "structure_mask": np.random.choice([True, False], size=(16, 16, 16)),
            "structure_types": np.random.randint(0, 3, size=(16, 16, 16), dtype=np.uint8),
        }

        output_paths = self.viz_suite.visualize_training_sample(sample, "test_sample")

        # Should generate multiple visualizations
        assert len(output_paths) >= 2  # At least parent and target
        assert all(path.suffix == ".png" for path in output_paths)
        assert any("parent" in path.name for path in output_paths)
        assert any("target" in path.name for path in output_paths)

        # Should call savefig for each visualization
        assert mock_savefig.call_count >= 2

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_visualize_model_predictions(self, mock_close, mock_savefig):
        """Test visualization of model predictions vs targets."""
        batch_size = 2

        # Create mock predictions and targets
        predictions = {
            "air_mask_logits": torch.randn(batch_size, 1, 16, 16, 16),
            "block_type_logits": torch.randn(batch_size, 5, 16, 16, 16),
        }

        targets = {
            "air_mask": torch.randint(0, 2, (batch_size, 16, 16, 16), dtype=torch.bool),
            "block_types": torch.randint(0, 5, (batch_size, 16, 16, 16), dtype=torch.uint8),
        }

        output_paths = self.viz_suite.visualize_model_predictions(
            predictions, targets, batch_idx=0, sample_id="test_pred"
        )

        # Should generate comparison visualizations
        assert len(output_paths) >= 1
        assert all(path.suffix == ".png" for path in output_paths)
        assert any("comparison" in path.name for path in output_paths)

    def test_visualize_minimal_sample(self):
        """Test visualization with minimal sample data."""
        # Sample with only required fields
        sample = {
            "target_types": np.random.randint(0, 3, size=(8, 8, 8), dtype=np.uint8),
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            output_paths = self.viz_suite.visualize_training_sample(sample, "minimal")

        # Should handle minimal data gracefully
        assert len(output_paths) >= 1
        assert any("target" in path.name for path in output_paths)

    @patch("builtins.open", create=True)
    def test_create_evaluation_report(self, mock_open):
        """Test creation of HTML evaluation report."""
        # Mock file writing
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Create test metrics and sample paths
        metrics = {
            "mask_accuracy": 0.85,
            "block_type_accuracy": 0.78,
            "structure_iou": 0.65,
            "mean_iou": 0.72,
        }

        sample_paths = [
            self.temp_dir / "sample1.png",
            self.temp_dir / "sample2.png",
        ]

        report_path = self.viz_suite.create_evaluation_report(metrics, sample_paths, "test_report")

        # Should create HTML file
        assert report_path.suffix == ".html"
        assert "test_report" in report_path.name

        # Should write HTML content
        mock_file.write.assert_called_once()
        html_content = mock_file.write.call_args[0][0]
        assert "<html>" in html_content
        assert "mask_accuracy" in html_content
        assert "0.85" in html_content

    def test_structure_aware_visualization(self):
        """Test structure-aware specific visualization features."""
        # Sample with structure data
        sample = {
            "target_types": np.random.randint(0, 3, size=(4, 4, 4), dtype=np.uint8),
            "structure_mask": np.random.choice([True, False], size=(4, 4, 4)),
            "structure_types": np.random.randint(0, 2, size=(4, 4, 4), dtype=np.uint8),
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            output_paths = self.viz_suite.visualize_training_sample(sample, "structure_test")

        # Should include structure visualization
        assert any("structures" in path.name for path in output_paths)

    def test_torch_tensor_handling(self):
        """Test that visualization suite handles torch tensors correctly."""
        # Create sample with torch tensors
        sample = {
            "target_types": torch.randint(0, 3, (4, 4, 4), dtype=torch.uint8),
            "target_mask": torch.randint(0, 2, (4, 4, 4), dtype=torch.bool),
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            output_paths = self.viz_suite.visualize_training_sample(sample, "torch_test")

        # Should handle torch tensors without errors
        assert len(output_paths) >= 1

    def test_large_voxel_data_handling(self):
        """Test visualization with larger voxel chunks."""
        # Test with 16x16x16 data (full resolution)
        sample = {
            "target_types": np.random.randint(0, 8, size=(16, 16, 16), dtype=np.uint8),
            "target_mask": np.random.choice([True, False], size=(16, 16, 16)),
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            output_paths = self.viz_suite.visualize_training_sample(sample, "large_test")

        # Should handle larger data efficiently
        assert len(output_paths) >= 1
