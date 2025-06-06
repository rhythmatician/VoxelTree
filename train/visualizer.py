"""
VoxelTree visualizer for 3D model predictions.

Provides utilities to generate visualizations of model predictions,
including side-by-side comparisons with ground truth and input data.
Also includes TensorBoard integration for training monitoring.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from torch import Tensor

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard integration for VoxelTree training visualization.

    Handles logging of metrics, model graphs, embeddings, and 3D visualizations
    to TensorBoard for monitoring training progress.
    """

    def __init__(self, log_dir: Union[str, Path], enabled: bool = True):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            enabled: Whether TensorBoard logging is enabled
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.log_dir = Path(log_dir)
        self.writer = None

        if self.enabled:
            try:
                self.log_dir.mkdir(exist_ok=True, parents=True)
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging enabled at {self.log_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step/iteration number
            prefix: Optional prefix for metric names (e.g., 'train/' or 'val/')
        """
        if not self.enabled or not self.writer:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_name = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(metric_name, value, step)

    def log_model_graph(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """
        Log model architecture graph to TensorBoard.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (including batch dimension)
        """
        if not self.enabled or not self.writer:
            return

        try:
            # Create dummy input tensor for tracing
            dummy_input = torch.zeros(input_shape, device=next(model.parameters()).device)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")

    def log_voxel_batch(
        self, inputs: Tensor, predictions: Tensor, targets: Tensor, step: int, max_samples: int = 4
    ):
        """
        Log 3D voxel visualizations to TensorBoard.

        Args:
            inputs: Input voxel batch (B, C, D, H, W)
            predictions: Predicted voxel batch (B, C, D, H, W)
            targets: Target voxel batch (B, C, D, H, W)
            step: Global step/iteration number
            max_samples: Maximum number of samples to visualize
        """
        if not self.enabled or not self.writer:
            return

        # Limit number of samples to visualize
        batch_size = inputs.shape[0]
        num_samples = min(batch_size, max_samples)

        try:
            for i in range(num_samples):
                # Convert to numpy for visualization
                input_np = inputs[i].detach().cpu().numpy()
                pred_np = predictions[i].detach().cpu().numpy()
                target_np = targets[i].detach().cpu().numpy()

                # Use VoxelVisualizer to create visualization
                fig = VoxelVisualizer.visualize_comparison(
                    input_np, pred_np, target_np, return_figure=True
                )

                if fig:
                    self.writer.add_figure(f"voxel_sample_{i}", fig, step)
        except Exception as e:
            logger.warning(f"Failed to log voxel visualizations: {e}")

    def log_embedding(
        self, features: Tensor, metadata: List[str], step: int, tag: str = "embeddings"
    ):
        """
        Log embeddings (e.g., LOD embeddings) to TensorBoard.

        Args:
            features: Feature tensor (N, D) where N is number of samples and D is dimensionality
            metadata: Labels for each embedding point
            step: Global step/iteration number
            tag: Tag name for the embedding visualization
        """
        if not self.enabled or not self.writer:
            return

        try:
            self.writer.add_embedding(features, metadata=metadata, tag=tag, global_step=step)
        except Exception as e:
            logger.warning(f"Failed to log embeddings: {e}")

    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()


class VoxelVisualizer:
    """
    Visualization utilities for VoxelTree model predictions.

    Generates static and animated visualizations of voxel data,
    including parent inputs, predictions, and ground truth targets.
    """

    @staticmethod
    def visualize_prediction(
        parent_voxel: Union[np.ndarray, Tensor],
        pred_mask: Union[np.ndarray, Tensor],
        pred_types: Union[np.ndarray, Tensor],
        target_mask: Optional[Union[np.ndarray, Tensor]] = None,
        target_types: Optional[Union[np.ndarray, Tensor]] = None,
        output_path: Optional[Path] = None,
        metadata: Optional[Dict] = None,
        mask_threshold: float = 0.5,
    ) -> Optional[Path]:
        """
        Generate a side-by-side visualization of model prediction.

        Args:
            parent_voxel: Parent voxel input (8³)
            pred_mask: Predicted mask probabilities or logits (16³)
            pred_types: Predicted type logits (C×16³)
            target_mask: Optional ground truth mask (16³)
            target_types: Optional ground truth types (16³)
            output_path: Path to save visualization
            metadata: Optional metadata to include in visualization
            mask_threshold: Threshold to binarize predicted mask

        Returns:
            Path to saved visualization if output_path provided, else None
        """
        # Convert to numpy if tensors
        if isinstance(parent_voxel, Tensor):
            parent_voxel = parent_voxel.detach().cpu().numpy()
        if isinstance(pred_mask, Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(pred_types, Tensor):
            pred_types = pred_types.detach().cpu().numpy()
        if isinstance(target_mask, Tensor) and target_mask is not None:
            target_mask = target_mask.detach().cpu().numpy()
        if isinstance(target_types, Tensor) and target_types is not None:
            target_types = target_types.detach().cpu().numpy()

        # Handle channel dimensions and batch dimensions
        if parent_voxel.ndim > 3:
            parent_voxel = parent_voxel.squeeze()
        if pred_mask.ndim > 3:
            pred_mask = pred_mask.squeeze()
        if target_mask is not None and target_mask.ndim > 3:
            target_mask = target_mask.squeeze()

        # Get predicted types
        if pred_types.ndim > 3:
            if pred_types.ndim == 5:  # (B, C, D, H, W)
                pred_types = pred_types[0]  # Take first batch
            pred_types = np.argmax(pred_types, axis=0)

        # Create figure
        num_plots = 3 if target_mask is None else 4
        fig = plt.figure(figsize=(15, 4))

        # Plot parent voxel
        ax1 = fig.add_subplot(1, num_plots, 1, projection="3d")
        VoxelVisualizer._plot_binary_voxel(ax1, parent_voxel, title="Parent Input (8³)")

        # Plot predicted mask
        ax2 = fig.add_subplot(1, num_plots, 2, projection="3d")
        pred_binary = pred_mask > mask_threshold
        VoxelVisualizer._plot_binary_voxel(ax2, pred_binary, title="Predicted Mask (16³)")

        # Plot predicted types (colored by type)
        ax3 = fig.add_subplot(1, num_plots, 3, projection="3d")
        VoxelVisualizer._plot_type_voxel(
            ax3, pred_binary, pred_types, title="Predicted Types (16³)"
        )

        # Plot ground truth if available
        if target_mask is not None and target_types is not None:
            ax4 = fig.add_subplot(1, num_plots, 4, projection="3d")
            VoxelVisualizer._plot_type_voxel(
                ax4, target_mask, target_types, title="Ground Truth (16³)"
            )

        # Add metadata
        if metadata:
            meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
            plt.figtext(0.5, 0.01, meta_str, ha="center")

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None

    @staticmethod
    def create_rotating_animation(
        voxel_data: Dict[str, Union[np.ndarray, Tensor]],
        output_path: Path,
        n_frames: int = 36,
        fps: int = 12,
        figsize: Tuple[int, int] = (15, 8),
    ) -> Path:
        """
        Create a rotating animation of 3D voxel data.

        Args:
            voxel_data: Dictionary with parent, pred_mask, pred_types, target_mask,
                       target_types entries
            output_path: Path to save animation
            n_frames: Number of frames in animation
            fps: Frames per second
            figsize: Figure size (width, height)

        Returns:
            Path to saved animation
        """
        # Convert tensors to numpy
        processed_data = {}
        for key, value in voxel_data.items():
            if value is None:
                processed_data[key] = None
                continue

            if isinstance(value, Tensor):
                value = value.detach().cpu().numpy()

            # Handle dimensions
            if key == "parent_voxel" and value.ndim > 3:
                value = value.squeeze()
            elif key == "pred_mask" and value.ndim > 3:
                value = value.squeeze()
            elif key == "target_mask" and value.ndim > 3:
                value = value.squeeze()
            elif key == "pred_types":
                if value.ndim == 5:  # (B, C, D, H, W)
                    value = value[0]  # Take first batch
                if value.ndim == 4:  # (C, D, H, W)
                    value = np.argmax(value, axis=0)

            processed_data[key] = value

        # Create figure
        num_plots = 3 if processed_data.get("target_mask") is None else 4
        fig = plt.figure(figsize=figsize)

        # Create subplot axes
        axes = []
        for i in range(num_plots):
            ax = fig.add_subplot(1, num_plots, i + 1, projection="3d")
            axes.append(ax)

        # Function to update plot for each frame
        def update(frame):
            # Rotate each plot
            angle = frame * (360 / n_frames)

            for ax in axes:
                ax.view_init(30, angle)

            return axes

        # Initialize plots
        VoxelVisualizer._plot_binary_voxel(
            axes[0], processed_data["parent_voxel"], title="Parent Input (8³)"
        )

        pred_binary = processed_data["pred_mask"] > 0.5
        VoxelVisualizer._plot_binary_voxel(axes[1], pred_binary, title="Predicted Mask (16³)")

        VoxelVisualizer._plot_type_voxel(
            axes[2], pred_binary, processed_data["pred_types"], title="Predicted Types (16³)"
        )

        if processed_data.get("target_mask") is not None:
            VoxelVisualizer._plot_type_voxel(
                axes[3],
                processed_data["target_mask"],
                processed_data["target_types"],
                title="Ground Truth (16³)",
            )

        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

        # Save animation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path, writer="pillow", fps=fps, dpi=100)
        plt.close(fig)

        return output_path

    @staticmethod
    def visualize_batch(
        batch_data: Dict[str, Tensor],
        model_outputs: Dict[str, Tensor],
        output_dir: Path,
        prefix: str = "batch_vis",
        max_samples: int = 4,
    ) -> List[Path]:
        """
        Visualize a batch of predictions.

        Args:
            batch_data: Dictionary with parent_voxel, target_mask, target_types
            model_outputs: Dictionary with air_mask_logits, block_type_logits
            output_dir: Directory to save visualizations
            prefix: Prefix for output filenames
            max_samples: Maximum number of samples to visualize

        Returns:
            List of paths to saved visualizations
        """
        batch_size = batch_data["parent_voxel"].shape[0]
        num_samples = min(batch_size, max_samples)
        output_paths = []

        for i in range(num_samples):
            # Extract data for this sample
            parent = batch_data["parent_voxel"][i]
            target_mask = batch_data["target_mask"][i] if "target_mask" in batch_data else None
            target_types = batch_data["target_types"][i] if "target_types" in batch_data else None

            pred_mask = torch.sigmoid(model_outputs["air_mask_logits"][i])
            pred_types = model_outputs["block_type_logits"][i]

            # Get metadata
            y_index = batch_data["y_index"][i].item() if "y_index" in batch_data else None
            lod = batch_data["lod"][i].item() if "lod" in batch_data else None

            metadata = {"sample_idx": i}
            if y_index is not None:
                metadata["y_index"] = y_index
            if lod is not None:
                metadata["lod"] = lod

            # Create output path
            output_path = output_dir / f"{prefix}_{i}.png"

            # Generate visualization
            vis_path = VoxelVisualizer.visualize_prediction(
                parent_voxel=parent,
                pred_mask=pred_mask,
                pred_types=pred_types,
                target_mask=target_mask,
                target_types=target_types,
                output_path=output_path,
                metadata=metadata,
            )

            output_paths.append(vis_path)

        return output_paths

    @staticmethod
    def _plot_binary_voxel(ax, voxels, title=None):
        """Plot binary voxel data on a 3D axis."""
        ax.voxels(voxels, edgecolor="k", alpha=0.3)
        if title:
            ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    @staticmethod
    def _plot_type_voxel(ax, mask, types, title=None):
        """Plot typed voxel data on a 3D axis with colors."""
        # Create colored voxel array
        if mask.ndim != 3 or types.ndim != 3:
            raise ValueError(f"Expected 3D arrays, got mask:{mask.shape}, types:{types.shape}")

        colored_voxels = np.zeros(mask.shape + (4,))  # RGBA

        # Fill with colors based on type
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if mask[i, j, k]:
                        block_type = types[i, j, k]
                        colored_voxels[i, j, k] = VoxelVisualizer._get_type_color(block_type)

        # Plot with colors
        ax.voxels(mask, facecolors=colored_voxels, edgecolor="k", alpha=0.3)
        if title:
            ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    @staticmethod
    def _get_type_color(block_type: int) -> Tuple[float, float, float, float]:
        """Get RGBA color for a block type."""
        # Define some preset colors for common block types
        common_colors = {
            0: (0.5, 0.5, 0.5, 1.0),  # Stone (gray)
            1: (0.6, 0.4, 0.2, 1.0),  # Dirt (brown)
            2: (0.0, 0.7, 0.0, 1.0),  # Grass (green)
            3: (0.9, 0.9, 0.2, 1.0),  # Sand (yellow)
            4: (0.2, 0.2, 0.9, 1.0),  # Water (blue)
            5: (0.6, 0.0, 0.0, 1.0),  # Ore (red)
        }

        if block_type in common_colors:
            return common_colors[block_type]

        # For other block types, generate a color based on hash
        # Use golden ratio to distribute colors nicely
        golden_ratio = 0.618033988749895
        h = (block_type * golden_ratio) % 1
        s = 0.7
        v = 0.95

        # HSV to RGB conversion
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        return r, g, b, 1.0  # RGBA
