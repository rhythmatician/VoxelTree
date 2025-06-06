"""
VoxelTree 3D Visualization Module - Phase 6.3 Implementation

This module provides 3D voxel rendering capabilities for VoxelTree model outputs
and training data visualization. Supports both matplotlib and web-based previews.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)


class VoxelRenderer:
    """3D voxel renderer for VoxelTree training data and model outputs."""

    def __init__(self, block_colors: Optional[Dict[int, str]] = None):
        """
        Initialize voxel renderer.

        Args:
            block_colors: Dictionary mapping block type IDs to hex colors
        """
        self.block_colors = block_colors or self._get_default_minecraft_colors()
        
    def _get_default_minecraft_colors(self) -> Dict[int, str]:
        """Get default Minecraft-inspired block colors."""
        return {
            0: "#87CEEB",    # Air (sky blue)
            1: "#8B4513",    # Stone (brown)
            2: "#228B22",    # Grass (green)
            3: "#8B4513",    # Dirt (brown)
            4: "#696969",    # Cobblestone (gray)
            5: "#DEB887",    # Wood (tan)
            6: "#32CD32",    # Leaves (lime green)
            7: "#000080",    # Water (navy)
            8: "#FFD700",    # Sand (gold)
            9: "#DC143C",    # Lava (red)
            10: "#4169E1",   # Ice (royal blue)
            11: "#FFFFFF",   # Snow (white)
            12: "#A0522D",   # Clay (sienna)
            13: "#FF6347",   # Netherrack (tomato)
            14: "#800080",   # Obsidian (purple)
            15: "#C0C0C0",   # Iron ore (silver)
        }

    def render_voxel_chunk(
        self,
        voxel_data: Union[np.ndarray, torch.Tensor],
        air_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Voxel Chunk",
        alpha: float = 0.8,
        figsize: Tuple[int, int] = (12, 10),
        view_angles: Tuple[float, float] = (30, 45),
    ) -> plt.Figure:
        """
        Render a 3D voxel chunk using matplotlib.

        Args:
            voxel_data: 3D array of block types (H, W, D) or (B, H, W, D)
            air_mask: Optional 3D boolean mask for air blocks
            output_path: Optional path to save the rendered image
            title: Title for the plot
            alpha: Transparency for voxel blocks
            figsize: Figure size tuple
            view_angles: Elevation and azimuth angles for 3D view

        Returns:
            matplotlib Figure object
        """
        # Convert to numpy if torch tensor
        if isinstance(voxel_data, torch.Tensor):
            voxel_data = voxel_data.detach().cpu().numpy()
        if isinstance(air_mask, torch.Tensor):
            air_mask = air_mask.detach().cpu().numpy()
            
        # Handle batch dimension
        if voxel_data.ndim == 4:
            voxel_data = voxel_data[0]  # Take first batch item
        if air_mask is not None and air_mask.ndim == 4:
            air_mask = air_mask[0]
            
        # Create figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique block types
        unique_blocks = np.unique(voxel_data)
        
        # Create colormap for block types
        colors = []
        for block_id in unique_blocks:
            color = self.block_colors.get(int(block_id), "#808080")  # Default gray
            colors.append(color)
        
        # Render each block type separately
        for i, block_id in enumerate(unique_blocks):
            # Skip air blocks if air_mask is provided
            if air_mask is not None and block_id == 0:
                continue
                
            # Get positions of this block type
            positions = np.where(voxel_data == block_id)
            
            # Filter out air blocks if mask provided
            if air_mask is not None:
                mask_positions = np.where(~air_mask[positions])
                positions = tuple(pos[mask_positions] for pos in positions)
            
            if len(positions[0]) > 0:  # Only render if blocks exist
                ax.scatter(
                    positions[0], positions[1], positions[2],
                    c=colors[i], s=30, alpha=alpha,
                    label=f"Block {int(block_id)}"
                )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set view angle
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        # Add legend if not too many block types
        if len(unique_blocks) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved voxel render to {output_path}")
        
        return fig

    def render_comparison(
        self,
        ground_truth: Union[np.ndarray, torch.Tensor],
        prediction: Union[np.ndarray, torch.Tensor],
        output_path: Optional[Union[str, Path]] = None,
        title_prefix: str = "Voxel Comparison",
        figsize: Tuple[int, int] = (20, 8),
    ) -> plt.Figure:
        """
        Render side-by-side comparison of ground truth vs prediction.

        Args:
            ground_truth: Ground truth voxel data
            prediction: Predicted voxel data  
            output_path: Optional path to save the comparison
            title_prefix: Prefix for subplot titles
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        # Convert to numpy if needed
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
            
        # Handle batch dimension
        if ground_truth.ndim == 4:
            ground_truth = ground_truth[0]
        if prediction.ndim == 4:
            prediction = prediction[0]
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Ground truth subplot
        ax1 = fig.add_subplot(121, projection='3d')
        self._render_single_voxel(ground_truth, ax1, f"{title_prefix} - Ground Truth")
        
        # Prediction subplot
        ax2 = fig.add_subplot(122, projection='3d')
        self._render_single_voxel(prediction, ax2, f"{title_prefix} - Prediction")
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved comparison render to {output_path}")
        
        return fig

    def _render_single_voxel(
        self, voxel_data: np.ndarray, ax, title: str, alpha: float = 0.8
    ):
        """Helper method to render a single voxel chunk on given axis."""
        unique_blocks = np.unique(voxel_data)
        
        for block_id in unique_blocks:
            if block_id == 0:  # Skip air
                continue
                
            positions = np.where(voxel_data == block_id)
            color = self.block_colors.get(int(block_id), "#808080")
            
            if len(positions[0]) > 0:
                ax.scatter(
                    positions[0], positions[1], positions[2],
                    c=color, s=20, alpha=alpha
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=30, azim=45)

    def render_structure_overlay(
        self,
        voxel_data: Union[np.ndarray, torch.Tensor],
        structure_mask: Union[np.ndarray, torch.Tensor],
        structure_types: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Structure-Aware Voxels",
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Render voxels with structure overlay highlighting.

        Args:
            voxel_data: 3D array of block types
            structure_mask: 3D boolean mask for structure regions
            structure_types: Optional structure type labels
            output_path: Optional path to save the rendered image
            title: Title for the plot
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        # Convert to numpy if needed
        if isinstance(voxel_data, torch.Tensor):
            voxel_data = voxel_data.detach().cpu().numpy()
        if isinstance(structure_mask, torch.Tensor):
            structure_mask = structure_mask.detach().cpu().numpy()
        if isinstance(structure_types, torch.Tensor):
            structure_types = structure_types.detach().cpu().numpy()
            
        # Handle batch dimension
        if voxel_data.ndim == 4:
            voxel_data = voxel_data[0]
        if structure_mask.ndim == 4:
            structure_mask = structure_mask[0]
        if structure_types is not None and structure_types.ndim == 3:
            structure_types = structure_types[0]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Render regular voxels (non-structure)
        non_structure_mask = ~structure_mask
        non_structure_positions = np.where(non_structure_mask & (voxel_data != 0))
        
        if len(non_structure_positions[0]) > 0:
            block_ids = voxel_data[non_structure_positions]
            colors = [self.block_colors.get(int(bid), "#808080") for bid in block_ids]
            
            ax.scatter(
                non_structure_positions[0], non_structure_positions[1], non_structure_positions[2],
                c=colors, s=20, alpha=0.5, label="Regular blocks"
            )
        
        # Render structure voxels with highlight
        structure_positions = np.where(structure_mask & (voxel_data != 0))
        
        if len(structure_positions[0]) > 0:
            ax.scatter(
                structure_positions[0], structure_positions[1], structure_positions[2],
                c='red', s=40, alpha=0.9, marker='^', label="Structure blocks"
            )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=30, azim=45)
        ax.legend()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved structure overlay render to {output_path}")
        
        return fig


class VoxelVisualizationSuite:
    """Complete visualization suite for VoxelTree evaluation."""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize visualization suite.

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.renderer = VoxelRenderer()
        logger.info(f"VoxelVisualizationSuite initialized, output: {self.output_dir}")

    def visualize_training_sample(
        self,
        sample: Dict[str, Union[np.ndarray, torch.Tensor]],
        sample_id: str = "sample",
    ) -> List[Path]:
        """
        Create comprehensive visualization of a training sample.

        Args:
            sample: Training sample dictionary
            sample_id: Identifier for the sample

        Returns:
            List of paths to generated visualization files
        """
        output_paths = []
        
        # Render parent voxel (8x8x8)
        if "parent_voxel" in sample:
            parent_path = self.output_dir / f"{sample_id}_parent.png"
            self.renderer.render_voxel_chunk(
                sample["parent_voxel"],
                output_path=parent_path,
                title=f"Parent Voxel - {sample_id}",
                figsize=(10, 8)
            )
            output_paths.append(parent_path)
            plt.close()
        
        # Render target chunk (16x16x16)
        if "target_types" in sample:
            target_path = self.output_dir / f"{sample_id}_target.png"
            air_mask = sample.get("target_mask")
            self.renderer.render_voxel_chunk(
                sample["target_types"],
                air_mask=air_mask,
                output_path=target_path,
                title=f"Target Chunk - {sample_id}",
                figsize=(12, 10)
            )
            output_paths.append(target_path)
            plt.close()
        
        # Render structure overlay if available
        if "structure_mask" in sample and "target_types" in sample:
            structure_path = self.output_dir / f"{sample_id}_structures.png"
            self.renderer.render_structure_overlay(
                sample["target_types"],
                sample["structure_mask"],
                structure_types=sample.get("structure_types"),
                output_path=structure_path,
                title=f"Structure Overlay - {sample_id}"
            )
            output_paths.append(structure_path)
            plt.close()
        
        logger.info(f"Generated {len(output_paths)} visualizations for {sample_id}")
        return output_paths

    def visualize_model_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch_idx: int = 0,
        sample_id: str = "prediction",
    ) -> List[Path]:
        """
        Visualize model predictions vs ground truth.

        Args:
            predictions: Model prediction dictionary
            targets: Ground truth targets dictionary
            batch_idx: Batch index to visualize
            sample_id: Identifier for the sample

        Returns:
            List of paths to generated visualization files
        """
        output_paths = []
        
        # Extract single sample from batch
        pred_sample = {k: v[batch_idx] if v.dim() > 3 else v for k, v in predictions.items()}
        target_sample = {k: v[batch_idx] if v.dim() > 3 else v for k, v in targets.items()}
        
        # Convert logits to predictions
        if "air_mask_logits" in pred_sample:
            pred_sample["air_mask"] = torch.sigmoid(pred_sample["air_mask_logits"]) > 0.5
        
        if "block_type_logits" in pred_sample:
            pred_sample["block_types"] = torch.argmax(pred_sample["block_type_logits"], dim=0)
        
        # Render block type comparison
        if "block_types" in pred_sample and "block_types" in target_sample:
            comparison_path = self.output_dir / f"{sample_id}_comparison.png"
            self.renderer.render_comparison(
                target_sample["block_types"],
                pred_sample["block_types"],
                output_path=comparison_path,
                title_prefix=f"Block Types - {sample_id}"
            )
            output_paths.append(comparison_path)
            plt.close()
        
        # Render air mask comparison
        if "air_mask" in pred_sample and "air_mask" in target_sample:
            mask_comparison_path = self.output_dir / f"{sample_id}_mask_comparison.png"
            self.renderer.render_comparison(
                target_sample["air_mask"].float(),
                pred_sample["air_mask"].float(),
                output_path=mask_comparison_path,
                title_prefix=f"Air Mask - {sample_id}"
            )
            output_paths.append(mask_comparison_path)
            plt.close()
        
        # Render structure comparison if available
        if "structure_mask" in pred_sample and "structure_mask" in target_sample:
            struct_comparison_path = self.output_dir / f"{sample_id}_structure_comparison.png"
            self.renderer.render_comparison(
                target_sample["structure_mask"].float(),
                pred_sample["structure_mask"].float(),
                output_path=struct_comparison_path,
                title_prefix=f"Structure Mask - {sample_id}"
            )
            output_paths.append(struct_comparison_path)
            plt.close()
        
        logger.info(f"Generated {len(output_paths)} prediction visualizations for {sample_id}")
        return output_paths

    def create_evaluation_report(
        self,
        metrics: Dict[str, float],
        sample_paths: List[Path],
        report_name: str = "evaluation_report",
    ) -> Path:
        """
        Create an HTML evaluation report with metrics and visualizations.

        Args:
            metrics: Dictionary of evaluation metrics
            sample_paths: List of paths to visualization images
            report_name: Name for the HTML report

        Returns:
            Path to generated HTML report
        """
        report_path = self.output_dir / f"{report_name}.html"
        
        # Generate HTML content
        html_content = self._generate_html_report(metrics, sample_paths, report_name)
        
        # Write HTML file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated evaluation report: {report_path}")
        return report_path

    def _generate_html_report(
        self, metrics: Dict[str, float], sample_paths: List[Path], title: str
    ) -> str:
        """Generate HTML content for evaluation report."""
        # Sort metrics by category
        mask_metrics = {k: v for k, v in metrics.items() if "mask" in k}
        block_metrics = {k: v for k, v in metrics.items() if "block" in k}
        structure_metrics = {k: v for k, v in metrics.items() if "structure" in k}
        iou_dice_metrics = {k: v for k, v in metrics.items() if "iou" in k or "dice" in k}
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-section {{ margin-bottom: 30px; }}
        .metric-table {{ border-collapse: collapse; width: 100%; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; }}
        .metric-table th {{ background-color: #f2f2f2; }}
        .visualization {{ margin: 20px 0; }}
        .visualization img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="metric-section">
        <h2>Air Mask Metrics</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
        
        for metric, value in mask_metrics.items():
            html += f"            <tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"
        
        html += """        </table>
    </div>
    
    <div class="metric-section">
        <h2>Block Type Metrics</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
        
        for metric, value in block_metrics.items():
            html += f"            <tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"
        
        html += """        </table>
    </div>
"""
        
        if structure_metrics:
            html += """    
    <div class="metric-section">
        <h2>Structure-Aware Metrics</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
            for metric, value in structure_metrics.items():
                html += f"            <tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"
            
            html += """        </table>
    </div>
"""
        
        if iou_dice_metrics:
            html += """
    <div class="metric-section">
        <h2>IoU / Dice Metrics</h2>
        <table class="metric-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
            for metric, value in iou_dice_metrics.items():
                html += f"            <tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"
            
            html += """        </table>
    </div>
"""
        
        # Add visualizations
        html += """
    <div class="metric-section">
        <h2>Visualizations</h2>
"""
        
        for img_path in sample_paths:
            rel_path = img_path.name  # Use relative path
            html += f"""        <div class="visualization">
            <h3>{img_path.stem}</h3>
            <img src="{rel_path}" alt="{img_path.stem}">
        </div>
"""
        
        html += """    </div>
</body>
</html>"""
        
        return html
