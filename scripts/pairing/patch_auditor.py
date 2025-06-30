"""
PatchQualityAuditor - Data quality audit tools for VoxelTree.

This module provides tools to analyze data characteristics and quality metrics
for training examples before full-scale training.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scripts.pairing.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class PatchQualityAuditor:
    """
    Analyzes data quality and characteristics of training examples.

    Provides metrics, histograms, and visualizations of data distributions
    to ensure dataset quality before training.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize PatchQualityAuditor with optional config path."""
        self.config_path = config_path
        self.validator = PatchValidator(config_path=config_path)
        self.metrics_cache = {}

        logger.info("PatchQualityAuditor initialized")

    def audit_patch_quality(self, patch_file: Path) -> Dict[str, Any]:
        """
        Perform comprehensive quality audit on a training example.

        Args:
            patch_file: Path to patch .npz file

        Returns:
            Dictionary of quality metrics
        """
        # First validate format correctness
        if not self.validator.validate_file(patch_file):
            errors = self.validator.get_file_errors(patch_file)
            return {"valid": False, "errors": errors, "metrics": {}, "filename": patch_file.name}

        # Load the file
        try:
            data = np.load(patch_file)
            example = {key: data[key] for key in data.keys()}

            # Extract basic metrics
            metrics = {
                "air_ratio": self._compute_air_ratio(example),
                "block_type_counts": self._compute_block_type_distribution(example),
                "y_index": int(example["y_index"]),
                "lod": int(example["lod"]),
                "spatial_entropy": self._compute_spatial_entropy(example),
                "has_biome_variance": self._has_biome_variance(example),
                "heightmap_stats": self._compute_heightmap_stats(example),
                "river_coverage": self._compute_river_coverage(example),
            }

            # Compute quality score (0-100)
            quality_score = self._compute_quality_score(metrics)
            metrics["quality_score"] = quality_score

            return {"valid": True, "metrics": metrics, "filename": patch_file.name}

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Error analyzing patch: {str(e)}"],
                "metrics": {},
                "filename": patch_file.name,
            }

    def audit_dataset(self, dataset_dir: Path, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Audit a dataset directory and generate aggregate statistics.

        Args:
            dataset_dir: Directory containing training examples
            sample_size: Optional limit on number of files to analyze

        Returns:
            Aggregate dataset quality metrics
        """
        npz_files = list(dataset_dir.glob("*.npz"))

        if not npz_files:
            logger.warning(f"No .npz files found in {dataset_dir}")
            return {"valid": False, "message": "No files found"}

        # Sample if requested
        if sample_size and sample_size < len(npz_files):
            import random

            npz_files = random.sample(npz_files, sample_size)

        # Analyze each file
        results = []
        valid_count = 0
        invalid_count = 0

        logger.info(f"Auditing {len(npz_files)} files in {dataset_dir}")

        for file_path in tqdm(npz_files, desc="Auditing patches"):
            audit_result = self.audit_patch_quality(file_path)
            results.append(audit_result)

            if audit_result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(results)

        return {
            "total_files": len(npz_files),
            "valid_files": valid_count,
            "invalid_files": invalid_count,
            "success_rate": valid_count / max(1, len(npz_files)),
            "aggregate_metrics": aggregate_metrics,
            "sample_results": results[:10],  # Include first 10 for reference
        }

    def visualize_dataset_metrics(
        self, audit_result: Dict[str, Any], output_dir: Path
    ) -> List[Path]:
        """
        Generate visualizations for dataset metrics.

        Args:
            audit_result: Result from audit_dataset method
            output_dir: Directory to save visualizations

        Returns:
            List of paths to generated visualization files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []

        # Skip visualization if no valid data
        if audit_result.get("valid_files", 0) == 0:
            return []

        aggregate_metrics = audit_result.get("aggregate_metrics", {})

        # 1. Air ratio distribution
        if "air_ratio_distribution" in aggregate_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            values = aggregate_metrics["air_ratio_distribution"]
            ax.hist(values, bins=20, alpha=0.7)
            ax.set_title("Air Ratio Distribution")
            ax.set_xlabel("Air Ratio")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "air_ratio_distribution.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_files.append(output_path)

        # 2. Block type distribution
        if "block_type_distribution" in aggregate_metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            block_counts = aggregate_metrics["block_type_distribution"]

            # Get top 20 block types
            top_blocks = sorted(block_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            blocks, counts = zip(*top_blocks) if top_blocks else ([], [])

            ax.bar(blocks, counts)
            ax.set_title("Top 20 Block Type Distribution")
            ax.set_xlabel("Block ID")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "block_type_distribution.png"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            output_files.append(output_path)

        # 3. Y-index distribution
        if "y_index_distribution" in aggregate_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            y_counts = aggregate_metrics["y_index_distribution"]
            y_indices = sorted(y_counts.keys())
            y_values = [y_counts[y] for y in y_indices]

            ax.bar(y_indices, y_values)
            ax.set_title("Y-Index Distribution")
            ax.set_xlabel("Y-Index")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "y_index_distribution.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_files.append(output_path)

        # 4. LOD level distribution
        if "lod_distribution" in aggregate_metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            lod_counts = aggregate_metrics["lod_distribution"]
            lod_levels = sorted(lod_counts.keys())
            lod_values = [lod_counts[lod] for lod in lod_levels]

            ax.bar(lod_levels, lod_values)
            ax.set_title("LOD Level Distribution")
            ax.set_xlabel("LOD Level")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "lod_distribution.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_files.append(output_path)

        # 5. Quality score distribution
        if "quality_score_distribution" in aggregate_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            scores = aggregate_metrics["quality_score_distribution"]
            ax.hist(scores, bins=10, range=(0, 100), alpha=0.7)
            ax.set_title("Quality Score Distribution")
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            output_path = output_dir / "quality_score_distribution.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_files.append(output_path)

        return output_files

    def render_voxel_patch(
        self, patch_file: Path, output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Render a 3D visualization of a training example using matplotlib.

        Args:
            patch_file: Path to patch .npz file
            output_path: Optional path to save the visualization

        Returns:
            Path to the saved visualization, or None if rendering failed
        """
        try:
            data = np.load(patch_file)

            # Create a figure with 3D plots for parent and target
            fig = plt.figure(figsize=(15, 8))

            # Parent voxel (smaller resolution)
            if "parent_voxel" in data:
                ax1 = fig.add_subplot(1, 2, 1, projection="3d")
                parent_voxel = data["parent_voxel"]
                self._plot_voxels(ax1, parent_voxel, title="Parent Voxel (8³)")

            # Target voxel (higher resolution)
            if "target_mask" in data and "target_types" in data:
                ax2 = fig.add_subplot(1, 2, 2, projection="3d")
                target_mask = data["target_mask"]
                target_types = data["target_types"]

                # Create a colored representation based on block types
                # But only where the mask is True (solid blocks)
                colored_voxels = np.zeros(target_mask.shape + (3,), dtype=float)

                # Simple coloring by block type
                for i in range(target_mask.shape[0]):
                    for j in range(target_mask.shape[1]):
                        for k in range(target_mask.shape[2]):
                            if target_mask[i, j, k]:
                                block_type = target_types[i, j, k]
                                # Generate a deterministic color based on block type
                                colored_voxels[i, j, k] = self._block_id_to_color(block_type)

                self._plot_colored_voxels(
                    ax2, target_mask, colored_voxels, title="Target Voxel (16³)"
                )

            # Add metadata
            metadata = {}
            for key in ["y_index", "lod", "chunk_x", "chunk_z"]:
                if key in data:
                    metadata[key] = data[key].item() if hasattr(data[key], "item") else data[key]

            plt.figtext(0.5, 0.01, f"Metadata: {metadata}", ha="center")
            plt.tight_layout()

            # Save or show
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                plt.show()
                plt.close(fig)
                return None

        except Exception as e:
            logger.error(f"Error rendering voxel patch {patch_file}: {e}")
            return None

    def _plot_voxels(self, ax, voxels, title=None):
        """Plot boolean voxel data on a matplotlib 3D axis."""
        ax.voxels(voxels, edgecolor="k", alpha=0.5)
        ax.set_title(title or "Voxel Data")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _plot_colored_voxels(self, ax, voxels, colors, title=None):
        """Plot colored voxel data on a matplotlib 3D axis."""
        ax.voxels(voxels, facecolors=colors, edgecolor="k", alpha=0.5)
        ax.set_title(title or "Voxel Data")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _block_id_to_color(self, block_id: int) -> Tuple[float, float, float]:
        """Convert block ID to RGB color values."""
        # Simple hash function to get consistent colors
        # Use the golden ratio to distribute colors nicely
        golden_ratio = 0.618033988749895
        h = (block_id * golden_ratio) % 1

        # HSV to RGB conversion for a pleasant color palette
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = 0.6  # Fixed value to avoid too light/dark colors
        q = 0.6 * (1 - f)
        t = 0.6 * f

        if h_i == 0:
            return (0.6, t, p)
        elif h_i == 1:
            return (q, 0.6, p)
        elif h_i == 2:
            return (p, 0.6, t)
        elif h_i == 3:
            return (p, q, 0.6)
        elif h_i == 4:
            return (t, p, 0.6)
        else:
            return (0.6, p, q)

    def _compute_air_ratio(self, example: Dict[str, Any]) -> float:
        """Compute ratio of air blocks to total blocks."""
        if "target_mask" not in example:
            return 0.0

        target_mask = example["target_mask"]
        total_blocks = target_mask.size
        solid_blocks = np.sum(target_mask)
        air_blocks = total_blocks - solid_blocks

        return float(air_blocks) / total_blocks if total_blocks > 0 else 0.0

    def _compute_block_type_distribution(self, example: Dict[str, Any]) -> Dict[int, int]:
        """Compute distribution of block types in target data."""
        if "target_types" not in example or "target_mask" not in example:
            return {}

        target_types = example["target_types"]
        target_mask = example["target_mask"]

        # Only count blocks where mask is True (solid blocks)
        masked_types = target_types[target_mask]

        # Count occurrences of each block type
        block_counts = Counter(masked_types.flatten())
        return dict(block_counts)

    def _compute_spatial_entropy(self, example: Dict[str, Any]) -> float:
        """Compute spatial entropy/complexity of the voxel structure."""
        if "target_mask" not in example:
            return 0.0

        target_mask = example["target_mask"]

        # Simple spatial complexity measure:
        # Count transitions between air/solid along each axis
        transitions = 0

        # X axis
        for y in range(target_mask.shape[1]):
            for z in range(target_mask.shape[2]):
                prev = False
                for x in range(target_mask.shape[0]):
                    curr = target_mask[x, y, z]
                    if curr != prev:
                        transitions += 1
                    prev = curr

        # Y axis
        for x in range(target_mask.shape[0]):
            for z in range(target_mask.shape[2]):
                prev = False
                for y in range(target_mask.shape[1]):
                    curr = target_mask[x, y, z]
                    if curr != prev:
                        transitions += 1
                    prev = curr

        # Z axis
        for x in range(target_mask.shape[0]):
            for y in range(target_mask.shape[1]):
                prev = False
                for z in range(target_mask.shape[2]):
                    curr = target_mask[x, y, z]
                    if curr != prev:
                        transitions += 1
                    prev = curr

        # Normalize by total number of possible transitions
        max_transitions = (
            target_mask.shape[0] * target_mask.shape[1]
            + target_mask.shape[0] * target_mask.shape[2]
            + target_mask.shape[1] * target_mask.shape[2]
        ) * 2  # Multiply by 2 as we could have at most 1 transition per cell

        return transitions / max_transitions if max_transitions > 0 else 0.0

    def _has_biome_variance(self, example: Dict[str, Any]) -> bool:
        """Check if the example has variance in biome types."""
        if "biome_patch" not in example:
            return False

        biome_patch = example["biome_patch"]
        unique_biomes = np.unique(biome_patch)

        return len(unique_biomes) > 1

    def _compute_heightmap_stats(self, example: Dict[str, Any]) -> Dict[str, float]:
        """Compute statistics about the heightmap."""
        if "heightmap_patch" not in example:
            return {}

        heightmap = example["heightmap_patch"]

        return {
            "min": float(np.min(heightmap)),
            "max": float(np.max(heightmap)),
            "mean": float(np.mean(heightmap)),
            "std": float(np.std(heightmap)),
            "variation": float(np.max(heightmap) - np.min(heightmap)),
        }

    def _compute_river_coverage(self, example: Dict[str, Any]) -> float:
        """Compute the coverage percentage of river features."""
        if "river_patch" not in example:
            return 0.0

        river_patch = example["river_patch"]
        # Use a threshold to determine river presence
        threshold = 0.3  # Adjust based on the specific river noise distribution

        river_cells = np.sum(river_patch > threshold)
        total_cells = river_patch.size

        return float(river_cells) / total_cells if total_cells > 0 else 0.0

    def _compute_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Compute an overall quality score based on collected metrics."""
        score = 50  # Base score

        # Add points for features that indicate good quality

        # 1. Air ratio - should be balanced (not all air, not all solid)
        air_ratio = metrics.get("air_ratio", 0.5)
        if 0.3 <= air_ratio <= 0.7:  # Good balance
            score += 10
        elif 0.1 <= air_ratio <= 0.9:  # Acceptable balance
            score += 5
        else:  # Too much air or too solid
            score -= 5

        # 2. Spatial complexity - higher is better
        spatial_entropy = metrics.get("spatial_entropy", 0.0)
        score += spatial_entropy * 20  # Max 20 points for high complexity

        # 3. Biome variance - better to have multiple biomes
        if metrics.get("has_biome_variance", False):
            score += 5

        # 4. Heightmap variation - some variation is good
        heightmap_stats = metrics.get("heightmap_stats", {})
        height_variation = heightmap_stats.get("variation", 0)
        if height_variation > 10:  # Significant terrain variation
            score += 10
        elif height_variation > 5:  # Some terrain variation
            score += 5

        # Ensure score is in range 0-100
        return max(0, min(100, score))

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple patch results."""
        valid_results = [r for r in results if r["valid"]]

        if not valid_results:
            return {}

        # Initialize aggregated metrics
        aggregate = {
            "air_ratio_distribution": [],
            "block_type_distribution": Counter(),
            "spatial_entropy_distribution": [],
            "y_index_distribution": Counter(),
            "lod_distribution": Counter(),
            "biome_variance_count": 0,
            "quality_score_distribution": [],
        }

        # Collect metrics
        for result in valid_results:
            metrics = result.get("metrics", {})

            # Air ratio
            if "air_ratio" in metrics:
                aggregate["air_ratio_distribution"].append(metrics["air_ratio"])

            # Block type counts
            if "block_type_counts" in metrics:
                aggregate["block_type_distribution"].update(metrics["block_type_counts"])

            # Spatial entropy
            if "spatial_entropy" in metrics:
                aggregate["spatial_entropy_distribution"].append(metrics["spatial_entropy"])

            # Y index
            if "y_index" in metrics:
                aggregate["y_index_distribution"][metrics["y_index"]] += 1

            # LOD level
            if "lod" in metrics:
                aggregate["lod_distribution"][metrics["lod"]] += 1

            # Biome variance
            if metrics.get("has_biome_variance", False):
                aggregate["biome_variance_count"] += 1

            # Quality score
            if "quality_score" in metrics:
                aggregate["quality_score_distribution"].append(metrics["quality_score"])

        # Calculate summary statistics
        aggregate["air_ratio_mean"] = (
            np.mean(aggregate["air_ratio_distribution"])
            if aggregate["air_ratio_distribution"]
            else 0
        )
        aggregate["air_ratio_std"] = (
            np.std(aggregate["air_ratio_distribution"])
            if aggregate["air_ratio_distribution"]
            else 0
        )

        aggregate["spatial_entropy_mean"] = (
            np.mean(aggregate["spatial_entropy_distribution"])
            if aggregate["spatial_entropy_distribution"]
            else 0
        )

        aggregate["biome_variance_rate"] = (
            aggregate["biome_variance_count"] / len(valid_results) if valid_results else 0
        )

        aggregate["quality_score_mean"] = (
            np.mean(aggregate["quality_score_distribution"])
            if aggregate["quality_score_distribution"]
            else 0
        )
        aggregate["quality_score_std"] = (
            np.std(aggregate["quality_score_distribution"])
            if aggregate["quality_score_distribution"]
            else 0
        )

        # Convert Counters to regular dicts for easier serialization
        aggregate["block_type_distribution"] = dict(aggregate["block_type_distribution"])
        aggregate["y_index_distribution"] = dict(aggregate["y_index_distribution"])
        aggregate["lod_distribution"] = dict(aggregate["lod_distribution"])

        # Count total blocks by type
        aggregate["total_block_count"] = sum(aggregate["block_type_distribution"].values())

        return aggregate
