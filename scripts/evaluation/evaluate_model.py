#!/usr/bin/env python3
"""
VoxelTree Model Evaluation CLI

This script provides comprehensive evaluation of trained VoxelTree models,
including structure-aware metrics for Phase 2 validation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.evaluation.metrics import AccuracyMetrics, DiceCalculator, IoUCalculator
from train.dataset import VoxelTreeDataset
from train.unet3d import VoxelUNet3D


class ModelEvaluator:
    """Main evaluator for VoxelTree models."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto"):
        """
        Initialize model evaluator.

        Args:
            config_path: Path to evaluation config file
            checkpoint_path: Path to model checkpoint
            device: Device to use for evaluation
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device(device)
        self.model = self._load_model(checkpoint_path)
        self.metrics = AccuracyMetrics()
        self.iou_calc = IoUCalculator()
        self.dice_calc = DiceCalculator()

    def _load_config(self, config_path: str) -> Dict:
        """Load evaluation configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    def _load_model(self, checkpoint_path: str) -> VoxelUNet3D:
        """Load trained model from checkpoint."""
        # Initialize model with config
        model_config = self.config.get("model", {})
        model = VoxelUNet3D(
            in_channels=model_config.get("in_channels", 1),
            out_channels=model_config.get("out_channels", 2),
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 4),
            use_structure_conditioning=model_config.get("structure_aware", True),
            use_heightmap_conditioning=model_config.get("heightmap_conditioning", True),
            use_biome_conditioning=model_config.get("biome_conditioning", True),
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def evaluate_dataset(self, dataset_path: str) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataset_path: Path to evaluation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        # Create dataset and dataloader
        dataset = VoxelTreeDataset(
            data_dir=dataset_path, config=self.config.get("dataset", {}), mode="eval"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("eval_batch_size", 8),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True if self.device.type == "cuda" else False,
        )

        print(f"Evaluating on {len(dataset)} samples...")

        all_metrics = []
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Forward pass
                predictions = self._forward_pass(batch)

                # Extract targets
                targets = self._extract_targets(batch)

                # Compute metrics
                batch_metrics = self._compute_batch_metrics(predictions, targets)
                all_metrics.append(batch_metrics)

                total_samples += batch["parent_voxel"].size(0)

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        print(f"\nEvaluation complete. Processed {total_samples} samples.")
        return aggregated_metrics

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run model forward pass."""
        # Prepare inputs
        parent_voxel = batch["parent_voxel"]

        # Optional conditioning inputs
        biome_patch = batch.get("biome_patch")
        heightmap = batch.get("heightmap")
        structure_mask = batch.get("structure_mask")
        structure_types = batch.get("structure_types")
        structure_positions = batch.get("structure_positions")

        # Forward pass
        outputs = self.model(
            parent_voxel,
            biome_patch=biome_patch,
            heightmap=heightmap,
            structure_mask=structure_mask,
            structure_types=structure_types,
            structure_positions=structure_positions,
        )

        return outputs

    def _extract_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract target tensors from batch."""
        targets = {}

        # Standard targets
        if "air_mask" in batch:
            targets["air_mask"] = batch["air_mask"]
        if "block_types" in batch:
            targets["block_types"] = batch["block_types"]

        # Structure targets
        if "structure_mask" in batch:
            targets["structure_mask"] = batch["structure_mask"]
        if "structure_types" in batch:
            targets["structure_types"] = batch["structure_types"]

        return targets

    def _compute_batch_metrics(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for a single batch."""
        # Core accuracy metrics
        metrics = self.metrics.compute_metrics(predictions, targets)

        # IoU metrics for block types
        if "block_type_logits" in predictions and "block_types" in targets:
            iou_metrics = self.iou_calc.calculate_iou(
                predictions["block_type_logits"], targets["block_types"]
            )
            metrics.update(iou_metrics)

        # Dice metrics for air mask
        if "air_mask_logits" in predictions and "air_mask" in targets:
            dice_metrics = self.dice_calc.calculate_dice(
                predictions["air_mask_logits"], targets["air_mask"]
            )
            metrics.update(dice_metrics)

        return metrics

    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all batches."""
        if not all_metrics:
            return {}

        # Get all metric keys
        all_keys = set()
        for metrics in all_metrics:
            all_keys.update(metrics.keys())

        # Average each metric
        aggregated = {}
        for key in all_keys:
            if "per_class" in key:
                # Handle per-class metrics separately
                continue

            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print metrics in a formatted way."""
        print("\n" + "=" * 60)
        print("🧠 VOXELTREE MODEL EVALUATION RESULTS")
        print("=" * 60)

        # Core metrics
        print("\n📊 CORE METRICS:")
        for key, value in metrics.items():
            if not any(x in key for x in ["structure", "class", "top3", "iou", "dice"]):
                print(f"  {key:<30}: {value:.4f}")

        # Structure-aware metrics
        structure_metrics = {k: v for k, v in metrics.items() if "structure" in k}
        if structure_metrics:
            print("\n🏗️  STRUCTURE-AWARE METRICS:")
            for key, value in structure_metrics.items():
                print(f"  {key:<30}: {value:.4f}")

        # Advanced metrics
        advanced_metrics = {
            k: v for k, v in metrics.items() if any(x in k for x in ["top3", "iou", "dice"])
        }
        if advanced_metrics:
            print("\n🎯 ADVANCED METRICS:")
            for key, value in advanced_metrics.items():
                print(f"  {key:<30}: {value:.4f}")

        print("=" * 60)

    def save_metrics(self, metrics: Dict[str, float], output_path: str) -> None:
        """Save metrics to JSON file."""
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to: {output_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate VoxelTree model")
    parser.add_argument("--config", required=True, help="Evaluation config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint file")
    parser.add_argument("--dataset", required=True, help="Evaluation dataset path")
    parser.add_argument("--output", help="Output metrics file (JSON)")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda/mps)")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.config, args.checkpoint, args.device)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(args.dataset)

    # Print results
    evaluator.print_metrics(metrics)

    # Save results if requested
    if args.output:
        evaluator.save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
