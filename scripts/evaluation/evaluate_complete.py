#!/usr/bin/env python3
"""
VoxelTree Complete Evaluation CLI - Phase 6 Integration

Command-line interface for comprehensive VoxelTree model evaluation.
Combines metrics computation and 3D visualization for complete assessment.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

from scripts.evaluation.evaluate_model import ModelEvaluator
from scripts.evaluation.visualization import VoxelVisualizationSuite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VoxelTree Model Evaluation with Metrics and Visualization"
    )

    # Required arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint (.pth file)"
    )

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to evaluation dataset directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results and visualizations",
    )

    # Optional arguments
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.evaluation.yaml",
        help="Path to evaluation configuration file",
    )

    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize (default: 5)"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Evaluation batch size (default: 8)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation (default: auto)",
    )

    parser.add_argument(
        "--include_structure",
        action="store_true",
        help="Include structure-aware evaluation (if model supports it)",
    )

    parser.add_argument(
        "--generate_report", action="store_true", help="Generate HTML evaluation report"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run evaluation
    run_complete_evaluation(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        include_structure=args.include_structure,
        generate_report=args.generate_report,
    )


def run_complete_evaluation(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    config_path: str = "config.evaluation.yaml",
    num_samples: int = 5,
    batch_size: int = 8,
    device: str = "auto",
    include_structure: bool = False,
    generate_report: bool = True,
) -> Dict[str, float]:
    """
    Run complete VoxelTree model evaluation with metrics and visualization.

    Args:
        model_path: Path to trained model checkpoint
        dataset_path: Path to evaluation dataset
        output_dir: Directory for outputs
        config_path: Evaluation configuration file
        num_samples: Number of samples to visualize
        batch_size: Evaluation batch size
        device: Device for evaluation
        include_structure: Whether to include structure-aware evaluation
        generate_report: Whether to generate HTML report

    Returns:
        Dictionary of computed evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting VoxelTree Complete Evaluation")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"📁 Dataset: {dataset_path}")
    logger.info(f"📁 Output: {output_dir}")

    # Load configuration
    config = load_evaluation_config(config_path)
    if batch_size != 8:  # Override config if specified
        config["eval_batch_size"] = batch_size

    # Set up device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔧 Using device: {device}")

    # Initialize evaluator
    logger.info("📊 Computing evaluation metrics...")
    evaluator = ModelEvaluator(config_path=config_path, device=device)

    # Run metrics evaluation
    metrics = evaluator.evaluate_dataset(model_path=model_path, dataset_path=dataset_path)

    # Save metrics to file
    metrics_path = output_path / "evaluation_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f, default_flow_style=False)
    logger.info(f"💾 Saved metrics to {metrics_path}")

    # Print key metrics
    print_key_metrics(metrics)

    # Generate visualizations
    logger.info("🎨 Generating visualizations...")
    viz_suite = VoxelVisualizationSuite(output_dir=output_path)

    # Load sample data for visualization
    sample_paths = visualize_evaluation_samples(
        evaluator=evaluator,
        viz_suite=viz_suite,
        model_path=model_path,
        dataset_path=dataset_path,
        num_samples=num_samples,
        include_structure=include_structure,
    )

    # Generate HTML report if requested
    if generate_report:
        logger.info("📝 Generating evaluation report...")
        report_path = viz_suite.create_evaluation_report(
            metrics=metrics, sample_paths=sample_paths, report_name="voxeltree_evaluation_report"
        )
        logger.info(f"📄 Generated report: {report_path}")

        # Print report URL for easy access
        print(f"\n🌐 Open evaluation report: file://{report_path.absolute()}")

    logger.info("✅ Evaluation complete!")
    return metrics


def load_evaluation_config(config_path: str) -> Dict:
    """Load evaluation configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {
            "eval_batch_size": 8,
            "metrics": {
                "include_iou": True,
                "include_dice": True,
                "include_structure": True,
            },
        }

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_key_metrics(metrics: Dict[str, float]):
    """Print key evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("📊 KEY EVALUATION METRICS")
    print("=" * 60)

    # Air mask metrics
    mask_accuracy = metrics.get("mask_accuracy", 0.0)
    mask_iou = metrics.get("mask_iou", 0.0)
    print(f"🎯 Air Mask Accuracy:      {mask_accuracy:.4f}")
    print(f"🎯 Air Mask IoU:           {mask_iou:.4f}")

    # Block type metrics
    block_accuracy = metrics.get("block_type_accuracy", 0.0)
    block_top3 = metrics.get("block_type_top3_accuracy", 0.0)
    print(f"🧱 Block Type Accuracy:    {block_accuracy:.4f}")
    print(f"🧱 Block Type Top-3:       {block_top3:.4f}")

    # Structure metrics (if available)
    structure_metrics = {k: v for k, v in metrics.items() if "structure" in k}
    if structure_metrics:
        print("\n🏗️ STRUCTURE-AWARE METRICS")
        print("-" * 30)
        for metric, value in structure_metrics.items():
            display_name = metric.replace("_", " ").title()
            print(f"🏗️ {display_name:<20}: {value:.4f}")

    # IoU/Dice metrics
    iou_dice_metrics = {k: v for k, v in metrics.items() if "iou" in k or "dice" in k}
    if iou_dice_metrics:
        print("\n📐 IoU / DICE METRICS")
        print("-" * 20)
        for metric, value in iou_dice_metrics.items():
            display_name = metric.replace("_", " ").title()
            print(f"📐 {display_name:<20}: {value:.4f}")

    print("=" * 60)


def visualize_evaluation_samples(
    evaluator: ModelEvaluator,
    viz_suite: VoxelVisualizationSuite,
    model_path: str,
    dataset_path: str,
    num_samples: int = 5,
    include_structure: bool = False,
) -> List[Path]:
    """
    Generate visualizations for evaluation samples.

    Args:
        evaluator: Model evaluator instance
        viz_suite: Visualization suite instance
        model_path: Path to model checkpoint
        dataset_path: Path to dataset
        num_samples: Number of samples to visualize
        include_structure: Whether to include structure visualizations

    Returns:
        List of paths to generated visualization files
    """
    from torch.utils.data import DataLoader

    from train.dataset import VoxelTreeDataset

    # Load dataset
    dataset = VoxelTreeDataset(data_dir=dataset_path, return_tensors=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = evaluator._load_model(model_path)
    model.eval()

    all_sample_paths = []

    # Process samples
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        logger.info(f"Visualizing sample {i+1}/{num_samples}")

        # Generate training sample visualization
        sample_data = {k: v[0] for k, v in batch.items()}  # Remove batch dimension
        training_paths = viz_suite.visualize_training_sample(sample_data, sample_id=f"sample_{i+1}")
        all_sample_paths.extend(training_paths)

        # Generate model prediction visualization
        with torch.no_grad():
            # Prepare model inputs
            model_inputs = evaluator._prepare_model_inputs(batch)
            predictions = model(**model_inputs)

            # Create target dictionary
            targets = {
                "air_mask": batch.get("target_mask"),
                "block_types": batch.get("target_types"),
            }

            # Add structure targets if available
            if include_structure and "structure_mask" in batch:
                targets["structure_mask"] = batch["structure_mask"]
            if include_structure and "structure_types" in batch:
                targets["structure_types"] = batch["structure_types"]

            # Generate prediction visualizations
            pred_paths = viz_suite.visualize_model_predictions(
                predictions, targets, batch_idx=0, sample_id=f"prediction_{i+1}"
            )
            all_sample_paths.extend(pred_paths)

    logger.info(f"Generated {len(all_sample_paths)} visualization files")
    return all_sample_paths


if __name__ == "__main__":
    main()
