#!/usr/bin/env python3
"""
VoxelTree Hyperparameter Optimization

This script performs hyperparameter tuning for the VoxelTree model using
Optuna optimization framework. It helps find optimal hyperparameters for
architecture, learning rate, regularization, etc.

Usage:
    python scripts/hyperparameter_tune.py --config config.yaml
    python scripts/hyperparameter_tune.py --config config.yaml --trials 50
"""

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler

# Try to import optuna with fallback
try:
    import optuna
    from optuna.trial import Trial
    from optuna.visualization import plot_contour, plot_optimization_history, plot_param_importances

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Append project root to path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.dataset import VoxelTreeDataset
from train.trainer import VoxelTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the hyperparameter tuning script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("hyperparam_tune.log")],
    )


def load_config(config_path: Path) -> Dict:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def sample_hyperparameters(trial: Trial) -> Dict[str, Any]:
    """Sample hyperparameters using Optuna trial."""
    # Learning rate (log scale)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

    # Weight decay (L2 regularization)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    # Model architecture
    base_channels = trial.suggest_int("base_channels", 16, 64, step=8)
    depth = trial.suggest_int("depth", 2, 4)

    # Dropout rate
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Loss weights
    mask_weight = trial.suggest_float("mask_weight", 0.5, 5.0)
    type_weight = trial.suggest_float("type_weight", 0.5, 5.0)

    # Embedding dimensions
    biome_embed_dim = trial.suggest_int("biome_embed_dim", 8, 32, step=8)
    lod_embed_dim = trial.suggest_int("lod_embed_dim", 16, 64, step=16)

    # Activation function
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])

    # Batch normalization
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    # Combine into hyperparameter dictionary
    hyperparams = {
        "training": {
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
        },
        "model": {
            "base_channels": base_channels,
            "depth": depth,
            "dropout_rate": dropout_rate,
            "use_batch_norm": use_batch_norm,
            "activation": activation,
            "biome_embed_dim": biome_embed_dim,
            "lod_embed_dim": lod_embed_dim,
        },
        "loss": {
            "mask_weight": mask_weight,
            "type_weight": type_weight,
        },
    }

    return hyperparams


def update_config_with_hyperparams(config: Dict, hyperparams: Dict) -> Dict:
    """Create a new config with updated hyperparameters."""
    new_config = deepcopy(config)

    # Update training config
    if "training" not in new_config:
        new_config["training"] = {}
    new_config["training"].update(hyperparams.get("training", {}))

    # Update model config
    if "model" not in new_config:
        new_config["model"] = {}
    new_config["model"].update(hyperparams.get("model", {}))

    # Update loss config
    if "loss" not in new_config:
        new_config["loss"] = {}
    new_config["loss"].update(hyperparams.get("loss", {}))

    return new_config


def create_val_dataloader(
    dataset_path: Path,
    batch_size: int,
    val_samples: int = 200,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, VoxelTreeDataset]:
    """Create validation dataloader from dataset."""
    logger = logging.getLogger(__name__)

    # Create dataset
    dataset = VoxelTreeDataset(dataset_path)

    # Create validation split
    dataset_size = len(dataset)
    val_samples = min(val_samples, dataset_size)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create indices and sampler
    indices = np.random.choice(dataset_size, val_samples, replace=False)
    sampler = SubsetRandomSampler(indices)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    logger.info(
        f"Created validation dataloader with {val_samples} samples, batch size {batch_size}"
    )
    return dataloader, dataset


def objective(trial: Trial, base_config: Dict, dataset_path: Path) -> float:
    """Objective function for Optuna optimization."""
    # TODO: Add hook for external objective function customization
    # This will allow for custom objective functions to be injected
    # from outside scripts without modifying this core hyperparameter tuning logic

    logger = logging.getLogger(__name__)

    # Sample hyperparameters
    hyperparams = sample_hyperparameters(trial)

    # Update config with hyperparameters
    config = update_config_with_hyperparams(base_config, hyperparams)

    # Log trial config
    logger.info(f"Trial {trial.number}: {json.dumps(hyperparams, indent=2)}")

    # Create validation dataloader
    batch_size = config.get("training", {}).get("batch_size", 32)
    val_loader, dataset = create_val_dataloader(
        dataset_path,
        batch_size=batch_size,
        val_samples=config.get("tuning", {}).get("validation_samples", 200),
    )

    try:
        # Create trainer with this config
        trainer = VoxelTrainer(config)

        # Train for a small number of epochs
        num_epochs = config.get("tuning", {}).get("epochs_per_trial", 3)
        logger.info(f"Training for {num_epochs} epochs")

        val_losses = []

        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = trainer.train_one_epoch(val_loader)

            # Validate
            val_metrics = trainer.validate_one_epoch(val_loader)
            val_loss = val_metrics["loss"]
            val_losses.append(val_loss)

            # Report intermediate value
            trial.report(val_loss, epoch)

            # Handle pruning (early stopping for unpromising trials)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

        # Return the average of the last validation loss
        return val_losses[-1]

    except (RuntimeError, optuna.exceptions.TrialPruned) as e:
        # Handle errors (usually CUDA out of memory)
        if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
            logger.warning(f"Trial {trial.number} failed with CUDA OOM: {e}")

            # Return a high loss to discourage these hyperparameters
            return 1e10

        # Re-raise pruning exceptions
        if isinstance(e, optuna.exceptions.TrialPruned):
            raise e

        # Return a high loss for other errors
        logger.error(f"Trial {trial.number} failed with error: {e}")
        return 1e10


def run_hyperparameter_search(
    config: Dict,
    dataset_path: Path,
    n_trials: int = 50,
    study_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run hyperparameter search using Optuna."""
    logger = logging.getLogger(__name__)

    if not OPTUNA_AVAILABLE:
        logger.error(
            "Optuna not available. Please install with 'pip install optuna'. "
            "For visualization support, also install 'pip install plotly'"
        )
        raise ImportError("Optuna is required for hyperparameter tuning")

    # Create output directory
    if output_dir is None:
        output_dir = Path("hyperparam_results") / f"study_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create database path
    db_path = output_dir / "optuna.db"
    study_name = study_name or f"voxeltree_study_{int(time.time())}"

    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    # Run optimization
    logger.info(f"Starting hyperparameter search with {n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, config, dataset_path),
        n_trials=n_trials,
        catch=(RuntimeError,),
        gc_after_trial=True,
    )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best loss: {best_value:.6f}")
    logger.info(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")

    # Generate plots if visualization is available
    try:
        fig_history = plot_optimization_history(study)
        fig_contour = plot_contour(study)
        fig_importance = plot_param_importances(study)

        fig_history.write_html(str(output_dir / "optimization_history.html"))
        fig_contour.write_html(str(output_dir / "contour_plot.html"))
        fig_importance.write_html(str(output_dir / "param_importances.html"))

        logger.info(f"Visualization plots saved to {output_dir}")
    except:
        logger.warning("Failed to generate visualization plots")

    # Generate updated config with best hyperparameters
    best_hyperparams = {}
    for param_name, param_value in best_params.items():
        # Map flat parameter names back to nested structure
        if (
            param_name.startswith("learning_rate")
            or param_name.startswith("optimizer")
            or param_name.startswith("weight_decay")
        ):
            if "training" not in best_hyperparams:
                best_hyperparams["training"] = {}
            best_hyperparams["training"][param_name] = param_value
        elif param_name.startswith("mask_weight") or param_name.startswith("type_weight"):
            if "loss" not in best_hyperparams:
                best_hyperparams["loss"] = {}
            best_hyperparams["loss"][param_name] = param_value
        else:
            # Model architecture parameters
            if "model" not in best_hyperparams:
                best_hyperparams["model"] = {}
            best_hyperparams["model"][param_name] = param_value

    # Create complete best config
    best_config = update_config_with_hyperparams(config, best_hyperparams)

    # Save best configuration
    with open(output_dir / "best_config.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    # Save study statistics
    study_stats = {
        "best_trial": best_trial.number,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_completed_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "study_name": study_name,
        "study_duration": study.trials[-1].datetime_complete - study.trials[0].datetime_start,
    }

    with open(output_dir / "study_stats.json", "w") as f:
        json.dump(study_stats, f, indent=2)

    # Save summary.txt with tuning results
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"VoxelTree Hyperparameter Tuning Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Completed trials: {study_stats['n_completed_trials']}\n")
        f.write(f"Pruned trials: {study_stats['n_pruned_trials']}\n\n")
        f.write(f"Best trial: {best_trial.number}\n")
        f.write(f"Best validation loss: {best_value:.6f}\n\n")
        f.write(f"Best hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")

    logger.info(f"Hyperparameter search completed. Results saved to {output_dir}")
    return {
        "best_config": best_config,
        "stats": study_stats,
        "output_dir": output_dir,
    }


def finalize_hyperparameters(
    config: Dict,
    best_config: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Finalize hyperparameters and create production training config."""
    logger = logging.getLogger(__name__)

    # Merge configs, giving priority to best_config
    final_config = deepcopy(config)

    # Update model config
    if "model" in best_config:
        if "model" not in final_config:
            final_config["model"] = {}
        final_config["model"].update(best_config["model"])

    # Update training config
    if "training" in best_config:
        if "training" not in final_config:
            final_config["training"] = {}
        final_config["training"].update(best_config["training"])

    # Update loss config
    if "loss" in best_config:
        if "loss" not in final_config:
            final_config["loss"] = {}
        final_config["loss"].update(best_config["loss"])

    # Save finalized config
    if output_path is None:
        output_path = Path("config_production.yaml")

    with open(output_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)

    logger.info(f"Finalized hyperparameters saved to {output_path}")
    return output_path


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="VoxelTree Hyperparameter Tuning")
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to dataset for tuning (defaults to train split from config)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--output-dir", type=Path, help="Directory to save tuning results")
    parser.add_argument(
        "--finalize", action="store_true", help="Generate final production config after tuning"
    )
    parser.add_argument("--study-name", type=str, help="Name for the Optuna study")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Check for Optuna
    if not OPTUNA_AVAILABLE:
        logger.error(
            "Optuna not available. Please install with 'pip install optuna'. "
            "For visualization support, also install 'pip install plotly'"
        )
        return 1

    try:
        # Load config
        config = load_config(args.config)

        # Determine dataset path
        dataset_path = args.dataset_path
        if dataset_path is None:
            processed_dir = Path(config["data"]["processed_data_dir"])
            dataset_path = processed_dir / "train"

        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return 1

        # Run hyperparameter search
        results = run_hyperparameter_search(
            config,
            dataset_path,
            n_trials=args.trials,
            study_name=args.study_name,
            output_dir=args.output_dir,
        )

        # Finalize hyperparameters if requested
        if args.finalize:
            final_config_path = finalize_hyperparameters(
                config,
                results["best_config"],
                output_path=(
                    results["output_dir"] / "final_config.yaml" if args.output_dir else None
                ),
            )
            logger.info(f"Production config saved to {final_config_path}")

        logger.info(f"Hyperparameter tuning completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())
