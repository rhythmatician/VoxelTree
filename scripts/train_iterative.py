#!/usr/bin/env python3
"""
VoxelTree Iterative Training Pipeline

This script implements the continuous training cycle:
1. Generate a batch of world chunks headlessly
2. Train the model on those chunks
3. Delete the chunks to save disk space
4. Repeat until convergence or max iterations

This approach allows training on virtually unlimited data while maintaining
a small disk footprint, perfect for LOD-aware voxel super-resolution.

Usage:
    python scripts/train_iterative.py --config config.yaml --max-iterations 100 --chunks-per-iteration 1000
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.extraction.chunk_extractor import ChunkExtractor
from scripts.pairing.patch_pairer import PatchPairer
from scripts.pairing.seed_input_linker import SeedInputLinker
from scripts.worldgen.bootstrap import FabricWorldGenBootstrap
from train.dataset import VoxelTreeDataset
from train.logger import TrainingLogger
from train.trainer import VoxelTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the iterative training pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("iterative_training.log"),
        ],
    )


def load_config(config_path: Path) -> Dict:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["worldgen", "extraction", "pairing", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def get_current_seed_and_region(
    base_seed: int, iteration: int, regions_per_seed: int = 16, config: Dict = None
) -> Tuple[int, int, int]:
    """
    Get the current seed and region coordinates for this iteration.

    Supports both regular training and expanded stronghold training modes.

    Args:
        base_seed: Starting seed value
        iteration: Current iteration number
        regions_per_seed: How many regions to process per seed before moving to next seed
        config: Configuration dict for stronghold training settings

    Returns:
        Tuple of (seed, region_x, region_z)
    """
    # Check if stronghold training is enabled
    stronghold_config = config.get("stronghold_training", {}) if config else {}
    stronghold_enabled = stronghold_config.get("enabled", False)

    if stronghold_enabled:
        # Use expanded grid for stronghold training
        regions_per_seed = stronghold_config.get("regions_per_seed", 1369)
        grid_size = int(regions_per_seed**0.5)  # Should be 37 for stronghold training
    else:
        # Regular training mode
        grid_size = int(regions_per_seed**0.5)

    seed_index = iteration // regions_per_seed
    region_index = iteration % regions_per_seed

    seed = base_seed + seed_index

    # Generate region coordinates in grid around spawn
    x = region_index % grid_size - grid_size // 2
    z = region_index // grid_size - grid_size // 2

    return seed, x, z


def generate_single_region_batch(
    config: Dict,
    seed: int,
    region_x: int,
    region_z: int,
    temp_dir: Path,
) -> Path:
    """Generate chunks from exactly one region - optimal batch size of ~1024 chunks."""
    logger = logging.getLogger(__name__)

    # Create temporary directories
    batch_dir = temp_dir / f"batch_s{seed}_r{region_x}_{region_z}_{int(time.time())}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_chunk_files = []

    try:
        # Generate world for this seed and region
        logger.debug(
            f"Generating seed {seed}, region ({region_x}, {region_z}) - expect ~1024 chunks"
        )

        # Create bootstrap and generate world
        bootstrap = FabricWorldGenBootstrap(
            seed=str(seed), java_heap=config["worldgen"].get("java_heap", "4G")
        )

        # Generate this specific region only (32x32 chunks = 1024 chunks)
        x_range = (region_x, region_x)  # Single region
        z_range = (region_z, region_z)  # Single region
        world_dir = bootstrap.generate_region_batch(x_range, z_range)

        # Extract chunks from this region
        extractor = ChunkExtractor(config["extraction"])
        chunk_output_dir = extractor.extract_all_chunks(world_dir)

        # Copy chunk files to batch directory
        for chunk_file in chunk_output_dir.glob("*.npz"):
            dest_file = batch_dir / f"r{region_x}_{region_z}_{chunk_file.name}"
            shutil.copy2(chunk_file, dest_file)
            all_chunk_files.append(dest_file)

        # Clean up temporary world files immediately
        if world_dir.exists():
            shutil.rmtree(world_dir)
        if chunk_output_dir.exists():
            shutil.rmtree(chunk_output_dir)

    except Exception as e:
        logger.error(f"Failed to generate region ({region_x}, {region_z}) for seed {seed}: {e}")
        return batch_dir

    logger.info(
        f"Generated {len(all_chunk_files)} chunks from seed {seed}, region ({region_x}, {region_z})"
    )
    return batch_dir


def create_training_pairs(
    config: Dict,
    chunk_dir: Path,
    temp_dir: Path,
) -> Path:
    """Create LOD training pairs from chunks."""
    logger = logging.getLogger(__name__)

    # Create patch pairs
    pairer = PatchPairer(config["pairing"])
    pairs_dir = pairer.create_patch_pairs(chunk_dir)

    # Link with seed inputs
    linker = SeedInputLinker(config["pairing"])
    linked_dir = linker.link_pairs_with_seed_inputs(pairs_dir)

    logger.info(f"Created {len(list(linked_dir.glob('*.npz')))} training pairs")
    return linked_dir


def train_on_batch(
    trainer: VoxelTrainer,
    training_logger: TrainingLogger,
    pairs_dir: Path,
    config: Dict,
    iteration: int,
) -> Tuple[float, Dict]:
    """Train the model on a batch of data."""
    logger = logging.getLogger(__name__)

    # Create dataset and dataloader for this batch
    dataset = VoxelTreeDataset(pairs_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
    )

    if len(dataset) == 0:
        logger.warning("No training data found in batch")
        return float("inf"), {}

    logger.info(f"Training on {len(dataset)} examples")

    # Train for one epoch on this batch
    trainer.model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        try:
            # Forward pass and compute loss
            loss, metrics = trainer.train_step(batch)
            total_loss += loss
            num_batches += 1

            # Log batch metrics if needed
            if batch_idx % config["training"].get("log_every", 10) == 0:
                step = iteration * len(dataloader) + batch_idx
                training_logger.log_metrics(
                    {
                        "batch_loss": loss,
                        "iteration": iteration,
                        "step": step,
                        **metrics,
                    }
                )

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            continue

    # Compute average loss for this batch
    avg_loss = total_loss / max(num_batches, 1)

    # Aggregate metrics
    batch_metrics = {
        "avg_loss": avg_loss,
        "num_examples": len(dataset),
        "num_batches": num_batches,
    }

    return avg_loss, batch_metrics


def cleanup_batch(batch_dir: Path) -> None:
    """Clean up temporary batch files."""
    logger = logging.getLogger(__name__)

    if batch_dir.exists():
        shutil.rmtree(batch_dir)
        logger.debug(f"Cleaned up batch directory: {batch_dir}")


def save_iteration_checkpoint(
    trainer: VoxelTrainer,
    iteration: int,
    loss: float,
    metrics: Dict,
    checkpoint_dir: Path,
) -> Path:
    """Save checkpoint after each iteration."""
    checkpoint_path = checkpoint_dir / f"iteration_{iteration:04d}.pt"
    trainer.save_checkpoint(checkpoint_path, iteration, loss)

    # Also save metrics
    metrics_path = checkpoint_dir / f"iteration_{iteration:04d}_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)

    return checkpoint_path


def should_stop_training(
    loss_history: List[float],
    min_iterations: int = 10,
    patience: int = 5,
    min_improvement: float = 0.001,
) -> bool:
    """Determine if training should stop based on loss history."""
    if len(loss_history) < min_iterations:
        return False

    # Check if loss has plateaued
    if len(loss_history) >= patience:
        recent_losses = loss_history[-patience:]
        if max(recent_losses) - min(recent_losses) < min_improvement:
            return True

    return False


def run_iterative_training(
    config: Dict,
    max_iterations: int = 100,
    base_seed: int = 10000,
    regions_per_seed: int = 16,
    resume_checkpoint: Optional[Path] = None,
) -> None:
    """Run the main iterative training loop with one region per batch."""
    logger = logging.getLogger(__name__)

    # Setup directories
    runs_dir = Path("runs") / f"iterative_run_{int(time.time())}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = runs_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    checkpoint_dir = runs_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config for this run
    with open(runs_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize trainer
    trainer = VoxelTrainer(config)

    # Initialize logger
    training_logger = TrainingLogger(
        log_dir=runs_dir / "logs", use_tensorboard=config["training"].get("use_tensorboard", True)
    )

    # Resume from checkpoint if provided
    start_iteration = 0
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
        # Extract iteration from checkpoint filename
        checkpoint_name = resume_checkpoint.stem
        if "iteration_" in checkpoint_name:
            start_iteration = int(checkpoint_name.split("_")[1]) + 1

    # Track training progress
    loss_history = []
    best_loss = float("inf")

    logger.info(f"Starting iterative training for {max_iterations} iterations")
    logger.info("Using one region per batch (~1024 chunks each)")

    for iteration in range(start_iteration, max_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")

        try:
            # Step 1: Get current seed and region with stronghold awareness
            seed, region_x, region_z = get_current_seed_and_region(
                base_seed, iteration, regions_per_seed, config
            )

            # Calculate distance from spawn for stronghold validation
            region_distance = int(
                ((region_x**2 + region_z**2) ** 0.5) * 512
            )  # 512 blocks per region

            logger.info(f"Processing seed {seed}, region ({region_x}, {region_z})")
            logger.info(f"Distance from spawn: {region_distance:,} blocks")

            # Check if this region might contain strongholds
            stronghold_config = config.get("stronghold_training", {})
            if stronghold_config.get("enabled", False):
                max_distance = stronghold_config.get("max_distance_blocks", 8960)
                if region_distance <= max_distance:
                    logger.info("Region within stronghold training coverage")
                else:
                    logger.info("Region beyond stronghold coverage - terrain only")

            # Step 2: Generate chunk batch from single region
            chunk_batch_dir = generate_single_region_batch(
                config, seed, region_x, region_z, temp_dir
            )

            # Step 3: Create training pairs
            pairs_dir = create_training_pairs(config, chunk_batch_dir, temp_dir)

            # Step 4: Train on this batch
            batch_loss, batch_metrics = train_on_batch(
                trainer, training_logger, pairs_dir, config, iteration
            )

            # Step 5: Track progress
            loss_history.append(batch_loss)
            logger.info(f"Iteration {iteration + 1} completed - Loss: {batch_loss:.4f}")

            # Step 6: Save checkpoint
            if batch_loss < best_loss:
                best_loss = batch_loss
                best_checkpoint = save_iteration_checkpoint(
                    trainer, iteration, batch_loss, batch_metrics, checkpoint_dir
                )
                logger.info(f"New best model saved: {best_checkpoint}")

            # Regular checkpoint every N iterations
            if (iteration + 1) % config["training"].get("save_every", 10) == 0:
                checkpoint_path = save_iteration_checkpoint(
                    trainer, iteration, batch_loss, batch_metrics, checkpoint_dir
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Step 6: Cleanup
            cleanup_batch(chunk_batch_dir)
            cleanup_batch(pairs_dir)

            # Log iteration metrics
            iteration_metrics = {
                "iteration": iteration + 1,
                "loss": batch_loss,
                "best_loss": best_loss,
                "current_seed": seed,
                "region_x": region_x,
                "region_z": region_z,
                **batch_metrics,
            }
            training_logger.log_metrics(iteration_metrics)

            # Step 7: Check stopping criteria
            if should_stop_training(loss_history):
                logger.info("Training converged - stopping early")
                break

        except Exception as e:
            logger.error(f"Iteration {iteration + 1} failed: {e}")
            # Clean up any partial files
            try:
                cleanup_batch(chunk_batch_dir)
                cleanup_batch(pairs_dir)
            except Exception:
                pass
            continue

    # Final cleanup and summary
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    training_logger.close()

    logger.info("Iterative training completed!")
    logger.info(f"Total iterations: {len(loss_history)}")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Results saved to: {runs_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="VoxelTree Iterative Training Pipeline")
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Maximum number of training iterations"
    )
    parser.add_argument(
        "--base-seed", type=int, default=10000, help="Base seed for world generation"
    )
    parser.add_argument(
        "--regions-per-seed",
        type=int,
        default=16,
        help="Number of regions per seed (16=4x4 grid, 1369=37x37 for strongholds)",
    )
    parser.add_argument(
        "--stronghold-training",
        action="store_true",
        help="Enable stronghold training mode (expands coverage to rings 1-3)",
    )
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume training from")
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

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Override stronghold training if specified via CLI
        if args.stronghold_training:
            if "stronghold_training" not in config:
                config["stronghold_training"] = {}
            config["stronghold_training"]["enabled"] = True
            logger.info("Stronghold training enabled via CLI argument")

            # Adjust regions per seed if not explicitly set
            if args.regions_per_seed == 16:  # Default value
                args.regions_per_seed = config["stronghold_training"].get("regions_per_seed", 1369)
                logger.info(f"Using stronghold training regions per seed: {args.regions_per_seed}")

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Run iterative training
    try:
        run_iterative_training(
            config,
            max_iterations=args.max_iterations,
            base_seed=args.base_seed,
            regions_per_seed=args.regions_per_seed,
            resume_checkpoint=args.resume,
        )
        logger.info("Iterative training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Iterative training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
