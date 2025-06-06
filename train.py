#!/usr/bin/env python3
"""
VoxelTree Training CLI - Main orchestration script for full-scale training

This script coordinates the complete training pipeline:
1. Dataset generation and validation
2. Model training with metrics and logging
3. Checkpoint management and resumption
4. Export verification

Usage:
    python train.py --config config.yaml --action [generate|train|export]
    python train.py --help
"""

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from train.dataset import VoxelTreeDataset
from train.logger import TrainingLogger
from train.trainer import VoxelTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the training pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["worldgen", "training", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def generate_dataset(config: Dict[str, Any]) -> None:
    """Generate the full training corpus using the extraction pipeline."""
    from scripts.extraction.chunk_extractor import ChunkExtractor
    from scripts.pairing.patch_pairer import PatchPairer
    from scripts.pairing.seed_input_linker import SeedInputLinker
    from scripts.worldgen.bootstrap import FabricWorldGenBootstrap

    logger = logging.getLogger(__name__)
    logger.info("Starting dataset generation pipeline...")

    # Step 1: Generate world chunks
    logger.info("Step 1: Generating world chunks...")
    bootstrap = FabricWorldGenBootstrap(config["worldgen"])
    world_dir = bootstrap.generate_world_regions()

    # Step 2: Extract chunks to NPZ format
    logger.info("Step 2: Extracting chunks to NPZ...")
    extractor = ChunkExtractor(config["extraction"])
    chunk_output_dir = extractor.extract_all_chunks(world_dir)

    # Step 3: Create LOD patch pairs
    logger.info("Step 3: Creating LOD patch pairs...")
    pairer = PatchPairer(config["pairing"])
    pairs_dir = pairer.create_patch_pairs(chunk_output_dir)

    # Step 4: Link with seed inputs
    logger.info("Step 4: Linking with seed inputs...")
    linker = SeedInputLinker(config["pairing"])
    final_dataset_dir = linker.link_pairs_with_seed_inputs(pairs_dir)

    # Step 5: Split into train/val/test
    logger.info("Step 5: Splitting dataset...")
    split_dataset(final_dataset_dir, config["data"])

    logger.info(f"Dataset generation complete! Output: {final_dataset_dir}")


def split_dataset(dataset_dir: Path, data_config: Dict[str, Any]) -> None:
    """Split the dataset into train/validation/test sets."""
    logger = logging.getLogger(__name__)

    # Get all patch files
    patch_files = list(Path(dataset_dir).glob("*.npz"))
    total_files = len(patch_files)

    if total_files == 0:
        raise ValueError(f"No .npz files found in {dataset_dir}")

    # Split ratios (default: 70% train, 20% val, 10% test)
    train_ratio = data_config.get("train_split", 0.7)
    val_ratio = data_config.get("val_split", 0.2)
    test_ratio = data_config.get("test_split", 0.1)

    # Calculate splits
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    logger.info(
        f"Splitting {total_files} files: {train_count} train, {val_count} val, {test_count} test"
    )

    # Create split directories
    splits_dir = Path(data_config["processed_data_dir"])
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    test_dir = splits_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    # Move files to split directories
    for i, patch_file in enumerate(patch_files):
        if i < train_count:
            dest_dir = train_dir
        elif i < train_count + val_count:
            dest_dir = val_dir
        else:
            dest_dir = test_dir

        dest_path = dest_dir / patch_file.name
        shutil.copy2(patch_file, dest_path)

    logger.info("Dataset split completed successfully")


def audit_dataset(config: Dict[str, Any]) -> None:
    """Perform data quality audit on the generated dataset."""
    from scripts.pairing.patch_validator import PatchValidator

    logger = logging.getLogger(__name__)
    logger.info("Starting dataset quality audit...")

    processed_dir = Path(config["data"]["processed_data_dir"])

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        logger.info(f"Auditing {split} split...")
        validator = PatchValidator()

        # Get sample of files for audit
        patch_files = list(split_dir.glob("*.npz"))
        sample_size = min(100, len(patch_files))  # Audit up to 100 files per split

        for patch_file in patch_files[:sample_size]:
            try:
                audit_result = validator.audit_patch_quality(patch_file)
                logger.debug(f"Audit result for {patch_file.name}: {audit_result}")
            except Exception as e:
                logger.error(f"Failed to audit {patch_file}: {e}")

    logger.info("Dataset audit completed")


def train_model(config: Dict[str, Any], resume_checkpoint: Optional[Path] = None) -> None:
    """Train the VoxelTree model with full logging and checkpointing."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    # Setup directories
    runs_dir = Path("runs") / f"run_{int(time.time())}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save config used for this run
    with open(runs_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create datasets
    data_config = config["data"]
    processed_dir = Path(data_config["processed_data_dir"])

    train_dataset = VoxelTreeDataset(processed_dir / "train")
    val_dataset = VoxelTreeDataset(processed_dir / "val")

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
    )

    # Initialize trainer
    trainer = VoxelTrainer(config)

    # Initialize logger
    training_logger = TrainingLogger(
        log_dir=runs_dir / "logs", use_tensorboard=config["training"].get("use_tensorboard", True)
    )

    # Resume from checkpoint if provided
    if resume_checkpoint:
        logger.info(f"Resuming training from {resume_checkpoint}")
        trainer.resume_from_checkpoint(resume_checkpoint)

    # Training loop
    epochs = config["training"]["epochs"]
    save_every = config["training"].get("save_every", 10)

    for epoch in range(trainer.current_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # Training phase
        trainer.model.train()
        train_metrics = trainer.train_one_epoch(train_loader)

        # Validation phase
        trainer.model.eval()
        val_metrics = trainer.validate_one_epoch(val_loader)

        # Log metrics
        epoch_metrics = {
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch": epoch + 1,
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }
        training_logger.log_metrics(epoch_metrics)

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = runs_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch + 1, train_metrics["loss"])
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if train_metrics["loss"] < trainer.best_loss:
            trainer.best_loss = train_metrics["loss"]
            best_model_path = runs_dir / "best_model.pt"
            trainer.save_checkpoint(best_model_path, epoch + 1, train_metrics["loss"])
            logger.info(f"New best model saved: {best_model_path}")

    # Final checkpoint
    final_checkpoint = runs_dir / "final_checkpoint.pt"
    trainer.save_checkpoint(final_checkpoint, epochs, trainer.best_loss)

    # Close logger
    training_logger.close()

    logger.info(f"Training completed! Results saved to {runs_dir}")


def export_model(config: Dict[str, Any], checkpoint_path: Path) -> None:
    """Export trained model to ONNX format and verify parity."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model export...")

    # Load trained model
    trainer = VoxelTrainer(config)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()

    # Create dummy input for export
    batch_size = 1
    dummy_input = {
        "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
        "biome_patch": torch.randint(0, 50, (batch_size, 16, 16)),
        "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
        "river_patch": torch.randn(batch_size, 1, 16, 16),
        "y_index": torch.randint(0, 24, (batch_size,)),
        "lod": torch.randint(1, 5, (batch_size,)),
    }

    # Export to ONNX
    onnx_path = checkpoint_path.parent / "model.onnx"
    torch.onnx.export(
        trainer.model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=[
            "parent_voxel",
            "biome_patch",
            "heightmap_patch",
            "river_patch",
            "y_index",
            "lod",
        ],
        output_names=["air_mask_logits", "block_type_logits"],
        dynamic_axes={
            "parent_voxel": {0: "batch_size"},
            "biome_patch": {0: "batch_size"},
            "heightmap_patch": {0: "batch_size"},
            "river_patch": {0: "batch_size"},
            "y_index": {0: "batch_size"},
            "lod": {0: "batch_size"},
            "air_mask_logits": {0: "batch_size"},
            "block_type_logits": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to ONNX: {onnx_path}")

    # Verify ONNX model parity
    try:
        import onnx
        import onnxruntime as ort

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(onnx_path))

        # Run inference with both models
        with torch.no_grad():
            torch_output = trainer.model(dummy_input)

            # Prepare ONNX input
            onnx_input = {
                name: dummy_input[name].numpy()
                for name in [
                    "parent_voxel",
                    "biome_patch",
                    "heightmap_patch",
                    "river_patch",
                    "y_index",
                    "lod",
                ]
            }

            onnx_output = ort_session.run(None, onnx_input)

        # Check parity
        torch_air_mask = torch_output[0].numpy()
        torch_block_types = torch_output[1].numpy()
        onnx_air_mask = onnx_output[0]
        onnx_block_types = onnx_output[1]

        air_mask_diff = abs(torch_air_mask - onnx_air_mask).max()
        block_types_diff = abs(torch_block_types - onnx_block_types).max()

        logger.info(f"ONNX parity check - Air mask max diff: {air_mask_diff:.6f}")
        logger.info(f"ONNX parity check - Block types max diff: {block_types_diff:.6f}")

        if air_mask_diff < 1e-5 and block_types_diff < 1e-5:
            logger.info("✅ ONNX export parity check PASSED")
        else:
            logger.warning("⚠️ ONNX export parity check FAILED - differences too large")

    except ImportError:
        logger.warning("ONNX verification skipped - onnx and onnxruntime not installed")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="VoxelTree Training Pipeline")
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--action",
        choices=["generate", "audit", "train", "export"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint for export")
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
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Execute requested action
    try:
        if args.action == "generate":
            generate_dataset(config)
        elif args.action == "audit":
            audit_dataset(config)
        elif args.action == "train":
            train_model(config, args.resume)
        elif args.action == "export":
            if not args.checkpoint:
                logger.error("--checkpoint required for export action")
                return 1
            export_model(config, args.checkpoint)
    except Exception as e:
        logger.error(f"Action '{args.action}' failed: {e}")
        return 1

    logger.info(f"Action '{args.action}' completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
