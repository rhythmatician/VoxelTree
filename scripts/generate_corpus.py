#!/usr/bin/env python3
"""
VoxelTree Training Corpus Generator

This script generates a comprehensive training corpus by:
1. Running the world generation pipeline with varied seeds
2. Executing the extraction pipeline for chunks
3. Creating LOD patch pairs
4. Linking with seed inputs
5. Organizing into train/val/test splits

Usage:
    python scripts/generate_corpus.py --config config.yaml --seed-range 1000-1100 --world-radius 8
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import yaml
from tqdm import tqdm

# Append project root to path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.extraction.chunk_extractor import ChunkExtractor
from scripts.pairing.patch_auditor import PatchQualityAuditor
from scripts.pairing.patch_pairer import PatchPairer
from scripts.pairing.patch_validator import PatchValidator
from scripts.pairing.seed_input_linker import SeedInputLinker
from scripts.worldgen.bootstrap import FabricWorldGenBootstrap


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the corpus generation pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("corpus_generation.log")],
    )


def load_config(config_path: Path) -> Dict:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["worldgen", "extraction", "pairing", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def check_disk_space(min_required_gb: float = 10.0) -> Tuple[float, bool]:
    """Check available disk space and ensure enough is available."""
    disk_usage = psutil.disk_usage(os.getcwd())
    available_gb = disk_usage.free / (1024**3)  # Convert bytes to GB

    logger = logging.getLogger(__name__)
    logger.info(f"Available disk space: {available_gb:.2f} GB")

    if available_gb < min_required_gb:
        logger.warning(
            f"Low disk space warning: {available_gb:.2f} GB available, {min_required_gb} GB recommended"
        )
        return available_gb, False

    return available_gb, True


def monitor_disk_space(min_required_gb: float = 5.0) -> bool:
    """Monitor disk space and return False if below threshold."""
    available_gb, sufficient = check_disk_space(min_required_gb)
    return sufficient


def generate_world_batch(
    config: Dict,
    seed: Union[int, str],
    output_dir: Path,
    world_radius: int = 4,
) -> Path:
    """Generate a world batch with the specified seed."""
    logger = logging.getLogger(__name__)

    # Create a copy of the config with the updated seed
    worldgen_config = config["worldgen"].copy()
    worldgen_config["seed"] = seed

    # Adjust region bounds based on world radius
    worldgen_config["chunk_region_bounds"] = {
        "x_min": -world_radius,
        "x_max": world_radius,
        "z_min": -world_radius,
        "z_max": world_radius,
    }

    # Create bootstrap and generate world
    bootstrap = FabricWorldGenBootstrap(worldgen_config)
    world_dir = bootstrap.generate_world_regions()

    # Create a uniquely named output directory for this seed
    seed_output_dir = output_dir / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generated world with seed {seed}, saved to {world_dir}")
    return world_dir


def extract_chunks(config: Dict, world_dir: Path, output_dir: Path) -> Path:
    """Extract chunks from the world directory."""
    logger = logging.getLogger(__name__)

    # Create extractor and run extraction
    extractor = ChunkExtractor(config["extraction"])
    chunk_output_dir = extractor.extract_all_chunks(world_dir)

    logger.info(f"Extracted chunks to {chunk_output_dir}")
    return chunk_output_dir


def create_patch_pairs(config: Dict, chunk_dir: Path, output_dir: Path) -> Path:
    """Create LOD patch pairs from extracted chunks."""
    logger = logging.getLogger(__name__)

    # Create pairer and generate pairs
    pairer = PatchPairer(config["pairing"])
    pairs_dir = pairer.create_patch_pairs(chunk_dir)

    logger.info(f"Created patch pairs in {pairs_dir}")
    return pairs_dir


def link_with_seed_inputs(config: Dict, pairs_dir: Path, output_dir: Path) -> Path:
    """Link patch pairs with seed inputs."""
    logger = logging.getLogger(__name__)

    # Create linker and link pairs with seed inputs
    linker = SeedInputLinker(config["pairing"])
    linked_dir = linker.link_pairs_with_seed_inputs(pairs_dir)

    logger.info(f"Linked pairs with seed inputs in {linked_dir}")
    return linked_dir


def split_dataset(
    dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Tuple[Path, Path, Path]:
    """Split dataset into train/validation/test sets."""
    logger = logging.getLogger(__name__)

    # Get all patch files
    patch_files = list(Path(dataset_dir).glob("*.npz"))
    total_files = len(patch_files)

    if total_files == 0:
        raise ValueError(f"No .npz files found in {dataset_dir}")

    # Shuffle files for randomization
    random.shuffle(patch_files)

    # Calculate splits
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    logger.info(
        f"Splitting {total_files} files: {train_count} train, {val_count} val, {test_count} test"
    )

    # Create split directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to split directories
    for i, patch_file in enumerate(patch_files):
        if i < train_count:
            dest_dir = train_dir
        elif i < train_count + val_count:
            dest_dir = val_dir
        else:
            dest_dir = test_dir

        dest_path = dest_dir / patch_file.name
        shutil.copy2(patch_file, dest_path)

    logger.info(f"Dataset split completed: {train_dir}, {val_dir}, {test_dir}")
    return train_dir, val_dir, test_dir


def audit_split(
    split_dir: Path,
    sample_size: int = 100,
    visualize: bool = True,
) -> Dict:
    """Audit a dataset split and visualize metrics."""
    logger = logging.getLogger(__name__)

    auditor = PatchQualityAuditor()
    validator = PatchValidator()

    # Get sample of files for audit
    patch_files = list(split_dir.glob("*.npz"))
    sample_size = min(sample_size, len(patch_files))
    sampled_files = (
        random.sample(patch_files, sample_size) if len(patch_files) > sample_size else patch_files
    )

    logger.info(f"Auditing {len(sampled_files)} files from {split_dir}")

    # Collect audit metrics
    metrics = []
    valid_count = 0
    error_count = 0

    for patch_file in tqdm(sampled_files, desc=f"Auditing {split_dir.name}"):
        try:
            # Validate file format first
            if validator.validate_file(patch_file):
                valid_count += 1

                # Perform quality audit if valid
                audit_result = auditor.audit_patch_quality(patch_file)
                metrics.append(audit_result["metrics"])
            else:
                error_count += 1
                logger.warning(f"Invalid file: {patch_file}")
        except Exception as e:
            error_count += 1
            logger.error(f"Error auditing {patch_file}: {e}")

    # Compute aggregate metrics
    aggregate_metrics = {}
    if metrics:
        # Compute averages for numerical metrics
        if metrics[0].get("air_ratio") is not None:
            aggregate_metrics["avg_air_ratio"] = sum(m["air_ratio"] for m in metrics) / len(metrics)

        if metrics[0].get("spatial_entropy") is not None:
            aggregate_metrics["avg_spatial_entropy"] = sum(
                m["spatial_entropy"] for m in metrics
            ) / len(metrics)

        if metrics[0].get("quality_score") is not None:
            aggregate_metrics["avg_quality_score"] = sum(m["quality_score"] for m in metrics) / len(
                metrics
            )

        # Count biome variance
        if metrics[0].get("has_biome_variance") is not None:
            biome_var_count = sum(1 for m in metrics if m["has_biome_variance"])
            aggregate_metrics["biome_variance_pct"] = (biome_var_count / len(metrics)) * 100

        # Block type distribution (aggregate)
        if metrics[0].get("block_type_counts") is not None:
            # Combine all block type counts
            combined_counts = {}
            for m in metrics:
                for block_id, count in m["block_type_counts"].items():
                    if block_id in combined_counts:
                        combined_counts[block_id] += count
                    else:
                        combined_counts[block_id] = count

            # Keep top 10 most common block types
            top_blocks = sorted(combined_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            aggregate_metrics["top_block_types"] = {str(k): v for k, v in top_blocks}

    # Add validation stats
    aggregate_metrics["valid_pct"] = (valid_count / len(sampled_files)) * 100
    aggregate_metrics["error_pct"] = (error_count / len(sampled_files)) * 100

    logger.info(f"Audit complete for {split_dir.name}: {aggregate_metrics}")
    return aggregate_metrics


def generate_corpus(
    config: Dict,
    seeds: List[Union[int, str]],
    world_radius: int = 4,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate complete training corpus using multiple seeds."""
    logger = logging.getLogger(__name__)

    # Initialize output directory
    if not output_dir:
        timestamp = int(time.time())
        output_dir = Path(config["data"]["processed_data_dir"]) / f"corpus_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating corpus in {output_dir}")

    # Process each seed
    final_linked_dirs = []

    for seed in seeds:
        logger.info(f"Processing seed {seed}")

        # Check disk space before starting
        if not monitor_disk_space(config["worldgen"].get("max_temp_disk_gb", 5)):
            logger.error("Insufficient disk space - aborting")
            raise RuntimeError("Insufficient disk space for corpus generation")

        try:
            # Generate world for this seed
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Generate world
            world_dir = generate_world_batch(config, seed, seed_dir, world_radius)

            # Step 2: Extract chunks
            chunk_dir = extract_chunks(config, world_dir, seed_dir)

            # Step 3: Create patch pairs
            pairs_dir = create_patch_pairs(config, chunk_dir, seed_dir)

            # Step 4: Link with seed inputs
            linked_dir = link_with_seed_inputs(config, pairs_dir, seed_dir)

            final_linked_dirs.append(linked_dir)

            # Clean up temporary files to save disk space
            if config["data"].get("cleanup_temp", True):
                # Keep extracted chunks and final linked pairs, remove intermediate steps
                if world_dir.exists():
                    shutil.rmtree(world_dir)
                logger.info(f"Cleaned up temporary files for seed {seed}")

        except Exception as e:
            logger.error(f"Error processing seed {seed}: {e}")
            # Continue with next seed

    # Combine all linked directories into final dataset
    final_dataset_dir = output_dir / "combined"
    final_dataset_dir.mkdir(exist_ok=True)

    file_count = 0
    for linked_dir in final_linked_dirs:
        for patch_file in linked_dir.glob("*.npz"):
            shutil.copy2(patch_file, final_dataset_dir / patch_file.name)
            file_count += 1

    logger.info(f"Combined {file_count} files into final dataset: {final_dataset_dir}")

    # Split dataset
    train_dir, val_dir, test_dir = split_dataset(
        final_dataset_dir,
        output_dir,
        train_ratio=config["data"].get("train_split", 0.7),
        val_ratio=config["data"].get("val_split", 0.2),
    )

    # Audit each split
    audit_train = audit_split(train_dir)
    audit_val = audit_split(val_dir)
    audit_test = audit_split(test_dir)

    # Save audit results
    audit_results = {
        "train": audit_train,
        "val": audit_val,
        "test": audit_test,
        "total_examples": file_count,
        "generation_timestamp": time.time(),
        "seeds_used": seeds,
        "world_radius": world_radius,
    }

    with open(output_dir / "audit_results.yml", "w") as f:
        yaml.dump(audit_results, f, default_flow_style=False)

    logger.info(f"Corpus generation complete: {file_count} total examples")
    return output_dir


def parse_seed_range(seed_range: str) -> List[int]:
    """Parse seed range string into list of seeds."""
    if "-" in seed_range:
        start, end = map(int, seed_range.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(s) for s in seed_range.split(",")]


def main():
    """Main entry point for corpus generation."""
    parser = argparse.ArgumentParser(description="VoxelTree Training Corpus Generator")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--seed-range",
        type=str,
        default="1000-1010",
        help="Range of seeds (e.g., '1000-1100') or comma-separated list (e.g., '1000,1001,1002')",
    )
    parser.add_argument(
        "--world-radius",
        type=int,
        default=4,
        help="Radius of world generation in regions (each region is 512×512 blocks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for corpus (defaults to processed_data_dir from config)",
    )
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

    try:
        # Load config
        config = load_config(args.config)

        # Parse seeds
        seeds = parse_seed_range(args.seed_range)
        logger.info(f"Using {len(seeds)} seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")

        # Generate corpus
        output_dir = generate_corpus(
            config, seeds, world_radius=args.world_radius, output_dir=args.output_dir
        )

        logger.info(f"Corpus generation completed successfully: {output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Corpus generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
