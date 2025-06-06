#!/usr/bin/env python3
"""
VoxelTree Training Benchmarker

This script benchmarks model training with different batch sizes and configurations
to determine optimal performance settings for full training runs.

Usage:
    python scripts/benchmark.py --config config.yaml
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm

# Append project root to path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.dataset import VoxelTreeDataset
from train.trainer import VoxelTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for benchmarking."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("benchmark.log")],
    )


def load_config(config_path: Path) -> Dict:
    """Load and validate configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_gpu_info() -> Dict:
    """Get GPU information if available."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
    }


def get_memory_usage() -> Dict:
    """Get current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss // (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms // (1024 * 1024),  # Virtual Memory Size in MB
        "system_total_mb": psutil.virtual_memory().total // (1024 * 1024),
        "system_available_mb": psutil.virtual_memory().available // (1024 * 1024),
    }


def get_gpu_memory_usage() -> Dict:
    """Get current GPU memory usage if available."""
    if not torch.cuda.is_available():
        return {"available": False}

    # Torch 1.10+ has a method to get reserved and allocated memory
    try:
        return {
            "available": True,
            "allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        }
    except AttributeError:
        # Fallback for older PyTorch versions
        return {
            "available": True,
            "allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
            "cached_mb": torch.cuda.memory_cached() // (1024 * 1024),
        }


def create_dummy_dataset(
    config: Dict,
    num_examples: int = 100,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Create a dummy dataset for benchmarking."""
    logger = logging.getLogger(__name__)

    if cache_dir is None:
        cache_dir = Path("data/benchmark_cache")

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating dummy dataset with {num_examples} examples in {cache_dir}")

    # Determine example dimensions from config
    parent_shape = (8, 8, 8)
    target_shape = (16, 16, 16)

    try:
        # Try to extract dimensions from config if available
        pairing_config = config.get("pairing", {})
        pair_format = pairing_config.get("pair_format", {})
        parent_shape = tuple(pair_format.get("parent_shape", parent_shape))
        target_shape = tuple(pair_format.get("target_shape", target_shape))
    except (KeyError, TypeError):
        logger.warning(
            f"Could not extract shapes from config, using defaults: {parent_shape}, {target_shape}"
        )

    for i in range(num_examples):
        example = {
            # Parent voxel (B×C×D×H×W)
            "parent_voxel": np.random.randint(0, 2, size=parent_shape).astype(np.float32),
            # Biome and condition data
            "biome_patch": np.random.randint(0, 50, size=(16, 16)).astype(np.int32),
            "heightmap_patch": np.random.randint(50, 100, size=(1, 16, 16)).astype(np.float32),
            "river_patch": np.random.randint(0, 2, size=(1, 16, 16)).astype(np.float32),
            # LOD and position info
            "y_index": np.array([np.random.randint(0, 24)]).astype(np.int32),
            "lod": np.array([np.random.randint(1, 5)]).astype(np.int32),
            # Target data
            "target_mask": np.random.randint(0, 2, size=(1,) + target_shape).astype(np.float32),
            "target_types": np.random.randint(0, 10, size=target_shape).astype(np.int32),
        }

        # Save to NPZ file
        output_path = cache_dir / f"example_{i:05d}.npz"
        np.savez_compressed(output_path, **example)

    logger.info(f"Created {num_examples} dummy examples")
    return cache_dir


def benchmark_batch_sizes(
    config: Dict,
    dataset_path: Path,
    batch_sizes: List[int],
    iterations: int = 10,
    warmup: int = 3,
) -> Dict:
    """Benchmark different batch sizes for training."""
    logger = logging.getLogger(__name__)
    logger.info(f"Benchmarking batch sizes: {batch_sizes}")

    # Create dataset
    try:
        dataset = VoxelTreeDataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    results = {}

    # Get baseline memory usage
    baseline_memory = get_memory_usage()
    baseline_gpu = get_gpu_memory_usage() if torch.cuda.is_available() else None

    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}")

        # Update config with current batch size
        test_config = config.copy()
        if "training" not in test_config:
            test_config["training"] = {}
        test_config["training"]["batch_size"] = batch_size

        # Create dataloader for this batch size
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
        )

        try:
            # Create trainer
            trainer = VoxelTrainer(test_config)

            # Memory usage before training
            before_memory = get_memory_usage()
            before_gpu = get_gpu_memory_usage() if torch.cuda.is_available() else None

            # Warmup
            logger.info(f"Performing {warmup} warmup iterations")
            for _ in range(warmup):
                trainer.train_one_epoch(dataloader)

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Benchmark
            times = []
            memory_usage = []
            gpu_usage = []

            logger.info(f"Running {iterations} benchmark iterations")
            for i in range(iterations):
                # Measure memory before iteration
                if i == 0:  # Only record first iteration's memory usage
                    pre_iter_memory = get_memory_usage()
                    pre_iter_gpu = get_gpu_memory_usage() if torch.cuda.is_available() else None

                # Time iteration
                start_time = time.time()
                metrics = trainer.train_one_epoch(dataloader)
                end_time = time.time()

                times.append(end_time - start_time)

                # Record memory usage after first iteration
                if i == 0:
                    post_iter_memory = get_memory_usage()
                    post_iter_gpu = get_gpu_memory_usage() if torch.cuda.is_available() else None

                    memory_usage.append(
                        {
                            "before": pre_iter_memory,
                            "after": post_iter_memory,
                            "diff_mb": post_iter_memory["rss_mb"] - pre_iter_memory["rss_mb"],
                        }
                    )

                    if torch.cuda.is_available():
                        gpu_usage.append(
                            {
                                "before": pre_iter_gpu,
                                "after": post_iter_gpu,
                                "diff_mb": post_iter_gpu["allocated_mb"]
                                - pre_iter_gpu["allocated_mb"],
                            }
                        )

            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            # Examples per second
            examples_per_sec = len(dataset) / avg_time

            # Record results
            results[batch_size] = {
                "avg_epoch_time": float(avg_time),
                "std_epoch_time": float(std_time),
                "min_epoch_time": float(min_time),
                "max_epoch_time": float(max_time),
                "examples_per_second": float(examples_per_sec),
                "memory_usage": memory_usage[0] if memory_usage else None,
                "gpu_usage": gpu_usage[0] if gpu_usage else None,
            }

            logger.info(
                f"Batch size {batch_size}: {avg_time:.2f}s ± {std_time:.2f}s per epoch, "
                f"{examples_per_sec:.2f} examples/sec"
            )

            # Clear memory
            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except RuntimeError as e:
            # Catch CUDA out of memory errors
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA out of memory for batch size {batch_size}")
                results[batch_size] = {"error": "CUDA out of memory", "message": str(e)}
            else:
                logger.error(f"Error for batch size {batch_size}: {e}")
                results[batch_size] = {"error": "Runtime error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error for batch size {batch_size}: {e}")
            results[batch_size] = {"error": type(e).__name__, "message": str(e)}

    # Determine optimal batch size based on examples/second and memory
    valid_batch_sizes = {bs: info for bs, info in results.items() if "error" not in info}

    if valid_batch_sizes:
        # Find batch size with highest examples/second
        optimal_throughput = max(
            valid_batch_sizes.items(), key=lambda x: x[1]["examples_per_second"]
        )[0]

        # Find batch size with best balance of throughput vs memory
        # This is more subjective - here we use examples/sec divided by memory increase
        memory_efficiency = {}
        for bs, info in valid_batch_sizes.items():
            memory_diff = info["memory_usage"]["diff_mb"] if info["memory_usage"] else 0
            if memory_diff <= 0:
                memory_diff = 1  # Avoid division by zero
            memory_efficiency[bs] = info["examples_per_second"] / memory_diff

        optimal_balance = max(memory_efficiency.items(), key=lambda x: x[1])[0]

        # Update summary
        summary = {
            "optimal_batch_size_throughput": optimal_throughput,
            "optimal_batch_size_balanced": optimal_balance,
            "recommended": optimal_balance,  # Default recommendation
            "system_info": {
                "cpu_count": os.cpu_count(),
                "base_memory_mb": baseline_memory["rss_mb"],
                "total_memory_mb": baseline_memory["system_total_mb"],
                "gpu_info": get_gpu_info(),
            },
        }

        results["summary"] = summary

    return results


def benchmark_trainer_performance(config: Dict, dataset_path: Path) -> Dict:
    """Benchmark trainer performance metrics beyond just batch size."""
    logger = logging.getLogger(__name__)

    # Create dataset
    try:
        dataset = VoxelTreeDataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    # Use recommended batch size from config
    batch_size = config.get("training", {}).get("batch_size", 32)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
    )

    # Metric tracking
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []

    # Create trainer with timing hooks
    trainer = VoxelTrainer(config)

    # Run benchmark
    logger.info("Running performance profiling for 10 iterations")
    for _ in range(10):
        # Track performance timings
        start_time = time.time()

        for batch in dataloader:
            # Forward pass timing
            forward_start = time.time()
            trainer.model.train()
            batch = {
                k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = trainer.model(
                parent_voxel=batch["parent_voxel"],
                biome_patch=batch["biome_patch"],
                heightmap_patch=batch["heightmap_patch"],
                river_patch=batch["river_patch"],
                y_index=batch["y_index"],
                lod=batch["lod"],
            )

            from train.losses import voxel_loss_fn

            loss = voxel_loss_fn(
                air_mask_logits=outputs["air_mask_logits"],
                block_type_logits=outputs["block_type_logits"],
                target_mask=batch["target_mask"],
                target_types=batch["target_types"],
                mask_weight=trainer.mask_weight,
                type_weight=trainer.type_weight,
            )
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)

            # Backward pass timing
            backward_start = time.time()
            trainer.optimizer.zero_grad()
            loss.backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            # Optimizer step timing
            optimizer_start = time.time()
            trainer.optimizer.step()
            optimizer_end = time.time()
            optimizer_times.append(optimizer_end - optimizer_start)

        epoch_time = time.time() - start_time
        total_times.append(epoch_time)

    # Calculate statistics
    results = {
        "batch_size": batch_size,
        "dataset_size": len(dataset),
        "forward_pass": {
            "avg_ms": float(np.mean(forward_times) * 1000),
            "min_ms": float(np.min(forward_times) * 1000),
            "max_ms": float(np.max(forward_times) * 1000),
        },
        "backward_pass": {
            "avg_ms": float(np.mean(backward_times) * 1000),
            "min_ms": float(np.min(backward_times) * 1000),
            "max_ms": float(np.max(backward_times) * 1000),
        },
        "optimizer_step": {
            "avg_ms": float(np.mean(optimizer_times) * 1000),
            "min_ms": float(np.min(optimizer_times) * 1000),
            "max_ms": float(np.max(optimizer_times) * 1000),
        },
        "total_epoch": {
            "avg_s": float(np.mean(total_times)),
            "min_s": float(np.min(total_times)),
            "max_s": float(np.max(total_times)),
        },
        "memory": get_memory_usage(),
        "gpu": get_gpu_memory_usage() if torch.cuda.is_available() else None,
    }

    # Calculate bottleneck analysis
    total_step_time = np.mean(forward_times) + np.mean(backward_times) + np.mean(optimizer_times)
    results["bottleneck_analysis"] = {
        "forward_pct": float(np.mean(forward_times) / total_step_time * 100),
        "backward_pct": float(np.mean(backward_times) / total_step_time * 100),
        "optimizer_pct": float(np.mean(optimizer_times) / total_step_time * 100),
    }

    # Theoretical throughput
    results["theoretical_throughput"] = {
        "examples_per_second": float(len(dataset) / np.mean(total_times)),
        "batches_per_second": float(len(dataloader) / np.mean(total_times)),
    }

    return results


def main():
    """Main entry point for benchmarking script."""
    parser = argparse.ArgumentParser(description="VoxelTree Training Benchmarker")
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to existing dataset for benchmarking (if not provided, will create dummy dataset)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of dummy examples to create if dataset path not provided",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations for each batch size test"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="benchmark_results.yml",
        help="Path to save benchmark results",
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
        logger.info(f"Loaded configuration from {args.config}")

        # Parse batch sizes
        batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")]

        # Ensure dataset exists
        if args.dataset_path and args.dataset_path.exists():
            dataset_path = args.dataset_path
            logger.info(f"Using existing dataset: {dataset_path}")
        else:
            logger.info("Creating dummy dataset...")
            dataset_path = create_dummy_dataset(config, args.num_examples)

        # Run batch size benchmarks
        logger.info(f"Starting batch size benchmark with {len(batch_sizes)} different sizes")
        batch_results = benchmark_batch_sizes(
            config, dataset_path, batch_sizes, iterations=args.iterations
        )

        # Run detailed performance benchmarks
        logger.info("Starting detailed performance benchmark")
        performance_results = benchmark_trainer_performance(config, dataset_path)

        # Combine results
        results = {
            "batch_size_results": batch_results,
            "performance_results": performance_results,
            "system_info": {
                "timestamp": time.time(),
                "cpu_count": os.cpu_count(),
                "memory": get_memory_usage(),
                "gpu": get_gpu_info(),
            },
        }

        # Output results
        with open(args.output_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        # Print summary
        if "summary" in batch_results:
            summary = batch_results["summary"]
            logger.info("\n----- BENCHMARK SUMMARY -----")
            logger.info(f"Recommended batch size: {summary['recommended']}")
            logger.info(f"Best throughput: {summary['optimal_batch_size_throughput']}")
            logger.info(f"Best balanced: {summary['optimal_batch_size_balanced']}")

            # Print bottleneck analysis
            bottleneck = performance_results["bottleneck_analysis"]
            logger.info("\n----- BOTTLENECK ANALYSIS -----")
            logger.info(f"Forward pass:    {bottleneck['forward_pct']:.1f}%")
            logger.info(f"Backward pass:   {bottleneck['backward_pct']:.1f}%")
            logger.info(f"Optimizer step:  {bottleneck['optimizer_pct']:.1f}%")

        logger.info(f"Full benchmark results saved to {args.output_file}")
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import os  # Required for os.cpu_count

    sys.exit(main())
