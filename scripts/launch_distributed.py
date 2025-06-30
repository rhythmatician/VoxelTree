#!/usr/bin/env python3
"""
VoxelTree Distributed Training Launch Helper

This script helps launch distributed training across multiple GPUs using
PyTorch's distributed data parallel (DDP) or Ray if available.

Usage:
    python scripts/launch_distributed.py --config config.yaml --num-gpus 4
    python scripts/launch_distributed.py --config config.yaml --ray
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_gpu_availability() -> int:
    """Check number of available GPUs."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} CUDA GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"  GPU {i}: {gpu_name}")
        return num_gpus
    else:
        logging.warning("No CUDA GPUs available, will use CPU")
        return 0


def launch_ddp_training(config_path: Path, num_gpus: int, additional_args: List[str] = None):
    """Launch distributed training using PyTorch DDP."""
    if additional_args is None:
        additional_args = []

    logging.info(f"Launching DDP training on {num_gpus} GPUs")

    # Prepare the command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",
        "train.py",
        "--config",
        str(config_path),
        "--distributed",
    ] + additional_args

    logging.info(f"Command: {' '.join(cmd)}")

    # Launch the process
    try:
        result = subprocess.run(cmd, check=True)
        logging.info("Training completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with exit code {e.returncode}")
        return e.returncode


def launch_ray_training(config_path: Path, num_gpus: int, additional_args: List[str] = None):
    """Launch distributed training using Ray."""
    try:
        import ray
        from ray import tune
    except ImportError:
        logging.error("Ray is not installed. Install with: pip install ray[tune]")
        return 1

    if additional_args is None:
        additional_args = []

    logging.info(f"Launching Ray training on {num_gpus} GPUs")

    # Initialize Ray
    ray.init(num_gpus=num_gpus)

    # TODO: Implement Ray training launch
    # This is a placeholder for Ray-based distributed training
    logging.warning("Ray training launch is not yet implemented")
    logging.info("Falling back to DDP training")

    ray.shutdown()
    return launch_ddp_training(config_path, num_gpus, additional_args)


def launch_single_gpu_training(
    config_path: Path, gpu_id: int = 0, additional_args: List[str] = None
):
    """Launch single GPU training."""
    if additional_args is None:
        additional_args = []

    logging.info(f"Launching single GPU training on GPU {gpu_id}")

    # Set CUDA_VISIBLE_DEVICES to use specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "train.py", "--config", str(config_path)] + additional_args

    logging.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, check=True)
        logging.info("Training completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with exit code {e.returncode}")
        return e.returncode


def launch_cpu_training(config_path: Path, additional_args: List[str] = None):
    """Launch CPU-only training."""
    if additional_args is None:
        additional_args = []

    logging.info("Launching CPU training")

    cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
    ] + additional_args

    logging.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        logging.info("Training completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with exit code {e.returncode}")
        return e.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch VoxelTree distributed training")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument(
        "--num-gpus", type=int, help="Number of GPUs to use (auto-detect if not specified)"
    )
    parser.add_argument("--gpu-id", type=int, help="Specific GPU ID for single GPU training")
    parser.add_argument("--ray", action="store_true", help="Use Ray for distributed training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument(
        "--action",
        choices=["generate", "train", "export"],
        default="train",
        help="Training action to perform",
    )

    args, unknown_args = parser.parse_known_args()

    setup_logging()

    # Validate config file
    if not args.config.exists():
        logging.error(f"Config file not found: {args.config}")
        return 1

    # Load config to check distributed settings
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return 1

    # Add action to unknown args
    unknown_args.extend(["--action", args.action])

    # Determine training mode
    if args.cpu:
        return launch_cpu_training(args.config, unknown_args)

    available_gpus = check_gpu_availability()

    if available_gpus == 0:
        logging.info("No GPUs available, falling back to CPU training")
        return launch_cpu_training(args.config, unknown_args)

    # Determine number of GPUs to use
    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, available_gpus)
    else:
        # Check config for distributed settings
        distributed_config = config.get("training", {}).get("distributed", {})
        if distributed_config.get("enabled", False):
            num_gpus_config = distributed_config.get("num_gpus", "auto")
            if num_gpus_config == "auto":
                num_gpus = available_gpus
            else:
                num_gpus = min(int(num_gpus_config), available_gpus)
        else:
            num_gpus = 1

    logging.info(f"Using {num_gpus} out of {available_gpus} available GPUs")

    # Launch appropriate training mode
    if args.gpu_id is not None:
        # Single GPU training on specific GPU
        if args.gpu_id >= available_gpus:
            logging.error(f"GPU {args.gpu_id} not available (only {available_gpus} GPUs)")
            return 1
        return launch_single_gpu_training(args.config, args.gpu_id, unknown_args)
    elif num_gpus == 1:
        # Single GPU training
        return launch_single_gpu_training(args.config, 0, unknown_args)
    elif args.ray:
        # Multi-GPU training with Ray
        return launch_ray_training(args.config, num_gpus, unknown_args)
    else:
        # Multi-GPU training with DDP
        return launch_ddp_training(args.config, num_gpus, unknown_args)


if __name__ == "__main__":
    sys.exit(main())
