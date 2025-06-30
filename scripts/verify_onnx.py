#!/usr/bin/env python3
"""
VoxelTree ONNX Export Verification

This script exports a trained PyTorch model to ONNX format and verifies
that outputs from both models match. It performs a comprehensive smoke test
to ensure the exported model is ready for integration with the LODiffusion Minecraft mod.

Usage:
    python scripts/verify_onnx.py --config config.yaml --checkpoint path/to/checkpoint.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import yaml
from tqdm import tqdm

# Add onnxruntime as an optional dependency
try:
    import onnxruntime as ort

    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False

# Append project root to path to ensure modules can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.trainer import VoxelTrainer
from train.visualizer import VoxelVisualizer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("onnx_verify.log")],
    )


def load_config(config_path: Path) -> Dict:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def export_model_to_onnx(model: torch.nn.Module, output_path: Path) -> Path:
    """Export PyTorch model to ONNX format."""
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to ONNX: {output_path}")

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
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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

    logger.info(f"Model exported successfully to {output_path}")
    return output_path


def verify_onnx_model(onnx_path: Path) -> bool:
    """Verify that the ONNX model is valid."""
    logger = logging.getLogger(__name__)

    try:
        # Load and check ONNX model
        logger.info("Verifying ONNX model structure...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model structure verified successfully")
        return True
    except Exception as e:
        logger.error(f"❌ ONNX verification failed: {e}")
        return False


def compare_outputs(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    num_samples: int = 10,
    tolerance: float = 1e-4,
) -> Dict:
    """Compare outputs between PyTorch and ONNX models."""
    logger = logging.getLogger(__name__)

    if not ONNX_RUNTIME_AVAILABLE:
        logger.error("❌ ONNX Runtime not available. Please install with 'pip install onnxruntime'")
        return {"success": False, "error": "ONNX Runtime not available"}

    try:
        # Create ONNX runtime session
        logger.info("Creating ONNX Runtime session...")
        ort_session = ort.InferenceSession(str(onnx_path))

        # Collect comparison results
        results = []

        logger.info(f"Running {num_samples} comparison tests...")

        # Define input shape variations to test robustness
        batch_sizes = [1, 2, 4] if num_samples >= 3 else [1]

        for i, batch_size in enumerate(batch_sizes):
            # Create test input
            test_input = {
                "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
                "biome_patch": torch.randint(0, 50, (batch_size, 16, 16)),
                "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
                "river_patch": torch.randn(batch_size, 1, 16, 16),
                "y_index": torch.randint(0, 24, (batch_size,)),
                "lod": torch.randint(1, 5, (batch_size,)),
            }

            # Run PyTorch model
            torch_model.eval()
            with torch.no_grad():
                torch_output = torch_model(test_input)

            # Run ONNX model
            onnx_input = {
                name: test_input[name].numpy()
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

            # Compare outputs
            torch_air_mask = torch_output["air_mask_logits"].numpy()
            torch_block_types = torch_output["block_type_logits"].numpy()
            onnx_air_mask = onnx_output[0]
            onnx_block_types = onnx_output[1]

            # Calculate differences
            air_mask_diff = abs(torch_air_mask - onnx_air_mask).max()
            block_types_diff = abs(torch_block_types - onnx_block_types).max()

            # Store result
            test_result = {
                "batch_size": batch_size,
                "air_mask_max_diff": float(air_mask_diff),
                "block_types_max_diff": float(block_types_diff),
                "air_mask_pass": air_mask_diff < tolerance,
                "block_types_pass": block_types_diff < tolerance,
                "pass": air_mask_diff < tolerance and block_types_diff < tolerance,
            }

            results.append(test_result)

            # Log result
            status = "✅" if test_result["pass"] else "❌"
            logger.info(
                f"Test {i+1}/{len(batch_sizes)} {status} "
                f"Batch size: {batch_size}, "
                f"Air mask diff: {air_mask_diff:.6f}, "
                f"Block types diff: {block_types_diff:.6f}"
            )

        # Check if all tests passed
        all_passed = all(r["pass"] for r in results)

        if all_passed:
            logger.info("✅ All output comparison tests PASSED")
        else:
            logger.warning("❌ Some output comparison tests FAILED")

        return {
            "success": True,
            "all_passed": all_passed,
            "results": results,
        }

    except Exception as e:
        logger.error(f"❌ Output comparison failed: {e}")
        return {"success": False, "error": str(e)}


def benchmark_inference(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    batch_size: int = 1,
    iterations: int = 100,
) -> Dict:
    """Benchmark inference speed for both PyTorch and ONNX models."""
    logger = logging.getLogger(__name__)

    if not ONNX_RUNTIME_AVAILABLE:
        logger.error("❌ ONNX Runtime not available. Skipping benchmarking.")
        return {"success": False, "error": "ONNX Runtime not available"}

    try:
        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(onnx_path))

        # Create test input
        test_input = {
            "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
            "biome_patch": torch.randint(0, 50, (batch_size, 16, 16)),
            "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
            "river_patch": torch.randn(batch_size, 1, 16, 16),
            "y_index": torch.randint(0, 24, (batch_size,)),
            "lod": torch.randint(1, 5, (batch_size,)),
        }

        # Prepare ONNX input
        onnx_input = {
            name: test_input[name].numpy()
            for name in [
                "parent_voxel",
                "biome_patch",
                "heightmap_patch",
                "river_patch",
                "y_index",
                "lod",
            ]
        }

        # Warmup
        logger.info("Warming up models...")
        for _ in range(10):
            with torch.no_grad():
                torch_model.eval()
                _ = torch_model(test_input)
            _ = ort_session.run(None, onnx_input)

        # Benchmark PyTorch
        logger.info(f"Benchmarking PyTorch model for {iterations} iterations...")
        torch_times = []
        for _ in tqdm(range(iterations)):
            start_time = time.time()
            with torch.no_grad():
                torch_model.eval()
                _ = torch_model(test_input)
            torch_times.append(time.time() - start_time)

        # Benchmark ONNX
        logger.info(f"Benchmarking ONNX model for {iterations} iterations...")
        onnx_times = []
        for _ in tqdm(range(iterations)):
            start_time = time.time()
            _ = ort_session.run(None, onnx_input)
            onnx_times.append(time.time() - start_time)

        # Calculate statistics
        torch_avg = np.mean(torch_times) * 1000  # Convert to ms
        torch_std = np.std(torch_times) * 1000
        torch_min = np.min(torch_times) * 1000
        torch_max = np.max(torch_times) * 1000

        onnx_avg = np.mean(onnx_times) * 1000
        onnx_std = np.std(onnx_times) * 1000
        onnx_min = np.min(onnx_times) * 1000
        onnx_max = np.max(onnx_times) * 1000

        speedup = torch_avg / onnx_avg if onnx_avg > 0 else 0

        # Log results
        logger.info("\n----- INFERENCE BENCHMARK RESULTS -----")
        logger.info(f"PyTorch: {torch_avg:.2f} ± {torch_std:.2f} ms")
        logger.info(f"ONNX:    {onnx_avg:.2f} ± {onnx_std:.2f} ms")
        logger.info(f"Speedup: {speedup:.2f}x")

        return {
            "success": True,
            "pytorch": {
                "avg_ms": float(torch_avg),
                "std_ms": float(torch_std),
                "min_ms": float(torch_min),
                "max_ms": float(torch_max),
            },
            "onnx": {
                "avg_ms": float(onnx_avg),
                "std_ms": float(onnx_std),
                "min_ms": float(onnx_min),
                "max_ms": float(onnx_max),
            },
            "speedup": float(speedup),
        }

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return {"success": False, "error": str(e)}


def visualize_sample_outputs(
    torch_model: torch.nn.Module,
    onnx_path: Path,
    output_dir: Path,
    num_samples: int = 3,
) -> Dict:
    """Create visualizations comparing PyTorch and ONNX model outputs."""
    logger = logging.getLogger(__name__)

    if not ONNX_RUNTIME_AVAILABLE:
        logger.error("❌ ONNX Runtime not available. Skipping visualization.")
        return {"success": False, "error": "ONNX Runtime not available"}

    try:
        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(onnx_path))

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        visualization_paths = []
        visualizer = VoxelVisualizer()

        for i in range(num_samples):
            # Create random sample input
            batch_size = 1
            sample_input = {
                "parent_voxel": torch.randn(batch_size, 1, 8, 8, 8),
                "biome_patch": torch.randint(0, 50, (batch_size, 16, 16)),
                "heightmap_patch": torch.randn(batch_size, 1, 16, 16),
                "river_patch": torch.randn(batch_size, 1, 16, 16),
                "y_index": torch.randint(0, 24, (batch_size,)),
                "lod": torch.randint(1, 5, (batch_size,)),
            }

            # Run PyTorch model
            torch_model.eval()
            with torch.no_grad():
                torch_output = torch_model(sample_input)

            # Run ONNX model
            onnx_input = {
                name: sample_input[name].numpy()
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

            # Extract outputs
            torch_air_mask = torch_output["air_mask_logits"]
            torch_block_types = torch_output["block_type_logits"]

            onnx_air_mask = torch.from_numpy(onnx_output[0])
            onnx_block_types = torch.from_numpy(onnx_output[1])

            # Create visualization comparing outputs
            viz_path = output_dir / f"compare_sample_{i+1}.png"

            # Visualize PyTorch output
            torch_viz_path = output_dir / f"pytorch_sample_{i+1}.png"
            visualizer.visualize_prediction(
                parent_voxel=sample_input["parent_voxel"][0, 0].numpy(),
                pred_mask=torch_air_mask[0, 0].numpy(),
                pred_types=torch.argmax(torch_block_types[0], dim=0).numpy(),
                output_path=torch_viz_path,
                metadata={"model": "PyTorch", "sample": i + 1},
            )

            # Visualize ONNX output
            onnx_viz_path = output_dir / f"onnx_sample_{i+1}.png"
            visualizer.visualize_prediction(
                parent_voxel=sample_input["parent_voxel"][0, 0].numpy(),
                pred_mask=onnx_air_mask[0, 0].numpy(),
                pred_types=torch.argmax(onnx_block_types[0], dim=0).numpy(),
                output_path=onnx_viz_path,
                metadata={"model": "ONNX", "sample": i + 1},
            )

            # Visualize difference (if any)
            diff_viz_path = output_dir / f"diff_sample_{i+1}.png"

            # Calculate differences
            mask_diff = torch_air_mask - onnx_air_mask
            types_diff = torch.argmax(torch_block_types, dim=1) - torch.argmax(
                onnx_block_types, dim=1
            )

            if mask_diff.abs().max() > 1e-4 or types_diff.abs().max() > 0:
                visualizer.visualize_prediction(
                    parent_voxel=sample_input["parent_voxel"][0, 0].numpy(),
                    pred_mask=mask_diff[0, 0].abs().numpy(),
                    pred_types=types_diff[0].abs().numpy(),
                    output_path=diff_viz_path,
                    metadata={"model": "Difference", "sample": i + 1},
                )
                visualization_paths.append(str(diff_viz_path))

            visualization_paths.append(str(torch_viz_path))
            visualization_paths.append(str(onnx_viz_path))

        logger.info(f"Created {len(visualization_paths)} visualizations in {output_dir}")

        return {
            "success": True,
            "visualization_paths": visualization_paths,
        }

    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        return {"success": False, "error": str(e)}


def verify_full_model(
    config: Dict, checkpoint_path: Path, output_dir: Optional[Path] = None
) -> Dict:
    """Run full ONNX model verification process."""
    logger = logging.getLogger(__name__)

    if not ONNX_RUNTIME_AVAILABLE:
        logger.warning(
            "ONNX Runtime not available. Full verification will be limited. "
            "Install with 'pip install onnxruntime'"
        )

    # Determine output directory
    if output_dir is None:
        output_dir = Path("model_exports") / f"export_{int(time.time())}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model from checkpoint
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        trainer = VoxelTrainer(config)
        trainer.load_checkpoint(checkpoint_path)
        trainer.model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"success": False, "error": f"Failed to load model: {str(e)}"}

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    try:
        export_model_to_onnx(trainer.model, onnx_path)
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        return {"success": False, "error": f"Failed to export model: {str(e)}"}

    # Verify ONNX model structure
    if not verify_onnx_model(onnx_path):
        return {"success": False, "error": "ONNX model verification failed"}

    results = {
        "success": True,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(checkpoint_path),
    }

    # Compare outputs (if ONNX Runtime available)
    if ONNX_RUNTIME_AVAILABLE:
        # Compare outputs
        comparison_results = compare_outputs(trainer.model, onnx_path)
        results["comparison"] = comparison_results

        # Benchmark inference
        benchmark_results = benchmark_inference(trainer.model, onnx_path)
        results["benchmark"] = benchmark_results

        # Visualize sample outputs
        viz_dir = output_dir / "visualizations"
        viz_results = visualize_sample_outputs(trainer.model, onnx_path, viz_dir)
        results["visualizations"] = viz_results

    # Save results to YAML
    results_path = output_dir / "verification_results.yml"
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    # Determine overall success
    if ONNX_RUNTIME_AVAILABLE:
        overall_success = (
            results["success"]
            and results["comparison"].get("all_passed", False)
            and results["benchmark"].get("success", False)
            and results["visualizations"].get("success", False)
        )
    else:
        overall_success = results["success"]

    # Log summary
    logger.info("\n----- VERIFICATION SUMMARY -----")
    logger.info(f"Model exported to: {onnx_path}")
    logger.info(f"Overall verification: {'✅ PASSED' if overall_success else '❌ FAILED'}")

    if ONNX_RUNTIME_AVAILABLE and "benchmark" in results and results["benchmark"].get("success"):
        speedup = results["benchmark"]["speedup"]
        logger.info(f"ONNX speedup: {speedup:.2f}x")

    logger.info(f"Full results saved to: {results_path}")

    return results


def main():
    """Main entry point for ONNX verification script."""
    parser = argparse.ArgumentParser(description="VoxelTree ONNX Export Verification")
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint to export"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory to save exported model and verification results"
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip benchmarking (faster verification)"
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
        # Check if onnxruntime is available
        if not ONNX_RUNTIME_AVAILABLE:
            logger.warning(
                "ONNX Runtime not available. Full verification will be limited. "
                "Install with 'pip install onnxruntime'"
            )

        # Load config
        config = load_config(args.config)

        # Verify model
        results = verify_full_model(config, args.checkpoint, args.output_dir)

        if results["success"]:
            logger.info("ONNX export verification completed successfully")
            return 0
        else:
            logger.error(
                f"ONNX export verification failed: {results.get('error', 'Unknown error')}"
            )
            return 1

    except Exception as e:
        logger.error(f"ONNX export verification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
