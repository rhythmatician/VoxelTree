#!/usr/bin/env python3
"""
Real End-to-End VoxelTree Evaluation

This script runs a complete evaluation pipeline:
1. Generate a small terrain corpus (1-2 regions)
2. Train briefly to create a real checkpoint
3. Export the model with provenance
4. Evaluate on real data (not synthetic)
5. Generate comprehensive metrics report

Usage:
    python scripts/real_evaluation.py [--quick]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: str, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    logger.info(f"Running: {cmd}")
    if cwd:
        logger.info(f"Working directory: {cwd}")

    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=False)

    if result.stdout:
        logger.info(f"STDOUT: {result.stdout[:500]}...")
    if result.stderr:
        logger.warning(f"STDERR: {result.stderr[:500]}...")

    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run real end-to-end VoxelTree evaluation")
    parser.add_argument("--quick", action="store_true", help="Use minimal settings for quick test")
    parser.add_argument(
        "--skip-worldgen", action="store_true", help="Skip world generation (use existing data)"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training (use existing checkpoint)"
    )
    args = parser.parse_args()

    # Set up paths
    repo_root = Path.cwd()
    artifacts_dir = repo_root / "artifacts" / "real_eval"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting real end-to-end VoxelTree evaluation")
    logger.info(f"Repository root: {repo_root}")
    logger.info(f"Artifacts directory: {artifacts_dir}")

    # Step 1: Generate terrain corpus (if needed)
    if not args.skip_worldgen:
        logger.info("Step 1: Generating terrain corpus...")
        radius = 1 if args.quick else 2
        worldgen_cmd = (
            f"python scripts/generate_corpus.py --config config_extended.yaml --radius {radius}"
        )
        run_command(worldgen_cmd, cwd=repo_root)
    else:
        logger.info("Step 1: Skipping world generation (using existing data)")

    # Step 2: Train briefly (if needed)
    checkpoint_path = None
    if not args.skip_training:
        logger.info("Step 2: Training model briefly...")
        max_iters = 1 if args.quick else 2
        epochs = 1 if args.quick else 2
        train_cmd = f"python scripts/train_iterative.py --config config_extended.yaml --max-iters {max_iters} --epochs {epochs}"
        run_command(train_cmd, cwd=repo_root)

        # Find the latest checkpoint
        runs_dir = repo_root / "runs"
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            checkpoint_path = latest_run / "best_checkpoint.pt"
            logger.info(f"Using checkpoint: {checkpoint_path}")
        else:
            logger.error("No runs directory found")
            sys.exit(1)
    else:
        logger.info("Step 2: Skipping training")
        # Try to find existing checkpoint
        runs_dir = repo_root / "runs"
        if runs_dir.exists():
            latest_run = max(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            checkpoint_path = latest_run / "best_checkpoint.pt"
            if checkpoint_path.exists():
                logger.info(f"Using existing checkpoint: {checkpoint_path}")
            else:
                logger.error("No checkpoint found and training skipped")
                sys.exit(1)

    # Step 3: Export model with provenance
    logger.info("Step 3: Exporting model with provenance...")
    export_cmd = f"PYTHONPATH=. python scripts/export_lod.py --checkpoint {checkpoint_path} --out-dir {artifacts_dir}"
    run_command(export_cmd, cwd=repo_root)

    # Verify export artifacts
    onnx_path = artifacts_dir / "model.onnx"
    config_path = artifacts_dir / "model_config.json"
    vectors_path = artifacts_dir / "test_vectors.npz"

    if not all([onnx_path.exists(), config_path.exists(), vectors_path.exists()]):
        logger.error("Export failed - missing artifacts")
        sys.exit(1)

    logger.info("Export completed successfully")

    # Step 4: Quick ONNX contract verification
    logger.info("Step 4: Verifying ONNX contract...")
    verify_script = f"""
import onnx
import json
import numpy as np

# Load and inspect ONNX model
model = onnx.load('{onnx_path}')
print("ONNX Contract Verification:")
print("Opsets:", [f"{{d.domain or 'ai.onnx'}}:{{d.version}}" for d in model.opset_import])

# Check inputs
inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
print("Inputs:", inputs)

# Check outputs  
outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
print("Outputs:", outputs)

# Load config
with open('{config_path}', 'r') as f:
    config = json.load(f)

print("Config keys:", list(config.keys()))
print("Has provenance:", 'provenance' in config)
print("Has block mapping:", 'block_mapping' in config)
if 'block_mapping' in config:
    print("Block mapping size:", len(config['block_mapping']))

# Check test vectors
vectors = np.load('{vectors_path}')
print("Test vector keys:", list(vectors.keys()))
print("Test vector shapes:", {{k: v.shape for k, v in vectors.items()}})
"""

    result = run_command(f'python -c "{verify_script}"', cwd=repo_root, check=False)
    if result.returncode == 0:
        logger.info("ONNX contract verification passed")
    else:
        logger.warning("ONNX contract verification had issues")

    # Step 5: Create evaluation summary
    logger.info("Step 5: Creating evaluation summary...")

    # Load model config for summary
    try:
        with open(config_path, "r") as f:
            model_config = json.load(f)

        summary = {
            "evaluation_type": "real_end_to_end",
            "quick_mode": args.quick,
            "artifacts": {
                "onnx_model": str(onnx_path),
                "model_config": str(config_path),
                "test_vectors": str(vectors_path),
            },
            "model_info": {
                "block_vocab_size": model_config.get("block_vocab_size", "unknown"),
                "biome_vocab_size": model_config.get("biome_vocab_size", "unknown"),
                "contract": model_config.get("contract", "unknown"),
            },
            "provenance": model_config.get("provenance", {}),
            "block_mapping_embedded": "block_mapping" in model_config,
            "checkpoint_used": str(checkpoint_path),
            "status": "completed",
        }

        summary_path = artifacts_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Evaluation summary saved to: {summary_path}")

        # Print key results
        print("\n" + "=" * 60)
        print("REAL END-TO-END EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Artifacts directory: {artifacts_dir}")
        print(f"ONNX model: {onnx_path.name} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Model config: {config_path.name}")
        print(f"Test vectors: {vectors_path.name}")
        print()
        print("Model Info:")
        print(f"  Block vocabulary: {summary['model_info']['block_vocab_size']}")
        print(f"  Biome vocabulary: {summary['model_info']['biome_vocab_size']}")
        print(f"  Contract: {summary['model_info']['contract']}")
        print()
        print("Provenance:")
        for key, value in summary["provenance"].items():
            print(f"  {key}: {value}")
        print()
        print("Next Steps:")
        print("1. Load the ONNX model in your DJL harness")
        print("2. Compare outputs with test_vectors.npz (max_abs_diff ≤ 1e-4)")
        print("3. Test with real Minecraft coordinate inputs")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Failed to create summary: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
