#!/usr/bin/env python3
"""Empirical calibration of sparse-root split threshold.

.. note::
    Shape comments in this file reference the legacy 4×2×4 spatial format.
    The code uses dynamic shapes (``C3``) and should work with v7 (4×4×4)
    data, but has not been validated against it.

Analyzes the distribution of root split logits from the ONNX model on a sample
of training data to recommend a threshold that balances tree expansion with
computational cost.

Usage
-----
  python VoxelTree/scripts/sparse_octree/calibrate.py \\
      --data noise_training_data/sparse_octree_pairs.npz \\
      --model LODiffusion/run/models/sparse_octree_fast80/sparse_octree.onnx \\
      [--samples 500] [--target-expand-rate 0.5]

Output
------
  - Print summary statistics of root split sigmoid distribution
  - Recommend threshold based on target expansion rate
  - Show impact of different threshold choices

The model exports its outputs in a deterministic order:
  [0] split_L4   [1,1]      root split logit
  [1] label_L4   [1,1,C]    root label logits
  [2] split_L3   [1,8]      ...
  ... etc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("noise_training_data/sparse_octree_pairs.npz"),
        help="Path to sparse_octree_pairs.npz training data",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to sparse_octree.onnx model (required unless --config is given)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to sparse_octree_config.json (will auto-locate model if --model not given)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of sections to sample from training data (default 500)",
    )
    parser.add_argument(
        "--target-expand-rate",
        type=float,
        default=0.5,
        help="Target fraction of sections that should expand at root (0.0-1.0, default 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-section logits",
    )
    return parser


def load_training_data(npz_path: Path, num_samples: int, seed: int = 42) -> np.ndarray:
    """Load training data and sample sections.

    Args:
        npz_path: Path to sparse_octree_pairs.npz
        num_samples: Number of sections to sample
        seed: Random seed

    Returns:
        noise_3d array of shape (num_samples, C3, 4, 2, 4)
    """
    print(f"Loading training data from {npz_path}...")
    data = np.load(npz_path)
    noise_3d = data["noise_3d"]

    total_samples = len(noise_3d)
    if num_samples >= total_samples:
        print(f"  Found {total_samples} sections; using all")
        return noise_3d.astype(np.float32)

    rng = np.random.RandomState(seed)
    indices = rng.choice(total_samples, size=num_samples, replace=False)
    sampled = noise_3d[indices]
    print(f"  Sampled {num_samples} / {total_samples} sections")
    return sampled.astype(np.float32)


def find_model_path(config_path: Path) -> Path:
    """Locate ONNX model from config file location."""
    config_dir = config_path.parent
    onnx_path = config_dir / "sparse_octree.onnx"
    if onnx_path.exists():
        return onnx_path

    # Try without directory suffix
    stem = config_dir.stem
    if stem.startswith("sparse_octree_"):
        return config_dir / "sparse_octree.onnx"

    raise FileNotFoundError(
        f"Cannot locate sparse_octree.onnx near {config_path}; " f"please specify --model explicitly"
    )


def run_inference(
    session: ort.InferenceSession,
    noise_3d: np.ndarray,
) -> np.ndarray:
    """Run ONNX inference and return root split logits.

    Args:
        session: ONNX Runtime session
        noise_3d: Unbatched noise_3d of shape (C3, 4, 2, 4)

    Returns:
        Root split logit (scalar float)
    """
    # Add batch dimension
    batched = noise_3d[np.newaxis, ...]  # (1, C3, 4, 2, 4)

    # Get input name (usually 'noise_3d' but check session)
    input_names = [inp.name for inp in session.get_inputs()]
    if not input_names:
        raise RuntimeError("ONNX model has no inputs")

    # Feed first input (should be noise_3d)
    inputs = {input_names[0]: batched}
    outputs = session.run(None, inputs)

    # First output is split_L4 with shape (1, 1)
    split_logit = outputs[0][0, 0]
    return split_logit


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def analyze_thresholds(
    sigmoids: np.ndarray,
    target_expand_rate: float = 0.5,
) -> dict[str, Any]:
    """Analyze sigmoid distribution and recommend thresholds.

    Args:
        sigmoids: Array of root split sigmoids (after sigmoid applied to logits)
        target_expand_rate: Desired fraction of sections that expand (0.0-1.0)

    Returns:
        Dictionary with statistics and recommendations
    """
    results = {}

    # Percentile-based stats
    results["count"] = len(sigmoids)
    results["mean_sigmoid"] = float(np.mean(sigmoids))
    results["std_sigmoid"] = float(np.std(sigmoids))
    results["min_sigmoid"] = float(np.min(sigmoids))
    results["max_sigmoid"] = float(np.max(sigmoids))
    results["median_sigmoid"] = float(np.median(sigmoids))

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    results["percentiles"] = {p: float(np.percentile(sigmoids, p)) for p in percentiles}

    # Recommended threshold by target expansion rate
    # To achieve target_expand_rate, we want the (100 - target_expand_rate*100)-th percentile
    recommended_percentile = (1.0 - target_expand_rate) * 100
    recommended_threshold = float(np.percentile(sigmoids, recommended_percentile))
    results["recommended_threshold"] = recommended_threshold
    results["recommended_percentile"] = recommended_percentile
    results["target_expand_rate"] = target_expand_rate

    # Show impact of different thresholds
    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results["threshold_impact"] = {}
    for thresh in test_thresholds:
        expand_rate = float(np.mean(sigmoids >= thresh))
        results["threshold_impact"][thresh] = expand_rate

    return results


def main() -> None:
    args = create_argument_parser().parse_args()

    # Resolve model path
    if args.model:
        model_path = args.model
    elif args.config:
        model_path = find_model_path(args.config)
    else:
        raise ValueError("Must specify either --model or --config")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not args.data.exists():
        raise FileNotFoundError(f"Data not found: {args.data}")

    print(f"\n{'='*70}")
    print("SPARSE ROOT SPLIT THRESHOLD CALIBRATION")
    print(f"{'='*70}")

    # Load data
    noise_3d_samples = load_training_data(args.data, args.samples, seed=args.seed)

    # Load ONNX model
    print(f"\nLoading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print(f"  ONNX model loaded; {len(session.get_outputs())} outputs")

    # Run inference on all samples
    print(f"\nRunning inference on {len(noise_3d_samples)} sections...")
    logits = []
    for i, noise_3d in enumerate(noise_3d_samples):
        logit = run_inference(session, noise_3d)
        logits.append(logit)
        if args.verbose and i < 10:
            print(f"  Section {i}: logit={logit:.4f}, sigmoid={sigmoid(logit):.4f}")
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1} / {len(noise_3d_samples)}")

    logits = np.array(logits)
    sigmoids = sigmoid(logits)

    # Analyze results
    results = analyze_thresholds(sigmoids, target_expand_rate=args.target_expand_rate)

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS (n={len(sigmoids)} sections)")
    print(f"{'='*70}")
    print("Root split sigmoid distribution:")
    print(f"  Mean:     {results['mean_sigmoid']:.4f}")
    print(f"  Std:      {results['std_sigmoid']:.4f}")
    print(f"  Min:      {results['min_sigmoid']:.4f}")
    print(f"  Max:      {results['max_sigmoid']:.4f}")
    print(f"  Median:   {results['median_sigmoid']:.4f}")

    print("\nPercentiles:")
    for p, val in results["percentiles"].items():
        print(f"  {p:3d}th:   {val:.4f}")

    print(f"\n{'RECOMMENDATION':-^70}")
    target_rate = args.target_expand_rate
    rec_thresh = results["recommended_threshold"]
    rec_percentile = results["recommended_percentile"]
    print(f"To achieve ~{target_rate*100:.0f}% root expansion:")
    print(f"  Set splitThreshold = {rec_thresh:.4f}")
    print(f"  (this is the {rec_percentile:.1f}th percentile of observed sigmoids)")

    print(f"\n{'IMPACT OF DIFFERENT THRESHOLDS':-^70}")
    print(f"{'Threshold':<15} {'Expansion Rate':<20}")
    for thresh in sorted(results["threshold_impact"].keys()):
        rate = results["threshold_impact"][thresh]
        print(f"  {thresh:.2f}{'':<10} {rate*100:>6.1f}%")

    # Write output if requested
    if args.output:
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    print(f"\n{'='*70}")
    print("Current runtime setting: sparseRootSplitThreshold = 0.6")
    print(f"  At threshold=0.6: {results['threshold_impact'][0.6]*100:.1f}% expansion")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
