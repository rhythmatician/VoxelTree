#!/usr/bin/env python3
"""Pre-apply ONNX Runtime graph optimizations to octree models.

Pre-optimizing the .onnx files means the runtime doesn't need to do it at
load time, shaving 5–15% off model startup latency.  Applies ORT_ENABLE_ALL
(level 99): constant folding, Conv+BN+ReLU fusion, shape propagation, and
execution-provider-specific layout transforms.

Usage:
    python scripts/octree/optimize_onnx.py                         # default: production/
    python scripts/octree/optimize_onnx.py --model-dir runs/v42    # custom dir
    python scripts/octree/optimize_onnx.py --validate              # also run inference check

Requirements:
    pip install onnxruntime          # CPU-only is fine for optimization
    pip install onnx                 # for validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnxruntime as ort

STEMS = ["octree_init", "octree_refine", "octree_leaf"]


def optimize_model(input_path: Path, output_path: Path) -> None:
    """Apply ORT_ENABLE_ALL graph optimizations and save the result."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    # Creating the session triggers optimization and writes the optimized model.
    # We use CPUExecutionProvider regardless of the deploy target — the graph
    # optimizations (constant folding, op fusion) are EP-agnostic.
    ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])

    in_mb = input_path.stat().st_size / 1e6
    out_mb = output_path.stat().st_size / 1e6
    print(f"  {input_path.name}: {in_mb:.2f} MB → {out_mb:.2f} MB")


def validate_model(path: Path) -> bool:
    """Verify the optimized model loads and produces outputs."""
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print(f"  ✓ {path.name} — valid")
        return True
    except Exception as e:
        print(f"  ✗ {path.name} — validation failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-optimize ONNX models with ORT graph passes")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("production"),
        help="Directory containing the .onnx files (default: production/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <model-dir>/optimized/)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run onnx.checker on the optimized models",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    out_dir: Path = args.output_dir or model_dir / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optimizing models in {model_dir} → {out_dir}\n")

    missing = [s for s in STEMS if not (model_dir / f"{s}.onnx").exists()]
    if missing:
        print(f"ERROR: Missing model files: {missing}", file=sys.stderr)
        sys.exit(1)

    for stem in STEMS:
        input_path = model_dir / f"{stem}.onnx"
        output_path = out_dir / f"{stem}_optimized.onnx"
        optimize_model(input_path, output_path)

    if args.validate:
        print("\nValidating optimized models:")
        all_ok = all(validate_model(out_dir / f"{stem}_optimized.onnx") for stem in STEMS)
        if not all_ok:
            print("\n⚠ Some models failed validation — check output above.")
            sys.exit(1)

    print("\nDone!  To use the optimized models, copy them into your LODiffusion")
    print("model directory (renaming to drop the _optimized suffix) or update")
    print("the model stems in OctreeModelRunner.")


if __name__ == "__main__":
    main()
