#!/usr/bin/env python3
"""Convert octree ONNX models from FP32 to FP16.

FP16 halves the model size and memory bandwidth requirements.  On the Intel
UHD 770 (Xe LP) via DirectML, expect 1.2–1.6× inference speedup without any
retraining, calibration data, or Java-side changes (I/O stays FP32).

Usage:
    python scripts/octree/convert_fp16.py                          # default: production/
    python scripts/octree/convert_fp16.py --model-dir runs/v42     # custom dir
    python scripts/octree/convert_fp16.py --validate               # also run shape check

Requirements:
    pip install onnx onnxconverter-common
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16

STEMS = ["octree_init", "octree_refine", "octree_leaf"]


def convert_model(input_path: Path, output_path: Path) -> None:
    """Convert a single ONNX model to FP16, keeping I/O in FP32."""
    model = onnx.load(str(input_path))

    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=True,  # inputs/outputs stay FP32 → no Java changes
        disable_shape_infer=False,
    )

    onnx.save(model_fp16, str(output_path))

    in_mb = input_path.stat().st_size / 1e6
    out_mb = output_path.stat().st_size / 1e6
    ratio = out_mb / in_mb
    print(f"  {input_path.name}: {in_mb:.2f} MB → {out_mb:.2f} MB ({ratio:.1%})")


def validate_model(path: Path) -> bool:
    """Check that the converted model loads and has valid structure."""
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        # Verify inputs are still FP32
        for inp in model.graph.input:
            elem_type = inp.type.tensor_type.elem_type
            if elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
                print(f"  ⚠ {path.name} input '{inp.name}' has unexpected type {elem_type}")
                return False
        print(f"  ✓ {path.name} — valid, inputs are FP32")
        return True
    except Exception as e:
        print(f"  ✗ {path.name} — validation failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX models to FP16")
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
        help="Output directory (default: <model-dir>/fp16/)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run onnx.checker on the converted models",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    out_dir: Path = args.output_dir or model_dir / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting models in {model_dir} → {out_dir}\n")

    missing = [s for s in STEMS if not (model_dir / f"{s}.onnx").exists()]
    if missing:
        print(f"ERROR: Missing model files: {missing}", file=sys.stderr)
        sys.exit(1)

    for stem in STEMS:
        input_path = model_dir / f"{stem}.onnx"
        output_path = out_dir / f"{stem}_fp16.onnx"
        convert_model(input_path, output_path)

    if args.validate:
        print("\nValidating converted models:")
        all_ok = all(validate_model(out_dir / f"{stem}_fp16.onnx") for stem in STEMS)
        if not all_ok:
            print("\n⚠ Some models failed validation — check output above.")
            sys.exit(1)

    print("\nDone!  To use the FP16 models, copy them into your LODiffusion")
    print("model directory (renaming to drop the _fp16 suffix) or update")
    print("the model stems in OctreeModelRunner.")


if __name__ == "__main__":
    main()
