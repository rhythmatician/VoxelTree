#!/usr/bin/env python3
"""
WS-1.3 Shader Parity Validator
================================
Compares GPU shader density output against Java vanilla reference for one chunk.

Usage
-----
    python tools/validate_shader_parity.py \\
        --gpu  <path/to/gpu_chunk_0_0.json> \\
        --java <path/to/java_chunk_0_0.json> \\
        [--tolerance 0.01] \\
        [--plot]

Typical paths (relative to LODiffusion run/ dir):
    --gpu  parity_reports/gpu_chunk_0_0.json   (written at world load)
    --java parity_reports/java_chunk_0_0.json  (written by /dumpnoise parity 0 0)

Exit code
---------
    0  all block errors < tolerance
    1  one or more errors >= tolerance
    2  argument / file error
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Indexing helpers — must match shader density_out layout
# ---------------------------------------------------------------------------
#   flat index = (lx + 16*lz) * 384 + (by + 64)
Y_MIN    =  -64
Y_MAX    =   320
Y_LEVELS =   384   # Y_MAX - Y_MIN


def flat_index(lx: int, lz: int, by: int) -> int:
    return (lx + 16 * lz) * Y_LEVELS + (by - Y_MIN)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_density(path: Path, label: str) -> List[float]:
    if not path.exists():
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(2)

    with path.open() as f:
        data = json.load(f)

    density = data.get("density")
    if density is None or len(density) != 16 * Y_LEVELS * 16:
        print(
            f"ERROR: {label} density array has wrong length "
            f"(expected {16 * Y_LEVELS * 16}, got {len(density) if density else 'None'})",
            file=sys.stderr,
        )
        sys.exit(2)

    meta = {k: data[k] for k in ("chunk_x", "chunk_z", "source", "y_min", "y_levels")
            if k in data}
    print(f"  Loaded {label}: chunk ({meta.get('chunk_x',0)},{meta.get('chunk_z',0)})"
          f"  source={meta.get('source','?')}"
          f"  y_min={meta.get('y_min',Y_MIN)}"
          f"  elements={len(density)}")
    return density


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(gpu: List[float], java: List[float], tolerance: float):
    """Return comprehensive error statistics."""
    n = len(gpu)
    assert n == len(java), "Density arrays have different length"

    max_err  = 0.0
    sum_err  = 0.0
    fail_cnt = 0
    worst_block = (0, 0, 0, 0.0)  # lx, lz, by, error  (type: Tuple[int,int,int,float])

    # Per column statistics
    col_max_err: Dict[Tuple[int, int], float] = {}
    for lx in range(16):
        for lz in range(16):
            col_err = 0.0
            for by in range(Y_MIN, Y_MAX):
                i = flat_index(lx, lz, by)
                err = abs(gpu[i] - java[i])
                sum_err += err
                if err > col_err:
                    col_err = err
                if err > max_err:
                    max_err = err
                    worst_block = (lx, lz, by, err)
                if err >= tolerance:
                    fail_cnt += 1
            col_max_err[(lx, lz)] = col_err

    mean_err = sum_err / n
    return {
        "n": n,
        "max_err": max_err,
        "mean_err": mean_err,
        "fail_count": fail_cnt,
        "fail_fraction": fail_cnt / n,
        "worst_block": worst_block,
        "col_max_err": col_max_err,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(stats: dict, tolerance: float, verbose: bool):
    n          = stats["n"]
    max_err    = stats["max_err"]
    mean_err   = stats["mean_err"]
    fail_cnt   = stats["fail_count"]
    fail_frac  = stats["fail_fraction"]
    lx, lz, by_worst, worst_err = stats["worst_block"]

    passed = max_err < tolerance

    print()
    print("=" * 60)
    print("  WS-1.3 Shader Parity Report")
    print("=" * 60)
    print(f"  Samples compared : {n:,}")
    print(f"  Max |err|        : {max_err:.6f}  {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"  Mean |err|       : {mean_err:.6f}")
    print(f"  Fail count       : {fail_cnt:,}  ({fail_frac:.2%} of {n:,})")
    print(f"  Tolerance        : {tolerance}")
    print(f"  Worst block      : lx={lx} lz={lz} by={by_worst}  err={worst_err:.6f}")
    print("=" * 60)

    if verbose and fail_cnt > 0:
        print()
        print("  Columns with max error >= tolerance:")
        bad = [(lx, lz, e) for (lx, lz), e in stats["col_max_err"].items() if e >= tolerance]
        bad.sort(key=lambda t: -t[2])
        for lx, lz, e in bad[:20]:
            bx_abs = lx   # block coords relative to chunk origin
            bz_abs = lz
            print(f"    lx={lx:2d}  lz={lz:2d}  max_col_err={e:.6f}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")

    print()
    if passed:
        print("  RESULT: PASS — all errors below tolerance.")
    else:
        print("  RESULT: FAIL — errors exceed tolerance.")
        print("  Check: NoiseRouterExtractor named indices, xzScale=0.25, ShiftedNoise params.")
    print()


# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def plot_error_heatmap(gpu: List[float], java: List[float], by_slice: int = 63):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] numpy/matplotlib not available — skipping heatmap")
        return

    err = np.zeros((16, 16))
    for lx in range(16):
        for lz in range(16):
            i = flat_index(lx, lz, by_slice)
            err[lx, lz] = abs(gpu[i] - java[i])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(err.T, origin="lower", cmap="hot_r")
    plt.colorbar(im, ax=ax, label="|GPU - Java|")
    ax.set_xlabel("lx (chunk-local X)")
    ax.set_ylabel("lz (chunk-local Z)")
    ax.set_title(f"Density error heatmap at Y={by_slice} (sea level)")
    plt.tight_layout()
    plt.savefig("parity_error_Y63.png", dpi=120)
    print(f"  Saved heatmap to parity_error_Y63.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu",       required=True,  help="Path to gpu_chunk_*.json")
    ap.add_argument("--java",      required=True,  help="Path to java_chunk_*.json")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="Max allowed |error| per block (default 0.01)")
    ap.add_argument("--plot",      action="store_true",
                    help="Generate a Y=63 error heatmap with matplotlib")
    ap.add_argument("--verbose",   action="store_true",
                    help="List the worst offending block columns")
    args = ap.parse_args()

    print(f"\nLoading parity files …")
    gpu_d  = load_density(Path(args.gpu),  "GPU")
    java_d = load_density(Path(args.java), "Java")

    print(f"\nComparing {len(gpu_d):,} density values …")
    stats = compare(gpu_d, java_d, args.tolerance)

    print_report(stats, args.tolerance, args.verbose)

    if args.plot:
        plot_error_heatmap(gpu_d, java_d, by_slice=63)

    sys.exit(0 if stats["max_err"] < args.tolerance else 1)


if __name__ == "__main__":
    main()
