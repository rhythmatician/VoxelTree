#!/usr/bin/env python3
"""Compute per-class frequency weights for block-type loss.

Reads the pre-built pair-cache files (*_pairs_v1.npz) and counts how often
each block type appears across ALL voxels (air included).  Applies
median-frequency balancing:

    w_c = median(freq_k | k seen) / freq_c     for  c seen
    w_c = 0                                     for  c unseen

Air (class 0) is included with its natural frequency — since ~75% of voxels
are air, its weight will be very low relative to solid blocks.

Weights are clipped to [0, max_weight] to prevent extreme values from very rare
blocks destabilising training, then saved as ``class_weights.npz`` next to the
pair-cache files.

Usage (standalone):
    python scripts/compute_class_weights.py --data-dir data/voxy_subset
    python scripts/compute_class_weights.py --data-dir data/voxy --max-weight 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

# Repo-root on path so imports work when called from anywhere
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def compute_weights(
    data_dir: Path,
    vocab_size: int,
    max_weight: float = 20.0,
    verbose: bool = True,
) -> npt.NDArray[np.float32]:
    """Scan pair-cache files and return per-class weights [vocab_size] float32.

    Args:
        data_dir:    Directory containing ``train_pairs_v1.npz``.
        vocab_size:  Number of block types (length of voxy_vocab.json).
        max_weight:  Hard cap on individual class weights.
        verbose:     Print progress and top/bottom class stats.

    Returns:
        weights: float32 array of shape [vocab_size] — weight for each block ID.
    """
    cache_file = data_dir / "train_pairs_v1.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Pair cache not found: {cache_file}\n" "Run 'python data-cli.py build-pairs' first."
        )

    if verbose:
        print(f"Loading pair cache: {cache_file}")

    data = np.load(cache_file)
    target_types = data["target_types"]  # (N, 16, 16, 16) int32

    if verbose:
        print(f"  Pairs: {target_types.shape[0]:,} — counting block frequencies…")

    # Count ALL voxels (air included) per class
    flat_types = target_types.reshape(-1).astype(np.int64)
    flat_types = np.clip(flat_types, 0, vocab_size - 1)

    counts = np.zeros(vocab_size, dtype=np.int64)
    np.add.at(counts, flat_types, 1)

    total_voxels = int(counts.sum())
    n_seen = int(np.sum(counts > 0))

    if verbose:
        air_pct = 100.0 * counts[0] / total_voxels
        print(f"  Total voxels scanned: {total_voxels:,}  (air: {air_pct:.1f}%)")
        print(f"  Classes seen (non-zero count): {n_seen} / {vocab_size}")

    # Median-frequency balancing (all seen classes, air included)
    seen_counts = counts[counts > 0]
    median_freq = float(np.median(seen_counts))

    weights = np.zeros(vocab_size, dtype=np.float32)
    for c in range(vocab_size):  # air (c=0) now participates
        if counts[c] > 0:
            weights[c] = float(median_freq / counts[c])

    # Clip extreme values (paranoid safety for 1-voxel-seen blocks)
    weights = np.clip(weights, 0.0, max_weight)

    if verbose:
        # Top-10 most up-weighted (rarest seen classes)
        seen_mask = counts > 0

        class_ids = np.where(seen_mask)[0]
        by_weight = class_ids[np.argsort(weights[class_ids])[::-1]]
        by_freq = class_ids[np.argsort(counts[class_ids])[::-1]]

        print("\n  Most up-weighted (rarest blocks):")
        for idx in by_weight[:10]:
            pct = 100.0 * counts[idx] / total_voxels
            print(
                f"    id={idx:4d}  weight={weights[idx]:.3f}  "
                f"count={counts[idx]:8,}  ({pct:.4f}%)"
            )

        print("\n  Least up-weighted (most common blocks):")
        for idx in by_freq[:10]:
            pct = 100.0 * counts[idx] / total_voxels
            print(
                f"    id={idx:4d}  weight={weights[idx]:.3f}  "
                f"count={counts[idx]:8,}  ({pct:.3f}%)"
            )

        print(f"\n  Median weight (seen classes): {np.median(weights[seen_mask]):.3f}")
        print(f"  Max weight after clip: {weights.max():.3f}")
        print(f"  Air (class 0) weight: {weights[0]:.6f}")

    return weights


def save_weights(weights: npt.NDArray[np.float32], output_path: Path) -> None:
    np.savez_compressed(output_path, class_weights=weights)
    print(f"\nSaved class weights → {output_path}  ({len(weights)} entries)")


def load_weights(path: Path) -> npt.NDArray[np.float32]:
    """Load class weights from an npz saved by this script."""
    d = np.load(path)
    return d["class_weights"].astype(np.float32)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute per-class block-type weights for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train_pairs_v1.npz",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=_REPO_ROOT / "config" / "voxy_vocab.json",
        help="Path to voxy_vocab.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for class_weights.npz (default: <data-dir>/class_weights.npz)",
    )
    p.add_argument(
        "--max-weight",
        type=float,
        default=20.0,
        help="Hard cap on individual class weight",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)

    vocab_path = args.vocab
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"Vocab: {vocab_size} entries from {vocab_path}")
    else:
        vocab_size = 1102
        print(f"Warning: vocab not found at {vocab_path}, using fallback size {vocab_size}")

    weights = compute_weights(
        data_dir=args.data_dir,
        vocab_size=vocab_size,
        max_weight=args.max_weight,
        verbose=True,
    )

    out = args.output or (args.data_dir / "class_weights.npz")
    save_weights(weights, out)


if __name__ == "__main__":
    main()
