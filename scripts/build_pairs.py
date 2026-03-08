#!/usr/bin/env python3
"""build_pairs.py — Pre-compute LOD training pairs from extracted NPZ chunks.

This is Phase 1b of the data preparation pipeline:

    Phase 1a (extract):   Voxy RocksDB  → data/voxy/*.npz    (raw LOD0 blocks)
    Phase 1b (here):      data/voxy/    → data/voxy/*_pairs_v2.npz
                          For each raw chunk, 4 LOD transitions are built
                          (init→LOD4, LOD4→3, LOD3→2, LOD2→1) and the full
                          stacked array is written to:
                            data/voxy/train_pairs_v2.npz
                            data/voxy/val_pairs_v2.npz
    Phase 2  (train):     *_pairs_v2.npz → model weights  (cache is loaded
                          by MultiLODDataset; training REQUIRES the cache)

NPZ files must contain at minimum: labels16, biome_patch, heightmap_patch,
and y_index.  Router6 noise data is no longer required (biome + heightmap
already encode the information router6 provided).

Running this step before training is MANDATORY.  The training script will
refuse to start without a pre-built pair cache.

Usage
-----
    python scripts/build_pairs.py --data-dir data/voxy
    python scripts/build_pairs.py --data-dir data/voxy --clean    # force rebuild
    python scripts/build_pairs.py --data-dir data/voxy --val-split 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without installing VoxelTree as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.multi_lod_dataset import (  # noqa: E402
    generate_pairs_from_npz_files,
    save_pairs_cache,
)


def _discover_npz_files(data_dir: Path, split: str) -> list[Path]:
    """Find source NPZ files for the given split, matching dataset conventions."""
    split_subdir = data_dir / split
    if split_subdir.is_dir():
        files = sorted(split_subdir.glob("*.npz"))
    else:
        files = sorted(data_dir.glob(f"*_{split}_*.npz"))
        if not files:
            # Fall back to all non-cache NPZ files in data_dir
            files = sorted(
                f
                for f in data_dir.glob("*.npz")
                if not f.name.endswith(("_pairs_v1.npz", "_pairs_v2.npz"))
            )
    return files


def build(
    data_dir: Path,
    *,
    val_split: float = 0.1,
    min_solid_fraction: float = 0.02,
    clean: bool = False,
) -> tuple[int, int]:
    """Build pre-computed pair caches for train and val splits.

    Returns (n_train_pairs, n_val_pairs).
    """
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # Respect clean flag by deleting existing cache files
    if clean:
        for cache_file in list(data_dir.glob("*_pairs_v2.npz")) + list(
            data_dir.glob("*_pairs_v1.npz")
        ):
            print(f"  Removing stale cache: {cache_file.name}")
            cache_file.unlink()

    # Discover source files for each split
    train_files = _discover_npz_files(data_dir, "train")
    val_files = _discover_npz_files(data_dir, "val")

    n_total = len(train_files)
    if n_total == 0:
        print(f"ERROR: No source NPZ files found in {data_dir}")
        print("  Run 'python data-cli.py extract' first.")
        sys.exit(1)

    print(f"Source chunks: train={len(train_files)}, val={len(val_files)}")
    print()

    # ---- TRAIN ----
    print("=" * 62)
    print("  Building TRAIN pairs cache ...")
    print("=" * 62)
    t0 = time.time()
    train_pairs = generate_pairs_from_npz_files(train_files, min_solid_fraction=min_solid_fraction)
    cache_path = data_dir / "train_pairs_v2.npz"
    if train_pairs:
        save_pairs_cache(train_pairs, cache_path)
    else:
        print("  WARNING: No train pairs generated!")
        print("  Most likely cause: NPZ files are missing required keys")
        print("  (labels16, biome_patch, heightmap_patch, y_index).")
        print("  Run 'python data-cli.py extract' first.")
    train_elapsed = time.time() - t0
    n_train_pairs = len(train_pairs)
    print(f"\n  Train cache: {n_train_pairs:,} pairs in {train_elapsed:.1f}s")

    # ---- VAL ----
    print()
    print("=" * 62)
    print("  Building VAL pairs cache ...")
    print("=" * 62)
    t0 = time.time()
    val_pairs = generate_pairs_from_npz_files(val_files, min_solid_fraction=min_solid_fraction)
    cache_path = data_dir / "val_pairs_v2.npz"
    if val_pairs:
        save_pairs_cache(val_pairs, cache_path)
    else:
        print("  WARNING: No val pairs generated!")
    val_elapsed = time.time() - t0
    n_val_pairs = len(val_pairs)
    print(f"\n  Val cache: {n_val_pairs:,} pairs in {val_elapsed:.1f}s")

    # ---- Summary ----
    print()
    print("=" * 62)
    print("  Pair build complete.")
    print(f"  Train pairs : {n_train_pairs:>10,}")
    print(f"  Val pairs   : {n_val_pairs:>10,}")
    print(f"  Total       : {n_train_pairs + n_val_pairs:>10,}")
    cache_files = sorted(data_dir.glob("*_pairs_v2.npz"))
    total_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    print(f"  Cache size  : {total_mb:.1f} MB  ({len(cache_files)} file(s))")
    print("=" * 62)

    return n_train_pairs, n_val_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute LOD training pairs from extracted NPZ chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/voxy"),
        metavar="DIR",
        help="Directory containing source NPZ files (default: data/voxy)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing *_pairs_v2.npz (and v1) cache files before rebuilding",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of chunks to reserve for validation (default: 0.1)",
    )
    parser.add_argument(
        "--min-solid",
        type=float,
        default=0.02,
        metavar="FRAC",
        help="Skip chunks with solid fraction below this threshold (default: 0.02)",
    )
    args = parser.parse_args()

    build(
        args.data_dir,
        val_split=args.val_split,
        min_solid_fraction=args.min_solid,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
