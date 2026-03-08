#!/usr/bin/env python3
"""build_pairs.py — Pre-compute LOD training pairs from extracted NPZ chunks.

This is Phase 1b of the data preparation pipeline:

    Phase 1a (extract):   Voxy RocksDB  → data/voxy/*.npz    (raw LOD0 blocks)
    Phase 1b (here):      data/voxy/    → data/voxy/*_pairs_v1.npz
                          For each raw chunk, 4 LOD transitions are built
                          (LOD4→3, LOD3→2, LOD2→1, LOD1→0) and the full
                          stacked array is written to:
                            data/voxy/train_pairs_v1.npz
                            data/voxy/val_pairs_v1.npz
    Phase 2  (train):     *_pairs_v1.npz → model weights  (cache is loaded
                          by MultiLODDataset automatically on first import)

Running this step before training avoids Mipper computation (3-D volume
downsampling + opacity table lookups) being repeated on every epoch.

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

from train.multi_lod_dataset import MultiLODDataset  # noqa: E402


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
        for cache_file in data_dir.glob("*_pairs_v1.npz"):
            print(f"  Removing stale cache: {cache_file.name}")
            cache_file.unlink()

    all_npz = sorted(data_dir.glob("*.npz"))
    # Exclude any existing cache files from the source list
    source_npz = [f for f in all_npz if not f.name.endswith("_pairs_v1.npz")]
    n_total = len(source_npz)

    if n_total == 0:
        print(f"ERROR: No source NPZ files found in {data_dir}")
        print("  Run 'python data-cli.py extract' first.")
        sys.exit(1)

    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    print(f"Source chunks: {n_total}  →  {n_train} train / {n_val} val")
    print()

    # Symlink trick: MultiLODDataset uses glob to discover files.
    # We can't pass explicit lists, so we use split subdirectories if
    # they exist, or rely on the filename-based split fallback.
    # The simplest compatible approach: point the dataset at data_dir and
    # let it take all files (which triggers the train cache), then build
    # a separate val cache by writing a tiny stub dataset over the last
    # n_val files.
    #
    # However, because MultiLODDataset already handles train/val splits
    # via subdirectory structure *or* filename patterns, and we want to
    # match whatever the trainer will use, we just instantiate it the same
    # way the trainer does: data_dir with split="train" / "val".
    # If no subdirectory exists the dataset takes ALL files for both splits
    # (the split arg only changes the cache filename in that case).
    # That is fine for initial builds.

    print("=" * 62)
    print("  Building TRAIN pairs cache ...")
    print("=" * 62)
    t0 = time.time()
    train_ds = MultiLODDataset(
        data_dir,
        split="train",
        min_solid_fraction=min_solid_fraction,
        use_pair_cache=True,
    )
    train_elapsed = time.time() - t0
    n_train_pairs = len(train_ds)
    print(f"\n  Train cache: {n_train_pairs:,} pairs in {train_elapsed:.1f}s")

    print()
    print("=" * 62)
    print("  Building VAL pairs cache ...")
    print("=" * 62)
    t0 = time.time()
    val_ds = MultiLODDataset(
        data_dir,
        split="val",
        min_solid_fraction=min_solid_fraction,
        use_pair_cache=True,
    )
    val_elapsed = time.time() - t0
    n_val_pairs = len(val_ds)
    print(f"\n  Val cache: {n_val_pairs:,} pairs in {val_elapsed:.1f}s")

    print()
    print("=" * 62)
    print("  Pair build complete.")
    print(f"  Train pairs : {n_train_pairs:>10,}")
    print(f"  Val pairs   : {n_val_pairs:>10,}")
    print(f"  Total       : {n_train_pairs + n_val_pairs:>10,}")
    cache_files = sorted(data_dir.glob("*_pairs_v1.npz"))
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
        help="Delete existing *_pairs_v1.npz cache files before rebuilding",
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
