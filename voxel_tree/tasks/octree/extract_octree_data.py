#!/usr/bin/env python3
"""extract_octree_data.py — Extract multi-LOD octree training data from Voxy RocksDB.

Reads WorldSections at ALL LOD levels (0–4) from Voxy databases, converts
per-world block IDs to canonical vocabulary IDs, computes per-section biome_2d
(32×32), and saves full 32³ sections as NPZ files.

Output directory structure::

    data/voxy_octree/
        level_0/   voxy_L0_x{X}_y{Y}_z{Z}.npz
        level_1/   voxy_L1_x{X}_y{Y}_z{Z}.npz
        level_2/   voxy_L2_x{X}_y{Y}_z{Z}.npz
        level_3/   voxy_L3_x{X}_y{Y}_z{Z}.npz
        level_4/   voxy_L4_x{X}_y{Y}_z{Z}.npz

Each NPZ contains::

    labels32             (32, 32, 32) int32  — canonical Voxy vocabulary IDs
    biome32              (32, 32)     int32  — biome IDs (column-wise majority)
    level                scalar       int64  — LOD level
    section_x            scalar       int64  — section X coordinate
    section_y            scalar       int64  — section Y coordinate
    section_z            scalar       int64  — section Z coordinate
    non_empty_children   scalar       uint8  — 8-bit occupancy mask from Voxy

Usage::

    python scripts/extract_octree_data.py \\
        "path/to/voxy/<hash>/storage" \\
        --output-dir data/voxy_octree \\
        --min-solid 0.02

    # Multiple databases:
    python scripts/extract_octree_data.py db1/storage db2/storage \\
        --output-dir data/voxy_octree
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from voxel_tree.tasks.voxy_reader import VoxyReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LOD_LEVEL = 4
_PKG_DIR = Path(__file__).resolve().parent.parent.parent  # VoxelTree/VoxelTree/ (the package)
DEFAULT_VOCAB_PATH = _PKG_DIR / "config" / "voxy_vocab.json"
DEFAULT_OUTPUT_DIR = Path("data/voxy_octree")

# ---------------------------------------------------------------------------
# Vocabulary helpers (shared with extract_voxy_training_data.py)
# ---------------------------------------------------------------------------


def build_voxy_vocab_from_worlds(db_paths: list[str], out_path: Path) -> dict[str, int]:
    """Scan Voxy worlds and build a canonical vocabulary from all unique block names."""
    all_names: set[str] = set()
    for p in db_paths:
        try:
            with VoxyReader(p) as r:
                all_names.update(r._block_names.values())
        except Exception as exc:
            print(f"  Warning: could not read {p}: {exc}")

    all_names.discard("minecraft:air")
    sorted_names = sorted(all_names)

    vocab: dict[str, int] = {"minecraft:air": 0}
    for i, name in enumerate(sorted_names, start=1):
        vocab[name] = i

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    print(f"Built Voxy vocabulary: {len(vocab)} entries → {out_path}")
    return vocab


def load_vocab(path: Path) -> dict[str, int]:
    """Load a canonical Voxy vocabulary from JSON."""
    with open(path) as f:
        return json.load(f)


def build_world_lut(reader: VoxyReader, vocab: dict[str, int]) -> npt.NDArray[np.int32]:
    """Build a per-world LUT mapping Voxy state IDs → canonical vocab IDs."""
    max_id = max(reader._block_names.keys()) if reader._block_names else 0
    lut = np.zeros(max_id + 1, dtype=np.int32)  # unmapped → 0 (air)

    mapped = 0
    for voxy_id, name in reader._block_names.items():
        if name in vocab:
            lut[voxy_id] = vocab[name]
            mapped += 1
        else:
            base = name.split("[")[0] if "[" in name else name
            lut[voxy_id] = vocab.get(base, 0)
            if base in vocab:
                mapped += 1

    return lut


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_biome_2d(biome_ids: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Collapse 3-D biome volume (y, z, x) → 2-D (z, x) via column majority.

    Works for any 32×32×32 volume regardless of LOD level.
    Returns (32, 32) int32 in (z, x) order.
    """
    Y, Z, X = biome_ids.shape
    cols = biome_ids.reshape(Y, -1)  # (Y, Z*X)

    def _col_mode(col: npt.NDArray[np.int32]) -> int:
        offset = int(col.min())
        shifted = col - offset if offset < 0 else col
        counts = np.bincount(shifted)
        return int(np.argmax(counts)) + offset

    modes_flat = np.apply_along_axis(_col_mode, axis=0, arr=cols)
    return modes_flat.reshape(Z, X).astype(np.int32)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_all_levels(
    db_path: str,
    output_dir: Path,
    vocab: dict[str, int],
    *,
    min_solid_fraction: float = 0.02,
    max_sections_per_level: int | None = None,
    prefix: str = "",
) -> dict[int, dict[str, int]]:
    """Extract training NPZ files from a Voxy RocksDB database at all LOD levels.

    Returns a dict mapping level → {"saved": N, "skipped": M, "total": T}.
    """
    stats: dict[int, dict[str, int]] = {}

    with VoxyReader(db_path) as reader:
        print(reader.summary())
        print()

        # Build per-world LUT: Voxy state ID → canonical vocab ID
        lut = build_world_lut(reader, vocab)
        n_mapped = int(np.sum(lut > 0))
        print(
            f"Block ID mapping: {n_mapped}/{len(reader._block_names)} "
            f"Voxy states → canonical vocab ({len(vocab)} entries)"
        )
        print()

        # Count available levels
        level_counts = reader.count_sections()
        available_levels = sorted(level_counts.keys())
        print(f"Available LOD levels: {available_levels}")
        for lvl in available_levels:
            print(f"  L{lvl}: {level_counts[lvl]:,} sections")
        print()

        # Global progress tracking (for GUI progress ring)
        _cap = max_sections_per_level
        total_expected = sum(
            min(level_counts.get(lvl, 0), _cap if _cap is not None else level_counts.get(lvl, 0))
            for lvl in range(MAX_LOD_LEVEL + 1)
        )
        global_processed = 0
        print_interval = max(1, total_expected // 20) if total_expected > 0 else 100

        # Extract each level
        for level in range(MAX_LOD_LEVEL + 1):
            if level not in level_counts:
                print(f"Skipping L{level} — no sections in database")
                stats[level] = {"saved": 0, "skipped": 0, "total": 0}
                continue

            level_dir = output_dir / f"level_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            skipped_empty = 0
            section_count = 0
            t0 = time.time()

            print(f"{'='*60}")
            print(f"  Extracting LOD {level}  ({level_counts[level]:,} sections)")
            print(f"  Output: {level_dir}")
            print(f"{'='*60}")

            for section in reader.iter_sections(level=level):
                if max_sections_per_level is not None and section_count >= max_sections_per_level:
                    break
                section_count += 1
                global_processed += 1

                block_ids = section.block_ids  # (32,32,32) (y,z,x)
                biome_ids = section.biome_ids  # (32,32,32)
                sx, sy, sz = section.x, section.y, section.z
                nec = section.non_empty_children

                # Map to canonical vocabulary
                max_id = lut.shape[0] - 1
                block_canon = lut[np.clip(block_ids, 0, max_id)]

                # Apply min-solid filter
                solid_frac = float(np.mean(block_canon > 0))
                if solid_frac < min_solid_fraction:
                    skipped_empty += 1
                    continue

                # Compute 2D biome map
                biome_2d = compute_biome_2d(biome_ids)

                # Save NPZ
                fname = f"{prefix}voxy_L{level}_x{sx}_y{sy}_z{sz}.npz"
                np.savez_compressed(
                    level_dir / fname,
                    labels32=block_canon.astype(np.int32),
                    biome32=biome_2d.astype(np.int32),
                    level=np.int64(level),
                    section_x=np.int64(sx),
                    section_y=np.int64(sy),
                    section_z=np.int64(sz),
                    non_empty_children=np.uint8(nec),
                )
                saved += 1

                if section_count % print_interval == 0:
                    elapsed = time.time() - t0
                    rate = section_count / elapsed if elapsed > 0 else 0
                    pct = global_processed * 100 // total_expected if total_expected > 0 else 0
                    print(
                        f"  ... L{level}: {section_count:,} processed, "
                        f"{saved:,} saved, {skipped_empty:,} skipped "
                        f"({rate:.0f}/s) — {pct}%"
                    )

            elapsed = time.time() - t0
            stats[level] = {
                "saved": saved,
                "skipped": skipped_empty,
                "total": section_count,
            }
            pct = global_processed * 100 // total_expected if total_expected > 0 else 0
            print(
                f"  L{level} complete: {saved:,} saved, {skipped_empty:,} skipped "
                f"in {elapsed:.1f}s  ({section_count:,} total) — {pct}%"
            )
            print()

    return stats


# ---------------------------------------------------------------------------
# Progress marker
# ---------------------------------------------------------------------------

MARKER_NAME = ".extract_octree_done"


def write_marker(output_dir: Path, stats: dict[int, dict[str, int]]) -> None:
    """Write a progress marker so downstream steps can verify extraction ran."""
    import json as _json

    marker = output_dir / MARKER_NAME
    marker.write_text(
        _json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "levels": {str(k): v for k, v in stats.items()},
            },
            indent=2,
        )
    )


def check_marker(output_dir: Path) -> bool:
    """Check whether extraction has completed previously."""
    return (output_dir / MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract multi-LOD octree training data from Voxy RocksDB.",
    )
    parser.add_argument(
        "db_path",
        nargs="+",
        help="Path(s) to Voxy storage directories (contains .sst files).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for NPZ files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Path to Voxy vocabulary JSON (default: config/voxy_vocab.json).",
    )
    parser.add_argument(
        "--min-solid",
        type=float,
        default=0.02,
        help="Minimum solid fraction to keep a section (default: 0.02).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Max sections per level per database (for testing).",
    )
    args = parser.parse_args(argv)

    # Load or build vocabulary
    if args.vocab.exists():
        vocab = load_vocab(args.vocab)
        print(f"Loaded Voxy vocabulary: {len(vocab)} entries from {args.vocab}")
    else:
        print(f"Vocabulary not found at {args.vocab} — building from input databases...")
        vocab = build_voxy_vocab_from_worlds(args.db_path, args.vocab)

    # Extract all databases
    all_stats: dict[int, dict[str, int]] = {}
    for i, db_path in enumerate(args.db_path):
        prefix = f"w{i}_" if len(args.db_path) > 1 else ""
        print("=" * 70)
        print(f"Database {i+1}/{len(args.db_path)}: {db_path}")
        print("=" * 70)
        db_stats = extract_all_levels(
            db_path=db_path,
            output_dir=args.output_dir,
            vocab=vocab,
            min_solid_fraction=args.min_solid,
            max_sections_per_level=args.max_sections,
            prefix=prefix,
        )
        # Merge stats
        for lvl, s in db_stats.items():
            if lvl not in all_stats:
                all_stats[lvl] = {"saved": 0, "skipped": 0, "total": 0}
            for k in ("saved", "skipped", "total"):
                all_stats[lvl][k] += s[k]
        print()

    # Summary
    print("=" * 70)
    print("  EXTRACTION SUMMARY")
    print("=" * 70)
    grand_total = 0
    for lvl in sorted(all_stats.keys()):
        s = all_stats[lvl]
        grand_total += s["saved"]
        print(
            f"  L{lvl}: {s['saved']:>8,} saved, {s['skipped']:>8,} skipped ({s['total']:,} total)"
        )
    print(f"  {'─'*50}")
    print(f"  Total saved: {grand_total:,}")
    print(f"  Output: {args.output_dir}")
    print()

    # Write marker
    write_marker(args.output_dir, all_stats)
    print(f"  Progress marker written: {args.output_dir / MARKER_NAME}")


if __name__ == "__main__":
    main()
