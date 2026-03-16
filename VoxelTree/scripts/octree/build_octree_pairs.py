#!/usr/bin/env python3
"""build_octree_pairs.py — Build parent/child octree training pairs.

Takes extracted NPZ files from ``extract_octree_data.py`` (per-level directories)
and builds training pairs for the 3-model octree architecture:

  Model A ``octree_init`` (L4): no parent → block_logits + occ_logits[8]
  Model B ``octree_refine`` (L3–L1 shared): parent_context + level → block_logits + occ_logits[8]
  Model C ``octree_leaf`` (L0): parent_context → block_logits only (no occ)

Parent→Child coordinate mapping::

    childX = (parentX << 1) + (octant & 1)
    childY = (parentY << 1) + ((octant >> 2) & 1)
    childZ = (parentZ << 1) + ((octant >> 1) & 1)

Octant index: ``(x&1) | ((z&1)<<1) | ((y&1)<<2)``

Parent context encoding: extract relevant 16³ octant from parent's 32³ grid,
upsample 2× via nearest-neighbor → 32³ parent context.

Output format (stacked NPZ cache per train/val split)::

    labels32             (N, 32, 32, 32) int32
    parent_labels32      (N, 32, 32, 32) int32   (zeros for L4 init)
    heightmap32          (N, 5, 32, 32)  float32
    biome32              (N, 32, 32)     int32
    y_position           (N,)            int64
    level                (N,)            int64
    non_empty_children   (N,)            uint8

Usage::

    python scripts/build_octree_pairs.py \\
        --data-dir data/voxy_octree \\
        --output-dir data/voxy_octree

    python scripts/build_octree_pairs.py \\
        --data-dir data/voxy_octree --val-split 0.1 --clean
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from VoxelTree.utils.progress import report as _report_progress

# Ensure project root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LOD_LEVEL = 4
HEIGHTMAP_PLANES = 5  # surface, ocean_floor, slope_x, slope_z, curvature

# Default placeholder heightmap (will be filled by column-heights step)
_EMPTY_HEIGHTMAP = np.zeros((HEIGHTMAP_PLANES, 32, 32), dtype=np.float32)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def child_coords_from_parent(px: int, py: int, pz: int, octant: int) -> tuple[int, int, int]:
    """Compute child section coordinates from parent coords and octant index.

    Octant bits: bit0 = x, bit1 = z, bit2 = y
    """
    cx = (px << 1) + (octant & 1)
    cy = (py << 1) + ((octant >> 2) & 1)
    cz = (pz << 1) + ((octant >> 1) & 1)
    return cx, cy, cz


def parent_coords_and_octant(cx: int, cy: int, cz: int) -> tuple[int, int, int, int]:
    """Compute parent section coordinates and octant index from child coords.

    Returns (parent_x, parent_y, parent_z, octant).
    """
    px = cx >> 1
    py = cy >> 1
    pz = cz >> 1
    octant = (cx & 1) | ((cz & 1) << 1) | ((cy & 1) << 2)
    return px, py, pz, octant


def extract_octant(
    parent_labels: npt.NDArray[np.int32],
    octant: int,
) -> npt.NDArray[np.int32]:
    """Extract the 16³ octant from a parent's 32³ grid at native Voxy resolution.

    The parent_labels array is (32, 32, 32) in (y, z, x) order.
    Octant bits: bit0 = x, bit1 = z, bit2 = y

    Returns (16, 16, 16) int32 — the octant voxels exactly as Voxy stores them,
    with no upsampling applied.  This is the authoritative comparison target
    for the hierarchical consistency loss.
    """
    dx = octant & 1
    dz = (octant >> 1) & 1
    dy = (octant >> 2) & 1

    y_slice = slice(dy * 16, (dy + 1) * 16)
    z_slice = slice(dz * 16, (dz + 1) * 16)
    x_slice = slice(dx * 16, (dx + 1) * 16)

    return parent_labels[y_slice, z_slice, x_slice].astype(np.int32)


def extract_octant_and_upsample(
    parent_labels: npt.NDArray[np.int32],
    octant: int,
) -> npt.NDArray[np.int32]:
    """Extract a 16³ octant from parent's 32³ grid and upsample 2× to 32³.

    The parent_labels array is (32, 32, 32) in (y, z, x) order.
    Octant bits: bit0 = x, bit1 = z, bit2 = y

    Returns (32, 32, 32) int32 — nearest-neighbor upsampled parent context
    suitable for use as the model's 32³ input feature.
    """
    sub = extract_octant(parent_labels, octant)  # (16, 16, 16)
    # Nearest-neighbor upsample 2×: repeat each voxel in all 3 dimensions
    upsampled = np.repeat(np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1), 2, axis=2)
    return upsampled.astype(np.int32)


# ---------------------------------------------------------------------------
# Section index builder
# ---------------------------------------------------------------------------


def build_section_index(
    data_dir: Path,
    level: int,
) -> dict[tuple[int, int, int], Path]:
    """Build a coordinate → file path index for a given LOD level.

    Scans ``data_dir/level_{level}/`` for NPZ files matching the naming
    convention ``voxy_L{level}_x{X}_y{Y}_z{Z}.npz``.

    Returns dict mapping (x, y, z) → Path.
    """
    import re

    level_dir = data_dir / f"level_{level}"
    if not level_dir.is_dir():
        return {}

    index: dict[tuple[int, int, int], Path] = {}
    pattern = re.compile(r"voxy_L\d+_x(-?\d+)_y(-?\d+)_z(-?\d+)\.npz$")

    for f in level_dir.iterdir():
        if not f.suffix == ".npz":
            continue
        m = pattern.search(f.name)
        if m:
            x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
            index[(x, y, z)] = f

    return index


def load_section_npz(path: Path) -> dict[str, Any]:
    """Load an NPZ section file and return a dict of arrays."""
    data = np.load(path)
    result = {k: data[k] for k in data.files}
    data.close()
    return result


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------


def compute_true_non_empty_children(
    parent_coords: tuple[int, int, int],
    child_level: int,
    child_index: dict[tuple[int, int, int], Path],
) -> np.uint8:
    """Compute ground-truth nonEmptyChildren bitmask for a parent section.

    Checks which of the 8 child octants actually exist in the child level's index.
    """
    px, py, pz = parent_coords
    mask = np.uint8(0)
    for octant in range(8):
        cx, cy, cz = child_coords_from_parent(px, py, pz, octant)
        if (cx, cy, cz) in child_index:
            mask |= np.uint8(1 << octant)
    return mask


def build_pairs_for_level(
    child_level: int,
    data_dir: Path,
    child_index: dict[tuple[int, int, int], Path],
    parent_index: dict[tuple[int, int, int], Path],
) -> list[dict[str, Any]]:
    """Build training pairs for a single child level.

    For each child section, finds its parent at child_level+1, extracts
    the parent context, and assembles the training pair.

    For L4 (init model), parent_labels32 is zeros.

    Returns a list of pair dicts.
    """
    parent_level = child_level + 1
    is_init = parent_level > MAX_LOD_LEVEL  # L4 children have no parent

    pairs: list[dict[str, Any]] = []
    skipped_no_parent = 0

    items = sorted(child_index.items())
    desc = f"L{child_level}" + (f"<-L{parent_level}" if not is_init else " (init)")

    total_items = len(items)
    for idx, ((cx, cy, cz), child_path) in enumerate(
        tqdm(items, desc=f"  Pairs {desc}", unit="sect")
    ):
        _report_progress(idx, total_items)
        # Load child data
        child_data = load_section_npz(child_path)
        child_labels = child_data["labels32"]  # (32,32,32) int32

        # Load heightmap if present (column-heights step may not have run yet)
        if "heightmap32" in child_data:
            heightmap = child_data["heightmap32"]  # (5, 32, 32) float32
        else:
            heightmap = _EMPTY_HEIGHTMAP.copy()

        biome = child_data["biome32"]  # (32, 32) int32
        raw_y = int(child_data["section_y"])
        # Translate section_y into the model's embedding range [0,24).
        # Raw values are often negative (e.g. -4…19).  Add +4 offset so
        # -4→0, 19→23; clamp to avoid out-of-bounds.
        y_pos = raw_y + 4
        y_pos = max(0, min(y_pos, 24 - 1))
        # Stored NEC may be stale; we recompute below for robustness.
        nec_stored = np.uint8(child_data["non_empty_children"])

        if is_init:
            # L4 init model: no parent context
            parent_labels = np.zeros((32, 32, 32), dtype=np.int32)
            parent_octant16 = np.zeros((16, 16, 16), dtype=np.int32)  # placeholder
            # NEC for L4 is irrelevant for training the init model, but we
            # still include it so the dataset schema stays uniform.
            nec = nec_stored
        else:
            # Find parent section coordinates and octant index
            px, py, pz, octant = parent_coords_and_octant(cx, cy, cz)
            if (px, py, pz) not in parent_index:
                skipped_no_parent += 1
                continue

            parent_data = load_section_npz(parent_index[(px, py, pz)])
            parent_full = parent_data["labels32"]  # (32,32,32)

            # Native 16³ Voxy octant — used by the consistency loss directly.
            parent_octant16 = extract_octant(parent_full, octant)  # (16,16,16)
            # NN-upsampled 32³ context — used as model input feature.
            parent_labels = np.repeat(
                np.repeat(np.repeat(parent_octant16, 2, axis=0), 2, axis=1), 2, axis=2
            ).astype(np.int32)

            # Compute true NEC by inspecting the next-lower level index.
            # This ensures the target matches our dataset rather than the
            # possibly outdated stored bitmask.
            if child_level > 0:
                gc_index = build_section_index(data_dir, child_level - 1)
                nec = compute_true_non_empty_children((cx, cy, cz), child_level, gc_index)
            else:
                # L0 has no children
                nec = np.uint8(0)

        pairs.append(
            {
                "labels32": child_labels,
                "parent_labels32": parent_labels,
                "parent_octant16": parent_octant16,
                "heightmap32": heightmap,
                "biome32": biome,
                "y_position": np.int64(y_pos),  # shifted +4 for embedding
                "level": np.int64(child_level),
                "non_empty_children": nec,
            }
        )

    if skipped_no_parent > 0:
        print(f"    Skipped {skipped_no_parent:,} sections with no parent at L{parent_level}")

    return pairs


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def deterministic_split(
    pairs: list[dict[str, Any]],
    val_fraction: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs into train/val using a deterministic shuffle.

    Uses a hash-based approach so the split is reproducible regardless
    of file-system ordering.
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_val >= len(shuffled):
        n_val = max(1, len(shuffled) // 10)

    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Stacking and saving
# ---------------------------------------------------------------------------


def stack_and_save(pairs: list[dict[str, Any]], cache_path: Path) -> int:
    """Stack a list of pair dicts into arrays and save as compressed NPZ.

    Returns the number of pairs saved.
    """
    if not pairs:
        print(f"  WARNING: No pairs to save for {cache_path.name}")
        return 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(pairs)
    all_labels = np.stack([p["labels32"] for p in pairs])  # (N,32,32,32)
    all_parents = np.stack([p["parent_labels32"] for p in pairs])  # (N,32,32,32)
    all_octant16 = np.stack([p["parent_octant16"] for p in pairs])  # (N,16,16,16)
    all_heightmaps = np.stack([p["heightmap32"] for p in pairs])  # (N,5,32,32)
    all_biomes = np.stack([p["biome32"] for p in pairs])  # (N,32,32)
    all_y_pos = np.array([p["y_position"] for p in pairs], np.int64)  # (N,)
    all_levels = np.array([p["level"] for p in pairs], np.int64)  # (N,)
    all_nec = np.array([p["non_empty_children"] for p in pairs], np.uint8)  # (N,)

    np.savez_compressed(
        cache_path,
        labels32=all_labels,
        parent_labels32=all_parents,
        parent_octant16=all_octant16,
        heightmap32=all_heightmaps,
        biome32=all_biomes,
        y_position=all_y_pos,
        level=all_levels,
        non_empty_children=all_nec,
    )

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    # Use ASCII arrow to avoid UnicodeEncodeError on Windows consoles.
    print(f"  Saved {n:,} pairs -> {cache_path}  ({size_mb:.1f} MB)")
    return n


def compute_root_features(blocks16: npt.NDArray[np.int32]) -> np.ndarray:
    """Compute a fixed-length root feature vector for a 16³ subchunk.

    Features (length 16):
      [air_frac, nonair_frac, unique_frac, top8_nonair_freqs..., padding]
    """

    flat = blocks16.ravel()
    total = float(flat.size)
    air_mask = flat == 0
    air_frac = float(np.count_nonzero(air_mask) / total)
    nonair_frac = 1.0 - air_frac

    unique_frac = float(len(np.unique(flat)) / total)

    nonair = flat[~air_mask]
    if nonair.size > 0:
        labels, counts = np.unique(nonair, return_counts=True)
        order = np.argsort(counts)[::-1]
        top_counts = counts[order][:8]
        top_freqs = top_counts.astype(np.float32) / total
        if top_freqs.size < 8:
            top_freqs = np.pad(top_freqs, (0, 8 - top_freqs.size), constant_values=0.0)
    else:
        top_freqs = np.zeros(8, dtype=np.float32)

    feats = np.concatenate(
        (
            np.array([air_frac, nonair_frac, unique_frac], dtype=np.float32),
            top_freqs.astype(np.float32),
        )
    )

    if feats.size < 16:
        feats = np.pad(feats, (0, 16 - feats.size), constant_values=0.0)

    return feats


def stack_and_save_sparse_root(pairs: list[dict[str, Any]], cache_path: Path) -> int:
    """Stack and save sparse-root pairs as a compressed NPZ."""
    if not pairs:
        print(f"  WARNING: No sparse-root pairs to save for {cache_path.name}")
        return 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(pairs)
    all_subchunks = np.stack([p["subchunk16"] for p in pairs])  # (N,16,16,16)
    all_octants = np.array([p["octant"] for p in pairs], np.int64)  # (N,)
    all_levels = np.array([p["level"] for p in pairs], np.int64)  # (N,)
    all_y_pos = np.array([p["y_position"] for p in pairs], np.int64)  # (N,)
    all_root_feats = np.stack([p["root_features"] for p in pairs]).astype(np.float32)  # (N,16)

    np.savez_compressed(
        cache_path,
        subchunk16=all_subchunks,
        octant=all_octants,
        level=all_levels,
        y_position=all_y_pos,
        root_features=all_root_feats,
    )

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {n:,} sparse-root pairs -> {cache_path}  ({size_mb:.1f} MB)")
    return n


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


# Map model_type → (child_levels, include_l4_init)
_MODEL_TYPE_LEVELS: dict[str, list[int]] = {
    "all": [4, 3, 2, 1, 0],
    "init": [4],
    "refine": [3, 2, 1],
    "leaf": [0],
}


def build(
    data_dir: Path,
    output_dir: Path,
    *,
    val_split: float = 0.1,
    clean: bool = False,
    model_type: str = "all",
    sparse_root: bool = False,
    sparse_root_output: Path | None = None,
) -> tuple[int, int]:
    """Build octree training pair caches.

    Args:
        data_dir: Directory containing level_N/ subdirectories.
        output_dir: Directory to save pair caches.
        val_split: Fraction of pairs for validation.
        clean: Delete existing caches before rebuilding.
        model_type: Which model's pairs to build: ``"all"``, ``"init"``,
            ``"refine"``, or ``"leaf"``.  Non-``"all"`` values produce
            ``{model_type}_train_octree_pairs.npz`` output files.
        sparse_root: If True, generate an additional sparse-root cache.
        sparse_root_output: Output path for sparse-root cache (default:
            ``<output_dir>/sparse_root_pairs.npz``).

    Returns (n_train_pairs, n_val_pairs).
    """
    if model_type not in _MODEL_TYPE_LEVELS:
        raise ValueError(f"Unknown model_type {model_type!r}. Valid: {sorted(_MODEL_TYPE_LEVELS)}")
    active_levels = _MODEL_TYPE_LEVELS[model_type]

    # Output filename prefix: empty for legacy "all", else "{model_type}_"
    prefix = "" if model_type == "all" else f"{model_type}_"

    # Determine sparse-root cache path (can be overridden when writing).
    sparse_root_cache = sparse_root_output or (output_dir / "sparse_root_pairs.npz")

    if clean:
        if model_type == "all":
            for cache_file in output_dir.glob("*_octree_pairs.npz"):
                print(f"  Removing stale cache: {cache_file.name}")
                cache_file.unlink()
        else:
            for split_name in ("train", "val"):
                cache_file = output_dir / f"{prefix}{split_name}_octree_pairs.npz"
                if cache_file.exists():
                    print(f"  Removing stale cache: {cache_file.name}")
                    cache_file.unlink()

        if sparse_root and sparse_root_cache.exists():
            print(f"  Removing stale cache: {sparse_root_cache.name}")
            sparse_root_cache.unlink()

    # Build indices for all levels
    print("Building section indices...")
    indices: dict[int, dict[tuple[int, int, int], Path]] = {}
    for level in range(MAX_LOD_LEVEL + 1):
        idx = build_section_index(data_dir, level)
        indices[level] = idx
        print(f"  L{level}: {len(idx):,} sections indexed")
    print()

    # Check we have data
    total_sections = sum(len(idx) for idx in indices.values())
    if total_sections == 0:
        print("ERROR: No sections found. Run extract-octree first.")
        sys.exit(1)

    if model_type != "all":
        print(f"Building pairs for model_type={model_type!r} (levels: {active_levels})")
        print()

    # Build pairs for each child level (filtered by active_levels)
    all_pairs: list[dict[str, Any]] = []

    # L4 sections → init model (no parent)
    if 4 in active_levels and indices[4]:
        print(f"Building L4 init pairs ({len(indices[4]):,} sections)...")
        l4_pairs = build_pairs_for_level(
            child_level=4,
            data_dir=data_dir,
            child_index=indices[4],
            parent_index={},  # no parent for L4
        )
        all_pairs.extend(l4_pairs)
        print(f"  L4 init: {len(l4_pairs):,} pairs")
        print()

    # L3 through L0: refine/leaf models (have parent at level+1)
    for child_level in range(3, -1, -1):
        if child_level not in active_levels:
            continue
        parent_level = child_level + 1
        if not indices[child_level]:
            print(f"Skipping L{child_level} — no sections")
            continue
        if not indices[parent_level]:
            print(f"Skipping L{child_level} — no parent sections at L{parent_level}")
            continue

        # Use ASCII arrow to avoid encoding issues on Windows consoles
        print(f"Building L{child_level}<-L{parent_level} pairs...")
        level_pairs = build_pairs_for_level(
            child_level=child_level,
            data_dir=data_dir,
            child_index=indices[child_level],
            parent_index=indices[parent_level],
        )
        all_pairs.extend(level_pairs)
        print(f"  L{child_level}: {len(level_pairs):,} pairs")
        print()

    if not all_pairs:
        print("ERROR: No pairs were generated!")
        sys.exit(1)

    print(f"Total pairs across all levels: {len(all_pairs):,}")
    print()

    # Train/val split
    train_pairs, val_pairs = deterministic_split(all_pairs, val_split)
    print(f"Split: {len(train_pairs):,} train, {len(val_pairs):,} val " f"(val_split={val_split})")
    print()

    # --- Level distribution summary ---
    print("Level distribution:")
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        level_counts: dict[int, int] = {}
        for p in split_pairs:
            lvl = int(p["level"])
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        dist_str = ", ".join(f"L{lvl}={cnt}" for lvl, cnt in sorted(level_counts.items()))
        print(f"  {split_name}: {dist_str}")
    print()

    # Save caches
    print("=" * 62)
    print("  Saving pair caches...")
    print("=" * 62)
    n_train = stack_and_save(train_pairs, output_dir / f"{prefix}train_octree_pairs.npz")
    n_val = stack_and_save(val_pairs, output_dir / f"{prefix}val_octree_pairs.npz")

    sparse_root_count = 0
    if sparse_root:
        print("""
  Building sparse-root pairs (from all sections)...
""")
        sparse_pairs: list[dict[str, Any]] = []
        for p in all_pairs:
            labels = p["labels32"]
            y_pos = int(p["y_position"])
            level = int(p["level"])
            for octant in range(8):
                subchunk = extract_octant(labels, octant)
                sparse_pairs.append(
                    {
                        "subchunk16": subchunk.astype(np.int32),
                        "octant": np.int64(octant),
                        "level": np.int64(level),
                        "y_position": np.int64(y_pos),
                        "root_features": compute_root_features(subchunk),
                    }
                )
        sparse_root_count = stack_and_save_sparse_root(sparse_pairs, sparse_root_cache)
        print(f"  Sparse-root pairs: {sparse_root_count:,} (from {len(all_pairs):,} sections)")
        print()

    # Write completion marker
    marker_name = f".{prefix}build_octree_pairs_done" if prefix else ".build_octree_pairs_done"
    marker = output_dir / marker_name
    marker.write_text(
        f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"model_type: {model_type}\n"
        f"train_pairs: {n_train}\n"
        f"val_pairs: {n_val}\n"
        f"sparse_root_pairs: {sparse_root_count}\n"
    )

    # Summary
    print()
    print("=" * 62)
    print("  OCTREE PAIR BUILD COMPLETE")
    print(f"  Train pairs : {n_train:>10,}")
    print(f"  Val pairs   : {n_val:>10,}")
    print(f"  Total       : {n_train + n_val:>10,}")
    cache_files = sorted(output_dir.glob("*_octree_pairs.npz"))
    total_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    print(f"  Cache size  : {total_mb:.1f} MB  ({len(cache_files)} file(s))")
    print("=" * 62)

    return n_train, n_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build octree parent/child training pairs from extracted NPZ sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/voxy_octree"),
        metavar="DIR",
        help="Directory containing level_N/ subdirectories (default: data/voxy_octree)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for pair caches (default: same as --data-dir)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of pairs for validation (default: 0.1)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing *_octree_pairs.npz caches before rebuilding",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["all", "init", "refine", "leaf"],
        metavar="TYPE",
        help=(
            "Which model's training pairs to build (default: all). "
            "'init'=L4 only, 'refine'=L1-L3 only, 'leaf'=L0 only, 'all'=all levels. "
            "Non-'all' values write '{TYPE}_train/val_octree_pairs.npz'."
        ),
    )
    parser.add_argument(
        "--sparse-root",
        action="store_true",
        help="Generate an additional sparse-root pair cache.",
    )
    parser.add_argument(
        "--sparse-root-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for sparse-root cache (default: <output_dir>/sparse_root_pairs.npz).",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.data_dir

    build(
        args.data_dir,
        output_dir,
        val_split=args.val_split,
        clean=args.clean,
        model_type=args.model_type,
        sparse_root=args.sparse_root,
        sparse_root_output=args.sparse_root_output,
    )


if __name__ == "__main__":
    main()
