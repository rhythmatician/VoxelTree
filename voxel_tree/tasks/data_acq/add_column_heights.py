#!/usr/bin/env python3
"""add_column_heights.py — Merge vanilla heightmaps and biomes from /dumpnoise JSON into NPZs.

Supports two modes:

  **Legacy mode** (default): Processes voxy_lod0_*.npz files in 16×16 layout.
  **Octree mode** (``--octree``): Processes data/voxy_octree/level_N/ multi-LOD files
    in 32×32 layout with 5-plane heightmap features.

Legacy mode:
  The Java runtime computes heightmaps via ``ChunkGenerator.getHeight()`` with
  ``Heightmap.Type.WORLD_SURFACE_WG``.  This script loads ``chunk_<cx>_<cz>.json``
  files and merges ``heightmap_surface`` / ``heightmap_ocean_floor`` into NPZs.

Octree mode:
  For each LOD level, vanilla chunk heightmaps are stitched/pooled to match the
  section's world footprint (32×(2^level) blocks per axis), then downsampled to
  32×32.  Five feature planes are computed:
    0: surface height (normalised)
    1: ocean floor height (normalised)
    2: slope_x (finite-difference gradient along x)
    3: slope_z (finite-difference gradient along z)
    4: curvature (Laplacian)

  Heightmap scaling per level:
    L0 → 32×32 blocks footprint  → stitch 2×2 chunk heightmaps → 32×32
    L1 → 64×64 blocks footprint  → pool 64×64 vanilla → 32×32
    L2 → 128×128 blocks footprint → pool 128×128 → 32×32
    L3 → 256×256 blocks footprint → pool 256×256 → 32×32
    L4 → 512×512 blocks footprint → pool 512×512 → 32×32

Usage::

    python scripts/add_column_heights.py data/voxy_octree \\
        --noise-dump-dir tools/fabric-server/runtime/noise_dumps
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from voxel_tree.utils.biome_mapping import BIOME_NAME_TO_ID, UNKNOWN_BIOME_ID
from tqdm import tqdm

from voxel_tree.utils.progress import report as _report_progress

# ---------------------------------------------------------------------------
# Height normalisation constant (MC world height range: -64 to 320 = 384)
# ---------------------------------------------------------------------------
HEIGHT_NORM = 384.0


def load_noise_dumps(noise_dump_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load all chunk_<cx>_<cz>.json files into a dict keyed by (cx, cz).

    Each JSON contains:
      - ``heightmap_surface``: flat 256-element array (x-major, 16×16)
      - ``heightmap_ocean_floor``: flat 256-element array (x-major, 16×16)
      - ``biome_names``: flat 256-element string array (x-major, 16×16)
      - ``chunk_x``, ``chunk_z``: chunk coordinates

    Returns:
        dict mapping (chunk_x, chunk_z) → parsed JSON dict
    """
    print(f"Loading noise dumps from {noise_dump_dir}...")
    pattern = str(noise_dump_dir / "chunk_*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"ERROR: No chunk_*.json files found in {noise_dump_dir}")
        sys.exit(1)

    dumps: dict[tuple[int, int], dict[str, Any]] = {}
    total = len(files)
    for idx, fpath in enumerate(
        tqdm(files, desc="Loading noise dumps", unit="file", dynamic_ncols=True)
    ):
        # emit explicit progress as well so the GUI sees it even if tqdm
        # suppresses output when run under a pipe
        _report_progress(idx, total)
        with open(fpath) as f:
            data = json.load(f)
        cx = data["chunk_x"]
        cz = data["chunk_z"]
        dumps[(cx, cz)] = data
    _report_progress(total, total)

    print(f"Loaded {len(dumps)} noise dump JSON files from {noise_dump_dir}")
    return dumps


def parse_heightmap(flat_array: list[int | float]) -> npt.NDArray[np.float32]:
    """Convert a flat 256-element x-major heightmap to (16, 16) float32.

    The Java /dumpnoise command writes heightmaps in x-major order::

        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                append(grid[x][z])

    The NPZ convention uses (z, x) indexing for heightmap arrays, matching
    the block array layout (y, z, x).  So we reshape to (16, 16) as (x, z)
    from the flat array and transpose to (z, x).
    """
    arr = np.array(flat_array, dtype=np.float32).reshape(16, 16)  # (x, z)
    return arr.T  # → (z, x)


def parse_biome_names(flat_names: list[str]) -> npt.NDArray[np.int32]:
    """Convert a flat 256-element biome name list to (16, 16) int32.

    Maps each biome name (e.g. ``"minecraft:plains"``) to its canonical
    integer ID using the shared ``biome_mapping.BIOME_NAME_TO_ID`` dict.
    Unknown biomes map to ``UNKNOWN_BIOME_ID`` (255).

    Layout follows the same x-major → (z, x) transpose as heightmaps.
    """
    ids = np.array(
        [BIOME_NAME_TO_ID.get(name, UNKNOWN_BIOME_ID) for name in flat_names],
        dtype=np.int32,
    ).reshape(
        16, 16
    )  # (x, z)
    return ids.T  # → (z, x)


# ---------------------------------------------------------------------------
# Octree heightmap helpers (32×32 multi-level)
# ---------------------------------------------------------------------------


def stitch_chunk_heightmaps(
    dumps: dict[tuple[int, int], dict[str, Any]],
    chunk_coords: list[tuple[int, int]],
    field: str,
) -> npt.NDArray[np.float32] | None:
    """Stitch multiple 16×16 chunk heightmaps into a larger grid.

    ``chunk_coords`` must be a list of (cx, cz) in row-major order.
    ``field`` is the JSON key (e.g. ``"heightmap_surface"``).

    Returns the stitched array in (z, x) order, or None if any chunk is missing.
    """
    # Determine grid dimensions from chunk coords
    cxs = sorted(set(c[0] for c in chunk_coords))
    czs = sorted(set(c[1] for c in chunk_coords))
    nx, nz = len(cxs), len(czs)

    grid = np.zeros((nz * 16, nx * 16), dtype=np.float32)  # (z, x)

    cx_to_col = {cx: i for i, cx in enumerate(cxs)}
    cz_to_row = {cz: j for j, cz in enumerate(czs)}

    for cx, cz in chunk_coords:
        dump = dumps.get((cx, cz))
        if dump is None or field not in dump:
            return None
        patch = parse_heightmap(dump[field])  # (16, 16) (z, x)
        row = cz_to_row[cz]
        col = cx_to_col[cx]
        grid[row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16] = patch

    return grid


def pool_heightmap_to_32(
    full_grid: npt.NDArray[np.float32],
    target_size: int = 32,
) -> npt.NDArray[np.float32]:
    """Average-pool a (H, W) heightmap down to (target_size, target_size).

    H and W must be divisible by target_size.
    """
    h, w = full_grid.shape
    bh = h // target_size
    bw = w // target_size
    # Truncate to exact multiple
    trimmed = full_grid[: bh * target_size, : bw * target_size]
    reshaped = trimmed.reshape(target_size, bh, target_size, bw)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


def compute_height_planes(
    surface_32: npt.NDArray[np.float32],
    ocean_floor_32: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute 5-plane heightmap features from surface and ocean-floor heights.

    Returns (5, 32, 32) float32:
      0: normalised surface height
      1: normalised ocean floor height
      2: slope_x (gradient along x axis)
      3: slope_z (gradient along z axis)
      4: curvature (discrete Laplacian)
    """
    planes = np.zeros((5, 32, 32), dtype=np.float32)

    # Normalise heights
    surf_norm = surface_32 / HEIGHT_NORM
    ocean_norm = ocean_floor_32 / HEIGHT_NORM

    planes[0] = surf_norm
    planes[1] = ocean_norm

    # Finite-difference gradients
    # slope_x: gradient along x axis (axis=1 in (z,x) layout)
    planes[2, :, 1:] = np.diff(surf_norm, axis=1)
    planes[2, :, 0] = planes[2, :, 1]

    # slope_z: gradient along z axis (axis=0 in (z,x) layout)
    planes[3, 1:, :] = np.diff(surf_norm, axis=0)
    planes[3, 0, :] = planes[3, 1, :]

    # Curvature: discrete Laplacian
    padded = np.pad(surf_norm, 1, mode="edge")
    laplacian = (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    planes[4] = laplacian

    return planes


def get_chunk_coords_for_section(
    section_x: int,
    section_z: int,
    level: int,
) -> list[tuple[int, int]]:
    """Compute all chunk (cx, cz) coordinates that overlap a section's footprint.

    A section at level L covers (32 × 2^L) blocks per axis.
    Each chunk covers 16 blocks, so the section covers (32 × 2^L / 16) = 2^(L+1) chunks.

    Section coordinates map to world block origin:
      block_x_origin = section_x × 32 × 2^level
      block_z_origin = section_z × 32 × 2^level

    Chunk coordinates:
      chunk_x_start = block_x_origin / 16
      chunk_z_start = block_z_origin / 16
    """
    scale = 2**level  # voxel size in blocks
    blocks_per_section = 32 * scale  # world block span of one section
    chunks_per_axis = blocks_per_section // 16  # how many chunks fit

    block_x_origin = section_x * blocks_per_section
    block_z_origin = section_z * blocks_per_section

    cx_start = block_x_origin // 16
    cz_start = block_z_origin // 16

    coords = []
    for cz in range(cz_start, cz_start + chunks_per_axis):
        for cx in range(cx_start, cx_start + chunks_per_axis):
            coords.append((cx, cz))

    return coords


def build_octree_heightmap(
    dumps: dict[tuple[int, int], dict[str, Any]],
    section_x: int,
    section_z: int,
    level: int,
) -> npt.NDArray[np.float32] | None:
    """Build a (5, 32, 32) heightmap for an octree section.

    Returns None if insufficient chunk data is available.
    """
    chunk_coords = get_chunk_coords_for_section(section_x, section_z, level)

    surface_full = stitch_chunk_heightmaps(dumps, chunk_coords, "heightmap_surface")
    ocean_full = stitch_chunk_heightmaps(dumps, chunk_coords, "heightmap_ocean_floor")

    if surface_full is None or ocean_full is None:
        return None

    h, w = surface_full.shape
    if h < 32 or w < 32:
        return None

    # Pool down to 32×32
    surface_32 = pool_heightmap_to_32(surface_full, 32)
    ocean_32 = pool_heightmap_to_32(ocean_full, 32)

    return compute_height_planes(surface_32, ocean_32)


def run_octree(args: argparse.Namespace) -> None:
    """Merge 5-plane heightmaps into octree-extracted NPZ files at all levels."""

    dumps = load_noise_dumps(args.noise_dump_dir)

    data_dir: Path = args.data_dir
    max_level = 4

    total_updated = 0
    total_skipped = 0
    total_missing = 0

    print(f"\n{'='*60}")
    print("  VoxelTree — Add Column Heights to NPZs")
    print(f"{'='*60}\n")

    for level in range(max_level + 1):
        level_dir = data_dir / f"level_{level}"
        if not level_dir.is_dir():
            print(f"  L{level}: directory not found ({level_dir}), skipping")
            continue

        files = sorted(level_dir.glob("*.npz"))
        if not files:
            print(f"  L{level}: no NPZ files, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  L{level}: Processing {len(files):,} files")
        print(f"  Section footprint: {32 * 2**level}×{32 * 2**level} blocks")
        print(f"  Chunks per section axis: {2**(level+1)}")
        print(f"{'='*60}")

        updated = 0
        skipped_nocoord = 0
        skipped_nodata = 0

        total2 = len(files)
        for idx, fpath in enumerate(tqdm(files, desc=f"  L{level} heightmaps", unit="file")):
            _report_progress(idx, total2)
            # Load existing NPZ
            try:
                npz = np.load(fpath)
                sx = int(npz["section_x"])
                sz = int(npz["section_z"])
            except Exception:
                skipped_nocoord += 1
                continue

            if args.dry_run:
                # Just check if heightmap can be built
                hm = build_octree_heightmap(dumps, sx, sz, level)
                if hm is not None:
                    updated += 1
                else:
                    skipped_nodata += 1
                npz.close()
                continue

            # Build 5-plane heightmap
            hm = build_octree_heightmap(dumps, sx, sz, level)
            if hm is None:
                skipped_nodata += 1
                npz.close()
                continue

            # Re-save with heightmap32 added
            data = {k: npz[k] for k in npz.files}
            npz.close()
            data["heightmap32"] = hm
            np.savez_compressed(fpath, **data)
            updated += 1

        print(
            f"  L{level}: {updated:,} updated, "
            f"{skipped_nodata:,} missing noise data, "
            f"{skipped_nocoord:,} bad coords"
        )
        total_updated += updated
        total_skipped += skipped_nocoord
        total_missing += skipped_nodata

    print(f"\n{'='*60}")
    print("  OCTREE COLUMN-HEIGHTS COMPLETE")
    print(f"  Updated:      {total_updated:,}")
    print(f"  Missing data: {total_missing:,}")
    print(f"  Bad coords:   {total_skipped:,}")
    print(f"{'='*60}")

    if total_updated == 0:
        print("\nERROR: No files were updated — check noise dump coverage")
        sys.exit(1)

    # Write marker
    marker = data_dir / ".column_heights_octree_done"
    marker.write_text(f"updated: {total_updated}\n")
    print(f"  Marker written: {marker}")

    # Verify a sample
    for level in range(max_level + 1):
        level_dir = data_dir / f"level_{level}"
        if not level_dir.is_dir():
            continue
        sample_files = sorted(level_dir.glob("*.npz"))[:1]
        for sf in sample_files:
            d = np.load(sf)
            if "heightmap32" in d:
                hm = d["heightmap32"]
                if hm is None:
                    raise ValueError("heightmap32 is None")
                print(f"\n  Verification L{level} ({sf.name}):")
                print(
                    f"    heightmap32: shape={hm.shape} " f"min={hm.min():.3f} max={hm.max():.3f}"
                )
                labels = ["surface", "ocean_floor", "slope_x", "slope_z", "curvature"]
                for i, lbl in enumerate(labels):
                    plane = hm[i]
                    print(
                        f"      [{i}] {lbl:14s}: "
                        f"min={plane.min():.4f} max={plane.max():.4f} "
                        f"mean={plane.mean():.4f}"
                    )
            d.close()
            break


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    print("\n" + "=" * 60)
    print("  VoxelTree — Add Column Heights to NPZs")
    print("=" * 60 + "\n")

    parser = argparse.ArgumentParser(
        description="Merge vanilla heightmaps from /dumpnoise JSON into training NPZ files.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing NPZ files " "(level_N/ subdirs).",
    )
    parser.add_argument(
        "--noise-dump-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing /dumpnoise chunk_*.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just report what would be done; don't modify files.",
    )
    args = parser.parse_args(argv)

    run_octree(args)
    return


if __name__ == "__main__":
    main()
    main()
