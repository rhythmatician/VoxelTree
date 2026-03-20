"""Build v7 training pairs from v7 noise dumps + Voxy L4 sections.

Each output sample is one 16³ voxel subchunk (octant of a Voxy L4 section)
paired with the 15-channel v7 RouterField noise context, biome IDs, and
per-column heightmaps for that section.

V7 noise dumps
--------------
  Files: section_{cx}_{sy}_{cz}.json  (chunk x, section y, chunk z coordinates)
  Each file has 15 RouterField channels + biome_ids, each 32 elements (4×2×4),
  indexed [qx*8 + qy*4 + qz].
  Channels match RouterField.java ordinals 0-14 (in order):
          temperature, vegetation, continents, erosion, depth, ridges,
          preliminary_surface_level, final_density,
          barrier, fluid_level_floodedness, fluid_level_spread, lava,
          vein_toggle, vein_ridged, vein_gap
  biome_ids: 32 discrete biome indices (int), same layout
  heightmap_surface: 256 ints (16×16, x-major)
  heightmap_ocean_floor: 256 ints (16×16, x-major)

  Legacy (pre-v7) dumps with 13 cave-noise channels are also supported for
  backward compatibility via auto-detection.

Voxy L4 sections
----------------
  Files: {voxy_dir}/level_4/voxy_L4_x{X}_y{Y}_z{Z}.npz
  X = chunk_x, Z = chunk_z, Y = section_y  (range -4..19)
  labels32: (32, 32, 32) int32 in (y, z, x) order  [Voxy's native ordering]
  Each section spans 16 blocks in each axis (16 sections × 16 blocks = 384 blocks)

Output
------
  subchunk16           : (N, 16, 16, 16)   int32   — native Voxy voxels (one octant per section)
  noise_3d             : (N, 15, 4, 2, 4)  float32 — all 15 v7 RouterField channels
  biome_ids            : (N, 4, 2, 4)      int32   — biome index per quart cell
  heightmap5           : (N, 5, 16, 16)    float32 — 5-plane heightmap (surface, ocean, slope_x, slope_z, curvature)

Usage
-----
  python scripts/build_sparse_octree_pairs.py \\
      --dumps  VoxelTree/tools/fabric-server/runtime/v7_dumps \\
      --voxy   VoxelTree/data/voxy_octree \\
      --output noise_training_data/sparse_octree_pairs_v7.npz

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v7 RouterField channels — must match RouterField.java ordinals 0-14 and
# router_field_contract.yaml exactly.  These are the 15 vanilla NoiseRouter
# output fields sampled at 4×2×4 quart-cell resolution.
NOISE_FIELDS = [
    "temperature",  # 0  climate
    "vegetation",  # 1  climate
    "continents",  # 2  climate
    "erosion",  # 3  climate
    "depth",  # 4  climate (3D)
    "ridges",  # 5  climate
    "preliminary_surface_level",  # 6  density
    "final_density",  # 7  density
    "barrier",  # 8  aquifer
    "fluid_level_floodedness",  # 9  aquifer
    "fluid_level_spread",  # 10 aquifer
    "lava",  # 11 aquifer
    "vein_toggle",  # 12 ore
    "vein_ridged",  # 13 ore
    "vein_gap",  # 14 ore
]
N_FIELDS = len(NOISE_FIELDS)  # 15
assert N_FIELDS == 15, f"Expected 15 v7 RouterField channels, got {N_FIELDS}"

# Legacy 13-channel field names for backward-compat auto-detection.
# @deprecated: v7 dumps use NOISE_FIELDS (15 channels) above.
_LEGACY_NOISE_FIELDS = [
    "offset",  # 0  overworld/offset
    "factor",  # 1  overworld/factor
    "jaggedness",  # 2  overworld/jaggedness
    "depth",  # 3  router.depth()
    "sloped_cheese",  # 4  overworld/sloped_cheese
    "y",  # 5  cell-centre Y coordinate
    "entrances",  # 6  overworld/caves/entrances
    "pillars",  # 7  overworld/caves/pillars (cheese caves)
    "spaghetti_2d",  # 8  overworld/caves/spaghetti_2d
    "spaghetti_roughness",  # 9  overworld/caves/spaghetti_roughness_function
    "noodle",  # 10 overworld/caves/noodle
    "base_3d_noise",  # 11 overworld/base_3d_noise
    "final_density",  # 12 router.finalDensity()
]

# Each section file covers exactly one (cx, sy, cz) triplet — 4×2×4 = 32 quart cells.
# Section Y range in Minecraft overworld: -4 to 19 inclusive (24 sections × 16 blocks = 384 blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_height_planes(hm_surface: np.ndarray, hm_ocean: np.ndarray) -> np.ndarray:
    """Compute 5-plane heightmap from raw surface + ocean-floor heightmaps.

    This mirrors ``LodGenerationService.computeOctreeHeightPlanes()`` in Java.
    Works on any (H, W) grid (16×16 training, 32×32 runtime).

    Returns
    -------
    planes : ndarray, float32, shape (5, H, W)
        [0] surface_norm       = surface / 320
        [1] ocean_floor_approx = min(surface, 62) / 320
        [2] slope_x            = central-difference of surface_norm along x (cols)
        [3] slope_z            = central-difference of surface_norm along z (rows)
        [4] curvature          = Laplacian (d²surface/dx² + d²surface/dz²)
    """
    HEIGHT_RANGE = 320.0
    SEA_LEVEL_PLANE = 62.0

    hm_surface = hm_surface.astype(np.float32)
    hm_ocean = hm_ocean.astype(np.float32)
    H, W = hm_surface.shape

    surf_norm = hm_surface / HEIGHT_RANGE
    ocean_approx = np.minimum(hm_surface, SEA_LEVEL_PLANE) / HEIGHT_RANGE

    # slope_x: central difference along columns (axis=1)
    slope_x = np.empty_like(surf_norm)
    slope_x[:, 0] = surf_norm[:, 1] - surf_norm[:, 0]
    slope_x[:, -1] = surf_norm[:, -1] - surf_norm[:, -2]
    slope_x[:, 1:-1] = (surf_norm[:, 2:] - surf_norm[:, :-2]) / 2.0

    # slope_z: central difference along rows (axis=0)
    slope_z = np.empty_like(surf_norm)
    slope_z[0, :] = surf_norm[1, :] - surf_norm[0, :]
    slope_z[-1, :] = surf_norm[-1, :] - surf_norm[-2, :]
    slope_z[1:-1, :] = (surf_norm[2:, :] - surf_norm[:-2, :]) / 2.0

    # curvature: second-order central difference (Laplacian)
    dsx = np.empty_like(slope_x)
    dsx[:, 0] = slope_x[:, 1] - slope_x[:, 0]
    dsx[:, -1] = slope_x[:, -1] - slope_x[:, -2]
    dsx[:, 1:-1] = (slope_x[:, 2:] - slope_x[:, :-2]) / 2.0

    dsz = np.empty_like(slope_z)
    dsz[0, :] = slope_z[1, :] - slope_z[0, :]
    dsz[-1, :] = slope_z[-1, :] - slope_z[-2, :]
    dsz[1:-1, :] = (slope_z[2:, :] - slope_z[:-2, :]) / 2.0

    curvature = dsx + dsz

    return np.stack([surf_norm, ocean_approx, slope_x, slope_z, curvature], axis=0).astype(
        np.float32
    )


def extract_octant(labels32: np.ndarray, octant: int) -> np.ndarray:
    """Extract 16³ octant from a 32³ (y,z,x) labels array.

    Octant bit layout: bit0=x, bit1=z, bit2=y
    Returns (16, 16, 16) int32.
    """
    dx = octant & 1
    dz = (octant >> 1) & 1
    dy = (octant >> 2) & 1
    return labels32[
        dy * 16 : (dy + 1) * 16,
        dz * 16 : (dz + 1) * 16,
        dx * 16 : (dx + 1) * 16,
    ].astype(np.int32)


def build_voxy_index(voxy_dir: Path) -> dict[tuple[int, int, int], Path]:
    """Build (x, y, z) -> Path index for all Voxy L4 npz files."""
    level4_dir = voxy_dir / "level_4"
    if not level4_dir.is_dir():
        raise FileNotFoundError(f"Voxy level_4 directory not found: {level4_dir}")

    pattern = re.compile(r"voxy_L4_x(-?\d+)_y(-?\d+)_z(-?\d+)\.npz$")
    index: dict[tuple[int, int, int], Path] = {}
    for f in level4_dir.iterdir():
        m = pattern.search(f.name)
        if m:
            x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
            index[(x, y, z)] = f

    print(f"  Indexed {len(index):,} Voxy L4 sections from {level4_dir}")
    return index


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_pairs(
    dumps_dir: Path,
    voxy_dir: Path,
    output_path: Path,
) -> tuple[int, dict[str, int]]:
    """Build and save sparse-root training pairs.

    Returns (pairs_saved, failure_stats) where failure_stats contains:
      - pairs_saved: total training samples written
      - matched_sections: sections with both a dump and a Voxy NPZ
      - total_dump_files: dump files discovered in dumps_dir
      - total_voxy_sections: Voxy L4 sections discovered in voxy_dir/level_4
      - skipped_no_voxy: dump files with no matching Voxy section
      - skipped_no_dump: Voxy sections with no matching dump file
      - total_skipped: skipped_no_voxy + skipped_no_dump
    """
    dump_files = sorted(dumps_dir.glob("section_*.json"))
    if not dump_files:
        print(f"ERROR: No section_*.json files found in {dumps_dir}")
        sys.exit(1)
    print(f"  Found {len(dump_files):,} sparse_octree section dump files")

    voxy_index = build_voxy_index(voxy_dir)

    section_pattern = re.compile(r"section_(-?\d+)_(-?\d+)_(-?\d+)\.json$")

    subchunks: list[np.ndarray] = []  # each (16, 16, 16) int32
    noise_slices: list[np.ndarray] = []  # each (15, 4, 2, 4) float32  [v7] or (13, ...) [legacy]
    biome_slices: list[np.ndarray] = []  # each (4, 2, 4) int32
    hm5_slices: list[np.ndarray] = []  # each (5, 16, 16) float32
    block_y_min_list: list[int] = []  # absolute block Y of each octant's bottom

    # Build a set of all (cx, sy, cz) keys parseable from dump files so we can
    # detect Voxy sections that have no corresponding noise dump.
    dump_keys: set[tuple[int, int, int]] = set()
    matched_sections = 0
    skipped_no_voxy = 0  # dump exists, but no matching Voxy section

    for dump_path in dump_files:
        m = section_pattern.search(dump_path.name)
        if not m:
            continue
        cx, sy, cz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dump_keys.add((cx, sy, cz))

        voxy_key = (cx, sy, cz)
        if voxy_key not in voxy_index:
            skipped_no_voxy += 1
            continue

        # Load JSON — noise fields are flat 32-value arrays indexed [qx*8 + qy*4 + qz]
        with open(dump_path) as f:
            raw = json.load(f)

        # Auto-detect v7 (15-ch) vs legacy (13-ch) based on first field name.
        is_v7 = NOISE_FIELDS[0] in raw  # v7 has "temperature" key
        active_fields = NOISE_FIELDS if is_v7 else _LEGACY_NOISE_FIELDS

        # Parse noise fields: 32-value flat → (4, 2, 4)
        field_arrays: list[np.ndarray] = []
        for field in active_fields:
            arr = np.array(raw[field], dtype=np.float32)  # (32,)
            arr = arr.reshape(4, 2, 4)  # (qx, qy, qz)
            field_arrays.append(arr)
        noise_block = np.stack(field_arrays)  # (15, 4, 2, 4) or (13, 4, 2, 4)

        # Parse biome IDs: 32-value flat → (4, 2, 4)
        biome_arr = np.array(raw["biome_ids"], dtype=np.int32).reshape(4, 2, 4)

        # Parse heightmaps: 256-value flat → (16, 16)
        # Fall back to zeros if the dump was collected before heightmap support was added.
        if "heightmap_surface" in raw:
            hm_surface = np.array(raw["heightmap_surface"], dtype=np.float32).reshape(16, 16)
            hm_ocean = np.array(raw["heightmap_ocean_floor"], dtype=np.float32).reshape(16, 16)
        else:
            hm_surface = np.zeros((16, 16), dtype=np.float32)
            hm_ocean = np.zeros((16, 16), dtype=np.float32)

        # Derive 5-plane heightmap: [surface_norm, ocean_approx, slope_x, slope_z, curvature]
        hm5 = compute_height_planes(hm_surface, hm_ocean)  # (5, 16, 16)

        # Load Voxy L4 section
        with np.load(voxy_index[voxy_key]) as vf:
            labels32 = vf["labels32"]  # (32, 32, 32) int32, order (y, z, x)

        matched_sections += 1

        # Extract all 8 octants as independent training samples
        for octant in range(8):
            sub = extract_octant(labels32, octant)  # (16, 16, 16) int32
            subchunks.append(sub)
            noise_slices.append(noise_block)
            biome_slices.append(biome_arr)
            hm5_slices.append(hm5)
            # Absolute block-Y of this octant's bottom edge.
            # Voxy sections are 16 blocks each; labels32 covers 32 blocks.
            dy = (octant >> 2) & 1
            block_y_min_list.append(sy * 16 + dy * 16)

    if not subchunks:
        print("ERROR: No pairs generated — check that dumps_dir and voxy_dir overlap.")
        sys.exit(1)

    # Voxy sections for which no dump file was found.
    skipped_no_dump = sum(1 for key in voxy_index if key not in dump_keys)
    total_skipped = skipped_no_voxy + skipped_no_dump

    n = len(subchunks)
    print(f"  Total pairs: {n:,} ({matched_sections:,} sections × 8 octants)")
    if skipped_no_voxy:
        print(f"  Skipped (no Voxy section):  {skipped_no_voxy:,}")
    if skipped_no_dump:
        print(f"  Skipped (no noise dump):    {skipped_no_dump:,}")

    # Stack and save
    all_subchunks = np.stack(subchunks).astype(np.int32)  # (N, 16, 16, 16)
    all_noise_3d = np.stack(noise_slices).astype(np.float32)  # (N, C, 4, 2, 4) C=15 or 13
    all_biome_ids = np.stack(biome_slices).astype(np.int32)  # (N, 4, 2, 4)
    all_hm5 = np.stack(hm5_slices).astype(np.float32)  # (N, 5, 16, 16)
    all_block_y_min = np.array(block_y_min_list, dtype=np.int32)  # (N,)
    n_ch = all_noise_3d.shape[1]
    print(f"  Noise channels: {n_ch} ({'v7 RouterField' if n_ch == 15 else 'legacy cave-noise'})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        subchunk16=all_subchunks,
        noise_3d=all_noise_3d,
        biome_ids=all_biome_ids,
        heightmap5=all_hm5,
        block_y_min=all_block_y_min,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved -> {output_path}  ({size_mb:.1f} MB)")

    failure_stats: dict[str, int] = {
        "pairs_saved": n,
        "matched_sections": matched_sections,
        "total_dump_files": len(dump_files),
        "total_voxy_sections": len(voxy_index),
        "skipped_no_voxy": skipped_no_voxy,
        "skipped_no_dump": skipped_no_dump,
        "total_skipped": total_skipped,
    }
    return n, failure_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dumps",
        type=Path,
        default=Path("VoxelTree/tools/fabric-server/runtime/v7_dumps"),
        metavar="DIR",
        help="Directory containing section_*.json v7 noise dumps",
    )
    parser.add_argument(
        "--voxy",
        type=Path,
        default=Path("VoxelTree/data/voxy_octree"),
        metavar="DIR",
        help="Voxy data directory containing level_4/ subdirectory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("noise_training_data/sparse_octree_pairs_v7.npz"),
        metavar="FILE",
        help="Output npz file path",
    )
    args = parser.parse_args(argv)

    print("=" * 62)
    print("  Building v7 training pairs (15ch 4×2×4)")
    print("=" * 62)
    print(f"  Dumps dir : {args.dumps}")
    print(f"  Voxy dir  : {args.voxy}")
    print(f"  Output    : {args.output}")
    print()

    n, failure_stats = build_pairs(args.dumps, args.voxy, args.output)

    print()
    print("=" * 62)
    print(f"  DONE — {n:,} pairs saved")
    print("=" * 62)
    print(f"[STEP_RESULT]{json.dumps(failure_stats, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
