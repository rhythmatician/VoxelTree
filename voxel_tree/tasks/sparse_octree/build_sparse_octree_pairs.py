"""Build v7 training pairs from v7 noise dumps + Voxy L4 sections.

Each output sample is one 16³ voxel subchunk (octant of a Voxy L4 section)
paired with the 13-channel SparseOctree cave-noise context, biome IDs, and
per-column heightmaps for that section.

V7 noise dumps
--------------
  Files: section_{cx}_{sy}_{cz}.json  (chunk x, section y, chunk z coordinates)
  Each file has 13 cave noise channels + biome_ids, each 32 elements (4×2×4),
  indexed [qx*8 + qy*4 + qz].
  Channels match WorldNoiseAccess.NOISE_3D_PATHS in LODiffusion (in order):
          offset, factor, jaggedness, depth, sloped_cheese, y,
          entrances, pillars, spaghetti_2d, spaghetti_roughness,
          noodle, base_3d_noise, final_density
  biome_ids: 32 discrete biome indices (int), same layout
  heightmap_surface: 256 ints (16×16, x-major)
  heightmap_ocean_floor: 256 ints (16×16, x-major)

Voxy L4 sections
----------------
  Files: {voxy_dir}/level_4/voxy_L4_x{X}_y{Y}_z{Z}.npz
  X = chunk_x, Z = chunk_z, Y = section_y  (range -4..19)
  labels32: (32, 32, 32) int32 in (y, z, x) order  [Voxy's native ordering]
  Each section spans 16 blocks in each axis (16 sections × 16 blocks = 384 blocks)

Output
------
  subchunk16           : (N, 16, 16, 16)   int32   — native Voxy voxels (one octant per section)
  noise_3d             : (N, 13, 4, 2, 4)  float32 — all 13 SparseOctree noise channels
  biome_ids            : (N, 4, 2, 4)      int32   — biome index per quart cell
  heightmap_surface    : (N, 16, 16)       int32   — WORLD_SURFACE_WG heights (x-major)
  heightmap_ocean_floor: (N, 16, 16)       int32   — OCEAN_FLOOR_WG heights (x-major)

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

# SparseOctree noise channels — must match WorldNoiseAccess.NOISE_3D_PATHS in LODiffusion.
# These are cave density function values sampled at quart-cell resolution,
# NOT the climate RouterField values (temperature, vegetation, etc.).
NOISE_FIELDS = [
    "offset",  # overworld/offset
    "factor",  # overworld/factor
    "jaggedness",  # overworld/jaggedness
    "depth",  # router.depth()
    "sloped_cheese",  # overworld/sloped_cheese
    "y",  # cell-centre Y coordinate
    "entrances",  # overworld/caves/entrances
    "pillars",  # overworld/caves/pillars (cheese caves)
    "spaghetti_2d",  # overworld/caves/spaghetti_2d
    "spaghetti_roughness",  # overworld/caves/spaghetti_roughness_function
    "noodle",  # overworld/caves/noodle
    "base_3d_noise",  # overworld/base_3d_noise
    "final_density",  # router.finalDensity()
]
N_FIELDS = len(NOISE_FIELDS)  # 13
assert N_FIELDS == 13, f"Expected 13 SparseOctree noise channels, got {N_FIELDS}"

# Each section file covers exactly one (cx, sy, cz) triplet — 4×2×4 = 32 quart cells.
# Section Y range in Minecraft overworld: -4 to 19 inclusive (24 sections × 16 blocks = 384 blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
) -> int:
    """Build and save sparse-root training pairs.

    Returns number of pairs saved.
    """
    dump_files = sorted(dumps_dir.glob("section_*.json"))
    if not dump_files:
        print(f"ERROR: No section_*.json files found in {dumps_dir}")
        sys.exit(1)
    print(f"  Found {len(dump_files):,} sparse_octree section dump files")

    voxy_index = build_voxy_index(voxy_dir)

    section_pattern = re.compile(r"section_(-?\d+)_(-?\d+)_(-?\d+)\.json$")

    subchunks: list[np.ndarray] = []  # each (16, 16, 16) int32
    noise_slices: list[np.ndarray] = []  # each (13, 4, 2, 4) float32
    biome_slices: list[np.ndarray] = []  # each (4, 2, 4) int32
    hm_surface_slices: list[np.ndarray] = []  # each (16, 16) int32
    hm_ocean_slices: list[np.ndarray] = []  # each (16, 16) int32

    matched_sections = 0
    skipped_sections = 0

    for dump_path in dump_files:
        m = section_pattern.search(dump_path.name)
        if not m:
            continue
        cx, sy, cz = int(m.group(1)), int(m.group(2)), int(m.group(3))

        voxy_key = (cx, sy, cz)
        if voxy_key not in voxy_index:
            skipped_sections += 1
            continue

        # Load JSON — noise fields are flat 32-value arrays indexed [qx*8 + qy*4 + qz]
        with open(dump_path) as f:
            raw = json.load(f)

        # Parse 15 noise fields: 32-value flat → (4, 2, 4)
        field_arrays: list[np.ndarray] = []
        for field in NOISE_FIELDS:
            arr = np.array(raw[field], dtype=np.float32)  # (32,)
            arr = arr.reshape(4, 2, 4)  # (qx, qy, qz)
            field_arrays.append(arr)
        noise_block = np.stack(field_arrays)  # (13, 4, 2, 4)

        # Parse biome IDs: 32-value flat → (4, 2, 4)
        biome_arr = np.array(raw["biome_ids"], dtype=np.int32).reshape(4, 2, 4)

        # Parse heightmaps: 256-value flat → (16, 16)
        # Fall back to zeros if the dump was collected before heightmap support was added.
        if "heightmap_surface" in raw:
            hm_surface = np.array(raw["heightmap_surface"], dtype=np.int32).reshape(16, 16)
            hm_ocean = np.array(raw["heightmap_ocean_floor"], dtype=np.int32).reshape(16, 16)
        else:
            hm_surface = np.zeros((16, 16), dtype=np.int32)
            hm_ocean = np.zeros((16, 16), dtype=np.int32)

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
            hm_surface_slices.append(hm_surface)
            hm_ocean_slices.append(hm_ocean)

    if not subchunks:
        print("ERROR: No pairs generated — check that dumps_dir and voxy_dir overlap.")
        sys.exit(1)

    n = len(subchunks)
    print(f"  Total pairs: {n:,} ({matched_sections:,} sections × 8 octants)")

    # Stack and save
    all_subchunks = np.stack(subchunks).astype(np.int32)  # (N, 16, 16, 16)
    all_noise_3d = np.stack(noise_slices).astype(np.float32)  # (N, 13, 4, 2, 4)
    all_biome_ids = np.stack(biome_slices).astype(np.int32)  # (N, 4, 2, 4)
    all_hm_surface = np.stack(hm_surface_slices).astype(np.int32)  # (N, 16, 16)
    all_hm_ocean = np.stack(hm_ocean_slices).astype(np.int32)  # (N, 16, 16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        subchunk16=all_subchunks,
        noise_3d=all_noise_3d,
        biome_ids=all_biome_ids,
        heightmap_surface=all_hm_surface,
        heightmap_ocean_floor=all_hm_ocean,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved -> {output_path}  ({size_mb:.1f} MB)")
    return n


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

    n = build_pairs(args.dumps, args.voxy, args.output)

    print()
    print("=" * 62)
    print(f"  DONE — {n:,} pairs saved")
    print("=" * 62)


if __name__ == "__main__":
    main()
