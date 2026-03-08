#!/usr/bin/env python3
"""add_column_heights.py — Compute column-level surface heights for training NPZs.

The original extraction stores a per-slab heightmap (highest non-air Y within
the 16-block section, normalised to [0, 1]).  This is wrong for conditioning
because the model needs the *column-level* surface height — the real world-Y
coordinate of the terrain surface — to match what the Java runtime computes
via ``ChunkGenerator.getHeight()``.

This script:
  1. Scans all ``voxy_lod0_*.npz`` files in a directory
  2. Groups them by (x, z) column
  3. For each column, loads all Y slices and computes the true per-column
     surface height (highest non-air block Y in world coordinates)
  4. Stores ``heightmap_surface`` [16, 16] float32 in world-Y coordinates
     (e.g., 65.0 for a block at Y=65) and ``heightmap_ocean_floor`` as
     the lowest non-air Y per column (a rough approximation)
  5. Re-saves each NPZ with the new fields added

The training code normalises these by ``/ 320`` to match the Java runtime's
``AnchorSampler.computeHeightPlanes()``.

Usage::

    python scripts/add_column_heights.py data/voxy_subset/train
    python scripts/add_column_heights.py data/voxy_subset/val
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


# World limits
BOTTOM_Y = -64  # Minecraft 1.18+ overworld bottom
HEIGHT_RANGE = 384  # -64 to 320


def compute_column_surface_heights(
    column_files: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column surface and ocean-floor heights.

    Args:
        column_files: list of (section_y, file_path) for one (x, z) column,
                      sorted by section_y ascending.

    Returns:
        surface_height: (16, 16) float32 — highest non-air block Y (world coords)
        ocean_floor:    (16, 16) float32 — lowest non-air block Y (world coords)
    """
    # Track highest and lowest non-air block Y for each (z, x) column
    # Initialise: surface at bottom, floor at top
    surface = np.full((16, 16), float(BOTTOM_Y), dtype=np.float32)
    floor = np.full((16, 16), float(BOTTOM_Y + HEIGHT_RANGE), dtype=np.float32)

    for section_y, fpath in column_files:
        data = np.load(fpath)
        # terrain_labels required: terrain-only blocks (vegetation filtered)
        # If missing, file was extracted with old code — must re-extract.
        if "terrain_labels" not in data:
            raise KeyError(
                f"{fpath}: missing 'terrain_labels' field. "
                "File was extracted with old code. Re-run extract_voxy_training_data.py."
            )
        block_ids = data["terrain_labels"]  # (16, 16, 16) — (y, z, x), vegetation-filtered

        base_y = section_y * 16  # world Y of bottom of this section

        for z in range(16):
            for x in range(16):
                col = block_ids[:, z, x]  # 16 values along Y
                nz = np.nonzero(col > 0)[0]
                if len(nz) > 0:
                    highest_world_y = base_y + float(nz[-1]) + 1.0
                    lowest_world_y = base_y + float(nz[0])
                    if highest_world_y > surface[z, x]:
                        surface[z, x] = highest_world_y
                    if lowest_world_y < floor[z, x]:
                        floor[z, x] = lowest_world_y

    # Clamp floor: if no terrain block was found, set to surface
    no_solid = surface <= BOTTOM_Y
    surface[no_solid] = 62.0  # sea level default
    floor[no_solid] = 62.0
    # If floor ended up above surface (shouldn't happen), clamp
    floor = np.minimum(floor, surface)

    return surface, floor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add column-level heightmap_surface to training NPZ files.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing voxy_lod0_*.npz files (e.g., data/voxy_subset/train).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just report what would be done; don't modify files.",
    )
    args = parser.parse_args()

    # Find all NPZ files
    pattern = str(args.data_dir / "voxy_lod0_*.npz")
    files = glob.glob(pattern)
    if not files:
        # Try recursive
        pattern = str(args.data_dir / "**" / "voxy_lod0_*.npz")
        files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No voxy_lod0_*.npz files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} NPZ files")

    # Group by (x, z) column
    columns: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for f in files:
        m = re.search(r"x(-?\d+)_y(-?\d+)_z(-?\d+)", f)
        if not m:
            print(f"  Skipping (no coord in name): {f}")
            continue
        wx, wy, wz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        columns[(wx, wz)].append((wy, f))

    print(f"Grouped into {len(columns)} unique (x, z) columns")

    # Check how many already have heightmap_surface
    sample = np.load(files[0])
    has_surface = "heightmap_surface" in sample
    if has_surface:
        print("WARNING: Files already contain heightmap_surface — will overwrite")

    if args.dry_run:
        print("Dry run — no files will be modified")
        return

    # Process each column
    updated = 0
    for (wx, wz), col_files in tqdm(columns.items(), desc="Computing column heights"):
        col_files.sort(key=lambda t: t[0])  # sort by section Y
        surface, floor = compute_column_surface_heights(col_files)

        # Re-save each section's NPZ with the new fields
        for section_y, fpath in col_files:
            data = dict(np.load(fpath))
            data["heightmap_surface"] = surface
            data["heightmap_ocean_floor"] = floor
            np.savez_compressed(fpath, **data)
            updated += 1

    print(f"\nUpdated {updated} NPZ files with column-level heightmap_surface")

    # Verify a sample
    sample_file = files[0]
    d = np.load(sample_file)
    hs = d["heightmap_surface"]
    print(f"\nVerification ({Path(sample_file).name}):")
    print(f"  heightmap_surface:    shape={hs.shape} min={hs.min():.1f} max={hs.max():.1f}")
    if "heightmap_ocean_floor" in d:
        hof = d["heightmap_ocean_floor"]
        print(f"  heightmap_ocean_floor: shape={hof.shape} min={hof.min():.1f} max={hof.max():.1f}")


if __name__ == "__main__":
    main()
