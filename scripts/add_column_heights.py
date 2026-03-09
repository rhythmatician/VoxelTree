#!/usr/bin/env python3
"""add_column_heights.py — Merge vanilla heightmaps and biomes from /dumpnoise JSON into NPZs.

The Java runtime computes heightmaps via ``ChunkGenerator.getHeight()`` with
``Heightmap.Type.WORLD_SURFACE_WG`` — this is pure vanilla terrain-generation
math that works for ANY coordinate, including chunks that have never been
loaded.  The ``/dumpnoise`` command exports heightmaps and biome names to JSON.

This script:
  1. Loads all ``chunk_<cx>_<cz>.json`` files from the noise-dump directory
  2. Scans all ``voxy_lod0_*.npz`` files in the data directory
  3. Matches each NPZ to its chunk JSON by (x, z) coordinate
  4. Stores ``heightmap_surface`` [16, 16] float32 and ``heightmap_ocean_floor``
     [16, 16] float32 in each NPZ — values are world-Y coordinates (e.g. 65.0)
  5. Stores ``biome_patch`` [16, 16] int32 — canonical biome IDs from
     ``biome_mapping.py`` (alphabetical overworld biome ordering, 0–53)
  6. Re-saves each NPZ with the new fields added

The training code normalises heights by ``/ 320`` to match the Java runtime's
``AnchorSampler.computeHeightPlanes()``.  Biome IDs index into a learned
``nn.Embedding(256, 32)`` in the model.

Usage::

    python scripts/add_column_heights.py data/voxy/train \\
        --noise-dump-dir LODiffusion/run/noise_dumps

    python scripts/add_column_heights.py data/voxy/val \\
        --noise-dump-dir LODiffusion/run/noise_dumps
"""

from __future__ import annotations
import numpy.typing as npt

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from biome_mapping import BIOME_NAME_TO_ID, UNKNOWN_BIOME_ID


def load_noise_dumps(noise_dump_dir: Path) -> dict[tuple[int, int], dict[str, list[float]]]:
    """Load all chunk_<cx>_<cz>.json files into a dict keyed by (cx, cz).

    Each JSON contains:
      - ``heightmap_surface``: flat 256-element array (x-major, 16×16)
      - ``heightmap_ocean_floor``: flat 256-element array (x-major, 16×16)
      - ``biome_names``: flat 256-element string array (x-major, 16×16)
      - ``chunk_x``, ``chunk_z``: chunk coordinates

    Returns:
        dict mapping (chunk_x, chunk_z) → parsed JSON dict
    """
    pattern = str(noise_dump_dir / "chunk_*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"ERROR: No chunk_*.json files found in {noise_dump_dir}")
        sys.exit(1)

    dumps: dict[tuple[int, int], dict[str, list[float]]] = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        cx = data["chunk_x"]
        cz = data["chunk_z"]
        dumps[(cx, cz)] = data

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge vanilla heightmaps and biomes from /dumpnoise JSON into training NPZ files.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing voxy_lod0_*.npz files (e.g. data/voxy/train).",
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
    args = parser.parse_args()

    # Load noise dumps
    dumps = load_noise_dumps(args.noise_dump_dir)

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

    # Check how many already have heightmap_surface
    sample = np.load(files[0])
    has_surface = "heightmap_surface" in sample
    if has_surface:
        print("WARNING: Files already contain heightmap_surface — will overwrite")

    if args.dry_run:
        print("Dry run — no files will be modified")
        # Still check coverage
        matched = 0
        missing = 0
        for f in files:
            m = re.search(r"x(-?\d+)_y(-?\d+)_z(-?\d+)", f)
            if m and (int(m.group(1)), int(m.group(3))) in dumps:
                matched += 1
            else:
                missing += 1
        print(f"  Would update {matched} files, {missing} have no matching JSON")
        return

    # Process each NPZ file
    updated = 0
    skipped = 0
    missing_chunks: list[tuple[int, int]] = []

    for fpath in tqdm(files, desc="Merging heightmaps"):
        m = re.search(r"x(-?\d+)_y(-?\d+)_z(-?\d+)", fpath)
        if not m:
            print(f"  Skipping (no coord in name): {fpath}")
            skipped += 1
            continue

        wx = int(m.group(1))  # section x = chunk x
        wz = int(m.group(3))  # section z = chunk z

        dump = dumps.get((wx, wz))
        if dump is None:
            missing_chunks.append((wx, wz))
            skipped += 1
            continue

        surface = parse_heightmap(dump["heightmap_surface"])
        ocean_floor = parse_heightmap(dump["heightmap_ocean_floor"])

        data = dict(np.load(fpath))
        data["heightmap_surface"] = surface
        data["heightmap_ocean_floor"] = ocean_floor

        # Overlay biome_patch if biome_names present in dump
        if "biome_names" in dump:
            data["biome_patch"] = parse_biome_names(dump["biome_names"])

        np.savez_compressed(fpath, **data)
        updated += 1

    print(f"\nUpdated {updated} NPZ files with vanilla heightmaps + biomes")
    if skipped:
        print(f"Skipped {skipped} files (no matching noise dump)")
    if missing_chunks:
        unique_missing = sorted(set(missing_chunks))
        print(f"  Missing chunks: {len(unique_missing)} unique (x,z) positions")
        if len(unique_missing) <= 10:
            for cx, cz in unique_missing:
                print(f"    chunk ({cx}, {cz})")
        else:
            for cx, cz in unique_missing[:5]:
                print(f"    chunk ({cx}, {cz})")
            print(f"    ... and {len(unique_missing) - 5} more")

    if updated == 0:
        print("\nERROR: No files were updated — check that noise dump coords match NPZ coords")
        sys.exit(1)

    # Verify a sample
    sample_file = files[0]
    d = np.load(sample_file)
    if "heightmap_surface" in d:
        hs = d["heightmap_surface"]
        print(f"\nVerification ({Path(sample_file).name}):")
        print(f"  heightmap_surface:     shape={hs.shape} min={hs.min():.1f} max={hs.max():.1f}")
        if "heightmap_ocean_floor" in d:
            hof = d["heightmap_ocean_floor"]
            print(
                f"  heightmap_ocean_floor: shape={hof.shape} min={hof.min():.1f} max={hof.max():.1f}"
            )
        if "biome_patch" in d:
            bp = d["biome_patch"]
            print(
                f"  biome_patch:           shape={bp.shape} unique={len(np.unique(bp))} "
                f"range=[{bp.min()},{bp.max()}]"
            )


if __name__ == "__main__":
    main()
