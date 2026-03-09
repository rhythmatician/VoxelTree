#!/usr/bin/env python3
"""extract_voxy_training_data.py — Extract training data from Voxy RocksDB.

Reads LOD-0 WorldSections from a Voxy database, converts per-world Voxy
block IDs to canonical Voxy vocabulary IDs, computes biomes and heightmaps,
and saves as NPZ files compatible with ``MultiLODDataset``.

Usage::

    python scripts/extract_voxy_training_data.py \\
        "path/to/voxy/<hash>/storage" \\
        --output-dir data/voxy \\
        --min-solid 0.02

The output NPZ files contain:
    labels16          (16, 16, 16) int32  — canonical Voxy vocabulary IDs
    biome_patch       (16, 16)    int32   — biome IDs (column-wise majority)
    heightmap_patch   (16, 16)    float32 — placeholder (replaced by add_column_heights.py)
    y_index           scalar      int64   — 0-based Y index (section_y + 4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Minecraft section Y base — matches LODiffusion's Y_BASE_SECTION = -4
Y_BASE_SECTION = -4

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.voxy_reader import VoxyReader


# ------------------------------------------------------------------
# Vocabulary helpers
# ------------------------------------------------------------------

DEFAULT_VOCAB_PATH = Path("config/voxy_vocab.json")


def build_voxy_vocab_from_worlds(db_paths: list[str], out_path: Path) -> dict[str, int]:
    """Scan Voxy worlds and build a canonical vocabulary from all unique block names.

    Writes the vocabulary to *out_path* and returns the mapping.
    Air is always assigned ID 0.
    """
    all_names: set[str] = set()
    for p in db_paths:
        try:
            with VoxyReader(p) as r:
                all_names.update(r._block_names.values())
        except Exception as exc:
            print("  Warning: could not read %s: %s" % (p, exc))

    all_names.discard("minecraft:air")
    sorted_names = sorted(all_names)

    vocab: dict[str, int] = {"minecraft:air": 0}
    for i, name in enumerate(sorted_names, start=1):
        vocab[name] = i

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    print("Built Voxy vocabulary: %d entries → %s" % (len(vocab), out_path))
    return vocab


def load_vocab(path: Path) -> dict[str, int]:
    """Load a canonical Voxy vocabulary from JSON."""
    with open(path) as f:
        return json.load(f)


def build_world_lut(reader: VoxyReader, vocab: dict[str, int]) -> npt.NDArray[np.int32]:
    """Build a per-world LUT mapping Voxy state IDs → canonical vocab IDs.

    Each Voxy world assigns its own sequential IDs to block states.  This
    function maps them to the canonical vocabulary by matching block names.
    """
    max_id = max(reader._block_names.keys()) if reader._block_names else 0
    lut = np.zeros(max_id + 1, dtype=np.int32)  # unmapped → 0 (air)

    mapped = 0
    for voxy_id, name in reader._block_names.items():
        if name in vocab:
            lut[voxy_id] = vocab[name]
            mapped += 1
        else:
            # Try base name without properties (shouldn't happen since Voxy
            # already strips properties, but just in case)
            base = name.split("[")[0] if "[" in name else name
            lut[voxy_id] = vocab.get(base, 0)
            if base in vocab:
                mapped += 1

    return lut


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def _compute_heightmap(block_ids: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Compute a normalised heightmap from a (16, 16, 16) block-ID volume.

    The volume is indexed as (y, z, x).  For each (z, x) column, the
    heightmap value is the highest Y coordinate that contains a non-air
    block, normalised to [0, 1] (0 = bottom, 1 = top).

    Returns (16, 16) float32 in (z, x) order.
    """
    occ = block_ids > 0  # (y, z, x) bool
    height = np.zeros((16, 16), dtype=np.float32)
    for z in range(16):
        for x in range(16):
            col = occ[:, z, x]
            nz = np.nonzero(col)[0]
            if len(nz) > 0:
                height[z, x] = (nz[-1] + 1) / 16.0
    return height


def _compute_biome_2d(biome_ids: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Collapse 3-D biome volume (y, z, x) to 2-D (z, x) via column majority.

    Returns (16, 16) int32.
    """
    result = np.zeros((16, 16), dtype=np.int32)
    for z in range(16):
        for x in range(16):
            col = biome_ids[:, z, x]
            vals, counts = np.unique(col, return_counts=True)
            result[z, x] = vals[np.argmax(counts)]
    return result


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------


def extract(
    db_path: str,
    output_dir: Path,
    vocab: dict[str, int],
    *,
    min_solid_fraction: float = 0.02,
    max_sections: int | None = None,
    prefix: str = "",
) -> int:
    """Extract training NPZ files from a Voxy RocksDB database.

    Each NPZ includes:
      - labels16: canonical vocab IDs
      - biome_patch: 2D biome IDs
      - heightmap_patch: placeholder (will be replaced by add_column_heights.py)
      - y_index: section Y coordinate
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with VoxyReader(db_path) as reader:
        print(reader.summary())
        print()

        # Build per-world LUT: Voxy state ID → canonical vocab ID
        lut = build_world_lut(reader, vocab)
        n_mapped = int(np.sum(lut > 0))  # exclude air-to-air
        print(
            "Block ID mapping: %d/%d Voxy states → canonical vocab (%d entries)"
            % (n_mapped, len(reader._block_names), len(vocab))
        )
        print()

        total_subblocks = 0
        saved = 0
        skipped_empty = 0
        section_count = 0

        for section in reader.iter_sections(level=0):
            if max_sections is not None and section_count >= max_sections:
                break
            section_count += 1

            for sub in reader.section_to_subblocks(section):
                total_subblocks += 1
                blk_voxy = sub["block_ids"]  # (16,16,16) — (y,z,x), per-world IDs
                bio_voxy = sub["biome_ids"]  # (16,16,16)

                # Convert per-world Voxy IDs → canonical vocab IDs
                max_id = lut.shape[0] - 1
                blk_canon = lut[np.clip(blk_voxy, 0, max_id)]

                solid_frac = float(np.mean(blk_canon > 0))
                if solid_frac < min_solid_fraction:
                    skipped_empty += 1
                    continue

                biome_2d = _compute_biome_2d(bio_voxy)
                heightmap = _compute_heightmap(blk_canon)
                # 0-based y index: section Y -4 → 0, -3 → 1, ..., 11 → 15
                # Matches inference: yIndex = sectionY - Y_BASE_SECTION
                y_index = sub["world_y"] - Y_BASE_SECTION

                wx, wy, wz = sub["world_x"], sub["world_y"], sub["world_z"]
                fname = "%svoxy_lod0_x%d_y%d_z%d.npz" % (prefix, wx, wy, wz)

                np.savez_compressed(
                    output_dir / fname,
                    labels16=blk_canon.astype(np.int32),
                    biome_patch=biome_2d.astype(np.int32),
                    heightmap_patch=heightmap.astype(np.float32),
                    y_index=np.int64(y_index),
                )
                saved += 1

            if section_count % 500 == 0:
                print(
                    "  ... processed %d sections, %d sub-blocks, %d saved"
                    % (section_count, total_subblocks, saved)
                )

    print()
    print("Extraction complete:")
    print("  LOD-0 sections processed: %d" % section_count)
    print("  Total 16³ sub-blocks: %d" % total_subblocks)
    print("  Saved (solid >= %.0f%%): %d" % (min_solid_fraction * 100, saved))
    print("  Skipped (near-empty): %d" % skipped_empty)
    print("  Output directory: %s" % output_dir)
    return saved


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract training data from Voxy RocksDB databases.",
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
        default=Path("data/voxy"),
        help="Output directory for NPZ files (default: data/voxy).",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Path to Voxy vocabulary JSON (default: config/voxy_vocab.json). "
        "If it doesn't exist, one will be built from the input databases.",
    )
    parser.add_argument(
        "--min-solid",
        type=float,
        default=0.02,
        help="Minimum solid fraction to keep a sub-block (default: 0.02).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Max LOD-0 sections to process per database (for testing).",
    )
    args = parser.parse_args()

    # Load or build vocabulary
    if args.vocab.exists():
        vocab = load_vocab(args.vocab)
        print("Loaded Voxy vocabulary: %d entries from %s" % (len(vocab), args.vocab))
    else:
        print("Vocabulary not found at %s — building from input databases..." % args.vocab)
        vocab = build_voxy_vocab_from_worlds(args.db_path, args.vocab)

    # Extract
    total = 0
    for i, db_path in enumerate(args.db_path):
        prefix = "w%d_" % i if len(args.db_path) > 1 else ""
        print("=" * 60)
        print("Database %d/%d: %s" % (i + 1, len(args.db_path), db_path))
        print("=" * 60)
        total += extract(
            db_path=db_path,
            output_dir=args.output_dir,
            vocab=vocab,
            min_solid_fraction=args.min_solid,
            max_sections=args.max_sections,
            prefix=prefix,
        )
        print()

    if len(args.db_path) > 1:
        print("Grand total: %d training chunks from %d worlds" % (total, len(args.db_path)))


if __name__ == "__main__":
    main()
