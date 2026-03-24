"""Build multi-level training pairs from v7 noise dumps + all Voxy LOD levels.

Each output sample is one 16-block noise section paired with ground-truth
labels at all 5 Voxy LOD levels (L0-L4), plus the 15-channel v7 RouterField
noise context, biome IDs, and per-column heightmaps.

Multi-level Voxy ground truth
-----------------------------
Voxy stores block data at 5 LOD levels (L0-L4), each as 32³ WorldSections:

  - L0: 1 block/voxel  (finest),  32-block WorldSections
  - L1: 2 blocks/voxel,           64-block WorldSections
  - L2: 4 blocks/voxel,           128-block WorldSections
  - L3: 8 blocks/voxel,           256-block WorldSections
  - L4: 16 blocks/voxel (coarsest), 512-block WorldSections

For a 16-block noise section, the ground truth at each level is:

  - L4: 1×1×1  = 1 voxel      (root of the sparse octree)
  - L3: 2×2×2  = 8 voxels
  - L2: 4×4×4  = 64 voxels
  - L1: 8×8×8  = 512 voxels
  - L0: 16×16×16 = 4096 voxels (block-level leaves)

Coordinate mapping (noise section → Voxy WorldSection at level L)::

    ws           = section_coord >> (L + 1)
    local_section = section_coord - (ws << (L + 1))   # 0..2^(L+1)-1
    n_voxels      = 16 >> L                             # per axis
    voxel_start   = local_section * n_voxels            # in 32³ grid

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

Output
------
  noise_3d       : (N, 15, 4, 2, 4)  float32 — all 15 v7 RouterField channels
  biome_ids      : (N, 4, 2, 4)      int32   — biome index per quart cell
  heightmap5     : (N, 5, 16, 16)    float32 — 5-plane heightmap
  block_y_min    : (N,)              int32   — sy * 16 (actual section block Y)
  labels_L0      : (N, 16, 16, 16)   int32   — Voxy L0 ground truth
  labels_L1      : (N, 8, 8, 8)      int32   — Voxy L1 ground truth
  labels_L2      : (N, 4, 4, 4)      int32   — Voxy L2 ground truth
  labels_L3      : (N, 2, 2, 2)      int32   — Voxy L3 ground truth
  labels_L4      : (N, 1, 1, 1)      int32   — Voxy L4 ground truth
  finest_level   : (N,)              int32   — finest Voxy level available (0-4)

Usage
-----
This module has no CLI entry point.  Invoke via the pipeline step runner::

  voxel-tree --step build_v7_pairs --run --profile NAME

All paths (dumps source, Voxy dir, output NPZ) are read from the named
profile YAML under ``data.v7_dumps_db`` / ``data.v7_dumps_dir``,
``data.data_dir``, and ``data.v7_pairs_npz``.

The public API is :func:`build_pairs`, which can also be imported directly
for use in tests or notebooks::

  from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (
      _DumpSourceSQLite,
      build_pairs,
  )
  build_pairs(_DumpSourceSQLite(db_path), voxy_dir, output_path)

"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

from voxel_tree.utils.coords import section_to_world_section

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_VT_ROOT = Path(__file__).resolve().parents[3]  # VoxelTree repo root
_DEFAULT_VOCAB_REMAP = _VT_ROOT / "voxel_tree" / "config" / "vocab_remap.json"


# ---------------------------------------------------------------------------
# Vocabulary remap
# ---------------------------------------------------------------------------


def _build_remap_lut(remap_path: Path | None) -> np.ndarray | None:
    """Build a numpy look-up table from ``vocab_remap.json``.

    The LUT maps old Voxy block IDs (0-1103) to new reduced IDs (0-512).
    Excluded blocks (remap value == -1) are mapped to 0 (air).

    Returns ``None`` when no remap file is found (labels pass through raw).
    """
    path = remap_path if remap_path is not None else _DEFAULT_VOCAB_REMAP
    if not path.exists():
        print(f"  Vocab remap not found at {path} — labels will use raw Voxy IDs")
        return None

    remap: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    max_old = max(int(k) for k in remap)
    lut = np.zeros(max_old + 1, dtype=np.int32)  # unmapped → 0 (air)
    for old_str, new_id in remap.items():
        lut[int(old_str)] = max(new_id, 0)  # -1 (excluded) → 0 (air)

    n_kept = sum(1 for v in remap.values() if v >= 0)
    n_excluded = sum(1 for v in remap.values() if v < 0)
    print(f"  Loaded vocab remap: {len(remap)} entries ({n_kept} kept, {n_excluded} → air)")
    return lut


def _apply_remap(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Remap block IDs in *labels* using the LUT, preserving -1 sentinels."""
    mask = labels >= 0
    clamped = np.clip(labels, 0, len(lut) - 1)
    return np.where(mask, lut[clamped], labels).astype(np.int32)


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

# Each section file covers exactly one (cx, sy, cz) triplet — 4×2×4 = 32 quart cells.
# Section Y range in Minecraft overworld: -4 to 19 inclusive (24 sections × 16 blocks = 384 blocks)


# ---------------------------------------------------------------------------
# SQLite loading helpers
# ---------------------------------------------------------------------------


def _unpack_noise_blob(blob: bytes) -> np.ndarray:
    """Unpack a noise BLOB (float32[480]) → (15, 4, 2, 4) float32."""
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    return arr.reshape(N_FIELDS, 4, 2, 4)


def _unpack_biome_blob(blob: bytes) -> np.ndarray:
    """Unpack a biome BLOB (int32[32]) → (4, 2, 4) int32."""
    arr = np.frombuffer(blob, dtype=np.int32).copy()
    return arr.reshape(4, 2, 4)


def _unpack_heightmap_blob(blob: bytes) -> np.ndarray:
    """Unpack a heightmap BLOB (int32[256]) → (16, 16) float32."""
    arr = np.frombuffer(blob, dtype=np.int32).copy()
    return arr.reshape(16, 16).astype(np.float32)


class _DumpSourceJSON:
    """Iterate noise dumps from a directory of JSON files."""

    def __init__(self, dumps_dir: Path):
        self.dump_files = sorted(dumps_dir.glob("section_*.json"))
        self._pattern = re.compile(r"section_(-?\d+)_(-?\d+)_(-?\d+)\.json$")
        if not self.dump_files:
            raise FileNotFoundError(f"No section_*.json files in {dumps_dir}")
        print(f"  Found {len(self.dump_files):,} JSON dump files")

    def __len__(self) -> int:
        return len(self.dump_files)

    def __iter__(self):
        for dump_path in self.dump_files:
            m = self._pattern.search(dump_path.name)
            if not m:
                continue
            cx, sy, cz = int(m.group(1)), int(m.group(2)), int(m.group(3))
            with open(dump_path) as f:
                raw = json.load(f)

            # Noise: 15 fields × 32 → (15, 4, 2, 4)
            field_arrays = []
            for field in NOISE_FIELDS:
                field_arrays.append(np.array(raw[field], dtype=np.float32).reshape(4, 2, 4))
            noise = np.stack(field_arrays)

            biome = np.array(raw["biome_ids"], dtype=np.int32).reshape(4, 2, 4)
            hm_surface = np.array(raw["heightmap_surface"], dtype=np.float32).reshape(16, 16)
            hm_ocean = np.array(raw["heightmap_ocean_floor"], dtype=np.float32).reshape(16, 16)

            yield cx, sy, cz, noise, biome, hm_surface, hm_ocean


class _DumpSourceSQLite:
    """Iterate noise dumps from a consolidated SQLite database.

    Supports optional spatial pre-filtering via *section_bounds* to avoid
    streaming millions of rows that fall outside the Voxy coverage area.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        section_bounds: dict[str, tuple[int, int]] | None = None,
    ):
        self.db_path = db_path
        self._bounds = section_bounds  # {"cx": (lo, hi), "sy": …, "cz": …}

        conn = sqlite3.connect(str(db_path))
        if section_bounds:
            where = self._where_clause()
            (self._count,) = conn.execute(f"SELECT COUNT(*) FROM sections WHERE {where}").fetchone()
        else:
            (self._count,) = conn.execute("SELECT COUNT(*) FROM sections").fetchone()
        conn.close()
        bounds_tag = "  (pre-filtered)" if section_bounds else ""
        print(f"  SQLite DB: {db_path} ({self._count:,} sections){bounds_tag}")

    # ── helpers ───────────────────────────────────────────────────────────

    def _where_clause(self) -> str:
        clauses: list[str] = []
        if self._bounds:
            for col, dbcol in [("cx", "chunk_x"), ("sy", "section_y"), ("cz", "chunk_z")]:
                if col in self._bounds:
                    lo, hi = self._bounds[col]
                    clauses.append(f"{dbcol} BETWEEN {lo} AND {hi}")
        return " AND ".join(clauses) if clauses else "1"

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA cache_size=-256000")  # 256 MB

        # Pre-load all heightmaps into a dict for fast lookup
        hm_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for cx, cz, surf_blob, ocean_blob in conn.execute(
            "SELECT chunk_x, chunk_z, surface, ocean_floor FROM heightmaps"
        ):
            hm_cache[(cx, cz)] = (
                _unpack_heightmap_blob(surf_blob),
                _unpack_heightmap_blob(ocean_blob),
            )
        print(f"  Loaded {len(hm_cache):,} heightmaps into cache")

        # Stream sections (with optional spatial pre-filter)
        where = self._where_clause() if self._bounds else "1"
        cursor = conn.execute(
            f"SELECT chunk_x, section_y, chunk_z, noise_data, biome_ids "
            f"FROM sections WHERE {where} "
            f"ORDER BY chunk_x, chunk_z, section_y"
        )
        for cx, sy, cz, noise_blob, biome_blob in cursor:
            noise = _unpack_noise_blob(noise_blob)
            biome = _unpack_biome_blob(biome_blob)
            hm_surface, hm_ocean = hm_cache.get(
                (cx, cz),
                (np.zeros((16, 16), dtype=np.float32), np.zeros((16, 16), dtype=np.float32)),
            )
            yield cx, sy, cz, noise, biome, hm_surface, hm_ocean

        conn.close()


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


def extract_section_subcube(
    labels32: np.ndarray,
    sx: int,
    sy: int,
    sz: int,
    level: int,
) -> np.ndarray:
    """Extract the per-level sub-cube for a player-section from a 32³ Voxy grid.

    For section coords (sx, sy, sz) at Voxy level L::

        WorldSection:   ws = s >> (L + 1)
        Local section:  ls = s - (ws << (L + 1))     # 0..2^(L+1)-1
        Voxels/axis:    n  = 16 >> L
        Voxel start:    vs = ls * n                    # in 32³ grid

    Args:
        labels32: (32, 32, 32) Voxy grid in (y, z, x) order.
        sx, sy, sz: Player-section coordinates (chunk x, section y, chunk z).
        level: Voxy level (0-4).

    Returns:
        ndarray of shape (n, n, n) in (y, z, x) order, where n = 16 >> level.
    """
    n = 16 >> level  # voxels per axis at this level

    def _voxel_range(s: int) -> tuple[int, int]:
        ws = s >> (level + 1)
        ls = s - (ws << (level + 1))
        vs = ls * n
        return vs, vs + n

    vy0, vy1 = _voxel_range(sy)
    vz0, vz1 = _voxel_range(sz)
    vx0, vx1 = _voxel_range(sx)

    return labels32[vy0:vy1, vz0:vz1, vx0:vx1].astype(np.int32)


def compute_voxy_section_bounds(
    voxy_dir: Path,
) -> dict[str, tuple[int, int]] | None:
    """Compute the tightest section-coordinate bounds from Voxy data.

    Scans all 5 levels and returns the intersection, i.e. the section coords
    that *could* have all 5 levels available.  Useful for pre-filtering a
    SQLite dump to avoid iterating millions of irrelevant rows.

    Returns ``{"cx": (lo, hi), "sy": (lo, hi), "cz": (lo, hi)}`` or ``None``
    if any level directory is missing.
    """
    bounds_per_axis: dict[str, list[tuple[int, int]]] = {"cx": [], "sy": [], "cz": []}

    for level in range(5):
        level_dir = voxy_dir / f"level_{level}"
        if not level_dir.is_dir():
            return None
        pat = re.compile(rf"(?:w\d+_)?voxy_L{level}_x(-?\d+)_y(-?\d+)_z(-?\d+)\.npz$")
        xs, ys, zs = [], [], []
        for f in level_dir.iterdir():
            m = pat.search(f.name)
            if m:
                xs.append(int(m.group(1)))
                ys.append(int(m.group(2)))
                zs.append(int(m.group(3)))
        if not xs:
            return None
        shift = level + 1
        # Convert world-section range to section-coordinate range
        for axis, vals in [("cx", xs), ("sy", ys), ("cz", zs)]:
            lo_section = min(vals) << shift
            hi_section = ((max(vals) + 1) << shift) - 1
            bounds_per_axis[axis].append((lo_section, hi_section))

    # Intersection: take the tightest (max of lows, min of highs)
    result: dict[str, tuple[int, int]] = {}
    for axis in ("cx", "sy", "cz"):
        lo = max(lo for lo, _ in bounds_per_axis[axis])
        hi = min(hi for _, hi in bounds_per_axis[axis])
        if lo > hi:
            return None  # empty intersection — should never happen with real data
        result[axis] = (lo, hi)
    return result


def build_voxy_indices(
    voxy_dir: Path,
    levels: list[int] | None = None,
) -> dict[int, dict[tuple[int, int, int], Path]]:
    """Build (x, y, z) → Path index for Voxy sections at each LOD level.

    Args:
        voxy_dir: Root Voxy data directory containing level_0/ through level_4/.
        levels: Which levels to index (default: all five).

    Returns:
        Dict mapping level → {(ws_x, ws_y, ws_z): Path}.
    """
    if levels is None:
        levels = [0, 1, 2, 3, 4]

    indices: dict[int, dict[tuple[int, int, int], Path]] = {}
    for level in levels:
        level_dir = voxy_dir / f"level_{level}"
        if not level_dir.is_dir():
            print(f"  Warning: Voxy level_{level} directory not found: {level_dir}")
            indices[level] = {}
            continue

        # Match both old (voxy_L4_...) and new (w0_voxy_L4_...) naming.
        pat = re.compile(rf"(?:w\d+_)?voxy_L{level}_x(-?\d+)_y(-?\d+)_z(-?\d+)\.npz$")
        index: dict[tuple[int, int, int], Path] = {}
        for f in level_dir.iterdir():
            m = pat.search(f.name)
            if m:
                x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
                index[(x, y, z)] = f

        indices[level] = index
        print(f"  Indexed {len(index):,} Voxy L{level} sections from {level_dir}")

    return indices


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_pairs(
    dump_source: _DumpSourceJSON | _DumpSourceSQLite,
    voxy_dir: Path,
    output_path: Path,
    *,
    require_all_levels: bool = True,
    vocab_remap_lut: np.ndarray | None = None,
) -> tuple[int, dict[str, int]]:
    """Build and save multi-level sparse-octree training pairs.

    For each noise dump at section (cx, sy, cz), extracts the corresponding
    sub-cube from Voxy at every available LOD level (L0-L4).  One training
    sample per dump — no octant fan-out.

    Args:
        dump_source: A _DumpSourceJSON or _DumpSourceSQLite iterable.
        require_all_levels: When True (default), only include samples where
            all 5 Voxy levels are available.  When False, include partial
            samples and mark missing levels with -1 sentinel.

    Returns:
        (pairs_saved, stats_dict).
    """
    n_dumps = len(dump_source)
    print(f"  Processing {n_dumps:,} dump sections")

    # ── Index all 5 Voxy LOD levels ──────────────────────────────────────
    voxy_indices = build_voxy_indices(voxy_dir)

    # ── Per-level label accumulators ─────────────────────────────────────
    # {level: list of ndarrays}, shapes: L4=(1,1,1), L3=(2,2,2), ..., L0=(16,16,16)
    label_lists: dict[int, list[np.ndarray]] = {lv: [] for lv in range(5)}
    noise_slices: list[np.ndarray] = []  # (15, 4, 2, 4) float32
    biome_slices: list[np.ndarray] = []  # (4, 2, 4) int32
    hm5_slices: list[np.ndarray] = []  # (5, 16, 16) float32
    block_y_min_list: list[int] = []  # sy * 16
    finest_level_list: list[int] = []  # finest available Voxy level

    # Voxy grid cache: (level, ws_tuple) → labels32 (32,32,32)
    voxy_cache: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {}

    skipped_no_voxy = 0
    skipped_partial = 0
    processed = 0
    progress_interval = max(1, n_dumps // 20)  # ~5% increments

    for cx, sy, cz, noise_block, biome_arr, hm_surface, hm_ocean in dump_source:
        processed += 1
        if processed % progress_interval == 0:
            pct = 100.0 * processed / n_dumps
            matched = len(noise_slices)
            print(
                f"  [{pct:5.1f}%] {processed:,}/{n_dumps:,} processed, "
                f"{matched:,} matched, {skipped_no_voxy:,} no-voxy, "
                f"{skipped_partial:,} partial",
                flush=True,
            )
        # ── Check Voxy availability at each level ────────────────────────
        available_ws: dict[int, tuple[int, int, int]] = {}
        for level in range(5):
            ws = (
                section_to_world_section(cx, level),
                section_to_world_section(sy, level),
                section_to_world_section(cz, level),
            )
            if ws in voxy_indices.get(level, {}):
                available_ws[level] = ws

        if not available_ws:
            skipped_no_voxy += 1
            continue

        if require_all_levels and len(available_ws) < 5:
            skipped_partial += 1
            continue

        finest = min(available_ws.keys())

        # Heightmaps → 5-plane
        hm5 = compute_height_planes(hm_surface, hm_ocean)  # (5, 16, 16)

        # ── Extract per-level sub-cubes from Voxy ────────────────────────
        level_grids: dict[int, np.ndarray] = {}
        for level, ws in available_ws.items():
            cache_key = (level, ws)
            if cache_key not in voxy_cache:
                with np.load(voxy_indices[level][ws]) as vf:
                    voxy_cache[cache_key] = vf["labels32"]  # (32, 32, 32)
            labels32 = voxy_cache[cache_key]
            level_grids[level] = extract_section_subcube(labels32, cx, sy, cz, level)

        # ── Accumulate sample ────────────────────────────────────────────
        noise_slices.append(noise_block)
        biome_slices.append(biome_arr)
        hm5_slices.append(hm5)
        block_y_min_list.append(sy * 16)  # actual section block Y
        finest_level_list.append(finest)

        for lv in range(5):
            if lv in level_grids:
                label_lists[lv].append(level_grids[lv])
            else:
                # Sentinel for missing levels.
                n = 16 >> lv
                label_lists[lv].append(np.full((n, n, n), -1, dtype=np.int32))

    if not noise_slices:
        print("ERROR: No pairs generated — check that dumps_dir and voxy_dir overlap.")
        print("  Hint: ensure Voxy data exists at all 5 levels (level_0/ .. level_4/)")
        print("        for the spatial region covered by the noise dumps.")
        sys.exit(1)

    n = len(noise_slices)
    print(f"  Total samples: {n:,} (1 per noise dump)")
    if skipped_no_voxy:
        print(f"  Skipped (no Voxy at any level):   {skipped_no_voxy:,}")
    if skipped_partial:
        print(f"  Skipped (missing some levels):     {skipped_partial:,}")
    level_counts = {lv: sum(1 for a in label_lists[lv] if a.min() >= 0) for lv in range(5)}
    for lv in range(5):
        side = 16 >> lv
        print(f"  L{lv} coverage: {level_counts[lv]:,}/{n:,} samples ({side}³ per sample)")

    # ── Stack and save ───────────────────────────────────────────────────
    all_noise_3d = np.stack(noise_slices).astype(np.float32)  # (N, 15, 4, 2, 4)
    all_biome_ids = np.stack(biome_slices).astype(np.int32)  # (N, 4, 2, 4)
    all_hm5 = np.stack(hm5_slices).astype(np.float32)  # (N, 5, 16, 16)
    all_block_y_min = np.array(block_y_min_list, dtype=np.int32)  # (N,)
    all_finest = np.array(finest_level_list, dtype=np.int32)  # (N,)

    save_dict: dict[str, np.ndarray] = {
        "noise_3d": all_noise_3d,
        "biome_ids": all_biome_ids,
        "heightmap5": all_hm5,
        "block_y_min": all_block_y_min,
        "finest_level": all_finest,
    }
    for lv in range(5):
        save_dict[f"labels_L{lv}"] = np.stack(label_lists[lv]).astype(np.int32)

    # ── Apply vocabulary remap (1104 → 513) ──────────────────────────────
    if vocab_remap_lut is not None:
        raw_max = max(
            int(save_dict[f"labels_L{lv}"][save_dict[f"labels_L{lv}"] >= 0].max())
            for lv in range(5)
            if (save_dict[f"labels_L{lv}"] >= 0).any()
        )
        for lv in range(5):
            save_dict[f"labels_L{lv}"] = _apply_remap(save_dict[f"labels_L{lv}"], vocab_remap_lut)
        new_max = max(
            int(save_dict[f"labels_L{lv}"][save_dict[f"labels_L{lv}"] >= 0].max())
            for lv in range(5)
            if (save_dict[f"labels_L{lv}"] >= 0).any()
        )
        print(f"  Vocab remap applied: max block ID {raw_max} → {new_max}")

    print(f"  Noise channels: {all_noise_3d.shape[1]} (v7 RouterField)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved -> {output_path}  ({size_mb:.1f} MB)")

    failure_stats: dict[str, int] = {
        "pairs_saved": n,
        "total_dump_files": n_dumps,
        "skipped_no_voxy": skipped_no_voxy,
        "skipped_partial": skipped_partial,
        "all_5_levels": level_counts.get(0, 0),
    }
    for lv in range(5):
        failure_stats[f"L{lv}_coverage"] = level_counts[lv]
    return n, failure_stats
