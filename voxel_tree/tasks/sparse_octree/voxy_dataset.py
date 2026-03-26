"""Per-level tiling dataset for Voxy-native models.

Tiles noise, biome, and heightmap data from the v7 noise-dumps DB into
WorldSection-sized inputs for each Voxy LOD level.

**Golden Rule — Input Resolution Invariant**:
Inputs never exceed vanilla Minecraft's native sampling resolution.
L0 = exactly vanilla's grid.  L1–L4 ⊆ vanilla (strict subset).

A Voxy WorldSection at level L covers ``32 × 2^L`` blocks per axis, which
spans ``2^(L+1)`` noise sections in each dimension.  This dataset:

1. Discovers all WorldSection coordinates at a given level that have
   **complete** coverage in the noise-dumps DB (every constituent section
   present) AND a matching Voxy ground-truth grid.
2. On ``__getitem__``, tiles the constituent noise/biome/heightmap blobs
   into the native-resolution tensors expected by :class:`VoxyL{level}Model`.
3. Returns ``(inputs_dict, labels_32)`` where ``labels_32`` is the Voxy
   ``int32[32, 32, 32]`` ground-truth voxel grid.

Coordinate conventions
----------------------
- Section DB uses ``(chunk_x, section_y, chunk_z)`` — 16-block aligned.
- ``section_to_world_section(coord, level) = coord >> (level + 1)``.
- Voxy WorldSection DB uses ``(level, ws_x, ws_y, ws_z)``.
- Heightmap DB uses ``(chunk_x, chunk_z)`` — one entry per 16×16 chunk column.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .build_sparse_octree_pairs import (
    N_FIELDS,
    _unpack_biome_blob,
    _unpack_heightmap_blob,
    _unpack_noise_blob,
    compute_height_planes,
)
from .voxy_models import L2_NOISE_CHANNELS, L3_NOISE_CHANNELS, L4_NOISE_CHANNELS

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════

# Noise cell grid per 16-block section: 4 × 2 × 4 (X, Y, Z)
_NOISE_QX = 4
_NOISE_QY = 2
_NOISE_QZ = 4

# Block-resolution heightmap per chunk
_HM_BLOCK_RES = 16

# Subsampled grid size for L2–L4 2D inputs
_SUBSAMPLE_XZ = 8

# Feature selection: channel indices per level
_LEVEL_NOISE_CHANNELS = {
    2: L2_NOISE_CHANNELS,
    3: L3_NOISE_CHANNELS,
    4: L4_NOISE_CHANNELS,
}

# Vanilla density cell width (blocks per cell XZ)
_CELL_WIDTH = 4
# Vanilla density cell height (blocks per cell Y)
_CELL_HEIGHT = 8


# ══════════════════════════════════════════════════════════════════════
#  Unpack helpers (re-exported for convenience)
# ══════════════════════════════════════════════════════════════════════


def _unpack_voxy_blob(blob: bytes) -> np.ndarray:
    """Decompress a zlib-compressed int32[32,32,32] Voxy grid."""
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=np.int32).reshape(32, 32, 32).copy()


# ══════════════════════════════════════════════════════════════════════
#  Sample discovery
# ══════════════════════════════════════════════════════════════════════


def _discover_complete_world_sections(
    conn: sqlite3.Connection,
    level: int,
    min_coverage: float = 1.0,
) -> List[Tuple[int, int, int]]:
    """Find all WorldSection coordinates at ``level`` with sufficient noise-DB coverage.

    A WorldSection at level L tiles ``T × T × T`` sections where ``T = 2^(L+1)``.
    Minecraft's overworld Y range is only 24 sections (-4..19), so for coarser
    levels the actual Y-span per WorldSection may be less than T.  We compute
    the realistic expected count per WS by clamping the Y range.

    We require at least ``min_coverage`` fraction of the *realistic* expected
    sections to exist in the ``sections`` table AND a matching row in
    ``voxy_sections``.

    Returns a list of ``(ws_x, ws_y, ws_z)`` tuples.
    """
    shift = level + 1
    T = 1 << shift

    # Determine actual Y extent in section coordinates from the DB
    y_range = conn.execute(
        "SELECT MIN(section_y), MAX(section_y) FROM sections"
    ).fetchone()
    if y_range[0] is None:
        return []
    y_min_global, y_max_global = y_range  # e.g. -4, 19

    # Group sections by their WorldSection coordinate
    rows = conn.execute(
        """
        SELECT chunk_x >> :shift AS ws_x,
               section_y >> :shift AS ws_y,
               chunk_z >> :shift AS ws_z,
               COUNT(*) AS cnt
        FROM sections
        GROUP BY ws_x, ws_y, ws_z
        """,
        {"shift": shift},
    ).fetchall()

    candidates = []
    for ws_x, ws_y, ws_z, cnt in rows:
        # Compute how many Y-sections actually exist in this WS's Y range
        ws_y_lo = ws_y * T       # first section_y in this WS
        ws_y_hi = ws_y_lo + T    # exclusive upper bound
        # Clamp to the global Y extent
        actual_y = max(0, min(ws_y_hi, y_max_global + 1) - max(ws_y_lo, y_min_global))
        expected = T * actual_y * T  # X × Y_actual × Z
        if expected == 0:
            continue
        threshold = int(expected * min_coverage)
        if cnt >= threshold:
            candidates.append((ws_x, ws_y, ws_z))

    if not candidates:
        return []

    # Filter to those that also have a Voxy ground-truth grid
    has_voxy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='voxy_sections'"
    ).fetchone()
    if not has_voxy:
        log.warning("No voxy_sections table found — cannot build training samples.")
        return []

    valid = []
    for ws_x, ws_y, ws_z in candidates:
        row = conn.execute(
            "SELECT 1 FROM voxy_sections WHERE level=? AND ws_x=? AND ws_y=? AND ws_z=?",
            (level, ws_x, ws_y, ws_z),
        ).fetchone()
        if row is not None:
            valid.append((ws_x, ws_y, ws_z))

    return valid


# ══════════════════════════════════════════════════════════════════════
#  Tiling logic
# ══════════════════════════════════════════════════════════════════════


def _tile_noise_and_biome(
    conn: sqlite3.Connection,
    ws_x: int,
    ws_y: int,
    ws_z: int,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tile noise and biome from constituent sections into WorldSection tensors.

    Args:
        conn: SQLite connection to noise-dumps DB.
        ws_x, ws_y, ws_z: WorldSection coordinates.
        level: Voxy LOD level.

    Returns:
        noise_3d: ``float32[15, Tx*4, Ty*2, Tz*4]``
        biome_3d: ``int32[Tx*4, Ty*2, Tz*4]``

        where ``T = 2^(level+1)`` = sections per axis.
    """
    shift = level + 1
    T = 1 << shift

    # Output shapes
    noise_out = np.zeros((N_FIELDS, T * _NOISE_QX, T * _NOISE_QY, T * _NOISE_QZ), dtype=np.float32)
    biome_out = np.zeros((T * _NOISE_QX, T * _NOISE_QY, T * _NOISE_QZ), dtype=np.int32)

    # Coordinate ranges for constituent sections
    cx_base = ws_x << shift
    sy_base = ws_y << shift
    cz_base = ws_z << shift

    rows = conn.execute(
        """
        SELECT chunk_x, section_y, chunk_z, noise_data, biome_ids
        FROM sections
        WHERE chunk_x >= ? AND chunk_x < ?
          AND section_y >= ? AND section_y < ?
          AND chunk_z >= ? AND chunk_z < ?
        """,
        (cx_base, cx_base + T, sy_base, sy_base + T, cz_base, cz_base + T),
    ).fetchall()

    for cx, sy, cz, noise_blob, biome_blob in rows:
        dx = cx - cx_base
        dy = sy - sy_base
        dz = cz - cz_base

        noise = _unpack_noise_blob(noise_blob)  # (15, 4, 2, 4)
        biome = _unpack_biome_blob(biome_blob)  # (4, 2, 4)

        # Place into tiled array
        x0, x1 = dx * _NOISE_QX, (dx + 1) * _NOISE_QX
        y0, y1 = dy * _NOISE_QY, (dy + 1) * _NOISE_QY
        z0, z1 = dz * _NOISE_QZ, (dz + 1) * _NOISE_QZ

        noise_out[:, x0:x1, y0:y1, z0:z1] = noise
        biome_out[x0:x1, y0:y1, z0:z1] = biome

    return noise_out, biome_out


def _tile_heightmap(
    conn: sqlite3.Connection,
    ws_x: int,
    ws_z: int,
    level: int,
) -> np.ndarray:
    """Tile heightmaps from constituent chunk columns into WorldSection 5-plane.

    Args:
        conn: SQLite connection to noise-dumps DB.
        ws_x, ws_z: WorldSection XZ coordinates.
        level: Voxy LOD level.

    Returns:
        ``float32[5, H, W]`` where ``H = W = 2^(level+1) * 4`` (at cell resolution).
    """
    shift = level + 1
    T = 1 << shift  # chunks per axis in this WorldSection

    # We tile at block resolution (16 per chunk) then downsample to cell resolution
    block_res = T * _HM_BLOCK_RES  # total blocks per axis
    surface_full = np.zeros((block_res, block_res), dtype=np.float32)
    ocean_full = np.zeros((block_res, block_res), dtype=np.float32)

    cx_base = ws_x << shift
    cz_base = ws_z << shift

    rows = conn.execute(
        """
        SELECT chunk_x, chunk_z, surface, ocean_floor
        FROM heightmaps
        WHERE chunk_x >= ? AND chunk_x < ?
          AND chunk_z >= ? AND chunk_z < ?
        """,
        (cx_base, cx_base + T, cz_base, cz_base + T),
    ).fetchall()

    for cx, cz, surface_blob, ocean_blob in rows:
        dx = cx - cx_base
        dz = cz - cz_base

        hm_s = _unpack_heightmap_blob(surface_blob)  # (16, 16) float32
        hm_o = _unpack_heightmap_blob(ocean_blob)

        x0, x1 = dx * _HM_BLOCK_RES, (dx + 1) * _HM_BLOCK_RES
        z0, z1 = dz * _HM_BLOCK_RES, (dz + 1) * _HM_BLOCK_RES

        surface_full[x0:x1, z0:z1] = hm_s
        ocean_full[x0:x1, z0:z1] = hm_o

    # Downsample to density-cell resolution (stride 4)
    surface_ds = surface_full[::_CELL_WIDTH, ::_CELL_WIDTH].copy()
    ocean_ds = ocean_full[::_CELL_WIDTH, ::_CELL_WIDTH].copy()

    # Compute 5-plane features
    return compute_height_planes(surface_ds, ocean_ds)


def _select_climate_2d(
    noise_3d: np.ndarray,
    biome_3d: np.ndarray,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature-select and collapse 3D noise/biome to 2D for L2–L4.

    1. Select only the relevant noise channels for this level.
    2. Average across the Y axis (dim 2 of ``[C, Nx, Ny, Nz]``).
    3. Subsample XZ to 8×8 using strided indexing.
    4. Biome: take the Y=0 slice (biomes are Y-invariant), subsample to 8×8.

    Args:
        noise_3d: ``float32[15, Nx, Ny, Nz]`` — full tiled noise.
        biome_3d: ``int32[Nx, Ny, Nz]`` — full tiled biome.
        level: Voxy LOD level (2, 3, or 4).

    Returns:
        climate_2d: ``float32[C_sel, 8, 8]`` — selected channels, 2D.
        biome_2d:   ``int32[8, 8]`` — biome IDs, 2D.
    """
    channels = _LEVEL_NOISE_CHANNELS[level]

    # Channel selection + Y-axis average
    selected = noise_3d[channels]  # [C_sel, Nx, Ny, Nz]
    climate_yz_avg = selected.mean(axis=2)  # [C_sel, Nx, Nz]

    # Subsample XZ to 8×8
    nx, nz = climate_yz_avg.shape[1], climate_yz_avg.shape[2]
    sx = max(1, nx // _SUBSAMPLE_XZ)
    sz = max(1, nz // _SUBSAMPLE_XZ)
    climate_2d = climate_yz_avg[:, ::sx, ::sz][:, :_SUBSAMPLE_XZ, :_SUBSAMPLE_XZ].copy()

    # Biome: take Y=0 slice (biomes are Y-invariant), subsample
    biome_flat = biome_3d[:, 0, :]  # [Nx, Nz]
    biome_2d = biome_flat[::sx, ::sz][:_SUBSAMPLE_XZ, :_SUBSAMPLE_XZ].copy()

    return climate_2d, biome_2d


# ══════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════


class VoxyLevelDataset(Dataset):  # type: ignore[type-arg]
    """Per-level tiling dataset for Voxy-native models.

    Each sample is a complete WorldSection at the specified Voxy LOD level.
    The dataset tiles noise/biome/heightmap from the constituent 16-block
    sections and retrieves the matching Voxy ground-truth grid.

    Thread-safe: each DataLoader worker opens its own SQLite connection.

    Args:
        db_path: Path to the v7 noise-dumps DB (must have ``sections``,
            ``heightmaps``, and ``voxy_sections`` tables).
        level: Voxy LOD level (0–4).
        min_coverage: Fraction of constituent sections required for a
            WorldSection to be included (default 1.0 = all required).
        vocab_remap: Optional mapping from raw Voxy IDs to training IDs.
    """

    def __init__(
        self,
        db_path: str | Path,
        level: int,
        min_coverage: float = 1.0,
        vocab_remap: Optional[Dict[int, int]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self.level = level
        self.min_coverage = min_coverage
        self.vocab_remap = vocab_remap

        # Thread-local storage for SQLite connections
        self._local = threading.local()

        # Discover available samples (uses a temporary connection)
        conn = sqlite3.connect(self.db_path)
        self.samples = _discover_complete_world_sections(conn, level, min_coverage)
        conn.close()
        log.info(
            "VoxyLevelDataset(level=%d): found %d complete WorldSections (min_coverage=%.0f%%)",
            level,
            len(self.samples),
            min_coverage * 100,
        )

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude unpicklable threading.local from serialization."""
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore threading.local after deserialization."""
        self.__dict__.update(state)
        self._local = threading.local()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ws_x, ws_y, ws_z = self.samples[idx]
        conn = self._get_conn()

        # ── Tile noise/biome ──────────────────────────────────────
        noise_3d, biome_3d = _tile_noise_and_biome(conn, ws_x, ws_y, ws_z, self.level)

        # ── Tile heightmap ────────────────────────────────────────
        heightmap5 = _tile_heightmap(conn, ws_x, ws_z, self.level)

        # ── Y position ────────────────────────────────────────────
        # WorldSection Y in section units → y_position index
        # Overworld: block Y range [-64, 320) → section Y range [-4, 20)
        # We shift to 0-based: y_position = ws_y + 4 (for L0) … but this
        # depends on the level. For simplicity, use the block Y midpoint.
        shift = self.level + 1
        sy_base = ws_y << shift
        block_y_min = sy_base * 16
        # Section-Y index (0-based, overworld: 0 = bottom at -64)
        y_position = sy_base + 4  # shift so -4 → 0

        # ── Voxy ground-truth labels ──────────────────────────────
        row = conn.execute(
            "SELECT labels32 FROM voxy_sections WHERE level=? AND ws_x=? AND ws_y=? AND ws_z=?",
            (self.level, ws_x, ws_y, ws_z),
        ).fetchone()
        assert row is not None, f"Missing voxy_sections: L{self.level} ({ws_x},{ws_y},{ws_z})"
        labels32 = _unpack_voxy_blob(row[0])  # int32 (32, 32, 32) in (Y, Z, X) order

        # Remap vocabulary if needed
        if self.vocab_remap is not None:
            remap_arr = np.full(max(self.vocab_remap.keys()) + 1, -1, dtype=np.int32)
            for src, dst in self.vocab_remap.items():
                remap_arr[src] = dst
            labels32 = remap_arr[np.clip(labels32, 0, len(remap_arr) - 1)]

        # ── Pack as tensors ───────────────────────────────────────
        result: Dict[str, Any] = {
            "heightmap": torch.from_numpy(heightmap5),  # [5, Hx, Hz]
            "y_position": torch.tensor(y_position, dtype=torch.long),
            "labels32": torch.from_numpy(labels32).long(),  # [32, 32, 32]
            "block_y_min": torch.tensor(block_y_min, dtype=torch.long),
            "ws_coords": torch.tensor([ws_x, ws_y, ws_z], dtype=torch.long),
        }

        # L0/L1: full 3D noise + biome.  L2-L4: 2D feature-selected climate + biome.
        if self.level >= 2:
            climate_2d, biome_2d = _select_climate_2d(noise_3d, biome_3d, self.level)
            result["climate_2d"] = torch.from_numpy(climate_2d)  # [C_sel, 8, 8]
            result["biome_2d"] = torch.from_numpy(biome_2d).int()  # [8, 8]
            # Subsample heightmap to 8×8 to match climate grid
            hm_h, hm_w = heightmap5.shape[1], heightmap5.shape[2]
            sh = max(1, hm_h // _SUBSAMPLE_XZ)
            sw = max(1, hm_w // _SUBSAMPLE_XZ)
            hm_sub = heightmap5[:, ::sh, ::sw][:, :_SUBSAMPLE_XZ, :_SUBSAMPLE_XZ].copy()
            result["heightmap"] = torch.from_numpy(hm_sub)  # [5, 8, 8]
        else:
            result["noise_3d"] = torch.from_numpy(noise_3d)  # [15, Nx, Ny, Nz]
            result["biome_3d"] = torch.from_numpy(biome_3d).int()  # [Nx, Ny, Nz]

        return result


# ══════════════════════════════════════════════════════════════════════
#  Collate function
# ══════════════════════════════════════════════════════════════════════


def voxy_level_collate(
    batch: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Collate a batch of VoxyLevelDataset samples into stacked tensors.

    Works for both 3D (L0/L1) and 2D (L2-L4) input schemas.
    """
    result: Dict[str, torch.Tensor] = {
        "heightmap": torch.stack([s["heightmap"] for s in batch]),
        "y_position": torch.stack([s["y_position"] for s in batch]),
        "labels32": torch.stack([s["labels32"] for s in batch]),
        "block_y_min": torch.stack([s["block_y_min"] for s in batch]),
        "ws_coords": torch.stack([s["ws_coords"] for s in batch]),
    }

    # L0/L1: 3D noise + biome
    if "noise_3d" in batch[0]:
        result["noise_3d"] = torch.stack([s["noise_3d"] for s in batch])
        result["biome_3d"] = torch.stack([s["biome_3d"] for s in batch])

    # L2-L4: 2D climate + biome
    if "climate_2d" in batch[0]:
        result["climate_2d"] = torch.stack([s["climate_2d"] for s in batch])
        result["biome_2d"] = torch.stack([s["biome_2d"] for s in batch])

    # Parent blocks (L0-L3)
    if "parent_blocks" in batch[0]:
        result["parent_blocks"] = torch.stack([s["parent_blocks"] for s in batch])

    return result


# ══════════════════════════════════════════════════════════════════════
#  Parent extraction utility
# ══════════════════════════════════════════════════════════════════════


def extract_parent_from_coarser(
    labels32_coarser: np.ndarray,
    octant_idx: int,
) -> np.ndarray:
    """Extract a 16³ octant from a 32³ parent grid and nearest-neighbor upsample to 32³.

    The coarser-level Voxy grid represents the same spatial region at half
    resolution.  Each octant (0–7) is a 16³ sub-cube that, when upsampled
    2× in each axis, provides the parent conditioning for the finer level.

    Octant index uses Voxy convention: bit0=X, bit1=Z, bit2=Y.

    Args:
        labels32_coarser: ``int32[32, 32, 32]`` — coarser-level Voxy grid (Y, Z, X).
        octant_idx: 0–7 child index.

    Returns:
        ``int32[32, 32, 32]`` — nearest-neighbor upsampled parent context.
    """
    # Decode octant index → offsets
    dx = (octant_idx & 1) * 16
    dz = ((octant_idx >> 1) & 1) * 16
    dy = ((octant_idx >> 2) & 1) * 16

    # Extract 16³ octant
    octant = labels32_coarser[dy : dy + 16, dz : dz + 16, dx : dx + 16]

    # Nearest-neighbor upsample to 32³
    return np.repeat(np.repeat(np.repeat(octant, 2, axis=0), 2, axis=1), 2, axis=2)


# ══════════════════════════════════════════════════════════════════════
#  Dataset with parent conditioning (for L0–L3)
# ══════════════════════════════════════════════════════════════════════


class VoxyLevelWithParentDataset(Dataset):  # type: ignore[type-arg]
    """Per-level dataset that includes parent conditioning from the coarser level.

    Wraps a :class:`VoxyLevelDataset` and augments each sample with:
    - ``parent_blocks [32, 32, 32]`` — upsampled from the matching octant of
      the coarser-level Voxy grid.

    This requires the coarser level (level + 1) to also have data in the DB.

    Args:
        db_path: Path to the v7 noise-dumps DB.
        level: Target Voxy LOD level (0–3; L4 has no parent).
        min_coverage: Coverage threshold for the target level.
        vocab_remap: Optional vocabulary remap.
    """

    def __init__(
        self,
        db_path: str | Path,
        level: int,
        min_coverage: float = 1.0,
        vocab_remap: Optional[Dict[int, int]] = None,
    ) -> None:
        assert 0 <= level <= 3, "VoxyLevelWithParentDataset: level must be 0–3 (L4 has no parent)"
        self.db_path = str(db_path)
        self.level = level
        self.vocab_remap = vocab_remap
        self._local = threading.local()

        # The base dataset for this level
        self.base = VoxyLevelDataset(db_path, level, min_coverage, vocab_remap)

        # Pre-filter: only keep samples where the parent (coarser) level also exists
        conn = sqlite3.connect(self.db_path)
        coarser = level + 1
        filtered = []
        for ws_x, ws_y, ws_z in self.base.samples:
            # Parent WorldSection at coarser level
            # A WorldSection at level L maps to WorldSection at level L+1 via >> 1
            parent_ws_x = ws_x >> 1
            parent_ws_y = ws_y >> 1
            parent_ws_z = ws_z >> 1
            row = conn.execute(
                "SELECT 1 FROM voxy_sections WHERE level=? AND ws_x=? AND ws_y=? AND ws_z=?",
                (coarser, parent_ws_x, parent_ws_y, parent_ws_z),
            ).fetchone()
            if row is not None:
                filtered.append((ws_x, ws_y, ws_z))

        # Replace base samples with filtered ones
        dropped = len(self.base.samples) - len(filtered)
        self.base.samples = filtered
        conn.close()
        if dropped > 0:
            log.info(
                "VoxyLevelWithParentDataset(L%d): dropped %d samples missing parent at L%d",
                level,
                dropped,
                coarser,
            )

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._local = threading.local()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        ws_x, ws_y, ws_z = self.base.samples[idx]
        conn = self._get_conn()

        # Fetch parent grid from coarser level
        coarser = self.level + 1
        parent_ws_x = ws_x >> 1
        parent_ws_y = ws_y >> 1
        parent_ws_z = ws_z >> 1

        row = conn.execute(
            "SELECT labels32 FROM voxy_sections WHERE level=? AND ws_x=? AND ws_y=? AND ws_z=?",
            (coarser, parent_ws_x, parent_ws_y, parent_ws_z),
        ).fetchone()
        assert row is not None
        parent_labels32 = _unpack_voxy_blob(row[0])

        # Determine which octant we are within the parent WorldSection
        octant_x = ws_x & 1
        octant_z = ws_z & 1
        octant_y = ws_y & 1
        octant_idx = octant_x | (octant_z << 1) | (octant_y << 2)

        parent_blocks = extract_parent_from_coarser(parent_labels32, octant_idx)

        if self.vocab_remap is not None:
            remap_arr = np.full(max(self.vocab_remap.keys()) + 1, -1, dtype=np.int32)
            for src, dst in self.vocab_remap.items():
                remap_arr[src] = dst
            parent_blocks = remap_arr[np.clip(parent_blocks, 0, len(remap_arr) - 1)]

        sample["parent_blocks"] = torch.from_numpy(parent_blocks).long()
        return sample


__all__ = [
    "VoxyLevelDataset",
    "VoxyLevelWithParentDataset",
    "voxy_level_collate",
    "extract_parent_from_coarser",
]
