"""voxy_writer.py — Scaffold for encoding VoxelTree output into Voxy's on-disk format.

Voxy stores each subchunk section as a flat array of 64-bit "voxel longs", packed in a
Z-order (Morton) traversal of the 8×8×8 block grid.  Each long encodes:

    bits 63-24  block-state ID  (40 bits, Voxy internal ID)
    bits 23-16  biome ID        (8 bits)
    bits  7- 4  sky light       (4 bits)
    bits  3- 0  block light     (4 bits)

Section keys pack (level, x, y, z) into a single long.

Status:  **SCAFFOLD** — function signatures and bit-level recipes are correct per
         docs/VOXY-FORMAT.md; the wire-up to DJL / the LODiffusion mod is deferred to
         Milestone 5 (integration).

TODO(milestone-5): implement pack_patches_to_section() body and connect to mod.
"""

from __future__ import annotations

import struct
from typing import Dict, Sequence

# ---------------------------------------------------------------------------
# Bit-field layout (from Voxy source — ColumnDataPoint)
# ---------------------------------------------------------------------------
_BLOCK_ID_SHIFT = 24
_BIOME_SHIFT = 16
_SKY_LIGHT_SHIFT = 4
_BLOCK_LIGHT_SHIFT = 0

_BLOCK_ID_MASK = (1 << 40) - 1  # 40 bits
_BIOME_MASK = 0xFF  # 8 bits
_LIGHT_MASK = 0xF  # 4 bits


def encode_voxel_long(
    block_id: int,
    biome_id: int = 0,
    sky_light: int = 15,
    block_light: int = 0,
) -> int:
    """Pack a single voxel into Voxy's 64-bit long format.

    Args:
        block_id:    Voxy internal block-state ID (0 = air).
        biome_id:    Vanilla biome ID, 0–255.
        sky_light:   Sky-light level 0–15.
        block_light: Block-light level 0–15.

    Returns:
        Unsigned 64-bit integer ready to be written to a Voxy section blob.
    """
    return (
        ((block_id & _BLOCK_ID_MASK) << _BLOCK_ID_SHIFT)
        | ((biome_id & _BIOME_MASK) << _BIOME_SHIFT)
        | ((sky_light & _LIGHT_MASK) << _SKY_LIGHT_SHIFT)
        | ((block_light & _LIGHT_MASK) << _BLOCK_LIGHT_SHIFT)
    )


def make_section_key(level: int, x: int, y: int, z: int) -> int:
    """Encode a Voxy section coordinate into a 64-bit map key.

    Voxy places sections at LOD-dependent granularity.  The key packs:
        bits 63-60  level (4 bits)
        bits 59-40  x     (20 bits, signed via two's-complement in range)
        bits 39-20  y     (20 bits)
        bits 19- 0  z     (20 bits)

    Args:
        level: LOD level (0 = full resolution, 1 = 2× coarser, …).
        x, y, z: Section coordinates in section-space (not block-space).

    Returns:
        Unsigned 64-bit section key.
    """
    # Mask to 20 bits each (handles negative coords via Python's arbitrary int)
    _20 = (1 << 20) - 1
    return ((level & 0xF) << 60) | ((x & _20) << 40) | ((y & _20) << 20) | (z & _20)


# ---------------------------------------------------------------------------
# Z-order (Morton) helpers for the 8×8×8 traversal
# ---------------------------------------------------------------------------


def _part1by2(n: int) -> int:
    """Spread the low 10 bits of n to every 3rd bit (for 3-D Morton interleave)."""
    n &= 0x3FF  # keep 10 bits
    n = (n | (n << 16)) & 0x30000FF
    n = (n | (n << 8)) & 0x300F00F
    n = (n | (n << 4)) & 0x30C30C3
    n = (n | (n << 2)) & 0x9249249
    return n


def lin2z(idx: int) -> int:
    """Convert a linear block index (0–511 inside an 8³ section) to Morton code.

    Voxy iterates 8³ sections in Z-order: the Morton index is used as the array
    offset inside each section blob.

    Args:
        idx: Linear index = x*64 + z*8 + y  (x outer, y inner — Voxy convention).

    Returns:
        Morton (Z-order) index 0–511.
    """
    x = (idx >> 6) & 0x7
    z = (idx >> 3) & 0x7
    y = idx & 0x7
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def _compact1by2(n: int) -> int:
    """Inverse of _part1by2: extract every 3rd bit back to a contiguous value."""
    n &= 0x9249249
    n = (n | (n >> 2)) & 0x30C30C3
    n = (n | (n >> 4)) & 0x300F00F
    n = (n | (n >> 8)) & 0x300FF
    n = (n | (n >> 16)) & 0x3FF
    return n


def z2lin(morton: int) -> int:
    """Inverse of :func:`lin2z`: convert Morton code back to linear index.

    Args:
        morton: Z-order index 0–511.

    Returns:
        Linear index = x*64 + z*8 + y.
    """
    x = _compact1by2(morton)
    y = _compact1by2(morton >> 1)
    z = _compact1by2(morton >> 2)
    return x * 64 + z * 8 + y


# ---------------------------------------------------------------------------
# Section packing
# ---------------------------------------------------------------------------


def pack_patches_to_section(
    patches_8x: "dict[str, object]",
    voxy_id_map: Dict[int, int],
    biome_grid: "Sequence[int] | None" = None,
    default_sky_light: int = 15,
) -> bytes:
    """Pack an 8³ block-type array into a Voxy section blob (Morton-ordered longs).

    Args:
        patches_8x:    Dict containing at least ``"block_ids"`` — a (8,8,8) int array
                       in (x, z, y_local) axis order matching the NPZ convention.
        voxy_id_map:   Mapping from VoxelTree block ID → Voxy internal block-state ID.
                       Block IDs absent from the map are treated as air (0).
        biome_grid:    Optional flat 64-element sequence of biome IDs in (x,z) order
                       (used to fill the biome field; defaults to 0).
        default_sky_light: Sky-light level to write for all blocks (0–15).

    Returns:
        Raw bytes: 512 × 8 = 4096 bytes, little-endian unsigned 64-bit longs in
        Morton traversal order, ready to be stored in a Voxy database entry.

    .. note::
        TODO(milestone-5): This is a scaffold.  The body is implemented; wire-up to
        the DJL inference harness and the LODiffusion mod is deferred.
    """
    import numpy as np

    _raw = patches_8x["block_ids"]  # (8, 8, 8) int array: axes (x, z, y_local)
    # Accept numpy arrays or PyTorch tensors.
    if type(_raw).__name__ == "Tensor":
        import torch as _torch  # local import to avoid hard dependency

        _cpu = _torch.as_tensor(_raw).detach().cpu()  # type: ignore[arg-type]
        block_ids: np.ndarray = np.asarray(_cpu, dtype=np.int64)
    else:
        block_ids = np.asarray(_raw, dtype=np.int64)
    assert block_ids.shape == (8, 8, 8), f"Expected (8,8,8), got {block_ids.shape}"

    biomes: "Sequence[int]" = biome_grid if biome_grid is not None else [0] * 64

    longs = bytearray(512 * 8)
    for lin in range(512):
        x = lin >> 6
        z = (lin >> 3) & 0x7
        y = lin & 0x7
        bid_vt = int(block_ids[x, z, y])
        bid_voxy = voxy_id_map.get(bid_vt, 0)
        biome = int(biomes[x * 8 + z])
        vlong = encode_voxel_long(bid_voxy, biome, default_sky_light, 0)
        morton = lin2z(lin)
        struct.pack_into("<Q", longs, morton * 8, vlong)

    return bytes(longs)
