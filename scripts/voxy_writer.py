"""voxy_writer.py — Encode VoxelTree output into Voxy's on-disk format.

Voxy stores each VoxelizedSection as 16×16×16 blocks in linear YZX order
(y-major, z-middle, x-minor), plus a mip pyramid (8³+4³+2³+1 = 585 extra
entries, total 4681 longs per section).

Each voxel is a 64-bit long packed as:

    bits 63-56  light           (8 bits: upper 4 = blockLight, lower 4 = skyLight)
    bits 55-47  biome ID        (9 bits)
    bits 46-27  block-state ID  (20 bits, Voxy internal ID)
    bits 26- 0  unused          (27 bits, zero)

Section keys pack (level, y, z, x) into a 64-bit long:

    bits 63-60  level  (4 bits)
    bits 59-52  y      (8 bits, signed)
    bits 51-28  z      (24 bits, signed)
    bits 27- 4  x      (24 bits, signed)
    bits  3- 0  spare  (4 bits, zero)

Blocks within a section are indexed as: y*256 + z*16 + x  (linear YZX).
"""

from __future__ import annotations

import struct
from typing import Dict, Sequence

# ---------------------------------------------------------------------------
# Bit-field layout (from Voxy source — Mapper.composeMappingId)
# ---------------------------------------------------------------------------
_BLOCK_ID_SHIFT = 27
_BLOCK_ID_BITS = 20
_BLOCK_ID_MASK = (1 << _BLOCK_ID_BITS) - 1  # 20 bits = 0xFFFFF

_BIOME_ID_SHIFT = 47
_BIOME_ID_BITS = 9
_BIOME_ID_MASK = (1 << _BIOME_ID_BITS) - 1  # 9 bits = 0x1FF

_LIGHT_SHIFT = 56
_LIGHT_MASK = 0xFF  # 8 bits


def encode_voxel_long(
    block_id: int,
    biome_id: int = 0,
    sky_light: int = 15,
    block_light: int = 0,
) -> int:
    """Pack a single voxel into Voxy's 64-bit long format.

    Args:
        block_id:    Voxy internal block-state ID (0 = air).
        biome_id:    Biome ID, 0–511 (9-bit).
        sky_light:   Sky-light level 0–15 (lower nibble of light byte).
        block_light: Block-light level 0–15 (upper nibble of light byte).

    Returns:
        Unsigned 64-bit integer ready to be written to a Voxy section blob.
    """
    light = ((block_light & 0xF) << 4) | (sky_light & 0xF)
    return (
        (light << _LIGHT_SHIFT)
        | ((biome_id & _BIOME_ID_MASK) << _BIOME_ID_SHIFT)
        | ((block_id & _BLOCK_ID_MASK) << _BLOCK_ID_SHIFT)
    )


def make_section_key(level: int, x: int, y: int, z: int) -> int:
    """Encode a Voxy section coordinate into a 64-bit map key.

    Layout (from Voxy WorldEngine.getWorldSectionId):
        bits 63-60  level (4 bits)
        bits 59-52  y     (8 bits, signed via mask)
        bits 51-28  z     (24 bits, signed via mask)
        bits 27- 4  x     (24 bits, signed via mask)
        bits  3- 0  spare (zero)

    Args:
        level: LOD level (0 = full resolution, up to 4).
        x, y, z: Section coordinates in section-space (not block-space).

    Returns:
        Unsigned 64-bit section key.
    """
    return (
        ((level & 0xF) << 60)
        | ((y & 0xFF) << 52)
        | ((z & ((1 << 24) - 1)) << 28)
        | ((x & ((1 << 24) - 1)) << 4)
    )


# ---------------------------------------------------------------------------
# Linear YZX indexing for 16×16×16 sections
# ---------------------------------------------------------------------------


def l0_index(x: int, y: int, z: int) -> int:
    """Get the linear index for a block in a 16³ VoxelizedSection (level-0).

    Ordering: y-major, z-middle, x-minor (YZX).

    Args:
        x, y, z: Block coordinates within the section (0–15 each).

    Returns:
        Array index (0–4095).
    """
    return (y << 8) | (z << 4) | x


# ---------------------------------------------------------------------------
# Section packing
# ---------------------------------------------------------------------------


def pack_blocks_to_section(
    block_ids_16: "object",
    voxy_id_map: Dict[int, int],
    biome_grid: "Sequence[int] | None" = None,
    default_sky_light: int = 15,
) -> bytes:
    """Pack a 16³ block-type array into a Voxy VoxelizedSection blob.

    The output contains 4681 longs:
      - 4096 entries for L0 (16×16×16)
      - 512 entries for L1 mip (8×8×8)
      - 64 entries for L2 mip (4×4×4)
      - 8 entries for L3 mip (2×2×2)
      - 1 entry for L4 mip (1×1×1)

    Mip entries use opacity-biased selection (most opaque child in each 2³
    group).  For simplicity, this implementation uses the first non-air child.

    Args:
        block_ids_16: A (16,16,16) int array in (X, Y, Z) or (X, Z, Y) axis
                      order.  Will be accessed as [x][y][z] in linear YZX order.
        voxy_id_map:  Mapping from VoxelTree block ID → Voxy internal block-state ID.
                      Block IDs absent from the map are treated as air (0).
        biome_grid:   Optional 256-element sequence of biome IDs in (x,z) order
                      (used to fill the biome field; defaults to 0).
        default_sky_light: Sky-light level to write for all blocks (0–15).

    Returns:
        Raw bytes: 4681 × 8 = 37448 bytes, little-endian unsigned 64-bit longs
        in linear YZX order, ready to be stored in a Voxy database entry.
    """
    import numpy as np

    _raw = block_ids_16
    if type(_raw).__name__ == "Tensor":
        import torch as _torch

        _cpu = _torch.as_tensor(_raw).detach().cpu()
        block_ids: np.ndarray = np.asarray(_cpu, dtype=np.int64)
    else:
        block_ids = np.asarray(_raw, dtype=np.int64)
    assert block_ids.shape == (16, 16, 16), f"Expected (16,16,16), got {block_ids.shape}"

    biomes: "Sequence[int]" = biome_grid if biome_grid is not None else [0] * 256
    total_entries = 16**3 + 8**3 + 4**3 + 2**3 + 1  # 4681
    longs = [0] * total_entries

    # Fill L0: 16×16×16 in YZX order
    for y in range(16):
        for z in range(16):
            for x in range(16):
                bid_vt = int(block_ids[x, y, z])
                bid_voxy = voxy_id_map.get(bid_vt, 0)
                biome = int(biomes[x * 16 + z]) if (x * 16 + z) < len(biomes) else 0
                vlong = encode_voxel_long(bid_voxy, biome, default_sky_light, 0)
                longs[l0_index(x, y, z)] = vlong

    # Fill mip levels with opacity-biased selection
    _fill_mip_pyramid(longs)

    # Serialize to little-endian bytes
    out = bytearray(total_entries * 8)
    for i, v in enumerate(longs):
        struct.pack_into("<Q", out, i * 8, v)
    return bytes(out)


def _fill_mip_pyramid(longs: list) -> None:
    """Fill mip levels 1–4 from the L0 data using first-non-air selection."""
    offsets = [0, 4096, 4096 + 512, 4096 + 512 + 64, 4096 + 512 + 64 + 8]
    sizes = [16, 8, 4, 2, 1]

    for level in range(1, 5):
        parent_off = offsets[level - 1]
        child_off = offsets[level]
        parent_size = sizes[level - 1]
        child_size = sizes[level]

        for cy in range(child_size):
            for cz in range(child_size):
                for cx in range(child_size):
                    # Take first non-air from the 2×2×2 parent group
                    best = 0
                    for dy in range(2):
                        for dz in range(2):
                            for dx in range(2):
                                px = cx * 2 + dx
                                py = cy * 2 + dy
                                pz = cz * 2 + dz
                                pidx = parent_off + (
                                    py * parent_size * parent_size + pz * parent_size + px
                                )
                                v = longs[pidx]
                                if best == 0 and v != 0:
                                    best = v
                    cidx = child_off + (cy * child_size * child_size + cz * child_size + cx)
                    longs[cidx] = best
