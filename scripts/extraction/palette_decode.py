# scripts/extraction/palette_decode.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def _bits_per_block(palette_size: int) -> int:
    if palette_size <= 1:
        return 4
    return max(4, int(math.ceil(math.log2(palette_size))))


def decode_palette_indices(states: Sequence[int], palette_size: int) -> np.ndarray:
    """
    Decode Mojang's bit-packed palette indices from a section's block_states.data (LongArray).
    Returns indices shaped (16, 16, 16) in (x, z, y) order.

    - states: iterable of signed 64-bit ints as returned by NBT readers
    - palette_size: len(section["block_states"]["palette"])
    """
    # Empty or single-valued palette => all zeros
    if palette_size <= 1 or not states:
        return np.zeros((16, 16, 16), dtype=np.uint16)

    bpb = _bits_per_block(palette_size)
    total = 16 * 16 * 16
    mask = (1 << bpb) - 1

    # Coerce to uint64 (Mojang stores as signed longs but we treat as raw bits)
    u64 = np.array([int(x) & 0xFFFFFFFFFFFFFFFF for x in states], dtype=np.uint64)

    out = np.empty(total, dtype=np.uint32)
    bit_index = 0
    for i in range(total):
        word_index = bit_index >> 6  # // 64
        bit_offset = bit_index & 63  # % 64
        val = u64[word_index] >> np.uint64(bit_offset)
        rem = 64 - bit_offset
        if rem < bpb:
            val |= u64[word_index + 1] << np.uint64(rem)
        out[i] = int(val & mask)
        bit_index += bpb

    # Mojang order is (y, z, x) when you linearly walk the 4096 values.
    # Reshape (y, z, x) then transpose to (x, z, y) to match downstream arrays.
    arr_yxz = out.reshape(16, 16, 16)  # (y, z, x)
    arr_xzy = np.transpose(arr_yxz, (2, 1, 0))  # -> (x, z, y)
    return arr_xzy.astype(np.uint16, copy=False)


def map_palette_to_block_ids(
    palette: List[Dict[str, Any]],
    indices: np.ndarray,
    block_id_map: Mapping[str, int],
    default_id: int = 0,
) -> np.ndarray:
    """
    Convert palette indices -> your integer block IDs using Name-only mapping.
    `indices` is (16,16,16) (x,z,y). Returns uint16 array of same shape.
    """
    names = [p.get("Name", "minecraft:air") for p in palette]
    lut = np.array([block_id_map.get(n, default_id) for n in names], dtype=np.uint16)
    return lut[indices]
