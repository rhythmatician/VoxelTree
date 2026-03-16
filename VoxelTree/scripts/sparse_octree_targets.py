"""Sparse-octree target builders for Voxy-aligned subchunks.

This module converts a dense voxel cube (typically a 16×16×16 Voxy subchunk)
into explicit top-down sparse-octree supervision.

For a 16³ subchunk, the resulting hierarchy is:

  L4: 1³   root voxel / node
  L3: 2³
  L2: 4³
  L1: 8³
  L0: 16³ leaves

Each node carries three training-relevant signals:

  - ``label``     : block ID if the node is a leaf, else ``split_label``
  - ``is_leaf``   : whether the node is homogeneous and can terminate
  - ``child_mask``: 8-bit occupancy mask for non-empty children when split

This is closer to the runtime object we want to predict than a dense 32³ grid:
the model can emit a sparse tree top-down and stop whenever the requested LOD
has been satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SparseOctreeLevel:
    """Per-level sparse-octree targets.

    Attributes:
        labels: ``int32[size, size, size]`` node labels.  For split nodes,
            ``split_label`` is stored instead.
        is_leaf: ``bool[size, size, size]`` indicating whether the node is a
            terminal homogeneous region.
        child_mask: ``uint8[size, size, size]`` occupancy mask for the node's
            eight children.  Zero for leaves.
    """

    labels: npt.NDArray[np.int32]
    is_leaf: npt.NDArray[np.bool_]
    child_mask: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class SparseOctreeNode:
    """Flattened node view useful for inspection and runtime-style traversal."""

    level: int
    y: int
    z: int
    x: int
    label: int
    is_leaf: bool
    child_mask: int


def _require_power_of_two_cube(blocks: npt.NDArray[np.int32]) -> int:
    if blocks.ndim != 3:
        raise ValueError(f"Expected a 3-D cube, got shape {blocks.shape}")
    if not (blocks.shape[0] == blocks.shape[1] == blocks.shape[2]):
        raise ValueError(f"Expected a cubic array, got shape {blocks.shape}")

    size = int(blocks.shape[0])
    if size <= 0 or size & (size - 1) != 0:
        raise ValueError(f"Cube edge must be a positive power of two, got {size}")
    return size


def _majority_label(cube: npt.NDArray[np.int32], air_id: int) -> int:
    """Return the dominant block ID, preferring non-air on ties."""

    values, counts = np.unique(cube, return_counts=True)
    if len(values) == 0:
        return int(air_id)

    max_count = int(counts.max())
    tied = values[counts == max_count]
    non_air = tied[tied != air_id]
    if len(non_air) > 0:
        return int(non_air.min())
    return int(tied.min())


def child_occupancy_mask(cube: npt.NDArray[np.int32], air_id: int = 0) -> np.uint8:
    """Return the 8-bit child occupancy mask for a split cube.

    A child bit is set when that octant contains any non-air voxel.
    Octant bits follow the repo convention:

      bit0 = x, bit1 = z, bit2 = y
    """

    size = _require_power_of_two_cube(cube)
    if size < 2:
        return np.uint8(0)

    half = size // 2
    mask = 0
    for dy in range(2):
        for dz in range(2):
            for dx in range(2):
                octant = dx | (dz << 1) | (dy << 2)
                ys = slice(dy * half, (dy + 1) * half)
                zs = slice(dz * half, (dz + 1) * half)
                xs = slice(dx * half, (dx + 1) * half)
                child = cube[ys, zs, xs]
                if np.any(child != air_id):
                    mask |= 1 << octant
    return np.uint8(mask)


def build_sparse_octree_targets(
    blocks: npt.NDArray[np.int32],
    *,
    air_id: int = 0,
    split_label: int = -1,
) -> Dict[int, SparseOctreeLevel]:
    """Build sparse-octree supervision from a dense voxel cube.

    Args:
        blocks: Dense cubic voxel grid.  For the Voxy subchunk case this is
            typically ``int32[16, 16, 16]`` in ``(y, z, x)`` order.
        air_id: Block ID treated as empty space.
        split_label: Sentinel stored in ``labels`` for internal split nodes.

    Returns:
        Dict mapping octree level → :class:`SparseOctreeLevel`.
        For a 16³ input, levels are ``4, 3, 2, 1, 0``.
    """

    size = _require_power_of_two_cube(blocks)
    max_level = int(np.log2(size))

    result: Dict[int, SparseOctreeLevel] = {}
    for level in range(max_level, -1, -1):
        side = 2 ** (max_level - level)
        result[level] = SparseOctreeLevel(
            labels=np.full((side, side, side), np.int32(split_label), dtype=np.int32),
            is_leaf=np.zeros((side, side, side), dtype=np.bool_),
            child_mask=np.zeros((side, side, side), dtype=np.uint8),
        )

    def recurse(cube: npt.NDArray[np.int32], level: int, y: int, z: int, x: int) -> None:
        level_data = result[level]
        unique = np.unique(cube)

        if len(unique) == 1:
            level_data.labels[y, z, x] = np.int32(unique[0])
            level_data.is_leaf[y, z, x] = True
            level_data.child_mask[y, z, x] = np.uint8(0)
            return

        if level == 0:
            # Defensive fallback: at the leaf level a 1×1×1 cube should already
            # be uniform, but keep a deterministic label if malformed input
            # somehow reaches this branch.
            level_data.labels[y, z, x] = np.int32(_majority_label(cube, air_id))
            level_data.is_leaf[y, z, x] = True
            level_data.child_mask[y, z, x] = np.uint8(0)
            return

        level_data.labels[y, z, x] = np.int32(split_label)
        level_data.is_leaf[y, z, x] = False
        level_data.child_mask[y, z, x] = child_occupancy_mask(cube, air_id=air_id)

        half = cube.shape[0] // 2
        for dy in range(2):
            for dz in range(2):
                for dx in range(2):
                    ys = slice(dy * half, (dy + 1) * half)
                    zs = slice(dz * half, (dz + 1) * half)
                    xs = slice(dx * half, (dx + 1) * half)
                    recurse(cube[ys, zs, xs], level - 1, y * 2 + dy, z * 2 + dz, x * 2 + dx)

    recurse(blocks.astype(np.int32, copy=False), max_level, 0, 0, 0)
    return result


def iter_sparse_octree_nodes(
    targets: Dict[int, SparseOctreeLevel],
    *,
    skip_empty_leaves: bool = True,
    air_id: int = 0,
) -> Iterator[SparseOctreeNode]:
    """Yield flattened nodes from :func:`build_sparse_octree_targets` output."""

    for level in sorted(targets.keys(), reverse=True):
        data = targets[level]
        side = data.labels.shape[0]
        for y in range(side):
            for z in range(side):
                for x in range(side):
                    label = int(data.labels[y, z, x])
                    is_leaf = bool(data.is_leaf[y, z, x])
                    child_mask = int(data.child_mask[y, z, x])
                    if skip_empty_leaves and is_leaf and label == air_id:
                        continue
                    yield SparseOctreeNode(
                        level=level,
                        y=y,
                        z=z,
                        x=x,
                        label=label,
                        is_leaf=is_leaf,
                        child_mask=child_mask,
                    )
