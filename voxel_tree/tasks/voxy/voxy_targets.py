"""voxy-hierarchy target builders for Voxy-aligned subchunks.

This module converts a dense voxel cube (typically a 16×16×16 Voxy subchunk)
into explicit top-down voxy-hierarchy supervision.

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
class VoxyLevelTargets:
    """Per-level voxy-hierarchy targets.

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
class VoxyNode:
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


def build_voxy_targets(
    blocks: npt.NDArray[np.int32],
    *,
    air_id: int = 0,
    split_label: int = -1,
) -> Dict[int, VoxyLevelTargets]:
    """Build voxy-hierarchy supervision from a dense voxel cube.

    .. note::

        This function derives coarser levels by recursively subdividing a
        single 16³ block-level grid.  For multi-level Voxy ground truth
        (where each LOD level has its own labels), prefer
        :func:`build_multilevel_voxy_targets`.

    Args:
        blocks: Dense cubic voxel grid.  For the Voxy subchunk case this is
            typically ``int32[16, 16, 16]`` in ``(y, z, x)`` order.
        air_id: Block ID treated as empty space.
        split_label: Sentinel stored in ``labels`` for internal split nodes.

    Returns:
        Dict mapping octree level → :class:`VoxyLevelTargets`.
        For a 16³ input, levels are ``4, 3, 2, 1, 0``.
    """

    size = _require_power_of_two_cube(blocks)
    max_level = int(np.log2(size))

    result: Dict[int, VoxyLevelTargets] = {}
    for level in range(max_level, -1, -1):
        side = 2 ** (max_level - level)
        result[level] = VoxyLevelTargets(
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


def build_multilevel_voxy_targets(
    level_grids: Dict[int, npt.NDArray[np.int32]],
    *,
    air_id: int = 0,
    split_label: int = -1,
) -> Dict[int, VoxyLevelTargets]:
    """Build voxy-hierarchy targets using multi-level Voxy ground truth.

    Unlike :func:`build_voxy_targets` which recursively subdivides a
    single 16³ grid to derive coarser levels, this function uses Voxy's actual
    labels at each LOD level as ground truth.  ``is_leaf`` and ``child_mask``
    are computed by comparing adjacent Voxy levels.

    A node at level *L* is a **leaf** when all 8 children at level *L-1* share
    the same block ID (or when *L* is the finest available level).  Otherwise
    the node is a **split** and ``child_mask`` encodes which children at
    level *L-1* are non-air.

    Args:
        level_grids: Dict mapping Voxy level (0-4) to label grids.
            Expected shapes: L4=(1,1,1), L3=(2,2,2), L2=(4,4,4),
            L1=(8,8,8), L0=(16,16,16).  Missing levels are skipped.
        air_id: Block ID for empty space (used for child_mask).
        split_label: Sentinel stored in ``labels`` for internal split nodes.

    Returns:
        Dict mapping level → :class:`VoxyLevelTargets`.
        Only levels present in *level_grids* are included.
    """
    result: Dict[int, VoxyLevelTargets] = {}
    available = sorted(level_grids.keys(), reverse=True)  # e.g. [4, 3, 2, 1, 0]
    if not available:
        return result

    finest = min(available)

    for level in available:
        grid = level_grids[level].astype(np.int32, copy=False)
        S = grid.shape[0]  # 1, 2, 4, 8, or 16

        # Defaults: every node is a leaf with its own Voxy label.
        labels = grid.copy()
        is_leaf = np.ones((S, S, S), dtype=np.bool_)
        child_mask = np.zeros((S, S, S), dtype=np.uint8)

        # If there's a finer level available, compare to determine splits.
        if level > finest and (level - 1) in level_grids:
            finer = level_grids[level - 1].astype(np.int32, copy=False)  # (2S, 2S, 2S)

            # Reshape finer into 2×2×2 blocks aligned with this level's voxels.
            # After reshape + transpose:
            #   blocks[y, z, x, dy, dz, dx] = finer[2y+dy, 2z+dz, 2x+dx]
            blocks_6d = finer.reshape(S, 2, S, 2, S, 2)
            blocks = blocks_6d.transpose(0, 2, 4, 1, 3, 5)  # (S, S, S, 2, 2, 2)

            # Flatten (dy, dz, dx) → 8 children per node.
            blocks_flat = blocks.reshape(S, S, S, 8)
            bmin = blocks_flat.min(axis=-1)
            bmax = blocks_flat.max(axis=-1)
            homogeneous = bmin == bmax  # all 8 children identical?

            is_leaf = homogeneous
            labels = np.where(homogeneous, bmin, np.int32(split_label))

            # Child occupancy mask for split nodes.
            # Bit layout: bit = dx | (dz<<1) | (dy<<2).
            # The natural C-order reshape of (2,2,2)→(8,) gives index
            # dy*4 + dz*2 + dx which IS the same as dx|(dz<<1)|(dy<<2).
            non_air = (blocks_flat != air_id).astype(np.uint8)  # (S, S, S, 8)
            bit_weights = np.array(
                [1 << i for i in range(8)], dtype=np.uint8
            )  # [1, 2, 4, 8, 16, 32, 64, 128]
            mask = (
                non_air * bit_weights[np.newaxis, np.newaxis, np.newaxis, :]
            ).sum(axis=-1).astype(np.uint8)
            child_mask = np.where(homogeneous, np.uint8(0), mask)

        result[level] = VoxyLevelTargets(
            labels=labels,
            is_leaf=is_leaf,
            child_mask=child_mask,
        )

    return result


def iter_voxy_nodes(
    targets: Dict[int, VoxyLevelTargets],
    *,
    skip_empty_leaves: bool = True,
    air_id: int = 0,
) -> Iterator[VoxyNode]:
    """Yield flattened nodes from :func:`build_voxy_targets` output."""

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
                    yield VoxyNode(
                        level=level,
                        y=y,
                        z=z,
                        x=x,
                        label=label,
                        is_leaf=is_leaf,
                        child_mask=child_mask,
                    )

