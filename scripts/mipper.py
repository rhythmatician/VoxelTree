"""
Voxy Mipper — Opacity-biased LOD downsampling.

Implements the exact algorithm from Voxy's ``Mipper.java`` / ``WorldConversionFactory.mipSection()``:
for each 2×2×2 window, select the non-air voxel with the highest *opacity*; break ties by corner
priority (I111 highest, I000 lowest).  If all corners are air the result is air.

**Why this matters:** Voxy uses this algorithm at every LOD level.  Training data built with a
different downsampler (mode / max-pool / probability pooling) produces parent voxels that
systematically disagree with what Voxy renders, hurting inference-time accuracy.

**Simplified opacity tiers (Phase-1, terrain-only):**
  - 0  = air                  (block_id == 0)
  - 1  = transparent non-air  (water, lava, bubble_column, …)
  - 15 = opaque solid         (everything else)

This is correct for ≥99% of terrain voxels.  Pass a ``vocab`` dict to ``build_opacity_table``
to enable per-block resolution.

**Corner priority** (from Voxy source, axes = (x, z, y_local) as stored in our NPZ files):
  Array stacking order 0-7 and their Voxy priorities [0,4,2,6,1,5,3,7].
  I111 (corner 7 in our ordering) has Voxy priority 7 — highest tie-break.

Usage
-----
::

    from scripts.mipper import mip_volume_numpy, mip_volume_torch, build_opacity_table

    # Numpy (offline pipeline)
    opacity = build_opacity_table(n_blocks=1105)
    coarse_labels, coarse_occ = mip_volume_numpy(labels16, factor=2, opacity_table=opacity)

    # PyTorch (in-trainer, vectorised)
    opacity_t = torch.from_numpy(opacity).long()
    coarse_labels_t, coarse_occ_t = mip_volume_torch(labels_b_d_h_w, factor=2,
                                                      opacity_table=opacity_t)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Transparent block name fragments (Phase-1 list; extend for Phase-2)
# ---------------------------------------------------------------------------
_TRANSPARENT_FRAGMENTS: frozenset[str] = frozenset(
    {
        "water",
        "lava",
        "bubble_column",
        "ice",
        "frosted_ice",
        "glass",
        "glass_pane",
        "leaves",
        "cobweb",
        "barrier",
        "light",
        "chorus_plant",
        "chorus_flower",
        "slime_block",
        "honey_block",
        "scaffolding",
    }
)

# ---------------------------------------------------------------------------
# Corner priority constants
# ---------------------------------------------------------------------------
# Our NPZ arrays have axis order (x, z, y_local).
# Voxy priority formula: priority = (y_local << 2) | (x << 1) | z
# For our stacking order below (corners[0..7]):
#   corner 0: [0::2,0::2,0::2] → (x=0,z=0,y=0) → priority = 0
#   corner 1: [0::2,0::2,1::2] → (x=0,z=0,y=1) → priority = 4
#   corner 2: [1::2,0::2,0::2] → (x=1,z=0,y=0) → priority = 2
#   corner 3: [1::2,0::2,1::2] → (x=1,z=0,y=1) → priority = 6
#   corner 4: [0::2,1::2,0::2] → (x=0,z=1,y=0) → priority = 1
#   corner 5: [0::2,1::2,1::2] → (x=0,z=1,y=1) → priority = 5
#   corner 6: [1::2,1::2,0::2] → (x=1,z=1,y=0) → priority = 3
#   corner 7: [1::2,1::2,1::2] → (x=1,z=1,y=1) → priority = 7  ← highest (I111)
_CORNER_PRIORITIES_NP = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int64)


# ---------------------------------------------------------------------------
# Opacity table construction
# ---------------------------------------------------------------------------


def build_opacity_table(
    n_blocks: int,
    vocab: Optional[dict[str, int]] = None,
) -> np.ndarray:
    """Return a 1-D numpy int64 array of length ``n_blocks`` mapping block_id → opacity tier.

    Args:
        n_blocks: Number of block IDs (table length).  Must be > 0.
        vocab:    Optional ``{block_name: block_id}`` mapping (e.g. from ``block_vocab.json``).
                  When provided, Minecraft names matching ``_TRANSPARENT_FRAGMENTS`` get tier 1.
                  When omitted, all non-zero IDs get tier 15 (conservative/correct for terrain).

    Returns:
        ``opacity_table`` where ``opacity_table[id]`` is 0, 1, or 15.
    """
    table = np.full(n_blocks, 15, dtype=np.int64)
    table[0] = 0  # minecraft:air is always transparent with opacity 0

    if vocab is not None:
        for name, bid in vocab.items():
            if bid < 0 or bid >= n_blocks:
                continue
            name_lower = name.lower()
            for frag in _TRANSPARENT_FRAGMENTS:
                if frag in name_lower:
                    table[bid] = 1
                    break

    return table


# ---------------------------------------------------------------------------
# NumPy implementation
# ---------------------------------------------------------------------------


def mip_once_numpy(
    labels: np.ndarray,
    opacity_table: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one Voxy Mipper step (single 2× downsample) in NumPy.

    Args:
        labels:        Integer block-ID array of shape *(D, H, W)* with even dimensions.
                       Axis order must match ``_CORNER_PRIORITIES_NP`` (i.e. (x, z, y_local)).
        opacity_table: 1-D opacity lookup, typically from :func:`build_opacity_table`.

    Returns:
        ``(coarse_labels, coarse_occ)`` both of shape *(D//2, H//2, W//2)*.
    """
    # Stack 8 corners of each 2×2×2 window along axis 0
    # Shape: (8, D//2, H//2, W//2)
    corners = np.stack(
        [
            labels[0::2, 0::2, 0::2],
            labels[0::2, 0::2, 1::2],
            labels[1::2, 0::2, 0::2],
            labels[1::2, 0::2, 1::2],
            labels[0::2, 1::2, 0::2],
            labels[0::2, 1::2, 1::2],
            labels[1::2, 1::2, 0::2],
            labels[1::2, 1::2, 1::2],
        ],
        axis=0,
    )

    # Clamp IDs to table bounds
    clamped = np.clip(corners, 0, len(opacity_table) - 1)
    opacities = opacity_table[clamped]  # (8, D//2, H//2, W//2)

    # Voxy score: (opacity << 4) | corner_priority
    priorities = _CORNER_PRIORITIES_NP[:, None, None, None]  # (8, 1, 1, 1)
    scores = (opacities << 4) + priorities  # (8, D//2, H//2, W//2)

    winner_idx = np.argmax(scores, axis=0)  # (D//2, H//2, W//2)

    # Gather block IDs from the winning corners
    d2, h2, w2 = labels.shape[0] // 2, labels.shape[1] // 2, labels.shape[2] // 2
    i0 = np.arange(d2)[:, None, None]
    i1 = np.arange(h2)[None, :, None]
    i2 = np.arange(w2)[None, None, :]
    coarse_labels = corners[winner_idx, i0, i1, i2]
    coarse_occ = (coarse_labels != 0).astype(np.uint8)

    return coarse_labels, coarse_occ


def mip_volume_numpy(
    labels: np.ndarray,
    factor: int,
    opacity_table: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample ``labels`` by ``factor`` using recursive Voxy Mipper steps.

    Args:
        labels:        Integer array shape *(D, H, W)*, block IDs.
        factor:        Power-of-two factor in ``{1, 2, 4, 8, 16}``.
        opacity_table: From :func:`build_opacity_table`.  Built lazily with ``n_blocks =
                       labels.max() + 1`` (padded) when not provided.

    Returns:
        ``(coarse_labels, coarse_occ)`` each of shape *(D//factor, H//factor, W//factor)*.
    """
    if factor == 1:
        occ = (labels != 0).astype(np.uint8)
        return labels.copy(), occ

    n_steps = int(round(math.log2(factor)))
    if 2**n_steps != factor:
        raise ValueError(f"factor must be a power of 2, got {factor}")

    if opacity_table is None:
        opacity_table = build_opacity_table(max(int(labels.max()) + 2, 2))

    current = labels
    for _ in range(n_steps):
        current, _ = mip_once_numpy(current, opacity_table)

    occ = (current != 0).astype(np.uint8)
    return current, occ


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------


def mip_once_torch(
    labels: "torch.Tensor",  # noqa: F821
    opacity_table: "torch.Tensor",  # noqa: F821
) -> Tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
    """Apply one Voxy Mipper step (single 2× downsample) in PyTorch.

    Args:
        labels:        Long integer tensor of shape *(B, D, H, W)*.
                       Axis order (D, H, W) = (x, z, y_local) matching the NumPy convention.
        opacity_table: 1-D Long tensor mapping block_id → opacity tier.

    Returns:
        ``(coarse_labels, coarse_occ)`` both of shape *(B, D//2, H//2, W//2)*.
        ``coarse_occ`` is float32 (0.0 or 1.0).
    """
    import torch

    # Stack 8 corners: (B, 8, D//2, H//2, W//2)
    corners = torch.stack(
        [
            labels[:, 0::2, 0::2, 0::2],
            labels[:, 0::2, 0::2, 1::2],
            labels[:, 1::2, 0::2, 0::2],
            labels[:, 1::2, 0::2, 1::2],
            labels[:, 0::2, 1::2, 0::2],
            labels[:, 0::2, 1::2, 1::2],
            labels[:, 1::2, 1::2, 0::2],
            labels[:, 1::2, 1::2, 1::2],
        ],
        dim=1,
    )  # (B, 8, D//2, H//2, W//2)

    dev = corners.device
    clamped = corners.clamp(0, len(opacity_table) - 1)
    opacities = opacity_table.to(dev)[clamped]  # (B, 8, D//2, H//2, W//2)

    prios = torch.tensor([0, 4, 2, 6, 1, 5, 3, 7], dtype=torch.long, device=dev).view(1, 8, 1, 1, 1)
    scores = (opacities << 4) + prios  # (B, 8, D//2, H//2, W//2)

    winner_idx = scores.argmax(dim=1, keepdim=True)  # (B, 1, D//2, H//2, W//2)
    coarse_labels = corners.gather(1, winner_idx).squeeze(1)  # (B, D//2, H//2, W//2)
    coarse_occ = (coarse_labels != 0).float()

    return coarse_labels, coarse_occ


def mip_volume_torch(
    labels: "torch.Tensor",  # noqa: F821
    factor: int,
    opacity_table: Optional["torch.Tensor"] = None,  # noqa: F821
) -> Tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
    """Downsample ``labels`` by ``factor`` using recursive Voxy Mipper steps (PyTorch).

    Args:
        labels:        Long tensor shape *(B, D, H, W)*.
        factor:        Power-of-two factor in ``{1, 2, 4, 8, 16}``.
        opacity_table: 1-D Long tensor.  Built lazily (all opaque tier) when not provided.

    Returns:
        ``(coarse_labels, coarse_occ)`` — shapes *(B, D//f, H//f, W//f)*.
        ``coarse_occ`` is float32.
    """
    import torch

    if factor == 1:
        return labels, (labels != 0).float()

    n_steps = int(round(math.log2(factor)))
    if 2**n_steps != factor:
        raise ValueError(f"factor must be a power of 2, got {factor}")

    if opacity_table is None:
        n = int(labels.max().item()) + 2
        opacity_table = build_opacity_table(max(n, 2))
        opacity_table = torch.from_numpy(opacity_table).long()

    current = labels
    for _ in range(n_steps):
        current, _ = mip_once_torch(current, opacity_table)

    coarse_occ = (current != 0).float()
    return current, coarse_occ
