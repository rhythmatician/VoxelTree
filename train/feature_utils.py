"""Shared feature-engineering utilities for progressive LOD models.

The ``build_tabular_features_torch`` function mirrors the notebook-level
``build_tabular_features`` (NumPy) but uses only ONNX-exportable PyTorch
ops so it can run inside model ``forward()`` at inference time.

Feature vector layout (29 dims)::

    Plane 0 (surface):    mean, std, min, max, center  → 5
    Plane 1 (ocean_floor): ... same 5 stats             → 5
    Plane 2 (slope_x):     ...                           → 5
    Plane 3 (slope_z):     ...                           → 5
    Plane 4 (curvature):   ...                           → 5
    biome_center (float)                                  → 1
    biome_mode   (float)                                  → 1
    biome_unique_count                                    → 1
    y_index_normalized                                    → 1
                                                    Total: 29
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

TAB_DIM: int = 29
"""Dimensionality of the tabular feature vector (must match notebook)."""


# ── PyTorch version (ONNX-exportable, runs inside model forward) ─────────


def build_tabular_features_torch(
    height_planes: torch.Tensor,  # [B, 5, 16, 16]
    biome_indices: torch.Tensor,  # [B, 16, 16] int64
    y_index: torch.Tensor,  # [B] int64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 29-dim tabular conditioning vector (PyTorch / ONNX-safe).

    Returns:
        tabular:      [B, 29]  float32
        biome_center: [B]      int64
        biome_mode:   [B]      int64
    """
    stats: list[torch.Tensor] = []
    for i in range(5):
        plane = height_planes[:, i]  # [B, 16, 16]
        stats.extend(
            [
                plane.mean(dim=(1, 2)),
                plane.std(dim=(1, 2), correction=0),
                plane.amin(dim=(1, 2)),
                plane.amax(dim=(1, 2)),
                plane[:, 8, 8],
            ]
        )

    biome_center = biome_indices[:, 8, 8].long().clamp(0, 255)

    # biome_mode via scatter-add histogram (ONNX-exportable; no bincount)
    B = biome_indices.shape[0]
    flat = biome_indices.reshape(B, -1).long().clamp(0, 255)
    counts = torch.zeros(B, 256, device=biome_indices.device, dtype=torch.float32)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    biome_mode = counts.argmax(dim=1)

    biome_unique = (counts > 0).sum(dim=1).float()
    y_norm = y_index.float() / 24.0

    tabular = torch.stack(
        stats + [biome_center.float(), biome_mode.float(), biome_unique, y_norm],
        dim=1,
    )

    return tabular, biome_center, biome_mode


# ── NumPy version (single-sample, used by dataset __getitem__) ───────────


def build_tabular_features_numpy(
    height_planes: np.ndarray,  # (5, 16, 16)
    biome_idx: np.ndarray,  # (16, 16) int
    y_index: int,
) -> Tuple[np.ndarray, int, int]:
    """Build the 29-dim tabular vector for a single sample (NumPy).

    Returns:
        tabular:      (29,)  float32
        biome_center: int
        biome_mode:   int
    """
    stats: list[float] = []
    for i in range(5):
        plane = height_planes[i]  # (16, 16)
        stats.extend(
            [
                float(plane.mean()),
                float(plane.std()),
                float(plane.min()),
                float(plane.max()),
                float(plane[8, 8]),
            ]
        )

    biome_center = int(np.clip(biome_idx[8, 8], 0, 255))

    vals, counts = np.unique(np.clip(biome_idx, 0, 255), return_counts=True)
    biome_mode = int(vals[np.argmax(counts)])
    biome_unique = float(len(vals))

    y_norm = float(y_index) / 24.0

    tabular = np.array(
        stats + [float(biome_center), float(biome_mode), biome_unique, y_norm],
        dtype=np.float32,
    )
    return tabular, biome_center, biome_mode
