#!/usr/bin/env python3
"""Spatial Resolution Analysis for Voxy LOD Levels.

Answers: "At each LOD level, how many sub-voxel spatial samples of each
input signal (heightmap, noise) do we actually need per voxel?"

Key metrics
-----------
1. **Variance decomposition** (primary metric): Measures the ratio
   between-voxel variance / total variance.  If ~1.0, one sample
   per voxel captures nearly all information; if low, we need more.

2. **Reconstruction RMSE**: At each subsampling rate N (samples per
   voxel axis), subsample, then upsample back.  RMSE vs full-res
   tells us how much detail is lost at each rate.

3. **Aggregation strategy comparison**: R² of each strategy (mean,
   center, corner, max, min) vs label-derived surface height.

4. **Noise resolution sweep**: Same variance decomposition for noise
   channels at different XZ output resolutions.

Coordinate conventions
---------------------
- WorldSection at level L covers 32 × 2^L blocks per axis.
- shift = level + 1; T = 2^shift chunks per WS axis.
- Heightmap uses (chunk_x, chunk_z); one 16×16 grid per chunk.
- Labels are int32[Y=32, Z=32, X=32] in voxel coords.
- Surface height from labels = y_min_block + voxel_index * (2^level).

Usage:
    python scripts/spatial_resolution_analysis.py [--db PATH] [--samples N]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────
N_FIELDS = 15
_NOISE_QX, _NOISE_QY, _NOISE_QZ = 4, 2, 4
_HM_BLOCK_RES = 16
_CELL_WIDTH = 4

FIELD_NAMES = [
    "temperature",
    "vegetation",
    "continents",
    "erosion",
    "depth",
    "ridges",
    "preliminary_surface_level",
    "final_density",
    "barrier",
    "fluid_level_floodedness",
    "fluid_level_spread",
    "lava",
    "vein_toggle",
    "vein_ridged",
    "vein_gap",
]

LEVEL_CHANNELS = {
    2: [0, 1, 2, 3, 4, 5, 7],  # 6 climate + final_density
    3: [0, 1, 2, 3, 4, 5],  # 6 climate
    4: [0, 1, 2, 3, 5],  # 5 climate (no depth)
}


# ── DB helpers ──────────────────────────────────────────────────────


def _unpack_noise(blob: bytes) -> np.ndarray:
    return (
        np.frombuffer(blob, dtype=np.float32)
        .reshape(N_FIELDS, _NOISE_QX, _NOISE_QY, _NOISE_QZ)
        .copy()
    )


def _unpack_biome(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.int32).reshape(_NOISE_QX, _NOISE_QY, _NOISE_QZ).copy()


def _unpack_heightmap(blob: bytes) -> np.ndarray:
    return (
        np.frombuffer(blob, dtype=np.int32).reshape(_HM_BLOCK_RES, _HM_BLOCK_RES).astype(np.float32)
    )


def _unpack_voxy(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=np.int32).reshape(32, 32, 32).copy()


# ── Tiling ──────────────────────────────────────────────────────────


def tile_heightmap_blocks(
    conn: sqlite3.Connection,
    ws_x: int,
    ws_z: int,
    level: int,
) -> np.ndarray:
    """Tile heightmaps at full block resolution for a WorldSection.

    Returns: float32[H_blocks, W_blocks] where H = W = T * 16.
    """
    shift = level + 1
    T = 1 << shift
    block_res = T * _HM_BLOCK_RES
    surface = np.full((block_res, block_res), np.nan, dtype=np.float32)

    cx_base = ws_x << shift
    cz_base = ws_z << shift

    rows = conn.execute(
        "SELECT chunk_x, chunk_z, surface FROM heightmaps "
        "WHERE chunk_x >= ? AND chunk_x < ? AND chunk_z >= ? AND chunk_z < ?",
        (cx_base, cx_base + T, cz_base, cz_base + T),
    ).fetchall()

    for cx, cz, blob in rows:
        dx, dz = cx - cx_base, cz - cz_base
        hm = _unpack_heightmap(blob)
        surface[dx * 16 : (dx + 1) * 16, dz * 16 : (dz + 1) * 16] = hm

    return surface


def tile_noise(
    conn: sqlite3.Connection,
    ws_x: int,
    ws_y: int,
    ws_z: int,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tile noise + biome at quart-cell resolution.

    Returns:
        noise: float32[15, Nx, Ny, Nz]
        biome: int32[Nx, Ny, Nz]
    """
    shift = level + 1
    T = 1 << shift
    noise_out = np.zeros((N_FIELDS, T * 4, T * 2, T * 4), dtype=np.float32)
    biome_out = np.zeros((T * 4, T * 2, T * 4), dtype=np.int32)

    cx_base = ws_x << shift
    sy_base = ws_y << shift
    cz_base = ws_z << shift

    rows = conn.execute(
        "SELECT chunk_x, section_y, chunk_z, noise_data, biome_ids FROM sections "
        "WHERE chunk_x >= ? AND chunk_x < ? AND section_y >= ? AND section_y < ? "
        "AND chunk_z >= ? AND chunk_z < ?",
        (cx_base, cx_base + T, sy_base, sy_base + T, cz_base, cz_base + T),
    ).fetchall()

    for cx, sy, cz, noise_blob, biome_blob in rows:
        dx, dy, dz = cx - cx_base, sy - sy_base, cz - cz_base
        noise = _unpack_noise(noise_blob)
        biome = _unpack_biome(biome_blob)
        noise_out[:, dx * 4 : (dx + 1) * 4, dy * 2 : (dy + 1) * 2, dz * 4 : (dz + 1) * 4] = noise
        biome_out[dx * 4 : (dx + 1) * 4, dy * 2 : (dy + 1) * 2, dz * 4 : (dz + 1) * 4] = biome

    return noise_out, biome_out


def get_voxy_labels(
    conn: sqlite3.Connection,
    level: int,
    ws_x: int,
    ws_y: int,
    ws_z: int,
) -> np.ndarray | None:
    """Get 32³ voxy labels for a WorldSection. Returns None if missing."""
    row = conn.execute(
        "SELECT labels32 FROM voxy_sections WHERE level=? AND ws_x=? AND ws_y=? AND ws_z=?",
        (level, ws_x, ws_y, ws_z),
    ).fetchone()
    if row is None:
        return None
    return _unpack_voxy(row[0])


# ── Subsampling strategies ──────────────────────────────────────────


def subsample_heightmap(
    hm_blocks: np.ndarray,
    voxel_size_blocks: int,
    grid_size: int = 32,
    strategy: str = "mean",
    samples_per_voxel: int = 1,
) -> np.ndarray:
    """Subsample a full-res heightmap to different per-voxel resolutions.

    Args:
        hm_blocks: float32[H, W] at block resolution.
        voxel_size_blocks: How many blocks per voxel (= 2^level).
        grid_size: Output voxels per axis (32).
        strategy: 'mean', 'center', 'corner', 'max', 'min', 'std'.
        samples_per_voxel: If > 1, produces [samples, samples, 32, 32]-ish output.

    Returns:
        float32[grid_size, grid_size] or float32[spv, spv, grid_size, grid_size]
    """
    if strategy == "multi":
        # Return samples_per_voxel × samples_per_voxel sub-samples within each voxel
        spv = min(samples_per_voxel, voxel_size_blocks)
        step = voxel_size_blocks // spv
        result = np.zeros((spv, spv, grid_size, grid_size), dtype=np.float32)
        for si in range(spv):
            for sj in range(spv):
                for vx in range(grid_size):
                    for vz in range(grid_size):
                        bx = vx * voxel_size_blocks + si * step
                        bz = vz * voxel_size_blocks + sj * step
                        if bx < hm_blocks.shape[0] and bz < hm_blocks.shape[1]:
                            result[si, sj, vx, vz] = hm_blocks[bx, bz]
        return result

    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    for vx in range(grid_size):
        for vz in range(grid_size):
            bx0 = vx * voxel_size_blocks
            bz0 = vz * voxel_size_blocks
            bx1 = min(bx0 + voxel_size_blocks, hm_blocks.shape[0])
            bz1 = min(bz0 + voxel_size_blocks, hm_blocks.shape[1])
            patch = hm_blocks[bx0:bx1, bz0:bz1]

            if patch.size == 0 or np.all(np.isnan(patch)):
                result[vx, vz] = 0.0
                continue

            valid = patch[~np.isnan(patch)]
            if valid.size == 0:
                result[vx, vz] = 0.0
                continue

            if strategy == "mean":
                result[vx, vz] = valid.mean()
            elif strategy == "center":
                cx, cz = voxel_size_blocks // 2, voxel_size_blocks // 2
                result[vx, vz] = hm_blocks[min(bx0 + cx, bx1 - 1), min(bz0 + cz, bz1 - 1)]
            elif strategy == "corner":
                result[vx, vz] = hm_blocks[bx0, bz0]
            elif strategy == "max":
                result[vx, vz] = valid.max()
            elif strategy == "min":
                result[vx, vz] = valid.min()
            elif strategy == "std":
                result[vx, vz] = valid.std() if valid.size > 1 else 0.0
    return result


def subsample_noise_xz(
    noise_3d: np.ndarray,
    level: int,
    out_size: int = 8,
    strategy: str = "mean",
) -> np.ndarray:
    """Collapse noise to 2D and subsample XZ.

    Args:
        noise_3d: float32[15, Nx, Ny, Nz] at quart-cell resolution.
        level: LOD level.
        out_size: Target XZ grid size.
        strategy: 'mean', 'stride', 'max'.

    Returns:
        float32[C, out_size, out_size] — Y-averaged, XZ-subsampled.
    """
    channels = LEVEL_CHANNELS.get(level, list(range(N_FIELDS)))
    selected = noise_3d[channels]  # [C, Nx, Ny, Nz]
    y_avg = selected.mean(axis=2)  # [C, Nx, Nz]
    C, Nx, Nz = y_avg.shape

    if strategy == "stride":
        sx, sz = max(1, Nx // out_size), max(1, Nz // out_size)
        return y_avg[:, ::sx, ::sz][:, :out_size, :out_size].copy()

    # Block-average: reshape into out_size×out_size blocks
    bx, bz = Nx // out_size, Nz // out_size
    if bx < 1 or bz < 1:
        # Noise grid smaller than target — just return as-is padded
        result = np.zeros((C, out_size, out_size), dtype=np.float32)
        result[:, : min(Nx, out_size), : min(Nz, out_size)] = y_avg[:, :out_size, :out_size]
        return result

    trimmed = y_avg[:, : bx * out_size, : bz * out_size]
    reshaped = trimmed.reshape(C, out_size, bx, out_size, bz)

    if strategy == "mean":
        return reshaped.mean(axis=(2, 4))
    elif strategy == "max":
        return reshaped.max(axis=(2, 4))
    else:
        return reshaped.mean(axis=(2, 4))


# ── Metrics ─────────────────────────────────────────────────────────


def column_majority_block(labels32: np.ndarray) -> np.ndarray:
    """For each XZ column, find the most common non-air block type.

    Args:
        labels32: int32[Y=32, Z=32, X=32]

    Returns:
        int32[32, 32] — majority block per column (X, Z).
    """
    result = np.zeros((32, 32), dtype=np.int32)
    for x in range(32):
        for z in range(32):
            col = labels32[:, z, x]  # Y, Z, X ordering
            nonair = col[col > 0]
            if len(nonair) == 0:
                result[x, z] = 0
            else:
                counts = Counter(nonair.tolist())
                result[x, z] = counts.most_common(1)[0][0]
    return result


def surface_height_from_labels(
    labels32: np.ndarray,
    y_min_block: int,
    voxel_size_blocks: int = 1,
) -> np.ndarray:
    """Compute effective surface height from voxel labels.

    Finds the highest non-air voxel in each column and converts the
    voxel index to block coordinates:  y_min_block + index * voxel_size.

    labels32: [Y=32, Z=32, X=32]

    Returns: float32[32, 32] — surface Y in block coords (X, Z layout).
             NaN for all-air columns.
    """
    result = np.full((32, 32), np.nan, dtype=np.float32)
    for x in range(32):
        for z in range(32):
            col = labels32[:, z, x]
            nonair_y = np.where(col > 0)[0]
            if len(nonair_y) > 0:
                result[x, z] = y_min_block + nonair_y[-1] * voxel_size_blocks
    return result


def correlation_with_surface(
    subsampled_hm: np.ndarray,
    label_surface: np.ndarray,
) -> float:
    """Pearson correlation between subsampled heightmap and label-derived surface.

    Both should be [32, 32].
    """
    a = subsampled_hm.ravel()
    b = label_surface.ravel()
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 10:
        return 0.0
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def r_squared_by_resolution(
    hm_blocks: np.ndarray,
    label_surface: np.ndarray,
    voxel_size_blocks: int,
    strategies: List[str] = ["mean", "center", "corner", "max", "min"],
) -> Dict[str, float]:
    """Compute R² for each heightmap subsampling strategy.

    Returns dict mapping strategy → R² value.
    """
    results: Dict[str, float] = {}
    for strat in strategies:
        sub = subsample_heightmap(hm_blocks, voxel_size_blocks, 32, strat)
        r = correlation_with_surface(sub, label_surface)
        results[strat] = r**2
    return results


def sub_voxel_position_importance(
    hm_blocks: np.ndarray,
    label_surface: np.ndarray,
    voxel_size_blocks: int,
    grid_size: int = 32,
) -> np.ndarray:
    """For each sub-voxel position (i, j), compute R² with label surface.

    Returns: float32[voxel_size, voxel_size] — R² per position within voxel.
    """
    spv = voxel_size_blocks
    r2_map = np.zeros((spv, spv), dtype=np.float32)

    for si in range(spv):
        for sj in range(spv):
            sub = np.zeros((grid_size, grid_size), dtype=np.float32)
            for vx in range(grid_size):
                for vz in range(grid_size):
                    bx = vx * voxel_size_blocks + si
                    bz = vz * voxel_size_blocks + sj
                    if bx < hm_blocks.shape[0] and bz < hm_blocks.shape[1]:
                        sub[vx, vz] = hm_blocks[bx, bz]
            r = correlation_with_surface(sub, label_surface)
            r2_map[si, sj] = r**2

    return r2_map


def multi_sample_r_squared(
    hm_blocks: np.ndarray,
    label_surface: np.ndarray,
    voxel_size_blocks: int,
    n_samples_per_axis: List[int] = [1, 2, 4, 8, 16],
) -> Dict[int, float]:
    """Test how R² improves as we increase heightmap samples per voxel.

    For N samples-per-voxel-axis, we take N×N evenly spaced samples
    within each voxel, average them, and compute R² vs label surface.

    Returns: dict mapping n_samples → R².
    """
    results: Dict[int, float] = {}
    for n in n_samples_per_axis:
        if n > voxel_size_blocks:
            continue
        step = max(1, voxel_size_blocks // n)
        sub = np.zeros((32, 32), dtype=np.float32)
        for vx in range(32):
            for vz in range(32):
                bx0 = vx * voxel_size_blocks
                bz0 = vz * voxel_size_blocks
                vals = []
                for si in range(n):
                    for sj in range(n):
                        bx = bx0 + si * step
                        bz = bz0 + sj * step
                        if bx < hm_blocks.shape[0] and bz < hm_blocks.shape[1]:
                            v = hm_blocks[bx, bz]
                            if not np.isnan(v):
                                vals.append(v)
                sub[vx, vz] = np.mean(vals) if vals else 0.0
        r = correlation_with_surface(sub, label_surface)
        results[n] = r**2
    return results


def variance_decomposition(
    hm_blocks: np.ndarray,
    voxel_size_blocks: int,
    grid_size: int = 32,
) -> Dict[str, float]:
    """Decompose heightmap variance into within-voxel and between-voxel.

    If between / total ≈ 1.0, one sample per voxel captures nearly all
    the spatial information.  If < 0.5, we're losing significant detail.

    Returns:
        within_var, between_var, total_var, ratio (between/total),
        within_std, between_std.
    """
    within_vars: list[float] = []
    voxel_means: list[float] = []

    for vx in range(grid_size):
        for vz in range(grid_size):
            bx0, bz0 = vx * voxel_size_blocks, vz * voxel_size_blocks
            bx1 = min(bx0 + voxel_size_blocks, hm_blocks.shape[0])
            bz1 = min(bz0 + voxel_size_blocks, hm_blocks.shape[1])
            patch = hm_blocks[bx0:bx1, bz0:bz1]
            valid = patch[~np.isnan(patch)]
            if len(valid) > 1:
                within_vars.append(float(valid.var()))
                voxel_means.append(float(valid.mean()))

    if not within_vars:
        return {
            "within_var": 0.0,
            "between_var": 0.0,
            "total_var": 0.0,
            "ratio": 0.0,
            "within_std": 0.0,
            "between_std": 0.0,
        }

    within_var = float(np.mean(within_vars))
    between_var = float(np.var(voxel_means))
    total_var = within_var + between_var
    ratio = between_var / total_var if total_var > 0 else 0.0
    return {
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
        "ratio": ratio,
        "within_std": float(np.sqrt(within_var)),
        "between_std": float(np.sqrt(between_var)),
    }


def reconstruction_rmse_sweep(
    hm_blocks: np.ndarray,
    voxel_size_blocks: int,
    grid_size: int = 32,
    n_samples_list: List[int] | None = None,
) -> Dict[int, float]:
    """Measure reconstruction RMSE at different subsampling rates.

    For each N (samples per voxel axis), subsample the heightmap to
    N×N per voxel, then upsample back to block resolution with
    nearest-neighbor, and compute RMSE vs original.

    Returns: dict mapping N → RMSE (in blocks).
    """
    if n_samples_list is None:
        n_samples_list = [n for n in [1, 2, 4, 8, 16] if n <= voxel_size_blocks]

    results: Dict[int, float] = {}

    for n in n_samples_list:
        if n > voxel_size_blocks:
            continue
        step = voxel_size_blocks // n

        # Subsample: take every `step` blocks within each voxel
        reconstructed = np.full_like(hm_blocks, np.nan)
        for vx in range(grid_size):
            for vz in range(grid_size):
                bx0 = vx * voxel_size_blocks
                bz0 = vz * voxel_size_blocks
                for si in range(n):
                    for sj in range(n):
                        sample_bx = bx0 + si * step
                        sample_bz = bz0 + sj * step
                        if sample_bx >= hm_blocks.shape[0] or sample_bz >= hm_blocks.shape[1]:
                            continue
                        val = hm_blocks[sample_bx, sample_bz]
                        # Fill the region this sample represents
                        fill_bx0 = bx0 + si * step
                        fill_bz0 = bz0 + sj * step
                        fill_bx1 = min(fill_bx0 + step, bx0 + voxel_size_blocks, hm_blocks.shape[0])
                        fill_bz1 = min(fill_bz0 + step, bz0 + voxel_size_blocks, hm_blocks.shape[1])
                        reconstructed[fill_bx0:fill_bx1, fill_bz0:fill_bz1] = val

        valid = ~(np.isnan(hm_blocks) | np.isnan(reconstructed))
        if valid.sum() > 0:
            diff = hm_blocks[valid] - reconstructed[valid]
            results[n] = float(np.sqrt(np.mean(diff**2)))
        else:
            results[n] = float("nan")

    return results


def completeness_check(labels32: np.ndarray) -> Dict[str, float]:
    """Check how 'partial' a WorldSection is.

    Returns:
        all_air_frac: Fraction of columns that are entirely air.
        mean_fill: Average fraction of non-air voxels per non-air column.
        has_surface: Whether the WS likely contains the terrain surface
            (some columns go from solid below to air above).
    """
    total_cols = 32 * 32
    all_air = 0
    fill_fracs: list[float] = []
    surface_cols = 0

    for x in range(32):
        for z in range(32):
            col = labels32[:, z, x]
            n_nonair = (col > 0).sum()
            if n_nonair == 0:
                all_air += 1
            else:
                fill_fracs.append(n_nonair / 32)
                # Check if column has a solid-to-air transition (surface)
                nonair_idx = np.where(col > 0)[0]
                if nonair_idx[-1] < 31:  # not solid all the way up
                    surface_cols += 1

    return {
        "all_air_frac": all_air / total_cols,
        "mean_fill": float(np.mean(fill_fracs)) if fill_fracs else 0.0,
        "surface_frac": surface_cols / total_cols,
        "non_air_cols": total_cols - all_air,
    }


def noise_variance_decomposition(
    noise_3d: np.ndarray,
    level: int,
) -> Dict[str, Dict[str, float]]:
    """Variance decomposition for noise channels.

    At each level, noise is at quart-cell resolution (4×2×4 per section).
    Per voxel in XZ, there are (voxel_size / 4) quart-cells.
    We decompose variance into within-voxel and between-voxel.

    Returns: dict mapping channel_name → {within_var, between_var, ratio}.
    """
    channels = LEVEL_CHANNELS.get(level, list(range(N_FIELDS)))
    voxel_size_blocks = 1 << level
    quarts_per_voxel_xz = max(1, voxel_size_blocks // _CELL_WIDTH)

    # Y-average the noise first
    selected = noise_3d[channels]  # [C, Nx, Ny, Nz]
    y_avg = selected.mean(axis=2)  # [C, Nx, Nz]

    results: Dict[str, Dict[str, float]] = {}

    for ci, ch_idx in enumerate(channels):
        ch_data = y_avg[ci]  # [Nx, Nz]
        Nx, Nz = ch_data.shape
        grid_x = Nx // quarts_per_voxel_xz
        grid_z = Nz // quarts_per_voxel_xz

        if grid_x < 2 or grid_z < 2 or quarts_per_voxel_xz < 2:
            continue

        within_vars: list[float] = []
        voxel_means: list[float] = []

        for vx in range(grid_x):
            for vz in range(grid_z):
                qx0 = vx * quarts_per_voxel_xz
                qz0 = vz * quarts_per_voxel_xz
                patch = ch_data[
                    qx0 : qx0 + quarts_per_voxel_xz, qz0 : qz0 + quarts_per_voxel_xz
                ].ravel()
                if len(patch) > 1 and patch.std() > 0:
                    within_vars.append(float(patch.var()))
                    voxel_means.append(float(patch.mean()))
                elif len(patch) >= 1:
                    within_vars.append(0.0)
                    voxel_means.append(float(patch.mean()))

        if not voxel_means:
            continue

        w = float(np.mean(within_vars))
        b = float(np.var(voxel_means))
        total = w + b
        results[FIELD_NAMES[ch_idx]] = {
            "within_var": w,
            "between_var": b,
            "ratio": b / total if total > 0 else 0.0,
        }

    return results


def noise_resolution_sweep(
    noise_3d: np.ndarray,
    labels32: np.ndarray,
    level: int,
    out_sizes: List[int] = [4, 8, 16, 32],
) -> Dict[str, Dict[str, float]]:
    """Test noise at different XZ output grid sizes.

    For each output size, compute correlation between each noise channel
    and the label-derived features (air fraction per column).

    Returns: dict mapping out_size_strategy → {metric: value}.
    """
    channels = LEVEL_CHANNELS.get(level, list(range(N_FIELDS)))

    # Compute label-derived features at 32×32
    air_frac = np.zeros((32, 32), dtype=np.float32)
    for x in range(32):
        for z in range(32):
            col = labels32[:, z, x]
            air_frac[x, z] = (col == 0).sum() / len(col)

    results: Dict[str, Dict[str, float]] = {}
    for out_size in out_sizes:
        for strategy in ["mean", "stride"]:
            sub = subsample_noise_xz(noise_3d, level, out_size, strategy)
            if out_size < 32:
                scale = 32 // out_size
                sub_up = np.repeat(np.repeat(sub, scale, axis=1), scale, axis=2)[:, :32, :32]
            else:
                sub_up = sub[:, :32, :32]

            chan_r2s = []
            for ci in range(len(channels)):
                chan = sub_up[ci].ravel()
                af = air_frac.ravel()
                valid = ~(np.isnan(chan) | np.isnan(af))
                if valid.sum() < 10 or chan[valid].std() < 1e-9 or af[valid].std() < 1e-9:
                    chan_r2s.append(0.0)
                    continue
                r = np.corrcoef(chan[valid], af[valid])[0, 1]
                chan_r2s.append(r**2 if not np.isnan(r) else 0.0)

            key_name = f"{out_size}_{strategy}"
            results[key_name] = {
                "mean_channel_r2": float(np.mean(chan_r2s)),
                "max_channel_r2": float(np.max(chan_r2s)),
                "per_channel_r2": {
                    FIELD_NAMES[channels[i]]: chan_r2s[i] for i in range(len(channels))
                },
            }

    return results


# ── Main analysis ───────────────────────────────────────────────────


def _hm_chunk_range(conn: sqlite3.Connection) -> Tuple[int, int, int, int]:
    """Return (x_min, x_max, z_min, z_max) of heightmap chunk coordinates."""
    row = conn.execute(
        "SELECT MIN(chunk_x), MAX(chunk_x), MIN(chunk_z), MAX(chunk_z) FROM heightmaps"
    ).fetchone()
    return row  # type: ignore[return-value]


def find_samples(
    conn: sqlite3.Connection,
    level: int,
    max_samples: int = 200,
    *,
    surface_y_range: Tuple[int, int] = (40, 200),
) -> List[Tuple[int, int, int]]:
    """Find WorldSections that contain the surface and have heightmap data.

    Filters for:
    1. ws_y whose block range includes ``surface_y_range``.
    2. Chunks fully within heightmap coordinate range (for complete data).
    3. At least 80 % heightmap coverage within the WS.
    """
    shift = level + 1
    T = 1 << shift
    voxel_size = 1 << level
    expected_hm = T * T

    hm_x_min, hm_x_max, hm_z_min, hm_z_max = _hm_chunk_range(conn)

    # Determine which ws_y values contain the surface
    valid_ws_y: set[int] = set()
    for ws_y_row in conn.execute(
        "SELECT DISTINCT ws_y FROM voxy_sections WHERE level=?", (level,)
    ).fetchall():
        ws_y = ws_y_row[0]
        y_lo = (ws_y << shift) * 16  # first block Y
        y_hi = y_lo + 32 * voxel_size - 1  # last block Y
        if y_lo <= surface_y_range[1] and y_hi >= surface_y_range[0]:
            valid_ws_y.add(ws_y)

    if not valid_ws_y:
        return []

    rows = conn.execute(
        "SELECT ws_x, ws_y, ws_z FROM voxy_sections WHERE level=?",
        (level,),
    ).fetchall()

    valid: list[tuple[int, int, int]] = []
    for ws_x, ws_y, ws_z in rows:
        if len(valid) >= max_samples:
            break
        if ws_y not in valid_ws_y:
            continue

        # Require chunks fully inside the heightmap range
        cx_lo = ws_x << shift
        cz_lo = ws_z << shift
        if cx_lo < hm_x_min or cx_lo + T - 1 > hm_x_max:
            continue
        if cz_lo < hm_z_min or cz_lo + T - 1 > hm_z_max:
            continue

        hmc = conn.execute(
            "SELECT count(*) FROM heightmaps WHERE chunk_x >= ? AND chunk_x < ? "
            "AND chunk_z >= ? AND chunk_z < ?",
            (cx_lo, cx_lo + T, cz_lo, cz_lo + T),
        ).fetchone()[0]
        if hmc >= expected_hm * 0.8:
            valid.append((ws_x, ws_y, ws_z))

    return valid


def run_heightmap_analysis(conn: sqlite3.Connection, level: int, max_samples: int = 100):
    """Run complete heightmap resolution analysis for one level."""
    shift = level + 1
    voxel_size_blocks = 1 << level  # 1 for L0, 2 for L1, 4 for L2, 8 for L3, 16 for L4

    print(f"\n{'='*72}")
    print(f"  HEIGHTMAP RESOLUTION ANALYSIS — Level {level}")
    print(f"  Voxel size: {voxel_size_blocks} blocks")
    print(f"  Vanilla HM cells per voxel: {voxel_size_blocks}×{voxel_size_blocks}")
    print(f"  Full HM grid: {32*voxel_size_blocks}×{32*voxel_size_blocks} blocks")
    print(f"{'='*72}")

    samples = find_samples(conn, level, max_samples)
    print(f"  Found {len(samples)} labeled WorldSections")
    if not samples:
        return

    # Accumulate results
    all_variance: list[dict] = []
    all_rmse: Dict[int, List[float]] = {}
    all_strategy_r2: Dict[str, List[float]] = {}
    all_multi_r2: Dict[int, List[float]] = {}
    all_completeness: list[dict] = []

    for i, (ws_x, ws_y, ws_z) in enumerate(samples):
        labels = get_voxy_labels(conn, level, ws_x, ws_y, ws_z)
        if labels is None:
            continue

        # Check completeness (partial WS detection)
        comp = completeness_check(labels)
        all_completeness.append(comp)

        # Skip mostly-air WSs (no useful surface)
        if comp["non_air_cols"] < 100:
            continue

        sy_base = ws_y << shift
        y_min_block = sy_base * 16

        hm = tile_heightmap_blocks(conn, ws_x, ws_z, level)
        if np.all(np.isnan(hm)):
            continue

        # PRIMARY METRIC: Variance decomposition
        vd = variance_decomposition(hm, voxel_size_blocks)
        if vd["total_var"] > 0:
            all_variance.append(vd)

        # Reconstruction RMSE sweep
        if voxel_size_blocks >= 2:
            rmse = reconstruction_rmse_sweep(hm, voxel_size_blocks)
            for n, val in rmse.items():
                if not np.isnan(val):
                    all_rmse.setdefault(n, []).append(val)

        # Strategy comparison R² (fixed: with voxel_size_blocks)
        label_surface = surface_height_from_labels(labels, y_min_block, voxel_size_blocks)
        strat_r2 = r_squared_by_resolution(hm, label_surface, voxel_size_blocks)
        for k, v in strat_r2.items():
            all_strategy_r2.setdefault(k, []).append(v)

        # Multi-sample sweep
        if voxel_size_blocks >= 2:
            possible_n = [n for n in [1, 2, 4, 8, 16] if n <= voxel_size_blocks]
            mr2 = multi_sample_r_squared(hm, label_surface, voxel_size_blocks, possible_n)
            for k, v in mr2.items():
                all_multi_r2.setdefault(k, []).append(v)

        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(samples)} samples")

    # ── Report ─────────────────────────────────────────────────────
    if all_completeness:
        air_fracs = [c["all_air_frac"] for c in all_completeness]
        surface_fracs = [c["surface_frac"] for c in all_completeness]
        print("\n-- Completeness check (partial WS detection) --")
        print(
            f"  Mean all-air fraction:  {np.mean(air_fracs):.1%} "
            f"(range {np.min(air_fracs):.1%} - {np.max(air_fracs):.1%})"
        )
        print(f"  Mean surface fraction:  {np.mean(surface_fracs):.1%}")
        usable = sum(1 for c in all_completeness if c["non_air_cols"] >= 100)
        print(f"  Usable WSs (>=100 non-air cols): {usable}/{len(all_completeness)}")

    if all_variance:
        ratios = [v["ratio"] for v in all_variance]
        within_stds = [v["within_std"] for v in all_variance]
        between_stds = [v["between_std"] for v in all_variance]
        print("\n-- PRIMARY: Heightmap variance decomposition --")
        print(
            f"  Between-voxel / total ratio: {np.mean(ratios):.4f} " f"(std {np.std(ratios):.4f})"
        )
        print(
            f"  Within-voxel std:  {np.mean(within_stds):.2f} blocks "
            f"(std {np.std(within_stds):.2f})"
        )
        print(
            f"  Between-voxel std: {np.mean(between_stds):.2f} blocks "
            f"(std {np.std(between_stds):.2f})"
        )
        print(f"  ==> Ratio ~1.0 means ONE sample per voxel suffices.")
        print(f"  ==> Within-voxel std tells how much info is lost per sample.")

    if all_rmse:
        print("\n-- Reconstruction RMSE at different subsampling rates --")
        print(f"  {'N/axis':<10} {'Mean RMSE':>12} {'Std':>8} {'blocks'}")
        for n in sorted(all_rmse.keys()):
            vals = all_rmse[n]
            print(f"  {n:<10} {np.mean(vals):>12.3f} {np.std(vals):>8.3f} blocks")

    if all_strategy_r2:
        print("\n-- Aggregation strategy comparison (R^2 vs label surface) --")
        print(f"  {'Strategy':<12} {'Mean R^2':>10} {'Std':>8} {'N':>5}")
        for strat, vals in sorted(all_strategy_r2.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"  {strat:<12} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} {len(vals):>5}")

    if all_multi_r2:
        print("\n-- Samples-per-voxel-axis sweep (mean agg, R^2) --")
        print(f"  {'N samples':<12} {'Mean R^2':>10} {'Std':>8} {'delta from N=1':>15}")
        baseline = np.mean(all_multi_r2.get(1, [0.0]))
        for n in sorted(all_multi_r2.keys()):
            vals = all_multi_r2[n]
            m = np.mean(vals)
            print(f"  {n:<12} {m:>10.4f} {np.std(vals):>8.4f} {m - baseline:>+15.4f}")


def run_noise_analysis(conn: sqlite3.Connection, level: int, max_samples: int = 100):
    """Run noise channel spatial resolution analysis for one level."""
    voxel_size_blocks = 1 << level
    quarts_per_voxel = voxel_size_blocks // _CELL_WIDTH

    if quarts_per_voxel < 2:
        print(
            f"\n  L{level}: noise is at/below voxel resolution "
            f"({quarts_per_voxel} quart per voxel) -- skip resolution sweep"
        )
        return

    print(f"\n{'='*72}")
    print(f"  NOISE RESOLUTION ANALYSIS -- Level {level}")
    print(f"  Quart-cells per voxel XZ: {quarts_per_voxel}")
    print(f"  Full noise grid: {32*quarts_per_voxel}x{32*quarts_per_voxel} quarts XZ")
    print(f"{'='*72}")

    samples = find_samples(conn, level, max_samples)
    if not samples:
        print("  No samples found")
        return

    # Accumulate variance decomposition AND resolution sweep
    all_var_decomp: Dict[str, list[dict]] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for i, (ws_x, ws_y, ws_z) in enumerate(samples):
        labels = get_voxy_labels(conn, level, ws_x, ws_y, ws_z)
        if labels is None:
            continue

        noise, biome = tile_noise(conn, ws_x, ws_y, ws_z, level)

        # Variance decomposition per channel
        nvd = noise_variance_decomposition(noise, level)
        for ch_name, vd in nvd.items():
            all_var_decomp.setdefault(ch_name, []).append(vd)

        # Resolution sweep (R² vs air fraction)
        nr = noise_resolution_sweep(noise, labels, level)
        for key, val in nr.items():
            all_results.setdefault(key, []).append(val)

        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(samples)} samples")

    # Report: variance decomposition
    if all_var_decomp:
        print("\n-- PRIMARY: Noise variance decomposition (between / total) --")
        print(f"  {'Channel':<30} {'Ratio':>8} {'Within std':>12} {'Between std':>12}")
        for ch_name in sorted(
            all_var_decomp.keys(), key=lambda k: -np.mean([v["ratio"] for v in all_var_decomp[k]])
        ):
            ratios = [v["ratio"] for v in all_var_decomp[ch_name]]
            w_stds = [np.sqrt(v["within_var"]) for v in all_var_decomp[ch_name]]
            b_stds = [np.sqrt(v["between_var"]) for v in all_var_decomp[ch_name]]
            print(
                f"  {ch_name:<30} {np.mean(ratios):>8.4f} {np.mean(w_stds):>12.6f} "
                f"{np.mean(b_stds):>12.6f}"
            )

    # Report: resolution sweep
    if all_results:
        print("\n-- Noise resolution sweep (mean channel R^2 vs air-fraction) --")
        print(f"  {'Grid_Strat':<20} {'Mean ch R^2':>12} {'Max ch R^2':>12}")
        for key in sorted(all_results.keys()):
            mean_r2 = np.mean([r["mean_channel_r2"] for r in all_results[key]])
            max_r2 = np.mean([r["max_channel_r2"] for r in all_results[key]])
            print(f"  {key:<20} {mean_r2:>12.4f} {max_r2:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Spatial resolution analysis for Voxy LOD")
    parser.add_argument(
        "--db", type=str, default=None, help="Path to v7_dumps.db (auto-detects if not given)"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Max WorldSections to sample per level"
    )
    parser.add_argument(
        "--levels", type=str, default="2,3,4", help="Comma-separated levels to analyze"
    )
    args = parser.parse_args()

    # Auto-detect DB
    db_path = args.db
    if db_path is None:
        candidates = [
            Path(__file__).resolve().parents[1]
            / "tools"
            / "fabric-server"
            / "runtime"
            / "v7_dumps.db",
            Path(__file__).resolve().parents[1] / "data" / "v7_dumps.db",
        ]
        for c in candidates:
            if c.exists():
                db_path = str(c)
                break
    if db_path is None:
        print("ERROR: Could not find v7_dumps.db. Use --db to specify path.")
        sys.exit(1)

    print(f"Database: {db_path}")
    conn = sqlite3.connect(db_path)

    levels = [int(x) for x in args.levels.split(",")]

    for level in levels:
        run_heightmap_analysis(conn, level, args.samples)
        run_noise_analysis(conn, level, args.samples)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
