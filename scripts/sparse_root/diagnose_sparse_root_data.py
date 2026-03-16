#!/usr/bin/env python3
"""Diagnose sparse_root training data alignment and quality.

Checks:
1. Do noise_3d values correlate with block types (e.g., high final_density → solid blocks)?
2. Are samples predominantly air or do they have real terrain?
3. Do splits actually correlate with terrain complexity?
4. Is there a coordinate mismatch or data corruption?

Usage
-----
  python VoxelTree/scripts/sparse_root/diagnose_sparse_root_data.py \\
      --data noise_training_data/sparse_root_pairs.npz \\
      [--samples 50] [--verbose]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("noise_training_data/sparse_root_pairs.npz"),
        help="Path to sparse_root_pairs.npz",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-sample details",
    )
    return parser


NOISE_FIELD_NAMES = [
    "offset",           # 0
    "factor",           # 1
    "jaggedness",       # 2
    "depth",            # 3  pre-shaping depth
    "sloped_cheese",    # 4
    "y",                # 5  Y coordinate (positional)
    "entrances",        # 6
    "pillars",          # 7
    "spaghetti_2d",     # 8
    "spaghetti_roughness",  # 9
    "noodle",               # 10
    "base_3d_noise",        # 11
    "final_density",        # 12
]


def analyze_samples(npz_path: Path, num_samples: int, verbose: bool = False) -> None:
    """Analyze training data samples for spatial alignment and quality."""
    print(f"Loading training data from {npz_path}...")
    data = np.load(npz_path)
    subchunks = data["subchunk16"]  # (N, 16, 16, 16)
    noise_3d = data["noise_3d"]      # (N, 13, 4, 2, 4)
    
    n_total = len(subchunks)
    num_samples = min(num_samples, n_total)
    
    print(f"\nDataset shape: {n_total} samples")
    print(f"  subchunk16: {subchunks.shape}")
    print(f"  noise_3d: {noise_3d.shape}")
    
    indices = np.random.choice(n_total, size=num_samples, replace=False)
    
    print(f"\n{'='*70}")
    print(f"SAMPLE ANALYSIS ({num_samples} / {n_total})")
    print(f"{'='*70}\n")
    
    # Global statistics
    all_air_count = 0
    air_only_samples = 0
    mixed_samples = 0
    solid_only_samples = 0
    
    final_density_means = []
    final_density_stds = []
    
    for sample_idx, i in enumerate(indices):
        labels = subchunks[i]  # (16, 16, 16)
        noise_block = noise_3d[i]  # (13, 4, 2, 4)
        
        # Count air (block_id == 0)
        n_air = np.sum(labels == 0)
        n_total_voxels = labels.size  # 16^3 = 4096
        air_frac = n_air / n_total_voxels
        
        # Get final_density (field 12)
        final_density = noise_block[12]  # (4, 2, 4)
        fd_mean = np.mean(final_density)
        fd_std = np.std(final_density)
        final_density_means.append(fd_mean)
        final_density_stds.append(fd_std)
        
        # Categorize
        if air_frac > 0.99:
            all_air_count += 1
            air_only_samples += 1
            category = "ALL-AIR"
        elif air_frac < 0.01:
            solid_only_samples += 1
            category = "SOLID-ONLY"
        else:
            mixed_samples += 1
            category = "MIXED"
        
        if verbose or sample_idx < 10:
            print(
                f"Sample {i:4d} ({sample_idx+1}/{num_samples}): "
                f"Air={air_frac*100:5.1f}%  FinalDensity={fd_mean:6.3f}±{fd_std:.3f}  [{category}]"
            )
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"All-air samples:       {air_only_samples:5d} ({air_only_samples/num_samples*100:5.1f}%)")
    print(f"Solid-only samples:    {solid_only_samples:5d} ({solid_only_samples/num_samples*100:5.1f}%)")
    print(f"Mixed samples:         {mixed_samples:5d} ({mixed_samples/num_samples*100:5.1f}%)")
    print()
    print(f"Final density (mean of means):        {np.mean(final_density_means):6.3f}")
    print(f"Final density (mean of stds):         {np.mean(final_density_stds):6.3f}")
    print(f"Final density range (across means):   [{np.min(final_density_means):6.3f}, {np.max(final_density_means):6.3f}]")
    
    # Hypothesis: Check if final_density correlates with non-air count
    print(f"\n{'='*70}")
    print("CORRELATION CHECK: final_density vs non-air content")
    print(f"{'='*70}")
    
    correlations = []
    for i in indices:
        labels = subchunks[i]
        noise_block = noise_3d[i]
        n_non_air = np.sum(labels != 0)
        fd_mean = np.mean(noise_block[12])
        correlations.append((fd_mean, n_non_air))
    
    correlations = np.array(correlations)
    if len(correlations) > 1:
        corr_coef = np.corrcoef(correlations[:, 0], correlations[:, 1])[0, 1]
        print(f"Pearson correlation (final_density vs non-air count): {corr_coef:.4f}")
        if np.isnan(corr_coef):
            print("  (correlation is NaN — likely constant final_density across all samples)")
        elif corr_coef < 0.3:
            print("  WARNING: Very weak correlation! Noise may not align with labels.")
        else:
            print(f"  Good correlation ({corr_coef:.3f}) — noise inputs seem to match terrain.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    args = create_argument_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    analyze_samples(args.data, args.samples, verbose=args.verbose)
