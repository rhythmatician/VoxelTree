"""Bias initialization from class-prior frequencies.

Initializes the block classification head's bias vector to log(frequency)
so that at epoch 0 the model already outputs the correct marginal
distribution (air ~75%, stone ~15%, etc.) before any gradient step.

Usage:
    from train.prior_init import init_block_head_bias
    init_block_head_bias(models, data_dir, vocab_size)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .progressive_lod_models import ProgressiveLODModel, ProgressiveLODModel0_Initial


def compute_block_frequencies(
    data_dir: Path,
    vocab_size: int,
    verbose: bool = True,
) -> np.ndarray:
    """Count every block (including air=0) across all training .npz pairs.

    Returns:
        log_freqs: float32 array [vocab_size] of log(freq + eps).
    """
    # choose the highest-version pair cache available (v1, v2, ...)
    candidates = sorted(data_dir.glob("train_pairs_v*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No pair cache (train_pairs_v*.npz) found in {data_dir}\n"
            "Run 'python data-cli.py build-pairs' first."
        )
    cache_file = candidates[-1]
    if verbose:
        print(f"Using pair cache: {cache_file.name}")

    data = np.load(cache_file)
    target_types = data["target_types"]  # (N, 16, 16, 16) int32

    if verbose:
        print(f"Computing block frequencies from {target_types.shape[0]:,} samples…")

    flat = target_types.reshape(-1).astype(np.int64)
    flat = np.clip(flat, 0, vocab_size - 1)

    counts = np.zeros(vocab_size, dtype=np.int64)
    np.add.at(counts, flat, 1)

    total = counts.sum()
    freqs = counts.astype(np.float64) / total
    log_freqs = np.log(freqs + 1e-8).astype(np.float32)

    if verbose:
        air_pct = 100.0 * counts[0] / total
        solid_pct = 100.0 - air_pct
        n_seen = int(np.sum(counts > 0))
        print(f"  Air (class 0): {air_pct:.1f}%  Solid: {solid_pct:.1f}%")
        print(f"  Classes with >=1 voxel: {n_seen} / {vocab_size}")
        # Top-5 most common
        top5 = np.argsort(counts)[::-1][:5]
        for idx in top5:
            pct = 100.0 * counts[idx] / total
            print(f"    id={idx:4d}  count={counts[idx]:12,}  ({pct:.2f}%)")

    return log_freqs


def init_block_head_bias(
    models: dict[str, torch.nn.Module],
    data_dir: Path,
    vocab_size: int,
    verbose: bool = True,
) -> None:
    """Set block_head bias to log-frequency priors for all 4 progressive models.

    Call AFTER model construction but BEFORE training or checkpoint loading.
    If resuming from a checkpoint the loaded weights will overwrite this init.
    """
    log_freqs = compute_block_frequencies(data_dir, vocab_size, verbose=verbose)
    bias_tensor = torch.from_numpy(log_freqs)

    for name, model in models.items():
        if isinstance(model, ProgressiveLODModel0_Initial):
            # Linear head: bias shape = [vocab_size]
            head = model.block_head
        elif isinstance(model, ProgressiveLODModel):
            # Conv3d head: bias shape = [vocab_size]
            head = model.block_head
        else:
            if verbose:
                print(f"  Skipping unknown model type for '{name}'")
            continue

        if head.bias is None:
            if verbose:
                print(f"  Warning: {name}.block_head has no bias — skipping")
            continue

        with torch.no_grad():
            head.bias.copy_(bias_tensor.to(head.bias.device))

        if verbose:
            print(f"  ✓ Initialized {name}.block_head bias from class priors")
