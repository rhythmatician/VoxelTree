"""Block-head bias initialization from training-split class priors.

Supports the current octree pair-cache schema (``*_octree_pairs.npz``) with
``labels32`` targets and optional ``level`` filtering per model family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL_LEVELS: dict[str, tuple[int, ...]] = {
    "init": (4,),
    "refine": (1, 2, 3),
    "leaf": (0,),
}


def _as_numpy_int64(labels32: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(labels32, torch.Tensor):
        arr = labels32.detach().cpu().numpy()
    else:
        arr = np.asarray(labels32)
    return arr.astype(np.int64, copy=False)


def compute_block_counts(
    labels32: np.ndarray | torch.Tensor,
    vocab_size: int,
) -> np.ndarray:
    """Count block IDs across a ``labels32`` tensor/array."""
    flat = _as_numpy_int64(labels32).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot compute block priors from an empty labels32 array")

    flat = np.clip(flat, 0, vocab_size - 1)
    return np.bincount(flat, minlength=vocab_size).astype(np.int64, copy=False)


def compute_block_log_priors(
    labels32: np.ndarray | torch.Tensor,
    vocab_size: int,
    *,
    eps: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """Convert block counts into log-frequency priors."""
    counts = compute_block_counts(labels32, vocab_size)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute block priors when total voxel count is zero")

    freqs = counts.astype(np.float64) / float(total)
    log_priors = np.log(freqs + eps).astype(np.float32)

    if verbose:
        _print_counts_summary(counts, total, source="labels32")

    return log_priors


def compute_block_log_priors_from_pair_cache(
    data_dir: Path | str,
    vocab_size: int,
    *,
    split: str = "train",
    levels: Sequence[int] | None = None,
    eps: float = 1e-8,
    verbose: bool = True,
) -> np.ndarray:
    """Compute log priors from a production octree pair cache."""
    cache_path = Path(data_dir) / f"{split}_octree_pairs.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Octree pair cache not found: {cache_path}\n" "Run the data-prep pipeline first."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        if "labels32" not in data:
            raise KeyError(
                f"{cache_path.name} is missing required key 'labels32'. "
                f"Available keys: {sorted(data.files)}"
            )

        labels32 = data["labels32"]
        level_suffix = "all levels"
        if levels is not None:
            if "level" not in data:
                raise KeyError(
                    f"{cache_path.name} is missing required key 'level' for level filtering"
                )

            levels_arr = np.asarray(tuple(levels), dtype=np.int64)
            level_mask = np.isin(data["level"].astype(np.int64), levels_arr)
            if not np.any(level_mask):
                raise ValueError(
                    f"No samples found for levels {list(levels_arr)} in {cache_path.name}"
                )
            labels32 = labels32[level_mask]
            level_suffix = f"levels={list(levels_arr)}"

        log_priors = compute_block_log_priors(labels32, vocab_size, eps=eps, verbose=False)

    if verbose:
        counts = compute_block_counts(labels32, vocab_size)
        total = int(counts.sum())
        _print_counts_summary(
            counts,
            total,
            source=f"{cache_path.name} ({level_suffix})",
        )

    return log_priors


def init_block_head_bias(
    model: nn.Module,
    block_log_priors: np.ndarray | torch.Tensor,
    *,
    verbose: bool = True,
    model_name: str | None = None,
) -> None:
    """Copy log priors into ``model.block_head.bias``."""
    head = getattr(model, "block_head", None)
    if head is None:
        raise AttributeError(f"{type(model).__name__} has no attribute 'block_head'")
    bias = getattr(head, "bias", None)
    if bias is None:
        raise ValueError(f"{type(model).__name__}.block_head has no bias parameter")

    prior_tensor = torch.as_tensor(block_log_priors, dtype=bias.dtype, device=bias.device)
    if prior_tensor.ndim != 1:
        raise ValueError(f"block_log_priors must be 1D, got shape {tuple(prior_tensor.shape)}")
    if prior_tensor.shape[0] != bias.shape[0]:
        raise ValueError(
            f"Prior length {prior_tensor.shape[0]} does not match bias length {bias.shape[0]}"
        )

    with torch.no_grad():
        bias.copy_(prior_tensor)

    if verbose:
        label = model_name or type(model).__name__
        print(f"  ✓ Initialized {label}.block_head bias from log-frequency priors")


def init_models_from_train_priors(
    models: Mapping[str, nn.Module],
    data_dir: Path | str,
    vocab_size: int,
    *,
    split: str = "train",
    model_levels: Mapping[str, Sequence[int]] | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Initialize each model's ``block_head.bias`` from its split-specific priors."""
    priors_by_model: dict[str, np.ndarray] = {}
    level_map = dict(DEFAULT_MODEL_LEVELS)
    if model_levels is not None:
        level_map.update({name: tuple(levels) for name, levels in model_levels.items()})

    for name, model in models.items():
        levels = level_map.get(name)
        if levels is None:
            if verbose:
                print(f"  Skipping {name}: no level mapping configured for prior init")
            continue

        if verbose:
            print(f"Computing block priors for {name} from {split} split, levels={list(levels)}")
        priors = compute_block_log_priors_from_pair_cache(
            data_dir,
            vocab_size,
            split=split,
            levels=levels,
            verbose=verbose,
        )
        init_block_head_bias(model, priors, verbose=verbose, model_name=name)
        priors_by_model[name] = priors

    return priors_by_model


def _print_counts_summary(counts: np.ndarray, total: int, *, source: str) -> None:
    air_pct = 100.0 * counts[0] / max(total, 1)
    seen = int(np.count_nonzero(counts))
    print(f"Computed block priors from {source}:")
    print(f"  Total voxels: {total:,} | air={air_pct:.2f}% | seen={seen}/{len(counts)} classes")
    topk = np.argsort(counts)[::-1][:5]
    for idx in topk:
        pct = 100.0 * counts[idx] / max(total, 1)
        print(f"    id={int(idx):4d}  count={int(counts[idx]):12,}  ({pct:.3f}%)")
