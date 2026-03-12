"""Octree training dataset — loads from the new pair-cache NPZ format.

The pair cache contains 32³ WorldSection training samples at all 5 LOD levels
(L0–L4).  Each sample has:

- ``labels32``            — target block IDs ``int32[32, 32, 32]``
- ``parent_labels32``     — parent block IDs ``int32[32, 32, 32]`` (zeros for L4)
- ``heightmap32``         — conditioning ``float32[5, 32, 32]``
- ``biome32``             — biome IDs ``int32[32, 32]``
- ``y_position``          — section Y index ``int64``
- ``level``               — LOD level (0–4) ``int64``
- ``non_empty_children``  — 8-bit occupancy bitmask ``uint8``

Samples are routed to one of three models based on level:

===========  ===============  ==================
Level        Model type       Has parent?
===========  ===============  ==================
L4           ``init``         No
L3, L2, L1   ``refine``       Yes (+ level emb)
L0           ``leaf``         Yes
===========  ===============  ==================
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Level → model type classification ────────────────────────────────


def _model_type_for_level(level: int) -> str:
    """Map LOD level to model type string."""
    if level == 4:
        return "init"
    elif level in (1, 2, 3):
        return "refine"
    elif level == 0:
        return "leaf"
    else:
        raise ValueError(f"Invalid LOD level: {level} (expected 0-4)")


# ── Dataset ──────────────────────────────────────────────────────────


class OctreeDataset(Dataset[Dict[str, Any]]):
    """Dataset for octree training from pair-cache NPZ files.

    Loads a pre-built pair cache (``{split}_octree_pairs.npz``) produced
    by the data-prep pipeline.  Use ``data-cli.py`` to create it.

    Samples are returned as dicts with tensors ready for the training loop.
    The ``level`` field determines which model processes the sample.

    Args:
        data_dir: Directory containing the pair cache.
        split: ``"train"`` or ``"val"``.
        level_sampling_weights: Optional per-level sampling weights.
            Keys are levels 0–4, values are relative weights.
            If ``None``, uniform sampling across all available samples.
        model_type: If set (``"init"``, ``"refine"``, or ``"leaf"``), load
            the per-model cache ``{model_type}_{split}_octree_pairs.npz``
            when it exists, falling back to the unified cache with level
            filtering.  ``None`` loads the unified cache (all levels).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        level_sampling_weights: Optional[Dict[int, float]] = None,
        model_type: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.model_type = model_type

        # Resolve cache path: prefer model-specific file, then unified
        _VALID_MODEL_TYPES = ("init", "refine", "leaf")
        if model_type is not None and model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type {model_type!r}. Valid: {_VALID_MODEL_TYPES} or None"
            )

        if model_type is not None:
            specific = self.data_dir / f"{model_type}_{split}_octree_pairs.npz"
            unified = self.data_dir / f"{split}_octree_pairs.npz"
            if specific.exists():
                cache_path = specific
            elif unified.exists():
                cache_path = unified
                # Will filter to relevant levels after loading
            else:
                raise FileNotFoundError(
                    f"No pair cache found for model_type={model_type!r}.\n\n"
                    f"Tried:\n"
                    f"  {specific}\n"
                    f"  {unified}\n\n"
                    f"Run the data-prep pipeline first:\n"
                    f"    python scripts/build_octree_pairs.py --model-type {model_type}"
                    f" --data-dir {self.data_dir}\n"
                )
        else:
            cache_path = self.data_dir / f"{split}_octree_pairs.npz"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Octree pair cache not found: {cache_path}\n\n"
                    f"Run the data-prep pipeline first:\n"
                    f"    python data-cli.py dataprep --data-dir {self.data_dir}\n"
                )

        self._load_cache(cache_path)

        # Levels relevant to this model_type (None → all levels)
        _model_levels: Optional[set] = None
        if model_type == "init":
            _model_levels = {4}
        elif model_type == "refine":
            _model_levels = {1, 2, 3}
        elif model_type == "leaf":
            _model_levels = {0}

        # Build per-level index for weighted sampling
        self._level_indices: Dict[int, List[int]] = {}
        for i in range(self._n_samples):
            lv = int(self._levels[i])
            if _model_levels is not None and lv not in _model_levels:
                continue  # skip levels that belong to a different model
            if lv not in self._level_indices:
                self._level_indices[lv] = []
            self._level_indices[lv].append(i)

        # Update _n_samples to count only relevant samples (matters when
        # model_type is set but a unified cache was loaded)
        relevant_count = sum(len(v) for v in self._level_indices.values())
        if relevant_count < self._n_samples:
            print(
                f"  Filtered to {relevant_count:,} of {self._n_samples:,} samples "
                f"for model_type={model_type!r}"
            )
            self._n_samples = relevant_count

        self._available_levels = sorted(self._level_indices.keys())

        if level_sampling_weights is not None:
            self._sampling_weights = level_sampling_weights
        else:
            # Default: weight proportional to level count (uniform per sample)
            self._sampling_weights = {lv: 1.0 for lv in self._available_levels}

        self._weight_levels = [
            lv for lv in self._available_levels if self._sampling_weights.get(lv, 0) > 0
        ]
        self._weight_values = [self._sampling_weights[lv] for lv in self._weight_levels]

        # Flat list of all relevant indices for the direct-index fallback in
        # __getitem__.  When model_type is set and a unified cache is loaded,
        # this contains only the subset of indices with relevant levels.
        self._all_relevant_indices: List[int] = sorted(
            idx for indices in self._level_indices.values() for idx in indices
        )

    _REQUIRED_KEYS: tuple = (
        "labels32",
        "parent_labels32",
        "heightmap32",
        "biome32",
        "y_position",
        "level",
        "non_empty_children",
    )

    def _load_cache(self, path: Path) -> None:
        """Load pair cache arrays into memory."""
        t0 = time.time()
        print(f"Loading octree pair cache: {path} ...")
        data = np.load(path, allow_pickle=False)

        # Validate required keys before accessing
        missing = [k for k in self._REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(
                f"Octree pair cache {path} is missing required key(s): {missing}\n"
                f"Available keys: {sorted(data.files)}\n"
                f"Re-run data-cli.py to rebuild the cache."
            )

        # Validate array shapes
        labels = data["labels32"]
        if labels.ndim != 4 or labels.shape[1:] != (32, 32, 32):
            raise ValueError(f"labels32 must have shape [N, 32, 32, 32], got {labels.shape}")
        heightmap = data["heightmap32"]
        if heightmap.ndim != 4 or heightmap.shape[1:] != (5, 32, 32):
            raise ValueError(f"heightmap32 must have shape [N, 5, 32, 32], got {heightmap.shape}")

        self._labels32: np.ndarray = data["labels32"]  # [N, 32, 32, 32]
        self._parent_labels32: np.ndarray = data["parent_labels32"]  # [N, 32, 32, 32]
        self._heightmap32: np.ndarray = data["heightmap32"]  # [N, 5, 32, 32]
        self._biome32: np.ndarray = data["biome32"]  # [N, 32, 32]
        self._y_position: np.ndarray = data["y_position"]  # [N]
        self._levels: np.ndarray = data["level"]  # [N]
        self._non_empty_children: np.ndarray = data["non_empty_children"]  # [N]

        self._n_samples = len(self._labels32)
        elapsed = time.time() - t0
        print(f"  Loaded {self._n_samples:,} samples in {elapsed:.1f}s")

        # Print per-level breakdown
        unique, counts = np.unique(self._levels, return_counts=True)
        for lv, cnt in zip(unique, counts):
            mt = _model_type_for_level(int(lv))
            print(f"    L{lv} ({mt}): {cnt:,} samples")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        With 10 % probability uses the exact index (for DataLoader shuffle
        compatibility); otherwise uses weighted level sampling.
        """
        if random.random() < 0.1:
            # Use _all_relevant_indices so filtered datasets (model_type set)
            # never accidentally return samples from other models' levels.
            i = self._all_relevant_indices[idx % len(self._all_relevant_indices)]
        else:
            # Pick level weighted, then random sample within that level
            chosen_level = random.choices(self._weight_levels, weights=self._weight_values, k=1)[0]
            indices = self._level_indices[chosen_level]
            i = random.choice(indices)

        level = int(self._levels[i])
        model_type = _model_type_for_level(level)

        sample: Dict[str, Any] = {
            "labels32": torch.from_numpy(self._labels32[i].astype(np.int64)),
            "heightmap32": torch.from_numpy(self._heightmap32[i].astype(np.float32)),
            "biome32": torch.from_numpy(self._biome32[i].astype(np.int64)),
            "y_position": torch.tensor(int(self._y_position[i]), dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.long),
            "non_empty_children": torch.tensor(
                int(self._non_empty_children[i]),  # uint8→int: safe for 8-bit occupancy masks
                dtype=torch.long,
            ),
            "model_type": model_type,  # str for collate routing
        }

        # Parent context (zeros for init model, real data for refine/leaf)
        if model_type in ("refine", "leaf"):
            sample["parent_labels32"] = torch.from_numpy(self._parent_labels32[i].astype(np.int64))
        else:
            # Init model (L4): parent is zeros — placeholder for uniform collation
            sample["parent_labels32"] = torch.zeros(32, 32, 32, dtype=torch.long)

        return sample


# ── Collate function ─────────────────────────────────────────────────


def _stack_group(samples: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
    """Stack a homogeneous group of samples into a batched dict."""
    batch: Dict[str, Any] = {
        "labels32": torch.stack([s["labels32"] for s in samples]),
        "parent_labels32": torch.stack([s["parent_labels32"] for s in samples]),
        "heightmap32": torch.stack([s["heightmap32"] for s in samples]),
        "biome32": torch.stack([s["biome32"] for s in samples]),
        "y_position": torch.stack([s["y_position"] for s in samples]),
        "level": torch.stack([s["level"] for s in samples]),
        "non_empty_children": torch.stack([s["non_empty_children"] for s in samples]),
        "model_type": model_type,
    }
    return batch


def collate_octree_batch(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate function grouping samples by model type.

    Within a DataLoader batch, samples may belong to different model types
    (init / refine / leaf).  This collate function keeps only the
    **majority model type** to ensure each batch routes to exactly one
    model.  Minority samples are dropped — the DataLoader's shuffling
    ensures all samples are visited across epochs.

    Refine-model samples at different levels (L1, L2, L3) can batch
    together since they share the same model with a level embedding.

    Args:
        samples: List of per-sample dicts from :class:`OctreeDataset`.

    Returns:
        Batched dict with stacked tensors and a ``model_type`` string key.
    """
    if not samples:
        return {"model_type": "empty", "batch_size": 0}

    # Classify by model type
    groups: Dict[str, List[Dict[str, Any]]] = {
        "init": [],
        "refine": [],
        "leaf": [],
    }
    for s in samples:
        mt = s["model_type"]
        assert isinstance(mt, str)
        groups[mt].append(s)

    # Pick the largest group
    model_type = max(groups, key=lambda k: len(groups[k]))
    group = groups[model_type]

    if not group:
        # Shouldn't happen, but be safe
        return {"model_type": "empty", "batch_size": 0}

    return _stack_group(group, model_type)
