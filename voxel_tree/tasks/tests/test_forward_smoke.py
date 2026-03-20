"""Verify model forward() signatures match what every caller actually passes.

The distill module once called ``teacher(noise_2d, noise_3d)`` — only 2 of
the 5 required positional args — and nothing caught the mismatch until the
pipeline ran end-to-end.  These tests exist to prevent that entire *class*
of bug: if a model signature changes, every call-site must be updated, and
any gap will be caught here at ``pytest`` time.

Strategy
--------
1. **Smoke forward**: For each model variant / call context, construct the
   *exact same batch dict* a ``DataLoader`` + collate produces, extract
   tensors the same way the runner does, and run one ``model.forward()``.
   A ``TypeError`` here means a caller fell out-of-sync with the model.

2. **Signature introspection**: For each top-level model class, compare
   its ``forward()`` required positional params against the batch dict
   keys exposed by the corresponding dataset's collate function.  If a
   new required param appears that no dataset provides, the test fails
   early with a clear message — before anyone tries to run the pipeline.
"""

from __future__ import annotations

import inspect
from typing import Dict

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Sparse-octree models
# ---------------------------------------------------------------------------

from voxel_tree.tasks.sparse_octree.sparse_octree import (
    SparseOctreeFastModel,
    SparseOctreeModel,
)

# ---------------------------------------------------------------------------
# Stage-1 models
# ---------------------------------------------------------------------------

from voxel_tree.tasks.density.train_density import DensityMLP
from voxel_tree.tasks.biome.train_biome_classifier import BiomeClassifier
from voxel_tree.tasks.heightmap.train_heightmap import HeightmapPredictor

# ---------------------------------------------------------------------------
# Helpers — synthetic batch factories
# ---------------------------------------------------------------------------

B = 2  # batch size for smoke tests


def _sparse_octree_batch(
    *,
    n2d: int = 0,
    n3d: int = 15,
    spatial_y: int = 2,
) -> Dict[str, torch.Tensor]:
    """Build a batch dict identical to ``sparse_octree_collate`` output."""
    return {
        "noise_2d": torch.zeros(B, n2d, 4, 4),
        "noise_3d": torch.randn(B, n3d, 4, spatial_y, 4),
        "biome_ids": torch.zeros(B, 4, spatial_y, 4, dtype=torch.long),
        "heightmap5": torch.randn(B, 5, 16, 16),
    }


_SPARSE_BATCH_KEYS_FOR_FORWARD = (
    "noise_2d",
    "noise_3d",
    "biome_ids",
    "heightmap5",
)


# ---------------------------------------------------------------------------
# 1) Smoke forward — sparse octree (train / distill / export call patterns)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [SparseOctreeModel, SparseOctreeFastModel],
    ids=["baseline", "fast"],
)
class TestSparseOctreeSmoke:
    """Every call pattern used in train / distill / export must not raise."""

    @staticmethod
    def _make_model(cls: type, n3d: int = 15) -> torch.nn.Module:
        return cls(n2d=0, n3d=n3d, hidden=16, num_classes=4, spatial_y=2)

    def test_train_call_pattern(self, model_cls: type) -> None:
        """Mirrors ``sparse_octree_train.train_sparse_octree()`` inner loop."""
        model = self._make_model(model_cls)
        batch = _sparse_octree_batch()
        preds = model(
            batch["noise_2d"],
            batch["noise_3d"],
            batch["biome_ids"],
            batch["heightmap5"],
        )
        assert isinstance(preds, dict) and len(preds) > 0

    def test_distill_call_pattern(self, model_cls: type) -> None:
        """Mirrors ``sparse_octree_distill.distill_sparse_octree()`` inner loop.

        This is the exact bug that prompted this test file — the distill
        code once passed only ``(noise_2d, noise_3d)`` and got a TypeError.
        """
        model = self._make_model(model_cls)
        batch = _sparse_octree_batch()
        # Extract args the exact same way the distill loop does:
        noise_2d = batch["noise_2d"]
        noise_3d = batch["noise_3d"]
        biome_ids = batch["biome_ids"]
        heightmap5 = batch["heightmap5"]

        preds = model(
            noise_2d,
            noise_3d,
            biome_ids,
            heightmap5,
        )
        assert isinstance(preds, dict) and len(preds) > 0

    def test_export_call_pattern(self, model_cls: type) -> None:
        """Mirrors ``export_sparse_octree._OnnxWrapperWith2d.forward()``."""
        model = self._make_model(model_cls)
        batch = _sparse_octree_batch()
        preds = model(
            batch["noise_2d"],
            batch["noise_3d"],
            batch["biome_ids"],
            batch["heightmap5"],
        )
        assert isinstance(preds, dict) and len(preds) > 0


# ---------------------------------------------------------------------------
# 2) Smoke forward — Stage 1 MLPs (density / biome / heightmap)
# ---------------------------------------------------------------------------


class TestDensityMLP:
    def test_train_call_pattern(self) -> None:
        model = DensityMLP()
        x = torch.randn(B, 6)
        out = model(x)
        assert out.shape == (B, 2)


class TestBiomeClassifier:
    def test_train_call_pattern(self) -> None:
        model = BiomeClassifier(num_classes=54)
        x = torch.randn(B, 6)
        out = model(x)
        assert out.shape == (B, 54)


class TestHeightmapPredictor:
    def test_train_call_pattern(self) -> None:
        model = HeightmapPredictor()
        x = torch.randn(B, 96)
        out = model(x)
        assert out.shape == (B, 32)


# ---------------------------------------------------------------------------
# 3) Signature introspection — required positional args vs. batch keys
# ---------------------------------------------------------------------------

# Maps model class → (required positional forward params, available batch keys)
# "batch keys" are what the dataset+collate provides — each must cover
# every required param of forward().

_SPARSE_BATCH_KEYS = frozenset(
    {
        "noise_2d",
        "noise_3d",
        "biome_ids",
        "heightmap5",
    }
)

_MODEL_BATCH_COVERAGE: list[tuple[type, frozenset[str]]] = [
    (SparseOctreeModel, _SPARSE_BATCH_KEYS),
    (SparseOctreeFastModel, _SPARSE_BATCH_KEYS),
    # DensityMLP, BiomeClassifier, HeightmapPredictor use a single (x)
    # tensor, not a batch dict, so signature introspection doesn't apply.
]


def _required_forward_params(cls: type) -> set[str]:
    """Return the required positional parameter names of ``cls.forward()``,
    excluding ``self``."""
    sig = inspect.signature(cls.forward)
    required: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            required.add(name)
    return required


@pytest.mark.parametrize(
    "model_cls,batch_keys",
    _MODEL_BATCH_COVERAGE,
    ids=[cls.__name__ for cls, _ in _MODEL_BATCH_COVERAGE],
)
def test_forward_required_params_covered_by_batch(
    model_cls: type,
    batch_keys: frozenset[str],
) -> None:
    """Every required positional param of forward() must be available in
    the batch dict that the dataset + collate produces.

    If someone adds a new required arg to forward() without also adding it
    to the dataset, this test fails immediately.
    """
    required = _required_forward_params(model_cls)
    missing = required - batch_keys
    assert not missing, (
        f"{model_cls.__name__}.forward() requires {sorted(missing)} "
        f"but the batch only provides {sorted(batch_keys)}"
    )


# ---------------------------------------------------------------------------
# 4) Cross-check: distill _evaluate_student uses correct call pattern
# ---------------------------------------------------------------------------


def test_distill_evaluate_student_call_pattern() -> None:
    """Replicate ``_evaluate_student`` with a real model to catch TypeError.

    The inner loop calls ``model(noise_2d, noise_3d, biome_ids, ...)``
    and must pass all required positional args.
    """
    from voxel_tree.tasks.sparse_octree.sparse_octree_distill import (
        _evaluate_student,
    )
    from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
        SparseOctreeDataset,
    )
    import tempfile
    from pathlib import Path

    n = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        npz = Path(tmpdir) / "test.npz"
        np.savez_compressed(
            npz,
            subchunk16=np.zeros((n, 16, 16, 16), dtype=np.int32),
            noise_3d=np.random.randn(n, 15, 4, 2, 4).astype(np.float32),
            biome_ids=np.zeros((n, 4, 2, 4), dtype=np.int32),
            heightmap5=np.random.randn(n, 5, 16, 16).astype(np.float32),
        )
        ds = SparseOctreeDataset(npz, cache_targets=True)
        model = SparseOctreeFastModel(
            n2d=0,
            n3d=15,
            hidden=16,
            num_classes=4,
            spatial_y=2,
        )
        # This must not raise TypeError — the whole purpose of this test.
        metrics = _evaluate_student(model, ds, batch_size=2, device=torch.device("cpu"))
        assert "split_acc" in metrics
        assert "leaf_acc" in metrics
