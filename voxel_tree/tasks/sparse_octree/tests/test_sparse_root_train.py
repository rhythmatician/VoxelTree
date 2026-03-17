from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

# Provide a tiny LODiffusion stub for importing sparse_octree_train helpers.
if "LODiffusion.models.sparse_octree" not in sys.modules:
    sparse_octree_mod = types.ModuleType("LODiffusion.models.sparse_octree")

    class _DummyModel:  # pragma: no cover - import shim only
        pass

    sparse_octree_mod.SparseOctreeFastModel = _DummyModel
    sparse_octree_mod.SparseOctreeModel = _DummyModel
    models_mod = types.ModuleType("LODiffusion.models")
    models_mod.sparse_octree = sparse_octree_mod
    lod_mod = types.ModuleType("LODiffusion")
    lod_mod.models = models_mod
    sys.modules["LODiffusion"] = lod_mod
    sys.modules["LODiffusion.models"] = models_mod
    sys.modules["LODiffusion.models.sparse_octree"] = sparse_octree_mod

from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
    _finalize_metrics,
    _sparse_octree_loss,
    _update_batch_metrics,
)


def test_sparse_octree_loss_masks_material_to_leaf_nodes() -> None:
    # One level, two nodes: first internal (split=1), second leaf (split=0).
    # Material logits for internal node are intentionally "wrong" but must be ignored.
    preds = {
        2: {
            "split": torch.tensor([[8.0, -8.0]], dtype=torch.float32),
            "label": torch.tensor([[[0.0, 9.0], [9.0, 0.0]]], dtype=torch.float32),
        }
    }
    targets = {
        2: {
            "split": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "label": torch.tensor([[-1, 0]], dtype=torch.int64),
            "is_leaf": torch.tensor([[False, True]]),
        }
    }

    perfect_leaf = _sparse_octree_loss(
        preds,
        targets,
        split_weight=1.0,
        label_weight=0.35,
        level_split_weights={2: 1.0},
        level_label_weights={2: 1.0},
    )

    # Change only the internal-node label logits; loss should be unchanged because
    # internal labels are outside the leaf mask.
    preds_internal_changed = {
        2: {
            "split": preds[2]["split"].clone(),
            "label": preds[2]["label"].clone(),
        }
    }
    preds_internal_changed[2]["label"][0, 0] = torch.tensor([9.0, 0.0])

    still_perfect_leaf = _sparse_octree_loss(
        preds_internal_changed,
        targets,
        split_weight=1.0,
        label_weight=0.35,
        level_split_weights={2: 1.0},
        level_label_weights={2: 1.0},
    )

    assert torch.allclose(perfect_leaf, still_perfect_leaf, atol=1e-6)


def test_batch_metric_accumulator_reports_split_and_leaf_quality() -> None:
    preds = {
        3: {
            "split": torch.tensor([[2.0, -2.0, 1.0, -1.0]], dtype=torch.float32),
            "label": torch.tensor(
                [[[3.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 2.0]]],
                dtype=torch.float32,
            ),
        }
    }
    targets = {
        3: {
            "split": torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            "label": torch.tensor([[-1, 1, 0, -1]], dtype=torch.int64),
            "is_leaf": torch.tensor([[False, True, True, False]]),
        }
    }

    accum = {
        "split_tp": 0.0,
        "split_tn": 0.0,
        "split_fp": 0.0,
        "split_fn": 0.0,
        "leaf_correct": 0.0,
        "leaf_total": 0.0,
        "pred_leaf_nodes": 0.0,
        "gt_leaf_nodes": 0.0,
    }
    _update_batch_metrics(preds, targets, accum)
    metrics = _finalize_metrics(accum)

    assert metrics["split_acc"] == 0.5
    assert metrics["split_precision"] == 0.5
    assert metrics["split_recall"] == 0.5
    assert metrics["split_f1"] == 0.5
    assert metrics["split_over_rate"] == 0.5
    assert metrics["split_under_rate"] == 0.5
    assert metrics["leaf_acc"] == 1.0
    assert metrics["leaf_node_ratio"] == 1.0
