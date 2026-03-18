from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

# sparse_octree.py now lives at voxel_tree/tasks/sparse_octree/sparse_octree.py;
# no LODiffusion stub is needed — the module is in the same package.

from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
    SparseOctreeDataset,
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


def test_sparse_octree_dataset_handles_missing_noise_2d() -> None:
    """v7 NPZs have no noise_2d key — dataset must zero-fill with shape (N, 0, 4, 4)."""
    n = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"
        np.savez_compressed(
            npz_path,
            subchunk16=np.zeros((n, 16, 16, 16), dtype=np.int32),
            noise_3d=np.random.randn(n, 15, 4, 4, 4).astype(np.float32),
            biome_ids=np.zeros((n, 4, 4, 4), dtype=np.int32),
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.noise_2d.shape == (n, 0, 4, 4)
        assert ds.noise_3d.shape == (n, 15, 4, 4, 4)
        assert ds.spatial_y == 4
        assert len(ds) == n


def test_sparse_octree_dataset_loads_noise_2d_when_present() -> None:
    """Legacy NPZs that include noise_2d should load it normally."""
    n = 3
    c2d = 6
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"
        np.savez_compressed(
            npz_path,
            subchunk16=np.zeros((n, 16, 16, 16), dtype=np.int32),
            noise_2d=np.random.randn(n, c2d, 4, 4).astype(np.float32),
            noise_3d=np.random.randn(n, 13, 4, 2, 4).astype(np.float32),
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.noise_2d.shape == (n, c2d, 4, 4)
        assert ds.noise_3d.shape == (n, 13, 4, 2, 4)
        assert ds.spatial_y == 2
        assert len(ds) == n
