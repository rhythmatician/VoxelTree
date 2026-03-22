from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from voxel_tree.tasks.sparse_octree.sparse_octree_train import (
    SparseOctreeDataset,
    _finalize_metrics,
    _sparse_octree_loss,
    _update_batch_metrics,
    sparse_octree_collate,
)

from voxel_tree.tasks.sparse_octree.sparse_octree import (
    SparseOctreeFastModel,
)

from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (
    NOISE_FIELDS,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal multi-level NPZ for test fixtures
# ---------------------------------------------------------------------------

def _make_test_npz(
    path: Path,
    n: int = 4,
    *,
    n3d: int = 15,
    spatial_y: int = 2,
    include_noise_2d: bool = False,
    n2d: int = 0,
    include_heightmap5: bool = True,
    include_biome_ids: bool = True,
) -> Path:
    """Create a multi-level NPZ with synthetic data."""
    rng = np.random.default_rng(42)
    save_dict: dict[str, np.ndarray] = {
        "noise_3d": rng.standard_normal((n, n3d, 4, spatial_y, 4)).astype(np.float32),
        "finest_level": np.zeros(n, dtype=np.int32),
        "block_y_min": np.arange(n, dtype=np.int32) * 16,
    }
    for lv in range(5):
        side = 16 >> lv
        save_dict[f"labels_L{lv}"] = rng.integers(0, 100, (n, side, side, side)).astype(np.int32)
    if include_noise_2d:
        save_dict["noise_2d"] = rng.standard_normal((n, n2d, 4, 4)).astype(np.float32)
    if include_heightmap5:
        save_dict["heightmap5"] = rng.standard_normal((n, 5, 16, 16)).astype(np.float32)
    if include_biome_ids:
        save_dict["biome_ids"] = rng.integers(0, 10, (n, 4, spatial_y, 4)).astype(np.int32)
    np.savez_compressed(path, **save_dict)
    return path


def test_noise_fields_count_is_15() -> None:
    """NOISE_FIELDS must contain exactly 15 v7 RouterField channel names."""
    assert len(NOISE_FIELDS) == 15
    assert NOISE_FIELDS[0] == "temperature"
    assert NOISE_FIELDS[-1] == "vein_gap"


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
    """NPZs without noise_2d key — dataset must zero-fill with shape (N, 0, 4, 4)."""
    n = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(
            Path(tmpdir) / "test.npz", n=n,
            include_noise_2d=False,
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.noise_2d.shape == (n, 0, 4, 4)
        assert ds.noise_3d.shape == (n, 15, 4, 2, 4)
        assert ds.spatial_y == 2
        assert len(ds) == n


def test_sparse_octree_dataset_loads_noise_2d_when_present() -> None:
    """NPZs that include noise_2d should load it normally."""
    n = 3
    c2d = 6
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(
            Path(tmpdir) / "test.npz", n=n,
            include_noise_2d=True, n2d=c2d,
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.noise_2d.shape == (n, c2d, 4, 4)
        assert ds.noise_3d.shape == (n, 15, 4, 2, 4)
        assert ds.spatial_y == 2
        assert len(ds) == n


def test_sparse_octree_dataset_handles_missing_heightmaps() -> None:
    """NPZs without heightmap5 key should zero-fill with shape (N, 5, 16, 16)."""
    n = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(
            Path(tmpdir) / "test.npz", n=n,
            include_heightmap5=False,
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.heightmap5.shape == (n, 5, 16, 16)
        assert ds.heightmap5.sum() == 0.0


def test_sparse_octree_dataset_loads_heightmaps_when_present() -> None:
    """NPZs that include heightmap5 should load them as float32."""
    n = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(
            Path(tmpdir) / "test.npz", n=n,
            include_heightmap5=True,
        )
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.heightmap5.shape == (n, 5, 16, 16)
        # Verify it loaded actual data (not zeros)
        assert ds.heightmap5.sum() != 0.0


def test_collate_includes_heightmaps() -> None:
    """Collation must stack heightmap5 tensors into batched tensors."""
    n = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(Path(tmpdir) / "test.npz", n=n)
        ds = SparseOctreeDataset(npz_path, cache_targets=True)
        batch = sparse_octree_collate([ds[i] for i in range(n)])
        assert batch["heightmap5"].shape == (n, 5, 16, 16)


def test_fast_model_forward_with_heightmaps() -> None:
    """SparseOctreeFastModel forward pass must accept heightmap5 input."""
    B = 2
    model = SparseOctreeFastModel(n2d=0, n3d=15, hidden=32, num_classes=10, spatial_y=2)
    noise_2d = torch.zeros(B, 0, 4, 4)
    noise_3d = torch.randn(B, 15, 4, 2, 4)
    biome_ids = torch.zeros(B, 4, 2, 4, dtype=torch.long)
    hm5 = torch.randn(B, 5, 16, 16)

    out = model(noise_2d, noise_3d, biome_ids, hm5)

    assert isinstance(out, dict)
    assert set(out.keys()) == {0, 1, 2, 3, 4}
    assert out[4]["occ"].shape == (B, 1, 8)
    assert out[4]["split"].shape == (B, 1)
    assert out[4]["label"].shape == (B, 1, 10)
    assert out[0]["occ"].shape == (B, 4096, 8)
    assert out[0]["split"].shape == (B, 4096)
    assert out[0]["label"].shape == (B, 4096, 10)


def test_v7_15ch_npz_trains_with_auto_n3d() -> None:
    """v7 15-channel NPZ should auto-detect n3d=15 and build a matching model."""
    n = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = _make_test_npz(Path(tmpdir) / "test.npz", n=n, n3d=15)
        ds = SparseOctreeDataset(npz_path, cache_targets=False)
        assert ds.noise_3d.shape == (n, 15, 4, 2, 4)
        sample = ds[0]
        n3d = sample["noise_3d"].shape[0]
        assert n3d == 15
        model = SparseOctreeFastModel(n2d=0, n3d=n3d, hidden=16, num_classes=4, spatial_y=2)
        B = 2
        out = model(
            torch.zeros(B, 0, 4, 4),
            torch.randn(B, 15, 4, 2, 4),
            torch.zeros(B, 4, 2, 4, dtype=torch.long),
            torch.randn(B, 5, 16, 16),
        )
        assert isinstance(out, dict)
        assert set(out.keys()) == {0, 1, 2, 3, 4}
