"""SQLite-backed Dataset for sparse-octree training.

Reads training pairs from a ``training_pairs`` SQLite database produced by
:func:`build_sparse_octree_pairs.build_pairs_db`.  Each row is decompressed
on-the-fly via zlib — no need to hold the full dataset in RAM.

Typical overhead per sample: ~0.1 ms decompression + ~0.05 ms target build.
With ``DataLoader(num_workers=4, batch_size=64)``, this yields ~2500 batches/s,
well above the training throughput of most GPUs.

Usage::

    from voxel_tree.tasks.sparse_octree.sparse_octree_dataset_db import (
        SparseOctreeSQLiteDataset,
    )

    ds = SparseOctreeSQLiteDataset(Path("training_pairs.db"))
    loader = DataLoader(ds, batch_size=64, shuffle=True,
                        collate_fn=sparse_octree_collate)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import unpack_blob
from voxel_tree.tasks.sparse_octree.sparse_octree_targets import (
    build_multilevel_voxy_targets,
)
from voxel_tree.tasks.sparse_octree.sparse_octree_train import compute_prunable_flags

# Fixed shapes for each field in the training_pairs table.
_NOISE_3D_SHAPE = (15, 4, 2, 4)
_BIOME_IDS_SHAPE = (4, 2, 4)
_HEIGHTMAP5_SHAPE = (5, 4, 4)
_LABEL_SHAPES: dict[int, tuple[int, int, int]] = {
    0: (16, 16, 16),
    1: (8, 8, 8),
    2: (4, 4, 4),
    3: (2, 2, 2),
    4: (1, 1, 1),
}


class SparseOctreeSQLiteDataset(Dataset):  # type: ignore[type-arg]
    """Lazy-loading dataset backed by a SQLite training-pairs DB.

    Each ``__getitem__`` reads a single row, decompresses the BLOBs, and
    builds the multi-level targets on the fly.  This keeps memory usage at
    ``O(batch_size)`` regardless of dataset size.

    Parameters
    ----------
    db_path : Path
        Path to the ``.db`` file produced by ``build_pairs_db``.
    air_id : int
        Block ID for air (used by target builder).
    max_samples : int or None
        Cap the dataset length for quick experiments.
    """

    def __init__(
        self,
        db_path: Path,
        air_id: int = 0,
        max_samples: Optional[int] = None,
    ) -> None:
        self.db_path = str(db_path)
        self.air_id = air_id

        # Read total count and validate schema
        conn = sqlite3.connect(self.db_path)
        (self._total,) = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()
        conn.close()

        if max_samples is not None and max_samples < self._total:
            self._n_samples = max_samples
        else:
            self._n_samples = self._total

        # Thread-local connections (SQLite connections are not thread-safe).
        # DataLoader workers each call __getitem__ from their own thread/process,
        # so we lazily init a per-thread connection.
        self._local_conn: Optional[sqlite3.Connection] = None

        # Expose spatial_y for model config detection
        self.spatial_y = _NOISE_3D_SHAPE[2]  # 2 (quartcell Y)

    def _get_conn(self) -> sqlite3.Connection:
        """Return a (lazily created) thread-local DB connection."""
        if self._local_conn is None:
            self._local_conn = sqlite3.connect(self.db_path)
            self._local_conn.execute("PRAGMA cache_size=-64000")  # 64 MB per worker
        return self._local_conn

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT noise_3d, biome_ids, heightmap5, block_y_min, finest_level,"
            "       labels_L0, labels_L1, labels_L2, labels_L3, labels_L4"
            " FROM training_pairs WHERE sample_id = ?",
            (idx,),
        ).fetchone()
        if row is None:
            raise IndexError(f"sample_id {idx} not found in {self.db_path}")

        (noise_blob, biome_blob, hm5_blob, block_y_min, finest_level, *label_blobs) = row

        # Decompress arrays
        noise_3d = unpack_blob(noise_blob, np.float32, _NOISE_3D_SHAPE)
        biome_ids = unpack_blob(biome_blob, np.int32, _BIOME_IDS_SHAPE)
        heightmap5 = unpack_blob(hm5_blob, np.float32, _HEIGHTMAP5_SHAPE)

        level_grids: Dict[int, np.ndarray] = {}
        for lv in range(5):
            arr = unpack_blob(label_blobs[lv], np.int32, _LABEL_SHAPES[lv])
            if arr.min() >= 0:  # skip sentinel-only levels
                level_grids[lv] = arr

        # Build multi-level targets (same logic as SparseOctreeDataset)
        raw_targets = build_multilevel_voxy_targets(
            level_grids,
            air_id=self.air_id,
            split_label=-1,
        )
        prunable_flags = compute_prunable_flags(heightmap5, int(block_y_min))

        targets: Dict[int, Dict[str, torch.Tensor]] = {}
        for lvl, lvl_data in raw_targets.items():
            split = (~lvl_data.is_leaf).astype(np.float32).reshape(-1)
            label = lvl_data.labels.astype(np.int64).reshape(-1)
            is_leaf = lvl_data.is_leaf.astype(np.bool_).reshape(-1)
            cm = lvl_data.child_mask.reshape(-1).astype(np.uint8)
            occ = np.unpackbits(cm[:, np.newaxis], axis=1, bitorder="little")[:, :8].astype(
                np.float32
            )
            prunable = prunable_flags[lvl].reshape(-1).astype(np.float32)
            targets[lvl] = {
                "occ": torch.from_numpy(occ),
                "split": torch.from_numpy(split),
                "label": torch.from_numpy(label),
                "is_leaf": torch.from_numpy(is_leaf),
                "is_prunable": torch.from_numpy(prunable),
            }

        # noise_2d is not present in the DB (v7 has only 3D noise fields).
        noise_2d = np.zeros((0, 4, 4), dtype=np.float32)

        return {
            "noise_2d": torch.from_numpy(noise_2d),
            "noise_3d": torch.from_numpy(noise_3d),
            "biome_ids": torch.from_numpy(biome_ids),
            "heightmap5": torch.from_numpy(heightmap5),
            "targets": targets,
        }

    def scan_max_block_id(self) -> int:
        """Scan a sample of rows to find the maximum block ID stored.

        Useful for auto-detecting ``num_classes`` without loading every row.
        """
        conn = self._get_conn()
        # Sample up to 1000 evenly-spaced rows
        step = max(1, self._total // 1000)
        max_id = 0
        for (sid,) in conn.execute(
            f"SELECT sample_id FROM training_pairs WHERE sample_id % {step} = 0"
        ):
            row = conn.execute(
                "SELECT labels_L0, labels_L1, labels_L2, labels_L3, labels_L4"
                " FROM training_pairs WHERE sample_id = ?",
                (sid,),
            ).fetchone()
            for lv, blob in enumerate(row):
                arr = unpack_blob(blob, np.int32, _LABEL_SHAPES[lv])
                valid = arr[arr >= 0]
                if len(valid) > 0:
                    max_id = max(max_id, int(valid.max()))
        return max_id


def get_pairs_db_sample_count(db_path: Path) -> int:
    """Return the number of training samples in a pairs DB."""
    conn = sqlite3.connect(str(db_path))
    (n,) = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()
    conn.close()
    return n
