"""Training utilities for the noise-conditioned sparse-root octree model.

Dataset layout expected in the .npz file
-----------------------------------------
subchunk16          : int32   [N, 16, 16, 16]   dense voxel labels
noise_2d            : float32 [N, C2, 4, 4]     vanilla 2-D climate fields per subchunk
noise_3d            : float32 [N, C3, 4, Y, 4]  vanilla 3-D volumetric fields (Y=2 for v7, matching vanilla cellHeight=8)
biome_ids           : int32   [N, 4, Y, 4]      discrete biome IDs at spatial resolution
heightmap5          : float32 [N, 5, 16, 16]    5-plane heightmap (surface_norm, ocean_approx, slope_x, slope_z, curvature)

Legacy NPZs with separate ``heightmap_surface`` / ``heightmap_ocean_floor``
arrays are auto-converted to the 5-plane format on load.

Targets are built on-the-fly by ``build_sparse_octree_targets`` and stored as
flat 1-D tensors per level so they can be directly compared against the model's
flat [B, N] output tensors.

  level 4 (side 1): N=1    occ [B,1,8],    split [B,1],     label [B,1]
  level 3 (side 2): N=8    occ [B,8,8],    split [B,8],     label [B,8]
  level 2 (side 4): N=64   occ [B,64,8],   split [B,64],    label [B,64]
  level 1 (side 8): N=512  occ [B,512,8],  split [B,512],   label [B,512]
  level 0 (side16): N=4096 occ [B,4096,8], split [B,4096],  label [B,4096]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .sparse_octree import SparseOctreeFastModel, SparseOctreeModel
from .sparse_octree_targets import build_sparse_octree_targets

# ---------------------------------------------------------------------------
# Geometric pruning helpers (Phase 5)
# ---------------------------------------------------------------------------

# Height-range constant matching LodGenerationService / build_sparse_octree_pairs.
_HEIGHT_RANGE = 320.0


def compute_prunable_flags(
    heightmap5: np.ndarray,
    block_y_min: int,
    max_level: int = 4,
    cube_side: int = 16,
) -> Dict[int, np.ndarray]:
    """Compute per-node ``is_prunable`` flags for geometric pruning.

    A node is *prunable* when its block-Y range does not intersect the
    heightmap surface range within its XZ footprint — exactly matching
    the zero-margin pruning logic in ``OctreeQueue.spawnChildren()``.

    Prunable above surface: ``node_y_min >= local_surf_max``
    Prunable below surface: ``node_y_max <= local_surf_min``

    Parameters
    ----------
    heightmap5 : ndarray (5, 16, 16)
        5-plane heightmap.  Channel 0 is ``surface_norm = raw_surface / 320``.
    block_y_min : int
        Absolute world block-Y of this subchunk's bottom edge.
    max_level : int
        Coarsest octree level (default 4).
    cube_side : int
        Voxel cube edge length (default 16).

    Returns
    -------
    Dict[level → bool ndarray (side, side, side)] where ``True`` ⇒ prunable.
    """
    # Recover raw surface heights in block coordinates.
    raw_surface = heightmap5[0] * _HEIGHT_RANGE  # (16, 16)

    result: Dict[int, np.ndarray] = {}
    for lvl in range(max_level, -1, -1):
        side = 2 ** (max_level - lvl)  # nodes per axis at this level
        cell = cube_side // side  # blocks per node per axis
        prunable = np.zeros((side, side, side), dtype=np.bool_)

        for y_idx in range(side):
            node_y_min = block_y_min + y_idx * cell
            node_y_max = block_y_min + (y_idx + 1) * cell
            for z_idx in range(side):
                for x_idx in range(side):
                    # Surface heights in this node's XZ footprint (16×16 hm)
                    hm_slice = raw_surface[
                        z_idx * cell : (z_idx + 1) * cell,
                        x_idx * cell : (x_idx + 1) * cell,
                    ]
                    surf_min = float(hm_slice.min())
                    surf_max = float(hm_slice.max())
                    # Zero-margin: prune if entirely above OR entirely below
                    if node_y_min >= surf_max or node_y_max <= surf_min:
                        prunable[y_idx, z_idx, x_idx] = True

        result[lvl] = prunable
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SparseOctreeDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for noise-conditioned sparse-root supervision."""

    def __init__(self, npz_path: Path, air_id: int = 0, cache_targets: bool = True, max_samples: Optional[int] = None) -> None:
        data = np.load(npz_path)
        self.subchunks = data["subchunk16"].astype(np.int32)  # (N,16,16,16)
        if max_samples is not None and max_samples < len(self.subchunks):
            self.subchunks = self.subchunks[:max_samples]
        self.noise_3d = data["noise_3d"].astype(np.float32)  # (N,C3,4,Y,4)
        if max_samples is not None and max_samples < len(self.noise_3d):
            self.noise_3d = self.noise_3d[:max_samples]
        # Detect spatial_y from noise_3d shape (index 2 = X=4, index 3 = Y)
        self.spatial_y = self.noise_3d.shape[3]  # 4 for v7, 2 for legacy
        _n_limit = len(self.subchunks)
        if "noise_2d" in data:
            self.noise_2d = data["noise_2d"].astype(np.float32)[:_n_limit]  # (N,C2,4,4)
        else:
            # v7 pipeline has no 2D noise channels — zero-length placeholder.
            self.noise_2d = np.zeros((_n_limit, 0, 4, 4), dtype=np.float32)
        if "biome_ids" in data:
            self.biome_ids = data["biome_ids"].astype(np.int32)[:_n_limit]  # (N,4,Y,4)
        else:
            # Zero-fill until training data is regenerated with biome IDs.
            self.biome_ids = np.zeros((_n_limit, 4, self.spatial_y, 4), dtype=np.int32)
        if "heightmap5" in data:
            self.heightmap5 = data["heightmap5"].astype(np.float32)[:_n_limit]  # (N,5,16,16)
        elif "heightmap_surface" in data:
            # Backward compat: derive 5-plane from legacy 2-channel heightmaps.
            from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (
                compute_height_planes,
            )

            hm_s = data["heightmap_surface"].astype(np.float32)[:_n_limit]  # (N,16,16)
            hm_o = data["heightmap_ocean_floor"].astype(np.float32)[:_n_limit]  # (N,16,16)
            n = _n_limit
            planes = np.stack(
                [compute_height_planes(hm_s[i], hm_o[i]) for i in range(n)]
            )  # (N,5,16,16)
            self.heightmap5 = planes
        else:
            self.heightmap5 = np.zeros((_n_limit, 5, 16, 16), dtype=np.float32)

        # Phase 5: absolute block-Y of each subchunk's bottom edge.
        if "block_y_min" in data:
            self.block_y_min = data["block_y_min"].astype(np.int32)[:_n_limit]  # (N,)
        else:
            # Legacy NPZ without block_y_min — assume surface-level (Y=64)
            # so heightmap comparisons are roughly centered. Pruning flags
            # won't be accurate but training can still proceed.
            self.block_y_min = np.full(_n_limit, 64, dtype=np.int32)

        self.air_id = air_id
        self._cache_targets = cache_targets
        self._target_cache: Optional[List[Dict[int, Dict[str, torch.Tensor]]]] = None

        n = len(self.subchunks)
        assert len(self.noise_2d) == n and len(self.noise_3d) == n and len(self.biome_ids) == n, (
            f"Length mismatch: subchunks={n}, "
            f"noise_2d={len(self.noise_2d)}, noise_3d={len(self.noise_3d)}, "
            f"biome_ids={len(self.biome_ids)}"
        )
        assert len(self.heightmap5) == n, (
            f"Heightmap5 length mismatch: subchunks={n}, " f"heightmap5={len(self.heightmap5)}"
        )

        if self._cache_targets:
            self._target_cache = [self._build_targets(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.subchunks)

    def _build_targets(self, idx: int) -> Dict[int, Dict[str, torch.Tensor]]:
        raw = build_sparse_octree_targets(self.subchunks[idx], air_id=self.air_id, split_label=-1)

        # Phase 5: compute geometric pruning flags from heightmap + block_y_min.
        prunable_flags = compute_prunable_flags(self.heightmap5[idx], int(self.block_y_min[idx]))

        targets: Dict[int, Dict[str, torch.Tensor]] = {}
        for lvl, lvl_data in raw.items():
            split = (~lvl_data.is_leaf).astype(np.float32).reshape(-1)
            label = lvl_data.labels.astype(np.int64).reshape(-1)
            is_leaf = lvl_data.is_leaf.astype(np.bool_).reshape(-1)
            # Decode child_mask uint8 → [N, 8] float occupancy bits
            # bit0=x, bit1=z, bit2=y matching Java octant convention
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
        return targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        targets = (
            self._target_cache[idx] if self._target_cache is not None else self._build_targets(idx)
        )

        return {
            "noise_2d": torch.from_numpy(self.noise_2d[idx]),  # [C2,4,4]
            "noise_3d": torch.from_numpy(self.noise_3d[idx]),  # [C3,4,Y,4]
            "biome_ids": torch.from_numpy(self.biome_ids[idx]),  # [4,Y,4]
            "heightmap5": torch.from_numpy(self.heightmap5[idx]),  # [5,16,16]
            "targets": targets,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def sparse_octree_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of samples into one batched dict."""
    noise_2d = torch.stack([b["noise_2d"] for b in batch], dim=0)  # [B,C2,4,4]
    noise_3d = torch.stack([b["noise_3d"] for b in batch], dim=0)  # [B,C3,4,Y,4]
    biome_ids = torch.stack([b["biome_ids"] for b in batch], dim=0)  # [B,4,Y,4]
    heightmap5 = torch.stack([b["heightmap5"] for b in batch], dim=0)  # [B,5,16,16]

    levels = sorted(batch[0]["targets"].keys(), reverse=True)
    targets: Dict[int, Dict[str, torch.Tensor]] = {}
    for lvl in levels:
        occ = torch.stack([b["targets"][lvl]["occ"] for b in batch], dim=0)  # [B,N,8]
        split = torch.stack([b["targets"][lvl]["split"] for b in batch], dim=0)  # [B,N]
        label = torch.stack([b["targets"][lvl]["label"] for b in batch], dim=0)  # [B,N]
        is_leaf = torch.stack([b["targets"][lvl]["is_leaf"] for b in batch], dim=0)  # [B,N]
        lvl_dict: Dict[str, torch.Tensor] = {
            "occ": occ,
            "split": split,
            "label": label,
            "is_leaf": is_leaf,
        }
        if "is_prunable" in batch[0]["targets"][lvl]:
            lvl_dict["is_prunable"] = torch.stack(
                [b["targets"][lvl]["is_prunable"] for b in batch], dim=0
            )  # [B,N]
        targets[lvl] = lvl_dict

    return {
        "noise_2d": noise_2d,
        "noise_3d": noise_3d,
        "biome_ids": biome_ids,
        "heightmap5": heightmap5,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _sparse_octree_loss(
    preds: Dict[int, Dict[str, torch.Tensor]],
    targets: Dict[int, Dict[str, torch.Tensor]],
    split_weight: float = 1.0,
    label_weight: float = 0.35,
    level_split_weights: Optional[Dict[int, float]] = None,
    level_label_weights: Optional[Dict[int, float]] = None,
    label_smoothing: float = 0.0,
    dynamic_split_pos_weight: bool = False,
    pruning_boost: float = 4.0,
) -> torch.Tensor:
    """Per-level split (BCE) + leaf-label (CE) loss with geometric pruning.

    Material/label CE is *explicitly* leaf-masked so only nodes with
    ``split_target == 0`` contribute. This keeps optimization focused on
    octree sparsity decisions first, then material labels at true leaves.

    Phase 5 — geometric pruning: when ``is_prunable`` flags are present in
    targets, the occ BCE loss for prunable nodes is amplified by
    ``pruning_boost`` to strongly push occ→0, teaching the model to agree
    with the runtime's heightmap-based geometric pruning.
    """
    ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    device = next(iter(preds.values()))["split"].device
    loss = torch.zeros((), device=device)
    level_split_weights = level_split_weights or {}
    level_label_weights = level_label_weights or {}

    for lvl, out in preds.items():
        label_pred = out["label"]  # [B, N, C]

        tgt = targets[lvl]
        label_tgt = tgt["label"].to(device)  # [B, N] int64

        # Occupancy / split BCE loss
        split_scale = split_weight * level_split_weights.get(lvl, 1.0)
        if split_scale > 0:
            # Use per-child occupancy when available, else legacy scalar split
            if "occ" in out and "occ" in tgt:
                bce_pred = out["occ"]  # [B, N, 8]
                bce_tgt = tgt["occ"].to(device)  # [B, N, 8]
            else:
                bce_pred = out["split"]  # [B, N]
                bce_tgt = tgt["split"].to(device)  # [B, N]

            # Phase 5: per-element weight for geometric pruning.
            # Prunable nodes get an amplified occ loss (push all bits → 0).
            prunable_weight: Optional[torch.Tensor] = None
            if "is_prunable" in tgt and pruning_boost > 0:
                prunable = tgt["is_prunable"].to(device)  # [B, N]
                # Expand to match occ shape ([B, N, 8] or [B, N])
                if bce_pred.ndim == 3:
                    pw = 1.0 + pruning_boost * prunable.unsqueeze(-1)  # [B, N, 1] → broadcast
                else:
                    pw = 1.0 + pruning_boost * prunable
                prunable_weight = pw

            bce: nn.Module
            if dynamic_split_pos_weight:
                pos = float(bce_tgt.sum().item())
                neg = float(bce_tgt.numel() - pos)
                if pos > 0.0 and neg > 0.0:
                    # Clamp to [0.5, 10] — at coarse levels the tiny per-batch
                    # sample count makes the raw ratio extremely noisy.
                    pw_val = max(0.5, min(neg / pos, 10.0))
                    pos_weight = torch.tensor([pw_val], device=device)
                    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
                else:
                    bce = nn.BCEWithLogitsLoss(reduction="none")
            else:
                bce = nn.BCEWithLogitsLoss(reduction="none")

            bce_unreduced = bce(bce_pred, bce_tgt)  # same shape as bce_pred
            if prunable_weight is not None:
                bce_unreduced = bce_unreduced * prunable_weight
            loss = loss + split_scale * bce_unreduced.mean()

        # Explicit leaf-only mask keeps material supervision restricted to
        # split_target == 0. This remains robust even if future target writers
        # stop encoding internal labels as -1.
        B, N, C = label_pred.shape
        if "is_leaf" in tgt:
            leaf_mask = tgt["is_leaf"].to(device=device, dtype=torch.bool)
        elif "occ" in tgt:
            leaf_mask = tgt["occ"].to(device).sum(dim=-1) == 0
        else:
            leaf_mask = tgt["split"].to(device) < 0.5
        if leaf_mask.any():
            label_scale = label_weight * level_label_weights.get(lvl, 1.0)
            loss = loss + label_scale * ce(label_pred[leaf_mask], label_tgt[leaf_mask])

    return loss


def _default_level_weights(max_level: int) -> tuple[Dict[int, float], Dict[int, float]]:
    split_weights: Dict[int, float] = {}
    label_weights: Dict[int, float] = {}
    for lvl in range(max_level, -1, -1):
        depth = max_level - lvl
        split_weights[lvl] = 1.0 + 0.15 * depth
        label_weights[lvl] = 1.0 + 0.4 * depth

    # No children exist below L0, so supervising split there only wastes capacity.
    split_weights[0] = 0.0
    return split_weights, label_weights


def _update_batch_metrics(
    preds: Dict[int, Dict[str, torch.Tensor]],
    targets: Dict[int, Dict[str, torch.Tensor]],
    accum: Dict[str, float],
) -> None:
    """Accumulate split-first metrics from one batch.

    In addition to the aggregate accumulators, per-level split metrics are
    tracked under ``split_tp_L{lvl}`` etc. so callers can diagnose per-level
    calibration without a separate evaluation pass.
    """
    for lvl, out in preds.items():
        # Derive split from occ: any child logit > 0 ↔ sigmoid > 0.5
        if "occ" in out:
            split_pred = out["occ"].max(dim=-1).values > 0
        else:
            split_pred = out["split"] > 0
        split_tgt = targets[lvl]["split"].to(split_pred.device) > 0.5

        tp = (split_pred & split_tgt).sum().item()
        tn = ((~split_pred) & (~split_tgt)).sum().item()
        fp = (split_pred & (~split_tgt)).sum().item()
        fn = ((~split_pred) & split_tgt).sum().item()
        accum["split_tp"] += float(tp)
        accum["split_tn"] += float(tn)
        accum["split_fp"] += float(fp)
        accum["split_fn"] += float(fn)

        # Per-level split accumulators (keys created lazily)
        for key, val in (("split_tp", tp), ("split_tn", tn), ("split_fp", fp), ("split_fn", fn)):
            lk = f"{key}_L{lvl}"
            accum[lk] = accum.get(lk, 0.0) + float(val)

        # Predicted/ground-truth node counts proxy complexity. A node is active
        # when it is internal (split=1) or a leaf with material supervision.
        pred_active = split_pred.numel() - split_pred.sum().item()
        gt_active = split_tgt.numel() - split_tgt.sum().item()
        accum["pred_leaf_nodes"] += float(pred_active)
        accum["gt_leaf_nodes"] += float(gt_active)

        label_pred = out["label"].argmax(dim=-1)
        if "is_leaf" in targets[lvl]:
            leaf_mask = targets[lvl]["is_leaf"].to(label_pred.device)
        else:
            leaf_mask = targets[lvl]["label"].to(label_pred.device) != -1
        if leaf_mask.any():
            label_tgt = targets[lvl]["label"].to(label_pred.device)
            accum["leaf_total"] += float(leaf_mask.sum().item())
            accum["leaf_correct"] += float(
                (label_pred[leaf_mask] == label_tgt[leaf_mask]).sum().item()
            )

        # Phase 5: pruning agreement — does the model predict occ=0 for
        # geometrically prunable nodes?
        if "is_prunable" in targets[lvl]:
            prunable = targets[lvl]["is_prunable"].to(split_pred.device).bool()
            if prunable.any():
                # For prunable nodes, the model should NOT split (occ=0).
                prune_agree = (~split_pred & prunable).sum().item()
                prune_total = prunable.sum().item()
                accum["prune_agree"] = accum.get("prune_agree", 0.0) + float(prune_agree)
                accum["prune_total"] = accum.get("prune_total", 0.0) + float(prune_total)


def _finalize_metrics(accum: Dict[str, float]) -> Dict[str, float]:
    tp = accum["split_tp"]
    tn = accum["split_tn"]
    fp = accum["split_fp"]
    fn = accum["split_fn"]
    split_total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    result: Dict[str, float] = {
        "split_acc": (tp + tn) / max(split_total, 1.0),
        "split_precision": precision,
        "split_recall": recall,
        "split_f1": f1,
        "split_over_rate": fp / max(fp + tn, 1.0),
        "split_under_rate": fn / max(fn + tp, 1.0),
        "leaf_acc": accum["leaf_correct"] / max(accum["leaf_total"], 1.0),
        "leaf_node_ratio": accum["pred_leaf_nodes"] / max(accum["gt_leaf_nodes"], 1.0),
    }

    # Per-level split F1 — only emitted for levels that have data.
    seen_levels: set[int] = set()
    for k in accum:
        if k.startswith("split_tp_L"):
            seen_levels.add(int(k.rsplit("L", 1)[1]))
    for lvl in sorted(seen_levels, reverse=True):
        ltp = accum.get(f"split_tp_L{lvl}", 0.0)
        lfp = accum.get(f"split_fp_L{lvl}", 0.0)
        lfn = accum.get(f"split_fn_L{lvl}", 0.0)
        lprec = ltp / max(ltp + lfp, 1.0)
        lrec = ltp / max(ltp + lfn, 1.0)
        result[f"split_f1_L{lvl}"] = 2.0 * lprec * lrec / max(lprec + lrec, 1e-12)

    # Phase 5: pruning agreement rate — fraction of geometrically-prunable
    # nodes where the model correctly predicts occ=0 (no split).
    prune_total = accum.get("prune_total", 0.0)
    if prune_total > 0:
        result["prune_agree_rate"] = accum.get("prune_agree", 0.0) / prune_total

    return result


def _build_model(
    model_variant: str,
    *,
    n2d: int,
    n3d: int,
    hidden: int,
    num_classes: int,
    spatial_y: int = 4,
) -> nn.Module:
    variant = model_variant.lower()
    if variant == "baseline":
        return SparseOctreeModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
            spatial_y=spatial_y,
        )
    if variant == "fast":
        return SparseOctreeFastModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
            spatial_y=spatial_y,
        )
    raise ValueError(f"Unknown sparse-root model_variant={model_variant!r}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_sparse_octree(
    data_path: Path,
    out_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden: int = 72,
    num_classes: int = -1,
    device: str = "cpu",
    model_variant: str = "fast",
    cache_targets: bool = True,
    max_samples: Optional[int] = None,
    split_weight: float = 1.0,
    label_weight: float = 0.35,
    label_smoothing: float = 0.03,
    pruning_boost: float = 4.0,
    resume_from: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train the SparseOctreeModel on noise-conditioned sparse-root pairs.

    Parameters
    ----------
    resume_from : Path, optional
        Path to a checkpoint saved by a previous training run.  When provided
        the model weights, optimizer state, and starting epoch are restored so
        training continues where it left off.  The ``epochs`` parameter is
        interpreted as the *total* target epoch count (not additional epochs).
    """

    data_path = Path(data_path)
    out_path = Path(out_path)
    _device = torch.device(device)
    ds = SparseOctreeDataset(data_path, cache_targets=cache_targets, max_samples=max_samples)
    print(f"  Dataset: {len(ds)} samples" + (" (limited from full set)" if max_samples else ""))

    # Auto-detect num_classes from the actual max block ID in the dataset.
    # Validate against the canonical Voxy vocab to prevent silent under-sizing
    # (blocks beyond num_classes become permanently unreachable).
    _VOCAB_PATH = (
        Path(__file__).resolve().parents[2] / "config" / "voxy_vocab.json"
    )  # voxel_tree/config/
    _vocab_size: Optional[int] = None
    if _VOCAB_PATH.exists():
        try:
            import json as _json

            _vmap = _json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
            _vocab_size = max(_vmap.values()) + 1 if _vmap else None
        except Exception:  # noqa: BLE001
            pass

    if num_classes <= 0:
        raw = np.load(data_path)
        num_classes = int(raw["subchunk16"].max()) + 1
        raw.close()
        if _vocab_size is not None and num_classes < _vocab_size:
            print(
                f"  WARNING: auto-detected num_classes={num_classes} from data, "
                f"but canonical vocab has {_vocab_size} entries. "
                f"{_vocab_size - num_classes} block(s) will be unreachable. "
                f"Consider passing --num-classes {_vocab_size}."
            )
            # Upgrade to vocab size so all blocks are representable.
            num_classes = _vocab_size
            print(f"  Upgraded num_classes to {num_classes} (canonical vocab size)")
        else:
            print(f"  auto-detected num_classes={num_classes}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=sparse_octree_collate)

    # Infer noise channel counts and spatial_y from the first sample
    sample = ds[0]
    n2d = sample["noise_2d"].shape[0]
    n3d = sample["noise_3d"].shape[0]
    spatial_y = ds.spatial_y  # detected from NPZ shape

    model = _build_model(
        model_variant,
        n2d=n2d,
        n3d=n3d,
        hidden=hidden,
        num_classes=num_classes,
        spatial_y=spatial_y,
    ).to(_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    max_level_attr = getattr(model, "max_level", 4)
    max_level = max_level_attr if isinstance(max_level_attr, int) else 4
    level_split_weights, level_label_weights = _default_level_weights(max_level)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    start_epoch = 1

    # ── Resume from checkpoint ─────────────────────────────────────────
    if resume_from is not None:
        resume_from = Path(resume_from)
        if resume_from.exists():
            ckpt = torch.load(resume_from, map_location=_device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_loss = ckpt.get("best_loss", float("inf"))
                if "best_state" in ckpt:
                    best_state = ckpt["best_state"]
                print(
                    f"[resume] Loaded checkpoint from {resume_from} "
                    f"(epoch {start_epoch - 1}, best_loss={best_loss:.4f})"
                )
            else:
                # Legacy checkpoint: bare state_dict
                model.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt)
                print(
                    f"[resume] Loaded legacy state_dict from {resume_from} "
                    f"(starting from epoch 1, no optimizer state)"
                )
        else:
            print(f"[resume] Checkpoint not found at {resume_from} — training from scratch")

    if start_epoch > epochs:
        print(
            f"[resume] Already trained to epoch {start_epoch - 1} "
            f"(requested {epochs}) — nothing to do"
        )
        return {
            "checkpoint": str(out_path),
            "best_loss": best_loss,
            "history": history,
            "model_variant": model_variant,
            "hidden": hidden,
        }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        metric_accum = {
            "split_tp": 0.0,
            "split_tn": 0.0,
            "split_fp": 0.0,
            "split_fn": 0.0,
            "leaf_correct": 0.0,
            "leaf_total": 0.0,
            "pred_leaf_nodes": 0.0,
            "gt_leaf_nodes": 0.0,
        }

        for batch in loader:
            noise_2d = batch["noise_2d"].to(_device)
            noise_3d = batch["noise_3d"].to(_device)
            biome_ids = batch["biome_ids"].to(_device)
            heightmap5 = batch["heightmap5"].to(_device)
            optimizer.zero_grad()
            preds = model(noise_2d, noise_3d, biome_ids, heightmap5)
            loss = _sparse_octree_loss(
                preds,
                batch["targets"],
                split_weight=split_weight,
                label_weight=label_weight,
                level_split_weights=level_split_weights,
                level_label_weights=level_label_weights,
                label_smoothing=label_smoothing,
                dynamic_split_pos_weight=True,
                pruning_boost=pruning_boost,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_batches += 1
            _update_batch_metrics(preds, batch["targets"], metric_accum)

        avg_loss = total_loss / max(total_batches, 1)
        row = {"epoch": float(epoch), "loss": avg_loss}
        row.update(_finalize_metrics(metric_accum))
        history.append(row)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_callback is not None:
            progress_callback(epoch, epochs, row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save full checkpoint with optimizer state for resume support.
    # Also save backward-compatible bare state_dict for export.
    full_ckpt = {
        "model_state_dict": best_state
        or {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "best_loss": best_loss,
        "model_variant": model_variant,
        "hidden": hidden,
        "num_classes": num_classes,
    }
    torch.save(full_ckpt, out_path)
    print(f"[train] Saved checkpoint to {out_path} (epoch {epochs}, best_loss={best_loss:.4f})")

    return {
        "checkpoint": str(out_path),
        "best_loss": best_loss,
        "history": history,
        "model_variant": model_variant,
        "hidden": hidden,
    }
