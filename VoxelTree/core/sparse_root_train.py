"""Training utilities for the noise-conditioned sparse-root octree model.

Dataset layout expected in the .npz file
-----------------------------------------
subchunk16  : int32   [N, 16, 16, 16]   dense voxel labels
noise_2d    : float32 [N, C2, 4, 4]     vanilla 2-D climate fields per subchunk
noise_3d    : float32 [N, C3, 4, 2, 4]  vanilla 3-D volumetric fields per subchunk
biome_ids   : int32   [N, 4, 2, 4]      discrete biome IDs at spatial resolution

Targets are built on-the-fly by ``build_sparse_octree_targets`` and stored as
flat 1-D tensors per level so they can be directly compared against the model's
flat [B, N] output tensors.

  level 4 (side 1): N=1    split [B,1],     label [B,1]
  level 3 (side 2): N=8    split [B,8],     label [B,8]
  level 2 (side 4): N=64   split [B,64],    label [B,64]
  level 1 (side 8): N=512  split [B,512],   label [B,512]
  level 0 (side16): N=4096 split [B,4096],  label [B,4096]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from LODiffusion.models.sparse_root import SparseRootFastModel, SparseRootModel
from .sparse_octree_targets import build_sparse_octree_targets

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SparseRootDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for noise-conditioned sparse-root supervision."""

    def __init__(self, npz_path: Path, air_id: int = 0, cache_targets: bool = True) -> None:
        data = np.load(npz_path)
        self.subchunks = data["subchunk16"].astype(np.int32)  # (N,16,16,16)
        self.noise_2d = data["noise_2d"].astype(np.float32)  # (N,C2,4,4)
        self.noise_3d = data["noise_3d"].astype(np.float32)  # (N,C3,4,2,4)
        if "biome_ids" in data:
            self.biome_ids = data["biome_ids"].astype(np.int32)  # (N,4,2,4)
        else:
            # Zero-fill until training data is regenerated with biome IDs.
            self.biome_ids = np.zeros((len(self.subchunks), 4, 2, 4), dtype=np.int32)
        self.air_id = air_id
        self._cache_targets = cache_targets
        self._target_cache: Optional[List[Dict[int, Dict[str, torch.Tensor]]]] = None

        n = len(self.subchunks)
        assert len(self.noise_2d) == n and len(self.noise_3d) == n and len(self.biome_ids) == n, (
            f"Length mismatch: subchunks={n}, "
            f"noise_2d={len(self.noise_2d)}, noise_3d={len(self.noise_3d)}, "
            f"biome_ids={len(self.biome_ids)}"
        )

        if self._cache_targets:
            self._target_cache = [self._build_targets(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.subchunks)

    def _build_targets(self, idx: int) -> Dict[int, Dict[str, torch.Tensor]]:
        raw = build_sparse_octree_targets(self.subchunks[idx], air_id=self.air_id, split_label=-1)

        targets: Dict[int, Dict[str, torch.Tensor]] = {}
        for lvl, lvl_data in raw.items():
            split = (~lvl_data.is_leaf).astype(np.float32).reshape(-1)
            label = lvl_data.labels.astype(np.int64).reshape(-1)
            is_leaf = lvl_data.is_leaf.astype(np.bool_).reshape(-1)
            targets[lvl] = {
                "split": torch.from_numpy(split),
                "label": torch.from_numpy(label),
                "is_leaf": torch.from_numpy(is_leaf),
            }
        return targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        targets = (
            self._target_cache[idx] if self._target_cache is not None else self._build_targets(idx)
        )

        return {
            "noise_2d": torch.from_numpy(self.noise_2d[idx]),  # [C2,4,4]
            "noise_3d": torch.from_numpy(self.noise_3d[idx]),  # [C3,4,2,4]
            "biome_ids": torch.from_numpy(self.biome_ids[idx]),  # [4,2,4]
            "targets": targets,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def sparse_root_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of samples into one batched dict."""
    noise_2d = torch.stack([b["noise_2d"] for b in batch], dim=0)  # [B,C2,4,4]
    noise_3d = torch.stack([b["noise_3d"] for b in batch], dim=0)  # [B,C3,4,2,4]
    biome_ids = torch.stack([b["biome_ids"] for b in batch], dim=0)  # [B,4,2,4]

    levels = sorted(batch[0]["targets"].keys(), reverse=True)
    targets: Dict[int, Dict[str, torch.Tensor]] = {}
    for lvl in levels:
        split = torch.stack([b["targets"][lvl]["split"] for b in batch], dim=0)  # [B,N]
        label = torch.stack([b["targets"][lvl]["label"] for b in batch], dim=0)  # [B,N]
        is_leaf = torch.stack([b["targets"][lvl]["is_leaf"] for b in batch], dim=0)  # [B,N]
        targets[lvl] = {"split": split, "label": label, "is_leaf": is_leaf}

    return {
        "noise_2d": noise_2d,
        "noise_3d": noise_3d,
        "biome_ids": biome_ids,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _sparse_root_loss(
    preds: Dict[int, Dict[str, torch.Tensor]],
    targets: Dict[int, Dict[str, torch.Tensor]],
    split_weight: float = 1.0,
    label_weight: float = 0.35,
    level_split_weights: Optional[Dict[int, float]] = None,
    level_label_weights: Optional[Dict[int, float]] = None,
    label_smoothing: float = 0.0,
    dynamic_split_pos_weight: bool = False,
) -> torch.Tensor:
    """Per-level split (BCE) + leaf-label (CE) loss.

    Material/label CE is *explicitly* leaf-masked so only nodes with
    ``split_target == 0`` contribute. This keeps optimization focused on
    octree sparsity decisions first, then material labels at true leaves.
    """
    ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    device = next(iter(preds.values()))["split"].device
    loss = torch.zeros((), device=device)
    level_split_weights = level_split_weights or {}
    level_label_weights = level_label_weights or {}

    for lvl, out in preds.items():
        split_pred = out["split"]  # [B, N]
        label_pred = out["label"]  # [B, N, C]

        tgt = targets[lvl]
        split_tgt = tgt["split"].to(device)  # [B, N] float
        label_tgt = tgt["label"].to(device)  # [B, N] int64

        split_scale = split_weight * level_split_weights.get(lvl, 1.0)
        if split_scale > 0:
            bce: nn.Module
            if dynamic_split_pos_weight:
                pos = float(split_tgt.sum().item())
                neg = float(split_tgt.numel() - pos)
                if pos > 0.0 and neg > 0.0:
                    pos_weight = torch.tensor([neg / pos], device=device)
                    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    bce = nn.BCEWithLogitsLoss()
            else:
                bce = nn.BCEWithLogitsLoss()
            loss = loss + split_scale * bce(split_pred, split_tgt)

        # Explicit leaf-only mask keeps material supervision restricted to
        # split_target == 0. This remains robust even if future target writers
        # stop encoding internal labels as -1.
        B, N, C = label_pred.shape
        if "is_leaf" in tgt:
            leaf_mask = tgt["is_leaf"].to(device=device, dtype=torch.bool)
        else:
            leaf_mask = split_tgt < 0.5
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
    """Accumulate split-first metrics from one batch."""
    for lvl, out in preds.items():
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


def _finalize_metrics(accum: Dict[str, float]) -> Dict[str, float]:
    tp = accum["split_tp"]
    tn = accum["split_tn"]
    fp = accum["split_fp"]
    fn = accum["split_fn"]
    split_total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "split_acc": (tp + tn) / max(split_total, 1.0),
        "split_precision": precision,
        "split_recall": recall,
        "split_f1": f1,
        "split_over_rate": fp / max(fp + tn, 1.0),
        "split_under_rate": fn / max(fn + tp, 1.0),
        "leaf_acc": accum["leaf_correct"] / max(accum["leaf_total"], 1.0),
        "leaf_node_ratio": accum["pred_leaf_nodes"] / max(accum["gt_leaf_nodes"], 1.0),
    }


def _build_model(
    model_variant: str,
    *,
    n2d: int,
    n3d: int,
    hidden: int,
    num_classes: int,
) -> nn.Module:
    variant = model_variant.lower()
    if variant == "baseline":
        return SparseRootModel(n2d=n2d, n3d=n3d, hidden=hidden, num_classes=num_classes)
    if variant == "fast":
        return SparseRootFastModel(n2d=n2d, n3d=n3d, hidden=hidden, num_classes=num_classes)
    raise ValueError(f"Unknown sparse-root model_variant={model_variant!r}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_sparse_root(
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
    split_weight: float = 1.0,
    label_weight: float = 0.35,
    label_smoothing: float = 0.03,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train the SparseRootModel on noise-conditioned sparse-root pairs."""

    data_path = Path(data_path)
    out_path = Path(out_path)
    _device = torch.device(device)
    ds = SparseRootDataset(data_path, cache_targets=cache_targets)

    # Auto-detect num_classes from the actual max block ID in the dataset
    if num_classes <= 0:
        raw = np.load(data_path)
        num_classes = int(raw["subchunk16"].max()) + 1
        raw.close()
        print(f"  auto-detected num_classes={num_classes}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=sparse_root_collate)

    # Infer noise channel counts from the first sample
    sample = ds[0]
    n2d = sample["noise_2d"].shape[0]
    n3d = sample["noise_3d"].shape[0]

    model = _build_model(
        model_variant,
        n2d=n2d,
        n3d=n3d,
        hidden=hidden,
        num_classes=num_classes,
    ).to(_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    max_level_attr = getattr(model, "max_level", 4)
    max_level = max_level_attr if isinstance(max_level_attr, int) else 4
    level_split_weights, level_label_weights = _default_level_weights(max_level)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
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
            optimizer.zero_grad()
            preds = model(noise_2d, noise_3d, biome_ids)
            loss = _sparse_root_loss(
                preds,
                batch["targets"],
                split_weight=split_weight,
                label_weight=label_weight,
                level_split_weights=level_split_weights,
                level_label_weights=level_label_weights,
                label_smoothing=label_smoothing,
                dynamic_split_pos_weight=True,
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
    torch.save(best_state or model.state_dict(), out_path)

    return {
        "checkpoint": str(out_path),
        "best_loss": best_loss,
        "history": history,
        "model_variant": model_variant,
        "hidden": hidden,
    }
