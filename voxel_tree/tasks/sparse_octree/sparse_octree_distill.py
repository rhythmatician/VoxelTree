"""Distillation utilities for sparse-root octree models.

This trains a `fast` sparse-root student from a stronger teacher checkpoint
while still keeping direct supervision from the real sparse-root dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sparse_octree import SparseOctreeFastModel, SparseOctreeModel
from .sparse_octree_train import (
    SparseOctreeDataset,
    _default_level_weights,
    _sparse_octree_loss,
    sparse_octree_collate,
)


def _unwrap_state(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state)!r}")
    return state


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    state = _unwrap_state(torch.load(path, map_location="cpu", weights_only=True))

    variant = "fast" if any(k.startswith("level_mod.") for k in state) else "baseline"
    if "label_head.out_proj.bias" in state:
        num_classes = int(state["label_head.out_proj.bias"].shape[0])
    elif "label_head.bias" in state:
        num_classes = int(state["label_head.bias"].shape[0])
    elif "label_head.out_proj.weight" in state:
        num_classes = int(state["label_head.out_proj.weight"].shape[0])
    elif "label_head.weight" in state:
        num_classes = int(state["label_head.weight"].shape[0])
    else:
        raise KeyError(f"Could not infer num_classes from checkpoint: {path}")

    if variant == "fast":
        hidden = int(state["root_proj.0.weight"].shape[0])
    else:
        hidden = int(state["root_proj.weight"].shape[0])

    return {
        "path": str(path),
        "state": state,
        "variant": variant,
        "hidden": hidden,
        "num_classes": num_classes,
    }


def _build_model(
    *,
    variant: str,
    n2d: int,
    n3d: int,
    hidden: int,
    num_classes: int,
) -> torch.nn.Module:
    if variant == "fast":
        return SparseOctreeFastModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
        )
    if variant == "baseline":
        return SparseOctreeModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown model variant: {variant!r}")


def _soft_label_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        if not mask.any():
            return student_logits.new_zeros(())
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def _distill_loss(
    student_preds: Dict[int, Dict[str, torch.Tensor]],
    teacher_preds: Dict[int, Dict[str, torch.Tensor]],
    targets: Dict[int, Dict[str, torch.Tensor]],
    *,
    hard_weight: float,
    split_distill_weight: float,
    label_distill_weight: float,
    temperature: float,
    label_smoothing: float,
    level_split_weights: Dict[int, float],
    level_label_weights: Dict[int, float],
) -> Dict[str, torch.Tensor]:
    device = next(iter(student_preds.values()))["split"].device

    hard = _sparse_octree_loss(
        student_preds,
        targets,
        level_split_weights=level_split_weights,
        level_label_weights=level_label_weights,
        label_smoothing=label_smoothing,
        dynamic_split_pos_weight=True,
    )

    split_distill = torch.zeros((), device=device)
    label_distill = torch.zeros((), device=device)
    for lvl in student_preds:
        student_split = student_preds[lvl]["split"]
        teacher_split = teacher_preds[lvl]["split"].detach()
        split_scale = level_split_weights.get(lvl, 1.0)
        if split_scale > 0:
            teacher_split_prob = torch.sigmoid(teacher_split / max(temperature, 1e-6))
            split_distill = split_distill + split_scale * F.binary_cross_entropy_with_logits(
                student_split,
                teacher_split_prob,
            )

        student_label = student_preds[lvl]["label"]
        teacher_label = teacher_preds[lvl]["label"].detach()
        label_mask = targets[lvl]["label"].to(device) != -1
        label_scale = level_label_weights.get(lvl, 1.0)
        if label_scale > 0:
            label_distill = label_distill + label_scale * _soft_label_kl(
                student_label,
                teacher_label,
                temperature=temperature,
                mask=label_mask,
            )

    total = (
        hard_weight * hard
        + split_distill_weight * split_distill
        + label_distill_weight * label_distill
    )
    return {
        "loss": total,
        "hard_loss": hard.detach(),
        "split_distill_loss": split_distill.detach(),
        "label_distill_loss": label_distill.detach(),
    }


@torch.no_grad()
def _evaluate_student(
    model: torch.nn.Module,
    dataset: SparseOctreeDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_octree_collate,
    )
    split_correct = 0.0
    split_total = 0.0
    leaf_correct = 0.0
    leaf_total = 0.0

    model.eval()
    for batch in loader:
        noise_2d = batch["noise_2d"].to(device)
        noise_3d = batch["noise_3d"].to(device)
        biome_ids = batch["biome_ids"].to(device)
        heightmap_surface = batch["heightmap_surface"].to(device)
        heightmap_ocean_floor = batch["heightmap_ocean_floor"].to(device)
        pos_bits = batch["position_bits"].to(device)
        preds = model(
            noise_2d,
            noise_3d,
            biome_ids,
            heightmap_surface,
            heightmap_ocean_floor,
            position_bits=pos_bits,
        )
        for lvl, out in preds.items():
            split_pred = (out["split"] > 0).to(torch.int64)
            split_tgt = batch["targets"][lvl]["split"].to(device).to(torch.int64)
            split_correct += float((split_pred == split_tgt).sum().item())
            split_total += float(split_tgt.numel())

            label_pred = out["label"].argmax(dim=-1)
            label_tgt = batch["targets"][lvl]["label"].to(device)
            leaf_mask = label_tgt != -1
            leaf_total += float(leaf_mask.sum().item())
            if leaf_mask.any():
                leaf_correct += float((label_pred[leaf_mask] == label_tgt[leaf_mask]).sum().item())

    return {
        "split_acc": split_correct / max(split_total, 1.0),
        "leaf_acc": leaf_correct / max(leaf_total, 1.0),
    }


def distill_sparse_octree(
    *,
    data_path: Path,
    teacher_checkpoint: Path,
    out_path: Path,
    summary_path: Optional[Path] = None,
    student_variant: str = "fast",
    student_hidden: int = 80,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    hard_weight: float = 0.5,
    split_distill_weight: float = 0.15,
    label_distill_weight: float = 0.35,
    temperature: float = 1.5,
    label_smoothing: float = 0.02,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Distill a sparse-root student from a teacher checkpoint."""

    data_path = Path(data_path)
    teacher_checkpoint = Path(teacher_checkpoint)
    out_path = Path(out_path)
    summary_path = Path(summary_path) if summary_path is not None else out_path.with_suffix(".json")
    _device = torch.device(device)

    dataset = SparseOctreeDataset(data_path, cache_targets=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_octree_collate,
    )
    sample = dataset[0]
    n2d = int(sample["noise_2d"].shape[0])
    n3d = int(sample["noise_3d"].shape[0])

    teacher_meta = checkpoint_metadata(teacher_checkpoint)
    teacher = _build_model(
        variant=teacher_meta["variant"],
        n2d=n2d,
        n3d=n3d,
        hidden=teacher_meta["hidden"],
        num_classes=teacher_meta["num_classes"],
    ).to(_device)
    teacher.load_state_dict(teacher_meta["state"], strict=True)
    teacher.eval()

    student = _build_model(
        variant=student_variant,
        n2d=n2d,
        n3d=n3d,
        hidden=student_hidden,
        num_classes=teacher_meta["num_classes"],
    ).to(_device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    max_level = int(getattr(student, "max_level", 4))
    level_split_weights, level_label_weights = _default_level_weights(max_level)

    history: list[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_split_kd = 0.0
        total_label_kd = 0.0
        batches = 0

        for batch in loader:
            noise_2d = batch["noise_2d"].to(_device)
            noise_3d = batch["noise_3d"].to(_device)
            biome_ids = batch["biome_ids"].to(_device)
            heightmap_surface = batch["heightmap_surface"].to(_device)
            heightmap_ocean_floor = batch["heightmap_ocean_floor"].to(_device)
            pos_bits = batch["position_bits"].to(_device)

            with torch.no_grad():
                teacher_preds = teacher(
                    noise_2d,
                    noise_3d,
                    biome_ids,
                    heightmap_surface,
                    heightmap_ocean_floor,
                    position_bits=pos_bits,
                )
            student_preds = student(
                noise_2d,
                noise_3d,
                biome_ids,
                heightmap_surface,
                heightmap_ocean_floor,
                position_bits=pos_bits,
            )

            losses = _distill_loss(
                student_preds,
                teacher_preds,
                batch["targets"],
                hard_weight=hard_weight,
                split_distill_weight=split_distill_weight,
                label_distill_weight=label_distill_weight,
                temperature=temperature,
                label_smoothing=label_smoothing,
                level_split_weights=level_split_weights,
                level_label_weights=level_label_weights,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()

            total_loss += float(losses["loss"].detach())
            total_hard += float(losses["hard_loss"])
            total_split_kd += float(losses["split_distill_loss"])
            total_label_kd += float(losses["label_distill_loss"])
            batches += 1

        scheduler.step()
        eval_metrics = _evaluate_student(student, dataset, batch_size=batch_size, device=_device)
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(batches, 1),
            "hard_loss": total_hard / max(batches, 1),
            "split_distill_loss": total_split_kd / max(batches, 1),
            "label_distill_loss": total_label_kd / max(batches, 1),
            "split_acc": eval_metrics["split_acc"],
            "leaf_acc": eval_metrics["leaf_acc"],
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        if progress_callback is not None:
            progress_callback(epoch, epochs, row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or student.state_dict(), out_path)

    result = {
        "checkpoint": str(out_path),
        "summary": str(summary_path),
        "best_loss": best_loss,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_variant": teacher_meta["variant"],
        "teacher_hidden": teacher_meta["hidden"],
        "student_variant": student_variant,
        "student_hidden": student_hidden,
        "num_classes": teacher_meta["num_classes"],
        "history": history,
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result


__all__ = ["checkpoint_metadata", "distill_sparse_octree"]
