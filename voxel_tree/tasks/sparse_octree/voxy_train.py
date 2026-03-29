"""Per-level training for Voxy-native models.

Trains one model at a time (L0, L1, …, L4).  Levels 0–3 include parent
conditioning; level 4 is the root and has no parent.

Usage
-----
::

    # Train L0
    python -m voxel_tree.tasks.sparse_octree.voxy_train \\
        --db data/v7_dumps.db --level 0 --epochs 40 --batch-size 16

    # Resume from checkpoint
    python -m voxel_tree.tasks.sparse_octree.voxy_train \\
        --db data/v7_dumps.db --level 0 --epochs 40 --resume checkpoints/voxy_L0.pt

The training data is assembled on the fly from the noise-dumps DB +
``voxy_sections`` table via :class:`VoxyLevelWithParentDataset`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .voxy_dataset import (
    VoxyLevelDataset,
    VoxyLevelWithParentDataset,
    voxy_level_collate,
)
from .voxy_models import VoxyModelConfig, create_model


# ══════════════════════════════════════════════════════════════════════
#  Loss
# ══════════════════════════════════════════════════════════════════════


def _compute_surface_weights(
    labels: torch.Tensor,
    interior_weight: float = 0.1,
    air_id: int = 0,
) -> torch.Tensor:
    """Compute per-voxel weights: 1.0 for visible (air-adjacent) blocks,
    *interior_weight* for fully buried blocks.

    A block is "visible" if any of its 6 face-neighbours is air.  Air blocks
    themselves are always weighted 1.0 (predicting air correctly matters for
    the silhouette).

    Boundary voxels (at the edge of the 32³ grid) are treated as visible
    because we cannot see their neighbours.

    Args:
        labels: ``[B, Y, Z, X]`` int block IDs.
        interior_weight: Weight for non-visible solid blocks (0–1).
        air_id: Block ID representing air.

    Returns:
        ``[B, Y, Z, X]`` float weight tensor.
    """
    is_air = labels == air_id  # [B, Y, Z, X]

    # Pad with True (air) on all 6 faces so boundary blocks count as visible
    padded = F.pad(is_air.float(), (1, 1, 1, 1, 1, 1), value=1.0)
    # Check 6-connected neighbours for air
    has_air_neighbour = (
        padded[:, :-2, 1:-1, 1:-1]  # Y-1
        + padded[:, 2:, 1:-1, 1:-1]  # Y+1
        + padded[:, 1:-1, :-2, 1:-1]  # Z-1
        + padded[:, 1:-1, 2:, 1:-1]  # Z+1
        + padded[:, 1:-1, 1:-1, :-2]  # X-1
        + padded[:, 1:-1, 1:-1, 2:]  # X+1
    ) > 0  # [B, Y, Z, X]

    # Weight: 1.0 for air blocks and visible solid blocks, interior_weight otherwise
    weights = torch.where(is_air | has_air_neighbour, 1.0, interior_weight)
    return weights


def _compute_semantic_weights(
    labels: torch.Tensor,
    *,
    air_id: int = 0,
    stone_id: int = 1,
    common_weight: float = 0.35,
    non_common_weight: float = 2.0,
) -> torch.Tensor:
    """Compute per-voxel semantic weights to de-emphasize air/stone.

    Air and stone dominate Minecraft volume statistics and can overwhelm
    optimization. This weighting promotes learning visually salient classes
    (grass, leaves, logs, etc.) by upweighting non-air/non-stone voxels.

    Args:
        labels: ``[B, Y, Z, X]`` int block IDs.
        air_id: Block ID representing air.
        stone_id: Block ID representing stone.
        common_weight: Weight for air/stone voxels.
        non_common_weight: Weight for all other block IDs.

    Returns:
        ``[B, Y, Z, X]`` float weight tensor.
    """
    is_common = (labels == air_id) | (labels == stone_id)
    common = torch.full_like(labels, common_weight, dtype=torch.float32)
    non_common = torch.full_like(labels, non_common_weight, dtype=torch.float32)
    return torch.where(is_common, common, non_common)


def _voxy_level_loss(
    block_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    occ_logits: Optional[torch.Tensor] = None,
    occ_target: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
    label_smoothing: float = 0.02,
    occ_weight: float = 1.0,
    interior_weight: float = 1.0,
    air_id: int = 0,
    stone_id: int = 1,
    common_weight: float = 0.35,
    non_common_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Compute per-level training loss.

    Args:
        block_logits: ``[B, V, 32, 32, 32]`` — block-type predictions.
        labels: ``[B, 32, 32, 32]`` — ground-truth block IDs (Y, Z, X order).
        occ_logits: ``[B, 8]`` — occupancy predictions (L1–L4 only).
        occ_target: ``[B, 8]`` — occupancy targets (float, 0/1).
        ignore_index: Label value to ignore (e.g. -1 for missing data).
        label_smoothing: Cross-entropy label smoothing.
        occ_weight: Weight for occupancy loss.
        interior_weight: Weight for non-visible (buried) blocks.  1.0 = equal
            weighting (default), 0.1 = 10× priority on visible blocks.
        air_id: Block ID representing air.
        stone_id: Block ID representing stone.
        common_weight: Weight for air/stone voxels.
        non_common_weight: Weight for non-air/non-stone voxels.

    Returns:
        Dict with 'loss', 'block_loss', 'occ_loss' (if applicable),
        and 'surface_frac' (fraction of voxels weighted at 1.0).
    """
    # ── Block classification loss ─────────────────────────────────
    # L4 outputs [B, V, 32, 32, 24] (Y trimmed to MC world height).
    # Trim labels to match if needed.
    y_out = block_logits.shape[2]  # block_logits: [B, V, Y, Z, X]
    if labels.shape[1] != y_out:
        labels = labels[:, :y_out, :, :]  # trim Y from 32 to 24

    # Reshape to [B*Y*Z*X, V] for cross_entropy
    logits_flat = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, block_logits.shape[1])
    labels_flat = labels.reshape(-1)

    use_surface = interior_weight < 1.0
    use_semantic = common_weight != 1.0 or non_common_weight != 1.0

    if use_surface or use_semantic:
        weights = torch.ones_like(labels, dtype=torch.float32)
        if use_surface:
            # Visibility-weighted loss: prioritise air-adjacent blocks
            weights = weights * _compute_surface_weights(
                labels, interior_weight, air_id=air_id
            )
        if use_semantic:
            # Semantic weighting: de-emphasize air/stone, emphasize others
            weights = weights * _compute_semantic_weights(
                labels,
                air_id=air_id,
                stone_id=stone_id,
                common_weight=common_weight,
                non_common_weight=non_common_weight,
            )
        weights_flat = weights.reshape(-1)

        per_voxel = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        # Zero weight for ignored voxels
        valid_mask = labels_flat != ignore_index
        valid_weights = weights_flat[valid_mask]
        block_loss = ((per_voxel * weights_flat)[valid_mask]).sum() / valid_weights.sum().clamp_min(
            1e-8
        )
        if use_surface:
            surface_flat = _compute_surface_weights(labels, interior_weight, air_id=air_id).reshape(-1)
            surface_frac = (surface_flat[valid_mask] == 1.0).float().mean().item()
        else:
            surface_frac = 1.0
        non_common_flat = ((labels != air_id) & (labels != stone_id)).reshape(-1)
        non_common_frac = non_common_flat[valid_mask].float().mean().item()
    else:
        block_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        surface_frac = 1.0
        valid_mask = labels_flat != ignore_index
        non_common_flat = ((labels != air_id) & (labels != stone_id)).reshape(-1)
        non_common_frac = non_common_flat[valid_mask].float().mean().item()

    result: Dict[str, torch.Tensor] = {
        "block_loss": block_loss,
        "surface_frac": torch.tensor(surface_frac),
        "non_common_frac": torch.tensor(non_common_frac),
    }

    # ── Occupancy loss ────────────────────────────────────────────
    if occ_logits is not None and occ_target is not None:
        occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occ_target)
        result["occ_loss"] = occ_loss
        result["loss"] = block_loss + occ_weight * occ_loss
    else:
        result["occ_loss"] = torch.tensor(0.0, device=block_logits.device)
        result["loss"] = block_loss

    return result


# ══════════════════════════════════════════════════════════════════════
#  Occupancy target computation
# ══════════════════════════════════════════════════════════════════════


def compute_occ_target(labels32: torch.Tensor, air_id: int = 0) -> torch.Tensor:
    """Compute 8-bit child-octant occupancy from a 32³ label grid.

    Splits the 32³ grid into 2×2×2 octants of 16³ each.  An octant is
    "occupied" if it contains any non-air block.

    Bit convention: bit0=X, bit1=Z, bit2=Y (Voxy standard).

    Args:
        labels32: ``[B, 32, 32, 32]`` int label grid.
        air_id: Block ID for air (default 0).

    Returns:
        ``[B, 8]`` float tensor with 0.0/1.0 per octant.
    """
    B = labels32.shape[0]
    occ = torch.zeros(B, 8, device=labels32.device, dtype=torch.float32)

    for y_half in range(2):
        for z_half in range(2):
            for x_half in range(2):
                octant_idx = x_half | (z_half << 1) | (y_half << 2)
                octant = labels32[
                    :,
                    y_half * 16 : (y_half + 1) * 16,
                    z_half * 16 : (z_half + 1) * 16,
                    x_half * 16 : (x_half + 1) * 16,
                ]
                has_nonair = (octant != air_id).any(dim=-1).any(dim=-1).any(dim=-1)
                occ[:, octant_idx] = has_nonair.float()

    return occ


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════


def _compute_block_accuracy(
    block_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
    air_id: int = 0,
    stone_id: int = 1,
) -> Dict[str, float]:
    """Compute block-type prediction accuracy."""
    preds = block_logits.argmax(dim=1)  # [B, Y, Z, X]
    # Trim labels Y-dim to match model output (L4 outputs Y=24, labels are Y=32)
    y_out = preds.shape[1]
    if labels.shape[1] != y_out:
        labels = labels[:, :y_out, :, :]
    valid = labels != ignore_index
    n_valid = valid.sum().item()
    if n_valid == 0:
        return {"block_acc": 0.0, "n_valid": 0, "non_common_acc": 0.0, "n_non_common": 0}
    correct = ((preds == labels) & valid).sum().item()
    non_common = valid & (labels != air_id) & (labels != stone_id)
    n_non_common = non_common.sum().item()
    if n_non_common > 0:
        non_common_correct = ((preds == labels) & non_common).sum().item()
        non_common_acc = non_common_correct / n_non_common
    else:
        non_common_acc = 0.0
    return {
        "block_acc": correct / n_valid,
        "n_valid": n_valid,
        "non_common_acc": non_common_acc,
        "n_non_common": n_non_common,
    }


# ══════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════


def train_voxy_level(
    db_path: Path,
    out_path: Path,
    level: int,
    *,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    label_smoothing: float = 0.02,
    occ_weight: float = 1.0,
    min_coverage: float = 1.0,
    resume_from: Optional[Path] = None,
    max_samples: Optional[int] = None,
    num_workers: Optional[int] = None,
    channels_last: bool = True,
    cache_dataset: bool = True,
    interior_weight: float = 0.1,
    air_id: int = 0,
    stone_id: int = 1,
    common_weight: float = 0.35,
    non_common_weight: float = 2.0,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train a single Voxy-level model.

    Args:
        db_path: Path to v7 noise-dumps DB (with ``voxy_sections`` table).
        out_path: Where to save the checkpoint.
        level: Voxy LOD level (0–4).
        epochs: Total epoch count.
        batch_size: Training batch size.
        lr: Learning rate (AdamW).
        device: PyTorch device string.
        label_smoothing: CE label smoothing.
        occ_weight: Occupancy loss multiplier.
        min_coverage: Fraction of constituent sections needed (1.0=all).
        resume_from: Path to a checkpoint to resume from.
        max_samples: Cap dataset size for debugging.
        num_workers: DataLoader worker count.  Default: 0 on Windows
            (SQLite + spawn multiprocessing deadlocks), 4 elsewhere.
        channels_last: Use channels-last-3d memory format for ~1.6x CPU
            speedup on 3D convolutions.  Default True.
        cache_dataset: Pre-load entire dataset into RAM to eliminate
            per-batch SQLite I/O overhead (~40% faster).  Default True.
        interior_weight: Weight for buried (non-visible) blocks.  0.1 means
            visible blocks get 10× priority.  Default 0.1.
        air_id: Block ID representing air.
        stone_id: Block ID representing stone.
        common_weight: Weight for air/stone voxels.
        non_common_weight: Weight for non-air/non-stone voxels.
        progress_callback: Called each epoch with (epoch, total, metrics).

    Returns:
        Dict with 'checkpoint', 'best_loss', 'history'.
    """
    db_path = Path(db_path)
    out_path = Path(out_path)
    _device = torch.device(device)

    # ── Auto-detect vocab size & load remap ───────────────────────
    _cfg_dir = Path(__file__).resolve().parents[2] / "config"
    _vocab_remap_path = _cfg_dir / "vocab_remap.json"
    num_classes = 513  # default: 512 block types + air=0
    _vocab_remap: Optional[Dict[int, int]] = None
    if _vocab_remap_path.exists():
        try:
            remap = json.loads(_vocab_remap_path.read_text(encoding="utf-8"))
            _vocab_remap = {int(k): int(v) for k, v in remap.items()}
            num_classes = max(_vocab_remap.values()) + 1 if _vocab_remap else num_classes
        except Exception:
            pass
    print(f"[L{level}] Vocab size: {num_classes}")

    # ── Dataset ───────────────────────────────────────────────────
    if level == 4:
        ds: VoxyLevelDataset | VoxyLevelWithParentDataset = VoxyLevelDataset(
            db_path, level, min_coverage, vocab_remap=_vocab_remap, cache=cache_dataset
        )
    else:
        ds = VoxyLevelWithParentDataset(
            db_path, level, min_coverage, vocab_remap=_vocab_remap, cache=cache_dataset
        )

    if max_samples is not None and len(ds) > max_samples:
        # Simple truncation for debugging
        if isinstance(ds, VoxyLevelWithParentDataset):
            ds.base.samples = ds.base.samples[:max_samples]
        else:
            ds.samples = ds.samples[:max_samples]

    print(f"[L{level}] Dataset: {len(ds)} samples")
    if len(ds) == 0:
        print(f"[L{level}] No training data — aborting.")
        return {"checkpoint": "", "best_loss": float("inf"), "history": []}

    # ── Model ─────────────────────────────────────────────────────
    cfg = VoxyModelConfig(block_vocab_size=num_classes)
    model = create_model(level, cfg).to(_device)

    # ── Channels-last 3D memory format (CPU speedup) ─────────
    if channels_last and hasattr(model, "unet"):
        model.unet.channels_last_3d = True
        print(f"[L{level}] Using channels-last-3d memory format")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[L{level}] Model params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ── DataLoader ────────────────────────────────────────────────
    if num_workers is None:
        # Default: 0 on Windows (SQLite + spawn causes deadlocks), 4 elsewhere
        _nw = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    else:
        _nw = num_workers
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=voxy_level_collate,
        num_workers=_nw,
        persistent_workers=_nw > 0,
        prefetch_factor=2 if _nw > 0 else None,
    )
    print(f"[L{level}] DataLoader: num_workers={_nw}")

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=_device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", float("inf"))
            print(f"[L{level}] Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.4f}")

    if start_epoch > epochs:
        print(f"[L{level}] Already trained to epoch {start_epoch - 1} — nothing to do")
        return {"checkpoint": str(out_path), "best_loss": best_loss, "history": []}

    # ── Training ──────────────────────────────────────────────────
    history: List[Dict[str, float]] = []
    n_batches = (len(ds) + batch_size - 1) // batch_size
    _log_interval = max(1, min(n_batches // 10, 50))
    has_parent = level < 4
    has_occ = level > 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_block_loss = 0.0
        total_occ_loss = 0.0
        total_surface_frac = 0.0
        total_non_common_frac = 0.0
        total_acc = 0.0
        total_non_common_acc = 0.0
        total_valid = 0
        total_non_common = 0
        total_batches = 0
        epoch_t0 = time.monotonic()

        for batch in loader:
            y_position = batch["y_position"].to(_device)
            labels32 = batch["labels32"].to(_device)

            # Forward pass — L0/L1 use 3D noise, L2-L4 use 2D climate
            if level >= 2:
                climate_2d = batch["climate_2d"].to(_device)
                biome_2d = batch["biome_2d"].to(_device)
                if has_parent:
                    parent_blocks = batch["parent_blocks"].to(_device)
                    preds = model(climate_2d, biome_2d, y_position, parent_blocks)
                else:
                    preds = model(climate_2d, biome_2d, y_position)
            else:
                noise_3d = batch["noise_3d"].to(_device)
                biome_3d = batch["biome_3d"].to(_device)
                if has_parent:
                    parent_blocks = batch["parent_blocks"].to(_device)
                    preds = model(noise_3d, biome_3d, y_position, parent_blocks)
                else:
                    preds = model(noise_3d, biome_3d, y_position)

            # Occupancy targets
            occ_target = None
            if has_occ:
                occ_target = compute_occ_target(labels32)

            losses = _voxy_level_loss(
                preds["block_logits"],
                labels32,
                occ_logits=preds.get("occ_logits"),
                occ_target=occ_target,
                label_smoothing=label_smoothing,
                occ_weight=occ_weight,
                interior_weight=interior_weight,
                air_id=air_id,
                stone_id=stone_id,
                common_weight=common_weight,
                non_common_weight=non_common_weight,
            )

            loss = losses["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_block_loss += losses["block_loss"].item()
            total_occ_loss += losses["occ_loss"].item()
            total_surface_frac += losses["surface_frac"].item()
            total_non_common_frac += losses["non_common_frac"].item()
            total_batches += 1

            # Accuracy (no grad needed)
            with torch.no_grad():
                acc = _compute_block_accuracy(
                    preds["block_logits"], labels32, air_id=air_id, stone_id=stone_id
                )
                total_acc += acc["block_acc"] * acc["n_valid"]
                total_valid += acc["n_valid"]
                total_non_common_acc += acc["non_common_acc"] * acc["n_non_common"]
                total_non_common += acc["n_non_common"]

            # Progress logging
            if total_batches % _log_interval == 0 or total_batches == n_batches:
                elapsed = time.monotonic() - epoch_t0
                pct = 100.0 * total_batches / n_batches
                avg = total_loss / total_batches
                spb = elapsed / total_batches
                remaining = (n_batches - total_batches + (epochs - epoch) * n_batches) * spb
                if remaining >= 3600:
                    eta = f"{remaining / 3600:.1f}h"
                elif remaining >= 60:
                    eta = f"{remaining / 60:.1f}m"
                else:
                    eta = f"{remaining:.0f}s"
                print(
                    f"  E{epoch}/{epochs} [{total_batches:>{len(str(n_batches))}}/{n_batches}]"
                    f" {pct:5.1f}%  loss={avg:.4f}  ETA {eta}",
                    flush=True,
                )

        avg_loss = total_loss / max(total_batches, 1)
        avg_block = total_block_loss / max(total_batches, 1)
        avg_occ = total_occ_loss / max(total_batches, 1)
        avg_surface = total_surface_frac / max(total_batches, 1)
        avg_acc = total_acc / max(total_valid, 1)
        avg_non_common_acc = total_non_common_acc / max(total_non_common, 1)
        avg_non_common_frac = total_non_common_frac / max(total_batches, 1)

        row = {
            "epoch": float(epoch),
            "loss": avg_loss,
            "block_loss": avg_block,
            "occ_loss": avg_occ,
            "block_acc": avg_acc,
            "non_common_acc": avg_non_common_acc,
            "non_common_frac": avg_non_common_frac,
            "surface_frac": avg_surface,
        }
        history.append(row)
        sfrac_str = f" surface={avg_surface:.1%}" if interior_weight < 1.0 else ""
        nc_str = f" non_common_acc={avg_non_common_acc:.3f} non_common_frac={avg_non_common_frac:.1%}"
        print(
            f"  [L{level}] E{epoch}: loss={avg_loss:.4f} block={avg_block:.4f} "
            f"occ={avg_occ:.4f} acc={avg_acc:.3f}{nc_str}{sfrac_str}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_callback is not None:
            progress_callback(epoch, epochs, row)

    # ── Save checkpoint ───────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": best_state
        or {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "best_loss": best_loss,
        "level": level,
        "config": cfg,
        "history": history,
    }
    torch.save(ckpt, out_path)
    print(f"[L{level}] Saved to {out_path} (best_loss={best_loss:.4f})")

    return {
        "checkpoint": str(out_path),
        "best_loss": best_loss,
        "history": history,
    }


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a single Voxy-level model")
    parser.add_argument("--db", type=Path, required=True, help="Path to v7 noise-dumps DB")
    parser.add_argument("--level", type=int, required=True, choices=range(5), help="Voxy level 0-4")
    parser.add_argument("--out", type=Path, default=None, help="Checkpoint path (default: auto)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--occ-weight", type=float, default=1.0)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: 0 on Windows, 4 on Linux)",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--no-channels-last", action="store_true", help="Disable channels-last-3d memory format"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable pre-loading dataset into RAM"
    )
    parser.add_argument(
        "--interior-weight",
        type=float,
        default=0.1,
        help="Loss weight for buried (non-visible) blocks. 0.1 = 10x priority on "
        "air-adjacent blocks. 1.0 = equal weighting (default: 0.1)",
    )
    parser.add_argument("--air-id", type=int, default=0, help="Block ID used for air")
    parser.add_argument("--stone-id", type=int, default=1, help="Block ID used for stone")
    parser.add_argument(
        "--common-weight",
        type=float,
        default=0.35,
        help="Loss weight for common classes (air/stone)",
    )
    parser.add_argument(
        "--non-common-weight",
        type=float,
        default=2.0,
        help="Loss weight for all non-air/non-stone classes",
    )
    args = parser.parse_args()

    out = args.out or Path(f"checkpoints/voxy_L{args.level}.pt")

    train_voxy_level(
        db_path=args.db,
        out_path=out,
        level=args.level,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        label_smoothing=args.label_smoothing,
        occ_weight=args.occ_weight,
        min_coverage=args.min_coverage,
        resume_from=args.resume,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        channels_last=not args.no_channels_last,
        cache_dataset=not args.no_cache,
        interior_weight=args.interior_weight,
        air_id=args.air_id,
        stone_id=args.stone_id,
        common_weight=args.common_weight,
        non_common_weight=args.non_common_weight,
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
