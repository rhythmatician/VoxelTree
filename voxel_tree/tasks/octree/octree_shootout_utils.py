from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss


def resolve_root() -> Path:
    """Resolve the VoxelTree project root from a notebook or script cwd."""
    root = Path.cwd()
    if root.name == "experimental":
        return root.parents[1]
    if root.name == "notebooks":
        return root.parent
    return root


ROOT = resolve_root()
DATA_DIR = ROOT / "data" / "voxy_octree"
ARTIFACT_DIR = ROOT / "notebooks" / "experimental" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_BLOCK_CLASSES = 1104
NUM_BIOMES = 256
NUM_Y_POSITIONS = 24
NUM_LEVELS = 5
SPATIAL = 32


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_block_counts(
    labels32: np.ndarray | torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    arr = (
        labels32.detach().cpu().numpy()
        if isinstance(labels32, torch.Tensor)
        else np.asarray(labels32)
    )
    flat = arr.astype(np.int64, copy=False).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot compute block priors from an empty labels32 array")
    flat = np.clip(flat, 0, num_classes - 1)
    return np.bincount(flat, minlength=num_classes).astype(np.int64, copy=False)


def compute_block_log_priors(
    labels32: np.ndarray | torch.Tensor,
    *,
    num_classes: int = NUM_BLOCK_CLASSES,
    verbose: bool = True,
) -> np.ndarray:
    """Compute log-frequency block priors from notebook labels."""
    counts = _compute_block_counts(labels32, num_classes)
    total = int(counts.sum())
    log_priors = np.log(counts.astype(np.float64) / max(total, 1) + 1e-8).astype(np.float32)

    if verbose:
        air_pct = 100.0 * counts[0] / max(total, 1)
        seen = int(np.count_nonzero(counts))
        print(
            f"Computed block priors: total_voxels={total:,} "
            f"air={air_pct:.2f}% seen={seen}/{num_classes}"
        )

    return log_priors


def init_block_head_bias(
    model: torch.nn.Module,
    block_log_priors: np.ndarray | torch.Tensor,
    *,
    verbose: bool = True,
    model_name: str | None = None,
) -> None:
    """Initialize a candidate model's ``block_head.bias`` from log priors."""
    head = getattr(model, "block_head", None)
    if head is None or getattr(head, "bias", None) is None:
        raise ValueError(f"{type(model).__name__} is missing a usable block_head.bias")

    bias = head.bias
    prior_tensor = torch.as_tensor(block_log_priors, dtype=bias.dtype, device=bias.device)
    if prior_tensor.ndim != 1 or prior_tensor.shape[0] != bias.shape[0]:
        raise ValueError(
            f"Prior shape {tuple(prior_tensor.shape)} does not match bias shape {tuple(bias.shape)}"
        )

    with torch.no_grad():
        bias.copy_(prior_tensor)

    if verbose:
        label = model_name or type(model).__name__
        print(f"  ✓ Initialized {label}.block_head bias from log-frequency priors")


def bitmask_to_binary(bitmask: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(8, device=bitmask.device).unsqueeze(0)
    return ((bitmask.unsqueeze(1).long() >> bits) & 1).float()


def _required_keys(include_parent: bool, include_level: bool, include_occ: bool) -> list[str]:
    keys = ["labels32", "heightmap32", "biome32", "y_position"]
    if include_parent:
        keys.append("parent_labels32")
    if include_level:
        keys.append("level")
    if include_occ:
        keys.append("non_empty_children")
    return keys


def load_pair_cache_subset(
    split: str,
    *,
    levels: Sequence[int],
    data_dir: Path = DATA_DIR,
    include_parent: bool = False,
    include_level: bool = False,
    include_occ: bool = False,
) -> dict[str, np.ndarray] | None:
    """Load and filter a production pair cache for notebook shootouts."""
    cache_path = data_dir / f"{split}_octree_pairs.npz"
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        missing = [
            key
            for key in _required_keys(include_parent, include_level, include_occ)
            if key not in data
        ]
        if missing:
            raise KeyError(f"{cache_path.name} missing required keys: {missing}")

        level_arr = data["level"].astype(np.int64)
        level_mask = np.isin(level_arr, np.array(list(levels), dtype=np.int64))
        if not np.any(level_mask):
            raise ValueError(f"No samples for levels {list(levels)} in {cache_path}")

        result: dict[str, np.ndarray] = {
            "labels32": data["labels32"][level_mask].astype(np.int32),
            "heightmap32": data["heightmap32"][level_mask].astype(np.float32),
            "biome32": data["biome32"][level_mask].astype(np.int32),
            "y_position": data["y_position"][level_mask].astype(np.int64),
        }
        if include_parent:
            result["parent_labels32"] = data["parent_labels32"][level_mask].astype(np.int32)
        if include_level:
            result["level"] = level_arr[level_mask]
        if include_occ:
            result["non_empty_children"] = data["non_empty_children"][level_mask].astype(np.uint8)
        return result


def maybe_load_class_weights(
    data_dir: Path = DATA_DIR,
    *,
    clamp_max: float = 20.0,
) -> torch.Tensor | None:
    """Load production class weights if present."""
    weights_path = data_dir / "class_weights.npz"
    if not weights_path.exists():
        return None

    with np.load(weights_path, allow_pickle=False) as data:
        if "class_weights" not in data:
            return None
        weights = data["class_weights"].astype(np.float32)

    if clamp_max > 0:
        weights = np.clip(weights, 0.0, clamp_max)
    return torch.from_numpy(weights)


def describe_split(name: str, data: dict[str, np.ndarray], *, show_levels: bool = False) -> None:
    print(f"{name}:")
    for key, value in data.items():
        print(f"  {key}: shape={value.shape} dtype={value.dtype}")
    if show_levels and "level" in data:
        unique, counts = np.unique(data["level"], return_counts=True)
        print("  levels:", {int(k): int(v) for k, v in zip(unique, counts)})


class ArrayDictDataset(Dataset[dict[str, torch.Tensor]]):
    """Minimal dataset wrapper over filtered numpy arrays."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        *,
        include_parent: bool = False,
        include_level: bool = False,
        include_occ: bool = False,
    ) -> None:
        self.labels = torch.from_numpy(data["labels32"]).long()
        self.heightmap = torch.from_numpy(data["heightmap32"]).float()
        self.biome = torch.from_numpy(data["biome32"]).long()
        self.y_pos = torch.from_numpy(data["y_position"]).long()
        self.parent = torch.from_numpy(data["parent_labels32"]).long() if include_parent else None
        self.level = torch.from_numpy(data["level"]).long() if include_level else None
        self.occ = (
            torch.from_numpy(data["non_empty_children"].astype(np.int64)).long()
            if include_occ
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "labels32": self.labels[idx],
            "heightmap32": self.heightmap[idx],
            "biome32": self.biome[idx],
            "y_position": self.y_pos[idx],
        }
        if self.parent is not None:
            sample["parent_labels32"] = self.parent[idx]
        if self.level is not None:
            sample["level"] = self.level[idx]
        if self.occ is not None:
            sample["non_empty_children"] = self.occ[idx]
        return sample


def make_loader(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    include_parent: bool = False,
    include_level: bool = False,
    include_occ: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = ArrayDictDataset(
        data,
        include_parent=include_parent,
        include_level=include_level,
        include_occ=include_occ,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def cross_entropy_loss(
    block_logits: torch.Tensor,
    target_blocks: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    vocab = block_logits.shape[1]
    flat_logits = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab)
    flat_target = target_blocks.reshape(-1)
    return F.cross_entropy(flat_logits, flat_target, weight=class_weights)


def occupancy_loss(
    occ_logits: torch.Tensor,
    occ_targets: torch.Tensor,
    *,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    if focal_gamma > 0:
        return sigmoid_focal_loss(
            occ_logits,
            occ_targets,
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
        )

    weight = torch.tensor(pos_weight, device=occ_logits.device) if pos_weight != 1.0 else None
    return F.binary_cross_entropy_with_logits(occ_logits, occ_targets, pos_weight=weight)


def compute_block_metrics(
    block_logits: torch.Tensor,
    target_blocks: torch.Tensor,
    *,
    air_id: int = 0,
    topk: int = 5,
) -> dict[str, float]:
    vocab = block_logits.shape[1]
    flat_logits = block_logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab)
    flat_target = target_blocks.reshape(-1)
    pred = flat_logits.argmax(dim=1)
    correct = pred == flat_target
    air_mask = flat_target == air_id
    nonair_mask = ~air_mask
    topk_idx = flat_logits.topk(min(topk, vocab), dim=1).indices
    topk_correct = (topk_idx == flat_target.unsqueeze(1)).any(dim=1)

    return {
        "block_acc": correct.float().mean().item(),
        "air_acc": correct[air_mask].float().mean().item() if air_mask.any() else 1.0,
        "nonair_acc": correct[nonair_mask].float().mean().item() if nonair_mask.any() else 1.0,
        f"top{topk}_acc": topk_correct.float().mean().item(),
    }


def compute_occ_metrics(occ_logits: torch.Tensor, occ_targets: torch.Tensor) -> dict[str, float]:
    occ_pred = (occ_logits > 0).float()
    tp = (occ_pred * occ_targets).sum().item()
    fp = (occ_pred * (1 - occ_targets)).sum().item()
    fn = ((1 - occ_pred) * occ_targets).sum().item()
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fnr = fn / max(tp + fn, 1.0)
    runtime_pred = (occ_logits > -0.8473).float()
    tp_rt = (runtime_pred * occ_targets).sum().item()
    fn_rt = ((1 - runtime_pred) * occ_targets).sum().item()
    recall_rt = tp_rt / max(tp_rt + fn_rt, 1.0)
    return {
        "occ_precision": precision,
        "occ_recall": recall,
        "occ_f1": f1,
        "occ_fnr": fnr,
        "occ_recall_rt": recall_rt,
    }


def maybe_corrupt_parent(
    parent: torch.Tensor,
    *,
    probability: float = 0.15,
    vocab_size: int = NUM_BLOCK_CLASSES,
    replace_fraction: float = 0.30,
) -> torch.Tensor:
    if probability <= 0 or random.random() >= probability:
        return parent
    if random.random() < 0.5:
        corrupted = parent.clone()
        non_air = corrupted > 0
        if int(non_air.sum().item()) == 0:
            return corrupted
        noise_mask = torch.rand_like(corrupted.float()) < replace_fraction
        replace_mask = non_air & noise_mask
        num_replace = int(replace_mask.sum().item())
        if num_replace > 0:
            random_ids = torch.randint(
                1,
                vocab_size,
                (num_replace,),
                device=corrupted.device,
                dtype=corrupted.dtype,
            )
            corrupted[replace_mask] = random_ids
        return corrupted
    return torch.zeros_like(parent)


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: str = "cpu",
    warmup: int = 5,
    steps: int = 20,
) -> dict[str, float]:
    dev = torch.device(device)
    model = model.to(dev).eval()
    moved_batch = {key: value.to(dev) for key, value in batch.items()}
    for _ in range(warmup):
        model(moved_batch)
    t0 = time.perf_counter()
    for _ in range(steps):
        model(moved_batch)
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / steps
    return {
        "ms_per_batch": ms_per_batch,
        "ms_per_sample": ms_per_batch / max(int(moved_batch["labels32"].shape[0]), 1),
    }
