"""
Multi-LOD Training Data Pipeline

This module creates training data for the flexible multi-LOD model by:
1. Loading full-resolution 16³ chunks
2. Downsampling to create parent voxels at different LOD levels
3. Creating training pairs for all LOD transitions

The downsampling process simulates what would happen during progressive
refinement, allowing the model to learn to reverse the process.
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.mipper import build_opacity_table, mip_volume_numpy

# Module-level opacity table (lazy)
_OPACITY_TABLE: np.ndarray | None = None


def _get_opacity_table(max_id: int = 4096) -> np.ndarray:
    global _OPACITY_TABLE
    if _OPACITY_TABLE is None or len(_OPACITY_TABLE) < max_id + 1:
        _OPACITY_TABLE = build_opacity_table(max(max_id + 1, 4096))
    return _OPACITY_TABLE


def create_occupancy_from_blocks(block_data: np.ndarray, air_id: int = 0) -> np.ndarray:
    """
    Create binary occupancy data from block IDs.

    Args:
        block_data: Array of block IDs
        air_id: ID of air blocks (default 0)

    Returns:
        Binary occupancy array (1 = solid, 0 = air)
    """
    return (block_data != air_id).astype(np.uint8)


def create_lod_training_pairs(
    labels16: np.ndarray,
    biome_patch: np.ndarray,
    heightmap_patch: np.ndarray,
    y_index: int,
    heightmap_surface: np.ndarray,
    heightmap_ocean_floor: np.ndarray,
    air_id: int = 0,
) -> List[Dict]:
    """
    Create training pairs for incremental LOD refinement from a single 16³ chunk.

    Each transition refines by exactly one LOD level (2× per axis).  The parent
    is the Mipper output at LOD N, upsampled to the canonical 8³ input.  The
    target is the Mipper output at LOD N-1, upsampled to 16³.

    Transitions produced (4 total, no LOD0 — vanilla handles that):
        init_to_lod4  → no parent,      target = mip(·,16)→16³ (LOD4)  — Init
        f=16 (lod4to3) → 1³→8³ parent (LOD4), target = mip(·,8)→16³ (LOD3)
        f=8  (lod3to2) → 2³→8³ parent (LOD3), target = mip(·,4)→16³ (LOD2)
        f=4  (lod2to1) → 4³→8³ parent (LOD2), target = mip(·,2)→16³ (LOD1)

    Conditioning inputs: biome + height_planes + y_index.

    Args:
        labels16:            (16, 16, 16) int array of block IDs at LOD0
        biome_patch:         (16, 16) int array of biome IDs
        heightmap_patch:     (16, 16) float array — per-slab normalised heights
        y_index:             Y-level slab index for this chunk
        heightmap_surface:   (16, 16) float32 — column-level surface height (world-Y)
        heightmap_ocean_floor:(16, 16) float32 — column-level ocean floor (world-Y)
        air_id:              block ID that represents air (default 0)

    Returns:
        List of training-sample dicts, one per coarsening factor.
    """
    tbl = _get_opacity_table(int(labels16.max()) + 1)

    # ------------------------------------------------------------------
    # Build conditioning tensors
    # ------------------------------------------------------------------
    # Biome: keep only compact index form; one-hot computed lazily in __getitem__
    biome_idx = biome_patch.astype(np.int64)  # (16,16)

    # Heightmap: normalise patch to [0,1]
    heightmap_norm = (heightmap_patch.astype(np.float32) - heightmap_patch.min()) / (
        max(heightmap_patch.max() - heightmap_patch.min(), 1e-6)
    )
    heightmap_tensor = heightmap_norm[None, ...]  # (1,16,16)

    # ------------------------------------------------------------------
    # Compute height_planes tensor [5, 16, 16]
    # ------------------------------------------------------------------
    HEIGHT_RANGE = 320.0
    surf = heightmap_surface.astype(np.float32) / HEIGHT_RANGE
    ofloor = heightmap_ocean_floor.astype(np.float32) / HEIGHT_RANGE

    # Terrain derivatives via central differences
    slope_x = np.gradient(surf, axis=1).astype(np.float32)
    slope_z = np.gradient(surf, axis=0).astype(np.float32)
    curvature = (np.gradient(slope_x, axis=1) + np.gradient(slope_z, axis=0)).astype(np.float32)

    height_planes = np.stack(
        [
            surf.astype(np.float32),
            ofloor.astype(np.float32),
            slope_x.astype(np.float32),
            slope_z.astype(np.float32),
            curvature.astype(np.float32),
        ],
        axis=0,
    )  # (5, 16, 16)

    # ------------------------------------------------------------------
    # Router6 removed — biome + heightmap is sufficient for LOD4→LOD1.
    # See docs/ChatGPTconversation.md for feature-selection rationale.
    # ------------------------------------------------------------------

    training_pairs: List[Dict] = []

    # ------------------------------------------------------------------
    # Init → LOD4 pair (no parent, target is the single-voxel mip)
    # ------------------------------------------------------------------
    init_labels, init_occ = mip_volume_numpy(labels16, 16, tbl)
    # init_labels / init_occ: shape (1, 1, 1)
    # Upsample to 16³ for uniform cache shape
    init_target_labels = np.repeat(
        np.repeat(np.repeat(init_labels, 16, axis=0), 16, axis=1), 16, axis=2
    ).astype(np.int64)
    init_target_occ = np.repeat(
        np.repeat(np.repeat(init_occ.astype(np.float32), 16, axis=0), 16, axis=1), 16, axis=2
    )

    training_pairs.append(
        {
            # --- model inputs (per-sample shapes, no batch dim) ---
            "parent_voxel": np.zeros((1, 8, 8, 8), dtype=np.float32),  # init model ignores parent
            "biome_patch": biome_idx,  # (16,16) int64 indices
            "biome_idx": biome_idx,  # (16,16) integer indices
            "heightmap_patch": heightmap_tensor,  # (1,16,16)  C=1
            "height_planes": height_planes,  # (5,16,16)
            "y_index": np.int64(y_index),  # scalar
            "lod": np.int64(4),  # LOD4 (coarsest)
            # --- targets (upsampled to 16³ for uniform storage) ---
            "target_mask": init_target_occ,  # (16,16,16) float32
            "target_types": init_target_labels,  # (16,16,16) int64
            # --- metadata ---
            "lod_transition": "init_to_lod4",
            "parent_size": 0,  # no parent for init
            "target_size": 16,
        }
    )

    # ------------------------------------------------------------------
    # Refinement pairs: LOD4→LOD3, LOD3→LOD2, LOD2→LOD1
    # (f=2 / LOD1→LOD0 deliberately dropped — vanilla handles LOD0)
    # ------------------------------------------------------------------
    for f in (4, 8, 16):
        # ------------------------------------------------------------------
        # Parent: coarsen labels16 by factor f using the Voxy Mipper,
        # then nearest-upsample to canonical 8³.
        # ------------------------------------------------------------------
        coarse_labels, coarse_occ = mip_volume_numpy(labels16, f, tbl)
        # shape: (16//f, 16//f, 16//f)

        coarse_size = coarse_labels.shape[0]  # 8, 4, 2, or 1
        if coarse_size != 8:
            scale = 8 // coarse_size
            coarse_occ_8 = np.repeat(
                np.repeat(np.repeat(coarse_occ.astype(np.float32), scale, axis=0), scale, axis=1),
                scale,
                axis=2,
            )
        else:
            coarse_occ_8 = coarse_occ.astype(np.float32)

        # ------------------------------------------------------------------
        # Target: one level finer than the parent (incremental refinement).
        #   f=4  → target is LOD1 = mip(labels16, 2)   → upsample to 16³
        #   f=8  → target is LOD2 = mip(labels16, 4)   → upsample to 16³
        #   f=16 → target is LOD3 = mip(labels16, 8)   → upsample to 16³
        # ------------------------------------------------------------------
        f_target = f // 2  # coarsening factor for the target LOD level

        # Target is the next-finer LOD level, upsampled to 16³
        tgt_labels, tgt_occ = mip_volume_numpy(labels16, f_target, tbl)
        # shape: (16//f_target, 16//f_target, 16//f_target)
        tgt_size = tgt_labels.shape[0]  # 8, 4, or 2
        up = 16 // tgt_size
        target_labels = np.repeat(
            np.repeat(np.repeat(tgt_labels, up, axis=0), up, axis=1),
            up,
            axis=2,
        ).astype(
            np.int64
        )  # (16,16,16)
        target_occ = np.repeat(
            np.repeat(np.repeat(tgt_occ.astype(np.float32), up, axis=0), up, axis=1),
            up,
            axis=2,
        )  # (16,16,16)

        lod_token = int(math.log2(f))  # 1, 2, 3, 4
        target_lod = lod_token - 1  # 0, 1, 2, 3

        training_pairs.append(
            {
                # --- model inputs (per-sample shapes, no batch dim) ---
                "parent_voxel": coarse_occ_8[None, ...],  # (1,8,8,8)  C=1
                "biome_patch": biome_idx,  # (16,16) int64 indices
                "biome_idx": biome_idx,  # (16,16) integer indices
                "heightmap_patch": heightmap_tensor,  # (1,16,16)  C=1
                "height_planes": height_planes,  # (5,16,16)
                "y_index": np.int64(y_index),  # scalar
                "lod": np.int64(lod_token),  # scalar
                # --- targets (16³, one LOD level finer than parent) ---
                "target_mask": target_occ,  # (16,16,16) float32
                "target_types": target_labels,  # (16,16,16) int64
                # --- metadata ---
                "lod_transition": f"lod{lod_token}to{target_lod}",
                "parent_size": 8,
                "target_size": 16,
            }
        )

    return training_pairs


# ------------------------------------------------------------------
# Standalone pair generation (used by scripts/build_pairs.py)
# ------------------------------------------------------------------


def generate_pairs_from_npz_files(
    npz_files: List[Path],
    *,
    min_solid_fraction: float = 0.02,
) -> List[Dict]:
    """Process raw NPZ chunk files into LOD training pairs.

    This is the **prep phase** — meant to run once, offline, *before* training.
    Required NPZ keys: ``labels16`` (or ``target_types``), ``biome_patch``
    (or ``biome16``), ``heightmap_patch`` (or ``height16``), ``y_index``,
    ``heightmap_surface``, ``heightmap_ocean_floor``.

    Run ``scripts/add_column_heights.py`` after extraction to add the
    column-level height fields to the NPZ files.

    Args:
        npz_files: Paths to raw chunk NPZ files.
        min_solid_fraction: Skip chunks with fewer solid voxels than this.

    Returns:
        List of training-pair dicts ready to be saved as a pair cache.
    """
    from tqdm import tqdm

    training_pairs: List[Dict] = []
    skipped_air = 0
    skipped_missing = 0
    total_files = len(npz_files)

    for npz_file in tqdm(npz_files, total=total_files, desc="Processing NPZ files", unit="file"):
        try:
            data = np.load(npz_file)

            # Extract required fields — support both canonical and legacy key names
            if "labels16" in data:
                labels16 = data["labels16"]
            elif "target_types" in data:
                labels16 = data["target_types"].astype(np.int32)
            else:
                print(f"Skipping {npz_file}: no labels16 or target_types key")
                skipped_missing += 1
                continue

            # Filter out nearly-all-air chunks (nothing to learn from)
            solid_frac = (labels16 != 0).mean()
            if solid_frac < min_solid_fraction:
                skipped_air += 1
                continue

            # ---- Biome: require real data ----
            if "biome16" in data:
                biome16 = data["biome16"]
            elif "biome_patch" in data:
                biome16 = data["biome_patch"]
            else:
                print(f"Skipping {npz_file}: no biome data (biome16 or biome_patch)")
                skipped_missing += 1
                continue

            # ---- Heightmap: require real data ----
            if "height16" in data:
                height16 = data["height16"]
            elif "heightmap_patch" in data:
                height16 = data["heightmap_patch"]
            else:
                print(f"Skipping {npz_file}: no heightmap data")
                skipped_missing += 1
                continue

            # Handle different height formats
            if height16.ndim == 3:
                height16 = height16[0]  # Take first channel

            # ---- Y-index: require real data ----
            if "y_index" not in data:
                print(f"Skipping {npz_file}: no y_index")
                skipped_missing += 1
                continue
            y_index = int(data["y_index"])

            # ---- Column-level heights: required ----
            if "heightmap_surface" not in data:
                print(
                    f"Skipping {npz_file}: no heightmap_surface "
                    f"(run scripts/add_column_heights.py first)"
                )
                skipped_missing += 1
                continue
            if "heightmap_ocean_floor" not in data:
                print(
                    f"Skipping {npz_file}: no heightmap_ocean_floor "
                    f"(run scripts/add_column_heights.py first)"
                )
                skipped_missing += 1
                continue

            # Create training pairs for all LOD transitions.
            pairs = create_lod_training_pairs(
                labels16=labels16,
                biome_patch=biome16,
                heightmap_patch=height16,
                y_index=y_index,
                heightmap_surface=data["heightmap_surface"],
                heightmap_ocean_floor=data["heightmap_ocean_floor"],
            )

            training_pairs.extend(pairs)

        except ValueError as e:
            # ValueError from create_lod_training_pairs means missing
            # required data (heightmap, biome, etc.)
            print(f"Skipping {npz_file}: {e}")
            skipped_missing += 1
            continue
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            continue

    print(
        f"Generated {len(training_pairs)} total training pairs "
        f"(skipped {skipped_air} near-empty, {skipped_missing} incomplete)"
    )

    # Print distribution by LOD transition
    lod_counts: Dict[str, int] = {}
    for pair in training_pairs:
        transition = pair["lod_transition"]
        lod_counts[transition] = lod_counts.get(transition, 0) + 1
    for transition, count in sorted(lod_counts.items()):
        print(f"  {transition}: {count} pairs")

    return training_pairs


def save_pairs_cache(pairs: List[Dict], path: Path) -> None:
    """Save training pairs as compressed NPZ for fast reloading.

    This is the public API used by ``scripts/build_pairs.py``.
    """
    n = len(pairs)
    print(f"Caching {n} training pairs to {path} ...")
    t0 = time.time()

    # Count source files from the pairs (approximate — for cache validation)
    # We store 0 since we no longer track npz_files in this context.
    np.savez_compressed(
        path,
        parent_voxel=np.array([p["parent_voxel"] for p in pairs], dtype=np.float32),
        biome_idx=np.array([p["biome_idx"] for p in pairs], dtype=np.int32),
        heightmap_patch=np.array([p["heightmap_patch"] for p in pairs], dtype=np.float32),
        height_planes=np.array([p["height_planes"] for p in pairs], dtype=np.float32),
        y_index=np.array([int(p["y_index"]) for p in pairs], dtype=np.int64),
        lod=np.array([int(p["lod"]) for p in pairs], dtype=np.int64),
        target_mask=np.array([p["target_mask"] for p in pairs], dtype=np.uint8),
        target_types=np.array([p["target_types"] for p in pairs], dtype=np.int32),
        lod_transition=np.array([p["lod_transition"] for p in pairs]),
        _n_source_files=np.array([0]),  # unknown at this level
        _min_solid_fraction=np.array([0.0]),
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved {size_mb:.1f} MB in {time.time() - t0:.1f}s")


class MultiLODDataset(Dataset):
    """
    Dataset that provides multi-LOD training pairs from NPZ chunk data.

    Requires a pre-built pair cache (``{split}_pairs_v2.npz``).
    Use ``scripts/build_pairs.py`` to create it.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        lod_sampling_weights: Optional[Dict[str, float]] = None,
        min_solid_fraction: float = 0.02,
        use_pair_cache: bool = True,
    ):
        """
        Initialize multi-LOD dataset from a pre-built pair cache.

        The pair cache MUST be built beforehand using ``scripts/build_pairs.py``.
        This enforces a clean separation between data preparation (which
        validates that all NPZ files contain real router6 noise data) and
        training (which only reads the cache).

        Args:
            data_dir: Directory containing the pair cache (``{split}_pairs_v2.npz``)
            split: "train" or "val"
            lod_sampling_weights: Weights for sampling different LOD transitions
            min_solid_fraction: Ignored when loading from cache (kept for API compat)
            use_pair_cache: Must be True.  Retained for backward compat; False raises.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.min_solid_fraction = min_solid_fraction

        if not use_pair_cache:
            raise ValueError(
                "use_pair_cache=False is no longer supported.  "
                "Run 'python scripts/build_pairs.py --data-dir <dir>' first, "
                "then use use_pair_cache=True (the default)."
            )

        # Default sampling weights (can emphasize certain LOD levels)
        if lod_sampling_weights is None:
            self.lod_sampling_weights = {
                "init_to_lod4": 0.15,  # Init step
                "lod4to3": 0.25,  # Coarsest refinement
                "lod3to2": 0.30,
                "lod2to1": 0.30,  # Finest (no LOD0 — vanilla handles that)
            }
        else:
            self.lod_sampling_weights = lod_sampling_weights

        # Load from pre-built pair cache — no inline NPZ processing
        self.training_pairs: List[Dict] = []
        cache_path = self.data_dir / f"{self.split}_pairs_v2.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Pair cache not found: {cache_path}\n\n"
                f"Training requires a pre-built pair cache.  Run:\n"
                f"    python scripts/build_pairs.py --data-dir {self.data_dir}\n\n"
                f"The pair cache is built from NPZ files with biome,\n"
                f"heightmap, and y_index conditioning (router6 removed)."
            )
        if not self._load_pairs_cache(cache_path):
            raise RuntimeError(
                f"Failed to load pair cache from {cache_path}.  "
                f"Delete it and rebuild: python scripts/build_pairs.py "
                f"--data-dir {self.data_dir} --clean"
            )

    def _generate_all_pairs(self):
        """Legacy method — inline pair generation is no longer supported.

        Use ``generate_pairs_from_npz_files()`` (module-level) instead,
        which is called by ``scripts/build_pairs.py``.
        """
        raise RuntimeError(
            "Inline pair generation during training is no longer supported.  "
            "Run 'python scripts/build_pairs.py --data-dir <dir>' to build "
            "the pair cache, then re-run training."
        )

    # ------------------------------------------------------------------
    # Pair cache I/O
    # ------------------------------------------------------------------

    def _load_pairs_cache(self, path: Path) -> bool:
        """Load pre-computed pairs from NPZ cache.  Returns True on success."""
        try:
            t0 = time.time()
            print(f"Loading cached pairs from {path} ...")
            data = np.load(path, allow_pickle=True)

            # Reconstruct list-of-dicts from stacked arrays
            parent_voxel = data["parent_voxel"]
            biome_idx = data["biome_idx"]
            heightmap_patch = data["heightmap_patch"]
            height_planes = data["height_planes"]
            y_index = data["y_index"]
            lod = data["lod"]
            target_mask = data["target_mask"]
            target_types = data["target_types"]
            transitions = data["lod_transition"]
            n = len(parent_voxel)

            self.training_pairs = []
            for i in range(n):
                self.training_pairs.append(
                    {
                        "parent_voxel": parent_voxel[i].astype(np.float32),
                        "biome_patch": biome_idx[i].astype(np.int64),
                        "biome_idx": biome_idx[i].astype(np.int64),
                        "heightmap_patch": heightmap_patch[i].astype(np.float32),
                        "height_planes": height_planes[i].astype(np.float32),
                        "y_index": np.int64(y_index[i]),
                        "lod": np.int64(lod[i]),
                        "target_mask": target_mask[i].astype(np.float32),
                        "target_types": target_types[i].astype(np.int64),
                        "lod_transition": str(transitions[i]),
                        "parent_size": 8,
                        "target_size": 16,
                    }
                )

            # Build transition index
            self._transition_indices = {}
            for i, pair in enumerate(self.training_pairs):
                t = pair["lod_transition"]
                if t not in self._transition_indices:
                    self._transition_indices[t] = []
                self._transition_indices[t].append(i)
            self._transitions = list(self._transition_indices.keys())
            self._transition_weights = [
                self.lod_sampling_weights.get(t, 1.0) for t in self._transitions
            ]

            elapsed = time.time() - t0
            print(f"  Loaded {n} pairs in {elapsed:.1f}s")
            for t in self._transitions:
                print(f"    {t}: {len(self._transition_indices[t])} pairs")
            return True

        except Exception as e:
            print(f"  Cache load failed ({e}) -- regenerating")
            return False

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        """Get a training sample."""
        # Choose sample based on LOD sampling weights (O(1) via pre-computed indices)
        if random.random() < 0.1:  # 10% direct index sampling
            pair = self.training_pairs[idx % len(self.training_pairs)]
        else:
            # Pick a transition type weighted by lod_sampling_weights
            chosen_transition = random.choices(
                self._transitions, weights=self._transition_weights, k=1
            )[0]
            indices = self._transition_indices[chosen_transition]
            pair = self.training_pairs[random.choice(indices)]

        # Convert to tensors
        # biome_patch stores compact int64 indices (16,16).
        # The model accepts integer indices directly (via Embedding).
        biome_idx_np = np.asarray(pair["biome_patch"])  # (16,16) int64

        sample = {
            "parent_voxel": torch.from_numpy(np.asarray(pair["parent_voxel"])).float(),
            "biome_patch": torch.from_numpy(biome_idx_np).long(),  # (16,16)
            "biome_idx": torch.from_numpy(biome_idx_np).long(),  # (16,16)
            "heightmap_patch": torch.from_numpy(np.asarray(pair["heightmap_patch"])).float(),
            "height_planes": torch.from_numpy(
                np.asarray(pair["height_planes"])
            ).float(),  # (5,16,16)
            "y_index": torch.tensor(int(pair["y_index"]), dtype=torch.long),
            "lod": torch.tensor(int(pair["lod"]), dtype=torch.long),
            "target_mask": torch.from_numpy(np.asarray(pair["target_mask"])).float(),
            "target_types": torch.from_numpy(np.asarray(pair["target_types"])).long(),
            "lod_transition": pair["lod_transition"],
        }

        return sample


def collate_multi_lod_batch(samples: List[Dict]) -> Dict:
    """
    Collate function for multi-LOD batches.

    Uses ALL samples regardless of LOD transition type.  The ``lod_transition``
    metadata is kept as a list, and the caller can inspect it for logging, but
    the model treats every sample identically (the LOD token is an input).
    """
    tensor_keys = [
        "parent_voxel",
        "biome_patch",
        "biome_idx",
        "heightmap_patch",
        "height_planes",
        "y_index",
        "lod",
        "target_mask",
        "target_types",
    ]

    batch: Dict[str, object] = {}
    for key in tensor_keys:
        if key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)

    # Keep lod_transition as the majority type (for logging)
    transitions = [s["lod_transition"] for s in samples]
    batch["lod_transition"] = max(set(transitions), key=transitions.count)
    return batch
