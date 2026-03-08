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

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.mipper import build_opacity_table, mip_volume_numpy
from train.anchor_conditioning import approximate_router6_from_biome

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
    y_index: int = 0,
    air_id: int = 0,
    router6: Optional[np.ndarray] = None,
    heightmap_surface: Optional[np.ndarray] = None,
    heightmap_ocean_floor: Optional[np.ndarray] = None,
    slope_x: Optional[np.ndarray] = None,
    slope_z: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    *,
    strict: bool = True,
) -> List[Dict]:
    """
    Create training pairs for incremental LOD refinement from a single 16³ chunk.

    Each transition refines by exactly one LOD level (2× per axis).  The parent
    is the Mipper output at LOD N, upsampled to the canonical 8³ input.  The
    target is the Mipper output at LOD N-1, upsampled to 16³.

    Transitions produced (coarsening factor → parent → target):
        f=2   → 8³ parent (LOD1), target = labels16       (LOD0, 16³)  — LOD1→LOD0
        f=4   → 4³→8³ parent (LOD2), target = mip(·,2)→16³ (LOD1)    — LOD2→LOD1
        f=8   → 2³→8³ parent (LOD3), target = mip(·,4)→16³ (LOD2)    — LOD3→LOD2
        f=16  → 1³→8³ parent (LOD4), target = mip(·,8)→16³ (LOD3)    — LOD4→LOD3

    At runtime, LODiffusion chains these: LOD4→LOD3→LOD2→LOD1→LOD0, each step
    adding one level of detail.  Training mirrors this so the model only ever
    learns tractable 2× super-resolution.

    Anchor channels (height_planes, router6) are passed through directly when
    available from NoiseDumper extraction.  When None, they are approximated
    from the existing heightmap and biome data.

    Args:
        labels16:            (16, 16, 16) int array of block IDs at LOD0
        biome_patch:         (16, 16) int array of biome IDs
        heightmap_patch:     (16, 16) float array of heights
        y_index:             Y-level slab index for this chunk
        air_id:              block ID that represents air (default 0)
        router6:             (6, 16, 16) float32 or None → approximate from biome
        heightmap_surface:   (16, 16) float32 or None → use heightmap_patch
        heightmap_ocean_floor:(16, 16) float32 or None → zeros if not available
        slope_x:             (16, 16) float32 or None → computed from surface height
        slope_z:             (16, 16) float32 or None → computed from surface height
        curvature:           (16, 16) float32 or None → computed from surface height

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
    if heightmap_surface is not None:
        surf = heightmap_surface.astype(np.float32)
        # Detect whether this is raw world-Y coords (values >> 1) or
        # already normalised legacy data (values in [0, 1]).
        # Raw world Y: overworld surface is typically 50–100+, max 320.
        # Normalised legacy: max is 1.0.
        if surf.max() > 2.0:
            # Raw world-Y coordinates — normalise to match Java runtime:
            #   AnchorSampler.computeHeightPlanes divides by HEIGHT_RANGE (320)
            #   ocean_floor = min(h, SEA_LEVEL=62) / 320
            SEA_LEVEL = 62.0
            HEIGHT_RANGE = 320.0
            if heightmap_ocean_floor is not None:
                ofloor = heightmap_ocean_floor.astype(np.float32) / HEIGHT_RANGE
            else:
                ofloor = np.minimum(surf, SEA_LEVEL) / HEIGHT_RANGE
            surf = surf / HEIGHT_RANGE
        else:
            # Already normalised (legacy or synthetic test data)
            if heightmap_ocean_floor is not None:
                ofloor = heightmap_ocean_floor.astype(np.float32)
            elif strict:
                raise ValueError(
                    "heightmap_ocean_floor is required (set strict=False to fall back "
                    "to zeros — not recommended)"
                )
            else:
                import warnings

                warnings.warn(
                    "heightmap_ocean_floor missing — falling back to zeros.",
                    stacklevel=2,
                )
                ofloor = np.zeros_like(surf)
    elif strict:
        raise ValueError(
            "heightmap_surface is required for accurate height conditioning. "
            "Run scripts/add_column_heights.py to add it to existing NPZ files, "
            "or set strict=False to fall back to per-slab heightmap_norm "
            "(WARNING: this produces a train/runtime mismatch)."
        )
    else:
        import warnings

        warnings.warn(
            "heightmap_surface missing — falling back to per-slab heightmap_norm. "
            "This is a known source of train/runtime mismatch! "
            "Run scripts/add_column_heights.py to fix.",
            stacklevel=2,
        )
        surf = heightmap_norm  # per-slab normalised — WRONG but won't crash
        ofloor = np.zeros_like(surf)

    if slope_x is None or slope_z is None or curvature is None:
        # Compute via central differences
        _sx = np.gradient(surf, axis=1).astype(np.float32)
        _sz = np.gradient(surf, axis=0).astype(np.float32)
        _lap = np.gradient(_sx, axis=1) + np.gradient(_sz, axis=0)
        if slope_x is None:
            slope_x = _sx
        if slope_z is None:
            slope_z = _sz
        if curvature is None:
            curvature = _lap.astype(np.float32)

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
    # Compute router6 [6, 16, 16] — real data required
    # ------------------------------------------------------------------
    if router6 is not None:
        router6_tensor = router6.astype(np.float32)  # (6, 16, 16)
    elif strict:
        raise ValueError(
            "router6 is required (set strict=False to approximate from "
            "biome+heightmap heuristics — NOT recommended, the approximation "
            "has a fundamentally different distribution than real router values)"
        )
    else:
        import warnings

        warnings.warn(
            "router6 missing — falling back to approximate_router6_from_biome. "
            "This produces a distribution mismatch vs real noise-router values.",
            stacklevel=2,
        )
        with torch.no_grad():
            _bx = torch.from_numpy(biome_idx).unsqueeze(0).float()  # (1,16,16)
            _hx = torch.from_numpy(heightmap_norm[None, None, ...])  # (1,1,16,16)
            _r6 = approximate_router6_from_biome(_bx, _hx)  # (1,6,16,16)
        router6_tensor = _r6.squeeze(0).numpy()  # (6, 16, 16)

    training_pairs: List[Dict] = []

    for f in (2, 4, 8, 16):
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
        #   f=2  → target is LOD0 = labels16           (already 16³)
        #   f=4  → target is LOD1 = mip(labels16, 2)   → upsample to 16³
        #   f=8  → target is LOD2 = mip(labels16, 4)   → upsample to 16³
        #   f=16 → target is LOD3 = mip(labels16, 8)   → upsample to 16³
        # ------------------------------------------------------------------
        f_target = f // 2  # coarsening factor for the target LOD level

        if f_target == 1:
            # Target is full-detail LOD0
            target_labels = labels16.astype(np.int64)  # (16,16,16)
            target_occ = create_occupancy_from_blocks(labels16, air_id).astype(np.float32)
        else:
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
                "router6": router6_tensor,  # (6,16,16)
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


class MultiLODDataset(Dataset):
    """
    Dataset that provides multi-LOD training pairs from NPZ chunk data.
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
        Initialize multi-LOD dataset.

        Args:
            data_dir: Directory containing NPZ files
            split: "train" or "val"
            lod_sampling_weights: Weights for sampling different LOD transitions
            min_solid_fraction: Skip chunks with fewer solid voxels than this
                                fraction (default 0.02 = 2%).  Filters the ~69%
                                all-air chunks that teach the model nothing
                                about block types.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.min_solid_fraction = min_solid_fraction
        self.use_pair_cache = use_pair_cache

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

        # Load NPZ file paths — check split subdirectory first, then
        # filename pattern, then fall back to all files in data_dir.
        split_subdir = self.data_dir / split
        if split_subdir.is_dir():
            self.npz_files = list(split_subdir.glob("*.npz"))
        else:
            self.npz_files = list(self.data_dir.glob(f"*_{split}_*.npz"))
            if not self.npz_files:
                self.npz_files = list(self.data_dir.glob("*.npz"))

        print(f"Found {len(self.npz_files)} NPZ files for {split} split")

        # Pre-generate all training pairs (memory permitting)
        self.training_pairs: List[Dict] = []
        self._generate_all_pairs()

    def _generate_all_pairs(self):
        """Generate all possible training pairs from NPZ files."""
        # Try loading pre-computed pairs from cache
        if self.use_pair_cache:
            cache_path = self.data_dir / f"{self.split}_pairs_v1.npz"
            if cache_path.exists():
                if self._load_pairs_cache(cache_path):
                    return

        print("Generating multi-LOD training pairs...")
        skipped_air = 0
        total_files = len(self.npz_files)

        for file_idx, npz_file in tqdm(
            enumerate(self.npz_files), total=total_files, desc="Processing NPZ files", unit="file"
        ):
            try:
                data = np.load(npz_file)

                # Extract required fields — support both canonical and legacy key names
                if "labels16" in data:
                    labels16 = data["labels16"]
                elif "target_types" in data:
                    labels16 = data["target_types"].astype(np.int32)
                else:
                    print(f"Skipping {npz_file}: no labels16 or target_types key")
                    continue

                # Filter out nearly-all-air chunks (nothing to learn from)
                solid_frac = (labels16 != 0).mean()
                if solid_frac < self.min_solid_fraction:
                    skipped_air += 1
                    continue

                # ---- Biome: require real data ----
                if "biome16" in data:
                    biome16 = data["biome16"]
                elif "biome_patch" in data:
                    biome16 = data["biome_patch"]
                else:
                    print(f"Skipping {npz_file}: no biome data (biome16 or biome_patch)")
                    continue

                # ---- Heightmap: require real data ----
                if "height16" in data:
                    height16 = data["height16"]
                elif "heightmap_patch" in data:
                    height16 = data["heightmap_patch"]
                else:
                    print(f"Skipping {npz_file}: no heightmap data (height16 or heightmap_patch)")
                    continue

                # Handle different height formats
                if height16.ndim == 3:
                    height16 = height16[0]  # Take first channel

                # ---- Y-index: require real data ----
                if "y_index" not in data:
                    print(f"Skipping {npz_file}: no y_index")
                    continue
                y_index = int(data["y_index"])

                # Create training pairs for all LOD transitions
                # strict=False allows legacy data without anchor fields,
                # but logs warnings so you can see exactly which samples
                # are using fallback conditioning.
                pairs = create_lod_training_pairs(
                    labels16=labels16,
                    biome_patch=biome16,
                    heightmap_patch=height16,
                    y_index=y_index,
                    router6=data.get("router6"),
                    heightmap_surface=data.get("heightmap_surface"),
                    heightmap_ocean_floor=data.get("heightmap_ocean_floor"),
                    slope_x=data.get("slope_x"),
                    slope_z=data.get("slope_z"),
                    curvature=data.get("curvature"),
                    strict=False,  # TODO: flip to True once all data has anchor fields
                )

                self.training_pairs.extend(pairs)

            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                continue

        print(
            f"Generated {len(self.training_pairs)} total training pairs "
            f"(skipped {skipped_air} near-empty chunks)"
        )

        # Print distribution by LOD transition
        lod_counts: Dict[str, int] = {}
        for pair in self.training_pairs:
            transition = pair["lod_transition"]
            lod_counts[transition] = lod_counts.get(transition, 0) + 1

        for transition, count in lod_counts.items():
            print(f"  {transition}: {count} pairs")

        # Pre-compute indices per LOD transition for O(1) sampling
        self._transition_indices: Dict[str, List[int]] = {}
        for i, pair in enumerate(self.training_pairs):
            t = pair["lod_transition"]
            if t not in self._transition_indices:
                self._transition_indices[t] = []
            self._transition_indices[t].append(i)

        # Build weighted transition list for random.choices
        self._transitions = list(self._transition_indices.keys())
        self._transition_weights = [
            self.lod_sampling_weights.get(t, 1.0) for t in self._transitions
        ]

        # Cache computed pairs for fast reloading on subsequent runs
        if self.use_pair_cache and self.training_pairs:
            cache_path = self.data_dir / f"{self.split}_pairs_v1.npz"
            self._save_pairs_cache(cache_path)

    # ------------------------------------------------------------------
    # Pair cache I/O
    # ------------------------------------------------------------------

    def _save_pairs_cache(self, path: Path) -> None:
        """Save computed pairs as compressed NPZ for fast reloading."""
        n = len(self.training_pairs)
        print(f"Caching {n} training pairs to {path} ...")
        t0 = time.time()
        np.savez_compressed(
            path,
            parent_voxel=np.array(
                [p["parent_voxel"] for p in self.training_pairs], dtype=np.float32
            ),
            biome_idx=np.array([p["biome_idx"] for p in self.training_pairs], dtype=np.int32),
            heightmap_patch=np.array(
                [p["heightmap_patch"] for p in self.training_pairs], dtype=np.float32
            ),
            height_planes=np.array(
                [p["height_planes"] for p in self.training_pairs], dtype=np.float32
            ),
            router6=np.array([p["router6"] for p in self.training_pairs], dtype=np.float32),
            y_index=np.array([int(p["y_index"]) for p in self.training_pairs], dtype=np.int64),
            lod=np.array([int(p["lod"]) for p in self.training_pairs], dtype=np.int64),
            target_mask=np.array([p["target_mask"] for p in self.training_pairs], dtype=np.uint8),
            target_types=np.array([p["target_types"] for p in self.training_pairs], dtype=np.int32),
            lod_transition=np.array([p["lod_transition"] for p in self.training_pairs]),
            _n_source_files=np.array([len(self.npz_files)]),
            _min_solid_fraction=np.array([self.min_solid_fraction]),
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Saved {size_mb:.1f} MB in {time.time() - t0:.1f}s")

    def _load_pairs_cache(self, path: Path) -> bool:
        """Load pre-computed pairs from NPZ cache.  Returns True on success."""
        try:
            t0 = time.time()
            print(f"Loading cached pairs from {path} ...")
            data = np.load(path, allow_pickle=True)

            # Validate cache matches current source data
            cached_n_files = int(data["_n_source_files"][0])
            cached_min_solid = float(data["_min_solid_fraction"][0])
            if cached_n_files != len(self.npz_files):
                print(
                    f"  Stale cache: {cached_n_files} source files cached "
                    f"vs {len(self.npz_files)} current -- regenerating"
                )
                return False
            if abs(cached_min_solid - self.min_solid_fraction) > 1e-6:
                print(
                    f"  Stale cache: min_solid_fraction changed "
                    f"({cached_min_solid} vs {self.min_solid_fraction}) -- regenerating"
                )
                return False

            # Reconstruct list-of-dicts from stacked arrays
            parent_voxel = data["parent_voxel"]
            biome_idx = data["biome_idx"]
            heightmap_patch = data["heightmap_patch"]
            height_planes = data["height_planes"]
            router6 = data["router6"]
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
                        "router6": router6[i].astype(np.float32),
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
            "router6": torch.from_numpy(np.asarray(pair["router6"])).float(),  # (6,16,16)
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
        "router6",
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
