# Dataset Respec Specification

**Status:** Design Phase  
**Target:** Python training pipeline  
**Purpose:** Read native FeatureBundle caches, emit shared inputs, generate LOD targets (1³/2³/4³/8³/16³)

## Overview

The dataset respec pipeline transforms vanilla-generated Minecraft worlds into training data that matches the 5-model LOD hierarchy. It reads native FeatureBundle caches (or generates them on-the-fly), extracts parent-child pairs at all LOD levels, and emits training samples with proper normalization.

## Pipeline Architecture

```
Vanilla World (.mca)
    ↓
[FeatureBundle Cache] (or generate on-the-fly)
    ↓
[Chunk Extraction] → Blocks + Metadata
    ↓
[LOD Pyramid Generation] → 1³/2³/4³/8³/16³ targets
    ↓
[Parent-Child Pairing] → Training samples
    ↓
[Shared Input Assembly] → x_height_planes, x_biome_quart, x_router6, etc.
    ↓
[Dataset Manifest] → Provenance + Statistics
```

## Input Sources

### 1. FeatureBundle Cache (Preferred)

If FeatureBundle cache exists (from Java mod or pre-generation):

```python
# Read from cache directory
cache_dir = Path(".lodiffusion/cache/anchors")
feature_bundle = load_feature_bundle_from_cache(chunk_pos, cache_dir)
```

**Advantages:**
- Consistent anchor signals across all samples
- No re-sampling of vanilla noise
- Matches runtime behavior exactly

### 2. On-the-Fly Generation (Fallback)

If cache doesn't exist, generate FeatureBundle from .mca + seed:

```python
# Use cubiomes or anvil-parser2 to sample vanilla noise
feature_bundle = generate_feature_bundle_from_world(
    chunk_pos, world_seed, mca_file
)
```

**Requirements:**
- Must match Java mod's NoiseTap implementation exactly
- Same normalization rules
- Same sampling coordinates

## LOD Target Generation

### LOD Pyramid Construction

For each 16×16×16 subchunk extracted from vanilla world:

```python
def generate_lod_pyramid(target_blocks_16x16x16):
    """
    Generate LOD pyramid: 16³ → 8³ → 4³ → 2³ → 1³
    
    Returns dict mapping LOD level to downsampled blocks/occupancy.
    """
    lod_pyramid = {}
    
    # LOD0: Full resolution (16×16×16)
    lod_pyramid[0] = {
        "blocks": target_blocks_16x16x16,  # [16, 16, 16]
        "occupancy": blocks_to_occupancy(target_blocks_16x16x16),  # [16, 16, 16]
        "size": 16
    }
    
    # LOD1: 8×8×8 (downsample by 2×2×2)
    lod_pyramid[1] = downsample_lod(lod_pyramid[0], factor=2)
    
    # LOD2: 4×4×4 (downsample by 2×2×2)
    lod_pyramid[2] = downsample_lod(lod_pyramid[1], factor=2)
    
    # LOD3: 2×2×2 (downsample by 2×2×2)
    lod_pyramid[3] = downsample_lod(lod_pyramid[2], factor=2)
    
    # LOD4: 1×1×1 (downsample by 2×2×2)
    lod_pyramid[4] = downsample_lod(lod_pyramid[3], factor=2)
    
    return lod_pyramid
```

### Downsampling Strategy

**For blocks:** Majority vote (most common block in 2×2×2 region)

```python
def downsample_blocks(blocks, factor):
    """
    Downsample blocks by factor using majority vote.
    
    Args:
        blocks: [D, D, D] block IDs
        factor: Downsampling factor (2, 4, 8, 16)
    
    Returns:
        [D//factor, D//factor, D//factor] downsampled blocks
    """
    # Reshape into blocks of factor×factor×factor
    # Take mode (most common) in each block
    # Return downsampled array
```

**For occupancy:** Mean (average occupancy in 2×2×2 region)

```python
def downsample_occupancy(occupancy, factor):
    """
    Downsample occupancy by factor using mean.
    
    Args:
        occupancy: [D, D, D] occupancy (0.0 = air, 1.0 = solid)
        factor: Downsampling factor
    
    Returns:
        [D//factor, D//factor, D//factor] downsampled occupancy
    """
    # Reshape into blocks of factor×factor×factor
    # Take mean in each block
    # Return downsampled array
```

## Training Sample Generation

### Per-Model Samples

For each of the 5 models, generate training samples:

#### Model 0: Init (Noise → LOD4)

```python
sample_init = {
    # Inputs
    "x_parent_prev": np.zeros([1, 1, 1, 1, 1], dtype=np.float32),  # Zeros
    "x_height_planes": feature_bundle.height_planes,  # [1, 5, 1, 16, 16]
    "x_biome_quart": feature_bundle.biome_quart,  # [1, 6, 4, 4, 4]
    "x_router6": feature_bundle.router6,  # [1, 6, 1, 16, 16]
    "x_chunk_pos": np.array([[chunk_x, chunk_z]], dtype=np.float32),  # [1, 2]
    "x_lod": np.array([[4]], dtype=np.float32),  # [1, 1]
    
    # Targets
    "block_logits_target": lod_pyramid[4]["blocks"],  # [1, 1, 1] → expand to [1, N_blocks, 1, 1, 1]
    "air_mask_target": lod_pyramid[4]["occupancy"],  # [1, 1, 1, 1, 1]
}
```

#### Octree sample specification

The dataset now provides three distinct sample types corresponding to the
octree pipeline. Each training example is independent; the recursive structure
is handled during inference by the job scheduler.

1. **Init sample**
   ```python
   sample_init = {
       # Inputs
       "x_height_planes": feature_bundle.height_planes,  # [1,5,16,16]
       "x_router6": feature_bundle.router6,             # [1,6,16,16]
       "x_biome": feature_bundle.biome,                # [1,16,16]
       "x_y_index": feature_bundle.y_index,            # [1]
       # no parent
       # Targets
       "block_logits_target": lod_pyramid[4]["blocks"],  # [1,N,1,1,1]
   }
   ```

2. **Refine sample** (used for all intermediate levels L4→L3, L3→L2, L2→L1, L1→L0):
   ```python
   sample_refine = {
       "x_parent": parent_blocks,      # e.g. [1,N,1,1,1] or [1,N,2,2,2], etc.
       "x_height_planes": feature_bundle.height_planes,
       "x_router6": feature_bundle.router6,
       "x_biome": feature_bundle.biome,
       "x_y_index": feature_bundle.y_index,
       # Targets
       "block_logits_target": child_blocks,  # [1,N,2,2,2]
   }
   ```
   The parent/child sizes double each step; sampling code can iterate levels or
   randomly pick a level to train on.

3. **Leaf sample** (final expansion to 32³):
   ```python
   sample_leaf = {
       "x_parent": l1_parent_blocks,   # [1,N,8,8,8]
       "x_height_planes": feature_bundle.height_planes,
       "x_router6": feature_bundle.router6,
       "x_biome": feature_bundle.biome,
       "x_y_index": feature_bundle.y_index,
       # Targets
       "block_logits_target": leaf_volume,  # [1,N,32,32,32]
   }
   ```

By unifying the sampling logic around octree node expansions, the dataset
avoids any hard-coded LOD labels and simplifies training of the shared
`octree_refine.onnx` model used at multiple depths.

## Shared Input Processing

### Height Planes

From FeatureBundle or vanilla heightmap:

```python
def process_height_planes(heightmap_surface, heightmap_ocean_floor):
    """
    Generate 5 height planes: surface, ocean_floor, slope_x, slope_z, curvature.
    
    Returns: [5, 16, 16] array
    """
    # Compute slopes
    slope_x = np.gradient(heightmap_surface, axis=1)
    slope_z = np.gradient(heightmap_surface, axis=0)
    
    # Compute curvature (second derivative)
    curvature = compute_curvature(heightmap_surface)
    
    # Stack: [surface, ocean_floor, slope_x, slope_z, curvature]
    return np.stack([
        heightmap_surface,
        heightmap_ocean_floor,
        slope_x,
        slope_z,
        curvature
    ], axis=0)
```

### Biome Quart

From FeatureBundle or biome sampling:

```python
def process_biome_quart(biomes_16x16):
    """
    Sample biome features at 4×4×4 quart lattice.
    
    Returns: [6, 4, 4, 4] array (temp, precip[3], isCold, downfall)
    """
    # Downsample biomes to 4×4
    # Extract temperature, precipitation, etc. from biome registry
    # Expand to 4×4×4 (replicate across Y)
    return biome_features_quart
```

### Router6

From FeatureBundle or noise sampling:

```python
def process_router6(noise_config, chunk_pos):
    """
    Sample NoiseRouter at Y=1 for 6 channels.
    
    Returns: [6, 16, 16] array
    """
    # Sample: temperature, vegetation, continents, erosion, depth, ridges
    # At 16×16 grid, Y=1
    return router_slices
```

## Normalization

Apply normalization according to `model_config.json`:

```python
def normalize_inputs(sample, model_config):
    """
    Normalize inputs according to model config.
    """
    # Heights: min-max by world limits
    if "heights" in model_config["normalization"]:
        norm_config = model_config["normalization"]["heights"]
        bottom_y = norm_config["bottomY"]  # -64
        height = norm_config["height"]  # 384
        sample["x_height_planes"] = (sample["x_height_planes"] - bottom_y) / height
    
    # Router6: z-score
    if "router6" in model_config["normalization"]:
        norm_config = model_config["normalization"]["router6"]
        mean = np.array(norm_config["mean"])
        std = np.array(norm_config["std"])
        sample["x_router6"] = (sample["x_router6"] - mean) / std
    
    # Coords: tanh scaling
    if "coords" in model_config["normalization"]:
        scale = model_config["normalization"]["coords"]["scale"]
        sample["x_chunk_pos"] = np.tanh(sample["x_chunk_pos"] / scale)
    
    return sample
```

## Dataset Manifest

Generate manifest with provenance:

```json
{
    "dataset_version": "1.0.0",
    "generation_date": "2025-01-XX",
    "world_seed": 12345,
    "coordinate_bounds": {
        "min_x": -100,
        "max_x": 100,
        "min_z": -100,
        "max_z": 100
    },
    "mod_list": [
        {"name": "fabric-api", "version": "0.125.3"},
        {"name": "carpet", "version": "1.4.112"}
    ],
    "code_commit": "abc123...",
    "block_vocab_version": "1.0",
    "statistics": {
        "total_samples": 10000,
        "samples_per_model": {
            "init": 2000,
            "lod4_to_3": 2000,
            "lod3_to_2": 2000,
            "lod2_to_1": 2000,
            "lod1_to_0": 2000
        },
        "biome_distribution": {...},
        "height_distribution": {...}
    },
    "channel_stats": {
        "height_planes": {
            "mean": [...],
            "std": [...]
        },
        "router6": {
            "mean": [...],
            "std": [...]
        }
    }
}
```

## Output Format

### Per-Sample NPZ Files

Each training sample saved as `.npz`:

```python
np.savez(
    output_path,
    # Inputs
    x_parent_prev=sample["x_parent_prev"],
    x_height_planes=sample["x_height_planes"],
    x_biome_quart=sample["x_biome_quart"],
    x_router6=sample["x_router6"],
    x_chunk_pos=sample["x_chunk_pos"],
    x_lod=sample["x_lod"],
    # Targets
    block_logits_target=sample["block_logits_target"],
    air_mask_target=sample["air_mask_target"],
    # Metadata
    lod_transition=sample["lod_transition"],
    chunk_x=chunk_x,
    chunk_z=chunk_z,
    y_index=y_index
)
```

### Dataset Directory Structure

```
data/
├── processed/
│   ├── model0_init/
│   │   ├── sample_0000.npz
│   │   ├── sample_0001.npz
│   │   └── ...
│   ├── model1_lod4to3/
│   │   └── ...
│   ├── model2_lod3to2/
│   │   └── ...
│   ├── model3_lod2to1/
│   │   └── ...
│   └── model4_lod1to0/
│       └── ...
├── manifest.json
└── channel_stats.json
```

## Implementation Notes

1. **Stratified Sampling:** Balance samples across biomes, LOD levels, and rare features
2. **Boundary Patches:** Include chunk-border samples to teach seam behavior
3. **Validation Split:** Reserve 10-20% for validation
4. **Parallel Processing:** Use multiprocessing for large-scale extraction
5. **Progress Tracking:** Log extraction progress and statistics

## Integration with Training

The dataset respec output feeds directly into PyTorch `Dataset` classes:

```python
class LODDataset(Dataset):
    def __init__(self, data_dir, model_name, split="train"):
        self.samples = load_samples_from_dir(data_dir / model_name / split)
    
    def __getitem__(self, idx):
        sample = np.load(self.samples[idx])
        return {
            "inputs": {
                "x_parent_prev": sample["x_parent_prev"],
                # ... etc
            },
            "targets": {
                "block_logits": sample["block_logits_target"],
                "air_mask": sample["air_mask_target"],
            }
        }
```

## Future Enhancements

- Support for optional channels (barrier, aquifer, cave_prior)
- Halo context generation for seam training
- Augmentation strategies (rotation, flipping with constraints)
- Streaming dataset for very large worlds
