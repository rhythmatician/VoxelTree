# Phase 1 Tensor Architecture - NoiseRouter→Neural Network Mapping

Detailed specification of input/output tensor shapes, data types, and generation strategies for the three parallel Phase 1 feature networks.

**NOTE (March 14, 2026):** This document describes **Phase 1A/B/C**, which learn from upstream Perlin noise to predict the 7 terrain-shaping features. This is **distinct** from **Stage1 Density**, which directly learns the 12 finaldensity computation inputs (see [MINECRAFT_TERRAIN_DAG_COMPLETE.md § 2.1.1](MINECRAFT_TERRAIN_DAG_COMPLETE.md)). Both approaches are valid; this doc focuses on the first strategy.

---

## Overview

The three Phase 1 networks decompose Minecraft's `NoiseRouter` into modular sub-problems. Each operates at **cell-level resolution** (4×4×8 = 768 sample points per chunk, not 98,304 blocks) to keep training tractable on CPU/mobile GPUs.

```
Raw Perlin Noise Maps + Coordinates
        │
        ├─────► [Phase 1A] Macro-Shape Net
        │       Output: continents, erosion, ridges
        │
        ├─────► [Phase 1B] Climate & Biome Net
        │       Output: temperature, vegetation
        │
        └─────► [Phase 1C] Subtractive Net
                Output: cave/aquifer probability
```

---

## Phase 1A: Macro-Shape Network
**Target:** Learn the three primary terrain-shaping DensityFunctions: `continents()`, `erosion()`, `ridges()`

### Input Tensor Specification

```
Tensor Name:    "perlin_input_macrofeatures"
Data Type:      float32 (or float16 for low-VRAM systems)
Shape:          (batch_size, grid_x=4, grid_y=48, grid_z=4, channels=11)
Range/Normalization: As specified below
Batch Size:     1000-2000 chunks recommended
Physical Unit:  Samples at 4-block (cellWidth) and 8-block (cellHeight) intervals
```

### Channel Specification (depth=11)

| Channel | Name | Derivation | Range | Purpose |
|---------|------|-----------|-------|---------|
| 0 | `norm_cell_x` | cellX / 4 | [0.0, 1.0] | Spatial position in chunk (X) |
| 1 | `norm_cell_z` | cellZ / 4 | [0.0, 1.0] | Spatial position in chunk (Z) |
| 2 | `norm_cell_y` | (cellY * 8 + minY) / (maxY - minY) | [0.0, 1.0] | Normalized altitude (Y) |
| 3-4 | `perlin_continents_octave[0:2]` | Sample Perlin at scale 43 blocks (octave 0), then 21.5 blocks (octave 1) | [-1.0, 1.0] | Multi-octave Perlin for continents |
| 5-7 | `perlin_erosion_octave[0:3]` | Octaves at scales: 52, 26, 13 blocks | [-1.0, 1.0] | Erosion noise (3 octaves for detail) |
| 8-9 | `perlin_ridges_octave[0:2]` | Octaves at scales: 20, 10 blocks | [-1.0, 1.0] | Ridge noise |
| 10 | `world_seed_feature` | Hash(world_seed, cellX, cellY, cellZ) normalized | [-1.0, 1.0] | Per-cell seed feature |

### Output Tensor Specification

```
Tensor Name:    "macrofeature_predictions"
Data Type:      float32
Shape:          (batch_size, grid_x=4, grid_y=48, grid_z=4, outputs=3)
Range:          [-2.0, 2.0] (allow some swing beyond [-1, 1])
Loss Function:  Mean Squared Error (MSE)
Grok Metric:    MSE < 0.001, MAE < 0.05
```

| Output Channel | Function | Ground Truth | Interpretation |
|---|---|---|---|
| 0 | `continents_density` | `router.continents().compute(context)` | < -0.5 = deep ocean, 0.0 = coast, > 0.5 = land |
| 1 | `erosion_density` | `router.erosion().compute(context)` | High = flattened/eroded, Low = sharp terrain |
| 2 | `ridges_density` | `router.ridges().compute(context)` | High = mountain ridges, Low = valleys |

### Why This Works

- **Shallow network** (2-3 hidden layers, 256-512 units) learns to map Perlin octaves → DensityFunction outputs
- **Cell-level**: Only 768 points per chunk vs 98,304 blocks = 128x fewer GPU cycles
- **Deterministic mapping**: Same input always produces same output (no randomness in Minecraft's functions)
- **Parallelizable**: No inter-chunk dependencies—train 2000 chunks simultaneously

---

## Phase 1B: Climate & Biome Network
**Target:** Learn climate parameters that determine biome distribution

### Input Tensor Specification

```
Tensor Name:    "perlin_input_climate"
Data Type:      float32
Shape:          (batch_size, grid_x=4, grid_y=48, grid_z=4, channels=8)
Batch Size:     Same as Phase 1A (1000-2000 chunks)
Note:           Climate is primarily XZ-dependent, but we keep 3D structure for consistency
```

### Channel Specification (depth=8)

| Channel | Name | Derivation | Range | Purpose |
|---------|------|-----------|-------|---------|
| 0-1 | `norm_cell_x`, `norm_cell_z` | cellX / 4, cellZ / 4 | [0.0, 1.0] | Horizontal position |
| 2 | `cell_y_identity` | Keep as single channel (climate varies minimally with Y) | [0.0, 1.0] | For consistency |
| 3-4 | `perlin_temperature_octave[0:2]` | Scales: 64 blocks, 32 blocks | [-1.0, 1.0] | Temperature oscillation |
| 5-6 | `perlin_vegetation_octave[0:2]` | Scales: 64 blocks, 32 blocks | [-1.0, 1.0] | Humidity/vegetation |
| 7 | `precombined_continents` | Output from Phase 1A, reused | [-1.0, 1.0] | Biomes depend on continents value too |

### Output Tensor Specification

```
Tensor Name:    "climate_predictions"
Data Type:      float32
Shape:          (batch_size, grid_x=4, grid_y=4, grid_z=4, outputs=4)
                NOTE: Reduced Y dimension to 4 (samples at 0, 96, 192, 288 blocks)
                      because climate doesn't vary heavily with Y
Range:          [-1.0, 1.0]
Loss Function:  MSE + Classification Loss (if biome labels available)
Grok Metric:    Biome prediction accuracy > 95%, MSE < 0.005
```

| Output Channel | Function | Ground Truth | Interpretation |
|---|---|---|---|
| 0 | `temperature_factor` | `router.temperature().compute(context)` | -1=cold (snow), 0=temperate, 1=hot (desert) |
| 1 | `vegetation_factor` | `router.vegetation().compute(context)` | -1=dry, 0=moderate, 1=wet (jungle) |
| 2 | `depth_factor` | `router.depth().compute(context)` | Used in biome selection formulas |
| 3 | `biome_confidence` | Derived from climate proximity to known biome centers | [0, 1] = confidence that this is the right biome |

### Why This Works

- **Independent from 1A/1C**: Climate networks can train in parallel without waiting
- **Reduced Y-complexity**: Climate barely changes with altitude, so we sample fewer Y levels (4 instead of 48)
- **Biome validation**: Ground truth can be validated against actual Minecraft biome assignments
- **Linear separability**: Climate parameters form distinct clusters in tensor space = easier for NN to learn

---

## Phase 1C: Subtractive Network (Aquifers & Carvers)
**Target:** Learn the 3D probability of "emptiness" (caves/aquifers) vs solid stone

### Input Tensor Specification

```
Tensor Name:    "perlin_input_carvers"
Data Type:      float32
Shape:          (batch_size, grid_x=4, grid_y=48, grid_z=4, channels=13)
Batch Size:     Same 1000-2000 chunks
Note:           This network handles the 3D carving patterns (caves, ravines, aquifers)
```

### Channel Specification (depth=13)

| Channel | Name | Derivation | Range | Purpose |
|---------|------|-----------|-------|---------|
| 0-2 | `norm_cell_x`, `norm_cell_y`, `norm_cell_z` | cellX/4, cellY/48, cellZ/4 | [0.0, 1.0] | Full 3D spatial position |
| 3 | `depth_gradient` | (cellY * 8 - minY) / height, clamped [0,1] | [0.0, 1.0] | How deep are we? (0 = surface, 1 = bedrock) |
| 4-6 | `perlin_cave_large_octave[0:3]` | Scales: 64, 32, 16 blocks | [-1.0, 1.0] | Large cave structure (3 octaves) |
| 7-9 | `perlin_cave_small_octave[0:3]` | Scales: 16, 8, 4 blocks | [-1.0, 1.0] | Small cave detail (3 octaves) |
| 10-11 | `perlin_aquifer_octave[0:2]` | Scales: 52, 26 blocks | [-1.0, 1.0] | Aquifer positioning noise |
| 12 | `seaLevel_relative_y` | (cellY * 8 - seaLevel) / 10 | [-6.4, 25.6] | Distance to sea level (in blocks, normalized) |

### Output Tensor Specification

```
Tensor Name:    "carver_predictions"
Data Type:      float32
Shape:          (batch_size, grid_x=4, grid_y=48, grid_z=4, outputs=4)
Range:          [0.0, 1.0] (probability outputs)
Loss Function:  Binary Cross-Entropy (BCE) + Dice Loss (for volume accuracy)
Grok Metric:    IoU > 0.75 (Intersection over Union for cave volume)
Ground Truth:   Generated from actual Minecraft carver + aquifer simulation
```

| Output Channel | Function | Ground Truth | Interpretation |
|---|---|---|---|
| 0 | `air_probability` | Is this cell air (1) or stone (0)? | Derived from inverse of finalDensity |
| 1 | `water_probability` | Is this cell water? | Derived from aquifer logic |
| 2 | `cave_uncertainty` | How confident is model about cave YES/NO? | Variance in ensemble predictions |
| 3 | `carver_influence` | How much do carvers affect this location? | From ConfiguredWorldCarver simulation |

### Why This Works

- **Subtractive Logic**: Caves and aquifers *remove* material from the terrain, not add
- **3D Patterns**: Unlike climate (mostly XZ), carving patterns are genuinely 3D (large caves span many Y levels)
- **Probabilistic Output**: Rather than outputting a single "is this air?" binary, output probabilities that can be thresholded
- **Ground Truth from Simulation**: Run the actual Minecraft aquifer + carver code to get labels
- **IoU Metric**: Intersection over Union directly measures cave generation quality (overlap with real caves)

---

## Data Generation Pipeline

### Extracting Ground Truth

```python
def generate_training_data_for_phase_1a(
    world_seed: int, 
    num_chunks: int = 1000
) -> Tuple[Tensor, Tensor]:
    """
    Extract Phase 1A training data from NoiseRouter
    """
    
    # Setup
    random_state = RandomState.create(world_seed, settings)
    router = random_state.router()
    
    inputs = np.zeros((num_chunks, 4, 48, 4, 11), dtype=np.float32)
    outputs = np.zeros((num_chunks, 4, 48, 4, 3), dtype=np.float32)
    
    for chunk_idx in range(num_chunks):
        # For each chunk in the dataset
        chunk_x = (chunk_idx % 32) * 16  # Spread across 32x32 chunk patch
        chunk_z = (chunk_idx // 32) * 16
        
        # For each cell in the chunk
        for cell_x in range(4):
            for cell_y in range(48):
                for cell_z in range(4):
                    block_x = chunk_x + cell_x * 4
                    block_y = -64 + cell_y * 8
                    block_z = chunk_z + cell_z * 4
                    
                    # ==== INPUTS ====
                    # Coordinates
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 0] = cell_x / 4.0
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 1] = cell_z / 4.0
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 2] = (block_y + 64) / 384.0
                    
                    # Perlin octaves (sampled from NoiseRouter internals)
                    perlin_continents_0 = sample_perlin_octave(
                        block_x, block_z, 
                        scale=43,  # First Minecraft octave for continents
                        seed=hash(world_seed, "continents_0")
                    )
                    perlin_continents_1 = sample_perlin_octave(
                        block_x, block_z,
                        scale=21.5,  # Second octave (half frequency)
                        seed=hash(world_seed, "continents_1")
                    )
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 3] = perlin_continents_0
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 4] = perlin_continents_1
                    
                    # Similar for erosion (channels 5-7) and ridges (channels 8-9)
                    # ... [repeat pattern for other octaves]
                    
                    inputs[chunk_idx, cell_x, cell_y, cell_z, 10] = \
                        hash_to_float(world_seed, block_x, block_y, block_z)
                    
                    # ==== OUTPUTS ====
                    context = DensityContext(block_x, block_y, block_z)
                    outputs[chunk_idx, cell_x, cell_y, cell_z, 0] = \
                        router.continents().compute(context)
                    outputs[chunk_idx, cell_x, cell_y, cell_z, 1] = \
                        router.erosion().compute(context)
                    outputs[chunk_idx, cell_x, cell_y, cell_z, 2] = \
                        router.ridges().compute(context)
    
    return torch.from_numpy(inputs), torch.from_numpy(outputs)
```

### Batch Construction Strategy

```python
class NoiseRouterDataLoader:
    """
    Loads pre-computed Minecraft NoiseRouter samples as training batches
    """
    
    def __init__(self, 
                 num_chunks: int = 10000,
                 chunk_sample_region: int = 128,  # 128x128 chunks
                 seed_distribution: str = "random"):
        
        self.num_chunks = num_chunks
        # Generate chunk coordinates sampling region
        self.chunk_coords = self._sample_chunk_region(chunk_sample_region)
        
        # Pre-generate all ground truth once
        self.precompute_all_tensors()
    
    def precompute_all_tensors(self):
        """
        Call the NoiseRouter for every cell location ONCE.
        Save results to disk.
        This is the expensive setup; training iteration is then fast.
        """
        # For Phase 1A, call router 10K chunks × 768 cells × 3 functions
        # = ~23M function evaluations (parallelizable, takes ~2-5 hours on CPU)
        
        # For Phase 1B, similar but with climate functions
        # For Phase 1C, similar with aquifer/carver simulation
        
        # Save to .npz (compressed numpy format) for fast loading
        np.savez_compressed("phase_1a_data.npz", 
                           inputs=self.inputs_1a,
                           outputs=self.outputs_1a)
    
    def __getitem__(self, batch_idx: int) -> Tuple[Tensor, Tensor]:
        # Load batch from pre-computed .npz
        start = batch_idx * batch_size
        end = start + batch_size
        
        return (torch.from_numpy(self.inputs[start:end]),
                torch.from_numpy(self.outputs[start:end]))
```

---

## Sampling Strategy: Perlin Octaves

Minecraft's DensityFunctions are built from **layered Perlin noise**. Here's how to extract raw octave values:

```python
def sample_perlin_octave(
    x: float, 
    z: float, 
    scale: float,           # 43, 21.5, 52, 26, 13, etc.
    seed: int,
    octave_idx: int = 0
) -> float:
    """
    Sample one octave of Perlin noise at a given spatial location.
    
    Args:
        x, z: block coordinates
        scale: sampling interval (larger = smoother, lower frequency)
        seed: noise seed (derived from world seed + function name)
    
    Returns:
        Noise value in [-1.0, 1.0]
    """
    # Normalize coordinates by scale
    sample_x = x / scale
    sample_z = z / scale
    
    # Sample 2D Perlin (Minecraft uses Simplex/Perlin hybrid)
    noise = perlin_noise_2d(sample_x, sample_z, seed)
    
    return noise
```

---

## Validation & Grok Metrics

### Phase 1A Validation

```python
def validate_phase_1a(model, val_loader):
    """
    Check if Phase 1A network has truly 'grokked' the macrofeatures
    """
    total_mse = 0.0
    
    for inputs, outputs in val_loader:
        predictions = model(inputs)
        mse = F.mse_loss(predictions, outputs)
        total_mse += mse.item()
    
    mean_mse = total_mse / len(val_loader)
    
    # Grok threshold
    if mean_mse < 0.001:
        print("✓ Phase 1A GROKKED: MSE < 0.001")
        return True
    else:
        print(f"✗ Phase 1A still learning: MSE = {mean_mse:.6f}")
        return False
```

### Phase 1C Validation (IoU for Caves)

```python
def validate_phase_1c(model, val_loader):
    """
    Intersection over Union for cave generation accuracy
    """
    total_iou = 0.0
    
    for inputs, outputs in val_loader:
        predictions = model(inputs).sigmoid()  # Convert to probabilities
        
        # Threshold at 0.5 (cell is air or not)
        pred_caves = (predictions[:, :, :, :, 0] > 0.5).float()
        true_caves = (outputs[:, :, :, :, 0] > 0.5).float()
        
        # Compute IoU
        intersection = (pred_caves * true_caves).sum()
        union = (pred_caves + true_caves - pred_caves * true_caves).sum()
        iou = intersection / (union + 1e-6)
        
        total_iou += iou.item()
    
    mean_iou = total_iou / len(val_loader)
    
    if mean_iou > 0.75:
        print(f"✓ Phase 1C GROKKED: IoU = {mean_iou:.3f}")
        return True
    else:
        print(f"✗ Phase 1C still learning: IoU = {mean_iou:.3f}")
        return False
```

---

## Memory & Compute Estimates

### Training Phase 1A

```
Input tensor:  (batch=1000, 4, 48, 4, 11) = 8.64 M floats = 34.5 MB per batch
Output tensor: (batch=1000, 4, 48, 4, 3) = 2.3 M floats = 9.2 MB per batch
Total batch:   ~44 MB

Network size:  2-3 hidden layers, 256-512 units
               Parameters: ~100K-200K (tiny model)
               Activations: ~50 MB per batch

Memory required: ~100-150 MB (fits on any GPU, even 2GB Jetson)
Forward pass:  ~1 sec / batch (CPU, optimized)
               ~0.1 sec / batch (GPU)

Training time (10K chunks, 10 epochs):
  CPU: ~28 hours
  GPU: ~2.8 hours
  Distributed (4 GPUs): ~0.7 hours
```

### Training Phase 1B

```
Similar to 1A but with reduced Y dimension (4 instead of 48)
Input tensor: (batch=1000, 4, 4, 4, 8) = 0.512 M floats = 2 MB per batch
Training time: ~3-5 hours (CPU) or ~20 minutes (GPU)
```

### Training Phase 1C

```
Same as 1A (full 3D grid)
Input tensor: (batch=1000, 4, 48, 4, 13) = 12.3 M floats = 49 MB per batch
Training time: ~40 hours (CPU) or ~4 hours (GPU)

Note: Ground truth for 1C requires simulating actual Minecraft aquifer + carver code
      (pre-compute once, reuse forever)
```

---

## Summary: Tensor Shapes at a Glance

| Phase | Input Shape | Input Channels | Output Shape | Output Channels | Training Time |
|-------|------------|---|------------|---|---|
| **1A** | (B, 4, 48, 4, **11**) | Continents/erosion/ridges Perlin + xyz | (B, 4, 48, 4, **3**) | continents, erosion, ridges | ~3-30 hours |
| **1B** | (B, 4, **4**, 4, **8**) | Temp/veg Perlin + xz + continents (reused) | (B, 4, **4**, 4, **4**) | temp, veg, depth, confidence | ~1-5 hours |
| **1C** | (B, 4, 48, 4, **13**) | Cave/aquifer Perlin + depth + xyz | (B, 4, 48, 4, **4**) | air_prob, water_prob, uncertainty, carver | ~4-40 hours |

All three can train **simultaneously** since they have no cross-dependencies during Phase 1.

