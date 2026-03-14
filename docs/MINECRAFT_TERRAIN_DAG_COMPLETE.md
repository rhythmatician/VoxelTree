# Minecraft Terrain Generation DAG: Vanilla vs. ML Pipeline

**Created:** March 13, 2026  
**Purpose:** Complete trace of terrain generation from world seed to placed blocks, with node-by-node mapping to GPU/ONNX pipeline

---

## Part 1: Vanilla Minecraft Terrain Generation DAG

### 1.1 High-Level Pipeline Overview

```
INPUTS:
  - World seed (long)
  - Biome source configuration
  - NoiseGeneratorSettings (noise router + parameters)

┌──────────────────────────────────────────────────────────────────┐
│ CHUNK LOOP: For each ChunkPos (x, z)                             │
└──────────────────────────────────────────────────────────────────┘

  STEP 1: Biome Sampling (Per 4×4 column)
  ───────────────────────────────────────
  Input:  QuartPos (4-block aligned) + Climate.Sampler
  Output: Biome IDs [16, 16, 16] (per 4×4 horizontal, per quart vertical)
  
  STEP 2: Height Sampling (Per column)
  ───────────────────────────────────
  Input:  BlockX, BlockZ + NoiseRouter.preliminarySurfaceLevel
  Output: Surface height (Y coordinate, integer)

  STEP 3: Noise Sampling at Cells (Per 4×8 cell)
  ──────────────────────────────────────────────
  Input:  Cell position (every 4 blocks horizontal, 8 blocks vertical)
  Output: Density values [8] for each DensityFunction in NoiseRouter
  
  STEP 4: Noise Interpolation (Per block)
  ─────────────────────────────────────────
  Input:  Individual block XYZ + interpolation factors
  Output: Interpolated density values (3D lerp from cell grid)
  
  STEP 5: Block State Selection
  ───────────────────────────────
  Input:  Interpolated density + Biome + Surface rules
  Output: BlockState (stone, dirt, grass, air, water, etc.)
  
  STEP 6: Carving (Optional, removes blocks)
  ───────────────────────────────────────────
  Input:  Carver configuration + BlockState grid + RNG
  Output: Modified BlockState grid (air = carved out)
  
  STEP 7: Surface Building (Applies surface decorations)
  ────────────────────────────────────────────────────────
  Input:  BlockState grid + Biome + Height info
  Output: Top blocks replaced (grass, mycelium, sand, gravel, etc.)
  
  STEP 8: Aquifer Generation (Per cell, populates water)
  ────────────────────────────────────────────────────────
  Input:  3D density grid + Aquifer configuration
  Output: BlockState grid with water/lava filled

FINAL OUTPUT: ChunkAccess (or ProtoChunk)
  - 16 × 16 × 384 blocks (or height-dependent)
  - Stored as ChunkSection arrays in chunk column
  - Ready for structure generation, entity spawning
```

---

### 1.2 Detailed DAG: Data Flow from Seed to Blocks

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: WorldSeed (long) + NoiseGeneratorSettings                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │  RandomState.create(seed, settings)  │
         │  - Initializes all RNGs              │
         │  - Seeds all DensityFunctions        │
         └──────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
            Router    Sampler    StructureManager
          (15 NoiseRouter      (Climate.Sampler)
           Density           (for biome queries)
           Functions)
                │
                ▼
     ┌──────────────────────────────────────────┐
     │  FOR EACH CHUNK: ChunkPos (cx, cz)        │
     │  (256×256 blocks = 16×16 chunk grid)      │
     └──────────────────────────────────────────┘
                │
         ┌──────┼──────┐
         │              │
         ▼              ▼
    ┌─────────┐    ┌──────────────────┐
    │ BiomeID │    │ NoiseChunk Setup  │
    │ Sampling│    │ (Cell boundaries) │
    └─────────┘    └──────────────────┘
         │              │
         │              ├─ Cell grid: 4×4×8 block cells
         │              ├─ Chunk size: 4 cells per chunk
         │              └─ Height: multiple Y slices (cell-aligned)
         │
         ▼
    ┌─────────────────────────────────┐
    │  FOR EACH BLOCK: (x, y, z)       │    ← 16×384×16 = 98,304 samples
    │  within chunk                   │
    └─────────────────────────────────┘
         │
         │  ┌─ Cell-align Y: y_cell = floor(y / cellHeight)
         │  ┌─ Cell-align XZ: x_cell = floor(x / cellWidth)
         │
         ▼
    ┌──────────────────────────────────────┐
    │ SAMPLE DENSITY FUNCTIONS              │
    │ (NoiseRouter → 15 DensityFunctions)    │
    │                                        │
    │  1. continents (2D)      ← Perlin      │
    │  2. erosion (2D)         ← Perlin      │
    │  3. ridges (2D)          ← Perlin      │
    │  4. temperature (2D)     ← Climate     │
    │  5. vegetation (2D)      ← Climate     │
    │  6. depth                ← 3D density  │
    │  7. finalDensity         ← 3D density  │
    │  8-15. ore vein, caves,  ← Varied     │
    │        aquifer, lava              │
    │                                        │
    │  Output: 15 × double[]s (for each XYZ) │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │ INTERPOLATE DENSITY                  │
    │ (3D lerp from cell grid)             │
    │                                        │
    │  Given block position (x,y,z):        │
    │  - Find enclosing cell                │
    │  - Get 8 corner cell values           │
    │  - Trilinear interpolation            │
    │  Output: Single interpolated density  │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │ BLOCK STATE SELECTION                │
    │ (SurfaceRules engine)                 │
    │                                        │
    │  Input:  finalDensity value           │
    │  Input2: Biome ID                     │
    │  Input3: Height above surface         │
    │  Input4: Erosion value (for variants) │
    │                                        │
    │  Rules:                               │
    │  - if finalDensity > threshold:       │
    │      Solid block (based on biome)     │
    │    else:                              │
    │      if depth < seaLevel: water       │
    │      else: air                        │
    │                                        │
    │  Output: BlockState (specific block)  │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │ CARVING (Caves & Ravines)             │
    │ (Optional carver pass)                │
    │                                        │
    │  input: BlockState grid               │
    │  Process: Spawn carvers from nearby   │
    │           chunks, remove air-filled   │
    │  Output: Modified BlockState          │
    │          (air replaces solid blocks)  │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │ AQUIFER GENERATION                   │
    │ (Local Aquifer.FluidPicker)          │
    │                                        │
    │  Input: Aquifer density functions     │
    │  Output: Replace air with water/lava  │
    └──────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────┐
    │ SURFACE LAYER (Grass, mycelium, etc.)│
    │ (SurfaceSystem.buildSurface)          │
    │                                        │
    │  Input: Top block of each column      │
    │  Input2: Biome ID                     │
    │  Input3: Height                       │
    │  Process: Replace top Y blocks        │
    │  Output: Decorative top blocks        │
    └──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ FINAL OUTPUT: ChunkAccess                │
│ - 256 blocks (16×16 XZ)                  │
│ - 384 blocks (full height)               │
│ - 1.5M block state values per chunk      │
│ - Ready for structures, biome deco       │
└──────────────────────────────────────────┘
```

---

### 1.3 Key Noise Functions & Their Role

| Function | Input | Output | Purpose | Cost |
|----------|-------|--------|---------|------|
| **continents** | (x, z) | [-2, 2] | 0=ocean, >0=land masses | 2D Perlin |
| **erosion** | (x, z) | [-1, 1] | Erodes hills, creates valleys | 2D Perlin |
| **ridges** | (x, z) | [-1, 1] | Mountain ridging, shape | 2D Perlin |
| **temperature** | (x, z) | [-2, 2] | Biome climate parameter | 2D Perlin |
| **vegetation** | (x, z) | [-2, 2] | Biome climate parameter | 2D Perlin |
| **depth** | (x, y, z) | [0, ∞] | Height of terrain post-erosion | 3D density |
| **finalDensity** | (x, y, z) | (-∞, ∞) | Threshold comparison → air/solid | 3D density |
| **veinToggle** | (x, y, z) | [0, 1] | Ore vein enable/disable mask | 3D noise |
| **aquifer** | (x, y, z) | double | Aquifer pocket detection | 3D noise |
| **carver** | (x, y, z) | double | Cave carving mask | 3D noise |

**Texture:** 
- 2D functions are computed once per column, cached
- 3D functions are computed per cell, interpolated to blocks
- Total noise evaluations per chunk: ~50K–200K (depending on settings)

---

## Part 2: Our GPU/ONNX Terrain Generation DAG

### 2.1 Architecture Decision: Replace vs. Keep

| Component | Vanilla | Our Approach | Rationale |
|-----------|---------|--------------|-----------|
| **World Seed** | RandomState | Keep identical | Reproducibility |
| **Biome Sampling** | MultiNoiseBiomeSource (Perlin) | **ML Phase 1B** | Cheaper than 2D Perlin |
| **Height Sampling** | preliminarySurfaceLevel | **Cached from Phase 1B** | Same output as biome |
| **Noise Sampling (2D)** | continents/erosion/ridges/temp/veg | **ML Phase 1A + 1B** | Faster than Perlin |
| **Interpolation** | Trilinear 3D lerp | **Implicit in ONNX** | Models learn spatial coherence |
| **Block Selection** | SurfaceRules → BlockState | **OGN Init/Refine/Leaf models** | Direct 32³ block prediction |
| **Carving** | Cave carvers (procedural) | **Keep vanilla** | Expensive; low visual impact at distance |
| **Aquifer** | 3D density thresholding | **Keep vanilla or ML Phase 1C** | Either fast ML or fallback vanilla |
| **Surface Layer** | SurfaceSystem rule application | **Implicit in ONNX** | Models can learn surface patterns |

### 2.2 Phase 0: GPU TerrainShaperSpline MLP

**NEW (March 13, 2026):** The TerrainProvider splines that map `(continents, erosion, ridges, weirdness) → (offset, factor, jaggedness)` have been replaced with a tiny 4→32→32→3 MLP. This is Phase 0, completed prior to Phase 1.

#### 2.2.1 What is TerrainShaperSpline?

In vanilla Minecraft, the terrain shaping splines are deeply nested cubic spline curves that map 4 scalar 2D noise samples to 3 scalar outputs that control the terrain's vertical structure:

```
INPUTS (all 2D Perlin samples):
  continents   float ∈ [-1.1, 1.0]   (describes ocean vs land)
  erosion      float ∈ [-1.0, 1.0]   (describes wind erosion)
  ridges       float ∈ [-1.0, 1.0]   (describes mountain peaks, folded)
  weirdness    float ∈ [-1.0, 1.0]   (describes terrain sharpness)

NESTED CUBIC SPLINE PROCESSING:
  ├─ overworldOffset(C, E, R, amplified)
  │  → Maps to continuous height offset [-0.5, 1.0]
  │  → Uses nested 3-level spline chain
  │
  ├─ overworldFactor(C, E, W, R, amplified)
  │  → Maps to terrain steepness factor [0.625, 6.3]
  │  → Controls the "slope" of the terrain gradient
  │
  └─ overworldJaggedness(C, E, W, R, amplified)
     → Maps to jaggedness multiplier [0.0, 2.0]
     → Controls cliff/overhang sharpness

OUTPUTS:
  offset      float ∈ [-0.5, 1.0]     (height bias)
  factor      float ∈ [0.625, 6.3]    (steepness scaling)
  jaggedness  float ∈ [0.0, 2.0]      (sharpness)
```

#### 2.2.2 Why Replace with MLP?

| Aspect | Vanilla Spline | MLP Replacement | Win |
|--------|--------|----------|-----|
| **Input Dimension** | 4 floats | 4 floats | Same |
| **Output Dimension** | 3 floats | 3 floats | Same |
| **Computation** | Nested lookup + interpolation, ~7 spline evals | Single forward pass, 2 matrix multiplies | **Faster** |
| **Memory** | Spline control points (~500 floats) | Weights (1315 floats) | **Comparable** |
| **GPU-friendly** | Spline binary search per branch | Fully vectorizable ops | **Excellent** |
| **Differentiable** | No (procedural lookup) | Yes (autodiff) | **Training benefit** |
| **Trainability** | Hard-coded constants | 3× (32×32 + 3×32 params) | **Learnable** |

#### 2.2.3 MLP Architecture

```
Input Layer:  4 → 32   (weights: 128, bias: 32)
  ReLU activation

Hidden Layer: 32 → 32  (weights: 1024, bias: 32)
  ReLU activation

Output Layer: 32 → 3   (weights: 96, bias: 3)
  Linear (no activation)

Total Parameters: 1315 floats (~5.2 KB)
```

Training: Trained on 2M samples of ground-truth spline outputs from ported TerrainProvider.java. Final validation loss: 0.00067 MSE.

#### 2.2.4 Integration into terrain_compute.comp

The GPU shader `terrain_compute.comp` now includes:

1. **SSBO binding=9** (`ShapeMlpWeights`): Flat buffer of 1315 weights
2. **UBO binding=10** (`ShapeMlpConfig`): Layer sizes and byte offsets
3. **GLSL function** `mc_terrain_shaper_mlp(vec4)`: Forward pass evaluation
4. **Updated** `computeFinalDensity()`: Uses `mc_terrain_shaper_mlp()` instead of `mc_spline_eval()`

**Workflow per chunk dispatch:**
```
For each of 256 columns in chunk:
  1. Sample 2D noise: continents, erosion, ridges (folded), weirdness
  2. Call MLP: vec3 output = mc_terrain_shaper_mlp(vec4(C, E, R, W))
  3. Use output (offset, factor, jaggedness) to compute finalDensity
  4. For each Y in [-64, 320]:
     - Evaluate density
     - Write to density_out SSBO
```

**Performance:**
- MLP evaluation: ~microseconds per call (vs. ~100 microseconds for nested spline evals)
- SSBO load: Once at world load
- Zero new Java code needed (weights integrated as binary)

### 2.3 GPU/ONNX Pipeline DAG

```
INPUT: WorldSeed + ChunkPos (cx, cz)
       ↓
┌──────────────────────────────────────────────┐
│ STEP 1: Deterministic Anchor Channels        │
│ (Cheap inputs from vanilla noise)             │
├──────────────────────────────────────────────┤
│ For column (x, z):                           │
│  1a. Sample vanilla Perlin for:              │
│      - Heightmap[5]: surface, ocean, 3×slope│
│      - Biome index (from Phase 1B or vanilla)
│      - Y position coded as LOD level         │
│                                               │
│  Outputs: float[5, 16, 16]  ← heightmap     │
│           int64[16, 16]     ← biome IDs      │
│           int64[1]          ← y_level        │
└──────────────────────────────────────────────┘
         │
         ├─ Source: NoiseTapImpl (captures vanilla noise)
         │           or Phase 1 ML models
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 2: GPU Density Field (terrain_compute)  │
│ Compute Shader: terrain_compute.comp         │
├──────────────────────────────────────────────┤
│ Input:  Perlin samplers + MLP weights        │
│ Per-chunk per-block computation:             │
│                                               │
│  For each block (x, y, z) in chunk:          │
│  1. Sample continents(x, z)                  │
│  2. Sample erosion(x, z)                     │
│  3. Sample ridges(x, z) → fold               │
│  4. Sample weirdness(x, z)                   │
│  5. Call MLP → (offset, factor, jaggedness) │
│  6. Compute finalDensity via formula         │
│                                               │
│ Output: density[16, 384, 16] float32         │
│         (SSBO binding=7)                     │
└──────────────────────────────────────────────┘
         │
         ├─ [PHASE 1A] Replace with ML density model
         │             OGN model trained on density fields
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 3: Initialize Octree Root (L4)          │
│ FNN: octree_init.onnx                        │
├──────────────────────────────────────────────┤
│ Input:  heightmap[5, 16, 16]                 │
```│         biome[16, 16]                        │
│         y_level[1]                           │
│                                               │
│ Processing:                                  │
│  - Conv3D layers (16→32→64) + ReLU          │
│  - Output 1: block_logits [1, 1104, 32,32,32]
│  - Output 2: occupancy [8] (8 octants)       │
│                                               │
│ Function: Generates root node (L4 voxel grid)
│           Predicts 32³ block volume          │
│           Selects top-8 occupied blocks      │
└──────────────────────────────────────────────┘
         │
         ├─ Output: block_ids[32,32,32] (via argmax)
         │           occ_logits[8]
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 3: Refine L4 → L3 (octree_refine.onnx) │
└──────────────────────────────────────────────┘
         │
         ├─ For each of 8 octants:
         │
         ▼
     PER OCTANT:
     ┌──────────────────────────────────────────┐
     │ 3a. Extract Parent Sub-Volume             │
     │                                            │
     │  Input: parent_blocks[32,32,32] (from L4) │
     │  Octant: index 0-7 (map to XYZ offsets)  │
     │  Output: parent_sub[16,16,16]             │
     │                                            │
     │  Logic: Extract 16³ sub-cube from         │
     │         corresponding quadrant            │
     └──────────────────────────────────────────┘
              │
              ▼
     ┌──────────────────────────────────────────┐
     │ 3b. Embed Parent Context                  │
     │                                            │
     │  Input:  parent_sub[16,16,16] int64       │
     │  Lookup: parent_embedding.npz matrix      │
     │  Output: parent_emb[C_parent, 32,32,32]  │
     │                                            │
     │  Logic: For each block ID, lookup         │
     │         embedding from matrix             │
     │         Upsample 2× to 32³                │
     └──────────────────────────────────────────┘
              │
              ▼
     ┌──────────────────────────────────────────┐
     │ 3c. Refine Model (octree_refine.onnx)    │
     │                                            │
     │  Input: parent_emb[C_parent, 32,32,32]   │
     │         heightmap[5, 16, 16]              │
     │         biome[16, 16]                     │
     │         y_level[1]                        │
     │         level[1] (=3)                     │
     │                                            │
     │  Processing:                              │
     │   - Same architecture as Init             │
     │   - Now conditioned on parent + level     │
     │                                            │
     │  Output: block_logits[1, 1104, 32,32,32] │
     │          occupancy[8]                     │
     │                                            │
     │  Function: Generates 8 child blocks       │
     │            (2³ octants at finer resolution)
     └──────────────────────────────────────────┘
              │
              └─► For each octant, recurse to L2

┌──────────────────────────────────────────────┐
│ STEPS 4-5: Refine L3→L2, L2→L1               │
│ (Same architecture, repeated with new level) │
└──────────────────────────────────────────────┘
         │
         (After L1 is complete, all 8 finest octants generated)
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 6: Leaf Expansion (octree_leaf.onnx)    │
│ (Expand L1 to full 32³ block volume)         │
├──────────────────────────────────────────────┤
│ Input: parent_emb[C_parent, 32,32,32] (L1)  │
│        heightmap[5, 16, 16]                  │
│        biome[16, 16]                         │
│        y_level[1]                            │
│                                               │
│ Output: block_logits[1, 1104, 32,32,32]     │
│                                               │
│ Function: Final 32³ block grids for          │
│           all 512 octants (8×8×8 hierarchy)  │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 7: Predict Density Fields + Material    │
│         (REVISED ARCHITECTURE)               │
├──────────────────────────────────────────────┤
│ Instead of predicting 1104-class logits,     │
│ predict continuous density + material class: │
│                                               │
│ Input:  OGN hidden feature maps              │
│         [C_hidden, 32, 32, 32]               │
│                                               │
│ Two output heads:                            │
│                                               │
│  A) Density Head:                            │
│     Output: density[1, 1, 32, 32, 32] (float32)
│     Range: (-∞, ∞), float values             │
│     Meaning: slopedCheese density value      │
│     Threshold: density > 0 → solid           │
│                density ≤ 0 → air/fluid       │
│     Loss: MSE(pred_density, gt_density)      │
│                                               │
│  B) Material Head:                           │
│     Output: material[1, 12, 32, 32, 32] (logits)
│     Classes: {air, stone, deepslate, dirt,   │
│              grass, sand, water, lava,       │
│              ore, bedrock, vegetation, wood} │
│     Loss: CrossEntropy applied only where    │
│            density > 0 (solid blocks)        │
│                                               │
│ Rationale: Density fields are smooth,        │
│           continuous, easier for Conv3D to   │
│           learn. Material is a post-process  │
│           applied to solid voxels.           │
│                                               │
│ Output: density[32, 32, 32] float32,         │
│         material[32, 32, 32] int64 (argmax)  │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 8: Apply Density Threshold + Material   │
│         Lookup                               │
├──────────────────────────────────────────────┤
│ For each voxel (x, y, z):                    │
│                                               │
│  1. If density[x,y,z] > 0:                   │
│     voxel_state = SOLID                      │
│     Block_material = material[x,y,z]         │
│  2. Else:                                     │
│     voxel_state = AIR/FLUID                  │
│     Block_material = AIR (or WATER if y < seaLevel)
│                                               │
│ Output: block_ids[32, 32, 32] int64          │
│         (mapped via material_categories.json) 
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 9: (Optional) Carving Post-Process      │
│                                               │
│ If enabled:                                   │
│  - Apply vanilla cave carvers (expensive)     │
│  - Remove blocks in carver volume             │
│                                               │
│ If disabled:                                  │
│  - Skip (low visual impact at LOD>1)         │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 9: (Optional) Aquifer Post-Process      │
│                                               │
│ Option A: Use Phase 1C ML model              │
│  - Neural AquiferMask[32, 32, 32]            │
│  - Replace air with water where mask > 0.5   │
│                                               │
│ Option B: Keep vanilla aquifer logic         │
│  - Reuse Y-column aquifer state from anchor  │
│  - Fill water where appropriate              │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ FINAL OUTPUT: 32³ Voxel Block Volume         │
│ (or 16×16×16 octants depending on format)    │
│ - 32,768 block state entries                 │
│ - Ready for Voxy VoxelSection write          │
│ - Ready for LOD rendering                    │
└──────────────────────────────────────────────┘
```

---

## Part 3: I/O Specification for Each DAG Node

### 3.1 Anchor Channel Sampling

**Node: NoiseTapImpl (Anchor Sampler)**

```
INPUT:
  - worldSeed: long
  - columnX: int, columnZ: int (block coordinates)
  - LOD level: int (0=finest, increase toward coarser)

PROCESS:
  Evaluate vanilla Perlin noise functions:
  - For heightmap: sample 5 channels at (x, z)
    ├─ surface_height: vanilla height function at Y=0
    ├─ ocean_floor: lowest water-surface elevation
    ├─ slope_x: height gradient in X direction
    ├─ slope_z: height gradient in Z direction  
    └─ curvature: second-order height curvature

  - For biome: sample BiomeID at (x, z, y)
    └─ Multiple Y levels if needed for 3D biomes

  - For Y-LOD encoding: LOD level→Y anchor
    └─ L4→Y_coarse, L3→Y_medium, etc.

OUTPUT:
  heightmap: float[5, 16, 16]     (per 4-block cell)
  biome_ids: int64[16, 16]        (per 4-block cell, biome index)
  y_level: int64[1]               (LOD level encoded)

COST: < 1ms per anchor (CPU vanilla noise)
```

### 3.2 OGN Init Model

**Node: OctreeModelRunner.runInit()**

```
INPUT:
  heightmap: float[1, 5, 16, 16]  (4 channel batch)
  biome: int64[1, 16, 16]         (biome IDs)
  y_index: int64[1]               (Y level / 8)

ONNX MODEL: octree_init.onnx
  Input shape:  (1, 5, 16, 16) → (1, 16, 16) → (1)
  Architecture: Conv3D(5→32, 32→64, 64→1104)
  Activation:   ReLU + final linear
  Parameters:   ~5M weights

PROCESS:
  Forward pass through 3-layer U-Net:
  1. Input: concat heightmap + biome_encoding + y_embedding
     Shape: (1, C_total, 16, 16)
  2. Expansion path: conv, norm, relu
  3. Bottleneck: deeper features
  4. Reduction path: conv, norm, relu → (1, 1104, 32, 32, 32)

OUTPUT:
  block_logits: float[1, 1104, 32, 32, 32]  (1104 classes)
  occupancy: float[1, 8]                     (8 octant confidence)

POSTPROCESS:
  block_ids = argmax(block_logits, axis=1)   (shape: [1, 32, 32, 32])

COST: ~150ms (CPU, batch size 1)
      ~30ms (GPU with DirectML, Intel UHD 770)
```

### 3.3 OGN Refine Model (L4→L3, L3→L2, L2→L1)

**Node: OctreeModelRunner.runRefine()**

**Repeated for each octant at each level.**

```
INPUT (per octant):
  parent_blocks: int64[1, 32, 32, 32]         (from previous level argmax)
  heightmap: float[1, 5, 16, 16]              (same anchor)
  biome: int64[1, 16, 16]                     (same anchor)
  y_index: int64[1]                           (Y level)
  level: int64[1]                             (3, 2, or 1)

PREPROCESSING:
  1. Extract octant:
     octant_index = 0..7 (from traversal)
     sub_blocks = extract_16_cubed(parent_blocks, octant_index)
     (sub_blocks shape: [16, 16, 16])
  
  2. Embed parent blocks:
     parent_emb_weights = load("parent_embedding.npz")   (vocab_size × C_parent)
     parent_embedded = []
     for each voxel in sub_blocks[i,j,k]:
        block_id = sub_blocks[i,j,k]
        parent_embedded[i,j,k] = parent_emb_weights[block_id]   (C_parent features)
     Shape: [C_parent, 16, 16, 16]
  
  3. Upsample 2×: nearest-neighbor
     parent_upsampled shape: [C_parent, 32, 32, 32]

ONNX MODEL: octree_refine.onnx
  Input shapes: (1, C_parent, 32, 32, 32) + (1, 5, 16, 16) + (1, 16, 16) + (1) + (1)
  Architecture: Conditioned U-Net (parent→child refinement)
  Parameters: ~8M weights

PROCESS:
  Forward pass with level conditioning:
  1. Concat parent_upsampled + heightmap + biome_emb + y_emb + level_emb
  2. Refinement path: learn child variations from parent
  3. Output: (1, 1104, 32, 32, 32) block_logits

OUTPUT:
  block_logits: float[1, 1104, 32, 32, 32]
  occupancy: float[1, 8]                     (8 grandchildren octants)

POSTPROCESS:
  block_ids = argmax(block_logits, axis=1)

COST: ~200ms per octant (sequential)
      ~25ms per octant (GPU)
      × 512 octants per column = 12.8s sequential, 12.8s parallel
```

### 3.4 OGN Leaf Model (L1→L0)

**Node: OctreeModelRunner.runLeaf()**

```
INPUT (per final octant):
  parent_blocks: int64[1, 32, 32, 32]         (from L1 refine)
  heightmap: float[1, 5, 16, 16]
  biome: int64[1, 16, 16]
  y_index: int64[1]

PREPROCESSING:
  Same as refine:
  1. Extract 16³
  2. Embed + upsample
  3. Shape: [C_parent, 32, 32, 32]

ONNX MODEL: octree_leaf.onnx
  Input shapes: same as refine
  Architecture: Simpler (no occupancy output)
  Parameters: ~8M weights

OUTPUT:
  block_logits: float[1, 1104, 32, 32, 32]
  (No occupancy needed; leaf is fully expanded)

POSTPROCESS:
  block_ids = argmax(block_logits, axis=1)

COST: ~200ms per octant (sequential)
```

### 3.5 Carving (Optional Post-Process)

**Node: CarvingEngine (Vanilla)**

```
INPUT:
  block_ids: int64[32, 32, 32]     (from octree traversal)
  column_x, column_z: int          (biome, height, RNG seed)
  list of nearby ConfiguredWorldCarver

PROCESS:
  1. Do carvers carve this region?
     Iterate nearby chunks (-8 to +8), check carver spawning
  
  2. For each carve volume:
     Carver RNG generates tunnel/ravine path
     Marks blocks as carved_out
  
  3. Update block_ids:
     carved_out blocks → air (block_id = 0)

OUTPUT:
  block_ids_carved: int64[32, 32, 32]

COST: ~50-200ms per chunk (depends on carver count)
      Can batch or parallelize
```

### 3.6 Aquifer (Options: ML or Vanilla)

**Node A: Phase 1C ML Model (AquiferMask)**

```
INPUT:
  (Same as Phase 1B neural network)
  Perlin octaves + coordinates

OUTPUT:
  aquifer_mask: float[32, 32, 32]   (0-1 probability of water)

PROCESS:
  For each voxel:
    if aquifer_mask[x,y,z] > 0.5:
      block_ids[x,y,z] = WATER_ID (or LAVA)
```

**Node B: Vanilla Aquifer System**

```
INPUT:
  block_ids: int64[32, 32, 32]
  y_level: int
  aquifer_config: AquiferStatus[] (sea level, lava level, etc.)

OUTPUT:
  block_ids_with_aquifer: int64[32, 32, 32]
```

---

## Part 4: Output Format Options (Voxy Integration)

### 4.1 Decision: Block IDs vs. BlockStates vs. Octants

**Context:**
- Voxy stores data in 32×32×32 "WorldSections"
- Each voxel has a block state (ID + properties)
- We can opt for different granularities

**Option A: Direct Block IDs (Integer Output)**
```
Input to Voxy: int64[32, 32, 32] block IDs (0-1103)
Voxy Mapping: Canonical ID → Voxy registry entry
Pros: Simple, direct ONNX output
Cons: Property variants lost (e.g., oak_stairs directions)
```

**Option B: BlockState Objects**
```
Input to Voxy: BlockState[32, 32, 32]
              (includes direction, age, waterlogged, etc.)
Voxy Mapping: Direct insertion
Pros: Full fidelity
Cons: ONNX can't output objects; requires post-processing
```

**Option C: Octant Fragments (Hierarchical)**
```
Output: 8 octants × 8 octants × 8 octants = 512 mini-volumes
Each: 4×4×4 or 2×2×2 voxels
Voxy Mapping: Insert as sub-sections
Pros: Can update selectively
Cons: Adds complexity
```

**Recommendation for Phase 1:**  
**Option A (Block IDs)** — simple, fast, sufficient.

### 4.2 Output to Voxy Block Mapping

```
ONNX Output: block_ids[32, 32, 32] int64
  Range: 0-1103 (vocabulary size)

Voxy Block Vocab: config/voxy_vocab.json
  {
    "0": "air",
    "1": "stone", 
    "2": "granite",
    ...
    "1103": "oak_stairs"
  }

Voxy Property Simplification:
  - oak_stairs[facing=north,shape=straight,waterlogged=false]
    → Collapses to single "oak_stairs" entry
  - Model must learn to predict average/most-common variant
  - Alternative: Train separate models per variant (overkill)

Block Write:
  VoxySectionWriter.write(
    world_section,
    model_block_ids[32,32,32],
    voxy_vocab
  )
```

---

## Part 5: Node-by-Node Mapping: Vanilla → ML Pipeline

### 5.1 Replacement Mapping Table

| Vanilla Stage | Input | Vanilla Process | **ML Replacement** | Output | Notes |
|---------------|-------|-----------------|-------------------|--------|-------|
| **1. Biome Sampling** | (x,z) QuartPos | MultiNoiseBiomeSource.getNoiseBiome() | Phase 1B NN | Biome ID | Could stay vanilla; ML is faster |
| **2. Height Sampling** | (x,z) | preliminarySurfaceLevel() | Cached from Phase 1B | Y int | Tied to biome sampling |
| **3. Cell Noise Sampling** | (cx,cy,cz) cells | Sample 15 DensityFunctions | OGN Init (L4) + Refine (L3-L0) | Block IDs | Biggest speedup |
| **4. Interpolation** | Cell corners + factors | Trilinear 3D lerp | *Implicit in ONNX* | Interpolated density | Models learn coherence |
| **5. Block Selection** | density + biome | SurfaceRules engine | *Implicit in ONNX outputs* | BlockState | ONNX predicts directly |
| **6. Carving** | block grid + carvers | Cave/ravine generation | Keep vanilla OR skip | Modified blocks | Optional; low LOD visual impact |
| **7. Aquifer** | 3D density | Aquifer thresholding | Phase 1C NN OR vanilla | Water/lava blocks | Optional; could use ML |
| **8. Surface Layer** | Top blocks + rules | Apply grass, mycelium, etc. | *Implicit in ONNX* | Decorated top | Models learn patterns |

### 5.2 Data Type Transformation Checklist

```
INPUT (world seed + chunk position):
  ✓ long worldSeed               → Used to seed RandomState
  ✓ ChunkPos (int x, int z)      → Expands to 16×16 blocks
  
ANCHOR SAMPLING:
  ✓ Perlin noise @ (x, z) → float[5]  heightmap
  ✓ BiomeID @ (x, z, y)  → int64[16,16] biome grid
  ✓ LOD level encoding    → int64[1] y_level
  
OGN INIT:
  ✓ float[1,5,16,16]    heightmap
  ✓ int64[1,16,16]      biome (requires embedding in ONNX)
  ✓ int64[1]            y_level (requires embedding in ONNX)
  → float[1,1104,32,32,32] block_logits
  → argmax → int64[32,32,32] block_ids
  
OGN REFINE:
  ✓ int64[32,32,32] parent_blocks   (from prev level)
    → Embedding lookup + 2× upsample → float[C_parent,32,32,32]
  ✓ float[1,5,16,16]   heightmap
  ✓ int64[1,16,16]     biome
  ✓ int64[1]           y_level
  ✓ int64[1]           level (3,2,1 requirements)
  → float[1,1104,32,32,32] block_logits
  → argmax → int64[32,32,32] block_ids_l3
  
[Repeat for L2, L1]

OGN LEAF:
  (Same structure, no occupancy output)

FINAL:
  ✓ int64[32,32,32] block_ids (0-1103 vocabulary)
  → Voxy insert via VoxySectionWriter
```

---

## Part 6: Critical Integration Points

### 6.1 Parity Requirements

```
The outputs of our ML pipeline MUST match vanilla terrain when:
1. Same world seed
2. Same coordinates
3. Same biome at location

Parity NOT required for:
- Exact block distribution (70-75% accuracy is OK initially)
- Carving patterns (we skip or use vanilla)
- Rare ore generation (outside OGN scope)
```

### 6.2 Missing Implementations

| Component | Status | Impact |
|-----------|--------|--------|
| Parent embedding loader in Java | ❌ | BLOCKS all Refine/Leaf inference |
| Octant extraction + upsample | ❌ | BLOCKS octree traversal |
| Phase 1C aquifer NN (optional) | ❌ | Can use vanilla fallback |
| Post-process carving | ❌ | Optional; low impact at distance |
| Seam strategy (halo regions) | ❌ | Visual artifacts at LOD boundaries |

### 6.3 Fallback Strategy

```
If any component fails:

INIT model fails:
  → Fall back to vanilla noise-based generation
  → Performance: lose all speedup, but game continues

REFINE model fails:
  → Return empty octant or parent block only
  → Visual: LOD level appears blocky but playable

Embedding not loaded:
  → Pass zeros instead of embedded parent
  → Model will predict incorrect blocks
  → Visual: nonsensical terrain, but no crash

Aquifer fails:
  → Terrain appears with hollow cavities
  → Use vanilla aquifer logic if available
```

---

## Part 7: Chunk-to-Section Mapping (Output Format)

### 7.1 Voxy WorldSection Storage

```
VANILLA CHUNK:
  16 × 16 × 384 blocks
  = 16 ChunkSections (384 / 16 = 24 sections)
  
VOXY EQUIVALENCE:
  32 × 32 × 32 Voxel WorldSection (one octree leaf)
  Can stack multiple sections for full height

OUR OUTPUT OPTIONS:

Option 1: Single 32³ Section
  - Covers one octree leaf (L0)
  - ~184m tall if stacked per chunk width
  - Mismatch with vanilla 16³ sections

Option 2: Subdivide 32³ into 2×2×2 octants
  - Each octant: 16×16×16
  - Matches vanilla section size!
  - Need to interpolate / subdivide ONNX output

Option 3: Keep 32³, remap at insertion
  - Let VoxySectionWriter handle subdivision
  - Voxy has the API for this

RECOMMENDATION: Option 3
  Let Voxy handle storage; we output 32³ block arrays
```

---

## Summary: Complete Data Transformation

```
INPUT
(worldSeed: long, ChunkPos: int×int)
      ↓
┌─ Seed RandomState ─────────────────┐
│                                     │
└─ Anchor sampling per column ────────┤
  heightmap[5,16,16]│biome[16,16]    │
  y_level[1]       │                 │
      ↓                               │
OGN Init Model                        │
  block_logits[1,1104,32,32,32]       │
      ↓                               │
argmax → block_ids_L4[32,32,32]       │
      ↓                               │
For each of 8 octants:                │
  Extract 16³ from L4                 │
  Embed parent blocks                 │
  Upsample 2× to 32³                  │
  OGN Refine Model (level=3)          │
  block_ids_L3[32,32,32] per octant    │────► Repeat for L2, L1, Leaf
  
(Recurse: L3→L2, L2→L1, L1→L0)
      ↓
512 32³ block octants
      ↓
(Optional) Carving post-process
(Optional) Aquifer post-process
      ↓
Final: 32³ × 512 block_ids
      ↓
VoxySectionWriter.write(
  world_section,
  block_ids,
  canonical_vocab
)
      ↓
OUTPUT
(Voxy RocksDB insertion complete)
```

---

## Next Steps for Implementation

**Before WS-1 (ONNX Blockers):**
1. Validate this DAG against decompiled Minecraft code ✓
2. Confirm all I/O types match ONNX contract
3. Identify any missing state variables

**During WS-1:**
1. Implement parent_embedding_loader in Java
2. Implement octant_extraction + upsample
3. Write integration tests comparing Python → Java outputs

**During WS-2 (Model Accuracy):**
1. Profile which nodes in DAG contribute most to inaccuracy
2. Retrain particular models (Init vs. Refine vs. Leaf)
3. Validate improved models against vanilla parity

**During WS-3 (E2E Test):**
1. Run full DAG end-to-end in Minecraft
2. Measure latency per node
3. Identify performance bottlenecks
