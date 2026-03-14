# LODiffusion / VoxelTree — Project Plan

**Revised:** March 13, 2026  
**Scope:** GPU-accelerated terrain generation pipeline for Minecraft LOD rendering  
**Repositories:** VoxelTree (Python ML training), LODiffusion (Java Fabric mod — runtime)

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture](#2-architecture)
3. [The Pipeline: Step by Step](#3-the-pipeline-step-by-step)
4. [What's Built vs. What's Needed](#4-whats-built-vs-whats-needed)
5. [Workstreams](#5-workstreams)
6. [Milestones](#6-milestones)
7. [Risk Register](#7-risk-register)

---

## 1. What This Project Does

**LODiffusion** is a Minecraft Fabric mod that generates distant terrain on the **client GPU** so players can see far-away landscape without the server ever loading those chunks. It feeds its output to **Voxy**, which renders the distant terrain as a sparse octree at 5 LOD levels.

**VoxelTree** is the offline training repo. It trains PyTorch models, exports them to ONNX, and LODiffusion loads those ONNX files at runtime.

### The Core Insight

Minecraft's vanilla terrain generation has two kinds of work:

| Work Type | Examples | GPU Strategy |
|-----------|----------|--------------|
| **Deterministic math** | Perlin noise, NormalNoise, YClampedGradient, peaksAndValleys, ShiftedNoise | **Port to GLSL** — exact, fast, trivially parallelizable |
| **Complex lookup tables** | Nested cubic splines (offset/factor/jaggedness) | **Replace with tiny NNs** — trained to match output, cheaper than the tree traversal |
| **Deeply nested branching** | WeirdScaledSampler, RangeChoice, cave pipeline (8+ noise sources with nested min/max/if), surface rules, biome lookup | **Progressive grokking** — train NN to the point of grokking, faster at runtime than thread-divergent branching on GPU |

We do NOT use neural networks to approximate noise functions. Noise is cheap branchless math — multiply, add, lerp. But anything with deeply nested if/then logic (cave carving, surface rules, block selection) causes **thread divergence** on the GPU — different threads in a warp take different branches, serializing execution. An NN that has grokked the same logic runs as uniform matrix multiplies with zero divergence.

### The End Goal

The density field is an intermediate step. The real goal is predicting **Voxy LOD representations** (block IDs at 5 resolution levels) directly from seed + position, fast enough for real-time gameplay. The early pipeline stages (noise router) are identical to vanilla terrain gen because we need that foundation to be exact before building the block prediction on top of it.

---

## 2. Architecture

### Design Principle: Progressive Grokking

Each stage of Minecraft's terrain pipeline becomes a layer in our network:

1. **Train a tiny NN on the earliest step** until it groks (loss ≈ 0)
2. **Freeze its weights**, turn its output layer into a hidden layer
3. **Add new layers for the next step**, train those
4. Repeat until the full pipeline is one forward pass

This works because each step in Minecraft's pipeline has a clean, well-defined input→output contract. If step N is perfectly learned, step N+1 only needs to learn the delta.

### Two Runtime Paths

```
Path A: GPU Compute Shader (density field)
─────────────────────────────────────────
  Seed + noise params (SSBOs, uploaded once)
    │
    ├── mc_normal_noise()         ← deterministic GLSL (ported vanilla Perlin)
    │   → continents, erosion, weirdness
    │
    ├── peaksAndValleys()         ← deterministic GLSL
    │   → ridges_folded
    │
    ├── TerrainShaperMLP          ← tiny NN (4→32→32→3, SSBO weights)
    │   (continents, erosion, ridges_folded, weirdness) → (offset, factor, jaggedness)
    │
    ├── mc_y_gradient()           ← deterministic GLSL
    ├── mc_normal_noise()         ← deterministic GLSL (sloped_cheese, jagged)
    │
    └── computeFinalDensity()     → float[16 × 384 × 16] density field


Path B: ONNX Block Prediction (OGN models)
───────────────────────────────────────────
  Anchor channels (heightmap, biome, y_position)
    │
    ├── OctreeInitModel           → block_logits[32³], occ_mask[8]     (L4)
    ├── OctreeRefineModel × 3     → block_logits[32³], occ_mask[8]     (L3→L2→L1)
    └── OctreeLeafModel           → block_logits[32³]                  (L0)
    │
    └── argmax → block IDs → VoxySectionWriter → Voxy renders them
```

**Today these are disconnected.** Path A produces density. Path B predicts blocks from anchor channels. The plan is to connect them: Path A's density output becomes conditioning input for Path B, and eventually Path A's stages get absorbed into Path B as frozen hidden layers.

### Voxy LOD Levels

| Level | Resolution | Block Footprint | Voxy Storage |
|-------|-----------|-----------------|--------------|
| L4 | 16 m/voxel | 512³ blocks | WorldSection 32³ × long[32768] |
| L3 | 8 m/voxel | 256³ blocks | WorldSection 32³ × long[32768] |
| L2 | 4 m/voxel | 128³ blocks | WorldSection 32³ × long[32768] |
| L1 | 2 m/voxel | 64³ blocks | WorldSection 32³ × long[32768] |
| L0 | 1 m/voxel | 32³ blocks | WorldSection 32³ × long[32768] |

Each voxel is a 64-bit packed value: block ID (20 bits) + biome ID (9 bits) + light (8 bits).

---

## 3. The Pipeline: Step by Step

This section traces Minecraft's vanilla terrain gen algorithm and maps each step to our implementation.

### Step 0: Seed → Noise Parameters

**Vanilla:** `RandomState.create(seed, settings)` initializes all RNGs, seeds all `DensityFunction` instances, creates permutation tables for Perlin noise.

**Our approach:** `NoiseRouterExtractor.extract(noiseRouter)` walks the vanilla `NoiseRouter` via reflection, serializes every `ImprovedNoise` permutation table, every `PerlinNoise` octave config, and every `NormalNoise` value factor into GPU-compatible buffers. `ShaderSSBOManager.uploadNoiseData()` pushes these to SSBOs (bindings 0–6). This happens **once at world load**.

**Status:** ✅ Done. 533-line `NoiseRouterExtractor.java` + 393-line `ShaderSSBOManager.java`.

### Step 1: 2D Noise Sampling (continents, erosion, ridges)

**Vanilla:** `NormalNoise.getValue(x, 0, z)` for each of continents, erosion, ridges. Each `NormalNoise` is two `PerlinNoise` instances sampled and blended. Each `PerlinNoise` is multiple `ImprovedNoise` octaves summed with amplitudes.

**Our approach:** Deterministic GLSL port in `terrain_compute.comp`:
- `mc_improved_noise()` in `improved_noise.glsl` — gradient noise with permutation lookup
- `mc_perlin_noise()` in `perlin_noise.glsl` — multi-octave sum
- `mc_normal_noise()` in `normal_noise.glsl` — two Perlin + valueFactor blend

**Status:** ✅ Done. Bit-accurate port. Called as `mc_normal_noise(router.nn_continents/nn_erosion/nn_ridges, bx, 0.0, bz)`.

### Step 2: Peaks and Valleys Transform

**Vanilla:** `NoiseRouterData.peaksAndValleys(ridges)` — folds the raw ridges noise into a shape that peaks at mountain ridges.

**Our approach:** Deterministic GLSL:
```glsl
float ridges_abs    = abs(weirdness) * 2.0 - 1.0;
float ridges_folded = mc_half_negative(1.0 - abs(ridges_abs));
```

**Status:** ✅ Done.

### Step 3: Terrain Shaping Splines → TerrainShaperMLP

**Vanilla:** Three deeply nested `CubicSpline` trees with hundreds of control points map `(continents, erosion, ridges_folded, weirdness)` → `(offset, factor, jaggedness)`. These control the terrain's vertical structure — where the surface is, how steep it is, whether cliffs form.

**Our approach:** **Neural network.** A 4→32→32→3 MLP trained on 2M ground-truth spline evaluations. Final validation MSE: 0.00067. Weights stored in SSBO binding 9 (1315 floats, ~5.2 KB). Forward pass runs entirely in GLSL.

**Why NN here:** The spline trees involve recursive branching, binary search, and cubic interpolation — all GPU-hostile operations. The MLP is 2 matrix multiplies with ReLU, fully vectorizable.

**Status:** ✅ Done. Trained, exported, integrated into `terrain_compute.comp` as `mc_terrain_shaper_mlp()`.

### Step 4: Y-Clamped Gradient (Depth)

**Vanilla:** `DensityFunctions.YClampedGradient` maps block Y to a depth value. Overworld: y=−64 → +1.5, y=320 → −1.5.

**Our approach:** Deterministic GLSL:
```glsl
float depth_gradient = mc_y_gradient(by, from_y, to_y, from_val, to_val);
float depth = depth_gradient + terrain_offset;
```

**Status:** ✅ Done.

### Step 5: 3D Noise (Sloped Cheese + Jagged)

**Vanilla:** Two 3D `NormalNoise` samples. "Sloped cheese" is the large-scale vertical variation. "Jagged" adds high-frequency detail near the surface.

**Our approach:** Deterministic GLSL: `mc_normal_noise(router.nn_depth_noise/nn_jagged, bx, by, bz)`.

**Status:** ✅ Done.

### Step 6: Final Density Combination

**Vanilla:**
```java
initialDensity = 4.0 * quarterNegative(factor * (depth + jaggedness));
finalDensity = squeeze(initialDensity + slopedCheese);
```

**Our approach:** Deterministic GLSL:
```glsl
float raw = 4.0 * mc_quarter_negative(terrain_factor * (depth + jaggedness_applied)) + sloped_cheese;
return clamp(mc_squeeze(raw) * 64.0, -64.0, 64.0);
```

**Status:** ✅ Done.

### Step 7: ShiftedNoise (XZ Coordinate Distortion)

**Vanilla:** Before 3D noise sampling, the X and Z coordinates are distorted by `shift_a` and `shift_b` noise values. This prevents axis-aligned artifacts.

**Our approach:** Deterministic GLSL (not yet implemented).

**Status:** ❌ TODO.

### Step 8: Cave Pipeline (WeirdScaledSampler + RangeChoice + underground)

**Vanilla:** The cave system is a deeply nested computation graph:
- **WeirdScaledSampler** — piecewise scaling of 3D noise based on weirdness parameter
- **RangeChoice** — `if (slopedCheese in [-1M, 1.5625]) use surface else use underground`
- **underground** — `max(min(min(cheeseCaves, entrances), spaghetti + roughness), pillars)`
- **noodle** — another RangeChoice with noodle ridge noise
- Result: `min(postProcess(slideOverworld(rangeChoice(...))), noodle)`

This involves 8+ noise sources with nested min/max/rangeChoice — the worst possible case for GPU thread divergence. Different threads in a warp hit different branches, serializing execution.

**Our approach:** **Neural network (progressive grokking Stage 1).** Instead of porting this branchy logic to GLSL, we train an NN on vanilla `finalDensity` dumps. The surface density from Steps 1-6 becomes the frozen backbone; new layers learn the cave modification + post-processing on top. The NN runs as uniform matrix multiplies with zero thread divergence.

**Status:** 🟣 NN target (WS-4 Stage 1). Surface density (no caves) works as M1 interim.

### Step 9: Slide + Blend + Squeeze (Post-Processing)

**Vanilla:** `postProcess(slideOverworld(density))` applies top/bottom edge fading, world blending, interpolation, and squeeze compression.

**Our approach:** Absorbed into the Stage 1 NN alongside caves. The NN learns the full `surface_density → finalDensity` transform including slide/blend/squeeze.

**Status:** 🟣 NN target (WS-4 Stage 1).

### Step 10: Block State Selection

**Vanilla:** `finalDensity < 0` → solid block. `finalDensity >= 0` → air (or water below sea level). Surface rules then replace the top few blocks with biome-appropriate blocks (grass, sand, mycelium, etc.).

**Our approach:** This is where the OGN models take over. Instead of applying Minecraft's `SurfaceRules` engine (which requires biome data, gradient checks, random noise, and dozens of conditional rules), we train the OGN models to **directly predict block IDs** from the density field + anchor channels.

**Status:** 🔄 OGN models exist (v12, ~70-75% accuracy) but are not yet connected to the density field. They currently predict blocks from heightmap/biome only. The density field output from the shader should become additional conditioning input.

### Step 11: Voxy LOD Hierarchy

**Vanilla:** N/A — vanilla Minecraft doesn't have LOD.

**Our approach:** OGN Init (L4) → OGN Refine × 3 (L3→L2→L1) → OGN Leaf (L0). Each level doubles the resolution. The octree structure is written to Voxy's WorldSection storage via `VoxySectionWriter`.

**Status:** 🔄 Models trained but Java integration has critical bugs (parent embedding not loaded, octant extraction not implemented).

---

## 4. What's Built vs. What's Needed

### GPU Compute Shader Pipeline

| Component | File | Status |
|-----------|------|--------|
| Noise parameter extraction | `NoiseRouterExtractor.java` (533 lines) | ✅ Done |
| SSBO upload + management | `ShaderSSBOManager.java` (393 lines) | ✅ Done |
| Shader compilation | `ShaderProgramManager.java` | ✅ Done |
| Compute dispatch | `TerrainComputeDispatcher.java` (355 lines) | ✅ Done |
| ImprovedNoise (gradient noise) | `improved_noise.glsl` | ✅ Done |
| PerlinNoise (multi-octave) | `perlin_noise.glsl` | ✅ Done |
| NormalNoise (dual-Perlin blend) | `normal_noise.glsl` | ✅ Done |
| TerrainShaperMLP | `terrain_compute.comp` (SSBO 9+10) | ✅ Done |
| computeFinalDensity | `terrain_compute.comp` | ✅ Surface density done (Steps 1-6) |
| RouterConfig UBO | Binding 8, 80 bytes | ✅ Done (indices default -1) |
| ShiftedNoise | — | ❌ TODO (GLSL — branchless math) |
| Cave pipeline (WeirdScaledSampler, RangeChoice, underground) | — | 🟣 NN target (WS-4 Stage 1) |
| Slide + blend + squeeze | — | 🟣 NN target (WS-4 Stage 1) |
| Density → block conversion | — | ❌ Not started |

### OGN Block Prediction Pipeline

| Component | File | Status |
|-----------|------|--------|
| Init model (L4) | `octree_init.onnx` (v12) | ✅ Trained, ~70-75% acc |
| Refine model (L3/L2/L1) | `octree_refine.onnx` (v12) | ✅ Trained |
| Leaf model (L0) | `octree_leaf.onnx` (v12) | ✅ Trained |
| Java ONNX runner | `OctreeModelRunner.java` (898 lines) | 🔴 Broken (embedding, octant) |
| Parent embedding loader | — | 🔴 Not implemented |
| Octant extraction/upsample | — | 🔴 Not implemented |
| Voxy section writer | `VoxySectionWriter.java` | ✅ Done |
| Block ID mapper | `VoxyBlockMapper.java` | ✅ Done |
| LOD generation scheduler | `LodGenerationService.java` | ✅ Done |
| Voxy demand queue | `ShadowRouterJobQueue.java` | ✅ Done |

### Training Pipeline (VoxelTree)

| Component | Status |
|-----------|--------|
| Octree data extraction from Voxy RocksDB | ✅ Done |
| OGN model architecture (Init/Refine/Leaf) | ✅ Done |
| Training loop | ✅ Done |
| ONNX export | ✅ Done |
| TerrainShaperMLP training | ✅ Done |
| Noise data extraction (NoiseDumperCommand) | ✅ Done |
| Progressive NN grokking pipeline | ❌ Not started |

---

## 5. Workstreams

### WS-1: Complete the Density Shader (Critical Path)

**Goal:** Finish `terrain_compute.comp` so it produces a correct **surface density** field (no caves). Cave carving and post-processing are NN targets (WS-4), not shader work.

| Task | Description | Effort |
|------|-------------|--------|
| 1.1 | Implement ShiftedNoise (XZ distortion via `shift_a`/`shift_b`) — branchless math | 1-2 days |
| 1.2 | Wire `NoiseRouterExtractor` named indices into `RouterConfig` (replace -1 defaults) | 1 day |
| 1.3 | Validate shader surface density against vanilla Java for 100+ positions | 1 day |

**Deliverable:** Shader produces a surface density field (mountains, oceans, plains) matching vanilla within floating-point tolerance. Caves come later via the NN pipeline.

**Why NOT port caves to GLSL:** WeirdScaledSampler, RangeChoice, and the underground function involve deeply nested if/then branching with 8+ noise sources feeding into min/max/rangeChoice chains. On a GPU, different threads in a warp take different branches, causing thread divergence that serializes execution. An NN trained to grok the same logic runs as uniform matrix multiplies — faster and with zero divergence.

### WS-2: Density → Block ID Conversion

**Goal:** Convert the shader's density output into actual block IDs that Voxy can render.

| Task | Description | Effort |
|------|-------------|--------|
| 2.1 | Implement threshold logic: `density < 0 → stone`, `density >= 0 && y < 62 → water`, else `air` | 1 day |
| 2.2 | Add surface layer heuristic (top 4 blocks of solid → grass_block/dirt/dirt/dirt or biome-appropriate) | 2-3 days |
| 2.3 | Write block IDs to Voxy format (64-bit packed voxels) via `VoxySectionWriter` | 1-2 days |
| 2.4 | End-to-end test: dispatch shader → read density → threshold → write to Voxy → see terrain | 1-2 days |

**Why this matters now:** This gives us a **working MVP** — the player sees distant terrain generated entirely on the GPU. It won't have biome-specific blocks or caves yet, but you'll see mountains, oceans, and plains in the right places. This is the fastest path to a visible result.

### WS-3: Fix OGN Java Integration

**Goal:** Make the existing ONNX octree models actually run in LODiffusion.

| Task | Description | Effort |
|------|-------------|--------|
| 3.1 | Implement parent embedding loader (read `parent_embedding.npz`, do lookup table in Java) | 2 days |
| 3.2 | Implement octant extraction (32³ → extract 16³ octant → upsample 2× → 32³) | 1 day |
| 3.3 | Wire into `LodGenerationService` octree traversal | 1 day |
| 3.4 | Integration tests: Java output matches Python for test vectors | 1-2 days |

**Note:** WS-3 is independent of WS-1/WS-2. OGN models predict blocks from heightmap/biome, not from the density field. This can proceed in parallel.

### WS-4: Progressive Grokking Pipeline (The New Training Strategy)

**Goal:** Build the training infrastructure for the progressive freeze-and-extend approach.

This is the core intellectual contribution of the project. Instead of training one big model, we train the pipeline in stages that mirror Minecraft's own computation graph.

#### Stage 0: TerrainShaperMLP — ✅ ALREADY DONE
- Input: `(continents, erosion, ridges_folded, weirdness)` — 4 floats
- Output: `(offset, factor, jaggedness)` — 3 floats
- Architecture: 4→32→32→3 MLP, ReLU
- Trained on 2M ground-truth spline evaluations. MSE: 0.00067.

#### Stage 1: Density Field Predictor (absorbs caves + post-processing)
- **Input:** Surface density features from shader (continents, erosion, weirdness, ridges_folded, offset, factor, jaggedness, depth, sloped_cheese, surface_density) + block position (x, y, z)
- **Output:** `finalDensity` float (the FULL density including caves, slide, blend, squeeze)
- **Training data:** Vanilla Java `NoiseRouter.finalDensity()` dumps at millions of (x,y,z) positions
- **Architecture:** Freeze TerrainShaperMLP as backbone. New layers learn: WeirdScaledSampler + RangeChoice + underground (cheese/spaghetti/entrances/noodle/pillars) + slide + blend + squeeze
- **Training strategy:** The shader produces surface density (Steps 1-6). The NN extends this with the branchy cave logic that would cause thread divergence in GLSL. Progressive growth: freeze TSM output→hidden, add cave layers, train to grok.
- **Why NN beats GLSL here:** Cave pipeline has 8+ noise sources feeding nested min/max/if branches. Thread divergence kills GPU perf. A grokked NN runs the same logic as uniform matmuls.
- **Grok metric:** MSE(predicted finalDensity, true finalDensity) < 0.001

#### Stage 2: Density → Block Classifier
- **Input:** density field (from Stage 1 or shader) + position context
- **Output:** block class logits (1104 classes) for a volume
- **Training data:** Paired (density field, Voxy block ID) from real Minecraft worlds
- **Architecture:** Freeze Stage 1 as hidden layers. Add classification head.
- **Training strategy:** Stage 1 already produces the density → the new layers learn block selection rules (stone vs. dirt vs. grass vs. water vs. air)
- **Grok metric:** Block accuracy > 90% on held-out test set

#### Stage 3: Multi-LOD Block Predictor
- **Input:** Seed + position (no intermediate density needed)
- **Output:** Voxy-format block volumes at all 5 LOD levels
- **Training data:** Voxy RocksDB octree data from real worlds
- **Architecture:** Unfreeze everything, fine-tune end-to-end with tiny learning rate
- **Training strategy:** The full pipeline (noise→density→blocks) is now one neural network. Fine-tuning lets it learn cross-step correlations the frozen stages missed.
- **Grok metric:** Per-LOD block accuracy > 95%, visual quality indistinguishable from vanilla at each LOD level

| Task | Description | Effort |
|------|-------------|--------|
| 4.1 | Build Stage 1 training dataset (density dumps from shader or Java) | 2-3 days |
| 4.2 | Train Stage 1 density predictor, validate against shader output | 1-2 weeks |
| 4.3 | Build Stage 2 training dataset (density + block ID pairs from Minecraft worlds) | 2-3 days |
| 4.4 | Train Stage 2 block classifier with frozen Stage 1 backbone | 1-2 weeks |
| 4.5 | Build Stage 3 dataset (full Voxy octree data at all LOD levels) | 2-3 days |
| 4.6 | Fine-tune full pipeline end-to-end for multi-LOD prediction | 2-4 weeks |
| 4.7 | Export final model to ONNX, deploy to LODiffusion | 1-2 days |

### WS-5: Integration, Polish & Release

**Goal:** Connect everything, optimize, ship.

| Task | Description | Effort |
|------|-------------|--------|
| 5.1 | Connect shader density output as OGN conditioning input | 2-3 days |
| 5.2 | LOD policy engine (distance → level, player-facing priority, caching) | 1 week |
| 5.3 | Seam strategy (halo overlap between LOD levels) | 1 week |
| 5.4 | INT8 quantization of final ONNX models | 2-3 days |
| 5.5 | In-game debug overlay (LOD boundaries, inference time, queue depth) | 1 week |
| 5.6 | Package as installable Fabric mod, publish | 1 week |

---

## 6. Milestones

### M1: Density Field MVP

**Definition:** Player joins a world → distant terrain shapes are visible (stone/water/air only, generated entirely on client GPU).

**Requires:** WS-1 complete, WS-2 complete.

**Success criteria:**
- [ ] Shader produces correct density field matching vanilla within float tolerance
- [ ] Density is thresholded to stone/water/air and written to Voxy
- [ ] Distant mountains, oceans, and plains appear in correct positions
- [ ] No server involvement for distant terrain
- [ ] Vanilla nearby terrain is unaffected

### M2: OGN Block Quality

**Definition:** Distant terrain has biome-appropriate block types at all 5 LOD levels.

**Requires:** WS-3 complete, WS-4 (Stages 1-2) complete.

**Success criteria:**
- [ ] OGN Init→Refine→Leaf chain runs without crashes
- [ ] Block prediction accuracy ≥ 90% on held-out test set
- [ ] Terrain looks recognizably like Minecraft (grass on plains, sand in deserts, etc.)
- [ ] LOD transitions are acceptable (no jarring seams)

### M3: Production Release

**Definition:** Polished mod ready for public use.

**Requires:** WS-4 (Stage 3) complete, WS-5 complete.

**Success criteria:**
- [ ] Single forward pass predicts blocks at all LOD levels
- [ ] Cold start < 100ms, warm inference < 50ms per chunk column
- [ ] Sprint and elytra flight work without stutter
- [ ] Packaged for CurseForge/Modrinth
- [ ] Works on integrated GPUs (Intel UHD 770)

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Stage 1 density predictor doesn't generalize across seeds | Medium | High | Train on multiple seeds; include seed-derived features as input |
| R2 | Progressive freeze causes information bottleneck | Medium | Medium | If frozen backbone is too lossy, increase its width before proceeding |
| R3 | Block classification can't learn surface rules from density alone | Medium | High | Add biome/erosion/temperature as auxiliary inputs to Stage 2 |
| R4 | Full end-to-end fine-tuning (Stage 3) destabilizes early layers | Low | High | Use very small learning rate (1e-5), layer-wise LR decay |
| R5 | Voxy API breaking changes | Medium | High | Pin Voxy version, abstract behind VoxyCompat |
| R6 | Model too large for integrated GPU VRAM | Medium | Medium | INT8 quantization, fewer channels, LOD-level-specific model sizes |
| R7 | Shader density doesn't match vanilla exactly (floating point) | Low | Medium | Acceptable if error < 0.01; ML stages absorb the noise anyway |
| R8 | Branchy GLSL (caves) causes thread divergence worse than expected | High | High | Don't port it — use NN grokking instead (this is now the plan) |
| R9 | Stage 1 NN can't learn full cave topology from surface features alone | Low | Medium | Add raw 3D noise values as auxiliary inputs; the noise itself is cheap GLSL, only the branching is expensive |

---

## What To Work On Next

```
Right now:
  1. WS-1: Finish the shader (ShiftedNoise only — 1-2 days)
  2. WS-2: Density→block conversion for Voxy (stone/water/air MVP)
  These two give you M1 — visible terrain shapes from surface density.
  No caves yet, but mountains/oceans/plains in the right places.

Parallel (independent):
  3. WS-3: Fix OGN Java integration (embedding, octant extraction)
  This is blocked on nothing and unblocks M2.

After M1:
  4. WS-4: Progressive grokking pipeline
     Stage 1 absorbs caves — train NN on finalDensity dumps.
     TerrainShaperMLP freezes as backbone, new layers learn the
     branchy cave logic as uniform matrix multiplies.
  This is the long pole. Start dataset extraction (finalDensity dumps
  from vanilla Java) as soon as M1 validates surface density.
```
