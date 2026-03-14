# TODO — LODiffusion / VoxelTree

**Branch:** `new-architecture` (both repos)  
**Updated:** March 13, 2026

Read [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for full context.  
See [docs/pipeline.mdd](docs/pipeline.mdd) for the full data-flow graph (Mermaid).

---

## How This Works (30-second version)

Everything runs inside **one GLSL compute shader** (`terrain_compute.comp`).  
The pipeline is a single `main()` function — no pausing, no buffers between stages.  
NN weights are imported as flat float arrays (SSBOs). Inference is literally `dot()` + `max(0,x)`.  
Between NN stages, regular GLSL does noise sampling and arithmetic — same ALU, zero cost to switch.

```
seed → noise params (SSBOs, once per world)
(x, y, z) → 2D noise → Stage 0 NN → surface density (GLSL) → cave noise (GLSL) → Stage 1 NN → Stage 2 NN → block_id → Voxy
              done         done           done                     TODO               TODO          TODO       TODO
```

---

## Priority Order

1. **WS-1** — Finish the density shader (M1 blocker)
2. **WS-2** — Density → block conversion (M1 blocker)
3. **WS-4** — Stage 1 + 2 NN training (M2)
4. **WS-3** — OGN Java integration fixes (independent, can parallel)
5. **WS-5** — Integration & polish (M3)

---

## WS-1: Complete the Density Shader

**Repo:** LODiffusion  
**File:** `src/main/resources/assets/lodiffusion/shaders/worldgen/terrain_compute.comp`  
**Goal:** Surface density field matches vanilla Minecraft (no caves yet)

- [x] **1.1 — Implement ShiftedNoise (XZ distortion)**
  - Port `ShiftedNoise.java` to GLSL
  - Two `mc_normal_noise()` calls: `shift_a`, `shift_b` → shifted x', z'
  - All downstream 2D noise sampling uses shifted coordinates
  - Reference: `NoiseRouterData.bootstrap()` lines 100-103 in `reference-code/26.1-snapshot-11/`
  - This is branchless math: sample two noise fields, add to coords
  - ✅ Implemented: xzScale=0.25 applied, shift_x/z computed from Noises.SHIFT (ShiftA/ShiftB coord
    permutation), all 5 2D noises now at correct (bx*0.25+shift_x, 0, bz*0.25+shift_z)
  - Effort: 1-2 days

- [x] **1.2 — Wire RouterConfig named indices**
  - Replace `-1` defaults in `RouterConfig` UBO (binding 8) with actual noise indices
  - `NoiseRouterExtractor.java` already extracts noise params into SSBOs 0-6
  - ✅ Implemented: fixed `visitNoise()` NoiseHolder handling so ShiftedNoise/ShiftA/ShiftB NormalNoise
    instances are registered; `wireNamedIndices()` resolves continents/erosion/ridges/shift via
    reflection; `ShaderSSBOManager` now calls `withNamedIndices()` from extracted data
  - Note: `nnDepthNoise`/`nnJagged` still -1 (buried in finalDensity tree — deferred)
  - Effort: 1 day

- [ ] **1.3 — Validate shader output vs. vanilla**
  - Run `NoiseDumperCommand` in-game (`/dump_noise`) at 100+ positions
  - Compare shader `surfDens` output to Java reference at same positions
  - Tolerance: `|shader - java| < 0.01`
  - Effort: 1 day

**Done when:** Shader produces correct surface density at verified positions.

---

## WS-2: Density → Block ID Conversion

**Repo:** LODiffusion  
**Goal:** Convert density float → block IDs → Voxy format → visible terrain

- [ ] **2.1 — Threshold logic in shader**
  - Add to `terrain_compute.comp` after `surfDens` computation:
    ```glsl
    int block_id;
    if (surfDens < 0.0) block_id = STONE;
    else if (y < 62 && surfDens >= 0.0) block_id = WATER;
    else block_id = AIR;
    ```
  - This is an MVP — no biome-specific blocks yet
  - Effort: 1 day

- [ ] **2.2 — Surface layer heuristic**
  - Top 4 blocks of solid: `grass_block / dirt / dirt / dirt`
  - Ocean floor: `sand` or `gravel`
  - Uses the `depth` value already computed in the shader
  - Effort: 2-3 days

- [ ] **2.3 — Write to Voxy format**
  - Pack into 64-bit voxels: `block(20b) + biome(9b) + light(8b)`
  - Use existing `VoxySectionWriter.java`
  - Output: `WorldSection` of `long[32768]` (32³ voxels)
  - Effort: 1-2 days

- [ ] **2.4 — End-to-end integration test**
  - Dispatch shader → read density buffer → threshold → write Voxy → see terrain
  - Player should see distant mountains, oceans, plains in correct positions
  - No caves, no grass colors — just shapes
  - Effort: 1-2 days

**Done when:** Player sees distant terrain generated entirely on client GPU.  
**This is Milestone M1.**

---

## WS-3: Fix OGN Java Integration (Independent)

**Repo:** LODiffusion  
**Can run in parallel with WS-1/WS-2** — shares no dependencies.

- [ ] **3.1 — Parent embedding loader**
  - `OctreeModelRunner.java` (898 lines) needs to load `parent_embedding.npz`
  - Implement embedding lookup table in Java
  - The ONNX model expects a `parent_embedding` input tensor — currently crashes
  - Effort: 2 days

- [ ] **3.2 — Octant extraction**
  - Input: parent 32³ block volume
  - Extract one 16³ octant → upsample 2× → feed as 32³ to refine model
  - There are 8 octants per parent — the model refines one at a time
  - Effort: 1 day

- [ ] **3.3 — Wire into LodGenerationService**
  - The octree traversal in `LodGenerationService.java` needs to call Init→Refine→Leaf chain
  - Currently the ONNX pipeline doesn't connect to the LOD scheduler
  - Effort: 1 day

- [ ] **3.4 — Integration tests**
  - Java output must match Python inference for identical inputs
  - Use test vectors from VoxelTree training to verify
  - Effort: 1-2 days

**Done when:** OGN Init→Refine→Leaf chain runs without crashes and produces correct blocks.

---

## WS-4: Progressive Grokking Training Pipeline

**Repo:** VoxelTree  
**Goal:** Train NNs that absorb the branchy cave/block logic as uniform matrix multiplies.

### How progressive grokking works:

1. Train a tiny NN on one step → loss ≈ 0 (grok)
2. Freeze its weights → its output layer becomes a hidden layer
3. Add new layers for the next step → train those
4. Repeat

Each stage's weights end up as a flat float array in a GLSL SSBO.  
At runtime, inference is `dot(weights, input) + bias` → `max(0, result)`. That's it.

### Stage 0: TerrainShaperMLP — DONE ✅

- Architecture: 4→32→32→3
- Input: `(continents, erosion, ridges_folded, weirdness)`
- Output: `(offset, factor, jaggedness)`
- MSE: 0.00067
- Weights exported to `terrain_shaper_weights.bin`, loaded into SSBO binding 9
- Files: `tools/train_terrain_shaper.py`, `tools/extract_terrain_shaper_weights.py`

### Stage 1: Cave + Density MLP — TODO

- [ ] **4.1 — Build Stage 1 training dataset**
  - Run vanilla Minecraft server with `NoiseDumperCommand`
  - Dump `finalDensity` at millions of (x, y, z) positions
  - Also dump the 12 input values per position:
    - `offset, factor, jaggedness` (from Stage 0 / spline eval)
    - `surfDens, slopedCheese` (surface density values)
    - `y` (block height)
    - `entrances, cheeseCaves, spaghetti2D, roughness, noodleToggle, noodleVal` (cave noise)
  - Save as `.npz` files
  - Effort: 2-3 days

- [ ] **4.2 — Train Stage 1**
  - Architecture: 12→64→64→1
  - Freeze TerrainShaperMLP weights (the first 3 outputs are frozen checkpoint values)
  - Train new layers on `MSE(predicted_finalDensity, true_finalDensity)`
  - This NN absorbs: `rangeChoice`, noodle caves, slide, blendDensity, squeeze
  - Grok metric: MSE < 0.001
  - Effort: 1-2 weeks

- [ ] **4.2a — Export Stage 1 weights to GLSL**
  - Same pattern as Stage 0: flatten weights → binary file → SSBO
  - Add `mc_stage1_mlp(float[12])` function to `terrain_compute.comp`
  - Effort: 1-2 days

### Stage 1 also needs cave noise in GLSL:

- [ ] **4.1a — Implement cave noise sampling (GLSL)**
  - 6 noise calls, all branchless `mc_normal_noise()`:
    - `spaghetti2D` — 1 noise call
    - `entrances` — 4 noise calls (multi-octave)
    - `cheeseCaves` — 1 noise call
    - `roughness` — 1 noise call
    - `noodleToggle` — 1 noise call
    - `noodleVal` — 2 noise calls (ridge)
  - These are inputs to Stage 1 NN, not the cave logic itself
  - The branchy part (min/max/if/rangeChoice) is what the NN learns
  - Reference: `NoiseRouter.java` and `NoiseSampler.java` in `reference-code/26.1-snapshot-11/`
  - Effort: 2-3 days

### Stage 2: Block Select MLP — TODO

- [ ] **4.3 — Build Stage 2 training dataset**
  - Extract paired `(finalDensity, block_id)` from real Minecraft worlds
  - Also need climate context: `continents, erosion, temperature, vegetation, depth, weirdness, y`
  - Source: Voxy RocksDB data or ChunkSerializer dumps
  - Effort: 2-3 days

- [ ] **4.4 — Train Stage 2**
  - Architecture: 8→128→128→K (K = number of block classes)
  - Input: `finalDensity` (frozen from Stage 1) + 7 climate values
  - Output: block class logits → argmax → block_id
  - Replaces: `Climate.TargetPoint` + RTree biome lookup + surface rules
  - Grok metric: block accuracy > 90%
  - Effort: 1-2 weeks

- [ ] **4.4a — Export Stage 2 weights to GLSL**
  - Same SSBO pattern
  - Add `mc_stage2_mlp(float[8])` function to `terrain_compute.comp`
  - Effort: 1-2 days

### Stage 3: End-to-End Fine-Tuning — FUTURE

- [ ] **4.5-4.7 — Unfreeze all, fine-tune, ONNX export**
  - Unfreeze everything, train end-to-end with tiny LR (1e-5)
  - Multi-LOD block prediction at all 5 Voxy LOD levels
  - Export to ONNX for runtime (alternative to GLSL SSBOs for larger models)
  - Effort: 2-4 weeks

---

## WS-5: Integration & Polish (After M2)

- [ ] **5.1** — Connect shader density as OGN conditioning input
- [ ] **5.2** — LOD policy engine (distance → level, player priority, caching)
- [ ] **5.3** — Seam strategy (halo overlap between LOD levels)
- [ ] **5.4** — INT8 quantization of final ONNX models
- [ ] **5.5** — In-game debug overlay (LOD boundaries, inference time, queue depth)
- [ ] **5.6** — Package as installable Fabric mod, publish

---

## Milestones

| Milestone | Definition | Requirements | Status |
|-----------|-----------|--------------|--------|
| **M1** | Player sees distant terrain shapes (stone/water/air) generated on client GPU | WS-1 + WS-2 | Not started |
| **M2** | Distant terrain has biome-appropriate blocks at all LOD levels | WS-3 + WS-4 (Stages 1-2) | Not started |
| **M3** | Polished mod, public release | WS-4 (Stage 3) + WS-5 | Not started |

---

## Repository Map

```
LODiffusion/                          (Java Fabric mod — runtime)
├── src/main/java/.../lodiffusion/
│   ├── command/NoiseDumperCommand.java      ← /dump_noise for data extraction
│   ├── gpu/
│   │   ├── NoiseRouterExtractor.java        ← extracts noise params → SSBOs
│   │   ├── ShaderSSBOManager.java           ← SSBO lifecycle
│   │   ├── ShaderProgramManager.java        ← shader compile/link
│   │   ├── TerrainComputeDispatcher.java    ← compute dispatch
│   │   └── TerrainShaperMlpSsbo.java        ← loads MLP weights → SSBO 9
│   ├── voxy/
│   │   ├── VoxySectionWriter.java           ← writes to Voxy format
│   │   └── VoxyBlockMapper.java             ← block name → Voxy ID
│   └── model/OctreeModelRunner.java         ← ONNX inference (broken — WS-3)
├── src/main/resources/.../shaders/worldgen/
│   ├── terrain_compute.comp                 ← THE compute shader (main work here)
│   ├── improved_noise.glsl                  ← gradient noise
│   ├── perlin_noise.glsl                    ← multi-octave Perlin
│   └── normal_noise.glsl                    ← dual-Perlin blend (NormalNoise)
└── build.gradle

VoxelTree/                            (Python ML training — offline)
├── tools/
│   ├── train_terrain_shaper.py              ← Stage 0 trainer (done)
│   ├── extract_terrain_shaper_weights.py    ← Stage 0 weight export (done)
│   ├── phase_1_data_extraction.py           ← density field extractor (framework)
│   ├── phase_1_train_macro_shape.py         ← macro-shape trainer (framework)
│   └── phase_1_train_climate.py             ← climate trainer (framework)
├── docs/
│   ├── PROJECT_PLAN.md                      ← master plan
│   ├── pipeline.mdd                         ← full pipeline DAG (Mermaid)
│   ├── architecture.png                     ← NN architecture diagram
│   └── PHASE_*.md                           ← phase-specific docs
└── schema/material_categories.json

reference-code/26.1-snapshot-11/      (Decompiled Minecraft source — read-only reference)
```

---

## Key Concepts for New Contributors

**Why not just port everything to GLSL?**  
Noise = cheap branchless math → GLSL.  
Caves/biomes = deeply nested if/then branching → GPU thread divergence → NN is faster.

**Why not one big neural network?**  
Progressive grokking: train each pipeline stage separately until loss ≈ 0, freeze, extend.  
Each stage has a clean input→output contract. If stage N is perfect, stage N+1 only learns the delta.

**What are SSBOs?**  
Shader Storage Buffer Objects. Flat arrays of floats uploaded to GPU memory once.  
The NN weights live there. At runtime: `float result = dot(weights[offset], input) + bias`.

**What is Voxy?**  
The LOD renderer. It stores blocks as a sparse octree of 32³ sections.  
Each voxel is a 64-bit `long`: `block_id(20 bits) + biome(9 bits) + light(8 bits)`.  
We generate these sections on the GPU and hand them to Voxy for rendering.

**Where's the reference code?**  
`reference-code/26.1-snapshot-11/` contains decompiled Minecraft 1.21.1 source.  
Key files: `NoiseRouter.java`, `NoiseSampler.java`, `NoiseRouterData.java`, `TerrainProvider.java`.
