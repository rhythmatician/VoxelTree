# LODiffusion — Comprehensive Project Outline

## 1. Vision & Philosophy

### 1.1 Core Idea

LODiffusion replaces most of Minecraft's forward terrain generation with a progressive, ML-driven, reverse-order refinement system. Instead of generating full-resolution terrain immediately, the system:

- Generates coarse LOD terrain first (far from player)
- Refines terrain progressively as the player approaches
- Anchors all macro-structure in vanilla noise functions
- Uses lightweight ONNX models for fast CPU inference
- Integrates with Voxy for multi-resolution rendering
- Preserves gameplay correctness inside simulation distance

### 1.2 Research Inspiration

This project is inspired by Tim Merino et al.'s work on **Discrete Absorbing Diffusion + VQ-VAE for Minecraft terrain generation** (NYU Game Innovation Lab, 2023). Key insights:

- **Discrete Absorbing Diffusion**: Unlike continuous diffusion, discrete diffusion uses masking (absorption) for categorical block data
- **VQ-VAE Compression**: Reduces computational cost by encoding 4×4×4 blocks into discrete codebook tokens
- **Two-Stage Architecture**: Separation of structure generation from texture/material assignment

However, LODiffusion diverges from this work in critical ways:

- **Not diffusion-based**: Uses single-pass CNN refinement (no iterative sampling)
- **Not latent evolution**: Uses hierarchical parent→child refinement, not latent space manipulation
- **Runtime-first**: Must run <100ms on CPU, not offline generation
- **Vanilla-anchored**: Anchors to vanilla noise, not prompt-driven

### 1.3 Core Philosophy: Vanilla-Anchored But Not Vanilla-Exact

**What vanilla noise provides:**

- Macro height structure (continentalness, erosion, peaks/valleys)
- Biome layout (climate parameters)
- Cave likelihood (3D noise fields, carver masks)
- River topology (riverbed corridors)
- Deterministic continuity (seed-stable, seamless)

**What the model learns:**

- Multi-resolution coherence (coarse LOD approximations that look believable)
- Sub-vanilla detail synthesis (naturalistic cliff breakup, organic cave walls)
- Material distribution beyond simple rules (realistic stratification, biome-blended transitions)
- Parent→child structural consistency (how coarse voxels expand into fine voxels)
- Hierarchical refinement (8³ → 16³, 4³ → 8³, etc.)

**What the model does NOT do:**

- Move mountains horizontally
- Ignore biome signals
- Invent caves where vanilla says none exist
- Create seams between patches
- Match vanilla bit-for-bit (statistical similarity is sufficient)

### 1.4 Success Criteria

The system is successful when:

- Player can stand still → terrain fills in all directions seamlessly
- Player can sprint → forward cone prioritized without stuttering
- Player can elytra at high speed → no invisible walls or collision failures
- Player increases render distance → no catastrophic performance drop
- Player restarts world → distant terrain instantly visible from cache
- World feels like vanilla → but smoother and more scalable

## 2. System Architecture

### 2.1 Languages & Tooling Stack

| Layer | Language/Tool | Responsibility |
|-------|---------------|----------------|
| **Training** | Python 3.13+ (PyTorch) | Dataset extraction, model training, ONNX export |
| **Runtime** | Java 21 (Fabric mod) | Anchor sampling, job scheduling, ONNX inference via DJL |
| **Model Contract** | ONNX 1.12+ | Static-shape deterministic inference |
| **LOD Rendering** | Voxy (Rust/Java) | Multi-resolution rendering and caching |
| **World Freeze** | Carpet Mod | Deterministic training world generation |
| **World Parsing** | anvil-parser2 | Reading .mca files for dataset extraction |

### 2.2 LOD Hierarchy

The system uses a 5-model family for progressive refinement:

| Model | Purpose | Input `x_parent_prev` | Output (`block_logits`, `air_mask`) |
|-------|---------|----------------------|-------------------------------------|
| **Init** | Noise → LOD4 | `[1,1,1,1,1]` (zeros) | `[1,N,1,1,1]`, `[1,1,1,1,1]` |
| **LOD4→3** | Refine LOD4 | `[1,1,1,1,1]` | `[1,N,2,2,2]`, `[1,1,2,2,2]` |
| **LOD3→2** | Refine LOD3 | `[1,1,2,2,2]` | `[1,N,4,4,4]`, `[1,1,4,4,4]` |
| **LOD2→1** | Refine LOD2 | `[1,1,4,4,4]` | `[1,N,8,8,8]`, `[1,1,8,8,8]` |
| **LOD1→0** | Refine LOD1 | `[1,1,8,8,8]` | `[1,N,16,16,16]`, `[1,1,16,16,16]` |

**Key principles:**

- LOD0 = full resolution (authoritative gameplay terrain)
- LOD1+ = progressively coarser representations
- All LOD levels are deterministic, worldspace-consistent, and seam-aware
- No upsampling in the mod; models contain static Resize/conv internally
- Vanilla `carve()` runs only at LOD0 to finalize caves/aquifers/structures

### 2.3 Anchor Channels (Shared Inputs)

All models share these deterministic signals derived from vanilla noise:

| Channel | Shape | Description |
|---------|-------|-------------|
| `x_height_planes` | `[1,5,1,16,16]` | Surface, ocean_floor, slope_x, slope_z, curvature |
| `x_biome_quart` | `[1,6,4,4,4]` | Temperature, precipitation[3], isCold, downfall |
| `x_router6` | `[1,6,1,16,16]` | Temperature, vegetation, continents, erosion, depth, ridges |
| `x_chunk_pos` | `[1,2]` | Chunk coordinates (x, z) for global coherence |
| `x_lod` | `[1,1]` | LOD timestep embedding |
| `x_barrier` (opt) | `[1,1,1,16,16]` | Coastal barrier mask |
| `x_aquifer3` (opt) | `[1,3,1,16,16]` | Aquifer surface wetness |
| `x_cave_prior4` (opt) | `[1,1,4,4,4]` | Coarse cave likelihood (LOD1→0 only) |

**Normalization:**

- Heights: min-max by world limits (-64 to 320)
- Router/Aquifer/Cave: z-score
- Flags: [0,1]
- Coords: affine/tanh scaling

### 2.4 Model Architecture

**Design constraints:**

- Pure Conv3D + GroupNorm + ReLU (+ Resize/nearest or strided-conv)
- No dynamic ops (required for ONNX export)
- Static shapes only
- CPU-friendly operations
- Deterministic inference

**Output schema:**

- `block_logits`: `[1, N_blocks, D, D, D]` where D is the target resolution
- `air_mask`: `[1, 1, D, D, D]` probability that a voxel is empty

**Block vocabulary (Phase 1):**

- Terrain-only: Air, Stone, Dirt/Grass, Sand/Gravel, Water
- Optional: Lava, Snow/Ice
- Trees/plants excluded for Phase 1
- Full vocabulary: 1104 Minecraft blocks (for future phases)

## 3. Dependencies & Environment

### 3.1 Pinned Dependency Matrix (Minecraft 1.21.5)

**Java Toolchain:**

- Java 21 (required for modern MC + Fabric ecosystem)

**Mod Stack:**

- Minecraft: 1.21.5
- Fabric Loader: (pinned version TBD)
- Fabric API: (pinned version TBD)
- Voxy: (pinned commit/branch TBD)
- Carpet Mod: (pinned version TBD)
- LODiffusion mod: (ours)

**ML Runtime Stack:**

- ONNX: 1.12+ (static shapes)
- DJL: (pinned version TBD)
- ONNX Runtime: CPU backend only

**Python Training Stack:**

- Python: 3.13+
- PyTorch: >=2.0
- NumPy, SciPy, PyYAML, tqdm
- anvil-parser2 (Minecraft .mca parsing)
- ONNX, ONNX Runtime (for export validation)

**What to avoid in demo milestone:**

- Extra worldgen mods
- Shader mods
- Anything that alters biome/noise settings
- "Nice-to-have" optimization mods that might affect timing

### 3.2 World Generation for Training

**Requirements:**

- Fixed seed(s) for reproducibility
- Defined region/radius to pregenerate
- Simulation freeze during and after generation
- Version-locked mod list

**Freeze strategy:**

- Use Carpet mod + gamerules:
  - `randomTickSpeed 0` (disable random ticks)
  - `doFireTick false` (prevent fire spread)
  - `mobGriefing false` (prevent mob block changes)
- Goal: World does not evolve between chunk samples

**Generation method:**

- Phase 1: Use pregen mod or scripted server loop
- Future: Consider headless generation harness for scale

### 3.3 Dataset Extraction Pipeline

**Inputs:**

- Vanilla-generated world files (.mca)
- Biome information / heightmaps from engine
- Anchor cache (optional; nice for reuse)

**Outputs:**

- Training samples: `(x_parent, anchors...) → y_child`
- Anchor cache (per region)
- Evaluation snapshots
- Dataset manifest with provenance

**Provenance tracking:**

- Seed
- Coordinate bounds
- Mod list + versions
- Code commit hash
- Block vocabulary version

## 4. Training Pipeline

### 4.1 Dataset Construction

**Process:**

1. Read .mca files using anvil-parser2
2. Generate parent-child pairs via LOD pyramid (2×2×2 max pooling)
3. Compute anchors (height, biome, router, etc.) from vanilla noise
4. Stratified sampling (biome balancing, rare-feature oversampling)
5. Boundary patches (to teach seam behavior)

**LOD pyramid factors:** [1, 2, 4, 8, 16]

- Factor 1: LOD0 (full resolution)
- Factor 2: LOD1 (8×8×8)
- Factor 4: LOD2 (4×4×4)
- Factor 8: LOD3 (2×2×2)
- Factor 16: LOD4 (1×1×1)

### 4.2 Training Regime

**Multi-LOD training:**

- Random coarsening factor sampling during training
- Dynamic parent generation from target occupancy
- Scheduled sampling: Probability ramp from 0.0→0.3 over epochs
- Teacher forcing reduction: Mixes model predictions with ground truth

**Loss functions:**

- Primary: CrossEntropy over `block_logits`
- Secondary: Binary mask loss for `air_mask`
- Anchor-consistency losses:
  - Surface loss: Predicted surface ≈ x_height
  - Cave loss: Air probability correlates with x_cave
  - River loss: Encourage valley/water continuity where x_rriver=1
  - Parent consistency: Child downsampled → parent

**Metrics:**

- Overall accuracy: Block-level prediction accuracy
- Air/Solid IoU: Intersection-over-Union for occupancy
- Per-LOD accuracy: Metrics broken down by LOD level
- Frequent-set accuracy: Top-K most common blocks
- Seam metrics: Continuity across chunk borders

### 4.3 Model Export

**Artifacts:**

- `model.onnx` (one per LOD step, 5 total)
- `model_config.json` (metadata: input/output names, shapes, normalization, block palette)
- `test_vectors.npz` (input/output examples for DJL parity validation)

**Export requirements:**

- Static shapes only
- Opset 17+ (ONNX)
- No unsupported ops
- Deterministic inference
- DJL compatibility verified

## 5. Runtime Integration

### 5.1 Job Scheduler

**Priority logic:**

1. Spawn sphere (force LOD0 for gameplay correctness)
2. Movement cone (narrow at speed, prioritize forward direction)
3. Idle fill-in (expand to 360° when stationary)
4. Background refinement (coarser LODs for distant terrain)

**Constraints:**

- Bound CPU time per tick (must not spike MSPT)
- Avoid blocking render thread
- Cache anchors and LODs (don't recompute)
- Graceful fallback if inference can't keep up (render coarser)

### 5.2 Caching Strategy

**Anchor cache:**

- Computed once per chunk/region
- Stored alongside .lod files
- Keyed by ChunkPos
- LRU eviction with optional disk sidecar

**LOD cache:**

- Generated sections cached per level
- Invalidation rules: seed change, mod change, model change
- Compression: zstd/lz4 for disk storage
- Versioning: model_config.json + cache version

### 5.3 Voxy Integration

**Requirements:**

- Generate LOD sections compatible with Voxy's 32³ section format
- Cache multi-level sections
- Ensure seam stability
- Support reload persistence

**Voxy section format:**

- 32×32×32 blocks per section
- Keyed by `(lvl, x, y, z)` in 64-bit key
- Storage: RocksDB with "world_sections" column family
- Serialization: palette + 16-bit indices + metadata

**Integration approach:**

- Generate our LOD patches (16³ or 32³)
- Assemble into Voxy sections
- Write to Voxy's cache format (or adapter layer)

### 5.4 Seam Strategy

**Problem:** Patchwise refinement can diverge at boundaries without neighbor context.

**Solutions (pick one):**

1. **Halo input (recommended):** Include 1-voxel border from neighbors in parent context
2. **Seam loss (training-time):** Penalty for boundary mismatches
3. **Deterministic post-pass stitch:** Reconcile boundary voxels after generation

**Implementation:**

- Condition on 10×10×10 parent (instead of 8×8×8)
- Output center 16×16×16 only
- This lets the model "see" neighbor's parent at boundary

## 6. Gameplay Boundaries

### 6.1 Simulation Distance

**Inside simulation distance:**

- LOD0 is authoritative (real blocks for gameplay)
- No invisible collisions
- Mob spawning works
- Random ticks operate normally
- Structures may be optionally vanilla (Phase 2)

**Outside simulation distance:**

- ML-generated LOD only (render-only)
- No vanilla chunk generation
- No gameplay mechanics (no mobs, no ticks, no collisions)

### 6.2 Authoritative Terrain Policy

**Philosophy:** Vanilla-anchored but ours

- We do not aim for bit-for-bit vanilla matching
- We aim for statistical similarity + gameplay correctness
- Terrain is generated using our model, anchored to vanilla noise
- Vanilla `carve()` may run at LOD0 for final cave/aquifer polish (optional)

**No double-work:**

- We do not generate proxy terrain then replace with vanilla
- We generate our terrain once, cache it, and use it
- Vanilla structures/features may be placed on top (Phase 2)

## 7. Performance Requirements

### 7.1 Inference Targets

- **CPU latency:** ≤100ms median, ≤150ms p95 per patch (16³ volume)
- **Memory:** ≤2MB incremental per inference
- **Throughput:** Must keep up with player movement at elytra speeds
- **Consistency:** Deterministic outputs (same input = same output)

### 7.2 Benchmarking

**Required measurements:**

- Per-patch inference time (all 5 models)
- Anchor sampling time
- IO time for caches
- Memory usage per region
- Worst-case spikes (MSPT impact)

**Profiling tools:**

- Python: `benchmark_progressive.py`, `benchmark_simple.py`
- Java: In-mod metrics + logging

## 8. Milestones

### Milestone 1: Plumbing & Infrastructure

- [ ] Mod stack launches on 1.21.5
- [ ] Voxy renders synthetic LOD terrain (no ML yet)
- [ ] Freeze + pregen pipeline works
- [ ] Anchor computation + caching functional

### Milestone 2: Dataset & Training Prep

- [ ] Extract parent-child samples from vanilla worlds
- [ ] Build training manifest with provenance
- [ ] Validate block vocab mapping
- [ ] Generate `test_vectors.npz` for DJL parity

### Milestone 3: Baseline Generator

- [ ] Deterministic non-ML refinement (proof of pipeline)
- [ ] Rendered via Voxy
- [ ] Seam strategy validated

### Milestone 4: Model Training

- [ ] Train Init model (noise → LOD4)
- [ ] Train refinement models (LOD4→3, 3→2, 2→1, 1→0)
- [ ] Achieve 99% accuracy on frequent blocks (goal)
- [ ] Export all 5 models to ONNX

### Milestone 5: ONNX Integration

- [ ] In-mod inference works (DJL + ONNX Runtime)
- [ ] Model outputs visible terrain via Voxy
- [ ] DJL parity verified (test vectors match)
- [ ] Performance benchmarks meet targets

### Milestone 6: Progressive Refinement

- [ ] Multi-level LOD chain functional
- [ ] Scheduler prioritizes correctly
- [ ] Caching prevents recomputation
- [ ] Seam stability validated

### Milestone 7: Performance Validation

- [ ] Profiling complete
- [ ] Meets CPU targets (<100ms per patch)
- [ ] Stable under fast travel (elytra speeds)
- [ ] No invisible collisions
- [ ] Restart preserves cached LODs

## 9. Acceptance Criteria (Phase 1 Complete)

The system is considered **working** when:

### Functional Requirements

- [ ] LOD terrain renders seamlessly at distance
- [ ] Refinement occurs as player approaches (no visible popping)
- [ ] No visible seam artifacts at chunk boundaries
- [ ] No invisible collisions (LOD0 is authoritative)
- [ ] Inference stays under CPU budget (<100ms per patch)
- [ ] Restart preserves cached LODs (persistence works)
- [ ] Terrain statistically resembles vanilla (heightmaps, biomes, caves)

### Quality Requirements

- [ ] Anchor constraints respected (height, rivers, caves align with vanilla)
- [ ] Seam metrics pass (continuity across borders)
- [ ] Coastline continuity validated (edge/PSNR at sea level)
- [ ] Performance benchmarks meet targets (latency, memory)

### Integration Requirements

- [ ] Voxy integration functional (sections render correctly)
- [ ] DJL parity verified (test vectors match PyTorch)
- [ ] ONNX export valid (static shapes, no unsupported ops)
- [ ] Cache invalidation works (seed/mod/model changes)

## 10. Out of Scope (Phase 1)

- Structure generation (Phase 2)
- Vegetation modeling (Phase 2)
- Nether/End support (future)
- Full vanilla parity (we aim for statistical similarity)
- Text-prompt terrain (not part of vision)
- GPU inference (CPU-only for mod compatibility)
- Interactive evolution (not part of architecture)

## 11. Long-Term Vision (Phase 2+)

- Structure-aware generation (two-codebook VQ-style latent system)
- Vegetation pass (trees, plants, decorations)
- Structure blending (villages, strongholds, etc.)
- Multi-biome transition smoothing
- Quantized inference (INT8 for speed)
- Adaptive patch size (larger patches for flat terrain)

## 12. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Seam instability | Visual artifacts, gameplay issues | Halo context early, seam loss in training, deterministic stitch |
| Anchor misalignment | Model fights constraints, poor quality | Strict anchor-consistency losses, validation tests |
| Model underfitting rare biomes | Poor quality in edge cases | Stratified sampling, rare-feature oversampling |
| Scheduler starvation at high speed | Visible popping, stuttering | Priority queue with movement cone, graceful fallback |
| Version drift in Fabric ecosystem | Breaking changes, incompatibility | Pin versions early, test matrix |
| Voxy format assumptions | Integration failures | Early adapter layer, format validation |
| Performance targets not met | Unusable in practice | Profiling from day 1, model size caps, quantization path |

## 13. Project Structure

```
VoxelTree/
├── train/              # Model definitions, training loop
├── scripts/             # Data processing, ONNX export, benchmarking
├── tests/               # Unit tests (PyTest)
├── models/              # Saved checkpoints + ONNX exports
├── data/                # Training data, intermediate files
├── docs/                # Architecture, project outline, reflections
├── Hephaistos/          # Minecraft world parsing library
├── schema/              # Model config JSON schemas
├── config.yaml          # Training configuration
└── requirements.txt     # Python dependencies
```

## 14. Next Steps (Immediate)

1. **Lock dependency matrix:** Pin MC 1.21.5 + Fabric + Voxy + Carpet versions
2. **Implement NoiseTap interface:** One-call capture per chunk for anchors
3. **Implement FeatureBundle cache:** LRU + optional disk sidecar
4. **Draft 5 model_config.json stubs:** Define exact I/O shapes for all models
5. **Build dataset respec:** Read native caches, emit shared inputs, generate LOD targets
6. **Start Init model:** Train noise → LOD4, verify end-to-end through DJL

---

**This outline defines the complete scope, architecture, and success criteria for LODiffusion Phase 1. All implementation should align with these principles and constraints.**
