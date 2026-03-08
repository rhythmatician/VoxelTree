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
| **LOD Rendering** | Voxy (Rust/Java) | Multi-resolution rendering, LOD storage (RocksDB) |
| **Training Data** | Voxy RocksDB | Ground-truth block data extraction (canonical source) |
| **World Freeze** | Carpet Mod | Deterministic training world generation |

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

**Block vocabulary:**

- **Voxy-native canonical vocabulary**: 1102 entries from `config/voxy_vocab.json`
- Auto-built by scanning all Voxy worlds; air = ID 0, remainder alphabetically sorted
- Property variants (e.g. `oak_stairs[facing=north]`) collapse to a single canonical ID
- Per-world Voxy state IDs are mapped to canonical IDs by block name at extraction time
- LODiffusion's `VoxyBlockMapper` reads `block_mapping` from `model_config.json` at runtime

## 3. Dependencies & Environment

### 3.1 Pinned Dependency Matrix (Minecraft 1.21.11)

**Java Toolchain:**

- Java 21 (required for modern MC + Fabric ecosystem)

**Mod Stack:**

- Minecraft: 1.21.11
- Fabric Loader: 0.18.4
- Fabric API: 0.141.3+1.21.11
- Voxy: 0.2.11-alpha
- Carpet Mod: 1.4.194+v251223
- Chunky: 1.4.55
- LODiffusion mod: (ours)

**ML Runtime Stack:**

- ONNX: 1.12+ (static shapes)
- DJL: (pinned version TBD)
- ONNX Runtime: CPU backend only

**Python Training Stack:**

- Python: 3.13+
- PyTorch: >=2.0
- NumPy, SciPy, PyYAML, tqdm
- rocksdict (Voxy RocksDB reading)
- zstandard (ZSTD decompression for Voxy data)
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

**Source:**

- Voxy RocksDB databases (LOD-0 WorldSections)
- Each Voxy world stores block data at `<saves>/<world>/voxy/<hash>/storage/`
- Block IDs are per-world sequential; mapped to canonical vocabulary by name

**Extraction tool:** `scripts/extract_voxy_training_data.py`

**Process:**

1. Open Voxy RocksDB via `scripts/voxy_reader.py` (SaveLoadSystem3 decoder)
2. Iterate LOD-0 sections, decode 32³ WorldSections into 16³ sub-blocks
3. Map per-world Voxy state IDs → canonical vocabulary IDs via `build_world_lut()`
4. Compute biome grid (16×16 column-wise majority) and heightmap (normalised max non-air Y)
5. Filter: skip sections below `--min-solid` threshold
6. Save as NPZ with keys: `labels16`, `biome_patch`, `heightmap_patch`, `y_index`

**Outputs:**

- NPZ files (one per 16³ sub-block): `data/voxy/<world>_<x>_<y>_<z>_<sub>.npz`
- Canonical vocabulary: `config/voxy_vocab.json` (auto-built if missing)

**Provenance tracking:**

- Voxy world path
- Vocabulary version (1102 entries)
- Code commit hash
- Extraction parameters (min-solid, max-sections)

## 4. Training Pipeline

### 4.1 Dataset Construction

**Process:**

1. Extract LOD-0 data from Voxy RocksDB databases (`scripts/extract_voxy_training_data.py`)
2. Build LOD pyramid targets using Voxy Mipper algorithm (`scripts/mipper.py`)
3. Randomly sample coarsening factor per training example
4. Compute anchor conditioning (height planes, router6) from extracted/cached data
5. Nearest-upsample coarsened parent to canonical 8³ for model input

**LOD coarsening:** Uses the **Voxy Mipper** (opacity-biased corner selection), not OR-pool
or majority vote. Source of truth: `scripts/mipper.py`.

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

> **Authoritative reference:** `docs/VOXY-FORMAT.md` (audited from `MCRcortex/voxy` source).
> All details below are ground-truth, not assumptions.

**Requirements:**

- Generate LOD sections compatible with Voxy's 32³ section format
- Cache multi-level sections
- Ensure seam stability
- Support reload persistence

**Voxy section format (verified):**

- `32×32×32` voxels per WorldSection, indexed as `(y<<10)|(z<<5)|x`
- **64-bit key** encoding: `(lvl<<60)|(y8<<52)|(z24<<28)|(x24<<4)` — 4 spare low bits
- Storage: **RocksDB + ZSTD** (default); LMDB/Redis/in-memory backends also present
- On-disk serialization per section:
  1. `key` (8 B) + `metadata` (8 B, low byte = `nonEmptyChildren` bitmask)
  2. `lutLen` (4 B) — number of unique voxel values
  3. LUT: `lutLen × 8 B` — each entry is the **full 64-bit voxel long** (block+biome+light)
  4. Indices: `32³ × 2 B = 65,536 B` — 16-bit LUT indices in **Morton (z-curve) order**
  5. `hash` (8 B)
- **Per-voxel `long` encoding:** bits 56–63 = light (sky<<4|block), bits 47–55 = biome ID
  (9-bit, Voxy-internal), bits 27–46 = block state ID (20-bit, **Voxy-internal mapped ID**,
  NOT the MC registry ID), bits 0–26 unused
- **Block ID mapping** is world-specific: persisted as a separate entry in the same RocksDB
  store; must be read before decoding any voxel data
- **LOD downsampling** (Mipper): opacity-biased corner selection, **not majority vote**.
  Each 2×2×2 group → pick the most-opaque non-air voxel (tie-break by I111 corner priority).
  If all air, average light values.

**LOD levels vs. vanilla chunk sections:**

- Voxy LOD 0: each 32³ WorldSection covers `32` world-blocks per axis (= two 16-block
  vanilla sections per axis, so 8 vanilla chunk sections compose one WorldSection)
- Voxy LOD n: each 32³ WorldSection covers `32 × 2ⁿ` world-blocks per axis
- During ingestion, `VoxelizedSection` holds a 5-level pyramid (16³+8³+4³+2³+1) for one
  vanilla chunk section, before assembly into the spanning WorldSection

**Integration approach:**

- To feed our model outputs into Voxy: pack each 16³ predicted patch into the correct
  octant of a LOD-0 WorldSection (8 patches share one section key), encode each voxel
  long using `(light<<56)|(biome<<47)|(voxy_block_id<<27)`, build LUT,
  Morton-sort indices, write with correct key
- The Voxy block ID mapping must be exported alongside the model's `model_config.json`
  (or translated via a shared `block_vocab.json` ↔ Voxy-ID table at inference time)
- See `docs/VOXY-FORMAT.md` §9 for a Python decoding/encoding recipe

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

- We do not generate proxy terrain then replace with vanilla (Late phase 1)
- We generate our terrain, latent Voxy representations, and training pairs once, cache them, and use them.  Only delete and regenerate with a new world seed for long runs, once everything is built and validated completely (At the very end of phase 1).
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

- [x] Mod stack launches on 1.21.11
- [x] Voxy renders synthetic LOD terrain (no ML yet) *(exceeded: ONNX model now active)*
- [x] Freeze + pregen pipeline works *(VoxelTree/data-cli.py: RCON orchestrator with Carpet freeze + Chunky pregen; requires server with RCON enabled)*
- [x] Anchor computation + caching functional

### Milestone 2: Dataset & Training Prep

- [x] Extract parent-child samples from vanilla worlds *(PatchPairer + data/chunks/ populated)*
- [x] Build training manifest with provenance *(dataset_respec.py + trainer.py provenance; training ran 8 epochs)*
- [x] Validate block vocab mapping *(standard_minecraft_blocks.json live in-game via VoxyBlockMapper)*
- [ ] Generate `test_vectors.npz` for DJL parity *(no .npz test vectors exist)*

### Milestone 3: Baseline Generator

- [x] Deterministic non-ML refinement (proof of pipeline) *(VanillaLikeTerrainGenerator + LodGenerationService sine/cosine heightmap)*
- [x] Rendered via Voxy *(VoxySectionWriter actively pushing sections; confirmed in game logs)*
- [ ] Seam strategy validated *(only crude tile-edge factor heuristic; no halo/XZ neighbor context)*

### Milestone 4: Model Training

- [ ] Train Init model (noise → LOD4) *(architecture pivoted to single unified model; no dedicated init model)*
- [ ] Train refinement models (LOD4→3, 3→2, 2→1, 1→0) *(training stalled at 8/20 epochs; transitions don't cascade as specified)*
- [ ] Achieve 99% accuracy on frequent blocks (goal) *(best: ~69% overall; ~70% block accuracy)*
- [ ] Export all 5 models to ONNX *(1 undertrained unified model exported; pipeline works)*

### Milestone 5: ONNX Integration

- [x] In-mod inference works (DJL + ONNX Runtime) *(DJL BOM 0.30.0, UnifiedModelRunner, confirmed in game logs)*
- [x] Model outputs visible terrain via Voxy *(LodGenerationService writing LOD4→1 sections; 200+ sections confirmed in log)*
- [ ] DJL parity verified (test vectors match) *(no test_vectors.npz generated; no Java parity test)*
- [ ] Performance benchmarks meet targets *(cold-start 359ms >> 100ms target; warm ~60ms; framework not run vs real model)*

### Milestone 6: Progressive Refinement

- [x] Multi-level LOD chain functional *(ProgressiveLODPipeline chains 5 stages; LodGenerationService runs 4 LOD passes confirmed in log)*
- [x] Scheduler prioritizes correctly *(buildSpiralSections() + PASS_RADIUS; closest sections generated first)*
- [x] Caching prevents recomputation *(parentCache HashMap in LodGenerationService; coarsened outputs reused across passes)*
- [ ] Seam stability validated *(no XZ neighbor context; no seam-specific tests)*

### Milestone 7: Performance Validation

- [ ] Profiling complete *(PerformanceMonitor + TerrainGenerationBenchmark exist but use simulateWork(), not real ONNX inference)*
- [ ] Meets CPU targets (<100ms per patch) *(cold-start 359ms; undertrained model only)*
- [ ] Stable under fast travel (elytra speeds)
- [ ] No invisible collisions
- [ ] Restart preserves cached LODs *(in-memory parentCache cleared on restart; rendered Voxy sections persist via RocksDB)*

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
- Vegetation modeling? (Phase 2? Maybe should be phase 1 though!)
- Nether/End support (future)
- Custom biomes (We're just training for Vanilla-like terrain)
- Text-prompt terrain (not part of vision)
- GPU inference (CPU-only for mod compatibility)
- Interactive evolution (not part of architecture)

## 11. Long-Term Ideas (Phase 2+)

- Structure-aware generation (two-codebook VQ-style latent system)
- Vegetation pass (trees, plants, decorations)
- Structure blending (villages, strongholds, etc.)
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
├── pipeline.py              # Two-phase orchestrator: extract → train → export → deploy
├── train_multi_lod.py       # Multi-LOD training CLI with Voxy vocab
├── config_multi_lod.yaml    # Model + training configuration
├── config/
│   └── voxy_vocab.json      # Canonical Voxy-native block vocabulary (1102 entries)
├── train/                   # Model architecture, dataset, losses, metrics
│   ├── unet3d.py            # SimpleFlexibleUNet3D (8→16 super-resolution)
│   ├── multi_lod_dataset.py # NPZ dataset with multi-LOD sampling
│   ├── anchor_conditioning.py # Height planes + router6 conditioning
│   ├── losses.py            # CE + air losses
│   └── metrics.py           # Per-step and rollout metrics
├── scripts/                 # Extraction, export, benchmarking
│   ├── extract_voxy_training_data.py  # Voxy RocksDB → NPZ patches
│   ├── voxy_reader.py       # RocksDB reader (SaveLoadSystem3 decoder)
│   ├── mipper.py            # Voxy Mipper (canonical LOD coarsening)
│   ├── export_lod.py        # Static ONNX export (opset ≥ 17)
│   ├── verify_onnx.py       # ONNX + test vector verification
│   └── extraction/          # Legacy MCA extraction (kept for reference)
├── tests/                   # Unit tests (PyTest)
├── models/                  # Saved checkpoints + ONNX exports (git-ignored)
├── data/                    # Extracted training NPZs (git-ignored)
├── docs/                    # Architecture, AC, reflections
└── requirements.txt         # Python dependencies
```

## 14. Next Steps (Immediate)

1. **Extract training data:** Run `pipeline.py extract` on all Voxy worlds
2. **Train model:** Run `pipeline.py train` with Voxy-native vocabulary (1102 blocks)
3. **Export ONNX:** Run `pipeline.py export` with static shapes
4. **Deploy to LODiffusion:** Copy ONNX + model_config.json to mod config
5. **Validate in-game:** Verify terrain renders correctly via Voxy

---

**This outline defines the complete scope, architecture, and success criteria for LODiffusion Phase 1. All implementation should align with these principles and constraints.**
