# LODiffusion — Comprehensive Project Outline

## 1. Vision & Philosophy

### 1.1 Core Idea

LODiffusion generates **distant LOD terrain only**, using a progressive, ML-driven, reverse-order refinement system. Vanilla terrain generation remains authoritative for playable-resolution terrain (LOD0). The system:

- Generates coarse LOD terrain first (far from player)
- Refines terrain progressively as the player approaches (LOD4 → LOD1 only)
- Anchors all macro-structure in vanilla noise functions
- Uses lightweight, **per-step ONNX models** for fast CPU inference
- Integrates with Voxy for multi-resolution rendering (insert-only; never overwrites Voxy data)
- Vanilla takes over at LOD0 — no model-generated playable terrain

> **Key architecture decision:** The model is a *render proxy* for distant terrain,
> not a terrain generator replacement. Players get exact vanilla terrain where they
> can interact with it, and believable approximations where they can only see it.

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

**What vanilla noise provides (cheap anchor channels only):**

- Macro height structure (continentalness, erosion, peaks/valleys)
- Biome layout (climate parameters)
- Surface height planes (surface, ocean floor, slopes, curvature)
- Router6 climate parameters (temperature, vegetation, continents, erosion, depth, ridges)
- Deterministic continuity (seed-stable, seamless)

> **Dropped from scope:** Cave carver noise, aquifer masks, barrier masks, and other
> expensive 3D noise functions are NOT computed at runtime. The model generates
> only the *visible outer shell* of distant terrain. Underground detail that
> players cannot see is not worth the compute cost.

**What the model learns:**

- Multi-resolution coherence (coarse LOD approximations that look believable from distance)
- Surface terrain envelope (skyline, mountain shapes, valley contours)
- Material distribution (biome-appropriate block types at coarse resolution)
- Parent→child structural consistency (how coarse voxels expand into finer voxels)
- Hierarchical refinement (1³ → 2³, 2³ → 4³, 4³ → 8³ only — NOT 8³ → 16³)

**What the model does NOT do:**

- Generate LOD0 / playable-resolution terrain (vanilla handles this)
- Replace vanilla terrain generation at any resolution players interact with
- Generate sealed underground volumes that are never visible
- Overwrite existing Voxy data (insert-only to RocksDB)
- Move mountains horizontally or ignore biome signals
- Compute expensive 3D noise fields (carvers, aquifers) at runtime

### 1.4 Success Criteria

The system is successful when:

- Player can stand still → distant terrain fills in all directions seamlessly
- Player can sprint → forward cone prioritized without stuttering
- Player can elytra at high speed → distant terrain streams without popping
- Player increases render distance → no catastrophic performance drop
- Player restarts world → distant terrain instantly visible from cache
- Distant terrain looks believable → smooth transition to vanilla as player approaches
- Vanilla terrain takes over seamlessly at LOD0 → no visual betrayal at the handoff

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

The system uses a **4-model family** of separate, per-step ONNX models for progressive refinement:

| Model | ONNX File | Input `x_parent` | Output (`block_logits`, `air_mask`) | Relative Size |
|-------|-----------|-------------------|-------------------------------------|---------------|
| **Init** | `init_to_lod4.onnx` | `[1,1,1,1,1]` (zeros) | `[1,N,1,1,1]`, `[1,1,1,1,1]` | Tiny (MLP) |
| **LOD4→3** | `refine_lod4_to_lod3.onnx` | `[1,1,1,1,1]` | `[1,N,2,2,2]`, `[1,1,2,2,2]` | Small |
| **LOD3→2** | `refine_lod3_to_lod2.onnx` | `[1,1,2,2,2]` | `[1,N,4,4,4]`, `[1,1,4,4,4]` | Medium |
| **LOD2→1** | `refine_lod2_to_lod1.onnx` | `[1,1,4,4,4]` | `[1,N,8,8,8]`, `[1,1,8,8,8]` | Medium-Large |

> **LOD1→0 dropped:** Vanilla terrain generation handles full-resolution (LOD0)
> terrain. The model stops at LOD1 (8³). This eliminates the hardest and most
> expensive refinement step and avoids the "terrain parity" problem entirely.

**Key principles:**

- **Separate ONNX models per step** — each has fixed tensor shapes, optimal for CPU cache and ONNX Runtime graph optimization
- **No LOD0 model** — vanilla terrain generation is authoritative for playable resolution
- LOD1+ = progressively coarser ML-generated representations (render proxy only)
- All LOD levels are deterministic, worldspace-consistent, and seam-aware
- No upsampling in the mod; models contain static Resize/conv internally
- Coarse models are smaller/faster; capacity scales with refinement difficulty
- Each model loaded once at startup, session kept alive (no per-call overhead)

### 2.3 Anchor Channels (Shared Inputs)

All models share these deterministic signals derived from **cheap** vanilla noise:

| Channel | Shape | Description | Status |
|---------|-------|-------------|--------|
| `x_parent` | varies per step | Parent voxels from previous LOD | **Active** |
| `x_height_planes` | `[1,5,16,16]` | Surface, ocean_floor, slope_x, slope_z, curvature | **Active** |
| `x_router6` | `[1,6,16,16]` | Temperature, vegetation, continents, erosion, depth, ridges | **Active** |
| `x_biome` | `[1,16,16]` int64 | Biome index per column | **Active** |
| `x_y_index` | `[1]` int64 | Y-slab position (0–23) | **Active** |

> **Dropped channels** (too expensive for distant LOD or no longer needed):
> - ~~`x_cave_prior4`~~ — Requires 3D noise evaluation; dropped with underground skipping
> - ~~`x_aquifer3`~~ — Expensive to compute; minimal visual impact at distance
> - ~~`x_barrier`~~ — Coastal barrier mask; minimal impact at coarse LODs
> - ~~`x_biome_quart`~~ — Replaced by simpler `x_biome` column-wise index
> - ~~`x_chunk_pos`~~ — Removed; global coherence comes from anchor channels
> - ~~`x_lod`~~ — No longer needed; each model handles exactly one LOD transition

**Normalization:**

- Heights: min-max by world limits (-64 to 320)
- Router6: z-score
- Biome: integer index → learned embedding
- Y-index: integer → learned embedding

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

**Artifacts (per LOD step, 4 models total):**

- `init_to_lod4.onnx` — Noise → LOD4 (tiny MLP)
- `refine_lod4_to_lod3.onnx` — LOD4 → LOD3 (small)
- `refine_lod3_to_lod2.onnx` — LOD3 → LOD2 (medium)
- `refine_lod2_to_lod1.onnx` — LOD2 → LOD1 (medium-large)
- `model_config.json` (metadata: input/output names, shapes per model, normalization, block palette)
- `test_vectors.npz` (input/output examples per model for DJL parity validation)

> **No LOD1→0 model.** Vanilla terrain generation handles LOD0.

**Export requirements:**

- Static shapes only (each model has fixed I/O shapes)
- Opset 17+ (ONNX)
- No unsupported ops
- Deterministic inference
- DJL compatibility verified
- Separate ONNX session per model (loaded once at startup)

## 5. Runtime Integration

### 5.1 Job Scheduler

**Priority logic:**

1. Movement cone (narrow at speed, prioritize forward direction)
2. Idle fill-in (expand to 360° when stationary)
3. Background refinement (coarser LODs for distant terrain)

> **Spawn sphere removed:** Vanilla handles LOD0/gameplay terrain. Our scheduler
> only manages distant LOD generation (LOD1–LOD4).

**Work queue design (target):**

- Deduplicated queue keyed by `(dimension, region_x, region_z, lod)`
- States: `MISSING → QUEUED → GENERATING → READY`, plus `FAILED_TEMP`, `STALE`
- Prerequisite chain resolution: request for LOD N generates missing chain coarsest-first
- Each generated level cached as scaffolding for future refinement requests
- `STALE` state for cache invalidation when model version changes

**Constraints:**

- Bound CPU time per tick (must not spike MSPT)
- Avoid blocking render thread
- Cache anchors and LODs (don't recompute)
- Graceful fallback if inference can't keep up (render coarser)
- Insert-only writes to Voxy RocksDB (never overwrite existing data)

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

**Data Authority Policy: Insert-Only**

> **Critical rule:** LODiffusion NEVER overwrites existing Voxy data in RocksDB.
> - If a section key already exists → skip (Voxy's data is authoritative)
> - Only write to keys where no data exists yet
> - Voxy always wins — it has ground-truth from vanilla terrain
> - This prevents data corruption and ensures vanilla parity where it matters
> - When vanilla terrain eventually loads, Voxy's real data naturally replaces our approximation

**Demand-Driven Generation (Target Architecture):**

> Hook into Voxy's RocksDB cache-miss path. When Voxy requests latent terrain
> that doesn't exist, enqueue it for generation at the requested LOD.
> - Pull-driven: only generate what Voxy actually wants to render
> - Natural prioritization from renderer demand
> - No wasted work on terrain nobody will see
> - Coarsest prerequisite generated first (cheapest, fastest first response)
> - Each generated level becomes cached scaffolding for future refinement
>
> **Prerequisite chain resolution:** A request for LOD N generates only the
> missing chain (coarsest-first), stopping at the requested level.
> Do not eagerly generate beyond what was requested.
>
> **Fallback behavior:** If terrain isn't ready yet, return "not available"
> rather than blocking the render path. Voxy renders fog/nothing until ready.

**Requirements:**

- Generate LOD sections compatible with Voxy's 32³ section format
- Insert-only writes (never overwrite existing Voxy data)
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

- **Vanilla terrain generation is authoritative** (LOD0 = real blocks)
- No invisible collisions
- Mob spawning works
- Random ticks operate normally
- Structures are vanilla

**Outside simulation distance:**

- ML-generated LOD terrain (LOD1–LOD4, render-only proxy)
- Visual approximation — not gameplay-authoritative
- No vanilla chunk generation needed yet
- No gameplay mechanics (no mobs, no ticks, no collisions)
- When player approaches, vanilla terrain generates and Voxy's real data replaces ours

### 6.2 Authoritative Terrain Policy

**Philosophy:** Vanilla is king; we're the distant preview

- Vanilla terrain generation runs normally for all playable terrain
- Our model generates ONLY distant LOD terrain (LOD1–LOD4) as a render proxy
- We do NOT aim for bit-for-bit vanilla matching at distance (visual plausibility is enough)
- We do NOT generate LOD0 (the model's finest output is LOD1 = 8³ resolution)
- When vanilla terrain eventually loads, it naturally replaces our approximation

**Insert-only data policy:**

- We insert terrain only where Voxy has no data yet
- We never overwrite Voxy's ground-truth data
- Vanilla terrain loading naturally supersedes our LOD approximations
- No reconciliation needed — Voxy's data always wins

### 6.3 Underground Terrain Optimization

**Philosophy:** Only generate what players can see from distance.

> Most distant terrain value is in the **surface shell**. Generating sealed
> underground volumes for distant LOD is wasted compute.

**Visibility rule (target):**

| Category | Generate? | Examples |
|----------|-----------|----------|
| **Always** | Yes | Surface shell, topography, biome-colored surfaces, coastlines |
| **Sometimes** | Conditional | Cave mouths near exposed surfaces, large overhangs, ravine walls |
| **Usually skip** | No | Sealed underground subchunks, deep carver detail, enclosed cave systems |

**Implementation approach:**

- Skip y-slabs that are fully below the surface heightmap with no exposed faces
- Focus model capacity on terrain envelope and visible surfaces
- This dramatically reduces both model complexity and runtime generation cost

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
- [ ] Seam strategy validated *(only crude tile-edge factor heuristic; no halo/XZ neighbor context. visibliy misaligned in-game)*

### Milestone 4: Model Training

- [ ] Train Init model (noise → LOD4) — **separate dedicated model (tiny MLP)**
- [ ] Train LOD4→3 refinement model — **separate ONNX, small capacity**
- [ ] Train LOD3→2 refinement model — **separate ONNX, medium capacity**
- [ ] Train LOD2→1 refinement model — **separate ONNX, medium-large capacity**
- ~~Train LOD1→0 refinement model~~ — **DROPPED: vanilla handles LOD0**
- [ ] Achieve 99% accuracy on frequent blocks (goal) *(best: ~69% overall; ~70% block accuracy - needs much improvement)*
- [ ] Export all 4 models to separate ONNX files *(pipeline exists; needs per-model export)*

> **Architecture reverted to separate models.** The unified model pivot (single model
> with `x_lod` conditioning over all transitions) was suboptimal: zero-padding different
> tensor sizes to 16³ wasted compute, prevented per-step capacity tuning, and hurt
> ONNX Runtime optimization. Each step now gets its own model with fixed tensor shapes.

### Milestone 5: ONNX Integration

- [x] In-mod inference works (DJL + ONNX Runtime) *(DJL BOM 0.30.0; confirmed in game logs)*
- [x] Model outputs visible terrain via Voxy *(LodGenerationService writing LOD sections; 200+ sections confirmed in log)*
- [ ] Migrate to per-step model loading (4 ONNX sessions, loaded once at startup)
- [ ] Implement insert-only RocksDB write guard (skip if key exists)
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

- **LOD0 generation** — Vanilla terrain is authoritative at full resolution
- **Cave carver noise / aquifer computation** — Too expensive for distant LOD
- **Underground subchunk generation** — Only generate visible terrain shell
- Structure generation (Phase 2)
- Vegetation modeling (Phase 2)
- Nether/End support (future)
- Custom biomes (We're just training for Vanilla-like terrain)
- Text-prompt terrain (not part of vision)
- GPU inference (CPU-only for mod compatibility)
- Interactive evolution (not part of architecture)

## 11. Long-Term Ideas (Phase 2+)

- Structure-aware generation (two-codebook VQ-style latent system)
- Vegetation pass (trees, plants, decorations)
- Structure blending (villages, strongholds, etc.)
- **Block-vocab reduction for Phase 2**: `blocklist.json` (repo root) contains a
  `generates_in_structures` field for every block type.  Once Phase 1 training
  covers all terrain biomes (and thus all terrain-generating blocks), a Phase 2
  vocabulary can be derived by filtering `blocklist.json` to only
  `generates_in_structures: "Yes"` entries.  This gives a compact, curated set
  of structure-specific blocks for a dedicated structure-generation model,
  without polluting the terrain vocabulary.
- Quantized inference (INT8 for speed)
- Adaptive patch size (larger patches for flat terrain)
- **Cave conditioning channel (`x_cave`)** — If cave topology at distance proves important,
  add a coarse cave likelihood mask. Stabilizes topology but requires 3D noise evaluation.
  Semi-hard constraint: if `cave_mask == 1`, strongly bias air probability.
- **Residual prediction mode** — Instead of predicting blocks from scratch,
  predict `child = upsample(parent) + neural_residual` to keep large structures stable.
- **Single-seed generalization test** — Train on one seed, evaluate on others.
  If model generalizes, it learned terrain rules, not memorization.
- **Neural terrain quality comparison** — Compare model output quality vs vanilla at same LOD.
  The model may produce *smoother* transitions than vanilla's hard thresholds.

## 12. Machine Learning Insights & Design Rationale

### Why Minecraft is Procedurally Learnable

Minecraft terrain generation is unusually learnable for ML because:

1. **Structural Regularity via Noise Functions**: Vanilla generation uses layered Perlin and simplex noise at multiple octaves. This creates self-similar fractal structure that neural networks can easily approximate through hierarchical feature learning.

2. **Biome Determinism**: Each biome is determined by climate parameters (temperature, humidity, continentalness) that form smooth manifolds in latent space. The model can learn to condition block generation on these continuous values rather than memorizing discrete patterns.

3. **Local Causality**: Block placement depends primarily on local neighborhood (within ~8 blocks vertically, ~16 blocks horizontally). Distant blocks don't affect local generation, making the problem highly parallelizable and learnable.

4. **Repetitive Patterns Across Biomes**: Slopes, water edges, and vegetation follow similar rules across different biomes, creating transferable feature representations.

This explains why LODiffusion's simple CNN refinement (not diffusion, not GANs, no latent evolution) works well: Minecraft's noise-based structure is more similar to traditional upsampling tasks than to free-form generation.

### Block Vocabulary Expansion (Phase 2 Strategy)

Phase 1 trains on terrain blocks only (139 unique blocks across 28 strategic biomes). For Phase 2 (structure generation), the vocabulary must expand to include structure-specific blocks:

**Phase 1→2 Expansion Path:**
- **Current vocab:** 139 terrain blocks from 28 biomes (covers all natural terrain generation)
- **Phase 2 target:** ~500+ blocks including structures (villages, strongholds, temples, mineshafts, ships)
- **Biome selection strategy:** Focus on biomes with feature-rich structures:
  - `lush_caves` — Vines, glow berries, moss expansion blocks
  - `badlands` — Terracotta color variety, copper/gold ore variants
  - `warm_ocean` — Coral reefs, sea pickle diversity
  - `deep_dark` — Sculk blocks, darkness indicator blocks
  - Remaining coverage: Nether/Fortress blocks (Phase 3+)

**Vocabulary derivation:** `blocklist.json` (repo root) already contains a `generates_in_structures` field for every Minecraft block. Once terrain training is complete, Phase 2 vocabulary is derived by filtering to `generates_in_structures: "Yes"` entries, ensuring a curated, non-redundant structure-specific palette.

### Cave Topology Sensitivity & Percolation Risks

While Phase 1 skips underground detail (only visible outer shell), Phase 2? cave generation carries a critical ML risk:

1. **Percolation Threshold Brittleness**: Cave systems depend on topological connectivity. Small changes to block placement near percolation thresholds can disconnect entire cave systems. A model trained on one seed's cave configuration may fail when cave parameters shift.

2. **Connectivity Invariance is Hard**: Unlike surface terrain (smooth height gradients), caves must satisfy global connectivity constraints. A local CNN cannot guarantee the model's cave structure remains connected after refinement.

3. **Mitigation Strategy**: _If_ Phase 2 includes cave conditioning, use a **conditioning mask** (coarse/binary "cave likely here" heatmap) as an input channel rather than training the model to generate caves from scratch. This preserves global structure while allowing fine detail refinement.

2. **Cross-Seed Validation Strategy**: Validate generalization by training the model on Seed A but testing on Seed B. If the model trained on one seed generalizes well to another seed, it has learned transferable terrain rules, not seed-specific memorization.

3. **Expected Behavior**: Grokking should occur when the model's capacity (parameter count) matches the "rule complexity" of terrain generation. Too small a model underfits; too large a model memorizes. The sweet spot is where generalization emerges naturally.

4. **Testing Plan**: Reserve a held-out test seed with identical biome distribution but different noise instances. Measure per-biome accuracy across test seed to confirm rule-learning.

## 13. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Seam instability | Visual artifacts | Halo context early, seam loss in training, deterministic stitch |
| Anchor misalignment | Model fights constraints, poor quality | Strict anchor-consistency losses, validation tests |
| Model underfitting rare biomes | Poor quality in edge cases | Stratified sampling, rare-feature oversampling |
| Scheduler starvation at high speed | Visible popping, stuttering | Priority queue with movement cone, graceful fallback |
| Version drift in Fabric ecosystem | Breaking changes, incompatibility | Pin versions early, test matrix |
| Voxy format assumptions | Integration failures | Early adapter layer, format validation |
| Performance targets not met | Unusable in practice | Profiling from day 1, model size caps, quantization path |
| **LOD→vanilla transition jarring** | Players notice handoff, lose trust | Smooth transition logic, blending at boundary, LOD1 quality must be high |
| **Locally correct, globally wrong** | Patch boundaries visible, caves dead-end | Conditioning channels anchor global structure, halo context, seam loss |
| **Insert-only policy violated** | Voxy data corruption | Code guards on RocksDB writes, key-exists check before every insert |

## 14. Project Structure

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

## 15. Next Steps (Immediate)

1. **Wire up separate model training:** Update `train_multi_lod.py` to train progressive per-step models from `train/progressive_lod_models.py`
2. **Drop LOD1→0 from pipeline:** Remove LOD0 training targets, update LOD sampling weights
3. **Export 4 separate ONNX models:** Update `scripts/export_lod.py` for per-step export
4. **Implement insert-only RocksDB guard:** In `VoxySectionWriter`, check key existence before writing
5. **Deploy to LODiffusion:** Copy 4 ONNX files + model_config.json to mod config
6. **Validate in-game:** Verify distant terrain renders correctly via Voxy

## 16. Experimental Ideas (Backlog)

These are not immediate priorities but worth tracking:

| Idea | Effort | Expected Value | Notes |
|------|--------|---------------|-------|
| Single-seed generalization test | Low | Medium | Cheap validation of rule-learning vs memorization |
| Residual prediction mode | Medium | Medium | `child = upsample(parent) + residual` — may stabilize boundaries |
| Cave conditioning channel | Medium | Medium | Useful if cave quality at LOD1 matters |
| Demand-driven Voxy hook | High | Very High | Pull-driven generation from Voxy cache-miss path |
| Underground y-slab skipping | Medium | Very High | Only generate y-slabs with exposed faces |
| Capacity profiling per model | Low | Medium | Right-size each model after initial training |
| LOD→vanilla transition blending | High | Critical | The make-or-break UX seam — highest priority after models work |

---

**This outline defines the complete scope, architecture, and success criteria for LODiffusion Phase 1. All implementation should align with these principles and constraints.**
