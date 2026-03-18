## Plan: VoxelTree Full Staged Pipeline Migration

**TL;DR:** Migrate the VoxelTree training pipeline from the legacy 13-channel/4×2×4 format to 15 RouterField channels at 4×4×4 quart resolution, and build four models for the full staged pipeline: density predictor, biome classifier, heightmap predictor, and block selector (sparse octree). This is a clean break — no backward compat. The data-harvester Fabric mod gets a new `/dumpnoise v7` command, the Python training code gets a new `router_field.py` canonical definition, and each model gets its own training script, ONNX export, and DAG step.

---

## Progress Summary

### LODiffusion (Java runtime) — implemented on `massive-overhaul` branch

These items are **not in this plan** but were prerequisites completed in prior sessions:

- [x] **Core interfaces**: `HeightmapData` record, `HeightmapProvider` interface, `BiomeProvider` interface, `UpstreamNoiseContext` bundle — all in `world.noise` package
- [x] **Shadow validation infrastructure**: `ParityConfig` (per-field thresholds, sampling strategies), `ParityReporter` (rolling-window stats, auto-flush), `ShadowValidatingSampler` (wraps reference + candidate, drives parity comparison)
- [x] **NoiseRouterSamplerFactory** updated: 4-arg `createSampler()`, `resolveBackendKey()` for `"vanilla"|"shadow"|"gpu"|"auto"`, wires `ParityConfig` from JSON config
- [x] **Batch inference**: `SparseOctreeModelRunner.runBatchInference(float[][] noiseBatch)` — stacks N noise arrays into `[N, 15, 4, 4, 4]`, splits output per-sample. Fallback to sequential if dynamic batch unsupported.
- [x] **BiomeProvider wired end-to-end**: `VanillaBiomeProvider` → `LodGenerationService.runSparseOctreePipeline()` calls `BiomeProvider.classifyBiomes()` when model `acceptsBiomeIds()`, passes result to `runInferenceWithBiome()`. `SparseOctreeModelRunner.buildBiomeTensor()` is now config-driven (reads `biomeIdsShape` from sidecar, defaults to `4×4×4` for v7).
- [x] **VanillaHeightmapProvider**: extracted from `WorldNoiseAccess.sampleBothHeightmaps()`, implements `HeightmapProvider`
- [x] **VanillaBiomeProvider**: uses `MultiNoiseBiomeSource` + climate fields from `SectionNoiseData`
- [x] **Legacy isolation**: `NoiseTapImpl` + `NoiseTap` moved to `world.noise.tools` sub-package. Legacy 13-channel `sampleNoise3DForSection()` deprecated.
- [x] **Config wiring**: `lodiffusion.defaults.json` has `terrainBackend`, `shadow.samplingRate/strategy/logLevel/aggregationWindow` fields. `Config.parityConfig()` feeds `NoiseRouterSamplerFactory`.
- [x] **Unit tests** (575 total, 0 failures):
  - `ParityConfigTest` (23 tests): `defaults()`, `thresholdFor()` all 5 field groups, `LogLevel` enum
  - `ParityReporterTest` (20 tests): `shouldSample()`, `compare()` identical/divergent/sign-flip, auto-flush, threshold violations
  - `ShadowValidatingSamplerTest` (11 tests): both samplers always called, reference returned, rate gating, `backendName()`, `close()` lifecycle
  - `ShadowModeIntegrationTest` (20 tests, **uncommitted**): full pipeline with stub samplers, `UpstreamNoiseContext` lifecycle, `resolveBackendKey()` resolution
- [x] **Bug fix**: `FeatureCache.java` + `FeatureCacheTest.java` imported `NoiseTap` from wrong package after tools isolation — fixed

**Not yet implemented (LODiffusion GPU pipeline):**
- [ ] **Quart-resolution GPU shader** (`quart_noise_compute.comp`): new shader with `layout(local_size_x=4, local_size_y=4, local_size_z=4)`, 64 threads per section, evaluates all 15 RouterField values at quart centres, outputs `float[N×960]` SSBO. Much simpler than existing block-res `terrain_compute.comp`.
- [ ] **ShadowRouterExtractor expansion**: needs to extract 7 additional NormalNoise indices (barrier, flood, spread, lava, vein_toggle, vein_ridged, vein_gap) into `RouterConfig` UBO. The GLSL `mc_normal_noise()` function already exists — just needs wiring.
- [ ] **RouterConfig UBO expansion**: current UBO is 112 bytes (std140). Needs additional fields for the 7 new NormalNoise indices. Draft extension exists at `docs/ROUTERCONFIG_EXTENSION.glsl` (proposes 256-byte layout with 13+ cave noise indices).
- [ ] **Async GPU dispatch queue**: `SectionRequestQueue` + `CompletableFuture`-based bridge between gen thread (`LODiffusion-Gen` daemon) and render thread (OpenGL context). Currently no such mechanism exists — `TerrainComputeDispatcher.dispatch()` is synchronous (dispatch → `glMemoryBarrier` → data ready). The existing `ShadowRouterJobQueue` + `VoxyShadowBridgeMixin` handles Voxy demand-driven requests but returns GPU buffer handles, not `SectionNoiseData`.
- [ ] **Double-buffered output SSBOs**: ping/pong SSBO pair to eliminate read-after-write stalls. Currently `ShaderSSBOManager` allocates single density (binding 7, 384KB) and block (binding 11, 384KB) output buffers.
- [ ] **`GpuNoiseRouterSampler` real implementation**: currently a stub that delegates 100% to `VanillaNoiseRouterSampler`. Blocked on quart shader + async dispatch.
- [ ] **`GpuHeightmapProvider`**: derive heightmaps from density zero-crossings via dedicated GPU pass. Plan calls for a separate shader, not embedded in the main terrain shader.
- [ ] **`GpuBiomeProvider`**: `BiomePaletteSSBO` already computes biome IDs on GPU (binding 13, quart resolution at 4×4×96 per chunk).  Needs to be wrapped into the `BiomeProvider` interface and adapted from chunk-column to per-section output.

---

## VoxelTree Steps

### Phase A — Data Infrastructure

1. - [x] **Create `router_field.py`** in `voxel_tree/utils/`. Created with `COUNT = 15`, all 15 field names matching Java `RouterField` ordinals, group helpers (`CLIMATE_FIELDS`, `DENSITY_FIELDS`, `AQUIFER_FIELDS`, `ORE_FIELDS`), and an `IntEnum` for programmatic access.

2. - [x] **Update data-harvester `WorldNoiseAccess.java`**: Added `sampleRouterFieldsForSection(sectionX, sectionY, sectionZ)` returning flat `float[960]` (15 fields × 4×4×4 quart cells). Uses `DensityFunction.SinglePointContext` at quart centres `(baseX + qx*4 + 2, ...)`. Also added a `sampleRouterFieldsForSectionStructured()` variant returning `float[15][4][4][4]`.

3. - [x] **Update data-harvester biome sampling**: Added `sampleBiomeIdsForSectionV7(sx, sy, sz)` returning `int[4][4][4]` at quart resolution. Uses `BiomeMapping.forBiome()` for canonical palette IDs.

4. - [x] **New `/dumpnoise v7 [radius]` sub-command**: Registered in `NoiseDumperCommand.register()`. `executeV7()` iterates chunk columns × 24 Y-sections (−64 to +320), calls `dumpSectionNoiseV7()` which writes JSON with `router_fields`, `biome_ids`, `heightmap_surface`, `heightmap_ocean_floor`, and `field_names` metadata array. Output to `run/v7_dumps/section_{cx}_{sy}_{cz}.json`.

5. - [x] **Update `build_sparse_octree_pairs.py`**: `N_FIELDS` → 15, spatial `(4,4,4)`, reads from `v7_dumps/` directory. Output NPZ contains `noise_3d: (N, 15, 4, 4, 4)`, `biome_ids: (N, 4, 4, 4)`, `heightmap_surface: (N, 16, 16)`, `heightmap_ocean_floor: (N, 16, 16)`. Dropped `noise_2d`.

### Phase B — Density Predictor

6. - [x] **`tasks/density_v2/train_density_v2.py`**: Created. `DensityV2MLP` class: `6 → 128 → 128 → 2` with ReLU. Trains on per-quart samples (6 climate inputs → FINAL_DENSITY + PRELIMINARY_SURFACE_LEVEL). MSE loss, AdamW optimizer, 80/20 train/val split. CLI entry point `main(argv)`. Includes basic ONNX export on completion.

7. - [x] **`tasks/density_v2/export_density_v2.py`**: Standalone ONNX exporter. Input `float32[batch, 6]` → output `float32[batch, 2]`. Writes sidecar JSON with contract `lodiffusion.v7.density_v2`, channel names, opset version. Also exports flat `.bin` weights for GPU SSBO upload.

### Phase C — Biome Classifier

8. - [x] **`tasks/biome/train_biome_classifier.py`**: Created. `BiomeClassifierMLP`: `6 → 64 → 64 → 54` with ReLU + CrossEntropyLoss. 54 classes from `biome_mapping.py`. Per-quart training samples. Reports top-1 and top-3 accuracy. CLI entry point.

9. - [x] **`tasks/biome/export_biome.py`**: ONNX export with contract `lodiffusion.v7.biome_classifier`. Input `float32[batch, 6]` → output logits `float32[batch, 54]`. Sidecar JSON includes biome palette names.

### Phase D — Heightmap Predictor

10. - [x] **`tasks/heightmap/train_heightmap.py`**: Created. Simplified from the original plan — uses 6 climate fields per quart column → 2 heights (`worldSurface`, `oceanFloor`). Architecture: `6 → 128 → 64 → 32` (predicts a 4×4 grid of 2 heights = 32 outputs). The heightmap data comes from the v7 dump NPZ `heightmap_surface`/`heightmap_ocean_floor` arrays. MSE loss.

11. - [x] **`tasks/heightmap/export_heightmap.py`**: ONNX export with contract `lodiffusion.v7.heightmap_predictor`. Sidecar JSON documents input/output shapes.

### Phase E — Block Selector (Sparse Octree Update)

12. - [x] **Update `sparse_octree.py` `_NoiseEncoder`**: Added `spatial_y` parameter (default `4` for v7, `2` for legacy). `flat_3d = n3d * 4 * spatial_y * 4`. Biome embed input `4 * spatial_y * 4`. Both `SparseOctreeModel` and `SparseOctreeFastModel` accept and forward `spatial_y`. **Note**: files were moved from `LODiffusion/models/` to `VoxelTree/voxel_tree/tasks/sparse_octree/` (they were in the wrong repo).

13. - [x] **Update `export_sparse_octree.py`**: Contract bumped to `lodiffusion.v7.sparse_octree`. `_STANDARD_NOISE_3D_CHANNELS` updated to 15 RouterField names. Dummy input `(1, 15, 4, 4, 4)`, biome `(1, 4, 4, 4)`. Sidecar config updated. `_VT_ROOT` path reference fixed (was double-VoxelTree bug). Moved to VoxelTree alongside `sparse_octree.py`.

14. - [x] **Update `sparse_octree_train.py`**: Auto-detects `spatial_y` from NPZ shape (`noise_3d.shape[3]`). Passes `spatial_y` to model constructors. Zero-fill biome fallback uses detected spatial.

### Phase F — Pipeline Orchestration

15. - [x] **Update `step_definitions.py`**: Added 4 new `ModelTrack`s (`density_v2`, `biome_classifier`, `heightmap_predictor`, `sparse_octree_v7`) each with `build_pairs → train → export → deploy` steps. Added `dumpnoise_v7` and `build_v7_pairs` data acquisition steps. All v7 tracks consume `v7_dumps` artifact. Also fixed all imports after `sparse_octree.py`/`export_sparse_octree.py` were moved from LODiffusion.

16. - [x] **`profiles/v7_staged.yaml`**: Full pipeline profile with all 4 model tracks.

17. - [x] **Update `MODEL-CONTRACT.md`**: Added §C documenting all 4 v7 contracts (`lodiffusion.v7.sparse_octree`, `lodiffusion.v7.density_v2`, `lodiffusion.v7.biome_classifier`, `lodiffusion.v7.heightmap_predictor`) with input/output shapes, ONNX opsets, sidecar formats.

18. - [x] **Deprecate legacy paths**: Added deprecation notices to: `NoiseDumperCommand` `sparse_root` sub-command, `WorldNoiseAccess.sampleNoise3DForSection()`, `diagnose.py`, `calibrate.py`, `export_sparse_octree.py` legacy channel list, `sparse_octree.py` legacy biome comment. Did **not** delete — deprecation markers only, allowing old checkpoints to still load for comparison.

### Additional completed work (not in original plan)

- [x] **Import fixes after file move**: Updated imports in `sparse_octree_train.py`, `sparse_octree_distill.py`, `step_definitions.py` (×3 sites), `test_sparse_root_train.py` (removed sys.modules stub), `__init__.py` (tombstone), `demo_sparse_octree.py`, both `.ipynb` notebooks, `MODEL-CONTRACT.md`
- [x] **`voxel_tree/contracts/` directory** — appears in `git status` as untracked (created during a session, purpose unclear)
- [x] **`profiles/new_profile.yaml`** — untracked, likely a working copy
- [x] **Modified `cli.py`** and density_v2 scripts — appear in `git status` as modified but uncommitted

---

## Verification (none run yet)

- [ ] **Data round-trip**: Run `/dumpnoise v7 8` in-game (8-chunk radius), then `build_sparse_octree_pairs.py` → confirm NPZ shapes are `(N, 15, 4, 4, 4)` for noise and `(N, 4, 4, 4)` for biomes
- [ ] **Density model smoke test**: Train density_v2 on a small dump, export ONNX, load in Python with `onnxruntime` and verify output shape `[1, 2]`
- [ ] **Biome classifier smoke test**: Train on small dump, verify top-1 accuracy > 80% on held-out quarts
- [ ] **Heightmap smoke test**: Train on small dump, verify MAE < 2 blocks vs vanilla heightmap
- [ ] **Sparse octree end-to-end**: Train updated model with 15ch input, export as v7 ONNX, load in Java `SparseOctreeModelRunner`, verify inference produces valid octree
- [ ] **Parity**: Use shadow mode (`terrainBackend=shadow`) to compare v7 model outputs against vanilla

---

## Decisions
- **Clean break from 13ch/4×2×4**: old dumps, old NPZs, old checkpoints are incompatible — no migration path needed
- **15 RouterField channels exclusively**: no derived features (offset/factor/jaggedness are gone — the model learns those relationships from raw vanilla fields)
- **Density predictor targets 2 outputs** (FINAL_DENSITY + PRELIMINARY_SURFACE_LEVEL): these are the two most expensive DensityFunctions and share upstream computation
- **Heightmap model vs analytical zero-crossing**: train an NN first for flexibility; the GPU analytical pass can coexist as a fallback
- **Biome classifier uses 6 climate inputs**: TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES — these are exactly what `MultiNoiseBiomeSource` uses internally

---

## Next Phase: GPU Pipeline

The next major work block (from conversation.md §§1–3) is the GPU-first pipeline. All CPU-side infrastructure is complete. The GPU side needs:

### 1. Quart-Resolution Shader (`quart_noise_compute.comp`)
New compute shader: `layout(local_size_x=4, local_size_y=4, local_size_z=4)` — 64 threads = 1 section. Each thread evaluates all 15 RouterField values at one quart centre. Output: `float[N * 960]` in a single SSBO. `glDispatchCompute(N, 1, 1)` for batched sections. **Dramatically simpler** than `terrain_compute.comp` — no surface detection, no material classification, no Y-column iteration.

**Existing GPU primitives available:**
- `mc_improved_noise()` — ImprovedNoise perlin (perm table, gradient, trilinear)
- `mc_perlin_noise()` — multi-octave PerlinNoise
- `mc_normal_noise()` — NormalNoise (two PerlinNoise blended)
- `mc_terrain_shaper_mlp()` — 4→32→32→3 MLP for offset/factor/jaggedness (binding 9+10)
- Spline evaluation from binding 6

**Missing:** 7 NormalNoise wiring calls (barrier, flood, spread, lava, vein_toggle, vein_ridged, vein_gap). The GLSL function exists; `ShadowRouterExtractor` just needs to extract these 7 additional noise indices into the UBO.

### 2. ShadowRouterExtractor Expansion
Extract indices for the 7 missing NormalNoise instances from vanilla `NoiseRouter`. Expand `RouterConfig` UBO from 112 to ~160+ bytes (or use the 256-byte layout from `ROUTERCONFIG_EXTENSION.glsl`).

### 3. Async GPU Dispatch Queue
`SectionRequestQueue` with `CompletableFuture<SectionNoiseData>` result delivery. Gen thread enqueues batch of 8 section origins → render thread dispatches `quart_noise_compute.comp` → reads back `float[8 × 960]` → completes futures. Double-buffered output SSBOs (ping/pong) to eliminate read-after-write stalls.

### 4. Wire `GpuNoiseRouterSampler`
Replace CPU fallback stub with real GPU SSBO readback via the async queue. This is the payoff — the gen thread submits noise requests to the GPU and gets back `SectionNoiseData` without touching `DensityFunction.sample()`.

### 5. `GpuHeightmapProvider` + `GpuBiomeProvider`
- Heightmaps: dedicated GPU pass for density zero-crossing (top-down surface scan)
- Biomes: wrap existing `BiomePaletteSSBO` (binding 13) into `BiomeProvider` interface