## Plan: VoxelTree Full Staged Pipeline Migration

**TL;DR:** Migrate the VoxelTree training pipeline from the legacy 13-channel/4×2×4 format to 15 RouterField channels at 4×4×4 quart resolution, and build four models for the full staged pipeline: density predictor, biome classifier, heightmap predictor, and block selector (sparse octree). This is a clean break — no backward compat. The data-harvester Fabric mod gets a new `/dumpnoise v7` command, the Python training code gets a new `router_field.py` canonical definition, and each model gets its own training script, ONNX export, and DAG step.

**Steps**

### Phase A — Data Infrastructure

1. **Create `router_field.py`** in voxel_tree/utils/: a Python-side canonical definition of the 15 RouterField channels matching Java's `RouterField` enum exactly — same names, same ordinals (0–14: TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES, PRELIMINARY_SURFACE_LEVEL, FINAL_DENSITY, BARRIER, FLUID_LEVEL_FLOODEDNESS, FLUID_LEVEL_SPREAD, LAVA, VEIN_TOGGLE, VEIN_RIDGED, VEIN_GAP). Include `COUNT = 15`, classification helpers (e.g., `CLIMATE_FIELDS`, `DENSITY_FIELDS`, `AQUIFER_FIELDS`, `ORE_FIELDS`).

2. **Update data-harvester `WorldNoiseAccess.java`** (WorldNoiseAccess.java): add a new `sampleRouterFieldsForSection(sectionX, sectionY, sectionZ)` method that samples all 15 vanilla `NoiseRouter` fields at **4×4×4 quart resolution** (cell size = 4 blocks in all axes). This replaces the old `sampleNoise3DForSection()` which samples 13 hand-picked fields at 4×2×4. Return `float[15][4][4][4]` (960 floats). The 15 fields map 1:1 to the `DensityFunction` router accessors: `temperature()`, `vegetation()`, `continents()`, `erosion()`, `depth()`, `ridges()`, `initialDensityWithoutJaggedness()`, `finalDensity()`, `barrierNoise()`, `fluidLevelFloodednessNoise()`, `fluidLevelSpreadNoise()`, `lavaNoise()`, `veinToggle()`, `veinRidged()`, `veinGap()`.

3. **Update data-harvester biome sampling**: add `sampleBiomeIdsForSectionV7(sx, sy, sz)` returning `int[4][4][4]` at quart resolution (matching the new spatial grid).

4. **New `/dumpnoise v7 [radius]` sub-command** in NoiseDumperCommand.java: dumps per-section JSON files to `run/v7_dumps/section_{cx}_{sy}_{cz}.json` containing:
   - `router_fields`: float[15][4][4][4] — all 15 channels
   - `biome_ids`: int[4][4][4] — canonical palette IDs
   - `heightmap_surface`: float[16][16] — vanilla `WORLD_SURFACE_WG`
   - `heightmap_ocean_floor`: float[16][16] — vanilla `OCEAN_FLOOR_WG`
   - Field names stored as metadata array matching `RouterField` ordinals

5. **Update `build_sparse_octree_pairs.py`** (build_sparse_octree_pairs.py): change `N_FIELDS` from 13 → 15, spatial reshape from `(4,2,4)` → `(4,4,4)`, read from `v7_dumps/` directory, output NPZ with shape `noise_3d: (N, 15, 4, 4, 4)`, `biome_ids: (N, 4, 4, 4)`. Drop the `noise_2d` placeholder entirely. Add `heightmap_surface: (N, 16, 16)` and `heightmap_ocean_floor: (N, 16, 16)` arrays to the output NPZ for heightmap model training.

### Phase B — Density Predictor

6. **New `tasks/density_v2/train_density_v2.py`**: a small MLP that predicts expensive density-related fields from cheap climate inputs. Input: 6 climate fields (TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES) at a single quart position → output: FINAL_DENSITY (and optionally PRELIMINARY_SURFACE_LEVEL). Architecture: `6 → 128 → 128 → 2` MLP with ReLU. Train on per-quart samples extracted from the v7 dump data (each section provides 64 training samples). MSE loss. This model enables skipping the most expensive vanilla DensityFunction evaluations at inference time.

7. **ONNX export for density_v2**: input `float32[batch, 6]` → output `float32[batch, 2]`. Also export flat `.bin` weights for GPU SSBO upload (matching the existing density SSBO pattern from TerrainShaperMlpSsbo.java).

### Phase C — Biome Classifier

8. **New `tasks/biome/train_biome_classifier.py`**: trains a lightweight classifier from climate noise → biome ID. Input: 6 climate fields (same as density predictor) per quart → output: softmax over 54 biome classes. Architecture: `6 → 64 → 64 → 54` MLP with ReLU, CrossEntropyLoss. Training data: (climate_noise, biome_id) pairs extracted from v7 dumps (64 quarts per section × all dumped sections). This replaces vanilla's `MultiNoiseBiomeSource.getBiome()` at inference time.

9. **ONNX export for biome classifier**: input `float32[batch, 6]` → output logits `float32[batch, 54]`. Argmax in Java runtime to get predicted biome ID.

### Phase D — Heightmap Predictor

10. **New `tasks/heightmap/train_heightmap.py`**: predicts surface height per (x, z) column from density values. Input: a column of FINAL_DENSITY values at quart Y positions (96 Y-quarts for the full -64→320 range, or 4 per section × 24 sections), plus the 6 climate fields for that column → output: `worldSurface` Y and `oceanFloor` Y (2 values). Architecture: `102 → 128 → 64 → 2` MLP (96 density values + 6 climate features). MSE loss, trained against vanilla heightmaps from the v7 dump. Alternative: a 1D convolution over the density column may generalize better — try both.

11. **ONNX export for heightmap**: input `float32[batch, 102]` → output `float32[batch, 2]`. Or if column-based: `float32[batch, 96+6]` → `float32[batch, 2]`. Note: the Java `HeightmapProvider` returns `float[16][16]` per section, so runtime calls this 256 times per section (once per block column) or batches.

### Phase E — Block Selector (Sparse Octree Update)

12. **Update `sparse_octree.py` `_NoiseEncoder`** (sparse_octree.py): change `flat_3d` calculation from `n3d * 4 * 2 * 4` → `n3d * 4 * 4 * 4`, change default `n3d` from 13 → 15 in both `SparseOctreeModel` and `SparseOctreeFastModel`. Update `forward()` shape comments. Update biome spatial from `4 × 2 × 4` → `4 × 4 × 4` (biome_embed goes from 256 → 512 flat features).

13. **Update `export_sparse_octree.py`** (export_sparse_octree.py): bump contract to `lodiffusion.v7.sparse_octree`, update `_STANDARD_NOISE_3D_CHANNELS` to the 15 RouterField names, change dummy input shape to `(1, 15, 4, 4, 4)`, update biome dummy to `(1, 4, 4, 4)`, update sidecar config and CLI defaults. Remove the pre-biome padding logic (clean break).

14. **Update training script** sparse_octree_train.py: update docstrings, default spatial shapes in `SparseOctreeDataset`, and the zero-fill biome fallback from `(n, 4, 2, 4)` → `(n, 4, 4, 4)`.

### Phase F — Pipeline Orchestration

15. **Update `step_definitions.py`** (step_definitions.py): add new model tracks for `density_v2`, `biome_classifier`, and `heightmap_predictor`, each with their own `build_pairs → train → export → deploy` phases. Add a `dumpnoise_v7` data acquisition step that produces `v7_dumps` artifact. Wire all three new tracks to consume `v7_dumps`. Update the `sparse_octree` track to consume `v7_dumps` instead of `noise_dumps`.

16. **New training profile** `profiles/v7_staged.yaml`: full DAG including `pregen → harvest → dumpnoise_v7 → extract_octree` → all four model tracks (density_v2, biome_classifier, heightmap_predictor, sparse_octree) → deploy all.

17. **Update `MODEL-CONTRACT.md`** (MODEL-CONTRACT.md): document the `lodiffusion.v7.sparse_octree` contract (15ch/4×4×4 input, 10-tensor octree output), plus new contracts for density_v2, biome_classifier, and heightmap_predictor (input/output shapes, ONNX opsets, sidecar formats).

18. **Delete or deprecate legacy code paths**: remove `/dumpnoise sparse_root` sub-command (replaced by `/dumpnoise v7`), remove `density_extraction.py` (placeholder, never implemented), update `MASTER_PLAN.md` to reflect the new staged pipeline.

**Verification**

- **Data round-trip**: Run `/dumpnoise v7 8` in-game (8-chunk radius), then `build_sparse_octree_pairs.py` → confirm NPZ shapes are `(N, 15, 4, 4, 4)` for noise and `(N, 4, 4, 4)` for biomes
- **Density model smoke test**: Train density_v2 on a small dump, export ONNX, load in Python with `onnxruntime` and verify output shape `[1, 2]`
- **Biome classifier smoke test**: Train on small dump, verify top-1 accuracy > 80% on held-out quarts
- **Heightmap smoke test**: Train on small dump, verify MAE < 2 blocks vs vanilla heightmap
- **Sparse octree end-to-end**: Train updated model with 15ch input, export as v7 ONNX, load in Java `SparseOctreeModelRunner`, verify inference produces valid octree
- **Parity**: Use shadow mode (`terrainBackend=shadow`) to compare v7 model outputs against vanilla

**Decisions**
- **Clean break from 13ch/4×2×4**: old dumps, old NPZs, old checkpoints are incompatible — no migration path needed
- **15 RouterField channels exclusively**: no derived features (offset/factor/jaggedness are gone — the model learns those relationships from raw vanilla fields)
- **Density predictor targets 2 outputs** (FINAL_DENSITY + PRELIMINARY_SURFACE_LEVEL): these are the two most expensive DensityFunctions and share upstream computation
- **Heightmap model vs analytical zero-crossing**: train an NN first for flexibility; the GPU analytical pass can coexist as a fallback
- **Biome classifier uses 6 climate inputs**: TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES — these are exactly what `MultiNoiseBiomeSource` uses internally