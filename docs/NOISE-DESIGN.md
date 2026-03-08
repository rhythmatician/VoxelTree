# Per-LOD Noise & Conditioning Design

> What conditioning inputs does each LOD transition need, and why router6 was dropped.

---

## 1. Decision: Router6 Dropped (March 2026)

The model originally accepted `x_router6 [1,6,16,16] float32` — six 2D noise
maps (temperature, vegetation, continentalness, erosion, depth, ridges). This
input has been **removed from the architecture entirely**.

### Why it was dropped

| Observation | Implication |
|-------------|------------|
| **Biome IS the output of router6.** MC's multi-noise system feeds temperature, vegetation, continentalness, erosion, depth, and ridges into a lookup table that produces a biome ID. The biome index already encodes the *outcome* of those 6 fields. | Router6 → biome is a many-to-one mapping. Giving the model both is largely redundant. |
| **Heightmap IS the output of depth/continentalness/erosion/ridges.** The terrain shape is determined by those noise fields. The heightmap captures their combined effect. | 4 of 6 router6 channels are already represented by the heightmap. |
| **No real router6 data existed.** All 53K+ NPZ training files contain only `labels16`, `biome_patch`, `heightmap_patch`, and `y_index`. Router6 was always approximated — `approximate_router6_from_biome()` was literally reconstructing a crude inverse of the biome→noise mapping, feeding the model information it already had. | The model was trained on circular/redundant data. Removing it loses nothing. |
| **No LOD0 generation.** We only generate LOD4→LOD1 (coarse terrain). Vanilla Minecraft handles LOD0 (block-level). At these coarse resolutions, biome + heightmap + y_index provide sufficient spatial context. | Router6's value would only appear at fine-grained (LOD0) cave/overhang detail. |
| **Massive simplification.** Dropping router6 eliminates the need for cubiomes CLI extensions, NoiseTap runtime wiring, and offline Java extractors — all of which were unbuilt infrastructure blocking training. | Unblocks the entire data pipeline with zero new tooling. |

### What the model uses now

| Input | Shape | Type | What it provides |
|-------|-------|------|-----------------|
| `height_planes` | `[B, 5, 16, 16]` | float32 | 5 normalised height features: raw heightmap, surface height, surface - slab_y, normalised Y position, height variance |
| `biome_idx` | `[B, 16, 16]` | int64 | Biome index per column → learned embedding (256 biomes) |
| `y_index` | `[B]` | int64 | Vertical slab index (0–23) → learned embedding. Tells the model "how deep" this section is. |
| `parent_voxel` | `[B, 1, P, P, P]` | float32 | Binary occupancy from parent LOD (refinement models only; P ∈ {1,2,4}) |

### ONNX contract (v3, post router6 removal)

```
Init model (init→LOD4):
  Inputs:  x_height_planes [1,5,16,16] float32
           x_biome         [1,16,16]   int64
           x_y_index       [1]         int64
  Outputs: block_logits    [1,V,1,1,1] float32
           air_mask        [1,1,1,1,1] float32

Refinement models (LOD4→3, LOD3→2, LOD2→1):
  Inputs:  x_height_planes [1,5,16,16] float32
           x_biome         [1,16,16]   int64
           x_y_index       [1]         int64
           x_parent        [1,1,P,P,P] float32
  Outputs: block_logits    [1,V,D,D,D] float32
           air_mask        [1,1,D,D,D] float32
```

---

## 2. What Router6 Was (Reference)

Minecraft 1.18+ determines biomes and terrain shape via the **multi-noise
system** (the "NoiseRouter"). Six of its density functions were the router6
channels:

| Channel | NoiseRouter Field | cubiomes Enum | What it controls |
|---------|------------------|---------------|-----------------|
| 0 | Temperature | `NP_TEMPERATURE` | Hot vs cold biomes |
| 1 | Vegetation/Humidity | `NP_HUMIDITY` | Lush vs barren |
| 2 | Continentalness | `NP_CONTINENTALNESS` | Ocean → inland gradient |
| 3 | Erosion | `NP_EROSION` | Flat vs mountainous |
| 4 | Depth | `NP_DEPTH` / `NP_SHIFT` | Vertical biome placement |
| 5 | Ridges/Weirdness | `NP_WEIRDNESS` | Peak shapes, cavern floors |

These are deterministic for a given seed + (x, y, z). They vary slowly in the
horizontal plane (biome scale ≈ 1:4).

**Key insight:** biome = f(temperature, vegetation, continentalness, erosion,
depth, ridges). Giving the model both the inputs and the output of this function
is redundant for coarse LOD generation.

---

## 3. Per-LOD Conditioning (Current)

All 4 LOD transitions use the same conditioning inputs — biome, heightmap,
and y_index. The LOD token tells the model which resolution it's operating at.

| Transition | Parent → Target | What's predicted | Conditioning |
|-----------|----------------|-----------------|-------------|
| Init→LOD4 | — → 1³ | Single-voxel seed | biome + height + y_index |
| LOD4→LOD3 | 1³ → 2³ | Continent outline | biome + height + y_index + parent |
| LOD3→LOD2 | 2³ → 4³ | Biome-scale terrain | biome + height + y_index + parent |
| LOD2→LOD1 | 4³ → 8³ | Regional detail | biome + height + y_index + parent |

> LOD1→LOD0 (8³ → 16³) is **not generated** — vanilla Minecraft handles
> full-resolution terrain. This is also the transition where router6 would
> have provided the most value (cave shapes, overhangs).

---

## 4. Architecture: AnchorConditioningFusion

The conditioning fusion module was simplified from 3 streams to 2:

### Before (with router6)
```
height_planes [B,5,H,W] → Conv → quarter of channels
router6       [B,6,H,W] → Conv → half of channels     ← REMOVED
biome_indices [B,H,W]   → Embedding → Conv → quarter of channels
                                  ↓
                    concat + y_embed → fusion MLP → [B, C, H, W]
```

### After (without router6)
```
height_planes [B,5,H,W] → Conv → third of channels
biome_indices [B,H,W]   → Embedding → Conv → third of channels
                                  ↓
                    concat + y_embed → fusion MLP → [B, C, H, W]
```

The biome stream is larger now (third vs quarter), which makes sense — biome
carries the information that router6 used to provide.

---

## 5. Existing Noise Infrastructure (Preserved but Unused)

The following infrastructure still exists and could be reactivated if a
future ablation study shows value in raw noise conditioning:

### Python side (VoxelTree)

| File | Purpose | Current status |
|------|---------|---------------|
| `scripts/extraction/chunk_extractor.py` | Voxy DB → NPZ extraction | Passes through router6 if present in source data (never is) |
| `scripts/dataset_respec.py` | Normalisation utilities | Has router6 normalisation code (unused) |
| `tools/voxeltree_cubiomes_cli.exe` | Cubiomes CLI | `biome` + `height` commands only; no `climate` command was ever added |

### Java side (LODiffusion)

| File | Purpose | Current status |
|------|---------|---------------|
| `NoiseTap.java` | Interface — 15 router fields, 4 tiers | **Ready** but not used by model |
| `NoiseTapImpl.java` | DensityFunction sampling | **Ready** |
| `NoiseDumperCommand.java` | `/dumpnoise` validation command | **Ready** — useful for debugging |
| `AnchorSampler.java` | Chunk → biome/height extraction | Biome + heightmap extraction still used; router6 path unused |

### What the Java runtime needs now

With router6 removed, `LodGenerationService` needs:
1. **Real heightmaps** from `NoiseTap` cache or chunk data (replaces synthetic sine)
2. **Real biomes** from chunk biome access (replaces hardcoded biome=1)
3. **y_index** from the section's Y coordinate

Router6 tensor construction can be removed from `UnifiedModelRunner`.

---

## 6. Future: When Would Router6 Come Back?

Router6 reintroduction would only be justified if:

1. **LOD0 generation is added** (8³ → 16³ block-level detail), where cave
   shapes depend on depth/ridges/erosion at specific Y values.
2. **An ablation study** on LOD2→LOD1 shows measurably better terrain quality
   with raw noise vs biome-only conditioning.
3. **Real noise data is available** — not approximated from biome IDs.

If reintroduced, the path would be:
- Extend cubiomes CLI with a `climate` command (design in Section 7 below)
- Re-extract training data with `router6_patch (6, 16, 16) float32` in NPZ files
- Widen `AnchorConditioningFusion` back to 3 streams
- Update ONNX contract to include `x_router6 [1,6,16,16]`
- Wire `NoiseTap` into `LodGenerationService` (GAP-5)

---

## 7. Reference: Cubiomes CLI Extension Design (Deferred)

> This section is preserved for future reference. The CLI extension was never
> built because router6 was dropped before it was needed.

To add a `climate` command to the cubiomes CLI:

```
voxeltree_cubiomes_cli climate <seed> <x> <z> <w> <h> [--y <y>] [--scale <s>]
```

Output: 6 floats per coordinate (temperature, humidity, continentalness,
erosion, depth, weirdness), one row per (x, z) position.

The cubiomes API call:

```c
BiomeNoise bn;
initBiomeNoise(&bn, MC_1_21);
setBiomeSeed(&bn, seed, /*large=*/0);

int64_t np[6];  // output: NP_TEMPERATURE..NP_WEIRDNESS
sampleBiomeNoise(&bn, np, x, y, z, NULL, 0);
// np[] now contains fixed-point climate values (divide by 10000.0 for float)
```

Extraction of 38.4M samples (~150k patches × 16×16) would take ~40 seconds.

---

## 8. Cubiomes ↔ MC Noise Mapping (Reference)

For future validation and cross-checking:

| cubiomes enum | `np[]` index | MC NoiseRouter field | NoiseTap `RouterField` | Former router6 channel |
|----|---|---|---|---|
| `NP_TEMPERATURE` | 0 | `temperature` | `TEMPERATURE` | 0 |
| `NP_HUMIDITY` | 1 | `vegetation` | `VEGETATION` | 1 |
| `NP_CONTINENTALNESS` | 2 | `continentalness` | `CONTINENTS` | 2 |
| `NP_EROSION` | 3 | `erosion` | `EROSION` | 3 |
| `NP_DEPTH` / `NP_SHIFT` | 4 | `depth` | `DEPTH` | 4 |
| `NP_WEIRDNESS` | 5 | `ridges` | `RIDGES` | 5 |

Note: cubiomes returns fixed-point `int64_t` values — divide by 10000.0 to
get the float equivalent. MC's `DensityFunction.sample()` returns raw doubles.

---

## 9. Code Changes (March 2026)

Files modified to remove router6:

| File | What changed |
|------|-------------|
| `train/anchor_conditioning.py` | Removed `approximate_router6_from_biome()`, ROUTER_* constants. Rewrote `AnchorConditioningFusion` from 3 streams to 2 (height + biome). `router6_channels` kept as ignored ctor arg for checkpoint compat. |
| `train/unet3d.py` | Removed `router6_channels` from `SimpleFlexibleConfig`. `forward()` accepts but ignores `router6` kwarg for backward compat. |
| `train/multi_lod_dataset.py` | Removed router6 from pair generation, saving, loading, `__getitem__`, and `collate_multi_lod_batch`. |
| `train_multi_lod.py` | Removed `--router6-channels` CLI arg and `router6` from `forward_step()`. |
| `train_progressive_quick.py` | Removed router6 from all model calls and dummy data. |
| `scripts/build_pairs.py` | Updated docstring and warning messages (no longer references router6). |
| `scripts/export_lod.py` | Removed `x_router6` from ONNX adapters, dummy inputs, input specs, sidecar config, and test vectors. |

### Files NOT modified (secondary, conditional usage):
- `scripts/dataset_respec.py` — router6 normalisation (only runs if data present)
- `scripts/extraction/chunk_extractor.py` — passes through router6 if in source (never is)
- `scripts/pairing/patch_pairer.py`, `seed_input_linker.py` — older pairing scripts (unused)
