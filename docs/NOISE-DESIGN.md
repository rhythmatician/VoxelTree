# Per-LOD Vanilla Noise Design

> How much vanilla noise does each LOD transition need, and where does it come from?

---

## 1. Problem Statement

The model accepts `x_router6 [1,6,16,16] float32` as a conditioning input — six
2D noise maps (temperature, vegetation, continentalness, erosion, depth, ridges).
Today, this input is **fake on both sides**:

| Side | What happens | Result |
|------|-------------|--------|
| **Training (Python)** | NPZ files contain no noise data. `approximate_router6_from_biome()` builds crude linear proxies from biome ID + heightmap. | Model sees redundant info it already gets from biome & height inputs. |
| **Runtime (Java)** | `LodGenerationService` uses synthetic sine/cosine heightmaps + hardcoded `biome=1`. Router6 approximated from those fakes. | Noise conditioning is meaningless. |

The `NoiseTap` infrastructure exists on the Java side (15 router fields, 4
performance tiers, benchmarked) but is **not wired into** the generation loop.
On the Python side, no mechanism exists to extract real noise during data
preparation.

### What "real noise" means

Minecraft 1.18+ determines biomes and terrain shape via the **multi-noise
system** (the "NoiseRouter"). Six of its density functions correspond directly
to our router6 channels:

| Channel | NoiseRouter Field | cubiomes Enum | What it controls |
|---------|------------------|---------------|-----------------|
| 0 | Temperature | `NP_TEMPERATURE` | Hot vs cold biomes |
| 1 | Vegetation/Humidity | `NP_HUMIDITY` | Lush vs barren |
| 2 | Continentalness | `NP_CONTINENTALNESS` | Ocean → inland gradient |
| 3 | Erosion | `NP_EROSION` | Flat vs mountainous |
| 4 | Depth | `NP_DEPTH` / `NP_SHIFT` | Vertical biome placement |
| 5 | Ridges/Weirdness | `NP_WEIRDNESS` | Peak shapes, cavern floors |

These are deterministic for a given seed + (x, y, z). They vary slowly in the
horizontal plane (biome scale ≈ 1:4) but provide spatial structure that the
model cannot recover from biome IDs alone.

---

## 2. Per-LOD Tier Mapping

Not every LOD transition needs the same noise fidelity. Coarse transitions
predict large-scale structure (land vs ocean, rough height bands) where 2D
climate fields suffice. Fine transitions reconstruct block-level detail where
3D density influences caves and overhangs.

### Chosen mapping

| Transition | Parent → Target | What's predicted | Noise tier | Router fields | Est. cost (Java) |
|-----------|----------------|-----------------|-----------|--------------|-----------------|
| LOD4→LOD3 | 1³ → 2³ | Continent outline | **CORE** | 6 (temp, veg, cont, eros, depth, ridges) | ~15 ms |
| LOD3→LOD2 | 2³ → 4³ | Biome-scale terrain | **CORE** | 6 | ~15 ms |
| LOD2→LOD1 | 4³ → 8³ | Regional detail | **CORE** | 6 | ~15 ms |
| LOD1→LOD0 | 8³ → 16³ | Block-level detail | **CORE** | 6 | ~15 ms |

> **Phase-1 simplification**: All transitions use CORE (6 fields).
> The ONNX contract has a fixed `x_router6 [1,6,16,16]` shape — this is
> already the right shape for CORE. Expanding later requires changing the
> model input channels, so we start uniform and add complexity only when
> ablation studies show a gain.

### Future expansion (post Phase-1)

If cave quality at LOD1→LOD0 proves insufficient:

| Transition | Noise tier | Fields added | Extra cost |
|-----------|-----------|-------------|-----------|
| LOD1→LOD0 | CAVE_AWARE | +`INITIAL_DENSITY_NO_JAG`, `FINAL_DENSITY` | ~50 ms |
| LOD2→LOD1 | EXTENDED | +`FLUID_FLOODEDNESS`, `FLUID_SPREAD`, `LAVA`, `BARRIER` | ~17 ms |

This would require widening `x_router6` from 6→8 or 6→12 channels and
retraining. The `AnchorConditioningFusion` module already parameterises
`router6_channels` so the architecture change is trivial; the data pipeline
change is the real cost.

---

## 3. Where to Get Real Noise

### 3a. Training side (Python)

The training pipeline runs **offline** — it reads pre-extracted Voxy RocksDB
data, not live Minecraft chunks. It cannot call `DensityFunction.sample()`.

**Options evaluated:**

| Approach | Status | Notes |
|---------|--------|-------|
| **pyubiomes** (Python wrapper for cubiomes C library) | **BROKEN** — v0.2.0 fails to compile on Python 3.13 / MSVC (javarnd.h syntax errors). No maintained fork. | Only version on PyPI; stale upstream. |
| **cubiomes-wrapper**, **pycubiomes** | **Don't exist** on PyPI. | |
| **Existing CLI** (`tools/voxeltree_cubiomes_cli.exe`) | Works, but only supports `biome` and `height` commands — **no router6**. | Prebuilt binary, no source in repo. |
| **Extend cubiomes CLI** for `climate` command | **Best option** — cubiomes C library exposes `sampleBiomeNoise(bn, np, x, y, z, ...)` which populates `np[6]` with all climate parameters. Need to add a `climate <seed> <x> <z> <w> <h>` command that outputs the 6 noise values per coordinate. | Requires rebuilding the CLI from cubiomes source. |
| **NoiseDumper command** (`/dumpnoise`) | **Already implemented** in LODiffusion. Dumps CORE router6 + heightmaps + biomes to JSON per chunk from a running MC client. | Requires a live MC session; limited to loaded chunks. Best for spot-checking / validation, not bulk extraction. |
| **Offline Java extractor** | Write a standalone Java tool that initialises `NoiseConfig` from a seed and samples DensityFunctions without a running server. | Most accurate — uses the exact same code as MC. Heavy setup (needs Minecraft mappings/classpath). |

#### Recommended: Extend cubiomes CLI

Add a new command to the cubiomes CLI:

```
voxeltree_cubiomes_cli climate <seed> <x> <z> <w> <h> [--y <y>] [--scale <s>]
```

Output: 6 floats per coordinate (temperature, humidity, continentalness,
erosion, depth, weirdness), one row per (x, z) position.

The cubiomes API call is:

```c
BiomeNoise bn;
initBiomeNoise(&bn, MC_1_21);
setBiomeSeed(&bn, seed, /*large=*/0);

int64_t np[6];  // output: NP_TEMPERATURE..NP_WEIRDNESS
sampleBiomeNoise(&bn, np, x, y, z, NULL, 0);
// np[] now contains fixed-point climate values (divide by 10000.0 for float)
```

This runs in microseconds per sample — extracting a 16×16 grid takes <1 ms.
For the full training set (~150k patches × 16×16 = ~38.4M samples), extraction
would take ~40 seconds.

#### Integration into extraction pipeline

Update `scripts/extract_voxy_training_data.py` to:

1. For each extracted NPZ patch, compute its world-space (x, z) range from
   the section key.
2. Shell out to the extended CLI (or use ctypes/cffi binding) to get the 6
   climate values on a 16×16 grid at surface Y.
3. Save as `router6_patch (6, 16, 16) float32` in the NPZ file.

Downstream, `MultiLODDataset.__getitem__()` loads `router6_patch` when present
and falls back to `approximate_router6_from_biome()` when absent (backward
compat).

### 3b. Runtime side (Java)

The infrastructure is already built:

- **`NoiseTap.java`** — interface with `captureAll(fields, heightmaps)` → `Cache`
- **`NoiseTapImpl.java`** — samples `DensityFunction.sample()` at 16×16×16
- **`NoiseDumperCommand.java`** — `/dumpnoise` command for validation
- **`AnchorSampler.java`** — `sample()` method extracts biomes + heightmap +
  router6 from a Chunk (uses `NoiseTap` under the hood)

**What's missing** is the wiring in `LodGenerationService.inferAndPushSection()`:

```
Current:  buildHeightmap() → synthetic sine
          biome → hardcoded 1
          router6 → approximateRouter6() from fakes

Needed:   NoiseTap.bind(chunk, noiseConfig, biomeAccess, seed)
          → captureAll(CORE_FIELDS, DEFAULT_HEIGHTMAPS)
          → use cache.heightmaps16, cache.biomes4, cache.router for model inputs
```

This is GAP-5 from the Java audit. The fix is to:
1. Plumb `NoiseConfig` into `LodGenerationService` (currently unavailable —
   needs reflection or API hook, same as `NoiseDumperCommand.tryGetNoiseConfig`)
2. Replace `buildHeightmap()` with `cache.getHeightmap(WORLD_SURFACE_WG)`
3. Replace hardcoded `biome=1` with `cache.biomes4` (upsampled 4×4→16×16)
4. Replace `approximateRouter6()` with real router data from cache

---

## 4. 2D vs 3D Noise

The ONNX contract uses `x_router6 [1,6,16,16]` — **2D** (one value per (x,z)
column). But noise router fields are actually **3D** — they vary with Y.

| Field | Y-variation | Impact |
|-------|------------|--------|
| Temperature | Negligible (2D noise at biome scale) | 2D is fine |
| Vegetation (Humidity) | Negligible | 2D is fine |
| Continentalness | Negligible | 2D is fine |
| Erosion | Negligible | 2D is fine |
| Depth | **Strong** — changes linearly with Y (depth = surface - y) | 2D fails underground |
| Ridges (Weirdness) | Moderate — affects cave shape at boundaries | 2D adequate for surface |

For Phase-1 (terrain-only, no caves), 2D sampling at surface Y is sufficient.
The `y_index` input already tells the model which vertical slab it's
reconstructing, which compensates for the lack of Y-varying depth.

For cave support (post Phase-1), we would need to either:
- Sample noise at the section's midpoint Y rather than surface Y
- Expand to 3D: `x_router6 [1,6,16,16,16]` (adds 15× more data per sample)
- Add separate depth/ridges channels sampled at section Y alongside 2D climate

---

## 5. ONNX Shape Constraint

The model contract is fixed for Phase-1:

```
x_router6: [1, 6, 16, 16] float32
```

All 6 channels are always populated. The LOD token (`x_lod`) tells the model
which transition it's performing; the router6 content is the same real noise
regardless of LOD level.

The key insight: **the model can learn to use different noise channels at
different LOD levels** via the LOD token conditioning, even though the input
shape is identical. At LOD4→LOD3 it may attend primarily to continentalness
and temperature; at LOD1→LOD0 it may attend more to erosion and ridges.

---

## 6. Implementation Roadmap

### Phase 1a: Extend cubiomes CLI (training data)

1. **Build cubiomes CLI from source** with a new `climate` command
   - Input: `climate <seed> <x> <z> <w> <h>`
   - Output: 6 floats per (x, z) coordinate: temp, humidity, cont, erosion, depth, weirdness
   - Uses `sampleBiomeNoise()` at sea level Y=63

2. **Add `--cli-path` option to extraction script**
   - `extract_voxy_training_data.py --cli-path tools/voxeltree_cubiomes_cli.exe`
   - Appends `router6_patch` to each NPZ file

3. **Update dataset loader**
   - Load `router6_patch` from NPZ when present
   - Fall back to `approximate_router6_from_biome()` when absent
   - Log warning ratio so we know how many patches have real vs approximate noise

### Phase 1b: Wire NoiseTap into runtime (Java)

4. **Plumb `NoiseConfig`** into `LodGenerationService`
   - Use same reflection approach as `NoiseDumperCommand.tryGetNoiseConfig()`
   - Cache at service startup; log warning if unavailable (graceful degradation)

5. **Replace synthetic inputs** in `inferAndPushSection()`
   - Real heightmap from `cache.getHeightmap(WORLD_SURFACE_WG)`
   - Real biomes from `cache.biomes4` → upsample 4→16
   - Real router6 from `cache.router` (CORE tier)
   - Fall back to approximation if `NoiseConfig` unavailable

### Phase 1c: Validate

6. **Spot-check consistency**: Use `/dumpnoise` to dump real router6 from a
   running MC session, compare against cubiomes CLI output for the same seed
   + coordinates. They should be close (cubiomes is a clean-room reimplementation
   so minor floating-point differences are expected).

7. **Ablation study**: Train two models — one with real noise, one with
   approximate noise — and compare metrics. This quantifies the value of real
   noise conditioning.

### Phase 2: Expand noise tiers (if needed)

8. Add EXTENDED/CAVE_AWARE fields when cave quality metrics warrant it.
   This touches: CLI output, NPZ format, model input channels, ONNX contract.

---

## 7. Cubiomes ↔ MC Noise Mapping

For validation and cross-checking:

| cubiomes enum | cubiomes `np[]` index | MC NoiseRouter field | NoiseTap `RouterField` | router6 channel |
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

## 8. Current Infrastructure Inventory

### Python side (VoxelTree)

| File | Purpose | Noise status |
|------|---------|-------------|
| `scripts/extract_voxy_training_data.py` | Voxy DB → NPZ | **No noise** — outputs labels16, biome, height, y_index only |
| `train/anchor_conditioning.py` | Router6 approximation + fusion | `approximate_router6_from_biome()` — crude linear proxy |
| `train/multi_lod_dataset.py` | Dataset loader | Always calls approximate; no NPZ router6 loading |
| `train/unet3d.py` | Model | Accepts router6 as optional kwarg |
| `tools/voxeltree_cubiomes_cli.exe` | Cubiomes CLI | `biome` + `height` only, no climate |
| `scripts/extraction/cubiomes_integration.py` | Cubiomes helper | Uses `find` command not available in current CLI |

### Java side (LODiffusion)

| File | Purpose | Noise status |
|------|---------|-------------|
| `NoiseTap.java` | Interface — 15 fields, 4 tiers | **Ready** |
| `NoiseTapImpl.java` | Implementation — DensityFunction sampling | **Ready** |
| `NoiseDumperCommand.java` | `/dumpnoise` command → JSON | **Ready** (dumps CORE fields) |
| `AnchorSampler.java` | Chunk → biome/height/router6 | **Ready** but NOT wired in |
| `LodGenerationService.java` | Generation loop | **Uses fakes** — sine heightmap, biome=1 |
| `UnifiedModelRunner.java` | Tensor construction | v2 contract correct, feeds whatever input it receives |

---

## 9. Risk & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| cubiomes noise doesn't match MC noise exactly | Medium — cubiomes is clean-room C reimpl | Validate with `/dumpnoise` on same seed; accept ≤5% deviation |
| Extending CLI requires C toolchain on Windows | Low — we already have MSVC | Use CMake from cubiomes repo |
| Real noise doesn't improve metrics vs approximate | Low — but possible for coarse LODs | If so, keep approximate for coarse; use real for LOD1→LOD0 only |
| `NoiseConfig` reflection breaks in future MC versions | Medium | Graceful fallback to approximation; log warnings |

---

## 10. Key Decision: Train with Real Noise from the Start

Rather than training with approximate noise now and retraining later, **we should
extend the CLI first** and re-extract training data with real router6 values.
This avoids:
- Wasting a training run on data we know is suboptimal
- Needing to re-extract 150k patches later
- Risk of the model learning to ignore the router6 input (since approximate ≈ biome + height)

The extraction overhead is minimal: ~40 seconds for 38M samples via cubiomes CLI.
