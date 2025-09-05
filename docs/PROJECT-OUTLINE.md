# 🌲 VoxelTree Project Outline

> **TDD workflow legend**
> **[X]** = Cycle complete (RED → GREEN → REFACTOR)  **[ ]** = Not started  **[~]** = In-progress  **[🆕]** = Added/changed in this outline

---

## ✅ Phase 0 — Developer Infrastructure

**Status:** Complete

| Cycle    | Goal            | Notes                                           |
| -------- | --------------- | ----------------------------------------------- |
| [X] 0.1 | Repo scaffold   | `train/`, `scripts/`, `tests/`, `docs/`         |
| [X] 0.2 | CI + pre-commit | GH Actions (Linux/Windows) + `pre-commit` hooks |
| [X] 0.3 | Config loader   | Central schema + env overrides                  |

---

## 🆕 Phase 0B — Worldgen Integration (Fabric **1.21.5**) & Caching

**Status:** [~] In-progress (new API-aligned design)

**Principles**

* **Cache at API granularity** (no upsampling in the mod).

  * Heightmaps: **16×16** planes (`WORLD_SURFACE_WG`, `OCEAN_FLOOR_WG`), plus derived `slope_x`, `slope_z`, `curvature`.
  * Biomes: **4×4×4** quart lattice features (temp, precip[3], isCold).
  * NoiseRouter slices: **16×16 @ 1 Y** for `temperature, vegetation, continents, erosion, depth, ridges` (Router-6).
  * Optional: `barrier` (16×16), `aquifer` trio (16×16), **coarse cave prior** (4×4×4 or 8×8×8).
* **Downsample only** when feeding early LODs; **all upsampling is done inside the ONNX models**.
* **Vanilla carve** is **kept** and called only at **LOD0**.

| Cycle      | Goal                             | Notes                                                                     |
| ---------- | -------------------------------- | ------------------------------------------------------------------------- |
| [🆕] 0B.1 | `NoiseTap` interface + impl      | One-call capture per chunk; returns native-grid tensors                   |
| [🆕] 0B.2 | `FeatureBundle` cache + LRU      | In-mem LRU + optional sidecar on disk; keyed by `ChunkPos`                |
| [🆕] 0B.3 | Normalization plumbing           | Heights (min-max by world limits), Router/Aquifer (z-score), flags [0,1] |
| [🆕] 0B.4 | Parity tests vs vanilla samplers | Unit tests for heights/biomes/router consistency                          |

---

## 🆕 Phase 1 — Dataset Respec (Native Caches → LOD Targets)

**Status:** [ ] Not started

| Cycle     | Goal                          | Notes                                                            |
| --------- | ----------------------------- | ---------------------------------------------------------------- |
| [🆕] 1.1 | Cache-driven dataset builder  | Read native caches; no upsampling; emit shared inputs            |
| [🆕] 1.2 | LOD targets (1³/2³/4³/8³/16³) | Majority downscaling for labels; `air_mask` as mean(air)         |
| [🆕] 1.3 | Channel stats export          | Per-channel mean/std (Router/Aquifer/Cave), world height min/max |
| [🆕] 1.4 | `test_vectors.npz` generation | Golden vectors for DJL parity                                    |

---

## 🆕 Phase 2 — Model Family v3 (5 Static-Shape ONNX Models)

**Status:** [ ] Not started

**Design rules**

* Five models: **Init (noise→LOD4)**, then **LOD4→3**, **3→2**, **2→1**, **1→0**.
* Pure Conv3D + GroupNorm + ReLU (+ Resize/nearest or strided-conv) — **no dynamic ops**.
* Inputs are **exact shared cached tensors**; only `x_parent_prev` size changes per LOD.
* Optional channels (`barrier`, `aquifer`, `cave_prior`) are **zero-fillable**.

| Cycle     | Goal                               | Notes                                             |
| --------- | ---------------------------------- | ------------------------------------------------- |
| [🆕] 2.1 | Init micro-UNet (outputs 1³)       | Fastest; establishes first coarse voxel           |
| [🆕] 2.2 | LOD refine UNets (2³, 4³, 8³, 16³) | Width caps scale with D; CPU-friendly             |
| [🆕] 2.3 | Internal resize blocks             | Models upsample planar/biome/parent internally    |
| [🆕] 2.4 | Config stubs per model             | `model_config.json` names, shapes, norms, palette |

---

## 🆕 Phase 3 — Training & Ablations

**Status:** [ ] Not started

| Cycle     | Goal                                     | Notes                               |
| --------- | ---------------------------------------- | ----------------------------------- |
| [🆕] 3.1 | Baseline: Height+Biome+Coords+LOD+Parent | No Router/Aquifer/Cave              |
| [🆕] 3.2 | + Router-6 (2D slice)                    | Check seam/shore coherence & IoU    |
| [🆕] 3.3 | + Barrier (2D)                           | Deltas on deltas; coastal crispness |
| [🆕] 3.4 | + Aquifer (2D)                           | Surface wetness fidelity            |
| [🆕] 3.5 | + Cave prior (4³) **only for LOD1→0**    | Underground fidelity vs cost        |
| [🆕] 3.6 | Timing: sampling+inference per LOD       | Keep <100 ms/patch budget           |

---

## 🆕 Phase 4 — Evaluation & Visualization

**Status:** [ ] Not started

| Cycle | Goal                                          |
| ----: | --------------------------------------------- |
|   4.1 | Accuracy (block IoU/Dice) + `air_mask` MAE    |
|   4.2 | **Seam metrics** across chunk borders         |
|   4.3 | Coastline continuity (edge/PSNR at sea level) |
|   4.4 | 3-D voxel preview (fast viewer)               |

---

## 🆕 Phase 5 — ONNX Export (x5) & Parity

**Status:** [ ] Not started

| Cycle | Goal                          | Notes                          |
| ----: | ----------------------------- | ------------------------------ |
|   5.1 | Export 5 models to ONNX 1.12+ | Static shapes only             |
|   5.2 | PyTorch↔ONNX numeric parity   | Use `test_vectors.npz`         |
|   5.3 | Op audit                      | Conv3D/GN/ReLU/Resize only     |
|   5.4 | Size & memory checks          | <~2 MB per patch at inference |

---

## 🆕 Phase 6 — LODiffusion Runtime Integration

**Status:** [ ] Not started

| Cycle | Goal                                                 |
| ----: | ---------------------------------------------------- |
|   6.1 | Java tensor packer (shared inputs → ONNX I/O shapes) |
|   6.2 | Progressive loop: Init → 4→3 → 3→2 → 2→1 → 1→0       |
|   6.3 | Caching: `FeatureBundle` LRU + optional sidecar      |
|   6.4 | **Vanilla carve** only at LOD0                       |
|   6.5 | Metrics & logs: sampling ms / inference ms per stage |

---

## 🆕 Phase 7 — Performance & Reliability

**Status:** [ ] Not started

| Cycle | Goal                                        |
| ----: | ------------------------------------------- |
|   7.1 | Disk-aware cache management (evict/TTL)     |
|   7.2 | Graceful fallback (missing cache → rebuild) |
|   7.3 | Seed/coord determinism tests                |
|   7.4 | Memory caps & GC-safe buffers               |

---

## 📌 Immediate Next Steps

1. Implement **NoiseTap (0B.1)** + **FeatureBundle (0B.2)** with unit tests.
2. Draft **five `model_config.json` stubs** (Phase 2.4) with shapes below.
3. Build **dataset respec** (Phase 1.1–1.3) and emit first `test_vectors.npz`.
4. Start **Init model** (Phase 2.1) and verify end-to-end through DJL with the vectors.

---

## 📎 Spec Appendix — **Unified I/O (5 Models)**

**Shared cached inputs (produced once; no upsampling in the mod)**

* `x_height_planes` → **[1,5,1,16,16]**
  *(surface, ocean_floor, slope_x, slope_z, curvature)*
* `x_biome_quart` → **[1,6,4,4,4]**
  *(temp, precip[3], isCold, downfall)*
* `x_router6` → **[1,6,1,16,16]**
  *(temperature, vegetation, continents, erosion, depth, ridges)*
* *(opt)* `x_barrier` → **[1,1,1,16,16]**
* *(opt)* `x_aquifer3` → **[1,3,1,16,16]**
* *(opt)* `x_cave_prior4` → **[1,1,4,4,4]**
* Scalars: `x_chunk_pos` **[1,2]**, `x_lod` **[1,1]**

> **Normalization**: heights (min-max by world limits), Router/Aquifer/Cave (z-score), flags [0,1], coords via affine/tanh.
> **Palette**: `N_blocks` documented per dataset in `model_config.json`.

**Per-model inputs/outputs**

| Model | Purpose     | `x_parent_prev`          | Outputs (`block_logits`, `air_mask`)     |
| ----: | ----------- | ------------------------ | ---------------------------------------- |
|     0 | Init → LOD4 | **[1,1,1,1,1]** (zeros) | **[1,N,1,1,1]**, **[1,1,1,1,1]**       |
|     1 | LOD4 → LOD3 | **[1,1,1,1,1]**         | **[1,N,2,2,2]**, **[1,1,2,2,2]**       |
|     2 | LOD3 → LOD2 | **[1,1,2,2,2]**         | **[1,N,4,4,4]**, **[1,1,4,4,4]**       |
|     3 | LOD2 → LOD1 | **[1,1,4,4,4]**         | **[1,N,8,8,8]**, **[1,1,8,8,8]**       |
|     4 | LOD1 → LOD0 | **[1,1,8,8,8]**         | **[1,N,16,16,16]**, **[1,1,16,16,16]** |

**Policy highlights**

* **No upsampling in the mod**; models contain static Resize/conv to align inputs with their internal grid.
* **Downsampling** only when feeding earlier LODs (handled **inside** the training/inference graphs for parity).
* **Vanilla `carve()`** runs only at **LOD0** to finalize caves/aquifers/structures.

---

### Notes on previous outline

* Old phases **0B/1/2/3/4/5** (seed-only + single UNet) are **superseded** by the five-model family and API-native caching.
* “Water depth” input was **removed**; the model learns `surface − ocean_floor` implicitly.
* Chunk coordinate pair `(chunkX,chunkZ)` is now an explicit scalar input for global coherence.
