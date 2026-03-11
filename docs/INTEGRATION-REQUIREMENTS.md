# 📦 Integration Instructions for VoxelTree

**⚠️ LEGACY DOCUMENT** This describes the now‑deprecated four‑model progressive pipeline (contract `lodiffusion.v3.progressive`).
For current octree architecture see `OCTREE-GENERATION-DESIGN.md` and `MODEL-CONTRACT.md` §A.

## 🎯 Purpose

The `VoxelTree` training pipeline must produce **four ONNX models** forming a progressive LOD refinement pipeline suitable for **real-time, just-in-time terrain generation** inside the **LODiffusion** Minecraft mod. Models are queried at runtime to generate plausible terrain in unexplored distant chunks, working from coarse (LOD4) down to fine (LOD1). LOD0 is always vanilla-authoritative.

Contract ID: `lodiffusion.v3.progressive`

---

## ✅ Output Model Requirements

All four models must meet **all** of the following:

### 1. **ONNX Export**

* Format: `.onnx` (opset 17, ONNX v1.14+)
* Input/output shapes must be *static* and explicitly defined for export.
* Must include all normalization, padding, and activation layers inline.

### 2. **Runtime Inference Compatibility**

* Must be compatible with **DJL (Deep Java Library)** + ONNX Runtime backend.
* No unsupported ops (e.g., dynamic shape manipulations, control flow).
* Prefer simple convolutional ops (`Conv3d`, `GroupNorm`, `ReLU`, etc.).

### 3. **Input Schema (all four models share these conditioning inputs)**

| Name              | Shape            | Type      | Description                                              |
| ----------------- | ---------------- | --------- | -------------------------------------------------------- |
| `x_height_planes` | `[1, 5, 16, 16]` | `float32` | Surface, ocean_floor, slope_x, slope_z, curvature        |
| `x_biome`         | `[1, 16, 16]`    | `int64`   | Vanilla biome index per (x,z) horizontal position        |
| `x_y_index`       | `[1]`            | `int64`   | Vertical slab index [0, 23]                              |
| `x_parent`        | `[1, 1, P, P, P]`| `float32` | Previous stage output; **refinement models only** — omit for `init_to_lod4` |

Parent tensor size P per model:

| Model                    | `x_parent` shape   |
| ------------------------ | ------------------ |
| `init_to_lod4`           | *absent*           |
| `refine_lod4_to_lod3`    | `[1, 1, 1, 1, 1]`  |
| `refine_lod3_to_lod2`    | `[1, 1, 2, 2, 2]`  |
| `refine_lod2_to_lod1`    | `[1, 1, 4, 4, 4]`  |

### 4. **Output Schema**

| Name           | Shape                       | Type      | Description                       |
| -------------- | --------------------------- | --------- | --------------------------------- |
| `block_logits` | `[1, N_blocks, D, D, D]`    | `float32` | Raw logits for each block type    |
| `air_mask`     | `[1, 1, D, D, D]`           | `float32` | Probability that a voxel is empty |

Output resolution D per model:

| Model                 | D (native output) | Java post-processing                                        |
| --------------------- | ----------------- | ----------------------------------------------------------- |
| `init_to_lod4`        | 1                 | —                                                           |
| `refine_lod4_to_lod3` | 2                 | —                                                           |
| `refine_lod3_to_lod2` | 4                 | —                                                           |
| `refine_lod2_to_lod1` | 8                 | `ProgressiveModelRunner` upsamples 2× → 16³ for Voxy write |

> The final 16³ `InferenceResult` produced by `ProgressiveModelRunner` is what gets written to Voxy.

---

## 🔁 Training Data Expectations

* Supervised pairs per model:
  * `x_height_planes` / `x_biome` / `x_y_index`: extracted from real Minecraft chunks via `VoxelTree/scripts/`
  * `x_parent`: argmax of previous stage output (or zeros for `init_to_lod4`)
  * Target: block IDs at native resolution D³
* Augmentation should preserve semantic terrain continuity (e.g., no flipping that breaks slope direction)

---

## 🔬 Loss Functions

* Primary: CrossEntropy over `block_logits`
* Secondary: Binary mask loss for `air_mask` (optional but recommended)
* Accuracy metrics: `argmax(block_logits) == block_target` and `air_mask` MAE vs true occupancy

---

## 🧠 Design Constraints from LODiffusion

LODiffusion's in-mod runtime has strict limits:

* **Inference time must be < 100ms combined across all four models** on midrange CPU (no GPU)
* Memory per chunk must stay under ~2 MB total
* Each chunk is generated *just before it's rendered* by DH → models must be stable, fast, and resilient to garbage input

---

## 🧪 Sanity Check Tests

Before merging to `main`, VoxelTree must:

* ✅ Export all four models to `.onnx` and verify with `verify_lodiffusion_v1.py`
* ✅ Confirm `pipeline_manifest.json` lists all required files with correct hashes
* ✅ Validate deterministic outputs (same input = same output for each model)
* ✅ Confirm shapes match contract exactly (fail-fast if not)

---

## 🔌 Post-Training Deliverables to LODiffusion

1. `init_to_lod4.onnx`
2. `refine_lod4_to_lod3.onnx`
3. `refine_lod3_to_lod2.onnx`
4. `refine_lod2_to_lod1.onnx`
5. `init_to_lod4_config.json` — block vocabulary, normalization specs
6. `refine_lod4_to_lod3_config.json`
7. `refine_lod3_to_lod2_config.json`
8. `refine_lod2_to_lod1_config.json`
9. `pipeline_manifest.json` — file list with checksums; validated by LODiffusion at startup
10. `init_to_lod4_test_vectors.npz`, `refine_lod4_to_lod3_test_vectors.npz`, `refine_lod3_to_lod2_test_vectors.npz`, `refine_lod2_to_lod1_test_vectors.npz`

Deploy to LODiffusion via: `python scripts/deploy_models.py production/vN` (copies to `LODiffusion/run/config/lodiffusion/`)

---

## 📍 Where It Goes in the Mod

Once deployed:

* `ProgressiveModelRunner` (loaded by `LodGenerationService`) loads all four ONNX models at startup
* At each chunk generation:
  * `AnchorSampler` prepares `x_height_planes`, `x_biome`, `x_y_index` from vanilla worldgen
  * `ProgressiveModelRunner.generate()` chains the four inference stages, carrying `x_parent` forward
  * Final 8³ output is upsampled 2× to 16³ and written to Voxy by `VoxySectionWriter`
