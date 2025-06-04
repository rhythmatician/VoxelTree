# 📦 Integration Instructions for VoxelTree

## 🎯 Purpose

The `VoxelTree` training pipeline must produce a model suitable for **real-time, just-in-time terrain generation** inside the **LODiffusion** Minecraft mod. This model will be queried at runtime to generate plausible terrain in **unexplored** distant chunks based on nearby terrain context and known environmental features.

---

## ✅ Output Model Requirements

The final model trained by VoxelTree must meet **all** of the following:

### 1. **ONNX Export**

* Format: `.onnx` (ONNX v1.12+)
* Input/output shapes must be *static* and explicitly defined for export.
* Must include all normalization, padding, and activation layers inline.

### 2. **Runtime Inference Compatibility**

* Must be compatible with **DJL (Deep Java Library)** + ONNX Runtime backend.
* No unsupported ops (e.g., dynamic shape manipulations, control flow).
* Prefer simple convolutional ops (`Conv3D`, `GroupNorm`, `ReLU`, etc.).

### 3. **Input Schema**

| Name         | Shape                    | Type      | Description                                                                |
| ------------ | ------------------------ | --------- | -------------------------------------------------------------------------- |
| `x_parent`   | `[1, 1, 8, 8, 8]`        | `float32` | Binary parent voxel block — 1 for solid, 0 for air                         |
| `x_biome`    | `[1, N_biomes, 8, 8, 1]` | `float32` | One-hot biome embedding collapsed across vertical axis                     |
| `x_height`   | `[1, 1, 8, 8, 1]`        | `float32` | Normalized heightmap values (0–1), sampled at each 8×8 horizontal position |
| `x_lod`      | `[1, 1]`                 | `float32` | Scalar LOD timestep embedded via sinusoidal positional encoding            |
| *(Optional)* | `[1, C, 8, 8, 8]`        | `float32` | Future conditioning channels like temperature or noise                     |

> All inputs must be normalized and **dimensionally consistent** with the output resolution.

### 4. **Output Schema**

| Name           | Shape                       | Type      | Description                                              |
| -------------- | --------------------------- | --------- | -------------------------------------------------------- |
| `block_logits` | `[1, N_blocks, 16, 16, 16]` | `float32` | Raw logits for each block type                           |
| `air_mask`     | `[1, 1, 16, 16, 16]`        | `float32` | Probability that a voxel is empty (for blending with DH) |

---

## 🔁 Training Data Expectations

* Supervised pairs of:

  * `parent_block`: 8³ binary voxel
  * `child_block`: 16³ label (block ID or class)
* Conditioning:

  * Biome IDs → one-hot or embedding
  * Surface heightmap → sampled to 8×8
  * LOD timestep → sinusoidal positional encoding (float)
* Augmentation should preserve semantic terrain continuity (e.g. no flipping that breaks slope direction).

---

## 🔬 Loss Functions

* Primary: CrossEntropy over `block_logits`
* Secondary: Binary mask loss for `air_mask` (optional but recommended)
* Accuracy metrics: `argmax(block_logits) == block_target` and `air_mask` MAE vs true occupancy

---

## 🧠 Design Constraints from LODiffusion

LODiffusion's in-mod runtime has strict limits:

* **Inference time must be < 100ms per patch** on midrange CPU (no GPU)
* Memory per patch must stay under \~2MB total
* Each patch is generated *just before it’s rendered* by DH → model must be stable, fast, and resilient to garbage input

---

## 🧪 Sanity Check Tests

Before merging to `main`, VoxelTree must:

* ✅ Export model to `.onnx` and run inference using DJL in Java
* ✅ Confirm model output changes with LOD timestep
* ✅ Validate deterministic outputs (same input = same output)
* ✅ Ensure gradients flow correctly during training

---

## 🔌 Post-Training Deliverables to LODiffusion

1. `model.onnx` — Trained model for DJL inference
2. `model_config.json` — Metadata describing:

   * Input/output names
   * LOD range
   * Biome block mappings
   * Normalization info (mean/std or min/max per channel)
3. `test_vectors.npz` — Input/output examples to validate runtime integration

---

## 📍 Where It Goes in the Mod

Once integrated:

* `DiffusionRunner.java` loads `model.onnx`
* At each chunk generation:

  * Uses surrounding chunk + DH LOD → prepares model input
  * Runs model → gets 16³ terrain patch prediction
  * Blends output with DH-rendered LOD0 edge if nearby
