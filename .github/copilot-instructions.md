# 🤖 Copilot Instructions — VoxelTree

This file guides GitHub Copilot (and contributors) in writing code for the `VoxelTree` project. It ensures consistency, avoids architectural pitfalls, and enforces our test-driven development (TDD) cycle.

---

## 🌲 Project Purpose

VoxelTree implements and trains a **LOD-aware, voxel super-resolution model** for Minecraft terrain. It progressively refines terrain data from coarser representations (e.g., 8×8×192) to higher fidelity forms (e.g., 16×16×384), conditioned on biome and heightmap data.

The trained model is exported to ONNX and used inside the LODiffusion Minecraft mod for **real-time, just-in-time terrain generation**.

---

## ✅ Core Constraints

- ⚙️ Use Python 3.10+
- 📦 Store extracted chunk data as `.npz` (use `np.savez_compressed`)
- 📁 Use `pathlib.Path` over `os.path`
- 🧵 Use `multiprocessing` (not threading) for batch `.mca` extraction
- 🚮 Never exceed 10–20 GB of disk usage during training
- 🧠 No deep learning during extraction — keep it fast and CPU-only
- 🔁 Only train on a **subset of chunks at a time**, then delete
- 🧰 Use `anvil-parser2` for MCA reading
- 🌍 Use `Chunky + Fabric` as headless vanilla-compatible world generator

---

## 🧪 Test-Driven Development (TDD)

Each feature is developed in a 3-phase cycle:

1. **RED** — Write a failing test
2. **GREEN** — Write just enough code to pass the test
3. **REFACTOR** — Reflect on structure, log insights, and update docs

### Commit Rules

- Every phase (`RED`, `GREEN`, `REFACTOR`) must be a separate commit
- Each TDD cycle must occur in a **feature branch** (e.g. `feat/mca-loader`)
- Only merge to `main` after REFACTOR is complete and documented

---

## 🗂️ Directory Overview

VoxelTree/
├── train/
│ ├── train.py
│ ├── dataset.py
│ ├── unet3d.py
│ ├── loss.py
│ └── config.yaml
├── scripts/
│ ├── export_onnx.py
│ ├── run_eval.py
│ └── generate_samples.py
├── models/ # Trained checkpoints and ONNX exports
├── tests/ # PyTest test suite
├── data/ # Temporary patch/chunk files
├── tools/ # Worldgen JARs and CLI tools (e.g., Chunky, cubiomes)
├── PROJECT-OUTLINE.md
├── ENVIRONMENT.md
└── .gitignore

---

## 🧠 Model Design Guidelines

- Input: `(parent_voxel, biome_patch, heightmap, lod_embedding)`
- Output: `(air_mask_logits, block_type_logits)`
- Architecture: 3D U-Net with skip connections
- Loss: BCE for mask + CE for block types
- Timestep embedding: sinusoidal or learned

---

## 🧱 Chunk Format

Extracted `.npz` chunk files must contain:

```python
{
  "block_types": uint8, shape=(16, 16, 384)
  "air_mask": bool,     shape=(16, 16, 384)
  "biomes": uint8 or int32, shape=(16, 16) or (16, 16, 384)
  "heightmap": uint8,   shape=(16, 16)
}
```

---

## 🧩 Training Data Preparation
Downsampled parent-child patch pairs should have:

- Parent voxel: (e.g. 8×8×8)
- Target mask: (e.g. 16×16×16)
- Target types: (same shape)

---

## 📦 Worldgen Tools

- **Primary**: Chunky (Fabric) + `fabric-server-*.jar` for `.mca` generation
- **Biome/Heightmap Source**: `voxeltree_cubiomes_cli.exe`
- **Parser**: `anvil-parser2` for reading `.mca`

All worldgen is CLI-invoked and fully headless.

---

## 🔁 Legacy Tooling Cleanup

Do not reference or use:
- `minecraft-worldgen.jar` (deprecated)
- `hephaistos.jar` (replaced by `anvil-parser2`)
- `opensimplex`, `noise`, or other procedural terrain gen tools

---

## 🔧 Configuration Changes

```yaml
# config.yaml:
worldgen:
  seed: "VoxelTree"
  java_heap: "4G"
  chunk_batch_size: 32
  java_tools:
    primary: "tools/fabric-server/fabric-server-mc.1.21.5-loader.0.16.14-launcher.1.0.3.jar"
    chunky: "tools/chunky/Chunky-Fabric-1.4.36.jar"
    cubiomes: "tools/voxeltree_cubiomes_cli/voxeltree_cubiomes_cli.exe"
```

---

Copilot, you build the brain.
We’ll handle the dirt.
