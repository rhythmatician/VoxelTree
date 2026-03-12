![VoxelTree logo](https://github.com/user-attachments/assets/c323591b-3fb1-48b6-a3c3-4a19cfcbeebf)

# 🌲 VoxelTree

VoxelTree trains a **3-model octree pipeline** for vanilla Minecraft terrain generation, aligned with Voxy's WorldSection hierarchy. Each model produces a complete **32³ block prediction** at a specific LOD level. Empty octants are pruned via learned occupancy prediction, and their subtrees never run inference.

The model runs in the **LODiffusion** mod to render distant terrain just-in-time.

---

## 🧠 What It Does

VoxelTree's octree pipeline consists of three models:

| Model | Level(s) | Input | Output |
|---|---|---|---|
| `OctreeInitModel` | L4 (root) | heightmap + biome + y_position | block_logits + occ_logits |
| `OctreeRefineModel` | L3, L2, L1 (shared) | parent_context + heightmap + biome + y_position + level | block_logits + occ_logits |
| `OctreeLeafModel` | L0 (leaf) | parent_context + heightmap + biome + y_position | block_logits (no occ head) |

Training data is extracted from **Voxy RocksDB** databases using a canonical
1104-entry block vocabulary. Parent→child context is passed via argmax block IDs
from each parent's 32³ predictions, octant-extracted and upsampled 2×.

---

## 🔬 Research Goals

- Train LOD-aware voxel refinement models using real-world Minecraft data
- Learn terrain continuity *without* neighbor context (initially)
- Support headless training on partial worlds to minimize disk usage
- Export ONNX models for runtime use in Minecraft (via Fabric mod)

---

## 🤖 AI-Assisted Development

This repository is part of a unique development experiment:
- 🧠 High-level supervision by **ChatGPT-4o**
- 🤖 Autonomous feature development by **Claude Sonnet 4 (Agent mode)** using GitHub Copilot
- 🧪 Strict TDD (Test-Driven Development) methodology enforced by `PROJECT-OUTLINE.md`

Each feature is implemented in a **micro-commit cycle**:
- 🔴 RED: failing test
- 🟢 GREEN: implementation
- ⚪ REFACTOR: reflection + docs

> The goal is to test how far Copilot can be pushed with precise architecture and supervision — even in complex research settings.

---

## 📁 Key Directories

| Path         | Purpose                                        |
|--------------|------------------------------------------------|
| `train/`     | Model architecture, dataset loader, losses, metrics |
| `scripts/`   | Voxy extraction, ONNX export, mipper, benchmarks |
| `config/`    | Canonical Voxy vocabulary (`voxy_vocab.json`)  |
| `tests/`     | Unit tests (PyTest)                            |
| `models/`    | Saved checkpoints + ONNX exports (git-ignored) |
| `data/`      | Extracted training NPZs (git-ignored)          |
| `docs/`      | Architecture, acceptance criteria, reflections |

---

## 🧰 Key Scripts

| Script                                    | Purpose                                          |
|-------------------------------------------|--------------------------------------------------|
| `pipeline.py`                             | Train → export → deploy orchestrator             |
| `data-cli.py`                             | Unified dataprep CLI                             |
| `train_octree.py`                         | 3-model octree training                          |
| `scripts/extract_octree_data.py`          | Voxy RocksDB → per-level NPZ                    |
| `scripts/voxy_reader.py`                  | RocksDB reader (SaveLoadSystem3 decoder)         |
| `scripts/add_column_heights.py`           | Merge vanilla heightmaps into NPZs               |
| `scripts/build_octree_pairs.py`           | NPZ → octree parent/child training pairs         |
| `scripts/export_octree.py`               | Checkpoint → 3 ONNX models                      |
| `scripts/deploy_models.py`               | Copy ONNX models to LODiffusion config dir       |

---

## 📘 Docs

- [Project Outline](docs/PROJECT-OUTLINE.md)
- [Acceptance Criteria](docs/AC.md)
- [Voxy Format](docs/VOXY-FORMAT.md)
- [Copilot Instructions](.github/copilot-instructions.md)

---

## 🛠 Status

VoxelTree is **in active development** — contributors welcome, but please read the CI and TDD guidelines first.

### Recent Progress:
- ✅ Voxy-native block vocabulary (1104 canonical entries)
- ✅ 3-model octree pipeline (init / refine / leaf)
- ✅ Octree parent→child context with 32³ WorldSection grids
- ✅ Occupancy prediction for octree pruning (learned child occupancy)
- ✅ Shared OctreeRefineModel with level embedding (L3/L2/L1)
- ✅ Pipeline orchestrator (dataprep → train → export → deploy)
- ✅ ONNX export and LODiffusion integration
- ✅ Static ONNX export with Voxy vocab embedding
- ⏳ Full training run on Voxy-extracted data
- ⏳ In-game deployment via LODiffusion mod

---

**Seed: `VoxelTree`**
