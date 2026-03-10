![VoxelTree logo](https://github.com/user-attachments/assets/c323591b-3fb1-48b6-a3c3-4a19cfcbeebf)

# 🌲 VoxelTree

VoxelTree is an experimental terrain refinement model designed to **progressively upscale Minecraft terrain data** using a voxel-based super-resolution architecture inspired by diffusion models.

Rather than generating entire terrain blocks from scratch, VoxelTree learns to **refine coarse, LOD-based voxel grids** into more detailed terrain using information derived from the Minecraft world seed (biome, heightmap, y-level, etc.). This enables scalable, just-in-time terrain generation with the potential for seamless integration into Minecraft via LODiffusion.

---

## 🧠 What It Does

VoxelTree models take:
- A coarse parent occupancy voxel (`x_parent`: 8×8×8, Mipper-derived)
- Anchor conditioning signals (`x_height_planes`, `x_biome`)
- Vertical position (`x_y_index`) and LOD coarseness token (`x_lod`)

And outputs:
- Block type predictions (`block_logits`: 1102-class Voxy-native vocabulary over 16×16×16)
- Air probability mask (`air_mask`: 16×16×16)

Training data is extracted directly from **Voxy RocksDB** databases using a canonical
1102-entry block vocabulary. LOD coarsening uses the **Voxy Mipper algorithm** (opacity-biased
corner selection), ensuring exact parity with Voxy's own LOD pipeline.

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
| `pipeline.py`                             | Two-phase orchestrator: extract → train → export → deploy |
| `train_multi_lod.py`                      | Multi-LOD training CLI with Voxy vocab           |
| `scripts/extract_voxy_training_data.py`   | Voxy RocksDB → NPZ training patches             |
| `scripts/voxy_reader.py`                  | RocksDB reader (SaveLoadSystem3 decoder)         |
| `scripts/mipper.py`                       | Voxy Mipper (canonical LOD coarsening)           |
| `scripts/export_lod.py`                   | Static ONNX export (opset ≥ 17)                  |
| `scripts/verify_onnx.py`                  | ONNX export + test vector verification           |

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
- ✅ Voxy-native block vocabulary (1102 canonical entries)
- ✅ Voxy RocksDB extraction pipeline (multi-world)
- ✅ Voxy Mipper (100% parity with Voxy's own LOD coarsening)
- ✅ V2 anchor-conditioned model (height planes + biome)
- ✅ Two-phase pipeline orchestrator (`pipeline.py`)
- ✅ Multi-LOD training with dynamic coarsening
- ✅ Static ONNX export with Voxy vocab embedding
- ⏳ Full training run on Voxy-extracted data
- ⏳ In-game deployment via LODiffusion mod

---

**Seed: `VoxelTree`**
