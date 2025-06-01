# 🌲 VoxelTree

VoxelTree is an experimental terrain refinement model designed to **progressively upscale Minecraft terrain data** using a voxel-based super-resolution architecture inspired by diffusion models.

Rather than generating entire terrain blocks from scratch, VoxelTree learns to **refine coarse, LOD-based voxel grids** into more detailed terrain using information derived from the Minecraft world seed (biome, heightmap, y-level, etc.). This enables scalable, just-in-time terrain generation with the potential for seamless integration into Minecraft via LODiffusion.

---

## 🧠 What It Does

VoxelTree models take:
- A coarse parent voxel (e.g., 8×8×8 air/solid mask or block types)
- Local seed-derived metadata (biome, heightmap, river signal, y-index)
- A target LOD timestep

And outputs:
- A refined terrain mask (16×16×16 air/solid)
- Detailed block type predictions (logits or classes)

This allows terrain to be generated **progressively and locally**, with higher detail closer to the player.

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

| Path         | Purpose                          |
|--------------|----------------------------------|
| `train/`     | Model, dataset, training loop    |
| `scripts/`   | Evaluation, ONNX export          |
| `tests/`     | Unit tests (PyTest)              |
| `models/`    | Saved checkpoints + ONNX         |
| `docs/`      | Architecture, project outline    |

---

## 📘 Docs

- [Project Outline](docs/PROJECT-OUTLINE.md)
- [Training Overview](docs/TRAINING-OVERVIEW.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Copilot Instructions](.github/copilot-instructions.md)

---

## 🛠 Status

VoxelTree is **in active development** — contributors welcome, but please read the CI and TDD guidelines first.

---

**Seed: `VoxelTree`**