
# 🌲 VoxelTree Project Outline — TDD-Driven

> Each numbered phase below corresponds to a **feature-driven TDD cycle**:
- Begin by writing a failing test (RED)
- Implement the logic to pass the test (GREEN)
- Reflect and document lessons or pitfalls (REFACTOR)
- Merge to `main` only after completing the cycle

---

## ✅ Phase 0 — Developer Infrastructure

| TDD Cycle | Goal                         | Description                          |
|-----------|------------------------------|--------------------------------------|
| [X] 0.1       | Repo scaffold                | Set up `train/`, `scripts/`, `tests/`, `docs/` |
| [X] 0.2       | CI with Pytest               | GitHub Actions or pre-commit hooks   |
| [X] 0.3       | config.yaml loader           | Validate all config options centrally|

---

## 🧱 Phase 1 — `.mca` Chunk Extraction

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [X] 1.1       | Load `.mca` file             | Fails if region is missing or unreadable         |
| [X] 1.2       | Extract subchunks            | Fails if 16³ slices are incorrect or misaligned  |
| [X] 1.3       | Downsample to 8³ parent      | Fails if pooling is inaccurate or shape mismatch |
| [X] 1.4       | Save input-output `.npz`     | Fails if file format or keys are malformed       |
| [X] 1.5       | Multiprocess batch extract   | Fails if total sample count is off               |

---

## 🧬 Phase 1B — Seed-Based Input Generation

> Headless extraction of biome, heightmap, and river signals using seed and position only.

| TDD Cycle | Goal                           | RED Test Focus                                      |
|-----------|--------------------------------|-----------------------------------------------------|
| [X] 1B.1      | Biome noise generator          | Fails if known `(x, z)` returns wrong biome         |
| [X] 1B.2      | Heightmap sampler              | Fails if height doesn’t match reference             |
| [X] 1B.3      | River noise patch              | Fails if expected signal is missing                 |
| [X] 1B.4      | Patch assembler (x, z window)  | Fails if array shapes or padding are invalid        |
| [X] 1B.5      | Save `.npz` seed-only input    | Fails if file missing expected keys                 |

---

## 🧮 Phase 2 — LOD Patch Pairing

| TDD Cycle | Goal                         | RED Test Focus                                 |
|-----------|------------------------------|------------------------------------------------|
| [X] 2.1       | Assemble parent-child pairs  | Fails if LOD mismatch or alignment issue       |
| [X] 2.2       | Link with seed-derived input | Fails if biome/heightmap pairing is incorrect  |
| [X] 2.3       | Validate patch format        | Fails if any training example is malformed     |

---

## 📦 Phase 3 — Dataset Loader

| TDD Cycle | Goal                         | RED Test Focus                              |
|-----------|------------------------------|----------------------------------------------|
| [X] 3.1       | Load `.npz` training patch   | Fails if file missing or data misaligned     |
| [ ] 3.2       | PyTorch Dataset impl         | Fails if `__getitem__` returns bad shapes    |
| [ ] 3.3       | Batch collation              | Fails if PyTorch DataLoader returns wrong batch |

---

## 🧠 Phase 4 — Model Architecture

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [X] 4.1       | 3D U-Net instantiate         | Fails if network doesn't build with config       |
| [ ] 4.2       | Forward pass (8³→16³)        | Fails if logits or mask shapes are incorrect     |
| [ ] 4.3       | Conditioning via inputs      | Fails if biome/heightmap/y-index not used        |
| [ ] 4.4       | LOD timestep embedding       | Fails if output doesn't vary by timestep         |
---

## 🏋️ Phase 5 — Training Loop

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [ ] 5.1       | Dry run 1 epoch              | Fails if no gradient/backprop recorded           |
| [ ] 5.2       | Checkpoint saving            | Fails if `.pt` missing or corrupted              |
| [ ] 5.3       | Resume training              | Fails if epoch count doesn’t continue            |
| [ ] 5.4       | CSV or TensorBoard logs      | Fails if logs not written                        |

---

## 🧪 Phase 6 — Evaluation + Visualization

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [ ] 6.1       | Accuracy metrics             | Fails if mask/type accuracy is incorrect         |
| [ ] 6.2       | IoU / Dice scores            | Fails if overlap metrics are invalid             |
| [ ] 6.3       | 3D voxel visualization       | Fails if matplotlib renders are missing/blank    |

---

## 📤 Phase 7 — ONNX Export

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [ ] 7.1       | Export ONNX                  | Fails if `.onnx` missing                         |
| [ ] 7.2       | Validate ONNX vs PyTorch     | Fails if outputs diverge                         |
| [ ] 7.3       | ONNX shape tests             | Fails if runtime inference fails                 |

---

## 🚦 Phase 8 — Disk-Aware Batch Controller

| TDD Cycle | Goal                         | RED Test Focus                                  |
|-----------|------------------------------|--------------------------------------------------|
| [ ] 8.1       | Cap disk usage               | Fails if chunk cache exceeds max GB             |
| [ ] 8.2       | Delete old data              | Fails if temporary files remain after batch     |
| [ ] 8.3       | Region history logging       | Fails if same (x, z) reused in epoch            |
| [ ] 8.4       | Retry failed patches         | Fails if skipped examples aren’t retried        |

---

## 📘 References

- Training goals and input/output schema: see [`docs/TRAINING-OVERVIEW.md`](docs/TRAINING-OVERVIEW.md)
- Model architecture and LOD support strategy: see [`copilot-instructions.md`](.github/copilot-instructions.md)
