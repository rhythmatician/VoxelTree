# 🌲 VoxelTree Project Outline — **Current Status (2025‑06‑05)**

> **TDD workflow legend**
> **[X]** = Cycle complete (RED → GREEN → REFACTOR)  **[ ]** = Not started  **[\~]** = In‑progress  **[🆕]** = Added or heavily modified since last outline

---

## ✅ Phase 0 — Developer Infrastructure

**Status:** Complete

| Cycle    | Goal                 | Notes                                                         |
| -------- | -------------------- | ------------------------------------------------------------- |
| [X] 0.1 | Repo scaffold        | `train/`, `scripts/`, `tests/`, `docs/` skeletons in place    |
| [X] 0.2 | CI + pre‑commit      | GitHub Actions matrix (Ubuntu & Windows) + `pre‑commit` hooks |
| [X] 0.3 | `config.yaml` loader | Centralised schema validation & env overrides                 |

---

## ✅ Phase 0B — Real Chunk Generation *(Vanilla 1.21.5 compatible)*

**Status:** Complete — 🆕 full refactor

Implemented headless Fabric server bootstrap plus Chunky pregeneration. Region decode now uses **`anvil` (package: `anvil‑parser2`)** with verified 1.18 + support. Outputs: 16³ tensors, downsampled 8³ parents, persisted as `.npz`.

| Cycle     | Goal                      | Result                                                           |
| --------- | ------------------------- | ---------------------------------------------------------------- |
| [X] 0B.1 | Headless chunkgen CLI     | `scripts/worldgen/bootstrap.py` spawns & scripts Fabric + Chunky |
| [X] 0B.2 | Validate `.mca` structure | Integrity tests ensure expected chunk sections exist             |
| [X] 0B.3 | Extract 16³ block arrays  | `scripts/extraction/chunk_extractor.py` converts NBT → numpy     |
| [X] 0B.4 | Downsample → 8³           | Verified pooling alignment unit tests                            |
| [X] 0B.5 | Save real‑data `.npz`     | Parity with seed‑only format confirmed                           |

---

## ✅ Phase 1 — `.mca` Chunk Extraction *(Legacy mock remains for regression)*

**Status:** Complete – superseded by 0B but kept for sanity tests.

| Cycle        | Goal                             | Result                                        |
| ------------ | -------------------------------- | --------------------------------------------- |
| [X] 1.1–1.5 | Mock extractor + multiproc batch | Tests still pass to guard against regressions |

---

## ✅ Phase 1B — Seed‑Based Input Generation

**Status:** Complete

Generates biome IDs, heightmap slices, river noise & patch coordinates purely from `(seed, x, z)` via `tools/voxeltree_cubiomes_cli/`. Output cached as `.npz`.

---

## ✅ Phase 2 — LOD Patch Pairing

**Status:** Complete

Parent 8³ + child 16³ + seed‑derived conditioning zipped into training samples; cross‑checked alignment tests.

---

## ✅ Phase 3 — Dataset Loader

**Status:** Complete (Phase 3.1 REF factor finished)

`train/dataset.py` + custom collator support lazy NPZ loading, optional RAM cache, and full type hints. All data‑shape validation tests pass.

---

## ✅ Phase 4 — Model Architecture

**Status:** Complete

`train/unet3d.py` implements multichannel 3‑D U‑Net with dual heads (block logits, air mask) and integrated conditioning (biome, height, river, LOD positional encoding).

| Cycle    | Goal                    |
| -------- | ----------------------- |
| [X] 4.1 | Instantiate network     |
| [X] 4.2 | Forward pass (8³ → 16³) |
| [X] 4.3 | Conditioning inputs     |
| [X] 4.4 | LOD timestep embedding  |

---

## ✅ Phase 5 — Training Loop

**Status:** Complete

`train/trainer.py` handles epoch loop, gradient step (`train/step.py`), checkpoint save/resume, CSV & TensorBoard logging (`train/logger.py`). End‑to‑end dry‑run integration test passes.

| Cycle    | Goal               |
| -------- | ------------------ |
| [X] 5.1 | One‑epoch dry run  |
| [X] 5.2 | Checkpoint saving  |
| [X] 5.3 | Resume training    |
| [X] 5.4 | Logging (CSV + TB) |

---

## 🧪 Phase 6 — Evaluation & Visualization

**Status:** [ ] Not started

| Planned Cycle | Goal                                 |
| ------------- | ------------------------------------ |
| 6.1           | Accuracy metrics (mask & block type) |
| 6.2           | IoU / Dice scores                    |
| 6.3           | 3‑D voxel render previews            |

---

## 📤 Phase 7 — ONNX Export

**Status:** [ ] Not started

| Planned Cycle | Goal                          |
| ------------- | ----------------------------- |
| 7.1           | Export to ONNX                |
| 7.2           | PyTorch vs ONNX parity tests  |
| 7.3           | Static‑shape compliance check |

---

## 🚦 Phase 8 — Disk‑Aware Batch Controller

**Status:** [ ] Not started

| Planned Cycle | Goal                            |
| ------------- | ------------------------------- |
| 8.1           | Cap disk usage during chunk gen |
| 8.2           | Auto‑purge old batches          |
| 8.3           | Generation history tracking     |
| 8.4           | Retry failed patch extracts     |

---

## 📌 Immediate Next Steps

1. Kick off Phase 6: draft RED tests for metrics & renders.
2. Prototype lightweight voxel visualiser (matplotlib ↔ trame) for QA.
3. Begin ONNX export early—catch unsupported ops ASAP.
4. Draft spec for Phase 8 (likely SQLite state + daemonized worker).

---

*Outline refreshed on **2025‑06‑05** based on branch `feat‑headless‑chunk‑maker`.*
