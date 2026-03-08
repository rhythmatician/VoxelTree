# Copilot Instructions — VoxelTree

> **Purpose:** Guide GitHub Copilot (and human contributors) to work within
> VoxelTree's single canonical pipeline. Last synced: 2026-03-08.

---

## RULE #1 — No New Files

**MODIFY existing files. NEVER create new ones** without explicit user approval.

This project has had repeated problems with Copilot generating new scripts
(`train_progressive_quick.py`, `run_training.py`, etc.) instead of editing the
canonical file. The result is a pile of dead code that must be deleted later.

If you think a new file is needed, **ask first**. The answer is almost always
"update the existing file instead."  Churn is preferred over cruft/proliferation.

### Canonical file list (exhaustive)

| Role | File | Notes |
|---|---|---|
| **Orchestrator** | `pipeline.py` | Subprocess phases: extract → column-heights → build-pairs → train → export → deploy |
| **Training** | `train.py` | THE training script. 4 progressive models. |
| **RCON CLI** | `data-cli.py` | freeze / pregen / status — RCON only, no training |
| **Extraction** | `scripts/extract_voxy_training_data.py` | Voxy RocksDB → NPZ |
| **Voxy reader** | `scripts/voxy_reader.py` | RocksDB reader |
| **Column heights** | `scripts/add_column_heights.py` | Post-extraction heightmap enrichment |
| **Pair builder** | `scripts/build_pairs.py` | NPZ → LOD transition pair NPZs |
| **Mipper** | `scripts/mipper.py` | Canonical LOD coarsening (Voxy algorithm) |
| **Class weights** | `scripts/compute_class_weights.py` | `--class-weights auto` support |
| **ONNX export** | `scripts/export_lod.py` | 4 ONNX models from checkpoint |
| **Dataset** | `train/multi_lod_dataset.py` | Pair cache + DataLoader |
| **Models** | `train/progressive_lod_models.py` | 4 progressive model architectures |
| **Conditioning** | `train/anchor_conditioning.py` | Height planes + biome fusion |
| **UNet** | `train/unet3d.py` | SimpleFlexibleUNet3D + SimpleFlexibleConfig |
| **Vocab** | `config/voxy_vocab.json` | 1102-entry block vocabulary |

Tests live in `tests/test_*.py`. Add tests to existing files or create
`tests/test_<module>.py` for NEW modules only.

**Everything else was deleted.** Do not recreate it.

---

## What NOT to create

These have been **deliberately removed**. Do not recreate or reference them:

- `config_multi_lod.yaml` — config lives in checkpoint now
- `x_router6` / router6 inputs — dropped; biome IS router6 output
- `train/losses.py`, `train/metrics.py`, `train/trainer.py`, `train/dataset.py` — consolidated into `train.py`
- `scripts/extraction/` — MCA extraction removed; Voxy only
- `scripts/worldgen/` — no longer needed
- `scripts/verify_onnx.py` — test vectors embedded in export
- `scripts/seed_inputs/` — dropped
- Any `train_*.py` or `run_*.py` variant at project root

---

## Project Purpose

VoxelTree trains a **progressive LOD voxel model** for vanilla Minecraft terrain.
The model runs in the **LODiffusion** mod to render distant terrain just-in-time.
LOD0 is NOT generated — vanilla terrain handles full resolution.

---

## Architecture — 4 Separate Progressive Models

| Step | Input | Output | Architecture |
|---|---|---|---|
| Init→LOD4 | conditioning only | 1×1×1 | tiny MLP |
| LOD4→LOD3 | 1³ parent | 2×2×2 | small Conv3D |
| LOD3→LOD2 | 2³ parent | 4×4×4 | medium Conv3D |
| LOD2→LOD1 | 4³ parent | 8×8×8 | medium-large Conv3D |

All models receive anchor conditioning:
- `x_height_planes`: [1, 5, 16, 16] float32 — surface, ocean_floor, slope_x, slope_z, curvature
- `x_biome`: [1, 16, 16] int64 — vanilla biome index
- `x_y_index`: [1] int64 — vertical 16-slab index

Refinement models additionally receive:
- `x_parent`: [1, 1, P, P, P] float32 — parent occupancy

All models output:
- `block_logits`: [1, N_blocks, D, D, D] float32
- `air_mask`: [1, 1, D, D, D] float32

Config is stored in checkpoints as `SimpleFlexibleConfig` — no external YAML.

---

## Canonical Pipeline

```
1. python data-cli.py pregen          # RCON: freeze world + Chunky pregen
2. /voxy import world <name>          # manual in-game command
3. python pipeline.py extract         # Voxy RocksDB → NPZ
4. python pipeline.py column-heights  # enrich NPZs with heightmap_surface/ocean_floor
5. python pipeline.py build-pairs     # NPZ → LOD transition pairs
6. python pipeline.py train           # train 4 progressive models
7. python pipeline.py export          # checkpoint → 4 ONNX files
8. python pipeline.py deploy          # copy to LODiffusion config dir
```

Or run steps 3-8 in one shot: `python pipeline.py run`

---

## Non-Negotiables

- **Terrain only**: `generate-structures=false` during worldgen
- **Deterministic inference**: same inputs → same outputs (CPU)
- **Static ONNX**: opset ≥ 17, NO dynamic axes
- **Truthful labels**: extract from Voxy RocksDB only — no synthetic fallbacks
- **No fallbacks**: every input is required. No Optional with fallback defaults.
- **No router6**: dropped entirely. Do not add router6/multinoise inputs.
- **heightmap_surface required**: every NPZ must have `heightmap_surface` and
  `heightmap_ocean_floor` (added by `scripts/add_column_heights.py`)

---

## Data Format

NPZ files in `data/voxy/` contain per-patch arrays. Required keys:

| Key | Shape | Dtype | Source |
|---|---|---|---|
| `labels16` | (16,16,16) | int16/int32 | Voxy block IDs |
| `biome_patch` | (16,16) | int16/int32 | vanilla biome index |
| `heightmap_patch` | (1,16,16) | float32 | section-level height |
| `y_index` | scalar | int | vertical slab index |
| `heightmap_surface` | (16,16) | float32 | column-level surface height |
| `heightmap_ocean_floor` | (16,16) | float32 | column-level ocean floor |

Block vocabulary: 1102 entries from `config/voxy_vocab.json`. Air = 0.

---

## Mipper (LOD Coarsening)

Single source of truth: `scripts/mipper.py`

- Opacity tiers: air=0, water/lava/glass/leaves=1, other solids=15
- Selection: `score = (opacity << 4) | corner_priority` — highest wins
- When all 8 corners equal opacity, corner I₁₁₁ wins (priority 7)

---

## Implementation Guardrails

- Python 3.11+; use `pathlib.Path`
- `rocksdict` + `zstandard` for Voxy RocksDB; `torch` + `numpy` for training
- Validate shapes/dtypes at boundaries
- No global mutable state
- No `Optional` with silent fallbacks — if data is missing, error or skip with message

---

## When editing code

1. **Find the canonical file** from the table above
2. **Modify it in place** — do not create a new file
3. **Update tests** in the corresponding `tests/test_*.py`
4. **Update this file** if the I/O contract or file list changes

---

## Tests

Existing test files:
- `tests/test_mipper.py` — Mipper invariants
- `tests/test_voxy_extraction.py` — Voxy reader + NPZ round-trip
- `tests/test_conditioning.py` — anchor conditioning
- `tests/test_forward_pass.py` — model forward pass shapes
- `tests/test_lod_embedding.py` — LOD embedding behavior
- `tests/test_unet3d.py` — UNet architecture

Run: `python -m pytest tests/ -q`
