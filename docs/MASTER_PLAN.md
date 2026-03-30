# VoxelTree / LODiffusion — Master Plan

**Version:** 1.0  
**Date:** June 2025  
**Status:** Active  

> A four-phase progressive training strategy for an ML-driven LOD terrain
> generator that mirrors Minecraft 1.21's vanilla NoiseRouter pipeline.
> Each phase builds on the previous one: parallel feature extraction →
> density combination → spatial upscaling → end-to-end fine-tuning.

### Current status (March 2026)

- The active pipeline is now centered around a **sparse-octree model** ("voxy") that replaces the legacy init/refine/leaf octree training strategy.
- The system supports a **GPU "shadow router" compute path** (GLSL + SSBO + Voxy request queue) for fast LOD generation, while retaining the option for CPU ONNX inference.
- The data pipeline is now: **dumpnoise → NPZ cache → training/distillation → ONNX export → runtime inference**. This flow is implemented between `LODiffusion` and `VoxelTree` via the new `dumpnoise sparse_octree` exporter, Python training scripts, and the ONNX runtime decoder.

---

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1 — Parallel Feature Networks](#3-phase-1--parallel-feature-networks)
   - [Phase 1A — Macro-Shape Net](#31-phase-1a--macro-shape-net)
   - [Phase 1B — Climate & Biome Net](#32-phase-1b--climate--biome-net)
   - [Phase 1C — Subtractive Net](#33-phase-1c--subtractive-net)
   - [Parallel Execution Strategy](#34-parallel-execution-strategy)
4. [Phase 2 — finalDensity Combiner](#4-phase-2--finaldensity-combiner)
5. [Phase 3 — Cell-to-Block Upscaler](#5-phase-3--cell-to-block-upscaler)
6. [Phase 4 — End-to-End Fine-Tuning](#6-phase-4--end-to-end-fine-tuning)
7. [Integration with OGN Octree Pipeline](#7-integration-with-ogn-octree-pipeline)
8. [Data Pipeline](#8-data-pipeline)
9. [Hardware & Timeline](#9-hardware--timeline)
10. [Risk Register & Mitigations](#10-risk-register--mitigations)
11. [Success Criteria](#11-success-criteria)
12. [References](#12-references)

---

## 1. Project Vision

LODiffusion generates **distant LOD terrain only** using lightweight ONNX
models for CPU inference. Vanilla Minecraft remains authoritative for
playable-resolution terrain (LOD 0). The system:

- Operates as a **render proxy** — believable approximations where players
  can only see, exact vanilla where they interact.
- Uses a **progressive octree** aligned with Voxy's `WorldSection` hierarchy
  (L4 → L1, skipping empty subtrees).
- Anchors all macro-structure in **vanilla noise functions** (heightmap, biome,
  y-position) so generated terrain is seed-stable and seamless.
- **Never samples any input at higher resolution than vanilla Minecraft** —
  L0 uses exactly vanilla's native grid; L1–L4 use a strict subset (fewer
  channels, equal or lower resolution).
- Targets **< 100 ms** per inference call on consumer CPUs.

The four-phase training plan described here produces the lightweight feature
extraction backbone that powers those ONNX models.

### Why Four Phases?

Minecraft's terrain generation is a **deeply nested composition** of
density functions. Training a single monolithic network to reproduce
`finalDensity` from raw Perlin octaves would require enormous depth, data,
and compute. Instead we decompose the problem:

```
Phase 1 (parallel)             Phase 2         Phase 3          Phase 4
┌──────────────────┐
│ 1A Macro-Shape   │─────┐
│ (continents,     │     │
│  erosion, ridges)│     │    ┌──────────┐    ┌──────────┐    ┌──────────┐
└──────────────────┘     ├───▶│ Combine  │───▶│ Upsample │───▶│ Fine-    │
┌──────────────────┐     │    │ Density  │    │ 4×4×8 →  │    │ Tune All │
│ 1B Climate/Biome │─────┤    │ (frozen  │    │ 16×16×32 │    │ (unfreeze│
│ (temp, veg,      │     │    │  1A/B/C  │    │ (deconv) │    │  tiny LR)│
│  depth, biome)   │     │    │  heads)  │    └──────────┘    └──────────┘
└──────────────────┘     │    └──────────┘
┌──────────────────┐     │
│ 1C Subtractive   │─────┘
│ (caves, aquifers)│
└──────────────────┘
```

Each phase trains faster and converges more reliably because the
optimisation surface is narrower and the ground truth is well-defined.

---

## 2. Architecture Overview

### 2.1 Minecraft's Terrain Pipeline (The Ground Truth)

Minecraft 1.21 generates terrain through its **NoiseRouter**, a directed
acyclic graph of `DensityFunction` nodes:

```
Raw Perlin Noise Octaves
  ├─ continentalness (2 octaves → continent shape)
  ├─ erosion         (3 octaves → flat vs mountainous)
  ├─ ridges          (2 octaves → peaks / valleys)
  ├─ temperature     (T octaves → hot vs cold)
  ├─ vegetation      (V octaves → lush vs dry)
  ├─ cave_large      (3 octaves → large cave structure)
  ├─ cave_small      (3 octaves → small cave detail)
  └─ aquifer         (2 octaves → water table)
         │
         ▼
  Intermediate Density Functions
  ├─ initialDensityWithoutJaggedness
  ├─ slopedCheese       ← terrain shape before caves
  ├─ depth / factor
  └─ temperature/vegetation → biome assignment
         │
         ▼
  finalDensity           ← single scalar per cell:
                            > 0 = solid, ≤ 0 = air/fluid
         │
         ▼
  Trilinear Interpolation  (4×4×8 cell → 16×16×32 blocks)
         │
         ▼
  SurfaceRules / Block Placement
```

**Critical insight:** The NoiseRouter operates at **cell resolution**
(4 × 4 × 8 = 768 cells per chunk), not full block resolution
(16 × 16 × 384 = 98,304 blocks). Our Phase 1 & 2 networks mirror
this — they learn **cell-level** density, not per-block predictions.

### 2.2 Resolution Map

| Component | Grid | Points / Chunk | Resolution |
|-----------|------|----------------|------------|
| Raw Perlin octaves | varies | varies | Input noise |
| Legacy feature networks (deprecated) | 4 × 48 × 4 | 768 | Cell-level (XY plane + full Y) |
| Legacy climate network (deprecated) | 4 × 4 × 4 | 64 | Cell-level (reduced Y — climate is XZ-dominant) |
| Phase 2 combiner | 4 × 48 × 4 | 768 | Cell-level |
| Phase 3 upscaler | 16 × 384 × 16 | 98,304 | Block-level |
| OGN octree models | 32 × 32 × 32 | 32,768 | Voxy WorldSection |

### 2.3 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Ground Truth Source                       │
│  Live Minecraft server + dumpnoise mod                      │
│  Extracts: per-cell Perlin octaves + NoiseRouter outputs    │
└────────────────────────┬────────────────────────────────────┘
                         │ JSON → NPZ conversion
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Training Data (NPZ)                       │
│  terrain_shaper_density_data.npz  inputs: (N, 4, 48, 4, 11)         │
│                           outputs: (N, 4, 48, 4, 1)         │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Legacy Phase 1 Feature Network Pipeline (Deprecated)

The original **Phase 1A/B/C** parallel feature extraction pipeline (macro-shape, climate/biome, and subtractive carving networks) is considered obsolete and is no longer used in the active training pipeline.

The current strategy focuses on learning the final density function directly (Stage 1 density model), which provides a single, stable target and simplifies the training and inference flow.

--

## 4. Phase 2 — finalDensity Combiner

**Goal:** Given the frozen outputs of Phase 1A, 1B, and 1C, learn the
**combination function** that produces Minecraft's `finalDensity` scalar.

**Why this is a separate phase:** In vanilla Minecraft, `finalDensity` is
computed by a deeply nested DAG that blends continentalness, erosion,
ridges, depth, factor, slopedCheese, and cave/aquifer masks. Rather than
training the entire graph end-to-end (difficult optimisation surface), we
let Phase 1 networks first learn each sub-function perfectly, then train
a small combiner that learns only the blending rules.

### 4.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 2 Network                           │
│                                                              │
│  Phase 1A output (frozen)  ──→ (B, 4, 48, 4, 3)             │
│  Phase 1B output (frozen)  ──→ (B, 4,  4, 4, 4) ── Upsample │
│                                   Y to 48 via      │        │
│                                  nearest repeat   ──┘        │
│  Phase 1C output (frozen)  ──→ (B, 4, 48, 4, 4)             │
│                                                              │
│        [Concatenate along channel axis]                      │
│              ↓                                               │
│        (B, 4, 48, 4, 11)    ← 3 + 4 + 4 = 11 channels       │
│              ↓                                               │
│  Conv3D(11 → 64, 3³) + ReLU                                 │
│              ↓                                               │
│  Conv3D(64 → 128, 3³) + ReLU                                │
│              ↓                                               │
│  Conv3D(128 → 64, 3³) + ReLU                                │
│              ↓                                               │
│  Conv3D(64 → 1, 3³)          ← linear output                │
│              ↓                                               │
│        (B, 4, 48, 4, 1)     ← finalDensity at cell level    │
│              ↓                                               │
│  Threshold: > 0 = solid, ≤ 0 = air/fluid                    │
└──────────────────────────────────────────────────────────────┘
```

**Y-dimension handling for Phase 1B:** The climate network outputs
(B, 4, 4, 4, 4) with only 4 Y-samples. Before concatenation, repeat
each Y-level 12× along the Y axis to produce (B, 4, 48, 4, 4). This
is justified because climate barely varies with altitude.

### 4.2 Input Tensor

```
Shape:    (batch, 4, 48, 4, 11)
dtype:    float32
```

| Channels | Source | Description |
|----------|--------|-------------|
| 0–2 | Phase 1A (frozen) | continentalness, erosion, ridges |
| 3–6 | Phase 1B (frozen, Y-upsampled) | temperature, vegetation, depth, biome_confidence |
| 7–10 | Phase 1C (frozen) | air_prob, water_prob, cave_uncertainty, carver_influence |

### 4.3 Output Tensor

```
Shape:    (batch, 4, 48, 4, 1)
dtype:    float32
Range:    unbounded (~-2 to +2 typical)
```

| Ch | Target | Ground Truth |
|----|--------|-------------|
| 0 | `finalDensity` | `router.finalDensity().compute(ctx)` |

### 4.4 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss | MSE | Continuous regression |
| Optimizer | Adam | |
| Learning rate | 5e-4 | Lower than Phase 1 — smaller network, fine blending |
| Batch size | 128 | |
| Epochs | 20–40 | Early stopping at MSE < 0.0005 |
| Phase 1 weights | **Frozen** | `requires_grad = False` for all Phase 1 params |
| New parameters | ~50 K | Only the combiner layers are trainable |

### 4.5 Grok Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| MSE | < 0.0005 | Tighter than Phase 1 — compound error budget |
| Binary accuracy | > 99.5% | Correct solid/air classification at threshold 0 |
| Surface MSE | < 0.0002 | Measured only at cells near the density=0 surface |

**Surface MSE** is the most important metric. A small error deep
underground or high in the air is invisible; a small error at the terrain
surface creates visible misplacement. We compute it by masking to cells
where |finalDensity| < 0.1.

### 4.6 Why Phase 2 Is Fast

- **Only ~50 K trainable parameters** (Phase 1's ~500 K are frozen)
- **Pre-learned features**: The combiner doesn't need to discover what
  "erosion" or "cave" means — those concepts are already represented
  as clean scalar outputs from Phase 1.
- **Expected convergence:** 2–5 hours on GPU, 10–20 hours on CPU.
- **Lower overfitting risk:** Fewer parameters + rich features →
  better generalisation.

### 4.7 Integration Code Pattern

```

Wall-Clock Timeline (GPU)

 Hour │  GPU #0 (1A)     GPU #1 (1B)      GPU #2 (1C)
──────┼──────────────────────────────────────────────────
  0   │  ▓ START         ▓ START           ▓ START
  1   │  ▓               ▓                 ▓
  2   │  ▓               ▓ DONE ✓          ▓
  3   │  ▓               (idle)            ▓
  4   │  ▓                                 ▓ DONE ✓
  5   │  ▓                                 (idle)
  …   │  ▓
 10   │  ▓ DONE ✓
──────┴──────────────────────────────────────────────────
Total wall-clock: ~10 h (GPU) or ~47 h (sequential CPU)
```

| Phase | GPU Time | CPU Time | Grok Gate |
|-------|----------|----------|-----------|
| 1A | ~3–10 h | ~20–30 h | MSE < 0.001 |
| 1B | ~0.5–2 h | ~3–5 h | Biome acc > 95% |
| 1C | ~4–10 h | ~25–40 h | IoU > 0.75 |
| **Parallel** | **~10 h** | **N/A** | |
| **Sequential** | **~18 h** | **~47 h** | |

**With Float16 + Positional Encoding:** ~15–20% additional speedup from
faster convergence and halved memory bandwidth.

---

## 4. Phase 2 — finalDensity Combiner

**Goal:** Given the frozen outputs of Phase 1A, 1B, and 1C, learn the
**combination function** that produces Minecraft's `finalDensity` scalar.

**Why this is a separate phase:** In vanilla Minecraft, `finalDensity` is
computed by a deeply nested DAG that blends continentalness, erosion,
ridges, depth, factor, slopedCheese, and cave/aquifer masks. Rather than
training the entire graph end-to-end (difficult optimisation surface), we
let Phase 1 networks first learn each sub-function perfectly, then train
a small combiner that learns only the blending rules.

### 4.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 2 Network                           │
│                                                              │
│  Phase 1A output (frozen)  ──→ (B, 4, 48, 4, 3)             │
│  Phase 1B output (frozen)  ──→ (B, 4,  4, 4, 4) ── Upsample │
│                                   Y to 48 via      │        │
│                                  nearest repeat   ──┘        │
│  Phase 1C output (frozen)  ──→ (B, 4, 48, 4, 4)             │
│                                                              │
│        [Concatenate along channel axis]                      │
│              ↓                                               │
│        (B, 4, 48, 4, 11)    ← 3 + 4 + 4 = 11 channels       │
│              ↓                                               │
│  Conv3D(11 → 64, 3³) + ReLU                                 │
│              ↓                                               │
│  Conv3D(64 → 128, 3³) + ReLU                                │
│              ↓                                               │
│  Conv3D(128 → 64, 3³) + ReLU                                │
│              ↓                                               │
│  Conv3D(64 → 1, 3³)          ← linear output                │
│              ↓                                               │
│        (B, 4, 48, 4, 1)     ← finalDensity at cell level    │
│              ↓                                               │
│  Threshold: > 0 = solid, ≤ 0 = air/fluid                    │
└──────────────────────────────────────────────────────────────┘
```

**Y-dimension handling for Phase 1B:** The climate network outputs
(B, 4, 4, 4, 4) with only 4 Y-samples. Before concatenation, repeat
each Y-level 12× along the Y axis to produce (B, 4, 48, 4, 4). This
is justified because climate barely varies with altitude.

### 4.2 Input Tensor

```
Shape:    (batch, 4, 48, 4, 11)
dtype:    float32
```

| Channels | Source | Description |
|----------|--------|-------------|
| 0–2 | Phase 1A (frozen) | continentalness, erosion, ridges |
| 3–6 | Phase 1B (frozen, Y-upsampled) | temperature, vegetation, depth, biome_confidence |
| 7–10 | Phase 1C (frozen) | air_prob, water_prob, cave_uncertainty, carver_influence |

### 4.3 Output Tensor

```
Shape:    (batch, 4, 48, 4, 1)
dtype:    float32
Range:    unbounded (~-2 to +2 typical)
```

| Ch | Target | Ground Truth |
|----|--------|-------------|
| 0 | `finalDensity` | `router.finalDensity().compute(ctx)` |

### 4.4 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss | MSE | Continuous regression |
| Optimizer | Adam | |
| Learning rate | 5e-4 | Lower than Phase 1 — smaller network, fine blending |
| Batch size | 128 | |
| Epochs | 20–40 | Early stopping at MSE < 0.0005 |
| Phase 1 weights | **Frozen** | `requires_grad = False` for all Phase 1 params |
| New parameters | ~50 K | Only the combiner layers are trainable |

### 4.5 Grok Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| MSE | < 0.0005 | Tighter than Phase 1 — compound error budget |
| Binary accuracy | > 99.5% | Correct solid/air classification at threshold 0 |
| Surface MSE | < 0.0002 | Measured only at cells near the density=0 surface |

**Surface MSE** is the most important metric. A small error deep
underground or high in the air is invisible; a small error at the terrain
surface creates visible misplacement. We compute it by masking to cells
where |finalDensity| < 0.1.

### 4.6 Why Phase 2 Is Fast

- **Only ~50 K trainable parameters** (Phase 1's ~500 K are frozen)
- **Pre-learned features**: The combiner doesn't need to discover what
  "erosion" or "cave" means — those concepts are already represented
  as clean scalar outputs from Phase 1.
- **Expected convergence:** 2–5 hours on GPU, 10–20 hours on CPU.
- **Lower overfitting risk:** Fewer parameters + rich features →
  better generalisation.

### 4.7 Integration Code Pattern

```python
class Phase2Combiner(nn.Module):
    """Trainable combiner on top of frozen Phase 1 feature extractors."""

    def __init__(self, phase1a_ckpt, phase1b_ckpt, phase1c_ckpt):
        super().__init__()

        # Load and freeze Phase 1 networks
        self.net_1a = load_and_freeze(phase1a_ckpt)   # → (B,4,48,4,3)
        self.net_1b = load_and_freeze(phase1b_ckpt)   # → (B,4, 4,4,4)
        self.net_1c = load_and_freeze(phase1c_ckpt)   # → (B,4,48,4,4)

        # Trainable combiner head
        self.combine = nn.Sequential(
            nn.Conv3d(11, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 1, 3, padding=1),
        )

    def forward(self, inputs_1a, inputs_1b, inputs_1c):
        with torch.no_grad():
            feat_a = self.net_1a(inputs_1a)                   # (B,4,48,4,3)
            feat_b = self.net_1b(inputs_1b)                   # (B,4, 4,4,4)
            feat_c = self.net_1c(inputs_1c)                   # (B,4,48,4,4)

        # Upsample 1B's Y dimension: 4 → 48 via nearest repeat
        feat_b = feat_b.repeat_interleave(12, dim=2)          # (B,4,48,4,4)

        combined = torch.cat([feat_a, feat_b, feat_c], dim=-1)  # (B,4,48,4,11)

        # Permute to (B, C, X, Y, Z) for Conv3D
        combined = combined.permute(0, 4, 1, 2, 3)           # (B,11,4,48,4)
        density = self.combine(combined)                       # (B, 1,4,48,4)

        return density.permute(0, 2, 3, 4, 1)                # (B,4,48,4,1)
```

---

## 5. Phase 3 — Cell-to-Block Upscaler

**Goal:** Take the cell-level `finalDensity` grid (4 × 48 × 4 per chunk)
produced by Phase 2 and upsample it to full block resolution
(16 × 384 × 16), mirroring Minecraft's trilinear interpolation.

### 5.1 Why Not Just Trilinear Interpolation?

In vanilla Minecraft, the cell-to-block expansion is pure trilinear
interpolation — a deterministic, non-learned operation. We have two
options:

| Approach | Pros | Cons |
|----------|------|------|
| **Hard-coded trilinear** | Exact parity, zero training | Rigid; can't learn corrections for accumulated Phase 1–2 errors |
| **Learned deconvolution** | Can compensate for upstream errors; learns surface sharpening | Requires training data; adds parameters |

**Decision: Learned deconvolution** — with trilinear interpolation as
the initialisation (warm-start). This gives us the best of both worlds:
we start from exact parity, and the fine-tuning in Phase 4 can adjust
the upsampler to compensate for any residual Phase 1–2 errors.

### 5.2 Architecture

```
Input: (B, 1, 4, 48, 4)         ← cell-level finalDensity from Phase 2

Stage 1 — Y expansion (48 → 384):
  ConvTranspose3d(1, 16, kernel=(1,8,1), stride=(1,8,1))
  GroupNorm(4, 16) + ReLU
  → (B, 16, 4, 384, 4)

Stage 2 — XZ expansion (4 → 16):
  ConvTranspose3d(16, 32, kernel=(4,1,4), stride=(4,1,4))
  GroupNorm(8, 32) + ReLU
  → (B, 32, 16, 384, 16)

Stage 3 — Refinement:
  Conv3D(32, 16, kernel=3³, padding=1)
  GroupNorm(4, 16) + ReLU
  Conv3D(16, 1, kernel=3³, padding=1)
  → (B, 1, 16, 384, 16)

Output: (B, 1, 16, 384, 16)     ← block-level density field
```

**Weight initialisation:** The transposed convolution kernels are
initialised to approximate trilinear interpolation. This means the
network starts by perfectly reproducing vanilla Minecraft's expansion
and only diverges during Phase 4 fine-tuning.

### 5.3 Input/Output Tensors

**Input:**
```
Shape:    (batch, 1, 4, 48, 4)
dtype:    float32
Source:   Phase 2 combiner output (frozen)
```

**Output:**
```
Shape:    (batch, 1, 16, 384, 16)
dtype:    float32
Range:    unbounded (density field)
Threshold: > 0 = solid, ≤ 0 = air
```

### 5.4 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss | MSE + Surface Smoothness Penalty | |
| Surface penalty | λ = 0.1 × Laplacian of density at surface | Penalises staircase artifacts |
| Optimizer | Adam | |
| Learning rate | 1e-4 | Very conservative — warm-started from trilinear |
| Batch size | 16 | Large tensors (16×384×16 each) |
| Epochs | 10–20 | |
| Phase 1+2 weights | **Frozen** | |

### 5.5 Ground Truth

The ground truth for Phase 3 is obtained by running vanilla Minecraft's
own trilinear interpolation on the ground-truth cell-level density:

```python
def generate_phase3_ground_truth(cell_density, chunk_x, chunk_z):
    """
    Expand cell-level density to block-level via vanilla trilinear interp.
    
    Args:
        cell_density: (4, 48, 4) cell-level finalDensity values
        chunk_x, chunk_z: chunk coordinates (for potential edge effects)
    
    Returns:
        block_density: (16, 384, 16) block-level density
    """
    block_density = np.zeros((16, 384, 16), dtype=np.float32)
    
    for bx in range(16):
        for by in range(384):
            for bz in range(16):
                # Find enclosing cell and interpolation weights
                cx = bx / 4.0    # [0, 4) → cell index
                cy = (by + 64) / 8.0  # block Y → cell Y
                cz = bz / 4.0
                
                # Trilinear interpolation between 8 nearest cell corners
                block_density[bx, by, bz] = trilinear_interp(
                    cell_density, cx, cy, cz
                )
    
    return block_density
```

### 5.6 Grok Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| MSE vs trilinear | < 0.0001 | Nearly identical to vanilla interpolation |
| Surface Laplacian | < 0.05 | No staircase artifacts on terrain slopes |
| Chunk-boundary continuity | < 0.001 | No visible seams between adjacent chunks |

### 5.7 Dual-Head Extension (Optional)

If the OGN dual-head redesign is active (density field + 12-class material
classification), Phase 3's upscaler produces two outputs:

```
Output Head A: block_density  (B, 1, 16, 384, 16)  — float32
Output Head B: block_material (B, 12, 16, 384, 16) — logits

Total Loss = λ₁ · MSE(density) + λ₂ · CrossEntropy(material[density>0])
           = 0.8 · MSE + 0.2 · CE
```

Material classification is masked to only where density > 0 (solid
voxels). Air voxels have no material. This mirrors the dual-head
approach described in `OGN_DUAL_HEAD_REDESIGN.md`.

---

## 6. Phase 4 — End-to-End Fine-Tuning

**Goal:** Unfreeze all weights across the entire pipeline
(Phase 1A/1B/1C → Phase 2 → Phase 3) and perform a final training pass
with a very small learning rate. This allows backpropagation to make
micro-adjustments that iron out edge cases where the independently-trained
components interact poorly.

### 6.1 Motivation

After Phases 1–3, the pipeline is assembled from components that were each
trained to match their *individual* ground truth. But compound errors
accumulate:

- Phase 1A's ±0.03 error on `erosion` →
- Phase 2's `finalDensity` is slightly off near mountain ridges →
- Phase 3's upsampled surface is shifted by 1–2 blocks

Phase 4 lets gradient flow from the final output all the way back through
Phase 1, allowing each layer to make sub-percent adjustments that minimise
the **end-to-end** error rather than each component's error individually.

### 6.2 Architecture

No new layers are added. The entire pipeline from Phase 1 inputs to
Phase 3 output is treated as a single end-to-end model:

```
Legacy Feature Net A ──┐
Legacy Feature Net B ──┼──→ Phase 2 Combiner ──→ Phase 3 Upscaler ──→ block_density
Legacy Feature Net C ──┘

All weights unfrozen. All gradients flow.
```

### 6.3 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss | End-to-end MSE on block-level density | Same as Phase 3 |
| Optimizer | Adam | |
| Learning rate | **1e-5** | 100× smaller than Phase 1 — micro-adjustments only |
| Batch size | 8 | Full pipeline in memory = large footprint |
| Epochs | 5–10 | Diminishing returns after ~5 |
| All weights | **Unfrozen** | `requires_grad = True` for everything |
| Gradient clipping | max_norm = 1.0 | Prevent catastrophic forgetting |

### 6.4 Differential Learning Rates

To prevent Phase 4 from destroying the well-learned Phase 1 features,
use **per-layer-group learning rates**:

```python
optimizer = torch.optim.Adam([
    {"params": model.feature_net_a.parameters(), "lr": 1e-6},  # Barely touch
    {"params": model.feature_net_b.parameters(), "lr": 1e-6},  # Barely touch
    {"params": model.feature_net_c.parameters(), "lr": 1e-6},  # Barely touch
    {"params": model.phase_2.parameters(),  "lr": 5e-6},  # Small tweaks
    {"params": model.phase_3.parameters(),  "lr": 1e-5},  # Most adjustment here
])
```

**Justification:** Phase 1 networks are already grokked — their features
are nearly perfect. Let them drift the least. Phase 3 (upscaler) benefits
most from end-to-end signal because its initial weights are a crude
trilinear approximation.

### 6.5 Catastrophic Forgetting Prevention

- **Gradient clipping** (max_norm = 1.0) prevents sudden large updates.
- **EWC (Elastic Weight Consolidation)** penalty (optional): For each
  Phase 1 parameter θ, add a penalty λ · F(θ) · (θ − θ*_frozen)² to the
  loss, where F(θ) is the Fisher information and θ* is the Phase 1
  checkpoint value. This penalises moving weights that were important for
  Phase 1 accuracy.
- **Validation monitoring:** If Phase 1A's MSE rises above 0.002 during
  Phase 4, reduce that layer group's LR by 10× or freeze it entirely.

### 6.6 Grok Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Block-level MSE | < 0.0001 | Combined pipeline accuracy |
| Surface MAE (blocks) | < 0.5 | Average surface height error < half a block |
| Biome-boundary IoU | > 0.90 | Terrain transitions match vanilla at biome edges |
| Legacy feature MSE | < 0.002 | No catastrophic forgetting on legacy feature extractors |

### 6.7 Output

After Phase 4 training, the complete pipeline is exported to ONNX:

```python
# Export the full pipeline
full_model = Phase4EndToEnd(
    legacy_feature_a, legacy_feature_b, legacy_feature_c,
    phase_2_combiner,
    phase_3_upscaler
)

torch.onnx.export(
    full_model,
    (dummy_inputs_a, dummy_inputs_b, dummy_inputs_c),
    "terrain_backbone.onnx",
    opset_version=17,
    input_names=["perlin_macro", "perlin_climate", "perlin_carvers"],
    output_names=["block_density"],
    dynamic_axes={"perlin_macro": {0: "batch"}, ...},
)
```

---

## 7. Integration with OGN Octree Pipeline

The four-phase backbone described above produces a **block-level density
field**. The OGN (Octree Generation Network) pipeline consumes this to
produce the final Voxy-compatible `WorldSection` predictions.

### 7.1 How They Connect

```
Phase 1–4 Backbone                 OGN Octree Pipeline
─────────────────                  ────────────────────
Perlin octaves →                   
  Legacy feature nets →             
    Phase 2 combiner →             
      Phase 3 upscaler →           
        block_density (16×384×16)  →  octree_init (L4)
                                        ↓ occ_logits
                                      octree_refine (L3, L2, L1)
                                        ↓ occ_logits
                                      octree_leaf (L0)
                                        ↓
                                      block_logits (32³) → Voxy
```

**In practice**, the four-phase backbone may be folded into the OGN
models' conditioning inputs rather than existing as a separate inference
call. The backbone's density field becomes an additional anchor channel
alongside heightmap and biome:

```
OGN model inputs:
  heightmap:    [N, 5, 32, 32]     ← from vanilla noise (cheap)
  biome:        [N, 32, 32]        ← from vanilla noise (cheap)
  y_position:   [N]                ← section coordinate
  density_field: [N, 1, 32, 32, 32] ← from Phase 1–4 backbone (new)
  parent_blocks: [N, 32, 32, 32]   ← from parent section (L3+ only)
```

### 7.2 Runtime Deployment (LODiffusion Mod)

At runtime in the Java mod, two strategies are possible:

**Strategy A — Pre-computed backbone:**  
Run the Phase 1–4 ONNX model once per chunk to produce the density
field, cache the result, then feed it to OGN init/refine/leaf calls
as an additional input.

**Strategy B — Baked into OGN weights:**  
During training, the OGN models are trained with ground-truth density
as an input. At export time, the backbone is fused into the first OGN
model (octree_init). This eliminates one ONNX call but makes the init
model larger.

**Current plan:** Strategy A for the demo milestone (simpler, easier to
debug), migrate to Strategy B for production (fewer inference calls at
runtime).

**Runtime LODiffusion direction (top-down, distance-gated):**

- L4 init-only model across full radius (1 ONNX call per 32³ section), write directly
  to Voxy L4 sections via `writeFullWorldSection` and set `nonEmptyChildren`.
- Mid-range: refine L3/L2/L1 only where view distance requires; each level expansion
  is an independent `octree_refine` call with parent context.
- Near: leaf L0 model (`octree_leaf`) only for visible/chunk-update sections.
- Avoid `mipSection()` for generated sections; `mipSection` remains the ingestion path
  for full L0 vanilla chunks.

This supports continuous migration from dense “bottom-up full L0” to efficient
“top-down progressive refinement” and enables large-radius DH rendering with
manageable ONNX call counts.

---

## 8. Data Pipeline

### 8.1 Extraction Flow

```
                  Minecraft Server (Fabric 1.21.11)
                  + dumpnoise mod
                  + Chunky (pregen)
                  + Carpet (tick freeze)
                            │
                            ▼
              /dumpnoise RCON command
              For each cell in pregen'd chunks:
                - Sample all Perlin octaves
                - Evaluate NoiseRouter intermediate functions
                - Record finalDensity
                            │
                            ▼
              noise_dumps/<profile>/*.json
              Raw per-cell noise + ground truth
                            │
                            ▼
              terrain_shaper_data_extraction.py
              JSON → NPZ conversion + tensor assembly
                            │
                            ▼
              terrain_shaper_density_data.npz
              (N, 4, 48, 4, 11) inputs
              (N, 4, 48, 4, 1)  finalDensity
```

### 8.2 Supplementary Data (OGN Training)

OGN models also need Voxy-format training data for the octree refinement
stages. This uses a separate extraction pipeline:

```
Minecraft Server → Voxy /voxy import world → RocksDB
             ↓
scripts/extract_voxy_training_data.py
             ↓
data/voxy/<world>_<coords>.npz
  - labels16:       (16,16,16) int16     block IDs
  - biome_patch:    (16,16)    int64     biome index
  - heightmap_patch: (16,16)   float32   normalised heights
  - y_index:        int                  section Y coordinate
```

### 8.3 Data Requirements

| Stage | Samples | Disk | Generation Time |
|-------|---------|------|-----------------|
| Stage 1 (legacy feature extraction) | 1,000–10,000 chunks | ~150–1500 MB | ~30–60 min |
| Stage 2 (finalDensity combiner) | Same chunks (finalDensity labels) | ~100–1000 MB | ~20 min |
| 3 | Same chunks (trilinear expansion) | ~500–5000 MB | ~1 h |
| OGN | 53,000+ sub-blocks from Voxy | ~5 GB | ~2 h |

**Seed strategy:** Train on 3+ seeds (e.g. 12345, -1, 999999) for
generalisation. Validate on held-out seeds.

---

## 9. Hardware & Timeline

### 9.1 Hardware Profiles

**Profile A — Multi-GPU (optimal)**

| Resource | Spec |
|----------|------|
| GPUs | 3× RTX 3060+ (4 GB VRAM minimum) |
| RAM | 16 GB system |
| Disk | 50 GB SSD for data + checkpoints |
| Phase 1 | ~10 h parallel |
| Phase 2 | ~3 h |
| Phase 3 | ~2 h |
| Phase 4 | ~4 h |
| **Total** | **~19 h** |

**Profile B — Single GPU**

| Resource | Spec |
|----------|------|
| GPU | 1× RTX 3060+ |
| Phase 1 | ~18 h sequential |
| Phases 2–4 | ~9 h |
| **Total** | **~27 h** |

**Profile C — CPU only ("the potato")**

| Resource | Spec |
|----------|------|
| CPU | Any modern 4+ core |
| Phase 1 | ~47 h sequential |
| Phases 2–4 | ~30 h |
| **Total** | **~77 h** |

### 9.2 End-to-End Timeline

```
Day 0
  ├─ Parity verification (30 min)               ← BLOCKING GATE
  ├─ Refinements integration (1 h)
  └─ Data extraction (1 h)

Day 0.5 – Day 1
  └─ Phase 1 training (10–47 h)                 ← longest phase
     ├─ 1B finishes first (~2 h)                   freeze ✓
     ├─ 1C finishes second (~5 h)                  freeze ✓
     └─ 1A finishes last (~10 h)                   freeze ✓

Day 1.5
  └─ Phase 2 training (3–5 h)                   freeze ✓

Day 2
  └─ Phase 3 training (2–4 h)                   freeze ✓

Day 2.5
  └─ Phase 4 fine-tuning (4–8 h)

Day 3
  └─ ONNX export + integration testing
  └─ Handoff to OGN octree pipeline
```

---

## 10. Risk Register & Mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | **Parity failure** — our Perlin implementation doesn't match Minecraft's | Legacy training data mismatch → garbage | BLOCKING gate: Ensure the training data matches vanilla noise outputs before training |
| 2 | **Phase 1A won't converge** — MSE stuck above 0.001 after 50 epochs | Surface terrain is wrong | Increase depth (+1–2 Conv3D layers), verify input normalisation, try different LR schedules |
| 3 | **Phase 1C cave recall too low** — network predicts "all solid" | No caves in generated terrain | Increase `pos_weight` (try 15.0, 20.0), add more cave examples to training set, try focal loss |
| 4 | **Phase 2 compound error** — small Phase 1 errors amplify in combination | finalDensity is off, surface shifts | Tighten Phase 1 grok thresholds (MSE < 0.0005), add surface-priority loss weighting |
| 5 | **Phase 3 staircase artifacts** — learned upscaler produces blocky terrain | Visually obvious at close range | Initialise from trilinear weights, add Laplacian smoothness penalty |
| 6 | **Phase 4 catastrophic forgetting** — unfreezing destroys Phase 1 features | MSE regresses, quality drops | Per-layer LR (1e-6 for Phase 1), gradient clipping, EWC penalty, monitor Phase 1 MSE during training |
| 7 | **Insufficient training data** — 1,000 chunks too few for generalisation | Overfits to training seeds | Scale to 5,000–10,000 chunks, train on 3+ seeds, validate on held-out seeds |
| 8 | **Memory overflow** — Phase 3/4 tensors too large for GPU | OOM crashes | Reduce batch size, use gradient accumulation, checkpoint activations (gradient checkpointing) |
| 9 | **ONNX export incompatibility** — custom ops or dynamic shapes break export | Can't deploy to LODiffusion mod | Use only ONNX-safe ops (no dynamic control flow), test export after each phase |
| 10 | **Ground truth data mismatch** — dumpnoise mod extracts wrong noise channels | Training targets are wrong | Cross-validate reference vectors from `reference-code/noise_reference_vectors.json` against multiple seeds |

---

## 11. Success Criteria

### Phase-Level Gates

| Phase | Gate | Metric | Allowed to Proceed? |
|-------|------|--------|---------------------|
| 0 | Parity | All Perlin samples within 1e-6 of vanilla | Phase 0 → Phase 1 |
| 1A | Grok | MSE < 0.001, R² > 0.99 | Freeze 1A weights |
| 1B | Grok | Biome accuracy > 95% | Freeze 1B weights |
| 1C | Grok | IoU > 0.75, Dice > 0.80 | Freeze 1C weights |
| 2 | Grok | MSE < 0.0005, Binary acc > 99.5% | Freeze Phase 2 |
| 3 | Grok | MSE vs trilinear < 0.0001, no staircase | Freeze Phase 3 |
| 4 | Converged | Block MSE < 0.0001, no catastrophic forgetting | Export to ONNX |

### System-Level Acceptance

| Criterion | Test |
|-----------|------|
| Terrain looks recognisable | Visual comparison: ML terrain vs vanilla screenshots |
| Surface height accuracy | Average deviation < 1 block at 16×16 column resolution |
| Biome transitions | Smooth, no abrupt edges where vanilla has gradients |
| No staircase artifacts | Laplacian smoothness metric passes |
| Inference speed | < 100 ms per chunk on i7/Ryzen 5 CPU |
| Seed stability | Same seed → same terrain (deterministic) |
| Seamless at LOD transitions | No visual pop-in or discontinuities as player approaches |

---

## 12. References

| Document | Purpose |
|----------|---------|
| [NOISE-DESIGN.md](NOISE-DESIGN.md) | Why router6 was dropped; per-LOD conditioning |
| [MODEL-CONTRACT.md](MODEL-CONTRACT.md) | ONNX tensor contracts for `lodiffusion.v5.octree` |
| [OCTREE-GENERATION-DESIGN.md](OCTREE-GENERATION-DESIGN.md) | OGN pipeline architecture (L4 → L0 traversal) |
| [TRAINING-OVERVIEW.md](TRAINING-OVERVIEW.md) | Progressive octree refinement training loop |
| [TRAINING-DATA-PIPELINE-PLAN.md](TRAINING-DATA-PIPELINE-PLAN.md) | Server bootstrap + Voxy LOD extraction |
| [PROJECT-OUTLINE.md](PROJECT-OUTLINE.md) | Full project architecture and dependency matrix |

