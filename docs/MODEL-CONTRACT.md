# 📐 Model Contract — VoxelTree ↔ LODiffusion

> **Active contract:** `lodiffusion.v5.octree` — see [§A](#a-contract-v5-octree) below.
>
> The legacy `lodiffusion.v3.progressive` contract is retained in [§B](#b-legacy-contract-v3-progressive)
> for reference but is **superseded** — new deployments should use v5.

---

## A. Contract: `lodiffusion.v5.octree`

**Contract ID:** `lodiffusion.v5.octree`
**Version:** `5.0.0`
**Authoritative source (Python):** [`VoxelTree/scripts/export_octree.py`](../scripts/export_octree.py)
**Authoritative source (Java):** [`LODiffusion/.../onnx/OctreeModelRunner.java`](../../LODiffusion/src/main/java/com/rhythmatician/lodiffusion/onnx/OctreeModelRunner.java)

### A.1 Pipeline Overview

Three ONNX models form a breadth-first octree traversal aligned with Voxy's
`WorldSection` hierarchy. Each model produces a complete **32³ Voxy WorldSection**
at a specific LOD level. Empty octants are pruned via predicted occupancy masks.

```
  octree_init (L4)
       │  occ_logits → spawnChildren
       ▼
  octree_refine (L3, L2, L1 — shared weights, level embedding)
       │  occ_logits → spawnChildren
       ▼
  octree_leaf (L0)
       (no occ_logits — block-resolution, no children)
```

### A.2 Per-Model Tensor Contract

**`octree_init.onnx` — L4 root (no parent)**

| Input | Shape | dtype | Notes |
|---|---|---|---|
| `heightmap` | `[N, 5, 32, 32]` | float32 | surface, ocean_floor, slope_x, slope_z, curvature |
| `biome` | `[N, 32, 32]` | int64 | biome index per (x,z) column |
| `y_position` | `[N]` | int64 | section Y index [0, 23] |

| Output | Shape | dtype |
|---|---|---|
| `block_logits` | `[N, V, 32, 32, 32]` | float32 |
| `occ_logits` | `[N, 8]` | float32 |

**`octree_refine.onnx` — L3/L2/L1 shared**

| Input | Shape | dtype | Notes |
|---|---|---|---|
| `parent_blocks` | `[N, 32, 32, 32]` | int64 | argmax block IDs from parent (octant-extracted, 2× upsampled) |
| `heightmap` | `[N, 5, 32, 32]` | float32 | scaled to section footprint |
| `biome` | `[N, 32, 32]` | int64 | biome index |
| `y_position` | `[N]` | int64 | section Y index |
| `level` | `[N]` | int64 | LOD level (1, 2, or 3) |

| Output | Shape | dtype |
|---|---|---|
| `block_logits` | `[N, V, 32, 32, 32]` | float32 |
| `occ_logits` | `[N, 8]` | float32 |

**`octree_leaf.onnx` — L0 leaf**

| Input | Shape | dtype | Notes |
|---|---|---|---|
| `parent_blocks` | `[N, 32, 32, 32]` | int64 | argmax block IDs from parent |
| `heightmap` | `[N, 5, 32, 32]` | float32 | block-resolution heights |
| `biome` | `[N, 32, 32]` | int64 | biome index |
| `y_position` | `[N]` | int64 | section Y index |

| Output | Shape | dtype |
|---|---|---|
| `block_logits` | `[N, V, 32, 32, 32]` | float32 |

`N` = dynamic batch size (≥ 1). `V` = block vocabulary size (default 1104).

### A.3 Parent Block Hand-off

The `block_logits` output of a parent section is argmaxed to produce `int[32³]`
block IDs. For each occupied child octant (from `occ_logits`):

1. Extract the relevant 16³ sub-volume from the parent's 32³ argmax
2. Nearest-neighbor upsample 2× → 32³
3. Pass as `parent_blocks` int64 input to the child model

The embedding lookup is **baked into the ONNX graph** — the Java runtime just
passes raw integer block IDs.

### A.4 Occupancy Mask

`occ_logits` is 8 floats. Apply `sigmoid > 0.5` to produce an 8-bit mask.
Bit index: `(x&1) | ((z&1)<<1) | ((y&1)<<2)` — matches Voxy's `nonEmptyChildren`.

### A.5 Sidecar Files

Each model has a companion `_config.json` with at minimum:

```json
{
  "version": "5.0.0",
  "contract": "lodiffusion.v5.octree",
  "model": "octree_init|octree_refine|octree_leaf",
  "block_vocab_size": 1104,
  "block_mapping": { ... }
}
```

### A.6 Pipeline Manifest

`pipeline_manifest.json` lists all required files:

```json
{
  "version": "5.0.0",
  "contract": "lodiffusion.v5.octree",
  "required_files": [
    "pipeline_manifest.json",
    "octree_init.onnx", "octree_init_config.json",
    "octree_refine.onnx", "octree_refine_config.json",
    "octree_leaf.onnx", "octree_leaf_config.json"
  ]
}
```

### A.7 Deployment

```bash
# Train + export
python scripts/export_octree.py --checkpoint octree_training/best_model.pt --out-dir production

# Deploy to mod
# Copy production/* → LODiffusion/run/config/lodiffusion/
```

---

## B. Legacy Contract: `lodiffusion.v3.progressive`

> **Status: SUPERSEDED** — retained for backward compatibility only.
> New deployments should use `lodiffusion.v5.octree` above.

**Contract ID:** `lodiffusion.v3.progressive`  
**Version:** `3.0.0`  
**Authoritative source (Python):** [`VoxelTree/scripts/export_lod.py`](../scripts/export_lod.py) — `MODEL_STEPS`  
**Authoritative source (Java):** [`LODiffusion/src/…/onnx/ProgressiveModelRunner.java`](../../LODiffusion/src/main/java/net/diffusion/lodiffusion/onnx/ProgressiveModelRunner.java) — `generate()`

This document is the single cross-boundary reference. Keep it in sync with both authoritative sources; if they disagree, fix the code, then update this doc.

---

### B.1 Pipeline Overview

Four ONNX models form a coarse-to-fine ladder. Each refines the output of the previous stage. LOD0 is vanilla-only — no model covers it.

```
  Init            LOD4→3          LOD3→2          LOD2→1
  ────────        ────────        ────────        ────────
  output D=1  →  output D=2  →  output D=4  →  output D=8
                                                    │
                                               Java 2× upsample
                                                    │
                                               InferenceResult [16³]
                                               (written to Voxy LOD1–4)
```

---

### B.2 Shared Conditioning Inputs

These three tensors are **identical** for every model call within a single chunk generation:

| Name | Shape | dtype | Source | Notes |
|---|---|---|---|---|
| `x_height_planes` | `[1, 5, 16, 16]` | float32 | `AnchorSampler` | surface, ocean\_floor, slope\_x, slope\_z, curvature |
| `x_biome` | `[1, 16, 16]` | int64 | `AnchorSampler` | vanilla biome ID per (x,z) column |
| `x_y_index` | `[1]` | int64 | `LodGenerationService` | vertical slab index [0, 23] |

---

### B.3 Per-Model Tensor Contract

| Model | ONNX file | `x_parent` shape | `block_logits` shape | `air_mask` shape |
|---|---|---|---|---|
| `init_to_lod4` | `init_to_lod4.onnx` | *absent* | `[1, N, 1, 1, 1]` | `[1, 1, 1, 1, 1]` |
| `refine_lod4_to_lod3` | `refine_lod4_to_lod3.onnx` | `[1, 1, 1, 1, 1]` | `[1, N, 2, 2, 2]` | `[1, 1, 2, 2, 2]` |
| `refine_lod3_to_lod2` | `refine_lod3_to_lod2.onnx` | `[1, 1, 2, 2, 2]` | `[1, N, 4, 4, 4]` | `[1, 1, 4, 4, 4]` |
| `refine_lod2_to_lod1` | `refine_lod2_to_lod1.onnx` | `[1, 1, 4, 4, 4]` | `[1, N, 8, 8, 8]` | `[1, 1, 8, 8, 8]` |

`N` = block vocabulary size (from `block_mapping` in each sidecar config).

#### B.3a Parent hand-off

The `block_logits` output of stage *k* becomes the `x_parent` input of stage *k+1*. `ProgressiveModelRunner` passes just the raw logit tensor — no argmax, no threshold.

---

### B.4 Post-Pipeline Upsampling (Java only)

`ProgressiveModelRunner.generate()` upsamples the final 8³ `block_logits` and `air_mask` 2× (nearest-neighbour) before packing into `InferenceResult`:

```
InferenceResult.blockLogits  →  float[1][N][16][16][16]
InferenceResult.airMask      →  float[1][1][16][16][16]
```

This is the shape consumed by `VoxySectionWriter`. The ONNX models themselves **never** output 16³.

---

### B.5 Sidecar Files

Each model has a companion JSON config:

```
init_to_lod4_config.json
refine_lod4_to_lod3_config.json
refine_lod3_to_lod2_config.json
refine_lod2_to_lod1_config.json
```

Minimum required keys:

```json
{
  "contract":       "lodiffusion.v3.progressive",
  "version":        "3.0.0",
  "block_mapping":  { "<model_index>": "<voxy_block_id>", … },
  "block_id_to_name": { "<voxy_block_id>": "<block_name>", … }
}
```

---

### B.6 Pipeline Manifest

`pipeline_manifest.json` lists every required file with its SHA-256 checksum. `ProgressiveModelRunner` validates all entries at startup and refuses to load if any file is missing or mismatched.

---

### B.7 Test Vectors

Each model ships a NumPy archive of golden input→output pairs:

```
init_to_lod4_test_vectors.npz
refine_lod4_to_lod3_test_vectors.npz
refine_lod3_to_lod2_test_vectors.npz
refine_lod2_to_lod1_test_vectors.npz
```

Validated by `LODiffusion/verify_lodiffusion_v1.py` on the Python side and by `ProgressiveModelRunner`'s startup smoke-test on the Java side.

---

### B.8 Deployment Steps

```bash
# 1. Train + export (VoxelTree side)
python scripts/export_lod.py --output production/vN

# 2. Verify shapes + checksums
python verify_lodiffusion_v1.py production/vN

# 3. Deploy to mod
python scripts/deploy_models.py production/vN
# → copies all files to LODiffusion/run/config/lodiffusion/
```

---

### B.9 y_index Range

`x_y_index` encodes the vertical 16-block slab within a Minecraft world:

| y_index | Block Y range |
|---|---|
| 0 | −64 … −49 |
| 1 | −48 … −33 |
| … | … |
| 23 | 304 … 319 |

24 slabs × 16 blocks = 384 blocks total (Minecraft 1.18+ world height).

---

### B.10 Invariants (enforced at runtime)

* All input shapes are **static** — no dynamic axes.
* `ProgressiveModelRunner` fails fast on any shape mismatch.
* `x_biome` values must be within range of the model's vocabulary; out-of-range values are clamped to 0 (unknown).
* LOD0 is **never** generated by the model pipeline; vanilla terrain always fills that level.
