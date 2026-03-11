# Octree Generation Pipeline — Design Document

## Status: ACTIVE — contract `lodiffusion.v5.octree` v5.0.0

## 1. Problem Statement

The current generation pipeline operates on **16³ Minecraft chunk sections** and always runs **4 sequential ONNX stages** per section, regardless of content. This is misaligned with Voxy's spatial hierarchy and wastes compute on sky/underground regions that contain no interesting detail.

## 2. Goal

Replace the per-subchunk 4-stage pipeline with a **true octree traversal** aligned with Voxy's `WorldSection` hierarchy. Each model call produces a complete **32³ Voxy WorldSection** at a specific LOD level. Empty octants are pruned, and their subtrees never run inference.

## 3. Voxy's Spatial Hierarchy (Reference)

A Voxy `WorldSection` is always **32³ voxels internally**, regardless of LOD level.

| Level | Voxel Size | Section Covers | Key Use Case |
|-------|-----------|----------------|--------------|
| L4 | 16m/voxel | 512³ blocks | Distant terrain silhouette |
| L3 | 8m/voxel | 256³ blocks | Mid-range terrain shape |
| L2 | 4m/voxel | 128³ blocks | Near terrain geology |
| L1 | 2m/voxel | 64³ blocks | Close terrain detail |
| L0 | 1m/voxel | 32³ blocks | Block-level resolution |

**Parent → Child:** A WorldSection at level L, position (px, py, pz) has 8 children at level L-1:
```
childX = (px << 1) + (octant & 1)
childY = (py << 1) + ((octant >> 2) & 1)
childZ = (pz << 1) + ((octant >> 1) & 1)
```

**`nonEmptyChildren`** is an 8-bit bitmask on each section. Bit index: `(x&1) | ((z&1)<<1) | ((y&1)<<2)`. The GPU renderer uses it to skip empty subtrees.

## 4. Architecture Overview

### 4.1 Pipeline Flow

```
 ┌──────────────────────────────────────────────────┐
 │  For each L4 WorldSection in player render range  │
 └──────────────────┬───────────────────────────────┘
                    ▼
            ┌───────────────┐
            │  L4 Model     │  Input: heightmap(32×32), biome(32×32), position
            │  32³ → 1104   │  Output: 32³ block predictions + nonEmptyChildren byte
            └───────┬───────┘
                    │ nonEmptyChildren = 0b00001111 (4 of 8 children occupied)
                    ▼
        ┌───────────────────────────┐
        │  For each set bit (4 of 8) │  Create child tasks at L3
        └───────────┬───────────────┘
                    ▼
            ┌───────────────┐
            │  L3 Model     │  Input: parent 32³ data (L4 window), heightmap, biome, pos
            │  32³ → 1104   │  Output: 32³ block predictions + nonEmptyChildren byte
            └───────┬───────┘
                    │ etc.
                    ▼
              L2 → L1 → L0
```

### 4.2 What Gets Skipped

If a parent section's `nonEmptyChildren` says octant 5 is empty:
- **No L(n-1) model call** for that child
- **No L(n-2) through L0** calls for the entire subtree under that child
- The region just doesn't exist in Voxy — Voxy's GPU renderer already handles this

**Conservative estimate for typical overworld terrain:**

| Level | Max Sections | Estimated Occupied | Sections Processed |
|-------|--------------|--------------------|--------------------|
| L4 | 1 root | 1 | 1 |
| L3 | 8 | ~4 (terrain surface) | 4 |
| L2 | 64 | ~16 | 16 |
| L1 | 512 | ~64 | 64 |
| L0 | 4096 | ~256 | 256 |
| **Total** | **4681** | | **~341** |

vs. current system: ~2048 chunk sections × 4 stages = **~8192 ONNX calls** for similar coverage.

## 5. Model Architecture

### 5.1 Three Models (L4 init, L3–L1 shared refinement, L0 leaf)

All models are **3D U-Nets** operating on 32³ spatial grids. L3–L1 share a single set of
weights with a **level embedding** input that tells the model which LOD scale it is refining.

**Model A — `octree_init` (L4 only):**
```
Input:
  - heightmap:      float[5, 32, 32]    Surface/ocean heights, slopes, curvature
  - biome:          int[32, 32]         Biome indices (embedded)
  - y_position:     int scalar          Section Y coordinate (embedded)
  (no parent_context — L4 is the root)

Output:
  - block_logits:   float[num_classes, 32, 32, 32]   Block type classification
  - occ_logits:     float[8]                          nonEmptyChildren prediction
```

**Model B — `octree_refine` (shared across L3, L2, L1):**
```
Input:
  - parent_blocks:  int[32, 32, 32]    Parent octant (16³ → upsampled 2× → 32³) block IDs
  - heightmap:      float[5, 32, 32]    Height planes scaled to section footprint
  - biome:          int[32, 32]         Biome indices (embedded)
  - y_position:     int scalar          Section Y coordinate (embedded)
  - level:          int scalar          LOD level (1, 2, or 3) — embedded

Output:
  - block_logits:   float[num_classes, 32, 32, 32]   Block type classification
  - occ_logits:     float[8]                          nonEmptyChildren prediction
```

**Model C — `octree_leaf` (L0 only):**
```
Input:
  - parent_blocks:  int[32, 32, 32]    Parent octant (16³ → upsampled 2× → 32³) block IDs
  - heightmap:      float[5, 32, 32]    Block-resolution heights
  - biome:          int[32, 32]         Biome indices (embedded)
  - y_position:     int scalar          Section Y coordinate (embedded)

Output:
  - block_logits:   float[num_classes, 32, 32, 32]   Block type classification
  (no occ_logits — individual blocks have no children)
```

**Why 3 models instead of 5:**
- L4 is architecturally distinct — no parent context, coarsest scale, effectively 2D.
- L3–L1 do the same job (refine a parent octant into finer detail + predict child occupancy)
  at different scales. A level embedding handles the differences, and pooling L3/L2/L1 training
  data into one model gives 3× more samples per gradient step.
- L0 is distinct — no occupancy head, highest block-level accuracy requirement.

### 5.2 Model Specification

```
3D U-Net with 3 resolution levels:

Encoder:
  32³ × C₀ → Conv3d(3,1,1) → BN → ReLU → Conv3d(3,1,1) → BN → ReLU → ↓2
  16³ × C₁ → Conv3d(3,1,1) → BN → ReLU → Conv3d(3,1,1) → BN → ReLU → ↓2
   8³ × C₂ → Conv3d(3,1,1) → BN → ReLU → Conv3d(3,1,1) → BN → ReLU

Bottleneck:
   8³ × C₂

Decoder:
   8³ × C₂ → ↑2 → cat(skip) → Conv3d → BN → ReLU → Conv3d → BN → ReLU → 16³ × C₁
  16³ × C₁ → ↑2 → cat(skip) → Conv3d → BN → ReLU → Conv3d → BN → ReLU → 32³ × C₀

Output heads:
  32³ × C₀ → Conv3d(1,0,0) → 32³ × num_classes     (block classification)
   8³ × C₂ → GAP → Linear(C₂, 32) → ReLU → Linear(32, 8)   (occupancy head — L4–L1 only)

The L0 model omits the occupancy head entirely, since individual blocks have no children.
```

**Suggested channel widths:**

| Model | C₀ | C₁ | C₂ | Est. Params | Rationale |
|-------|----|----|-----|-------------|-----------||
| A: `octree_init` (L4) | 24 | 48 | 96 | ~400K | Coarsest scale, no parent context |
| B: `octree_refine` (L3–L1) | 32 | 64 | 128 | ~800K | Shared across 3 levels via level embedding |
| C: `octree_leaf` (L0) | 48 | 96 | 192 | ~1.8M | Block-level accuracy, no occ head |

**Total: ~3.0M params across 3 models** (vs. ~1M current, but far fewer ONNX calls per column).

### 5.3 Parent Context Encoding

Each model (except L4) receives the **parent section's 32³ block predictions** as context. Since a child section corresponds to one **octant** of the parent, the relevant parent data is a 16³ sub-cube of the parent's 32³ grid. This gets upsampled 2× to 32³ to match the child's spatial resolution.

```
parent_32³[px, py, pz]  →  argmax → int[32³]  →  extract octant 16³  →  upsample ×2 → 32³  →  parent_blocks input
```

The embedding lookup from block IDs to dense features is **baked into the ONNX graph**,
so the Java runtime just passes raw integer block IDs.

This provides the structural skeleton: "your parent predicted stone at this location — now predict the finer detail."

### 5.4 Conditioning at Each Level

The heightmap/biome data must cover the **world footprint** of the section:

| Level | World Footprint | Heightmap Source |
|-------|----------------|-----------------|
| L0 | 32×32m | 32×32 vanilla heights (stitch 2×2 chunk heightmaps) |
| L1 | 64×64m | Pool 64×64 vanilla heights → 32×32 |
| L2 | 128×128m | Pool 128×128 → 32×32 |
| L3 | 256×256m | Pool 256×256 → 32×32 |
| L4 | 512×512m | Pool 512×512 → 32×32 |

All conditioning is resampled to **32×32** to match the section's spatial grid, regardless of LOD level.

## 6. Training Data Pipeline

### 6.1 Data Extraction Changes

The current extraction script (`extract_voxy_training_data.py`) splits 32³ Voxy sections into 16³ sub-blocks. For the new pipeline:

1. **Keep the full 32³ shape** — don't call `section_to_subblocks()`
2. **Extract all LOD levels** (0–4) — change `iter_sections(level=0)` to iterate all levels
3. **Save 32×32 heightmaps** — stitch from vanilla heightmaps covering the section's footprint
4. **Save parent/child relationships** — for each section, record its parent section ID and child occupancy

### 6.2 Training Pair Format

```python
# Per sample:
{
    "labels32":          np.int32[32, 32, 32],    # Target block IDs
    "parent_labels32":   np.int32[32, 32, 32],    # Parent's 32³ (octant-extracted + upsampled)
    "heightmap32":       np.float32[5, 32, 32],   # Height planes (surface, ocean, slopes, curvature)
    "biome32":           np.int32[32, 32],         # Biome indices
    "y_position":        np.int64,                 # Section Y coordinate
    "level":             np.int64,                 # LOD level (0–4)
    "non_empty_children": np.uint8,                # Ground-truth 8-bit occupancy mask
}
```

### 6.3 Ground Truth vs. Synthetic LOD Data

**Option A (recommended):** Use Voxy's actual LOD data from the database. This is ground truth — Voxy's own mipper produced it. Available via `VoxyReader.iter_sections(level=N)`.

**Option B (fallback):** Synthetically downsample L0 data to create L1–L4 using our Python mipper. This is what the current pipeline does. Less accurate but doesn't require Voxy data at higher levels.

**Hybrid:** Start with Option B for rapid iteration, validate with Option A later.

## 7. Java Runtime (LODiffusion)

### 7.1 New Task Structure

Replace `SectionTask` with `OctreeTask`:

```java
class OctreeTask {
    final int level;           // 0–4
    final int wsX, wsY, wsZ;   // WorldSection coordinates at this level
    final long wsKey;          // Packed Voxy WorldSection key
    final byte parentOccMask;  // Which octant of parent we are (0xFF for root)
    volatile long[] parentBlocks;  // Parent's 32³ predictions (upsampled octant, int64 block IDs)
    volatile State state;
    volatile int priority;
}
```

### 7.2 New Queue Structure

Replace `LodGenerationQueue` with `OctreeQueue`:

```java
class OctreeQueue {
    // 5 priority queues — one per LOD level
    PriorityBlockingQueue<OctreeTask>[] levelQueues = new PBQ[5];

    // Process L4 first, then L3, etc. — breadth-first by level
    OctreeTask pollNextTask(long timeoutMs);

    // After processing, spawn child tasks for occupied octants
    void spawnChildren(OctreeTask parent, byte occMask, float[][][] predictions);
}
```

**`spawnChildren` creates up to 8 child tasks:**
```java
void spawnChildren(OctreeTask parent, byte occMask, float[][][] preds) {
    for (int octant = 0; octant < 8; octant++) {
        if ((occMask & (1 << octant)) == 0) continue;  // SKIP empty octant

        int childX = (parent.wsX << 1) + (octant & 1);
        int childY = (parent.wsY << 1) + ((octant >> 2) & 1);
        int childZ = (parent.wsZ << 1) + ((octant >> 1) & 1);

        long[] parentBlocks = extractAndUpsampleOctant(preds, octant); // int64 block IDs
        OctreeTask child = new OctreeTask(parent.level - 1, childX, childY, childZ, parentBlocks);
        levelQueues[child.level].add(child);
    }
}
```

### 7.3 Write Path

Every task writes directly to its native Voxy level:

| Level | Write Method | Voxels Written |
|-------|-------------|----------------|
| L4 | `writeAtLevel(4, ...)` | 1 voxel in the L4 grid |
| L3 | `writeAtLevel(3, ...)` | 8 voxels in the L3 grid |
| L2 | `writeAtLevel(2, ...)` | 64 voxels in the L2 grid |
| L1 | `writeAtLevel(1, ...)` | 512 voxels in the L1 grid |
| L0 | `insertUpdate(...)` | 32³ via standard path |

Wait — re-reading VoxyCompat: `writeAtLevel` writes a sub-region of a WorldSection, not a full 32³. We need a new method: `writeWorldSection(level, wsX, wsY, wsZ, long[32³])` that writes the **entire 32³ grid** of a WorldSection directly.

### 7.4 ONNX Model Runner

Replace `ProgressiveModelRunner` with `OctreeModelRunner`:

```java
class OctreeModelRunner {
    private final OrtSession initModel;    // octree_init.onnx   (L4)
    private final OrtSession refineModel;  // octree_refine.onnx (L3–L1, shared)
    private final OrtSession leafModel;    // octree_leaf.onnx   (L0)

    StageOutput runLevel(int level, float[] parentContext,
                         float[] heightmap, int[] biome, int yPos) {
        OrtSession model = switch (level) {
            case 4 -> initModel;
            case 0 -> leafModel;
            default -> refineModel;  // levels 1, 2, 3
        };
        // Build input tensors (include level embedding for refineModel)
        // Run inference, extract outputs
        return new StageOutput(blockLogits32x32x32, occLogits8);
    }

    // Batch version: run multiple sections at same level in one call
    StageOutput[] runLevelBatch(int level, List<OctreeTask> tasks) { ... }
}
```

### 7.5 Worker Topology

```
Level 4 workers (1 thread)  — produces root predictions, spawns L3 children
Level 3 workers (2 threads) — processes L3 tasks, spawns L2 children
Level 2 workers (2 threads) — processes L2 tasks, spawns L1 children
Level 1 workers (4 threads) — processes L1 tasks, spawns L0 children
Level 0 workers (4 threads) — processes L0 tasks (highest volume)
```

Higher levels are serialized or low-parallelism (few tasks). Lower levels get more parallelism (many tasks).

## 8. ONNX Export

### 8.1 Three ONNX Files

```
octree_init.onnx    — Model A: L4 root (no parent context)
octree_refine.onnx  — Model B: shared L3/L2/L1 refinement (with level embedding)
octree_leaf.onnx    — Model C: L0 leaf (no occupancy head)
```

### 8.2 I/O Contracts

**`octree_init.onnx` (L4):**
```
Inputs:
  heightmap:      float32[N, 5, 32, 32]
  biome:          int64[N, 32, 32]
  y_position:     int64[N]

Outputs:
  block_logits:   float32[N, num_classes, 32, 32, 32]
  occ_logits:     float32[N, 8]
```

**`octree_refine.onnx` (L3–L1):**
```
Inputs:
  parent_blocks:  int64[N, 32, 32, 32]
  heightmap:      float32[N, 5, 32, 32]
  biome:          int64[N, 32, 32]
  y_position:     int64[N]
  level:          int64[N]                               (1, 2, or 3)

Outputs:
  block_logits:   float32[N, num_classes, 32, 32, 32]
  occ_logits:     float32[N, 8]
```

**`octree_leaf.onnx` (L0):**
```
Inputs:
  parent_blocks:  int64[N, 32, 32, 32]
  heightmap:      float32[N, 5, 32, 32]
  biome:          int64[N, 32, 32]
  y_position:     int64[N]

Outputs:
  block_logits:   float32[N, num_classes, 32, 32, 32]
```

### 8.3 Contract Version

```json
{
    "version": "5.0.0",
    "contract": "lodiffusion.v5.octree",
    "models": 3,
    "voxels_per_section": 32,
    "num_classes": 1104
}
```

## 9. Implementation Plan

### Phase 1: Data Pipeline (VoxelTree)
1. Modify `extract_voxy_training_data.py` to save 32³ sections without sub-splitting
2. Extract LOD levels 0–4 from Voxy database
3. Build 32×32 heightmap/biome patches scaled per level
4. Create new `OctreeDataset` class (parent/child pairs across levels)
5. Build training pair cache

### Phase 2: Model Architecture (VoxelTree)
6. Implement `UNet3D32` model class
7. Implement 5 level-specific factory functions with appropriate channel widths
8. Implement `OccupancyHead` (reuse existing)
9. Training loop with per-level loss + occupancy BCE
10. Validation with occupancy F1 and per-level accuracy metrics

### Phase 3: Export (VoxelTree)
11. ONNX export script for 5 models
12. Test vectors and sidecar configs
13. Pipeline manifest v5

### Phase 4: Java Runtime (LODiffusion)
14. `OctreeTask` class
15. `OctreeQueue` with per-level queues and `spawnChildren()`
16. `OctreeModelRunner` loading 5 ONNX models
17. New write path: `writeWorldSection()` for full 32³ Voxy writes
18. Worker topology and scheduling
19. Underground/sky heuristics at queue population time

### Phase 5: Integration Testing
20. End-to-end: train → export → load in Java → generate in-game
21. Performance benchmarking vs. old pipeline
22. Visual quality comparison

## 10. Migration Strategy

The old pipeline (`lodiffusion.v3.progressive` / `v4.sparse_progressive`) and the new octree pipeline (`lodiffusion.v5.octree`) can coexist:

- LODiffusion checks the `contract` field in the pipeline manifest
- If `v5.octree`: use `OctreeModelRunner` + `OctreeQueue`
- If `v3`/`v4`: use existing `ProgressiveModelRunner` + `LodGenerationQueue`

This allows gradual rollout — ship v5 models when they're ready without breaking v3/v4 compat.

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 32³ model too slow for ONNX RT | High | Profile early; fall back to channels [16,32,64] if needed |
| L0 generates too many sections | Medium | Make L0 optional; Voxy renders L1 acceptably at close range |
| Training data insufficient at LOD 1–4 | Medium | Start with synthetic downsampling from L0 |
| Parent context degrades at leaf levels | Medium | Train with teacher forcing (GT parent) and student forcing (predicted parent) |
| `nonEmptyChildren` prediction accuracy | High | Monitor F1 closely; false negatives = missing terrain; false positives = wasted compute only |

## 12. Open Questions

1. **L0 write path:** `writeAtLevel` rejects level 0. We need to either write via `insertUpdate()` (requires VoxelizedSection format) or add a new VoxyCompat method for direct L0 WorldSection writes.

2. **Border coherence:** Neighboring sections at the same level may produce inconsistent predictions at their shared boundary. Possible fix: overlap the parent context by 1–2 voxels.

3. **Noise conditioning:** The current pipeline uses `/dumpnoise` tabular features from vanilla worldgen. For the octree pipeline at higher LOD levels, these features need spatial aggregation. What statistics are most predictive at 512m scale?

4. **Memory budget:** 5 ONNX models loaded simultaneously. At ~1M params each with float32, that's ~20MB model weights — easily fits.

5. **Should the L4 model even use a 3D U-Net?** At L4, each voxel is 16m — the spatial structure is very coarse. A simpler architecture (MLP or 2D conv + height axis) might suffice.
