# docs/AC.md — Phase‑1 Acceptance Criteria & Strategy (Terrain‑only)

## 0) Scope

**Minecraft:** Java 1.21.x, **terrain only** (no generated structures). Training/eval worlds are created with `generate-structures=false`.

## 1) Functional & Integration

* **Determinism:** Same inputs ⇒ identical outputs (bit‑for‑bit) on CPU.
* **DJL/ONNX ready:**

  * `model.onnx` loads in a Java DJL harness and reproduces PyTorch outputs on provided vectors with **max\_abs\_diff ≤ 1e‑4**.
  * **Static shapes**, **opset ≥ 17**, **no dynamic axes**, no control‑flow/custom ops.
  * Output names **exactly:** `block_logits` (single-head: air = class 0 in unified softmax).
* **Artifacts shipped:** `model.onnx`, `model_config.json`, `test_vectors.npz` (1–3 realistic samples).

## 2) Runtime Performance (mid‑range CPU)

* **Latency (per 16³):** median ≤ **100 ms**, p95 ≤ **150 ms**.
* **Incremental memory:** ≤ **2 MB** above resident model.
* **Throughput:** With sprint + forward cone, player sees large render distance with no visible stalls (document scenario).

## 3) Quality (held‑out seeds; terrain blocks)

Report **overall** and **frequent‑set** (top ≈50 terrain blocks; air excluded from top‑1 unless noted).

* **LOD1 → LOD0 (8³→16³)** — primary gate:

  * Air/solid IoU (16³) ≥ **0.99**
  * Block top‑1 (frequent‑set) ≥ **0.99**
  * Block top‑1 (overall) ≥ **0.98**
* **Coarser transitions (incremental refinement — each step refines by one LOD level):**

  * LOD2→LOD1: frequent‑set ≥ **0.985**
  * LOD3→LOD2: frequent‑set ≥ **0.98**
  * LOD4→LOD3: frequent‑set ≥ **0.975**
* **Full rollout (self‑fed LOD4→LOD3→LOD2→LOD1→LOD0):**

  * Final LOD0 frequent‑set ≥ **0.985**; visual QA passes (readable biomes; rivers align).

> If ores are included, track a separate “ore subset” score; otherwise exclude ores from Phase‑1 gates.

## 4) Model I/O Contract (v2 — Anchor Conditioning)

**Inputs**

* `x_parent` **\[1,1,8,8,8] float32** — binary occupancy (Mipper-derived from child or prior prediction)
* `x_height_planes` **\[1,5,16,16] float32** — surface, ocean\_floor, slope\_x, slope\_z, curvature
* `x_router6` **\[1,6,16,16] float32** — temperature, vegetation, continents, erosion, depth, ridges
* `x_biome` **\[1,16,16] int64** — vanilla biome index per (x,z)
* `x_y_index` **\[1] int64** — vertical 16‑slab index
* `x_lod` **\[1] int64** — coarseness token: 1 for native 8→16; >1 when parent was coarsened in train

**Outputs**

* `block_logits` **\[1, 1104, 16,16,16] float32** — Voxy-native vocabulary (1104 block types); argmax(dim=1) yields block indices, where class 0 represents air.

> Block vocabulary: 1102 entries from canonical Voxy vocabulary (`config/voxy_vocab.json`), **air=class 0**.

## 5) Data Pipeline (truthful Voxy labels)

* Worldgen via Fabric server; `server.properties` includes `generate-structures=false` and numeric `level-seed`.
* Use `/voxy import world` to populate Voxy's RocksDB with LOD data.
* Extraction reads **Voxy RocksDB** databases via `scripts/voxy_reader.py` (SaveLoadSystem3 decoder). **No synthetic fallbacks.**
* Per-world Voxy state IDs mapped to **canonical vocabulary** (`config/voxy_vocab.json`, 1102 entries, air=0) by block name.
* Build LOD targets using **Voxy Mipper** (`scripts/mipper.py`): opacity-biased corner selection, not OR-pool or majority vote.
* Pipeline orchestrator: `pipeline.py extract` → `pipeline.py train` → `pipeline.py export`.

## 6) Training Strategy (single static model; incremental refinement)

* Always predict **16³** from an **8³** parent (both nearest-upsampled to canonical sizes).
* Each training sample represents one **LOD step**: the target is one level finer
  than the parent, not always LOD0.
* For each sample, randomly choose coarsening factor `f ∈ {2,4,8,16}`:

  * Parent: `mip_volume_numpy(labels16, f, tbl)` → nearest-upsample to **8³**
  * Target: `mip_volume_numpy(labels16, f//2, tbl)` → nearest-upsample to **16³**
    (for f=2 the target is `labels16` itself — full-detail LOD0)
  * `lod_token = log2(f)` (1,2,3,4)
  * `lod_transition = "lod{N}to{N-1}"` (lod1to0, lod2to1, lod3to2, lod4to3)
* Training samples are drawn from an **octree expansion process**:
  1. **Init samples**: anchor channels → root node (1³) with no parent.
  2. **Refine samples**: parent node of size `s` → eight child nodes of size `2s` (repeat for
     s=1,2,4,8). Each refine sample includes the parent occupancy and the corresponding
     multi-channel anchors.
  3. **Leaf samples**: final L1 parent (`8³`) → full 32³ leaf volume prediction.
  Each sample is treated independently; the recursive traversal is handled at
  inference time by the octree scheduler.
* **Scheduled sampling** remains useful for stability: with p≈0.1→0.3, child inputs
  may be formed from the model’s own argmax prediction (mipped back to parent size)
  before being used in the next refine step.
* **Loss**: `CE(block_logits, target_labels)` — unified cross-entropy on all 1102 classes
  (air=class 0).

## 7) Evaluation

* **Per‑step eval**: accuracy/IoU for each LOD transition (lod1to0, lod2to1, lod3to2, lod4to3).
* **Full rollout eval**: self‑fed chain from LOD4→LOD3→LOD2→LOD1→LOD0.
* Report frequent‑set vs overall, confusion matrix on frequent‑set, surface top‑block accuracy.

## 8) Export & Metadata

* Export **static ONNX** (opset ≥17), ordered inputs/outputs:

  * Inputs: `x_parent, x_height_planes, x_router6, x_biome, x_y_index, x_lod` (per-step models; init has no x_parent)
  * Outputs: `block_logits` (single-head: argmax(axis=1) → block indices, air=class 0)
* `model_config.json` includes: schemas & dtypes, normalization, `block_mapping` from `config/voxy_vocab.json` (Voxy-native), `block_id_to_name` reverse mapping, MC version, git SHAs, dataset ID.
* `test_vectors.npz` contains at least one realistic sample (inputs + expected outputs) for harness verification.
* LODiffusion's `VoxyBlockMapper` reads `block_mapping` from `model_config.json` at runtime.

## 9) Runtime (LODiffusion) — Just‑in‑Time

* **Octree traversal**: work queue seeded with L4 root nodes in the generation radius.
  - **Level workers** pull `OctreeTask` objects from `OctreeQueue` (L4→L3→L2→L1→leaf).
  - At each non‑leaf node, `OctreeModelRunner.runInit` (for L4) or `runRefine` is
    invoked; children are enqueued based on occupancy mask and radius.
  - Leaf nodes invoke `runLeaf` to produce a 32³ block volume, which the mod
    immediately slices into eight 16³ Voxy sections and writes them to RocksDB.
* **Scheduler**: speed‑aware forward cone; 360° when idle; preempt on player turn;
  tight D1 safety halo; cache anchors & octree results; evict far‑behind nodes first.
* **Fallback**: if inference cannot keep up, skip deeper levels and defer them until
  next tick; Voxy renders missing sections as fog until ready.
* **Vanilla handoff**: the leaf output is a distance render proxy only. When the
  player nears a region, vanilla terrain generation overwrites any proxy data
  (insert‑only policy guarantees no corruption).

## 10) CI / Definition of Done

* Unit: Mipper invariants, Voxy extraction round-trip, dataset contract
* Mini E2E: export ONNX + `test_vectors.npz` on CI
* DJL harness: Java loads ONNX, runs vectors, **max\_abs\_diff ≤ 1e‑4**
* Eval: per‑step + rollout metrics on held‑out seeds (config‑driven); artifacts saved
* Hygiene: batch cleanup verified; `config/voxy_vocab.json` canonical; `model_config.json` validated via JSON Schema
