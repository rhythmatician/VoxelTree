# docs/AC.md — Phase‑1 Acceptance Criteria & Strategy (Terrain‑only)

## 0) Scope

**Minecraft:** Java 1.21.x, **terrain only** (no generated structures). Training/eval worlds are created with `generate-structures=false`.

## 1) Functional & Integration

* **Determinism:** Same inputs ⇒ identical outputs (bit‑for‑bit) on CPU.
* **DJL/ONNX ready:**

  * `model.onnx` loads in a Java DJL harness and reproduces PyTorch outputs on provided vectors with **max\_abs\_diff ≤ 1e‑4**.
  * **Static shapes**, **opset ≥ 17**, **no dynamic axes**, no control‑flow/custom ops.
  * Output names **exactly:** `block_logits`, `air_mask`.
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
* **Coarser parent starts (simulated with lod token):**

  * 4³‑parent (LOD2→0 equiv): frequent‑set ≥ **0.985**
  * 2³‑parent (LOD3→0 equiv): frequent‑set ≥ **0.98**
  * 1³ / sub‑1³ (LOD4/5→0 equiv): frequent‑set ≥ **0.975**
* **Full rollout (self‑fed LOD5→4→3→2→1→0):**

  * Final LOD0 frequent‑set ≥ **0.985**; visual QA passes (readable biomes; rivers align).

> If ores are included, track a separate “ore subset” score; otherwise exclude ores from Phase‑1 gates.

## 4) Model I/O Contract (Phase‑1)

**Inputs**

* `parent_voxel` **\[1,1,8,8,8] float32** — 3D occupancy (OR‑pooled from child or prior prediction)
* `biome_patch` **\[1,16,16] int64** — vanilla biome index (surface)
* `heightmap_patch` **\[1,1,16,16] float32** — normalized motion‑blocking height \[0,1]
* `river_patch` **\[1,1,16,16] float32** — optional river mask
* *(optional but recommended)* `multinoise_{continentalness,erosion,ridge,weirdness,temperature,humidity}` **\[1,1,16,16] float32** — seed‑derived 2D maps
* `y_index` **\[1] int64** — vertical slab index
* `lod` **\[1] int64** — coarseness token: 1 for native 8→16; >1 when parent was coarsened in train

**Outputs**

* `block_logits` **\[1, N\_blocks, 16,16,16] float32** — full block vocab
* `air_mask` **\[1,1,16,16,16] float32** — P(air)

> (Optional, not Phase‑1 critical) `coarse_logits8 [1,N_coarse,8,8,8]`, `surface_logits [1,N_surface,16,16]` to improve far‑LOD visuals without 16³.

## 5) Data Pipeline (truthful vanilla labels)

* Worldgen via Fabric in a temp dir; `server.properties` includes `generate-structures=false` and numeric `level-seed`.
* Extraction parses `.mca` sections (palette + bit‑packed block states). **No synthetic fallbacks.**
* Full region coverage: iterate **32×32 chunks** per region; skip missing/corrupt chunks.
* Stable vocab: discovered automatically to `block_vocab.json` (`air` fixed to 0); dataset freezes a copy.
* Build LOD targets by **2×2×2 pooling**: occupancy=OR; blocks=probability‑pool (or mode).

## 6) Training Strategy (single static model; multi‑LOD capable)

* Always predict **16³** (LOD0) from an **8³** parent.
* For each sample, randomly choose coarsening factor `f ∈ {1,2,4,8,16}` (relative to 8³):

  * `parent_f = OR_pool(Occ16, window=f)` → upsample to **8³** for input
  * `lod = log2(f)+1`
  * Targets remain **true 16³** labels (plus air)
* **Scheduled sampling**: with p≈0.1→0.3, form `Occ16` (and parent) from the model’s own previous prediction to stabilize rollouts.
* **Loss**: `CE(block_logits, labels16) + λ_air * BCEWithLogits(air_mask, air_target)`; start `λ_air=0.25`.

## 7) Evaluation

* **Per‑step eval**: accuracy/IoU for each `f` above.
* **Full rollout eval**: self‑fed chain from LOD5→…→LOD0.
* Report frequent‑set vs overall, confusion matrix on frequent‑set, surface top‑block accuracy.

## 8) Export & Metadata

* Export **static ONNX** (opset ≥17), ordered inputs/outputs:

  * Inputs: `parent_voxel, biome_patch, heightmap_patch, river_patch, (multinoise_*×6), y_index, lod`
  * Outputs: `block_logits, air_mask`
* `model_config.json` includes: schemas & dtypes, normalization, `block_id↔name`, MC version, git SHAs, dataset ID, optional‑head flags.
* `test_vectors.npz` contains at least one realistic sample (inputs + expected outputs) for harness verification.

## 9) Runtime (LODiffusion) — Just‑in‑Time

* **LOD5 (1³):** biome/height tint (no 3D inference in Phase‑1 minimal)
* **LOD4/3/2:** (optional) pool down from an 8³ coarse head if added later; otherwise keep tint
* **LOD1 (8³):** run the model **once** for that patch; render 16³ result
* **LOD0:** render from 16³; optionally handoff to vanilla in sim distance
* Scheduler: speed‑aware forward cone; 360° when idle; preempt on player turn; tight D1 safety halo; cache & evict far‑behind first

## 10) CI / Definition of Done

* Unit: palette decode, downsample invariants
* Mini E2E: export ONNX + `test_vectors.npz` on CI
* DJL harness: Java loads ONNX, runs vectors, **max\_abs\_diff ≤ 1e‑4**
* Eval: per‑step + rollout metrics on held‑out seeds (config‑driven); artifacts saved
* Hygiene: batch cleanup verified; `block_vocab.json` frozen in dataset; `model_config.json` validated via JSON Schema
