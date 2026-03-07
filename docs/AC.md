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

## 4) Model I/O Contract (v2 — Anchor Conditioning)

**Inputs**

* `x_parent` **\[1,1,8,8,8] float32** — binary occupancy (Mipper-derived from child or prior prediction)
* `x_height_planes` **\[1,5,16,16] float32** — surface, ocean\_floor, slope\_x, slope\_z, curvature
* `x_router6` **\[1,6,16,16] float32** — temperature, vegetation, continents, erosion, depth, ridges
* `x_biome` **\[1,16,16] int64** — vanilla biome index per (x,z)
* `x_y_index` **\[1] int64** — vertical 16‑slab index
* `x_lod` **\[1] int64** — coarseness token: 1 for native 8→16; >1 when parent was coarsened in train

**Outputs**

* `block_logits` **\[1, 1102, 16,16,16] float32** — Voxy-native vocabulary (1102 block types)
* `air_mask` **\[1,1,16,16,16] float32** — P(air)

> Block vocabulary: 1102 entries from canonical Voxy vocabulary (`config/voxy_vocab.json`), air=0.

## 5) Data Pipeline (truthful Voxy labels)

* Worldgen via Fabric server; `server.properties` includes `generate-structures=false` and numeric `level-seed`.
* Use `/voxy import world` to populate Voxy's RocksDB with LOD data.
* Extraction reads **Voxy RocksDB** databases via `scripts/voxy_reader.py` (SaveLoadSystem3 decoder). **No synthetic fallbacks.**
* Per-world Voxy state IDs mapped to **canonical vocabulary** (`config/voxy_vocab.json`, 1102 entries, air=0) by block name.
* Build LOD targets using **Voxy Mipper** (`scripts/mipper.py`): opacity-biased corner selection, not OR-pool or majority vote.
* Pipeline orchestrator: `pipeline.py extract` → `pipeline.py train` → `pipeline.py export`.

## 6) Training Strategy (single static model; multi‑LOD capable)

* Always predict **16³** (LOD0) from an **8³** parent.
* For each sample, randomly choose coarsening factor `f ∈ {1,2,4,8,16}` (relative to 8³):

  * `parent_f = mip_volume_numpy(labels16, f, tbl)` → nearest-upsample to **8³** for input
  * `lod = log2(f)+1`
  * Targets remain **true 16³** labels (plus air)
* **Scheduled sampling**: with p≈0.1→0.3, form parent from the model's own previous prediction (argmax → Mipper) to stabilize rollouts.
* **Loss**: `CE(block_logits, labels16) + λ_air * BCEWithLogits(air_mask, air_target)`; start `λ_air=0.25`.

## 7) Evaluation

* **Per‑step eval**: accuracy/IoU for each `f` above.
* **Full rollout eval**: self‑fed chain from LOD5→…→LOD0.
* Report frequent‑set vs overall, confusion matrix on frequent‑set, surface top‑block accuracy.

## 8) Export & Metadata

* Export **static ONNX** (opset ≥17), ordered inputs/outputs:

  * Inputs: `x_parent, x_height_planes, x_router6, x_biome, x_y_index, x_lod`
  * Outputs: `block_logits, air_mask`
* `model_config.json` includes: schemas & dtypes, normalization, `block_mapping` from `config/voxy_vocab.json` (Voxy-native), `block_id_to_name` reverse mapping, MC version, git SHAs, dataset ID.
* `test_vectors.npz` contains at least one realistic sample (inputs + expected outputs) for harness verification.
* LODiffusion's `VoxyBlockMapper` reads `block_mapping` from `model_config.json` at runtime.

## 9) Runtime (LODiffusion) — Just‑in‑Time

* **LOD5 (1³):** biome/height tint (no 3D inference in Phase‑1 minimal)
* **LOD4/3/2:** (optional) pool down from an 8³ coarse head if added later; otherwise keep tint
* **LOD1 (8³):** run the model **once** for that patch; render 16³ result
* **LOD0:** render from 16³; optionally handoff to vanilla in sim distance
* Scheduler: speed‑aware forward cone; 360° when idle; preempt on player turn; tight D1 safety halo; cache & evict far‑behind first

## 10) CI / Definition of Done

* Unit: Mipper invariants, Voxy extraction round-trip, dataset contract
* Mini E2E: export ONNX + `test_vectors.npz` on CI
* DJL harness: Java loads ONNX, runs vectors, **max\_abs\_diff ≤ 1e‑4**
* Eval: per‑step + rollout metrics on held‑out seeds (config‑driven); artifacts saved
* Hygiene: batch cleanup verified; `config/voxy_vocab.json` canonical; `model_config.json` validated via JSON Schema
