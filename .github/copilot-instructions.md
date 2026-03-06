# 🤖 Copilot Instructions — VoxelTree (Expanded)

> **Purpose:** This file guides GitHub Copilot (and human contributors) to build VoxelTree
> consistently. It encodes our LOD-first design, data pipeline rules, training strategy,
> acceptance criteria, and CI checks.

---

## 🌲 Project Purpose

VoxelTree trains a **LOD-aware voxel super‑resolution model** for **vanilla terrain** (no structures in Phase‑1). The model runs in the **LODiffusion** mod to render huge distances **just‑in‑time**: only compute what the player can notice **right now**.

---

## ✅ Non‑Negotiables (Phase‑1)

* **Terrain‑only data**: `generate-structures=false` during worldgen.
* **Deterministic inference**: same inputs ⇒ same outputs (CPU).
* **Static ONNX**: opset ≥ 17, **no dynamic axes**, outputs named `block_logits`, `air_mask`.
* **Disk hygiene**: batch data is deleted after each train iteration; obey high‑watermark.
* **Truthful labels**: parse `.mca` **palette + bitpacked states**; **no synthetic fallbacks**.
* **Just‑in‑time**: far LODs avoid the heavy 16³ head; refine only when entering near bands.

---

## 🧭 Acceptance Criteria (excerpt)

* **LOD1→LOD0 (8³→16³)**: Air/solid IoU ≥ 0.99, frequent‑set block top‑1 ≥ 0.99, overall ≥ 0.98.
* **Coarser starts (4³/2³/1³ parents, simulated)**: frequent‑set ≥ 0.985/0.98/0.975.
* **Full rollout LOD5→…→LOD0** (self‑fed): frequent‑set at final LOD0 ≥ 0.985.
* **Runtime**: ≤ 100 ms median per 16³; ≤ 150 ms p95; ≤ 2 MB incremental mem.
* **Integration**: DJL harness loads ONNX; `max_abs_diff ≤ 1e‑4` vs PyTorch on test vectors.

> Full AC lives in `docs/AC.md`. Keep this file in sync.

---

## 🗂 Directory & Files (authoritative)

```
VoxelTree/
├─ train/
│  ├─ train.py            # CLI: train, eval, export
│  ├─ dataset.py          # Loads NPZ patches; builds multi‑LOD inputs
│  ├─ unet3d.py           # 8→16 SR UNet (+ heads if enabled)
│  ├─ exporter.py         # Static ONNX export (opset ≥17)
│  ├─ loss.py             # CE + air losses
│  └─ config.yaml         # Model + data config
├─ scripts/
│  ├─ worldgen/bootstrap.py     # Fabric + Chunky; structures=false
│  ├─ extraction/chunk_extractor.py
│  ├─ extraction/palette_decode.py
│  ├─ extraction/block_vocab.py  # auto‑discover mapping
│  ├─ verify_onnx.py             # export + test_vectors + model_config
│  ├─ run_eval.py                # per‑step + rollout metrics
│  └─ generate_corpus.py         # iter: worldgen→extract→pair→split
├─ docs/AC.md
├─ tests/                        # PyTest (unit + mini E2E)
├─ models/                       # checkpoints/onnx (ignored in git)
├─ data/                         # temp batch data (ignored)
└─ tools/                        # JARs, CLI (Chunky, Fabric, cubiomes)
```

---

## 🧠 Model I/O Contract (Phase‑1)

**Inputs**

* `parent_voxel`: **\[1,1,8,8,8] float32** — binary occupancy (3D OR‑pool from child or prev pred)
* `biome_patch`: **\[1,16,16] int64** — vanilla biome index per (x,z)
* `heightmap_patch`: **\[1,1,16,16] float32** — normalized \[0,1] motion‑blocking height
* `river_patch` *(opt)*: **\[1,1,16,16] float32** — river prior
* `multinoise_*` *(opt, 6 maps)*: **\[1,1,16,16] float32** — continentalness, erosion, ridge/peaks, weirdness, temperature, humidity (seed‑derived)
* `y_index`: **\[1] int64** — vertical 16‑slab index
* `lod`: **\[1] int64** — coarseness token (1 for native 8→16; >1 when parent was coarsened in train)

**Outputs**

* `block_logits`: **\[1, N\_blocks, 16,16,16] float32** — full vocab
* `air_mask`: **\[1,1,16,16,16] float32** — P(air)

> ONNX: **static shapes only**. No dict inputs/outputs.

---

## 📦 Data Extraction Rules

* **Parse truth from MCA**: use `anvil-parser2` to read each section’s `palette` and bitpacked `block_states.data`.
* **Block IDs**: resolve `Name` → integer via **auto‑discovered `block_vocab.json`** (ID 0 reserved for `minecraft:air`). Never hardcode giant enums.
* **Biomes**: use the chunk’s vanilla biomes; collapse to 2D **16×16** surface grid for `biome_patch`.
* **Heightmap**: motion‑blocking height per (x,z); normalize to `[0,1]` by world max Y.
* **Seed‑derived maps**: produce 2D **16×16** multinoise fields (optional in v1; recommended later).
* **Full region sweep**: iterate **32×32 chunks** per region; skip missing/corrupt chunks (don’t synthesize air).
* **Save as NPZ**: `np.savez_compressed` with typed arrays; keep keys stable.

**Required NPZ keys per patch** (minimum):

```
labels16:  int (HWC=[16,16,16])  # block IDs
occ16:     bool/uint8            # (non‑air)
biome16:   int16/int32  (16,16)
height16:  float32      (1,16,16) normalized
river16:   float32      (1,16,16) [optional]
```

---

## 🔁 Building the LOD Pyramid (Targets)

From `labels16` build coarser labels purely for **metrics** and optional heads:

* **Occupancy**: `occ8 = OR_pool(occ16, 2×2×2)`; repeat to get 4³, 2³, 1³.
* **Blocks**: probability pooling preferred (smoother):

  1. one‑hot `labels16` → `[N_blocks,16,16,16]`
  2. `avg_2x2x2` → `[N_blocks,8,8,8]` (and further)
  3. `argmax` per voxel → integer IDs at coarse scale.

**Invariant tests** (unit):

* OR‑pooled occupancy monotonicity
* Pooled class probs sum ≈ 1.0

---

## 🧪 Training Strategy (single static model)

* Always predict **16³** from an **8³** parent.
* Randomly draw **coarsening factor** `f ∈ {1,2,4,8,16}` per sample:

  * `parent_f = OR_pool(occ16, window=f)`  → nearest‑upsample back to **8³**
  * `lod = log2(f)+1`
* **Scheduled sampling** (anti‑drift): with p≈0.1→0.3, derive `occ16` (and `parent_f`) from the model’s **previous** pred (downsampled) instead of truth.
* **Loss**: `CE(block_logits, labels16) + λ_air * BCEWithLogits(air_mask, air_target)`; start `λ_air=0.25`.
* **Metrics**: per‑step (by f) and **full rollout** LOD5→…→LOD0 (self‑fed).

---

## 🧱 Worldgen Rules

* Always write `server.properties` before first boot:

  ```
  level-seed=<numeric>
  generate-structures=false
  ```
* Launch Fabric server headless (`nogui`) with `-Xmx` from config; never hardcode Java path—use `JAVA_HOME` or `which java`.
* Use **Chunky** commands (`center`, `radius`, `start`) and wait for region files to complete before shutdown.
* Generate into a **temp directory**; on success, copy `.mca` into the batch dir; then delete the temp world.

---

## 🧰 Implementation Guardrails

* Python **3.11+**; use `pathlib.Path`; avoid global state when possible.
* Multiprocessing for extraction; never block on per‑chunk logging.
* Keep tensors typed/sized exactly as the contract; validate in `dataset.__getitem__`.
* Limit batch RAM; stream from NPZ; pin shapes in `collate_fn`.
* Respect disk cap (10–20 GB); delete batch outputs after training unless `--keep`.

---

## 🧪 Tests (what to write first)

* **Palette decode**: synthetic bitpacked arrays round‑trip to indices.
* **Downsample invariants**: OR‑pool correctness; prob‑pool vs mode sanity.
* **Dataset contract**: shapes/dtypes are exact; raise if not.
* **Mini E2E**: worldgen(1 region) → extract(≥ 64 chunks) → pair → 1 epoch → export ONNX → onnxruntime forward.

---

## 🚀 Export & Model Config

* Use `train/exporter.py` to export **static** ONNX (opset ≥ 17): ordered inputs/outputs only.
* `scripts/verify_onnx.py` also writes `model_config.json` with:

  * Inputs/outputs (name, dtype, shape, normalization)
  * `block_id↔name` mapping (frozen for this dataset)
  * MC version, git SHA, dataset ID
* Ship `test_vectors.npz` (inputs + outputs) for DJL verification.

---

## 🧱 CI Expectations

Your existing Windows CI runs **lint/typecheck/tests**. Add two jobs (or steps) after tests:

1. **Export ONNX + vectors** (CPU Torch + onnxruntime)
2. **DJL verify**: Java loads ONNX; forward pass OK; (optional) compare to vectors with `max_abs_diff ≤ 1e‑4`.

See `docs/AC.md` for a ready workflow. Keep the Windows runner for parity with your dev environment.

---

## 🔁 TDD Cycle & Commit Rules

* **RED → GREEN → REFACTOR** as three commits per feature.
* Work in `feat/*` branches; merge to `main` only after REFACTOR with docs updated.
* Record assumptions & decisions in commit messages and link to `docs/AC.md` sections when relevant.

---

## 🧼 Do / Don’t

**Do**

* Validate shapes/dtypes aggressively at boundaries.
* Log per‑step metrics (f=1,2,4,8,16) and **rollout** results.
* Delete temp worlds & batches after use.

**Don’t**

* Don’t synthesize air/stone when chunk read fails—**skip** chunk.
* Don’t export ONNX with dynamic axes or dict I/O.
* Don’t hardcode Java paths or OS‑specific separators.

---

## 📎 Snippets Copilot Can Reuse

**3D OR‑pool (2×2×2):**

```python
occ8 = (
  occ16[0::2,0::2,0::2] | occ16[1::2,0::2,0::2] | occ16[0::2,1::2,0::2] | occ16[1::2,1::2,0::2] |
  occ16[0::2,0::2,1::2] | occ16[1::2,0::2,1::2] | occ16[0::2,1::2,1::2] | occ16[1::2,1::2,1::2]
)
```

**Probability pooling (class‑wise avg):**

```python
# logits_onehot: [C,16,16,16] one‑hot from labels
pooled = (
  logits_onehot[:,0::2,0::2,0::2] + logits_onehot[:,1::2,0::2,0::2] +
  logits_onehot[:,0::2,1::2,0::2] + logits_onehot[:,1::2,1::2,0::2] +
  logits_onehot[:,0::2,0::2,1::2] + logits_onehot[:,1::2,0::2,1::2] +
  logits_onehot[:,0::2,1::2,1::2] + logits_onehot[:,1::2,1::2,1::2]
) / 8.0
ids8 = pooled.argmax(0)
```

**Coarsen‑factor sampling in dataset:**

```python
f = random.choice([1,2,4,8,16])
parent_f = or_pool(occ16, f)      # shape: (8//f, 8//f, 8//f)
parent_8 = nearest_upsample(parent_f, out_shape=(8,8,8))
lod = int(math.log2(f)) + 1
```

---

## 🧩 Stretch (post‑Phase‑1)

* Add tiny **surface (2D)** and/or **coarse 8³** heads to color far LODs without 16³.
* Add multinoise inputs (if not already) to tighten long‑range fidelity.
* Explore INT8/FP16 quantization (verify opset + accuracy).

---

**Keep this file aligned with `docs/AC.md`. If they diverge, update both in the same PR.**
