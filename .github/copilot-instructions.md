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
* **Truthful labels**: extract from **Voxy RocksDB** databases (canonical vocabulary); **no synthetic fallbacks**.
* **Just‑in‑time**: far LODs avoid the heavy 16³ head; refine only when entering near bands.

---

## 🧭 Acceptance Criteria (excerpt)

* **LOD1→LOD0 (8³→16³)**: Air/solid IoU ≥ 0.99, frequent‑set block top‑1 ≥ 0.99, overall ≥ 0.98.
* **Coarser transitions (incremental refinement)**:
  * LOD2→LOD1: frequent‑set ≥ 0.985
  * LOD3→LOD2: frequent‑set ≥ 0.98
  * LOD4→LOD3: frequent‑set ≥ 0.975
* **Full rollout LOD4→LOD3→LOD2→LOD1→LOD0** (self‑fed): frequent‑set at final LOD0 ≥ 0.985.
* **Runtime**: ≤ 100 ms median per 16³; ≤ 150 ms p95; ≤ 2 MB incremental mem.
* **Integration**: DJL harness loads ONNX; `max_abs_diff ≤ 1e‑4` vs PyTorch on test vectors.

> Full AC lives in `docs/AC.md`. Keep this file in sync.

---

## 🗂 Directory & Files (authoritative)

```
VoxelTree/
├─ pipeline.py                   # Two-phase orchestrator: extract→train→export→deploy
├─ train_multi_lod.py            # CLI: multi-LOD training with Voxy vocab
├─ config_multi_lod.yaml         # Model + training config
├─ config/
│  └─ voxy_vocab.json            # Canonical Voxy-native block vocabulary (1102 entries)
├─ train/
│  ├─ multi_lod_dataset.py       # Loads NPZ patches; builds multi‑LOD inputs
│  ├─ unet3d.py                  # 8→16 SR UNet (SimpleFlexibleUNet3D)
│  ├─ anchor_conditioning.py     # Height planes + router6 conditioning
│  ├─ losses.py                  # CE + air losses
│  └─ metrics.py                 # Per‑step and rollout metrics
├─ scripts/
│  ├─ extract_voxy_training_data.py  # Voxy RocksDB → NPZ patches
│  ├─ voxy_reader.py             # RocksDB reader (SaveLoadSystem3 decoder)
│  ├─ mipper.py                  # Voxy Mipper (canonical LOD coarsening)
│  ├─ export_lod.py              # Static ONNX export (opset ≥17)
│  ├─ verify_onnx.py             # export + test_vectors + model_config
│  ├─ extraction/                # Legacy MCA extraction (kept for reference)
│  └─ worldgen/bootstrap.py      # Fabric + Chunky; structures=false
├─ docs/AC.md
├─ tests/                        # PyTest (unit + mini E2E)
├─ models/                       # checkpoints/onnx (ignored in git)
├─ data/                         # temp batch data (ignored)
└─ tools/                        # JARs, CLI (Chunky, Fabric, cubiomes)
```

---

## 🧠 Model I/O Contract (v2 — Anchor Conditioning)

**Inputs**

* `x_parent`: **\[1,1,8,8,8] float32** — binary occupancy (Mipper-derived from child or prev pred)
* `x_height_planes`: **\[1,5,16,16] float32** — surface, ocean\_floor, slope\_x, slope\_z, curvature
* `x_router6`: **\[1,6,16,16] float32** — temperature, vegetation, continents, erosion, depth, ridges
* `x_biome`: **\[1,16,16] int64** — vanilla biome index per (x,z)
* `x_y_index`: **\[1] int64** — vertical 16‑slab index
* `x_lod`: **\[1] int64** — coarseness token (1 for native 8→16; >1 when parent was coarsened in train)

**Outputs**

* `block_logits`: **\[1, 1102, 16,16,16] float32** — Voxy-native vocab (1102 block types)
* `air_mask`: **\[1,1,16,16,16] float32** — P(air)

> ONNX: **static shapes only**. No dict inputs/outputs.
> Block vocabulary: 1102 entries from canonical Voxy vocabulary (`config/voxy_vocab.json`), air=0.

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

From `labels16` build coarser labels using the **Voxy Mipper algorithm** (not OR-pool or
probability pooling).

* **Single source of truth**: `scripts/mipper.py` — `mip_once_numpy`, `mip_volume_numpy`,
  `mip_volume_torch`.
* **Opacity tiers**: air = 0, water/lava/glass/leaves/etc. = 1, all other solids = 15.
* **Selection rule**: score = `(opacity << 4) | corner_priority` — highest score wins.
  When all 8 corners are identical opacity the **I₁₁₁ corner** (axis-order: x=1,z=1,y=1)
  always wins (priority 7).
* **Occupancy**: derived automatically from the Mipper output (non-air ⇒ occupied).

**Invariant tests** (unit):

* Mipper occupancy is ≤ OR-pool occupancy (it can only be equal or denser, never more)
* All-air input → all-air output
* Single opaque voxel in a 2³ window → that voxel wins

---

## ✦✦✦✦ Training Strategy (single static model — incremental refinement)

* Always predict **16³** from an **8³** parent (both nearest-upsampled to canonical sizes).
* Each training sample represents a **single LOD step**: the target is one level
  finer than the parent, not always LOD0.
* Randomly draw **coarsening factor** `f ∈ {2,4,8,16}` per sample:

  * Parent: `mip_volume_numpy(labels16, f)` → nearest‑upsample to **8³**
  * Target: `mip_volume_numpy(labels16, f//2)` → nearest‑upsample to **16³**
    (for f=2 the target is `labels16` itself — full‑detail LOD0)
  * `lod_token = log2(f)` (1,2,3,4)
  * `lod_transition = "lod{N}to{N-1}"` (lod1to0, lod2to1, lod3to2, lod4to3)
* At runtime, LODiffusion chains: LOD4→LOD3→LOD2→LOD1→LOD0, each step feeding
  the previous prediction as the next parent.
* **Scheduled sampling** (anti‑drift): with p≈0.1→0.3, derive the parent from the model's
  **previous** pred (run argmax on logits, then `mip_volume_torch`) instead of truth.
* **Loss**: `CE(block_logits, target_labels) + λ_air * BCEWithLogits(air_mask, target_occ)`; start `λ_air=0.25`.
* **Metrics**: per‑step (by transition) and **full rollout** LOD4→…→LOD0 (self‑fed).

---

## 🧱 Worldgen Rules

* Always write `server.properties` before first boot:

  ```
  level-seed=<numeric>
  generate-structures=false
  difficulty=peaceful
  spawn-npcs=false
  spawn-animals=false
  spawn-monsters=false
  ```
* **Terrain purity** (see `tests/test_terrain_purity.py`):
  After server ready, **before Chunky starts**, send:
  1. Gamerules: `randomTickSpeed 0`, `doFireTick false`, `mobGriefing false`,
     `doWeatherCycle false`, `doDaylightCycle false`, `doMobSpawning false`
  2. `/tick freeze` (MC 1.20.3+) — halts ALL ticking (water flow, leaf decay, etc.)
  This ensures chunks are saved in their exact as-generated state.
* Constants live in `scripts/worldgen/bootstrap.py`: `TERRAIN_PURITY_GAMERULES`,
  `TICK_FREEZE_COMMAND`, `STRUCTURE_ONLY_BLOCKS`.
* Launch Fabric server headless (`nogui`) with `-Xmx` from config; never hardcode Java path—use `JAVA_HOME` or `which java`.
* Use **Chunky** commands (`center`, `radius`, `start`) and wait for region files to complete before shutdown.
* Generate into a **temp directory**; on success, copy `.mca` into the batch dir; then delete the temp world.

---

## 🧰 Implementation Guardrails

* Python **3.11+**; use `pathlib.Path`; avoid global state when possible.
* `rocksdict` + `zstandard` for Voxy RocksDB extraction; `torch` + `numpy` for training.
* Keep tensors typed/sized exactly as the contract; validate in `dataset.__getitem__`.
* Limit batch RAM; stream from NPZ; pin shapes in `collate_fn`.
* Respect disk cap (10–20 GB); delete batch outputs after training unless `--keep`.
* Use `pipeline.py` to orchestrate extract→train→export→deploy.

---

## 🧪 Tests (what to write first)

* **Terrain purity**: constants are complete; gamerules + tick freeze sent before Chunky;
  server.properties has all required settings; structure-only blocks detectable in vocab.
* **Mipper invariants**: all-air→all-air; single opaque voxel wins; occupancy ≤ OR-pool.
* **Voxy extraction**: round-trip vocab mapping; NPZ shapes/dtypes match contract.
* **Dataset contract**: shapes/dtypes are exact; raise if not.
* **Mini E2E**: extract(Voxy DB, ≥ 64 chunks) → pair → 1 epoch → export ONNX → onnxruntime forward.

---

## 🚀 Export & Model Config

* Use `scripts/export_lod.py` to export **static** ONNX (opset ≥ 17): ordered inputs/outputs only.
* Export also writes `model_config.json` with:

  * Inputs/outputs (name, dtype, shape, normalization)
  * `block_mapping` from `config/voxy_vocab.json` (Voxy-native block→ID)
  * `block_id_to_name` reverse mapping
  * MC version, git SHA, dataset ID
* Ship `test_vectors.npz` (inputs + outputs) for DJL verification.
* LODiffusion's `VoxyBlockMapper` reads `block_mapping` from `model_config.json` at runtime.

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

**Voxy Mipper (canonical LOD coarsening):**

```python
from scripts.mipper import build_opacity_table, mip_volume_numpy, mip_volume_torch

# NumPy (data pipeline, extraction)
tbl = build_opacity_table(n_blocks=4096)          # air=0, transparent=1, solid=15
labels8, occ8 = mip_volume_numpy(labels16, 2, tbl)  # 16³ → 8³
labels4, occ4 = mip_volume_numpy(labels16, 4, tbl)  # 16³ → 4³  (recursive inside)

# PyTorch (training, inference)
tbl_t = torch.from_numpy(tbl).long().to(device)
labels8_t, occ8_t = mip_volume_torch(labels16_t.long(), 2, tbl_t)  # (B,D,H,W), (B,D//2,H//2,W//2)
```

**Incremental refinement pair generation in dataset:**

```python
from scripts.mipper import build_opacity_table, mip_volume_numpy
import math, numpy as np

tbl = build_opacity_table(n_blocks=4096)
for f in (2, 4, 8, 16):
    # Parent: LOD N, coarsened by f, upsampled to 8³
    parent_labels, parent_occ = mip_volume_numpy(labels16, f, tbl)
    parent_8 = np.repeat(np.repeat(np.repeat(
        parent_occ, 8 // parent_occ.shape[0], axis=0),
        8 // parent_occ.shape[1], axis=1),
        8 // parent_occ.shape[2], axis=2)  # → 8³

    # Target: LOD N-1 (one level finer), upsampled to 16³
    f_target = f // 2
    if f_target == 1:
        target = labels16  # LOD0, already 16³
    else:
        tgt, _ = mip_volume_numpy(labels16, f_target, tbl)
        up = 16 // tgt.shape[0]
        target = np.repeat(np.repeat(np.repeat(tgt, up, 0), up, 1), up, 2)

    lod_token = int(math.log2(f))  # 1, 2, 3, 4
    transition = f"lod{lod_token}to{lod_token - 1}"  # lod1to0, lod2to1, ...
```

---

## 🧩 Stretch (post‑Phase‑1)

* Add tiny **surface (2D)** and/or **coarse 8³** heads to color far LODs without 16³.
* Add multinoise inputs (if not already) to tighten long‑range fidelity.
* Explore INT8/FP16 quantization (verify opset + accuracy).

---

**Keep this file aligned with `docs/AC.md`. If they diverge, update both in the same PR.**
