## 🎯 Training Goal: Learn to Refine Terrain via LOD Steps

> The model learns to take **seed-derived inputs** + **existing coarse terrain** and output a **finer resolution** version of that terrain.

Each step is a **progressive LOD refinement**, not full terrain generation from scratch.

---

## 🧠 Model Input and Output — Per Training Sample

| Component         | Source                   | Description                                  |
| ----------------- | ------------------------ | -------------------------------------------- |
| `parent_voxel`    | From `.mca`, downsampled | Coarse (8×8×8) air/solid mask or block types |
| `biome_patch`     | From seed                | Biome IDs for 16×16 area                     |
| `heightmap_patch` | From seed                | Surface heightmap for 16×16 block area       |
| `y_index`         | From coordinates         | Which vertical subchunk this is (0–23)       |
| `lod`             | Set by training loop     | Current LOD step / timestep                  |
| `target_mask`     | From `.mca`              | High-res (16×16×16) air/solid mask           |
| `target_types`    | From `.mca`              | High-res block type IDs or one-hot logits    |

> Each `.mca` yields many of these input-output **subchunk patches**.

---

## 🔁 Training Loop: Progressive Diffusion-Like Refinement

| Step | Operation                                      |
| ---- | ---------------------------------------------- |
| 1️⃣  | Sample `(x, z, y)` + LOD level                 |
| 2️⃣  | Load input features (biome, height, y-index)   |
| 3️⃣  | Load coarse `parent_voxel`                     |
| 4️⃣  | Load target `target_mask` and `target_types`   |
| 5️⃣  | Train on: **parent + context → refined voxel** |

---

## 🏗️ Dataset Construction Loop

For each `.mca` region:

1. Slice into 16×16×16 subchunks
2. For each subchunk:

   * Downsample to 8×8×8 (or LOD-coarsened parent)
   * Save `.npz`:

     ```python
     {
       'parent_voxel': (8, 8, 8),
       'target_mask': (16, 16, 16),
       'target_types': (16, 16, 16),
       'biome_patch': (16, 16),
       'heightmap': (16, 16),
       'y_index': int,
       'lod': int
     }
     ```

---

## ✅ Summary: What the Model Learns

> “Given the coarse voxel state of a subchunk and what I know from the seed (biomes, heightmap, etc.), **what should the refined block layout look like?**”

That’s it — no need to learn worldgen, no need to understand cave carving logic, just **refine coarse → fine** in the way Minecraft worlds naturally do.

---

## Optional Later Features

* Conditioning on nearby LODs (neighboring subchunks)
* Using diffusion noise schedules to learn denoising steps
* Predicting uncertainty or diversity (e.g., with dropout)
