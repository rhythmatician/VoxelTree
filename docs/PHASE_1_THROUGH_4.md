Here is the master plan for your staged pre-training. And yes, **several of these can (and should) be trained in parallel** to save time.

---

### 🔀 Phase 1: The Parallel "Feature" Networks

Instead of training one deep network sequentially from the ground up, we start by training three separate, shallow networks. These act as our "feature extractors." Because they don't depend on each other, you can train them simultaneously.

| Stage | Network Target | Input Tensors | Output / Grok Metric |
| --- | --- | --- | --- |
| **1A** | **Macro-Shape Net**<br>

<br>*(Continents, Erosion, Ridges)* | Base Perlin noise maps, X/Z coords | **Target:** Match the intermediate `DensityFunction` outputs.<br>

<br>**Grok Check:** Mean Squared Error (MSE) of continent/erosion values < 0.01. |
| **1B** | **Climate & Biome Net**<br>

<br>*(Temperature, Humidity)* | Climate noise maps, X/Z coords | **Target:** Predict the climate scalar values.<br>

<br>**Grok Check:** 98%+ accuracy on mapping climate vectors to the correct biome cluster. |
| **1C** | **Subtractive Net**<br>

<br>*(Aquifers & Carvers)* | Depth (Y coord), specific cave noise maps | **Target:** Predict the 3D probability of "emptiness" (water/air vs stone).<br>

<br>**Grok Check:** High Intersection over Union (IoU) for cave generation volume. |

**Why this works for the potato:** Shallow networks run lightning-fast on CPUs/low-end GPUs. We are forcing the model to learn the specific mathematical bounds of Minecraft's sub-systems before they ever interact.

---

### 🧬 Phase 2: The `finalDensity` Combiner (Sequential)

Once 1A, 1B, and 1C are fully grokked, we freeze their weights. They are now officially hidden layers. We bolt a new set of layers onto the end of them.

According to your `MINECRAFT_TERRAIN_GENERATION_QUICK_REFERENCE.md`, the magic happens when the `NoiseRouter` combines everything into `finalDensity`.

* **The Target:** We train these new layers to output the exact `finalDensity` scalar value.
* **The Challenge:** The model must learn how a high `erosion` value from Stage 1A flattens the terrain, or how an `aquifer` value from 1C floods a valley.
* **The Benefit:** Because the model already perfectly understands what "erosion" or a "cave" is, it requires vastly fewer parameters (and less training time) to figure out how they blend.

---

### 🧱 Phase 3: The Cell-to-Block Upscaler

Here is a critical optimization from your `MINECRAFT_TERRAIN_GENERATION_ANALYSIS.md` (Section 1.3): Minecraft does *not* calculate noise for every block. It samples at a coarse 4×4×8 cell resolution (768 points per chunk) and interpolates the rest to fill the 98,304 blocks.

We must mirror this hardware-saving trick in the ML pipeline.

* **The Strategy:** Phases 1 and 2 should *only* operate at the low-resolution 4×4×8 cell level.
* **The Upscaler:** We train a final, lightweight "Deconvolution" or "Upsampling" stage. Its entire job is to take the low-res 3D `finalDensity` grid and smooth it out into a high-resolution voxel/octree output.
* **Grok Check:** The output terrain must not have "stair-step" artifacts. The slopes must match vanilla Minecraft's trilinear interpolation.

---

### ⚙️ Phase 4: Fine-Tuning (The "Unfreeze")

Once the entire pipeline is assembled (Input -> Phase 1 Branches -> Phase 2 Combiner -> Phase 3 Upscaler) and generating recognizable terrain, we do one final training run.

* **The Unfreeze:** We unfreeze the weights of all layers, but we use an incredibly small learning rate (e.g., `1e-5`).
* **The Goal:** We let the backpropagation algorithm make micro-adjustments across the entire network to iron out any weird edge cases where caves meet oceans, or where biomes sharply transition.

---

### 🎯 Next Step

This architecture perfectly mimics the logic flow of the `NoiseBasedChunkGenerator` while exploiting ML concepts like feature freezing to keep the parameter count low enough for non-NVIDIA hardware.

To get started on Stage 1, we need to decide what the actual shape of our input tensors will be. **Would you like me to map out the exact input dimensions and data types we'll need to feed into the Phase 1 networks based on the `NoiseRouterExtractor` data?**