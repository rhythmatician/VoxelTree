# Stage-1 Density & Terrain Shaper Models — Complete Track

## Overview
Lightweight density prediction (Stage-1) with optional terrain shaper for rapid, lower-memory terrain generation. Complements octree models as an alternative fast path.

## Code Organization

### Model Implementation (`VoxelTree` package)
- **Training**: Core training loops in this folder (stage1_*.py scripts)
- **Density**: `scripts/stage1/train_density.py` → Density NN training
- **Terrain Shaper**: `scripts/stage1/train_terrain_shaper.py` → Biome/feature shaper
- **Distillation**: `scripts/stage1/distill_density.py` → Knowledge distillation
- **Extraction**: `scripts/stage1/extract_density_weights.py` and `extract_terrain_shaper_weights.py` → Export weights

### Training Scripts (this folder)
- `train_density.py` — Density model training
- `train_terrain_shaper.py` — Terrain shaper training  
- `distill_density.py` — Distillation from octree
- `extract_density_weights.py` — Export density weights
- `extract_terrain_shaper_weights.py` — Export terrain shaper weights
- `density_extraction.py` — Utilities for feature extraction
- `density_nn_shootout.ipynb` — Architecture selection
- `density_nn_experiments.ipynb` — Experimental validation

### Architecture & Design
- `ARCHITECTURE.md` — Stage-1 setup and design (moved from docs/DISTILL_DENSITY_SETUP.md)
- Lightweight models optimized for CPU inference
- Two-stage pipeline: density prediction → terrain refinement

## Training Workflow
1. **Data prerequisite**: Noise dumps from `dumpnoise` step
2. **Build** → `build_pairs_stage1` (uses noise dumps directly, not octree pairs)
3. **Train** → `train_stage1_density` (train_density.py)
4. **Distill** (optional) → `distill_density` (distill_density.py)
5. **Train terrain shaper** (optional) → `train_terrain_shaper` (train_terrain_shaper.py)
6. **Export** → `extract_stage1_weights` (extraction scripts)
7. **Deploy** → Suitable for LODiffusion Fabric mod

## Key Differences from Octree
- **Faster**: Simpler models, fewer parameters
- **More efficient**: Density-based, can skip block-level detail
- **Unified**: Single pass instead of 3-level hierarchy
- **Distillation ready**: Can learn from octree training data

## References
- Input: Noise dumps (` noise_dumps` artifact)
- Output: Density predictions (float32, H×W grid per chunk)
- Optional: Terrain shaper refinement for biome-aware generation
- Checkpoints: Suitable for mixed training with terrain shaper
