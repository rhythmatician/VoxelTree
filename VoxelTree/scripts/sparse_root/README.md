# Sparse Root Model — Complete Track

## Overview
Fast, spatial-pruning sparse voxel tree model for Minecraft terrain refinement. Complements the octree models with hierarchical block-type predictions.

## Code Organization

### Model Implementation (`VoxelTree` package)
- **Training**: `VoxelTree.scripts.sparse_root.sparse_root_train` → Training loop, data handling
- **Distillation**: `VoxelTree.scripts.sparse_root.sparse_root_distill` → Knowledge distillation from octree
- **Utilities**: `VoxelTree.scripts.sparse_octree_targets` → Ground truth generation

### Training Scripts (this folder)
- `train.py` — Training entry point launched via GUI (reads from `VoxelTree.scripts.sparse_root.*`)
- `distill.py` — Distillation from octree checkpoint
- `calibrate.py` — Threshold calibration for split decisions
- `calibrate_split_threshold.py` — Detailed threshold analysis
- `diagnose.py` — Debug/diagnostic utilities
- `diagnose_sparse_root_data.py` — Data validation
- `list_biomes_in_training_set.py` — Dataset inspection
- `sparse_root_exploration.ipynb` — Architecture experiments
- `sparse_root_runtime_tradeoffs.ipynb` — Speed vs. accuracy analysis

### Architecture & Design
- `README_distill.md` — Distillation methodology
- See `docs/` in root for additional architecture docs

## Training Workflow
1. **Data prerequisite**: Octree checkpoint from `train_octree`
2. **Build pairs** → `build_pairs_sparse_root` (creates input data)
3. **Train** → `train_sparse_root` (train.py)
4. **Distill** → `distill_sparse_root` (distill.py) — optional knowledge distillation
5. **Export** → (implement via export script)
6. **Deploy** → (implement via deploy script)

## Calibration & Tuning
The `calibrate*.py` scripts are for spatial pruning threshold tuning. Use to balance:
- Memory efficiency (fewer voxels)
- Detail retention (pruning thresholds)
- Runtime performance

## Split-first training defaults

The sparse-root objective is intentionally biased toward structural sparsity.

- Global lambda defaults: `split_weight=1.0`, `label_weight=0.35`
- Material CE is leaf-masked: label loss is only computed where `split_target == 0`
- Suggested first-pass acceptance gates:
  - `split_f1 >= 0.90`
  - `split_under_rate <= 0.05`
  - `leaf_node_ratio` (pred/gt) in `[0.95, 1.10]`
  - `leaf_acc >= 0.75`

Use `train.py --split-weight ... --label-weight ...` to sweep these values.

## References
- Training pairs from octree-derived targets (VoxelTree.scripts.sparse_octree_targets)
- Checkpoints needed: prior octree checkpoint
- Output: sparse root checkpoint suitable for refinement
