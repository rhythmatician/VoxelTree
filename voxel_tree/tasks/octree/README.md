# Octree Model — Complete Track

## Overview
The octree model is a 3-model architecture for generating Minecraft terrain at multiple detail levels (L0-L4).

## Code Organization

### Model Implementation (`VoxelTree` package)
- **Models**: `VoxelTree.scripts.octree.models` → OctreeInitModel, OctreeRefineModel, OctreeLeafModel
- **Dataset**: `VoxelTree.scripts.octree.dataset` → OctreeDataset, collate_octree_batch
- **Prior Init**: `VoxelTree.scripts.prior_init` → initialization utilities
- **Export**: `VoxelTree.scripts.octree.export` → ONNX/deployment export
- **Pair Building**: `VoxelTree.scripts.build_octree_pairs` → build training pair caches
- **Utilities**: `VoxelTree.scripts.octree.octree_shootout_utils` → evaluation and visualization

### Training Scripts (this folder)
- `train.py` — Main training entry point (reads from `VoxelTree.scripts.octree.*`)
- `convert_fp16.py` — Float16 quantization
- `optimize_onnx.py` — ONNX optimization post-export
- `octree_*_shootout.ipynb` — Model architecture experiments and selection

### Architecture & Design
- `ARCHITECTURE.md` — Design rationale for the 3-model approach
- See `docs/` in root for additional architecture docs

##Training Workflow
1. **Data acquisition** → `pregen`, `voxy_import`, `extract_octree`, `column_heights`
2. **Build pairs** → `build_pairs_octree` (VoxelTree/scripts/build_octree_pairs.py)
3. **Train** → `train_octree` (train.py)
4. **Export** → `export_octree` (VoxelTree/scripts/export_octree.py)
5. **Deploy** → Deploy to Fabric mod (VoxelTree/scripts/deploy_models.py)

## References
- Training data cached in `data/voxy_octree/{train_octree_pairs.npz, val_octree_pairs.npz}`
- Checkpoints saved to configured output directory
- Exported ONNX models suitable for LODiffusion Fabric mod
