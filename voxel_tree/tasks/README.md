# Scripts

Standalone CLI scripts organized by **model track** (swim lane).

| Folder | Track | Description |
|--------|-------|-------------|
| `terrain_shaper/` | Stage-1 Density | Train, extract, distill the tiny density NN + terrain shaper |
| `voxy/` | Sparse Root | Train, distill, calibrate the sparse-root occupancy model |
| `octree/` | Octree (init/refine/leaf) | Post-export ONNX optimization and FP16 conversion |
| `data_acq/` | Data Acquisition | Voxy utilities, world-data checks |

> **Note:** The core octree pipeline scripts (`build_octree_pairs`, `export_octree`,
> `deploy_models`) live inside the VoxelTree Python package at
> `VoxelTree/VoxelTree/scripts/` and are invoked via `python -m VoxelTree <cmd>`.
> Infrastructure tools (Fabric server, Hephaistos, cubiomes CLI) remain in `tools/`.

