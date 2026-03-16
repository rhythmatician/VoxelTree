# Data Acquisition Pipeline — Shared Across All Models

## Overview
Data acquisition (pregen, voxy import, noise dumps, heightmap extraction) is shared infrastructure that feeds all model tracks. Located in workspace `scripts/data_acq/` but logic is in VoxelTree core preprocessing.

## Code Organization

### Core Implementation (`VoxelTree` package)
- **Preprocessing**: `VoxelTree.preprocessing.pipeline` → Orchestration
- **Pre-generation**: `VoxelTree.cli.pregen` → Minecraft world generation
- **Voxel Import**: `VoxelTree.cli.voxy-import` → Voxel octree database import
- **Noise Dumps**: `VoxelTree.cli.dumpnoise` → Minecraft noise function dumps
- **Octree Extract**: `VoxelTree.scripts.extract_octree_data` → NPZ voxel extraction
- **Heightmap**: `VoxelTree.scripts.add_column_heights` → Heightmap generation

### Scripts (this folder)
- `copy_voxy_test_subset.py` — Test data subset creation
- (Notebooks for data exploration/validation would go here)

## artifact Dependencies
Each data-acq step produces artifacts consumed by downstream models:

- **pregen** → `mc_world` (Minecraft world save)
- **voxy_import** → `voxy_db` (requires `mc_world`)
- **dumpnoise** → `noise_dumps` (Minecraft noise function values)
- **extract_octree** → `octree_npz` (requires `voxy_db`)
- **column_heights** → `octree_with_heights` (requires `octree_npz` + `noise_dumps`)

## Used By
All models consume data from this pipeline:
- **Octree models**: train on `octree_with_heights` pairs
- **Stage-1 density**: trains on `noise_dumps` directly
- **Sparse root**: refines `octree` models

## Workflow
```
pregen → voxy_import ──┐
dumpnoise ────────────┤
                      ├→ extract_octree → column_heights → [all models]
```

##  Data Locations
- Minecraft world: `{data_dir}/` (game save directory)
- Voxel database: `{data_dir}/voxy_octree/` (Level 0 voxel grid)
- Noise dumps: `{noise_dump_dir}/` (JSON files for each dimension)
- Pair caches: `{data_dir}/train_*.npz, val_*.npz` (per-model training data)

## Configuration
- RCON settings: Server manager  for RCON commands (pregen, voxy_import, dumpnoise all require server access)
- Data directories: profile → `data.*` sections
- Noise dimensions: configurable per Minecraft version

## Server Requirements
The following steps require a running Minecraft server with Fabric mods:
1. `pregen` — Terrain pre-generation (RCON: `forceload`, worldborder expansion, fill)
2. `voxy_import` — Live octree data capture from voxel-tap stream (RCON: modconfig)
3. `dumpnoise` — Noise function values (RCON: `playsound` for state, custom mod API)
