# VoxelTree — Copilot Instructions

## Philosophy
Rapid prototyping, not production code. **Prefer replacing code over adding code.** Remove orphaned code from prior agents. Keep total model count minimal — revise existing models and their I/O contracts rather than creating new ones. Every model has 4 canonical steps: `build_pairs → train → export → deploy`.

Align model I/O contracts with Vanilla Minecraft's actual sampling grid (noise at lower resolution, interpolated to block grid). Final output targets Voxy's sparse octree format.

## Architecture Overview
VoxelTree is a Python 3.11+ / PyTorch training pipeline. It exports ONNX models + JSON sidecar configs to `LODiffusion/run/config/lodiffusion/` for in-game inference.

**4 model tracks**, each with 4 canonical steps auto-wired via artifact dependencies:

| Track | Model | Key I/O |
|-------|-------|---------|
| `sparse_octree` | `SparseOctreeFastModel` | 15ch noise 4×2×4 + biome + heightmap → 5-level octree |
| `density` | `DensityMLP` | 6 climate → PSL + final_density |
| `biome_classifier` | `BiomeClassifier` | 6 climate → 54 biome logits |
| `heightmap_predictor` | `HeightmapPredictor` | 96 climate grid → 32 heights |

**Data acquisition** (shared pipeline feeding all tracks):
```
pregen → harvest (voxy_db) → extract_octree → column_heights → build_v7_pairs
dumpnoise (noise_dumps) ──────────────────────────────────────┘
```

## Critical Patterns

### Contract system (`voxel_tree/contracts/`)
Single source of truth for model I/O shapes between Python and Java. Every export produces a `_config.json` sidecar with `contract` field — Java's `ConfigLoader` throws without it.
- Register contracts in `catalog.py`; query with `get_contract(name, rev)`
- `ModelTrack.contract_revision` is validated at step execution — stale = warning, missing = error
- Don't create manual prereqs on `StepDef` — `_wire_prereqs()` auto-computes from `produces`/`consumes`

### Profile-driven configuration
All step configuration flows through YAML profiles in `profiles/`. No per-step CLI flags. Step functions receive the full profile dict via stdin from `step_runner.py`.

### Lazy imports
Use function-level imports (`# noqa: PLC0415`) to keep CLI startup fast. The contract catalog loads on first access via `_ensure_catalog_loaded()`.

### Block vocabulary
`voxel_tree/config/voxy_vocab.json` — 1104-entry canonical mapping (`minecraft:air = 0`). Shared with Java's `BlockVocabulary.java`. Always use `num_classes=1104`.

### RouterField channel ordering
The **shared contract** lives at `../router_field_contract.yaml` (workspace root, outside both repos). It defines the canonical 15-channel ordering, normalization, and spatial layout. Python validates against it at training init via `voxel_tree.utils.contract_validator.validate_router_contract()`. Java's `RouterField.java` must match exactly. Any mismatch is a hard error.

## Build & Run
```bash
pip install -e .                                    # Install package
voxel-tree                                          # Launch PySide6 GUI (no args)
voxel-tree --step train_sparse_octree --run --profile phase6  # Run step via CLI
voxel-tree contracts sparse_octree                  # Inspect contracts
pytest voxel_tree/                                  # Run tests
```

Fabric server for data acquisition lives in `tools/fabric-server/` — start via `voxel-tree --server start --role train`.

## Gotchas
- `spatial_y=2` not 4 — v7 contract rev 3 is production. Rev 1 (4×4×4) was speculative and never matched real data.
- `noise_2d` is vestigial (zero channels) but must be included in ONNX export for backward compat.
- GUI + CLI coexist: `voxel-tree` with no args = GUI, with args = CLI. Step runner subprocess bridge isolates execution.
- Sparse octree track has a 5th non-canonical step (`distill_sparse_octree`) via `extra_steps`.
- Java auto-detects v6 (13ch, `split`) vs v7+ (15ch, `occ`) based on ONNX output names — hot-swap friendly.
- `reference-code/` (sibling to VoxelTree) contains Minecraft and Voxy source for implementation guidance.