# Worldgen Pipeline (Fabric + Chunky + Optional Carpet/Voxy)

This document describes the repeatable **generate terrain → freeze sim → run post-commands → export artifacts** workflow for VoxelTree.

## Configuration (`config.yaml`)

The worldgen pipeline is driven by `worldgen` settings in `config.yaml`:

```yaml
worldgen:
  seed: 6901795026152433433
  java_heap: "4G"
  generate_structures: false
  gamemode: "creative"
  difficulty: "peaceful"
  view_distance: 6
  simulation_distance: 4
  java_tools:
    primary: "tools/fabric-server/..."
    chunky: "tools/fabric-server/runtime/mods/Chunky-Fabric-1.4.36.jar"
  mods:
    - "tools/fabric-server/runtime/mods/fabric-api-*.jar"
    - "tools/fabric-server/runtime/mods/Chunky-Fabric-*.jar"
    - "tools/fabric-server/runtime/mods/fabric-carpet-*.jar"
    - "tools/fabric-server/runtime/mods/voxy-*.jar"
```

Key notes:

* `generate_structures: false` makes terrain-only worlds by writing `generate-structures=false` to `server.properties`.
* `gamemode`, `difficulty`, `view_distance`, and `simulation_distance` are optional and default to the current server defaults if omitted.
* `mods` is a **list of paths or glob patterns**. Every resolved JAR is copied into the temp server `mods/` directory; missing paths are logged as warnings.
* `java_tools.primary` (Fabric server) and `java_tools.chunky` are **required** for generation. Optional mods (Carpet/Voxy) may be omitted.

## Running the pipeline

Use the orchestration script:

```bash
python scripts/worldgen/generate_world_and_prepare_voxy_dataset.py \
  --center-x 0 \
  --center-z 0 \
  --radius 5 \
  --world-name voxeltree_world
```

Outputs:

* `.mca` region files are copied to `data/test_world/region/`.
* A manifest is written to `data/test_world/run_manifest.json` with seed, MC version (if detected), run parameters, copied mods, and timestamps.

## Voxy caveat (dedicated server)

Voxy commands are commonly **client-only**, and may not be registered on a dedicated server. The pipeline:

* Attempts to run a Voxy import command only if a Voxy JAR is configured.
* Scans server output for “unknown command” warnings.
* Logs a clear warning if `/voxy` commands are unavailable:
  * “/voxy commands are not available on this dedicated server; you likely need a client/integrated-server environment for Voxy import/export. Proceeding with only .mca output.”

If you need Voxy import/export, plan for a future step that drives a client or integrated server.
