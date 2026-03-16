# DataHarvester — Automated Voxy Training Data Extraction

A Fabric client mod + Python orchestration script that automates Voxy LOD
database population for ML training data.  Replaces the manual "connect and
teleport around" step in the data pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  harvest.py (Python orchestrator)                                   │
│  ┌──────────┐  RCON   ┌────────────────────┐  chunks  ┌─────────┐ │
│  │ teleport  │───────▶│  Fabric Server      │────────▶│ Voxy    │ │
│  │ spiral    │        │  + Chunky           │  (LOD)  │ client  │ │
│  │ + monitor │        │  + VoxyWorldGen v2  │────────▶│ RocksDB │ │
│  └──────────┘        └────────────────────┘          └─────────┘ │
│       │                        ▲                          │       │
│       │              MC client connects                   │       │
│       │              (DataHarvester mod                   ▼       │
│       │               auto-connects)               *.sst files   │
│       │                                            (training     │
│       └── monitors DB growth ──────────────────── data ready)    │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Fabric server** runs with Chunky (pregeneration) and VoxyWorldGen v2
   (generates chunks around players and streams LOD data to clients).
2. **DataHarvester mod** auto-connects to the server when launched.
3. **harvest.py** teleports the player via RCON in a spiral pattern,
   covering the desired area.  VoxyWorldGen generates chunks around each
   position and streams them to the client's Voxy instance.
4. **Voxy** (client-side) ingests the LOD data into its RocksDB database.
5. **harvest.py** monitors the RocksDB `.sst` files until the database
   stabilises (no growth for 60s), then signals completion.

## Quick Start

### 1. Build & install the DataHarvester mod

```powershell
cd VoxelTree\tools\data-harvester
.\build_and_install.bat
```

This builds the Fabric mod JAR and copies it to the Modrinth
"LODiffusion dependencies" profile's `mods/` folder.

### 2. Start the Fabric server

```powershell
cd VoxelTree\tools\fabric-server\runtime
java -jar fabric-server-mc.1.21.11-loader.0.18.4-launcher.1.1.1.jar nogui
```

### 3. Run the harvest

```powershell
# Terminal 1: Start the harvest orchestrator
python -m VoxelTree.preprocessing.harvest --password voxeltree --radius 2048

# Terminal 2: Launch Minecraft via Modrinth App
#   (the DataHarvester mod will auto-connect to localhost:25565)
```

Or via the unified CLI:

```powershell
python -m VoxelTree.preprocessing.cli harvest --password voxeltree --radius 2048
```

### 4. Continue the pipeline

Once the harvest completes, extract training data:

```powershell
python -m VoxelTree.preprocessing.cli dataprep --from-step extract-octree
```

## Configuration

### DataHarvester mod config

Placed at `config/dataharvester.json` in the Minecraft profile directory:

```json
{
  "serverAddress": "localhost:25565",
  "autoConnect": true,
  "autoConnectDelaySec": 5,
  "reconnectOnDisconnect": true,
  "reconnectDelaySec": 10
}
```

| Field | Default | Description |
|---|---|---|
| `serverAddress` | `localhost:25565` | Server to connect to |
| `autoConnect` | `true` | Auto-connect on game start |
| `autoConnectDelaySec` | `5` | Seconds to wait after title screen loads |
| `reconnectOnDisconnect` | `true` | Auto-reconnect if disconnected |
| `reconnectDelaySec` | `10` | Seconds between reconnect attempts |

### Harvest script options

```
  --radius BLOCKS       Harvest radius in blocks (default: 2048)
  --step BLOCKS         Teleport step size in blocks (default: 256)
  --dwell SECS          Seconds at each teleport position (default: 8)
  --stable-seconds SECS DB must be stable for N seconds (default: 60)
  --skip-pregen         Skip Chunky pregeneration
  --spiral-only         Skip pregen + player wait; just run spiral
  --monitor-only        Only monitor Voxy DB growth
```

## Project Structure

```
VoxelTree/tools/data-harvester/
├── build.gradle                 # Fabric Loom project
├── gradle.properties            # MC 1.21.11, Fabric 0.18.4
├── settings.gradle
├── build_and_install.bat        # Build + deploy to Modrinth profile
├── gradle/wrapper/
│   └── gradle-wrapper.properties
└── src/main/
    ├── java/com/voxeltree/harvester/
    │   ├── DataHarvesterClient.java    # ClientModInitializer
    │   ├── AutoConnectHandler.java     # Auto-connect + reconnect logic
    │   ├── HarvesterConfig.java        # Config POJO
    │   └── mixin/
    │       └── TitleScreenMixin.java   # Title screen logging
    └── resources/
        ├── fabric.mod.json
        └── dataharvester.mixins.json

VoxelTree/VoxelTree/preprocessing/
├── harvest.py                   # Python orchestration (teleport spiral, DB monitoring)
├── rcon.py                      # RCON client (used by harvest.py)
└── cli.py                       # Unified CLI (harvest subcommand added)
```

## Teleport Spiral Pattern

The harvest script teleports the player in a square spiral:

```
     step=256 blocks
  ╭───────────────╮
  │ 25  10  11  12│
  │ 24   9   2   3│
  │ 23   8   1   4│  ← starts at center
  │ 22   7   6   5│
  ╰───────────────╯
```

At each position, VoxyWorldGen generates chunks in a radius defined by
`generationRadius` in `voxyworldgenv2.json` (default: 128 chunks = 2048
blocks).  The `--step` parameter controls the distance between teleport
points — 256 blocks is conservative overlap ensuring full coverage.

## Estimated Times

| Radius | Positions | ~Time (8s dwell) |
|--------|-----------|------------------|
| 512    | 16        | ~2 min           |
| 1024   | 64        | ~9 min           |
| 2048   | 256       | ~34 min          |
| 4096   | 1024      | ~2.3 hours       |

Plus Chunky pregen time (if not skipped) and Voxy DB stabilisation.

## Prerequisites

- **Java 21+** on PATH
- **Fabric server** with: Chunky, VoxyWorldGen v2, Fabric API
- **Modrinth profile** "LODiffusion dependencies" with: Voxy, VoxyWorldGen v2, Fabric API
- **Server**: `online-mode=false`, `gamemode=spectator`, `enable-rcon=true`
- **Python 3.10+** with the VoxelTree package on the path
