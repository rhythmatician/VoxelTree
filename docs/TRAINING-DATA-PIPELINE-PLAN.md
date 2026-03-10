# Training Data Generation Pipeline Plan

**Status:** Planning  
**Target:** Generate high-quality training data using Voxy latent representations

## Overview

This pipeline will generate training data by:
1. Launching headless Fabric server in peaceful mode
2. Setting `generate_structures=false` for terrain-only training
3. Tick-freezing the game (no water/lava flow, no plant growth)
4. Generating chunks using Chunky mod
5. Using Voxy mod to generate LOD latent representations via `/voxy import world`
6. Extracting Voxy latents + vanilla blocks + anchor channels for training

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Training Data Generation Pipeline                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 1. Server Setup & Configuration       │
        │    - Fabric server + mods             │
        │    - Carpet (tick freeze)             │
        │    - Voxy (LOD generation)            │
        │    - Chunky (chunk pregen)            │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 2. World Generation                   │
        │    - Set peaceful mode                │
        │    - Set generate_structures=false   │
        │    - Apply tick freeze (Carpet)       │
        │    - Generate chunks (Chunky)         │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 3. Voxy LOD Generation                │
        │    - Execute /voxy import world       │
        │    - Wait for LOD processing          │
        │    - Verify LOD cache created         │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 4. Data Extraction                    │
        │    - Read .mca files (vanilla blocks) │
        │    - Read Voxy cache (latent LODs)    │
        │    - Extract anchor channels          │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 5. Training Sample Assembly           │
        │    - Combine blocks + Voxy latents    │
        │    - Generate LOD pyramid             │
        │    - Create training samples          │
        └───────────────────────────────────────┘
```

## Components

### 1. Enhanced Server Bootstrap

**File:** `scripts/worldgen/training_server.py`

**Responsibilities:**
- Launch headless Fabric server
- Configure server.properties:
  - `difficulty=peaceful`
  - `generate-structures=false`
  - `spawn-monsters=false`
  - `spawn-animals=false`
- Install required mods:
  - Fabric API
  - Carpet Mod (for tick freeze)
  - Voxy Mod (for LOD generation)
  - Chunky Mod (for chunk pregeneration)
- Execute server commands via stdin/stdout

**Key Methods:**
```python
class TrainingServerBootstrap:
    def __init__(self, seed, world_path, config):
        """Initialize with seed, world path, config"""
    
    def start_server(self) -> bool:
        """Start Fabric server, wait for ready"""
    
    def execute_command(self, command: str) -> str:
        """Execute server command, return output"""
    
    def apply_tick_freeze(self) -> bool:
        """Apply Carpet tick freeze commands"""
    
    def generate_chunks(self, center_x, center_z, radius) -> bool:
        """Use Chunky to generate chunks"""
    
    def import_voxy_world(self, world_name: str) -> bool:
        """Execute /voxy import world command"""
    
    def stop_server(self) -> bool:
        """Gracefully stop server"""
```

### 2. Tick Freeze Configuration

**Carpet Mod Commands:**
```python
TICK_FREEZE_COMMANDS = [
    "/carpet setDefault randomTickSpeed 0",  # Disable random ticks
    "/carpet setDefault doFireTick false",   # Prevent fire spread
    "/carpet setDefault mobGriefing false",  # Prevent mob block changes
    "/carpet setDefault doDaylightCycle false",  # Optional: freeze time
    "/gamerule doMobSpawning false",         # Disable mob spawning
    "/gamerule doMobLoot false",             # Disable mob drops
    "/gamerule doTileDrops false",           # Disable block drops
]
```

**Validation:**
- Wait for command confirmation
- Verify gamerules are set correctly
- Check server logs for errors

### 3. Voxy Integration

**Voxy Command Execution:**
```python
def import_voxy_world(self, world_name: str) -> bool:
    """
    Execute /voxy import world <world_name> command.
    
    This generates LOD representations for the world.
    """
    command = f"/voxy import world {world_name}"
    output = self.execute_command(command)
    
    # Wait for completion
    # Voxy may take time to process large worlds
    self._wait_for_voxy_completion()
    
    # Verify Voxy cache exists
    voxy_cache = self.world_path / ".voxy" / "cache"
    if not voxy_cache.exists():
        raise RuntimeError("Voxy cache not created")
    
    return True
```

**Voxy Cache Location:**
- Default: `world/.voxy/cache/` or `world/voxy_cache/`
- Format: RocksDB with section keys
- Structure: Multi-LOD sections keyed by `(lvl, x, y, z)`

### 4. Voxy Latent Reader

**File:** `scripts/extraction/voxy_reader.py`

**Responsibilities:**
- Read Voxy's RocksDB cache
- Extract LOD sections for all levels
- Convert to numpy arrays
- Map to chunk coordinates

**Key Methods:**
```python
class VoxyLatentReader:
    def __init__(self, voxy_cache_path: Path):
        """Initialize with path to Voxy cache"""
    
    def read_section(self, lvl: int, x: int, y: int, z: int) -> np.ndarray:
        """Read a single LOD section"""
    
    def read_chunk_lods(self, chunk_x: int, chunk_z: int) -> Dict[int, np.ndarray]:
        """Read all LOD levels for a chunk"""
    
    def list_available_sections(self) -> List[Tuple[int, int, int, int]]:
        """List all available (lvl, x, y, z) sections"""
```

**Voxy Section Format:**
- 32×32×32 blocks per section
- Stored as palette + indices
- Need to decode to block IDs or keep as latents

### 5. Training Data Assembler

**File:** `scripts/extraction/training_data_assembler.py`

**Responsibilities:**
- Combine vanilla blocks (.mca) + Voxy latents
- Generate anchor channels (height, biome, etc.)
- Create LOD pyramid from vanilla blocks
- Match Voxy latents to LOD levels
- Export training samples

**Key Methods:**
```python
class TrainingDataAssembler:
    def __init__(self, world_path: Path, voxy_cache: Path):
        """Initialize with world and Voxy cache paths"""
    
    def extract_chunk_data(self, chunk_x: int, chunk_z: int) -> Dict:
        """Extract all data for a chunk"""
    
    def combine_vanilla_voxy(self, vanilla_blocks, voxy_latents) -> Dict:
        """Combine vanilla blocks with Voxy latent representations"""
    
    def generate_training_samples(self, chunk_data: Dict) -> List[Dict]:
        """Generate training samples for all 5 models"""
```

## Implementation Steps

### Step 1: Enhanced Server Bootstrap

1. **Extend `WorldGenBootstrap`** or create `TrainingServerBootstrap`
   - Add Carpet mod installation
   - Add Voxy mod installation
   - Add tick freeze command execution
   - Add Voxy import command execution

2. **Server Command Interface**
   - Implement `execute_command()` method
   - Parse server output for command results
   - Handle async command execution
   - Add timeout handling

3. **Mod Installation**
   - Copy Carpet mod JAR to mods directory
   - Copy Voxy mod JAR to mods directory
   - Verify mods load correctly

### Step 2: Tick Freeze Implementation

1. **Carpet Mod Integration**
   - Execute tick freeze commands on server start
   - Verify commands succeed
   - Add error handling for missing Carpet

2. **Validation**
   - Check gamerules after setting
   - Verify world doesn't evolve (test with wait)
   - Log any warnings

### Step 3: Voxy Integration

1. **Voxy Command Execution**
   - Execute `/voxy import world <name>` after chunk generation
   - Wait for completion (poll server logs or Voxy cache)
   - Handle errors gracefully

2. **Voxy Cache Reading**
   - Research Voxy's cache format (RocksDB)
   - Implement reader for LOD sections
   - Map sections to chunk coordinates

3. **Latent Extraction**
   - Extract all LOD levels for each chunk
   - Convert to numpy arrays
   - Store in training data format

### Step 4: Data Extraction Pipeline

1. **Combine Data Sources**
   - Read vanilla blocks from .mca files
   - Read Voxy latents from cache
   - Extract anchor channels (height, biome, etc.)
   - Align by chunk coordinates

2. **Training Sample Generation**
   - Use existing `dataset_respec.py` pipeline
   - Enhance with Voxy latent integration
   - Generate samples for all 5 models

### Step 5: Integration & Testing

1. **End-to-End Test**
   - Generate small test world (e.g., 10×10 chunks)
   - Verify tick freeze works
   - Verify Voxy generates LODs
   - Verify training samples are created

2. **Validation**
   - Check Voxy latents match vanilla blocks
   - Verify LOD hierarchy is correct
   - Test with multiple seeds

## Configuration

**New Config Section:**
```yaml
training_data:
  server:
    peaceful_mode: true
    generate_structures: false
    tick_freeze: true
    mods:
      carpet: "tools/mods/carpet-1.4.112.jar"
      voxy: "tools/mods/voxy-*.jar"
      chunky: "tools/mods/Chunky-Fabric-1.4.36.jar"
  
  voxy:
    import_command: "/voxy import world {world_name}"
    cache_location: ".voxy/cache"
    wait_timeout: 3600  # seconds
    poll_interval: 5    # seconds
  
  chunk_generation:
    center_x: 0
    center_z: 0
    radius: 16  # chunks
    pattern: "spiral"  # or "grid"
  
  output:
    format: "voxy_latent"  # or "npz", "both"
    include_vanilla_blocks: true
    include_anchors: true
```

## Dependencies

**Required Mods:**
- Fabric API (already in use)
- Carpet Mod (for tick freeze)
- Voxy Mod (for LOD generation)
- Chunky Mod (already in use)

**Python Libraries:**
- `rocksdb` or `plyvel` (for reading Voxy's RocksDB cache)
- Existing: `anvil-parser2`, `numpy`, `yaml`

## File Structure

```
scripts/
├── worldgen/
│   ├── bootstrap.py              # Existing
│   └── training_server.py        # NEW: Enhanced server bootstrap
├── extraction/
│   ├── chunk_extractor.py        # Existing
│   ├── voxy_reader.py            # NEW: Voxy cache reader
│   └── training_data_assembler.py # NEW: Combine all data sources
└── dataset_respec.py             # Existing (enhance with Voxy)
```

## Testing Strategy

1. **Unit Tests**
   - Test tick freeze commands
   - Test Voxy command execution
   - Test Voxy cache reading

2. **Integration Tests**
   - Small world generation (5×5 chunks)
   - Verify tick freeze prevents changes
   - Verify Voxy generates LODs
   - Verify data extraction works

3. **Validation Tests**
   - Compare Voxy latents to vanilla blocks
   - Verify LOD hierarchy consistency
   - Check training sample format

## Success Criteria

✅ Server launches with all required mods  
✅ Tick freeze prevents world evolution  
✅ Chunks generate successfully  
✅ Voxy generates LOD representations  
✅ Voxy latents can be read from cache  
✅ Training samples combine vanilla + Voxy + anchors  
✅ Pipeline runs end-to-end without errors

## Next Steps

1. Research Voxy mod cache format and API
2. Implement enhanced server bootstrap
3. Implement tick freeze commands
4. Implement Voxy command execution
5. Implement Voxy cache reader
6. Integrate with existing dataset pipeline
7. Test end-to-end with small world
8. Scale to full training dataset generation
