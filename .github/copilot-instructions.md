# 🤖 Copilot Instructions — VoxelTree

This file guides GitHub Copilot (and contributors) in writing code for the `VoxelTree` project. It ensures consistency, avoids architectural pitfalls, and enforces our test-driven development (TDD) cycle.

---

## 🌲 Project Purpose

VoxelTree implements and trains a **LOD-aware, voxel super-resolution model** for Minecraft terrain. It progressively refines terrain data from coarser representations (e.g., 8×8×192) to higher fidelity forms (e.g., 16×16×384), conditioned on biome and heightmap data.

The trained model is exported to ONNX and used inside the LODiffusion Minecraft mod for **real-time, just-in-time terrain generation**.

---

## ✅ Core Constraints

- ⚙️ Use Python 3.10+
- 📦 Store extracted chunk data as `.npz` (use `np.savez_compressed`)
- 📁 Use `pathlib.Path` over `os.path`
- 🧵 Use `multiprocessing` (not threading) for batch `.mca` extraction
- 🚮 Never exceed 10–20 GB of disk usage during training
- 🧠 No deep learning during extraction — keep it fast and CPU-only
- 🔁 Only train on a **subset of chunks at a time**, then delete

---

## 🧪 Test-Driven Development (TDD)

Each feature is developed in a 3-phase cycle:

1. **RED** — Write a failing test
2. **GREEN** — Write just enough code to pass the test
3. **REFACTOR** — Reflect on structure, log insights, and update docs

### Commit Rules

- Every phase (`RED`, `GREEN`, `REFACTOR`) must be a separate commit
- Each TDD cycle must occur in a **feature branch** (e.g. `feat/mca-loader`)
- Only merge to `main` after REFACTOR is complete and documented

---

## 🗂️ Directory Overview

VoxelTree/
├── train/
│ ├── train.py
│ ├── dataset.py
│ ├── unet3d.py
│ ├── loss.py
│ └── config.yaml
├── scripts/
│ ├── export_onnx.py
│ ├── run_eval.py
│ └── generate_samples.py
├── models/ # Trained checkpoints and ONNX exports
├── tests/ # PyTest test suite
├── data/ # Temporary patch/chunk files
├── PROJECT-OUTLINE.md
├── README.md
└── .gitignore

yaml
Always show details

Copy

---

## 🧠 Model Design Guidelines

- Input: `(parent_voxel, biome_patch, heightmap, lod_embedding)`
- Output: `(air_mask_logits, block_type_logits)`
- Architecture: 3D U-Net with skip connections
- Loss: BCE for mask + CE for block types
- Timestep embedding: sinusoidal or learned

---

## 🧱 Chunk Format

Extracted `.npz` chunk files must contain:

```python
{
  "block_types": uint8, shape=(16, 16, 384)
  "air_mask": bool,     shape=(16, 16, 384)
  "biomes": uint8 or int32, shape=(16, 16) or (16, 16, 384)
  "heightmap": uint8,   shape=(16, 16)
}
```

## 🧩 Training Data Preparation
Downsampled parent-child patch pairs should have:

Parent voxel: (e.g. 8×8×8)

Target mask: (e.g. 16×16×16)

Target types: (same shape)

## 🌍 Phase 0B: World Generation Bootstrap

### 🎯 Tactical Briefing

Phase 0B establishes the foundation for generating Minecraft .mca files using a headless Java-based world generator. This phase must create a reliable, repeatable pipeline for terrain data extraction.

### 🛠️ Implementation Strategy

#### Java Tool Selection
- **Primary**: Use `minecraft-dev-tools` or similar headless worldgen jar
- **Fallback**: Custom Fabric mod with minimal dependencies 
- **Alternative**: Cuberite C++ server with custom plugins
- **Last Resort**: Parse existing world saves with `anvil-parser`

#### Required Infrastructure
```python
# Essential components to implement:
class WorldGenBootstrap:
    def __init__(self, seed: str = "VoxelTree", java_heap: str = "4G"):
        self.seed = self._hash_seed(seed)  # "VoxelTree" → 1903448982
        self.java_heap = java_heap
        self.temp_world_dir = Path("temp_worlds")
        
    def generate_region_batch(self, x_range: tuple, z_range: tuple) -> Path:
        """Generate .mca files for specified chunk ranges"""
        
    def cleanup_temp_worlds(self, keep_latest: int = 2):
        """Remove old temporary world folders"""
```

#### File Management Protocol
- **Temp worlds**: `temp_worlds/world_{timestamp}/region/`
- **Target chunks**: Generate only needed regions (e.g., r.0.0.mca, r.1.0.mca)
- **Auto-cleanup**: Remove temp worlds after successful .npz extraction
- **Disk limit**: Never exceed 5GB of temp .mca files at once

#### Java Process Management
```python
# Required subprocess patterns:
def run_worldgen_java(jar_path: Path, world_dir: Path, 
                     x_min: int, x_max: int, z_min: int, z_max: int) -> bool:
    """
    Execute: java -Xmx4G -jar worldgen.jar --seed 1903448982 
             --output ./temp_worlds/world_001 --region-x 0,4 --region-z 0,4
    """
    
def validate_mca_output(region_dir: Path) -> dict:
    """Verify .mca files contain expected chunks and aren't corrupted"""
```

#### Error Recovery Patterns
- **Java heap exhaustion**: Reduce batch size and retry
- **Corrupted .mca**: Regenerate specific region files
- **Missing chunks**: Validate chunk coordinates in .mca files
- **Disk space**: Emergency cleanup of temp directories

### 🧪 TDD Test Requirements

#### RED Phase Tests
```python
def test_worldgen_bootstrap_init():
    """Test bootstrap initialization with seed hashing"""
    
def test_generate_single_region():
    """Test generation of one .mca file with known chunks"""
    
def test_mca_file_validation():
    """Test parsing and validation of generated .mca files"""
    
def test_cleanup_temp_worlds():
    """Test automatic cleanup respects disk limits"""
```

#### GREEN Phase Implementation
- Implement minimal `WorldGenBootstrap` class
- Add basic Java subprocess execution
- Create .mca file existence validation
- Implement emergency disk cleanup

#### REFACTOR Phase Documentation
- Document Java tool selection rationale
- Add performance benchmarks for different batch sizes
- Create troubleshooting guide for common Java/worldgen errors
- Update config.yaml with worldgen parameters

### 🔧 Configuration Requirements

```yaml
# config.yaml additions:
worldgen:
  seed: "VoxelTree"  # Converts to 1903448982
  java_heap: "4G"
  batch_size: 16     # Chunks per .mca generation batch
  max_temp_disk_gb: 5
  chunk_region_bounds:
    x_min: 0
    x_max: 128        # 2048 chunks = 128 regions
    z_min: 0  
    z_max: 128
  java_tools:
    primary: "tools/minecraft-worldgen.jar"
    fallback: "tools/fabric-worldgen-mod.jar"
```

### 🚨 Critical Success Criteria

1. **Repeatability**: Same seed always produces identical .mca files
2. **Efficiency**: Generate 100+ chunks in <5 minutes on typical hardware
3. **Reliability**: Handle Java crashes and corrupted outputs gracefully
4. **Resource limits**: Never exceed 5GB temp disk usage
5. **Testability**: All worldgen operations must be unit-testable

### 🌍 World Generation Context

- **Seed**: "VoxelTree" → numeric 1903448982
- **Training region**: [0,2048] × [0,2048] chunks (≈500GB full, we subsample)
- **Output format**: Standard Minecraft .mca region files
- **Biome support**: Must preserve all biome data for conditioning
- **Height range**: Full 384-block height (Y=-64 to Y=320)

All terrain generation must be headless and repeatable

🔐 Behavior Constraints for Copilot
Never hardcode file paths — load from config.yaml

Always compress disk output

Use lazy loading and memory-mapping for large batches

Use torch.device("cuda" if available) for GPU support

Don't assume you can load the whole dataset at once

Avoid diffusion chains unless explicitly enabled

🌀 Optional Later Features
Discrete denoising diffusion (offline or distilled)

Curriculum learning across LOD levels

Caching biome rarity for adaptive sampling

ONNX runtime embedded testing

🧠 Reflect & Improve
Each REFACTOR step should:

Add markdown notes if something went wrong

Suggest updated tests or config safeguards

Improve variable/function naming

When in doubt, follow the TDD flow, use batch generators, and prefer clarity over cleverness.

VoxelTree builds terrain. Copilot, you build the brain.

---

## 🌍 Phase 0C: MCA to NPZ Extraction Pipeline

#### 🎯 Tactical Briefing

Phase 0C implements the core data extraction functionality to parse Minecraft .mca region files and convert them into compressed .npz training data. This phase creates the bridge between raw Minecraft world data and ML-ready tensors.

#### 🛠️ Implementation Strategy

##### Anvil Parser Integration
- **Primary**: Use `anvil-parser` library for robust .mca reading
- **Fallback**: Custom NBT parser with zlib decompression
- **Data format**: Extract blocks, biomes, heightmaps per 16×16×384 chunk
- **Validation**: Verify chunk coordinates and detect corruption

##### Required Infrastructure
```python
# Essential components to implement:
class ChunkExtractor:
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config = load_config(config_path)
        self.output_dir = Path(self.config["extraction"]["output_dir"])
        
    def extract_chunk_data(self, region_file: Path, chunk_x: int, chunk_z: int) -> dict:
        """Extract single chunk data from .mca file"""
        
    def extract_region_batch(self, region_file: Path) -> List[Path]:
        """Extract all chunks from region file to .npz files"""
        
    def process_block_data(self, chunk_data) -> tuple:
        """Convert NBT block data to numpy arrays (types, air_mask)"""
        
    def extract_biome_data(self, chunk_data) -> np.ndarray:
        """Extract biome IDs for chunk"""
        
    def compute_heightmap(self, block_types: np.ndarray) -> np.ndarray:
        """Compute surface heightmap from block data"""
```

##### NPZ Output Format
```python
# Each chunk_{x}_{z}.npz contains:
{
    "block_types": np.uint8,    # shape=(16, 16, 384), block type IDs
    "air_mask": np.bool_,       # shape=(16, 16, 384), True for air blocks
    "biomes": np.uint8,         # shape=(16, 16), biome IDs per column
    "heightmap": np.uint16,     # shape=(16, 16), surface height per column
    "chunk_x": np.int32,        # chunk X coordinate
    "chunk_z": np.int32,        # chunk Z coordinate
    "region_file": np.string_   # source .mca filename for debugging
}
```

##### Memory Management Protocol
- **Streaming**: Process one region file at a time
- **Batch limits**: Max 64 chunks in memory before writing to disk
- **Cleanup**: Delete temp .mca files after successful .npz extraction
- **Disk quotas**: Monitor .npz output size, delete oldest when >10GB

##### Multiprocessing Strategy
```python
# Required parallel processing patterns:
def extract_regions_parallel(region_files: List[Path], num_workers: int = 4) -> None:
    """
    Use multiprocessing.Pool to extract multiple regions simultaneously
    Each worker processes one .mca file independently
    """
    
def validate_extraction_results(output_dir: Path) -> dict:
    """Verify .npz files contain expected data and aren't corrupted"""
```

#### 🧪 TDD Test Requirements

##### RED Phase Tests
```python
def test_chunk_extractor_init():
    """Test ChunkExtractor initialization and config loading"""
    
def test_extract_single_chunk():
    """Test extraction of one chunk from .mca file"""
    
def test_block_data_processing():
    """Test conversion of NBT block data to numpy arrays"""
    
def test_biome_extraction():
    """Test biome ID extraction and validation"""
    
def test_heightmap_computation():
    """Test surface heightmap calculation from blocks"""
    
def test_npz_output_format():
    """Test .npz file structure and compression"""
    
def test_region_batch_extraction():
    """Test extraction of all chunks from one region"""
    
def test_parallel_region_processing():
    """Test multiprocessing extraction of multiple regions"""
    
def test_memory_management():
    """Test streaming processing without excessive memory usage"""
    
def test_extraction_validation():
    """Test detection of corrupted or incomplete extractions"""
```

##### GREEN Phase Implementation
- Implement minimal `ChunkExtractor` class
- Add basic .mca file reading with anvil-parser
- Create block data → numpy conversion
- Implement .npz output with compression
- Add simple multiprocessing for regions

##### REFACTOR Phase Documentation
- Document anvil-parser integration details
- Add performance benchmarks for extraction speed
- Create troubleshooting guide for common .mca parsing errors
- Update config.yaml with extraction parameters

#### 🔧 Configuration Requirements

```yaml
# config.yaml additions:
extraction:
  output_dir: "data/chunks"
  temp_dir: "temp_extraction"
  max_disk_usage_gb: 10
  batch_size: 64              # chunks per batch before disk write
  num_workers: 4              # multiprocessing workers
  compression_level: 6        # npz compression (1-9)
  validation:
    verify_checksums: true
    detect_corruption: true
  block_mapping:
    air_blocks: [0]           # block IDs considered "air"
    solid_blocks: [1, 2, 3]   # common solid block IDs
  heightmap:
    surface_blocks: [2, 3, 4] # blocks that count as "surface"
    min_height: -64           # world bottom (Y coordinate)
    max_height: 320           # world top (Y coordinate)
```

#### 🚨 Critical Success Criteria

1. **Format compliance**: All .npz files match exact schema
2. **Data integrity**: No corruption during extraction process
3. **Performance**: Extract 1000+ chunks in <2 minutes on typical hardware
4. **Memory efficiency**: Never exceed 2GB RAM during extraction
5. **Parallel scalability**: Linear speedup with multiple workers
6. **Error recovery**: Handle corrupted .mca files gracefully

#### 🌍 Extraction Context

- **Input**: .mca region files from Phase 0B worldgen
- **Output**: Compressed .npz chunk files in data/chunks/
- **Chunk range**: Full 16×16×384 blocks per chunk
- **Coordinate system**: Minecraft world coordinates (chunk_x, chunk_z)
- **Block mapping**: Convert Minecraft block IDs to training-friendly format
- **Biome support**: Preserve biome data for conditional generation

All extraction must be deterministic and reproducible for same input .mca files.

---