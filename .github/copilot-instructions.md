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