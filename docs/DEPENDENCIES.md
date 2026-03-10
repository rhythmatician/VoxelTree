# LODiffusion Dependency Matrix

**Target:** Minecraft 1.21.11  
**Last Updated:** 2026-03-05  
**Status:** Pinned for Phase 1 development

## Java Toolchain

- **Java:** 21 (required for modern MC + Fabric ecosystem)
- **Gradle:** (TBD - check Fabric template)
- **Kotlin:** (TBD - if used in mod)

## Mod Stack (Minecraft 1.21.11)

### Core Dependencies

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| **Minecraft** | 1.21.11 | Mojang | Target version |
| **Fabric Loader** | 0.18.4 | [FabricMC](https://fabricmc.net/) | Pinned |
| **Fabric API** | 0.141.3+1.21.11 | [FabricMC](https://modrinth.com/mod/fabric-api) | Pinned |
| **Voxy** | 0.2.11-alpha | [MCRcortex/voxy](https://github.com/MCRcortex/voxy) | Pinned |
| **Carpet Mod** | 1.4.194+v251223 | [gnembon/fabric-carpet](https://github.com/gnembon/fabric-carpet) | For world freeze + pregen |
| **Chunky** | 1.4.55 | [pop4959/Chunky](https://github.com/pop4959/Chunky) | Chunk pregeneration |

### LODiffusion Mod

- **LODiffusion:** (ours - version TBD)
- **DJL:** TBD (check latest stable)
- **ONNX Runtime:** TBD (CPU backend only)

## ML Runtime Stack

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| **ONNX** | 1.12+ | [ONNX](https://onnx.ai/) | Static shapes only |
| **DJL** | TBD | [Deep Java Library](https://djl.ai/) | Check latest stable |
| **ONNX Runtime** | TBD | [ONNX Runtime](https://onnxruntime.ai/) | CPU backend only |

## Python Training Stack

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| **Python** | 3.13+ | [python.org](https://www.python.org/) | 3.13.1+ recommended |
| **PyTorch** | >=2.0 | [PyTorch](https://pytorch.org/) | CPU/GPU support |
| **NumPy** | Latest | PyPI | Core dependency |
| **SciPy** | Latest | PyPI | Interpolation, pooling |
| **PyYAML** | Latest | PyPI | Config parsing |
| **tqdm** | Latest | PyPI | Progress bars |
| **anvil-parser2** | Latest | PyPI | Minecraft .mca parsing |
| **ONNX** | Latest | PyPI | Export validation |
| **ONNX Runtime** | Latest | PyPI | Export validation |

See `requirements.txt` for complete Python dependency list.

## World Generation Tools

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| **Carpet Mod** | 1.4.194+v251223 | [gnembon/fabric-carpet](https://github.com/gnembon/fabric-carpet) | World freeze + pregen |
| **Chunky** | 1.4.55 | [pop4959/Chunky](https://github.com/pop4959/Chunky) | Chunk pregeneration |
| **anvil-parser2** | Latest | PyPI | Python .mca reading |

## Excluded Dependencies (Demo Milestone)

**Do NOT include in Phase 1 demo:**

- Extra worldgen mods (may alter noise/biome behavior)
- Shader mods (may affect performance profiling)
- Optimization mods (may affect timing/threading)
- Any mod that alters biome/noise settings

## Version Locking Strategy

1. **Pin exact versions** in build files (Gradle `build.gradle.kts` for mod)
2. **Document in this file** with commit hashes/branch names where applicable
3. **Lock Python dependencies** via `requirements.txt` with version pins
4. **Test compatibility matrix** before proceeding to next milestone

## Next Steps

1. [x] Pin Fabric Loader version for 1.21.11 → 0.18.4
2. [x] Pin Fabric API version for 1.21.11 → 0.141.3+1.21.11
3. [x] Pin Voxy version for 1.21.11 → 0.2.11-alpha
4. [x] Pin Carpet Mod version for 1.21.11 → 1.4.194+v251223
5. [x] Pin Chunky version for 1.21.11 → 1.4.55
6. [ ] Research and pin DJL version (latest stable)
7. [ ] Research and pin ONNX Runtime version (CPU backend)
8. [ ] Create `mods.lock.json` or equivalent for Java dependencies
9. [ ] Update `requirements.txt` with pinned Python versions

## Compatibility Notes

- **Java 21** is required for modern Minecraft + Fabric ecosystem
- **ONNX 1.12+** required for static shape support
- **DJL** must support ONNX Runtime CPU backend
- All versions must be compatible with **Minecraft 1.21.11**

## References

- [Fabric Wiki - Getting Started](https://fabricmc.net/wiki/start)
- [Fabric API - Modrinth](https://modrinth.com/mod/fabric-api)
- [Voxy GitHub](https://github.com/MCRcortex/voxy)
- [Carpet Mod GitHub](https://github.com/gnembon/fabric-carpet)
- [DJL Documentation](https://djl.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
