# Java Worldgen Tools Directory

This directory contains tools for generating and extracting Minecraft world data for VoxelTree training.

## ✅ Current Toolchain (as of Phase 0B)

### 🧱 Chunk Generation

**Primary:** `fabric-server.jar` + `Chunky` mod

* **Purpose**: Vanilla-accurate chunk pregeneration (produces `.mca` files)
* **Location**: `tools/fabric-server/`, `tools/chunky/`
* **Usage**: Spawn a Fabric server with `Chunky-Fabric-<version>.jar` in `mods/`, then use `chunky center`, `chunky radius`, and `chunky start` commands to generate terrain

### ❌ Deprecated: `minecraft-worldgen.jar`

* **Status**: Removed from pipeline due to version limitations and lower maintainability
* **Reason**: Not compatible with 1.21.x; superseded by Fabric + Chunky setup

## 🔍 `.mca` Parsing and Extraction

**Current Parser**: [`anvil-parser2`](https://github.com/0xTiger/anvil-parser2) (Python)

* **Purpose**: Parses `.mca` region files, decodes block states and subchunk data
* **Usage**: Called from Python inside `VanillaChunkGenerator.extract_chunks_from_mca()`
* **Installation**:

  ```bash
  pip install anvil-parser2
  ```
* **No Java or `.jar` required** for extraction anymore.

## Installation Instructions

1. Download the Fabric server JAR and `Chunky-Fabric` mod JAR
2. Place them in `tools/fabric-server/` and `tools/chunky/`
3. Use `VanillaChunkGenerator` to script pregen and parsing
4. Make sure Java 17+ is installed and in PATH

## Troubleshooting

### Java Heap Exhaustion

* Reduce batch size in `config.yaml`
* Increase `java_heap` setting (default: "4G")
* Monitor system memory usage

### Missing Fabric Mods or CLI

* Ensure `Chunky-Fabric-*.jar` and `fabric-api-*.jar` are in `mods/`
* Check file permissions
* Validate Java install with: `java -version`

### `.mca` Parsing Fails

* Ensure file paths and chunk ranges are valid
* Try opening the `.mca` file manually with `anvil-parser2`
* Validate output structure (16x16x16 subchunks)

## References

* Chunk pregen: [https://modrinth.com/mod/chunky](https://modrinth.com/mod/chunky)
* Anvil parser: [https://github.com/0xTiger/anvil-parser2](https://github.com/0xTiger/anvil-parser2)
