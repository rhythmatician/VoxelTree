# Java Worldgen Tools Directory

This directory contains Java-based tools for generating Minecraft world files (.mca).

## Required Tools

### Primary: minecraft-worldgen.jar
- **Purpose**: Headless Minecraft world generation
- **Usage**: `java -Xmx4G -jar minecraft-worldgen.jar --seed 1903448982 --output ./world --region-x 0,4 --region-z 0,4`
- **Download**: [Link to be added when tool is identified]

### Fallback: fabric-worldgen-mod.jar
- **Purpose**: Fabric-based world generation mod
- **Usage**: Similar command-line interface
- **Download**: [Link to be added when tool is identified]

## Installation Instructions

1. Download the appropriate worldgen tool
2. Place the JAR file in this directory
3. Ensure Java 17+ is installed and in PATH
4. Test with: `java -jar minecraft-worldgen.jar --help`

## Troubleshooting

### Java Heap Exhaustion
- Reduce batch size in config.yaml
- Increase java_heap setting (default: "4G")
- Monitor system memory usage

### Missing Java Tools
- Bootstrap will automatically fall back to secondary tools
- Check file permissions on JAR files
- Verify Java installation: `java -version`

### Corrupted .mca Files
- Re-run generation for affected regions
- Check disk space and permissions
- Validate with `WorldGenBootstrap.validate_mca_output()`
