# NoiseTap Interface Specification

**Status:** Design Phase  
**Target:** Minecraft 1.21.5 + Fabric  
**Purpose:** One-call anchor capture per chunk for LODiffusion

## Overview

The `NoiseTap` interface provides a single-call mechanism to capture all vanilla noise-derived anchor signals for a given chunk. This eliminates the need for multiple sampling calls and ensures consistency across all anchor channels.

## Interface Definition

```java
package com.lodiffusion.anchor;

import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.chunk.ChunkGenerator;

/**
 * Captures vanilla noise-derived anchor signals for a chunk in a single call.
 * 
 * Returns native-grid tensors ready for normalization and caching.
 */
public interface NoiseTap {
    /**
     * Captures all anchor signals for a given chunk position.
     * 
     * @param chunkPos The chunk position to sample
     * @param chunkGenerator The chunk generator (for noise access)
     * @param worldSeed The world seed for determinism
     * @return FeatureBundle containing all anchor channels
     */
    FeatureBundle capture(ChunkPos chunkPos, ChunkGenerator chunkGenerator, long worldSeed);
}
```

## Implementation: VanillaNoiseTap

```java
package com.lodiffusion.anchor.impl;

import com.lodiffusion.anchor.FeatureBundle;
import com.lodiffusion.anchor.NoiseTap;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;

/**
 * Vanilla implementation of NoiseTap using Minecraft's native noise routers.
 */
public class VanillaNoiseTap implements NoiseTap {
    
    @Override
    public FeatureBundle capture(ChunkPos chunkPos, ChunkGenerator chunkGenerator, long worldSeed) {
        // 1. Get NoiseConfig from chunk generator
        NoiseConfig noiseConfig = chunkGenerator.getNoiseConfig();
        
        // 2. Sample height planes (16×16)
        float[][] heightSurface = sampleHeightmap(chunkPos, noiseConfig, Heightmap.Type.WORLD_SURFACE_WG);
        float[][] heightOceanFloor = sampleHeightmap(chunkPos, noiseConfig, Heightmap.Type.OCEAN_FLOOR_WG);
        float[][] slopeX = computeSlope(heightSurface, 'x');
        float[][] slopeZ = computeSlope(heightSurface, 'z');
        float[][] curvature = computeCurvature(heightSurface);
        
        // 3. Sample biome quart lattice (4×4×4)
        BiomeQuart[][] biomeQuart = sampleBiomeQuart(chunkPos, noiseConfig);
        
        // 4. Sample NoiseRouter slices (16×16 @ 1 Y)
        float[][] routerTemperature = sampleRouter(chunkPos, noiseConfig, RouterType.TEMPERATURE);
        float[][] routerVegetation = sampleRouter(chunkPos, noiseConfig, RouterType.VEGETATION);
        float[][] routerContinents = sampleRouter(chunkPos, noiseConfig, RouterType.CONTINENTS);
        float[][] routerErosion = sampleRouter(chunkPos, noiseConfig, RouterType.EROSION);
        float[][] routerDepth = sampleRouter(chunkPos, noiseConfig, RouterType.DEPTH);
        float[][] routerRidges = sampleRouter(chunkPos, noiseConfig, RouterType.RIDGES);
        
        // 5. Optional: barrier (16×16)
        float[][] barrier = sampleBarrier(chunkPos, noiseConfig);
        
        // 6. Optional: aquifer trio (16×16)
        float[][] aquiferSurface = sampleAquifer(chunkPos, noiseConfig, AquiferType.SURFACE);
        float[][] aquiferFlooded = sampleAquifer(chunkPos, noiseConfig, AquiferType.FLOODED);
        float[][] aquiferLava = sampleAquifer(chunkPos, noiseConfig, AquiferType.LAVA);
        
        // 7. Optional: coarse cave prior (4×4×4) - expensive, use sparingly
        float[][][] cavePrior = sampleCavePrior(chunkPos, noiseConfig);
        
        // 8. Build FeatureBundle
        return FeatureBundle.builder()
            .chunkPos(chunkPos)
            .heightPlanes(new HeightPlanes(heightSurface, heightOceanFloor, slopeX, slopeZ, curvature))
            .biomeQuart(biomeQuart)
            .router6(new Router6(routerTemperature, routerVegetation, routerContinents, 
                                routerErosion, routerDepth, routerRidges))
            .barrier(barrier)
            .aquifer3(new Aquifer3(aquiferSurface, aquiferFlooded, aquiferLava))
            .cavePrior(cavePrior)
            .build();
    }
    
    // Helper methods for sampling...
    private float[][] sampleHeightmap(ChunkPos pos, NoiseConfig config, Heightmap.Type type) {
        // Sample at 16×16 grid
        // Return normalized heights
    }
    
    private BiomeQuart[][] sampleBiomeQuart(ChunkPos pos, NoiseConfig config) {
        // Sample at 4×4×4 quart lattice
        // Return temperature, precipitation[3], isCold, downfall
    }
    
    private float[][] sampleRouter(ChunkPos pos, NoiseConfig config, RouterType type) {
        // Sample NoiseRouter at Y=1 (or appropriate level)
        // Return 16×16 grid
    }
    
    // ... additional sampling methods
}
```

## FeatureBundle Data Structure

```java
package com.lodiffusion.anchor;

import net.minecraft.util.math.ChunkPos;

/**
 * Immutable bundle of all anchor channels for a chunk.
 * 
 * All data is in native Minecraft grid format (not yet normalized).
 */
public class FeatureBundle {
    private final ChunkPos chunkPos;
    
    // Required channels
    private final HeightPlanes heightPlanes;      // [5, 16, 16]
    private final BiomeQuart biomeQuart;          // [6, 4, 4, 4]
    private final Router6 router6;                // [6, 16, 16]
    
    // Optional channels
    private final float[][] barrier;              // [1, 16, 16] or null
    private final Aquifer3 aquifer3;              // [3, 16, 16] or null
    private final float[][][] cavePrior;         // [1, 4, 4, 4] or null
    
    // Builder pattern
    public static Builder builder() { ... }
    
    // Getters
    public ChunkPos getChunkPos() { return chunkPos; }
    public HeightPlanes getHeightPlanes() { return heightPlanes; }
    // ... etc
}
```

## Tensor Shapes (Native Grid)

| Channel | Native Shape | Description |
|---------|--------------|-------------|
| `heightPlanes` | `[5, 16, 16]` | surface, ocean_floor, slope_x, slope_z, curvature |
| `biomeQuart` | `[6, 4, 4, 4]` | temp, precip[3], isCold, downfall |
| `router6` | `[6, 16, 16]` | temperature, vegetation, continents, erosion, depth, ridges |
| `barrier` (opt) | `[1, 16, 16]` | Coastal barrier mask |
| `aquifer3` (opt) | `[3, 16, 16]` | Surface, flooded, lava aquifer masks |
| `cavePrior` (opt) | `[1, 4, 4, 4]` | Coarse cave likelihood |

## Normalization

Normalization is **not** performed by NoiseTap. It returns raw Minecraft values. Normalization happens in the `FeatureBundle` cache layer or during tensor packing for ONNX.

**Normalization rules:**
- Heights: min-max by world limits (-64 to 320)
- Router/Aquifer/Cave: z-score (requires dataset statistics)
- Flags: [0,1] (already normalized)
- Coords: affine/tanh scaling

## Performance Considerations

- **Single call:** All sampling happens in one pass to ensure consistency
- **Lazy optional channels:** Expensive channels (cave prior) can be null
- **Caching:** FeatureBundle should be cached (see FeatureBundle cache spec)
- **Thread safety:** Implementation should be thread-safe for parallel chunk processing

## Testing

Unit tests should verify:
1. Deterministic output (same chunkPos + seed = same FeatureBundle)
2. Shape correctness (all tensors match expected dimensions)
3. Value ranges (heights in [-64, 320], flags in [0,1], etc.)
4. Consistency with vanilla chunk generation (parity tests)

## Integration Points

- **ChunkGenerator:** Access to NoiseConfig and noise routers
- **FeatureBundle Cache:** Output is cached for reuse
- **Tensor Packer:** FeatureBundle is converted to ONNX input tensors
- **Dataset Extraction:** Can be used for training data generation

## Future Extensions

- Support for custom noise configurations
- Per-dimension implementations (Overworld, Nether, End)
- Sampling at different Y levels for 3D features
- Integration with structure placement hints
