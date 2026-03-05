# FeatureBundle Cache Specification

**Status:** Design Phase  
**Target:** Minecraft 1.21.5 + Fabric  
**Purpose:** LRU cache with optional disk sidecar for anchor channels

## Overview

The `FeatureBundleCache` provides efficient storage and retrieval of `FeatureBundle` objects (anchor channels) keyed by `ChunkPos`. It supports both in-memory LRU caching and optional persistent disk storage for cross-session reuse.

## Interface Definition

```java
package com.lodiffusion.anchor;

import net.minecraft.util.math.ChunkPos;
import java.util.Optional;

/**
 * LRU cache for FeatureBundle objects with optional disk persistence.
 * 
 * Keyed by ChunkPos. Supports eviction, TTL, and versioning.
 */
public interface FeatureBundleCache {
    /**
     * Get a FeatureBundle from cache (memory or disk).
     * 
     * @param chunkPos The chunk position
     * @return Optional FeatureBundle if found, empty otherwise
     */
    Optional<FeatureBundle> get(ChunkPos chunkPos);
    
    /**
     * Store a FeatureBundle in cache (memory and optionally disk).
     * 
     * @param chunkPos The chunk position
     * @param bundle The FeatureBundle to cache
     */
    void put(ChunkPos chunkPos, FeatureBundle bundle);
    
    /**
     * Check if a FeatureBundle exists in cache.
     * 
     * @param chunkPos The chunk position
     * @return true if cached (memory or disk)
     */
    boolean contains(ChunkPos chunkPos);
    
    /**
     * Remove a FeatureBundle from cache.
     * 
     * @param chunkPos The chunk position
     */
    void evict(ChunkPos chunkPos);
    
    /**
     * Clear all cached entries.
     */
    void clear();
    
    /**
     * Get cache statistics.
     * 
     * @return CacheStats with hit/miss counts, size, etc.
     */
    CacheStats getStats();
}
```

## Implementation: LRUFeatureBundleCache

```java
package com.lodiffusion.anchor.impl;

import com.lodiffusion.anchor.FeatureBundle;
import com.lodiffusion.anchor.FeatureBundleCache;
import net.minecraft.util.math.ChunkPos;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.io.*;

/**
 * LRU cache implementation with optional disk sidecar.
 */
public class LRUFeatureBundleCache implements FeatureBundleCache {
    
    // In-memory LRU cache
    private final LinkedHashMap<ChunkPos, CacheEntry> memoryCache;
    private final int maxMemorySize;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    
    // Disk persistence (optional)
    private final boolean diskEnabled;
    private final Path cacheDir;
    private final CompressionType compression;
    
    // Statistics
    private final CacheStats stats = new CacheStats();
    
    public LRUFeatureBundleCache(
        int maxMemorySize,
        boolean diskEnabled,
        Path cacheDir,
        CompressionType compression
    ) {
        this.maxMemorySize = maxMemorySize;
        this.diskEnabled = diskEnabled;
        this.cacheDir = cacheDir;
        this.compression = compression;
        
        // LRU LinkedHashMap with access order
        this.memoryCache = new LinkedHashMap<ChunkPos, CacheEntry>(
            16, 0.75f, true // access order
        ) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<ChunkPos, CacheEntry> eldest) {
                if (size() > maxMemorySize) {
                    // Evict to disk if enabled
                    if (diskEnabled) {
                        evictToDisk(eldest.getKey(), eldest.getValue());
                    }
                    return true;
                }
                return false;
            }
        };
        
        if (diskEnabled) {
            cacheDir.toFile().mkdirs();
        }
    }
    
    @Override
    public Optional<FeatureBundle> get(ChunkPos chunkPos) {
        lock.readLock().lock();
        try {
            // Check memory first
            CacheEntry entry = memoryCache.get(chunkPos);
            if (entry != null) {
                stats.recordHit();
                return Optional.of(entry.bundle);
            }
            
            // Check disk if enabled
            if (diskEnabled) {
                Optional<FeatureBundle> diskBundle = loadFromDisk(chunkPos);
                if (diskBundle.isPresent()) {
                    stats.recordHit();
                    // Promote to memory
                    put(chunkPos, diskBundle.get());
                    return diskBundle;
                }
            }
            
            stats.recordMiss();
            return Optional.empty();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void put(ChunkPos chunkPos, FeatureBundle bundle) {
        lock.writeLock().lock();
        try {
            CacheEntry entry = new CacheEntry(bundle, System.currentTimeMillis());
            memoryCache.put(chunkPos, entry);
            
            // Write to disk if enabled
            if (diskEnabled) {
                saveToDisk(chunkPos, bundle);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    @Override
    public boolean contains(ChunkPos chunkPos) {
        lock.readLock().lock();
        try {
            if (memoryCache.containsKey(chunkPos)) {
                return true;
            }
            if (diskEnabled) {
                return diskFileExists(chunkPos);
            }
            return false;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void evict(ChunkPos chunkPos) {
        lock.writeLock().lock();
        try {
            memoryCache.remove(chunkPos);
            if (diskEnabled) {
                deleteFromDisk(chunkPos);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            memoryCache.clear();
            if (diskEnabled) {
                clearDiskCache();
            }
            stats.reset();
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    // Disk I/O methods
    private void saveToDisk(ChunkPos chunkPos, FeatureBundle bundle) {
        Path file = getDiskPath(chunkPos);
        try {
            // Serialize FeatureBundle
            byte[] data = serialize(bundle);
            
            // Compress if enabled
            if (compression == CompressionType.ZSTD) {
                data = compressZstd(data);
            } else if (compression == CompressionType.LZ4) {
                data = compressLz4(data);
            }
            
            // Write atomically
            Path tempFile = file.resolveSibling(file.getFileName() + ".tmp");
            Files.write(tempFile, data);
            Files.move(tempFile, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            // Log error but don't fail
            logger.warn("Failed to save FeatureBundle to disk: {}", chunkPos, e);
        }
    }
    
    private Optional<FeatureBundle> loadFromDisk(ChunkPos chunkPos) {
        Path file = getDiskPath(chunkPos);
        if (!Files.exists(file)) {
            return Optional.empty();
        }
        
        try {
            byte[] data = Files.readAllBytes(file);
            
            // Decompress if needed
            if (compression == CompressionType.ZSTD) {
                data = decompressZstd(data);
            } else if (compression == CompressionType.LZ4) {
                data = decompressLz4(data);
            }
            
            FeatureBundle bundle = deserialize(data);
            return Optional.of(bundle);
        } catch (IOException e) {
            logger.warn("Failed to load FeatureBundle from disk: {}", chunkPos, e);
            return Optional.empty();
        }
    }
    
    private Path getDiskPath(ChunkPos chunkPos) {
        // Organize by region: r.X.Z/featurebundle_c.X.Z.dat
        int regionX = chunkPos.x >> 5;
        int regionZ = chunkPos.z >> 5;
        String regionDir = String.format("r.%d.%d", regionX, regionZ);
        String filename = String.format("featurebundle_c.%d.%d.dat", chunkPos.x, chunkPos.z);
        return cacheDir.resolve(regionDir).resolve(filename);
    }
    
    // Serialization
    private byte[] serialize(FeatureBundle bundle) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (DataOutputStream dos = new DataOutputStream(baos)) {
            // Write version header
            dos.writeInt(CACHE_VERSION);
            
            // Write ChunkPos
            dos.writeInt(bundle.getChunkPos().x);
            dos.writeInt(bundle.getChunkPos().z);
            
            // Write height planes [5, 16, 16]
            HeightPlanes heights = bundle.getHeightPlanes();
            for (int i = 0; i < 5; i++) {
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        dos.writeFloat(heights.get(i, x, z));
                    }
                }
            }
            
            // Write biome quart [6, 4, 4, 4]
            // ... etc
            
            // Write optional channels (with flags)
            dos.writeBoolean(bundle.hasBarrier());
            if (bundle.hasBarrier()) {
                // Write barrier data
            }
            // ... etc
        }
        return baos.toByteArray();
    }
    
    private FeatureBundle deserialize(byte[] data) throws IOException {
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(data))) {
            // Read version
            int version = dis.readInt();
            if (version != CACHE_VERSION) {
                throw new IOException("Cache version mismatch: " + version);
            }
            
            // Read ChunkPos
            int x = dis.readInt();
            int z = dis.readInt();
            ChunkPos chunkPos = new ChunkPos(x, z);
            
            // Read height planes
            // ... etc
            
            // Build FeatureBundle
            return FeatureBundle.builder()
                .chunkPos(chunkPos)
                // ... set all fields
                .build();
        }
    }
    
    // Cache entry wrapper
    private static class CacheEntry {
        final FeatureBundle bundle;
        final long timestamp;
        
        CacheEntry(FeatureBundle bundle, long timestamp) {
            this.bundle = bundle;
            this.timestamp = timestamp;
        }
    }
    
    private static final int CACHE_VERSION = 1;
}
```

## Cache Configuration

```java
public class CacheConfig {
    public int maxMemorySize = 1000;  // Max entries in memory
    public boolean diskEnabled = true;
    public Path cacheDir = Paths.get(".lodiffusion/cache/anchors");
    public CompressionType compression = CompressionType.ZSTD;
    public long ttlSeconds = -1;  // -1 = no expiration
    public boolean enableStats = true;
}
```

## Cache Statistics

```java
public class CacheStats {
    private long hits = 0;
    private long misses = 0;
    private long evictions = 0;
    private long diskReads = 0;
    private long diskWrites = 0;
    
    public double hitRate() {
        long total = hits + misses;
        return total > 0 ? (double) hits / total : 0.0;
    }
    
    // Getters and record methods...
}
```

## Versioning & Invalidation

Cache entries include a version number. Invalidation triggers:
- World seed change
- Mod version change
- Model config change
- Manual cache clear

Invalidation strategy:
1. Check cache version on load
2. If mismatch, clear cache and rebuild
3. Store version in cache metadata file

## Performance Considerations

- **Memory:** LRU eviction keeps hot entries in memory
- **Disk:** Async writes (don't block on disk I/O)
- **Compression:** ZSTD or LZ4 for space efficiency
- **Thread safety:** Read-write locks for concurrent access
- **Atomic writes:** Use temp files + atomic move for crash safety

## Integration Points

- **NoiseTap:** Populates cache on first access
- **Tensor Packer:** Reads from cache for ONNX inputs
- **Dataset Extraction:** Can pre-populate cache for training
- **LOD Scheduler:** Cache lookup before expensive sampling

## Future Extensions

- TTL-based expiration
- Cache warming strategies
- Distributed cache (multi-server)
- Metrics export (Prometheus, etc.)
