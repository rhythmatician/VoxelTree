package com.voxeltree.harvester.noise;

import net.minecraft.core.Holder;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.Mth;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.NoiseSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.function.Predicate;


/**
 * Provides server-side access to Minecraft's noise generators for sampling
 * heightmap, biome, and density-router values at <em>any</em> (x, z) coordinate
 * — <b>without needing a loaded chunk</b>.
 *
 * <p>This replaces the synthetic sine-wave heightmap and constant-biome fallback
 * that was previously used for distant (unloaded) sections.  It works by
 * tapping into the server's {@link net.minecraft.world.level.levelgen.ChunkGenerator}
 * and {@link RandomState} directly.
 *
 * <h3>Limitations</h3>
 * <ul>
 *   <li>Only works when a server is available.
 *       Returns {@code null} from {@link #tryCreate(Level)} on failure.</li>
 *   <li>All sampling methods are pure computation (no world state mutation),
 *       so they are safe to call from worker threads.</li>
 * </ul>
 */
public final class WorldNoiseAccess {

    private static final Logger LOGGER = LoggerFactory.getLogger(WorldNoiseAccess.class);

    private final ServerLevel serverLevel;
    private final net.minecraft.world.level.chunk.ChunkGenerator generator;
    private final RandomState randomState;
    private final BiomeSource biomeSource;

    private WorldNoiseAccess(ServerLevel serverLevel,
                             net.minecraft.world.level.chunk.ChunkGenerator generator,
                             RandomState randomState) {
        this.serverLevel = serverLevel;
        this.generator = generator;
        this.randomState = randomState;
        this.biomeSource = generator.getBiomeSource();
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Try to create a {@code WorldNoiseAccess} from a server instance.
     *
     * @param server      the Minecraft server (integrated or dedicated)
     * @param clientWorld the client-side world (used to determine dimension)
     * @return a new instance, or {@code null} if {@code RandomState} is
     *         not available (e.g., non-noise chunk generator)
     */
    public static WorldNoiseAccess tryCreate(MinecraftServer server, Level clientWorld) {
        try {
            if (server == null) {
                LOGGER.info(
                        "[WorldNoiseAccess] No server provided — cannot bind noise pipeline");
                return null;
            }

            // Get the server-side world for the same dimension as the client
            ResourceKey<Level> dimKey = clientWorld.dimension();
            ServerLevel serverLevel = server.getLevel(dimKey);
            if (serverLevel == null) {
                LOGGER.warn(
                        "[WorldNoiseAccess] Could not get ServerLevel for dimension {}",
                        dimKey.identifier());
                return null;
            }

            return tryCreate(serverLevel);

        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] Failed to initialize: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Try to create a {@code WorldNoiseAccess} directly from a {@link ServerLevel}.
     *
     * <p>This overload is useful from server-side code (e.g., commands) that
     * already has a {@code ServerLevel} reference.
     *
     * @param serverLevel the server-side world
     * @return a new instance, or {@code null} if {@code RandomState} is unavailable
     */
    public static WorldNoiseAccess tryCreate(ServerLevel serverLevel) {
        try {
            net.minecraft.world.level.chunk.ChunkGenerator gen =
                    serverLevel.getChunkSource().getGenerator();

            RandomState rs = tryGetRandomState(serverLevel);
            if (rs == null) {
                LOGGER.warn(
                        "[WorldNoiseAccess] RandomState unavailable — cannot use noise access");
                return null;
            }

            LOGGER.info(
                    "[WorldNoiseAccess] Successfully bound to server noise pipeline");
            return new WorldNoiseAccess(serverLevel, gen, rs);

        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] Failed to initialize: {}", e.getMessage());
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Heightmap sampling
    // ------------------------------------------------------------------

    /**
     * Sample the real surface heightmap for a 16×16 section column.
     *
     * <p>Delegates to {@link #sampleBothHeightmaps} so that the fast
     * {@code NoiseChunk(4)} path is used for overworld-style generators
     * (~64× faster than 256 independent {@code getBaseHeight()} calls).
     *
     * @param sectionX section X coordinate (chunk X)
     * @param sectionZ section Z coordinate (chunk Z)
     * @return float[16][16] of surface Y values in block coordinates
     */
    public float[][] sampleHeightmap(int sectionX, int sectionZ) {
        return sampleBothHeightmaps(sectionX, sectionZ)[0];  // [0] = WORLD_SURFACE_WG
    }

    /**
     * Sample BOTH heightmaps (WORLD_SURFACE_WG and OCEAN_FLOOR_WG) for a chunk
     * in a single pass using a full-chunk {@link NoiseChunk}.
     *
     * <p>Constructs one {@code NoiseChunk(4, ...)} covering the entire
     * 16×16 chunk (4 horizontal noise cells), then walks all block columns
     * top-down using the same interpolation loop as
     * {@code NoiseBasedChunkGenerator.populateNoise()}, recording the first solid
     * block per column for each heightmap type.
     *
     * <p><b>Why this is fast (~64× over {@code getBaseHeight()} ×256):</b>
     * The dominant cost is {@code initializeForFirstCellX()} / {@code advanceCellX()},
     * which evaluate the full noise router tree to fill interpolator buffers.
     * Here we call these <em>5 times total</em> (once + 4 horizontal cells) for
     * the whole chunk. The equivalent with independent per-column samplers
     * ({@code horizontalCellCount=1}) is 512 calls (256 columns × 2 types).
     * Within each 4×4 cell, all 64 block states are cheap trilinear interpolation.
     *
     * <p><b>Thread safety:</b> Each invocation creates its own
     * {@code NoiseChunk} with independent interpolator state. No shared
     * mutable state — safe to call from N threads simultaneously.
     *
     * <p><b>No side effects:</b> No chunks are created or cached. Pure computation
     * identical to what happens inside {@code populateNoise()}, minus block writes.
     *
     * @param sectionX section X coordinate (chunk X)
     * @param sectionZ section Z coordinate (chunk Z)
     * @return {@code float[2][16][16]}: index 0 = WORLD_SURFACE_WG,
     *         index 1 = OCEAN_FLOOR_WG. Values are {@code topSolidBlockY + 1},
     *         consistent with {@link net.minecraft.world.level.levelgen.ChunkGenerator#getBaseHeight}.
     */
    public float[][][] sampleBothHeightmaps(int sectionX, int sectionZ) {
        if (!(generator instanceof NoiseBasedChunkGenerator ncg)) {
            // Non-noise generator (e.g. flat world) — fall back to per-column sampling
            float[][] surface = sampleHeightmap(sectionX, sectionZ, Heightmap.Types.WORLD_SURFACE_WG);
            float[][] ocean   = sampleHeightmap(sectionX, sectionZ, Heightmap.Types.OCEAN_FLOOR_WG);
            return new float[][][] { surface, ocean };
        }

        NoiseGeneratorSettings settings = ncg.generatorSettings().value();
        NoiseSettings noiseSettings = settings.noiseSettings().clampToHeightAccessor(serverLevel);

        // Standard overworld values: hCells=4, hCellB=4, vCellB=8, minCellY=-8, cellHeight=48
        int hCells     = 16 / noiseSettings.getCellWidth();
        int hCellB     = noiseSettings.getCellWidth();
        int vCellB     = noiseSettings.getCellHeight();
        int minCellY   = Mth.floorDiv(noiseSettings.minY(), vCellB);
        int cellHeight = Mth.floorDiv(noiseSettings.height(), vCellB);
        int startX     = sectionX * 16;
        int startZ     = sectionZ * 16;

        // Replicate NoiseBasedChunkGenerator.createFluidLevelSampler().
        // The private Supplier<FluidPicker> on NoiseBasedChunkGenerator is
        // inaccessible, so we reconstruct it from the public settings.
        int seaLevel = settings.seaLevel();
        Aquifer.FluidStatus lavaLevel = new Aquifer.FluidStatus(-54, Blocks.LAVA.defaultBlockState());
        Aquifer.FluidStatus seaFluid  = new Aquifer.FluidStatus(seaLevel, settings.defaultFluid());
        Aquifer.FluidPicker fluidPicker =
                (x, y, z) -> y < Math.min(-54, seaLevel) ? lavaLevel : seaFluid;

        // One sampler for the entire 16×16 chunk — 4 horizontal cells × 4.
        // Beardifier marker is a no-op since we have no structure bounding boxes.
        NoiseChunk sampler = new NoiseChunk(
                hCells, randomState, startX, startZ, noiseSettings,
                DensityFunctions.BeardifierMarker.INSTANCE,
                settings, fluidPicker, Blender.empty());

        Predicate<BlockState> surfacePred = Heightmap.Types.WORLD_SURFACE_WG.isOpaque();
        Predicate<BlockState> oceanPred   = Heightmap.Types.OCEAN_FLOOR_WG.isOpaque();

        int bottomY = noiseSettings.minY();
        float[][] surface    = new float[16][16];
        float[][] oceanFloor = new float[16][16];
        for (float[] row : surface)    Arrays.fill(row, bottomY);
        for (float[] row : oceanFloor) Arrays.fill(row, bottomY);

        // Per-column flags: once both heightmaps are found for a column (iterating
        // top-down), we skip getInterpolatedState() for it. updateFor*() still runs
        // in order so the sampler's state machine advances correctly.
        boolean[] surfaceDone = new boolean[256];
        boolean[] oceanDone   = new boolean[256];

        // Mirror of NoiseBasedChunkGenerator.populateNoise() — same loop structure,
        // same call order — but collecting heightmaps instead of writing blocks.
        sampler.initializeForFirstCellX();

        for (int o = 0; o < hCells; o++) {            // horizontal cell X (0-3)
            sampler.advanceCellX(o);

            for (int p = 0; p < hCells; p++) {        // horizontal cell Z (0-3)
                for (int r = cellHeight - 1; r >= 0; r--) {  // vertical cell, top → bottom
                    sampler.selectCellYZ(r, p);

                    for (int s = vCellB - 1; s >= 0; s--) {  // block within cell, top → bottom
                        int blockY = (minCellY + r) * vCellB + s;
                        sampler.updateForY(blockY, (double) s / vCellB);

                        for (int w = 0; w < hCellB; w++) {   // block X within cell
                            int blockX = startX + o * hCellB + w;
                            int lx     = o * hCellB + w;
                            sampler.updateForX(blockX, (double) w / hCellB);

                            for (int z = 0; z < hCellB; z++) { // block Z within cell
                                int blockZ = startZ + p * hCellB + z;
                                int lz     = p * hCellB + z;
                                sampler.updateForZ(blockZ, (double) z / hCellB);

                                int idx = lx * 16 + lz;
                                if (surfaceDone[idx] && oceanDone[idx]) continue;

                                BlockState state = sampler.getInterpolatedState();
                                // null from NoiseChunk means AIR; fall back
                                // to default block (stone) for density > 0 regions
                                BlockState actual = (state == null)
                                        ? settings.defaultBlock() : state;

                                if (!surfaceDone[idx] && surfacePred.test(actual)) {
                                    surface[lx][lz] = blockY + 1;
                                    surfaceDone[idx] = true;
                                }
                                if (!oceanDone[idx] && oceanPred.test(actual)) {
                                    oceanFloor[lx][lz] = blockY + 1;
                                    oceanDone[idx] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        sampler.stopInterpolation();
        return new float[][][] { surface, oceanFloor };
    }

    /**
     * Sample a heightmap of the given type for a 16×16 section column.
     *
     * <p>Uses {@link net.minecraft.world.level.levelgen.ChunkGenerator#getBaseHeight}
     * — pure computation, no loaded chunk needed.
     *
     * @param sectionX section X coordinate (chunk X)
     * @param sectionZ section Z coordinate (chunk Z)
     * @param type     the heightmap type (e.g. WORLD_SURFACE_WG, OCEAN_FLOOR_WG)
     * @return float[16][16] of Y values in block coordinates
     */
    public float[][] sampleHeightmap(int sectionX, int sectionZ, Heightmap.Types type) {
        float[][] hm = new float[16][16];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                hm[lx][lz] = generator.getBaseHeight(
                        baseX + lx, baseZ + lz,
                        type, serverLevel, randomState);
            }
        }
        return hm;
    }

    // ------------------------------------------------------------------
    // Biome sampling
    // ------------------------------------------------------------------

    /**
     * Sample biome names for a 16×16 section column using the
     * server-side {@link BiomeSource}.
     *
     * <p>Biomes are sampled at quarter-resolution (4-block steps) as per
     * Minecraft's biome storage convention, then each block column gets
     * the biome of its containing quarter.
     *
     * <p>Returns the biome's registry key name (e.g. {@code "minecraft:plains"}).
     * The Python pipeline maps these to canonical integer IDs via a shared
     * alphabetical biome mapping.
     *
     * @param sectionX section X coordinate
     * @param sectionZ section Z coordinate
     * @param heightmap surface heightmap for Y coordinate
     * @return String[16][16] of biome registry key names
     */
    public String[][] sampleBiomeNames(int sectionX, int sectionZ, float[][] heightmap) {
        String[][] biomes = new String[16][16];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;

        // Biomes are stored at quarter resolution — sample at quart coords
        // and fill the 4×4 block region with the same value.
        for (int qx = 0; qx < 4; qx++) {
            for (int qz = 0; qz < 4; qz++) {
                int bx = baseX + qx * 4 + 2;  // center of quartet
                int bz = baseZ + qz * 4 + 2;
                int surfaceY = (int) heightmap[qx * 4][qz * 4];

                // BiomeSource.getNoiseBiome works at quart coordinates
                Holder<Biome> biomeEntry = biomeSource.getNoiseBiome(
                        bx >> 2, surfaceY >> 2, bz >> 2,
                        randomState.sampler());

                // Extract registry key name (e.g. "minecraft:plains")
                String biomeName = biomeEntry.unwrapKey()
                        .map(key -> key.identifier().toString())
                        .orElse("minecraft:unknown");

                // Fill the 4×4 block region
                for (int dx = 0; dx < 4; dx++) {
                    for (int dz = 0; dz < 4; dz++) {
                        biomes[qx * 4 + dx][qz * 4 + dz] = biomeName;
                    }
                }
            }
        }
        return biomes;
    }

    /**
     * Sample biome integer indices for a 16×16 section column.
     *
     * @deprecated Use {@link #sampleBiomeNames} for stable canonical encoding.
     *     This method uses unstable {@code hashCode() % 256} encoding.
     */
    @Deprecated
    public int[][] sampleBiomes(int sectionX, int sectionZ, float[][] heightmap) {
        String[][] names = sampleBiomeNames(sectionX, sectionZ, heightmap);
        int[][] biomes = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                biomes[x][z] = Math.abs(names[x][z].hashCode()) % 256;
            }
        }
        return biomes;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /**
     * Get the {@link RandomState} from the world's chunk source.
     *
     * <p>In MC 1.21+ (Mojang), {@code ServerChunkCache} exposes the
     * {@code RandomState} — no reflection needed.
     */
    private static RandomState tryGetRandomState(ServerLevel level) {
        try {
            return level.getChunkSource().randomState();
        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] Failed to get RandomState: {}", e.getMessage());
            return null;
        }
    }

    /** Expose for diagnostics. */
    public boolean isAvailable() {
        return true;  // if constructed, it's available
    }

    // ------------------------------------------------------------------
    // Named DensityFunction registry lookup (WS-4.1)
    // ------------------------------------------------------------------

    /**
     * Look up a registered {@link DensityFunction} by its overworld resource path.
     *
     * <p>Density functions such as {@code "overworld/offset"},
     * {@code "overworld/caves/spaghetti_2d"}, etc. are stored in Minecraft's
     * dynamic {@code DensityFunction} registry and can be sampled via
     * {@link #sampleRouterField3D(DensityFunction, int, int)} at any world
     * coordinate without a loaded chunk.
     *
     * <p>Returns {@code null} if the registry or the specific ID is not found
     * (e.g. the world uses a non-overworld or custom generator).
     *
     * @param path  the overworld-relative resource path, e.g.
     *              {@code "overworld/offset"} or
     *              {@code "overworld/caves/spaghetti_2d"}
     * @return the {@link DensityFunction}, or {@code null} on failure
     */
    public DensityFunction lookupDensityFunction(String path) {
        try {
            var dfReg = serverLevel.registryAccess()
                    .lookupOrThrow(Registries.DENSITY_FUNCTION);
            Identifier id = Identifier.fromNamespaceAndPath("minecraft", path);
            DensityFunction df = dfReg.getValue(id);
            if (df == null) {
                LOGGER.debug(
                        "[WorldNoiseAccess] DensityFunction not found: minecraft:{}", path);
            }
            return df;
        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] lookupDensityFunction({}) failed: {}",
                    path, e.getMessage());
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Raw NoiseRouter field sampling
    // ------------------------------------------------------------------

    /**
     * Expose the {@link NoiseRouter} for direct {@link DensityFunction} access.
     *
     * <p>The returned router's functions can be sampled at arbitrary (x, y, z)
     * positions using {@link DensityFunction#compute(DensityFunction.FunctionContext)}.
     */
    public NoiseRouter getNoiseRouter() {
        return randomState.router();
    }

    // -------------------------------------------------------------------------
    // Block-resolution sampling (WS-1.3 parity validation)
    // -------------------------------------------------------------------------

    /**
     * Sample {@code router.finalDensity()} at every block in a chunk column,
     * producing the same 16×384×16 grid written by the GPU compute shader to Binding 7.
     *
     * <p>Indexing matches the shader: {@code [lx + 16*lz] * 384 + (by + 64)}.
     *
     * @param router   the NoiseRouter for the dimension
     * @param sectionX chunk X coordinate
     * @param sectionZ chunk Z coordinate
     * @return flat {@code float[16 * 384 * 16]} array
     */
    public float[] sampleFinalDensityBlockRes(NoiseRouter router,
                                               int sectionX, int sectionZ) {
        float[] out = new float[16 * 384 * 16];
        DensityFunction df = router.finalDensity();
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;
        for (int lx = 0; lx < 16; lx++) {
            int bx = baseX + lx;
            for (int lz = 0; lz < 16; lz++) {
                int bz = baseZ + lz;
                int colBase = (lx + 16 * lz) * 384;
                for (int by = -64; by < 320; by++) {
                    out[colBase + (by + 64)] = (float) df.compute(
                            new DensityFunction.SinglePointContext(bx, by, bz));
                }
            }
        }
        return out;
    }

    /**
     * Sample a {@link DensityFunction} at 16×16 <em>block</em> resolution for a single Y.
     * Higher-resolution than {@link #sampleRouterField2D} (which uses 4×4 cells).
     *
     * @param df       the density function to evaluate
     * @param sectionX chunk X coordinate
     * @param sectionZ chunk Z coordinate
     * @param sampleY  block Y at which to evaluate
     * @return {@code float[16][16]}, lx-outer / lz-inner (x-major)
     */
    public float[][] sampleRouterField2DBlockRes(DensityFunction df,
                                                  int sectionX, int sectionZ,
                                                  int sampleY) {
        float[][] out = new float[16][16];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                out[lx][lz] = (float) df.compute(
                        new DensityFunction.SinglePointContext(
                                baseX + lx, sampleY, baseZ + lz));
            }
        }
        return out;
    }

    /**
     * Sample a {@link DensityFunction} at 4×4 cell resolution at a fixed Y.
     *
     * <p>Cell centre coordinates: {@code X = chunkBaseX + cx*4 + 2},
     * {@code Z = chunkBaseZ + cz*4 + 2}.  The Y coordinate is fixed for
     * fields that do not vary vertically (continents, erosion, ridges,
     * temperature, vegetation).
     *
     * @param df       the density function to evaluate
     * @param sectionX chunk X coordinate
     * @param sectionZ chunk Z coordinate
     * @param sampleY  block Y at which to evaluate
     * @return {@code float[4][4]} grid, cx-outer / cz-inner (x-major)
     */
    public float[][] sampleRouterField2D(DensityFunction df,
                                         int sectionX, int sectionZ,
                                         int sampleY) {
        float[][] out = new float[4][4];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;
        for (int cx = 0; cx < 4; cx++) {
            int x = baseX + cx * 4 + 2;  // cell centre X
            for (int cz = 0; cz < 4; cz++) {
                int z = baseZ + cz * 4 + 2;  // cell centre Z
                out[cx][cz] = (float) df.compute(
                        new DensityFunction.SinglePointContext(x, sampleY, z));
            }
        }
        return out;
    }

    /**
     * Sample a {@link DensityFunction} at 4×48×4 cell resolution — the full
     * overworld noise grid from Y=-64 to Y=320 (48 cells of 8 blocks each).
     *
     * <p>Cell centre coordinates:
     * <ul>
     *   <li>X: {@code chunkBaseX + cx*4 + 2}</li>
     *   <li>Z: {@code chunkBaseZ + cz*4 + 2}</li>
     *   <li>Y: {@code -64 + cy*8 + 4}  (centre of 8-block cell)</li>
     * </ul>
     *
     * @param df       the density function to evaluate
     * @param sectionX chunk X coordinate
     * @param sectionZ chunk Z coordinate
     * @return {@code float[4][48][4]} grid, {@code [cx][cy][cz]} order
     */
    public float[][][] sampleRouterField3D(DensityFunction df,
                                           int sectionX, int sectionZ) {
        float[][][] out = new float[4][48][4];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;
        for (int cx = 0; cx < 4; cx++) {
            int x = baseX + cx * 4 + 2;
            for (int cz = 0; cz < 4; cz++) {
                int z = baseZ + cz * 4 + 2;
                for (int cy = 0; cy < 48; cy++) {
                    int y = -64 + cy * 8 + 4;  // cell centre Y
                    out[cx][cy][cz] = (float) df.compute(
                            new DensityFunction.SinglePointContext(x, y, z));
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------
    // [DEPRECATED] SparseRoot noise input (13 channels, 4×2×4)
    // Superseded by v7 RouterField-based sampling (15ch, 4×4×4).
    // See: getRouterFieldFunctions(), sampleRouterFieldsForSection()
    // ------------------------------------------------------------------

    /**
     * Registry paths and special handling for the 13 SparseRoot noise channels.
     *
     * @deprecated Use {@link #getRouterFieldFunctions()} (15 v7 channels) instead.
     *
     * <p>Layout matches the Python training pipeline:
     * <pre>
     *   0  offset           overworld/offset
     *   1  factor           overworld/factor
     *   2  jaggedness       overworld/jaggedness
     *   3  depth            router.depth()  (null path → special)
     *   4  sloped_cheese    overworld/sloped_cheese
     *   5  y                cell-centre Y (null path → special)
     *   6  entrances        overworld/caves/entrances
     *   7  cheese_caves     overworld/caves/pillars
     *   8  spaghetti_2d     overworld/caves/spaghetti_2d
     *   9  roughness        overworld/caves/spaghetti_roughness_function
     *  10  noodle           overworld/caves/noodle
     *  11  base_3d_noise    overworld/base_3d_noise
     *  12  final_density    router.finalDensity() (null path → special)
     * </pre>
     */
    private static final String[] NOISE_3D_PATHS = {
        "overworld/offset",
        "overworld/factor",
        "overworld/jaggedness",
        null,   // depth → router.depth()
        "overworld/sloped_cheese",
        null,   // y → cell-centre Y value
        "overworld/caves/entrances",
        "overworld/caves/pillars",
        "overworld/caves/spaghetti_2d",
        "overworld/caves/spaghetti_roughness_function",
        "overworld/caves/noodle",
        "overworld/base_3d_noise",
        null,   // final_density → router.finalDensity()
    };

    /** Number of SparseRoot noise channels. */
    public static final int N_NOISE_3D = NOISE_3D_PATHS.length; // 13

    /**
     * Lazily-resolved density functions for {@link #sampleNoise3DForSection}.
     * Index 5 (Y) stays null; indices 3 and 12 use router fields.
     * Protected by {@code this} monitor on first initialization.
     */
    private volatile DensityFunction[] noise3dFunctions = null;

    /**
     * Resolve (once) and cache all {@link DensityFunction} objects needed for
     * the 13-channel SparseRoot noise input.
     */
    private DensityFunction[] getNoise3dFunctions() {
        if (noise3dFunctions != null) return noise3dFunctions;
        synchronized (this) {
            if (noise3dFunctions != null) return noise3dFunctions;
            NoiseRouter router = randomState.router();
            DensityFunction[] dfs = new DensityFunction[N_NOISE_3D];
            for (int i = 0; i < N_NOISE_3D; i++) {
                String path = NOISE_3D_PATHS[i];
                if (path == null) {
                    if (i == 3)  { dfs[i] = router.depth(); }           // depth
                    else if (i == 12) { dfs[i] = router.finalDensity(); } // final_density
                    // i == 5 (Y): stays null, handled per-cell
                } else {
                    DensityFunction df = lookupDensityFunction(path);
                    if (df == null) {
                        LOGGER.warn(
                                "[WorldNoiseAccess] noise3d[{}] '{}' not found — using zero",
                                i, path);
                        df = DensityFunctions.zero();
                    }
                    dfs[i] = df;
                }
            }
            noise3dFunctions = dfs;
            return dfs;
        }
    }

    /**
     * Sample the 13-channel SparseRoot noise input for a single L0 Voxy section.
     *
     * <p>Returns a flat {@code float[N_NOISE_3D * 4 * 2 * 4]} array in
     * {@code [field][cx][cy][cz]} (channel-outermost, C-contiguous) order,
     * matching the Python training pipeline's <br>
     * {@code noise_3d shape=(N, 13, 4, 2, 4)}.
     *
     * <p>The two Y-cells that make up a 16-block section at vanilla cell resolution
     * (8 blocks/cell) are sliced from the full 48-cell column:
     * <pre>
     *   cy_start = (sectionY + 4) * 2
     *   cy values sampled: cy_start, cy_start + 1
     * </pre>
     *
     * @param chunkX   chunk X coordinate (= wsX at L0)
     * @param chunkZ   chunk Z coordinate (= wsZ at L0)
     * @param sectionY section Y in native (L0) units, range [-4, 19]
     * @return flat {@code float[13 * 4 * 2 * 4 = 416]}, or an all-zeros array
     *         if the noise pipeline is unavailable
     */
    public float[] sampleNoise3DForSection(int chunkX, int chunkZ, int sectionY) {
        DensityFunction[] dfs = getNoise3dFunctions();
        float[] flat = new float[N_NOISE_3D * 4 * 2 * 4];

        // cy_start = (sectionY + 4) * 2; clamp to valid cell range [0, 47]
        int cyStart = (sectionY + 4) * 2;
        cyStart = Math.max(0, Math.min(46, cyStart));  // ensure cy_start+1 ≤ 47

        int baseX = chunkX * 16;
        int baseZ = chunkZ * 16;

        int flatIdx = 0;
        for (int field = 0; field < N_NOISE_3D; field++) {
            DensityFunction df = dfs[field];
            for (int cx = 0; cx < 4; cx++) {
                int x = baseX + cx * 4 + 2;
                for (int localCy = 0; localCy < 2; localCy++) {
                    int cy = cyStart + localCy;
                    int y = -64 + cy * 8 + 4;  // cell-centre Y in blocks
                    for (int cz = 0; cz < 4; cz++) {
                        int z = baseZ + cz * 4 + 2;
                        float val;
                        if (df == null) {
                            // field 5 = Y: emit cell-centre Y
                            val = y;
                        } else {
                            val = (float) df.compute(
                                    new DensityFunction.SinglePointContext(x, y, z));
                        }
                        flat[flatIdx++] = val;
                    }
                }
            }
        }
        return flat;
    }

    /**
     * Sample biome IDs at 4×2×4 noise cell resolution for a section.
     *
     * <p>Used by SparseRoot training data export. Biomes are sampled at
     * quarter-block resolution and mapped to stable integer IDs via
     * biome registry position.
     *
     * @param chunkX   chunk X coordinate
     * @param chunkZ   chunk Z coordinate
     * @param sectionY section Y in native units, range [-4, 19]
     * @return {@code int[4][2][4]} grid, {@code [cx][localCy][cz]} order
     */
    public int[][][] sampleBiomeIdsForSection(int chunkX, int chunkZ, int sectionY) {
        int[][][] result = new int[4][2][4];
        int cyStart = (sectionY + 4) * 2;
        cyStart = Math.max(0, Math.min(46, cyStart));

        int baseX = chunkX * 16;
        int baseZ = chunkZ * 16;

        try {
            Registry<Biome> biomeReg = serverLevel.registryAccess()
                    .lookupOrThrow(Registries.BIOME);

            for (int cx = 0; cx < 4; cx++) {
                int x = baseX + cx * 4 + 2;
                for (int localCy = 0; localCy < 2; localCy++) {
                    int cy = cyStart + localCy;
                    int y = -64 + cy * 8 + 4;
                    for (int cz = 0; cz < 4; cz++) {
                        int z = baseZ + cz * 4 + 2;

                        // Sample biome at quart coordinates
                        Holder<Biome> biomeEntry = biomeSource.getNoiseBiome(
                                x >> 2, y >> 2, z >> 2,
                                randomState.sampler());

                        // Get the index in the biome registry
                        int biomeId = biomeReg.getId(biomeEntry.value());
                        result[cx][localCy][cz] = biomeId >= 0 ? biomeId : 0;
                    }
                }
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] Failed to sample biome IDs: {}", e.getMessage());
            // Return all zeros on error
            for (int cx = 0; cx < 4; cx++) {
                for (int cy = 0; cy < 2; cy++) {
                    for (int cz = 0; cz < 4; cz++) {
                        result[cx][cy][cz] = 0;
                    }
                }
            }
        }
        return result;
    }

    // ------------------------------------------------------------------
    // V7 RouterField sampling (15 channels, 4×4×4 quart resolution)
    // ------------------------------------------------------------------

    /**
     * The 15 NoiseRouter fields sampled by the v7 pipeline, in index order.
     *
     * <p>These map 1:1 to the Java {@code RouterField} enum and the Python
     * {@code router_field.py}. Each field is accessed via the corresponding
     * {@link NoiseRouter} accessor.
     *
     * <p>Index order:
     * <pre>
     *   0  TEMPERATURE               router.temperature()
     *   1  VEGETATION                 router.vegetation()
     *   2  CONTINENTS                 router.continents()
     *   3  EROSION                    router.erosion()
     *   4  DEPTH                      router.depth()
     *   5  RIDGES                     router.ridges()
     *   6  PRELIMINARY_SURFACE_LEVEL  router.preliminarySurfaceLevel()
     *   7  FINAL_DENSITY              router.finalDensity()
     *   8  BARRIER                    router.barrierNoise()
     *   9  FLUID_LEVEL_FLOODEDNESS    router.fluidLevelFloodednessNoise()
     *  10  FLUID_LEVEL_SPREAD         router.fluidLevelSpreadNoise()
     *  11  LAVA                       router.lavaNoise()
     *  12  VEIN_TOGGLE                router.veinToggle()
     *  13  VEIN_RIDGED                router.veinRidged()
     *  14  VEIN_GAP                   router.veinGap()
     * </pre>
     */
    public static final int N_ROUTER_FIELDS = 15;

    /**
     * Lazily-resolved density functions for the 15-channel v7 pipeline.
     * Protected by {@code this} monitor on first initialization.
     */
    private volatile DensityFunction[] routerFieldFunctions = null;

    /**
     * Resolve (once) and cache all 15 {@link DensityFunction} objects for
     * the v7 RouterField set.
     */
    private DensityFunction[] getRouterFieldFunctions() {
        if (routerFieldFunctions != null) return routerFieldFunctions;
        synchronized (this) {
            if (routerFieldFunctions != null) return routerFieldFunctions;
            NoiseRouter router = randomState.router();
            DensityFunction[] dfs = new DensityFunction[N_ROUTER_FIELDS];
            dfs[0]  = router.temperature();
            dfs[1]  = router.vegetation();
            dfs[2]  = router.continents();
            dfs[3]  = router.erosion();
            dfs[4]  = router.depth();
            dfs[5]  = router.ridges();
            dfs[6]  = router.preliminarySurfaceLevel();
            dfs[7]  = router.finalDensity();
            dfs[8]  = router.barrierNoise();
            dfs[9]  = router.fluidLevelFloodednessNoise();
            dfs[10] = router.fluidLevelSpreadNoise();
            dfs[11] = router.lavaNoise();
            dfs[12] = router.veinToggle();
            dfs[13] = router.veinRidged();
            dfs[14] = router.veinGap();
            routerFieldFunctions = dfs;
            return dfs;
        }
    }

    /**
     * Sample all 15 RouterField channels for a single section at
     * <b>4×4×4 quart resolution</b>.
     *
     * <p>Returns a flat {@code float[15 * 4 * 4 * 4 = 960]} array in
     * {@code [field][qx][qy][qz]} (channel-outermost, C-contiguous) order,
     * matching the Python training pipeline's
     * {@code router_fields shape=(N, 15, 4, 4, 4)}.
     *
     * <p>Each quart cell is 4 blocks wide on X/Z and 8 blocks tall on Y
     * (matching vanilla cellWidth=4, cellHeight=8).  For a 16-block section,
     * there are 4 quarts on X/Z and 2 on Y.  The sample point is at the cell centre:
     * <pre>
     *   x = sectionX * 16 + qx * 4 + 2
     *   y = sectionY * 16 + qy * 8 + 4
     *   z = sectionZ * 16 + qz * 4 + 2
     * </pre>
     *
     * @param sectionX chunk X coordinate (= section X at L0)
     * @param sectionY section Y in native units, range [-4, 19]
     * @param sectionZ chunk Z coordinate (= section Z at L0)
     * @return flat {@code float[480]}, or all-zeros if the noise pipeline
     *         is unavailable
     */
    public float[] sampleRouterFieldsForSection(int sectionX, int sectionY, int sectionZ) {
        DensityFunction[] dfs = getRouterFieldFunctions();
        float[] flat = new float[N_ROUTER_FIELDS * 4 * 2 * 4];

        int baseX = sectionX * 16;
        int baseY = sectionY * 16;
        int baseZ = sectionZ * 16;

        int flatIdx = 0;
        for (int field = 0; field < N_ROUTER_FIELDS; field++) {
            DensityFunction df = dfs[field];
            for (int qx = 0; qx < 4; qx++) {
                int x = baseX + qx * 4 + 2;
                for (int qy = 0; qy < 2; qy++) {
                    int y = baseY + qy * 8 + 4;
                    for (int qz = 0; qz < 4; qz++) {
                        int z = baseZ + qz * 4 + 2;
                        flat[flatIdx++] = (float) df.compute(
                                new DensityFunction.SinglePointContext(x, y, z));
                    }
                }
            }
        }
        return flat;
    }

    /**
     * Sample biome IDs at <b>4×2×4 quart resolution</b> for a section (v7).
     *
     * <p>Returns {@code int[4][2][4]} in {@code [qx][qy][qz]} order.
     * Biomes are sampled via {@link BiomeSource#getNoiseBiome} at quart
     * coordinates and mapped to stable registry IDs.
     *
     * @param sectionX chunk X coordinate
     * @param sectionY section Y in native units, range [-4, 19]
     * @param sectionZ chunk Z coordinate
     * @return {@code int[4][2][4]} grid of biome registry IDs
     */
    public int[][][] sampleBiomeIdsForSectionV7(int sectionX, int sectionY, int sectionZ) {
        int[][][] result = new int[4][2][4];

        int baseX = sectionX * 16;
        int baseY = sectionY * 16;
        int baseZ = sectionZ * 16;

        try {
            Registry<Biome> biomeReg = serverLevel.registryAccess()
                    .lookupOrThrow(Registries.BIOME);

            for (int qx = 0; qx < 4; qx++) {
                int x = baseX + qx * 4 + 2;
                for (int qy = 0; qy < 2; qy++) {
                    int y = baseY + qy * 8 + 4;
                    for (int qz = 0; qz < 4; qz++) {
                        int z = baseZ + qz * 4 + 2;

                        Holder<Biome> biomeEntry = biomeSource.getNoiseBiome(
                                x >> 2, y >> 2, z >> 2,
                                randomState.sampler());

                        int biomeId = biomeReg.getId(biomeEntry.value());
                        result[qx][qy][qz] = biomeId >= 0 ? biomeId : 0;
                    }
                }
            }
        } catch (Exception e) {
            LOGGER.warn(
                    "[WorldNoiseAccess] Failed to sample v7 biome IDs: {}", e.getMessage());
        }
        return result;
    }
}
