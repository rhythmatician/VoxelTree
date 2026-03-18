package com.voxeltree.harvester.noise;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.context.CommandContext;

import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.permissions.Permissions;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.NoiseRouter;

/**
 * Server command {@code /dumpnoise <radius>} that extracts vanilla noise
 * signals using {@link WorldNoiseAccess} and serialises them to JSON files
 * under {@code run/noise_dumps/}.
 *
 * <p><b>No loaded chunks required.</b> All data is computed directly from
 * the {@link net.minecraft.world.level.levelgen.ChunkGenerator} and
 * {@link net.minecraft.world.level.levelgen.RandomState} — pure math, no world
 * state needed. This means noise can be dumped for <em>any</em> coordinate,
 * even if no player has ever visited the area.
 *
 * <p>Each dump file contains:
 * <ul>
 *   <li>{@code heightmap_surface} — 16×16 WORLD_SURFACE_WG heights (x-major)</li>
 *   <li>{@code heightmap_ocean_floor} — 16×16 OCEAN_FLOOR_WG heights (x-major)</li>
 *   <li>{@code biome_names} — 16×16 biome registry key names at block resolution
 *       (e.g. "minecraft:plains"), x-major</li>
 *   <li>{@code seed}, {@code chunk_x}, {@code chunk_z}</li>
 * </ul>
 *
 * <p>Usage: {@code /dumpnoise [radius]}  (default radius = 8 chunks)
 *
 * <p>The output JSON can be consumed by the Python training pipeline through
 * {@code scripts/add_column_heights.py} when the
 * {@code --noise-dump-dir} option is supplied.
 */
public final class NoiseDumperCommand {

    private static final Logger LOG = LoggerFactory.getLogger(NoiseDumperCommand.class);

    private NoiseDumperCommand() {}

    /**
     * Register {@code /dumpnoise [radius]} and sub-commands with the
     * Brigadier dispatcher.
     * Should be called from {@link com.voxeltree.harvester.DataHarvesterMod}.
     */
    public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(Commands.literal("dumpnoise")
            .requires(src -> src.permissions().hasPermission(Permissions.COMMANDS_GAMEMASTER))  // op level 2
            // /dumpnoise            (default radius 8)
            .executes(ctx -> execute(ctx, 8))
            // /dumpnoise <radius>
            .then(Commands.argument("radius", IntegerArgumentType.integer(1, 512))
                .executes(ctx -> execute(ctx,
                        IntegerArgumentType.getInteger(ctx, "radius"))))
            // /dumpnoise parity <cx> <cz>
            // Writes block-resolution Java vanilla density to parity_reports/java_chunk_<cx>_<cz>.json
            // Compare against parity_reports/gpu_chunk_0_0.json (auto-written at world load)
            // then run:  python tools/validate_shader_parity.py
            .then(Commands.literal("parity")
                .then(Commands.argument("cx", IntegerArgumentType.integer(-512, 512))
                    .then(Commands.argument("cz", IntegerArgumentType.integer(-512, 512))
                        .executes(ctx -> executeParity(ctx,
                                IntegerArgumentType.getInteger(ctx, "cx"),
                                IntegerArgumentType.getInteger(ctx, "cz"))))))
            // /dumpnoise stage1 [radius]   — WS-4.1
            // Dumps the 12 Stage 1 NN input features + final_density output per 4×48×4 cell.
            // Output: run/stage1_dumps/chunk_<cx>_<cz>.json
            .then(Commands.literal("stage1")
                .executes(ctx -> executeStage1(ctx, 8))
                .then(Commands.argument("radius", IntegerArgumentType.integer(1, 512))
                    .executes(ctx -> executeStage1(ctx,
                            IntegerArgumentType.getInteger(ctx, "radius")))))
            // /dumpnoise sparse_root [radius]   — WS-5.1+   [DEPRECATED: use /dumpnoise v7]
            // Dumps the 13 noise_3d channels + biome_ids per section (4×2×4 spatial resolution).
            // Output: run/sparse_root_dumps/section_<cx>_<cy>_<cz>.json
            // @deprecated Superseded by /dumpnoise v7 (15ch, 4×4×4). Retained for legacy data.
            .then(Commands.literal("sparse_root")
                .executes(ctx -> executeSparseRoot(ctx, 4))
                .then(Commands.argument("radius", IntegerArgumentType.integer(1, 512))
                    .executes(ctx -> executeSparseRoot(ctx,
                            IntegerArgumentType.getInteger(ctx, "radius")))))
            // /dumpnoise v7 [radius]
            // Dumps 15 RouterField channels + biome_ids per section at 4×4×4 quart resolution,
            // plus per-column heightmaps.  Output: run/v7_dumps/section_<cx>_<sy>_<cz>.json
            .then(Commands.literal("v7")
                .executes(ctx -> executeV7(ctx, 4))
                .then(Commands.argument("radius", IntegerArgumentType.integer(1, 512))
                    .executes(ctx -> executeV7(ctx,
                            IntegerArgumentType.getInteger(ctx, "radius")))))
        );
    }

    // ------------------------------------------------------------------
    // Parity handler (/dumpnoise parity <cx> <cz>)
    // ------------------------------------------------------------------

    /**
     * WS-1.3: Dump block-resolution Java vanilla final_density for one chunk.
     *
     * <p>Writes {@code run/parity_reports/java_chunk_<cx>_<cz>.json} with a flat
     * {@code density} array of 98,304 floats indexed {@code [lx + 16*lz] * 384 + (by + 64)}.
     */
    private static int executeParity(CommandContext<CommandSourceStack> ctx,
                                     int cx, int cz) {
        CommandSourceStack source = ctx.getSource();
        ServerLevel world = source.getLevel();

        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendFailure(Component.literal("[NoiseDumper] parity: noise pipeline unavailable"));
            return 0;
        }

        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[NoiseDumper] parity: sampling Java density for chunk (%d,%d) — 98,304 positions...",
                        cx, cz)),
                false);

        long t0 = System.currentTimeMillis();
        NoiseRouter router = noise.getNoiseRouter();
        float[] javaDensity = noise.sampleFinalDensityBlockRes(router, cx, cz);
        long samplingMs = System.currentTimeMillis() - t0;

        // Compute a quick surface-level mean to verify non-trivial output
        float sum = 0f;
        for (int col = 0; col < 256; col++) {
            // Y=63 (sea level): index = col*384 + (63+64) = col*384 + 127
            sum += javaDensity[col * 384 + 127];
        }
        float meanAtSeaLevel = sum / 256f;

        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[NoiseDumper] parity: sampled 98,304 values in %dms; mean density @ Y=63: %.5f",
                        samplingMs, meanAtSeaLevel)),
                false);

        // Write JSON
        try {
            Path dir = Path.of("parity_reports");
            Files.createDirectories(dir);
            Path out = dir.resolve(String.format("java_chunk_%d_%d.json", cx, cz));

            StringBuilder sb = new StringBuilder(2 * 1024 * 1024);
            sb.append("{\n");
            sb.append("  \"chunk_x\": ").append(cx).append(",\n");
            sb.append("  \"chunk_z\": ").append(cz).append(",\n");
            sb.append("  \"source\": \"java\",\n");
            sb.append("  \"y_min\": -64,\n");
            sb.append("  \"y_levels\": 384,\n");
            sb.append("  \"note\": \"density[lx + 16*lz][by - y_min]\",\n");
            sb.append("  \"density\": [");
            for (int i = 0; i < javaDensity.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format("%.6g", javaDensity[i]));
            }
            sb.append("]\n}\n");

            Files.writeString(out, sb);
            final long sizeKB = out.toFile().length() / 1024;
            source.sendSuccess(
                    () -> Component.literal(String.format(
                            "[NoiseDumper] parity: wrote %s (%dKB) — now run: python tools/validate_shader_parity.py",
                            out.toAbsolutePath(), sizeKB)),
                    false);
            return 1;
        } catch (IOException e) {
            source.sendFailure(Component.literal("[NoiseDumper] parity: write failed — " + e.getMessage()));
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Main handler
    // ------------------------------------------------------------------

    private static int execute(CommandContext<CommandSourceStack> ctx, int radius) {
        CommandSourceStack source = ctx.getSource();
        ServerLevel world = source.getLevel();

        // Output directory: <run>/noise_dumps/
        Path outDir = Path.of("noise_dumps");
        try {
            Files.createDirectories(outDir);
        } catch (IOException e) {
            source.sendFailure(Component.literal("[NoiseDumper] Cannot create output dir: " + e.getMessage()));
            return 0;
        }

        // Create WorldNoiseAccess — chunk-free noise pipeline.
        // If this fails, we cannot proceed (no fallback to chunk-based sampling).
        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendFailure(Component.literal(
                    "[NoiseDumper] Failed to initialise noise pipeline. "
                    + "RandomState unavailable — is this a vanilla overworld?"));
            return 0;
        }

        long seed = world.getSeed();

        // Find player origin chunk (or fallback to 0,0)
        BlockPos origin;
        try {
            origin = BlockPos.containing(source.getPosition());
        } catch (UnsupportedOperationException e) {
            origin = BlockPos.ZERO;
        }
        int centerCx = origin.getX() >> 4;
        int centerCz = origin.getZ() >> 4;

        int totalChunks = (2 * radius + 1) * (2 * radius + 1);
        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[NoiseDumper] Dumping %d chunks (%d×%d) centred (%d,%d) → %s",
                        totalChunks, 2 * radius + 1, 2 * radius + 1,
                        centerCx, centerCz, outDir.toAbsolutePath())),
                false);

        // Parallel worker pool — limited to avoid starving the server tick loop.
        int threadCount = 4;
        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[NoiseDumper] Using %d worker threads", threadCount)),
                false);

        ExecutorService pool = Executors.newFixedThreadPool(threadCount, r -> {
            Thread t = new Thread(r, "NoiseDumper-Worker");
            t.setDaemon(true);
            return t;
        });

        AtomicInteger dumped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        long startTime = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>(totalChunks);

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                final int fcx = centerCx + dx;
                final int fcz = centerCz + dz;
                futures.add(pool.submit(() -> {
                    try {
                        dumpChunkNoise(noise, fcx, fcz, seed, outDir);
                        dumped.incrementAndGet();
                    } catch (Exception e) {
                        LOG.warn("[NoiseDumper] Failed chunk (" + fcx + "," + fcz + "): " + e);
                        failed.incrementAndGet();
                    }

                    // Throttled progress: every 100 chunks or on the last one
                    int done = dumped.get() + failed.get();
                    if (done % 100 == 0 || done == totalChunks) {
                        double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                        double rate = elapsed > 0 ? done / elapsed : 0;
                        double eta = rate > 0 ? (totalChunks - done) / rate : 0;
                        source.sendSuccess(
                                () -> Component.literal(String.format(
                                        "[NoiseDumper] %d/%d (%.1f/s, ETA %.0fs)",
                                        done, totalChunks, rate, eta)),
                                false);
                    }
                }));
            }
        }

        // Coordinator thread waits for all futures, then shuts down the pool
        Thread coordinator = new Thread(() -> {
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    LOG.warn("[NoiseDumper] Future error: " + e);
                }
            }
            pool.shutdown();
            double totalSec = (System.currentTimeMillis() - startTime) / 1000.0;
            final int d = dumped.get();
            final int fl = failed.get();
            source.sendSuccess(
                    () -> Component.literal(String.format(
                            "[NoiseDumper] Done. %d dumped, %d failed in %.1fs (%.1f chunks/s)",
                            d, fl, totalSec, d / totalSec)),
                    false);
        }, "NoiseDumper-Coordinator");
        coordinator.setDaemon(true);
        coordinator.start();

        return 1;
    }

    // ------------------------------------------------------------------
    // Per-chunk dump (chunk-free)
    // ------------------------------------------------------------------

    /**
     * Dump noise signals for a single chunk position to JSON.
     *
     * <p>All data is computed via {@link WorldNoiseAccess} — no loaded chunk
     * or world state is required.
     *
     * @param noise  the noise access (provides heightmaps, biomes)
     * @param cx     chunk X coordinate
     * @param cz     chunk Z coordinate
     * @param seed   world seed
     * @param outDir output directory
     */
    static void dumpChunkNoise(WorldNoiseAccess noise,
                               int cx, int cz, long seed,
                               Path outDir) throws IOException {
        String filename = String.format("chunk_%d_%d.json", cx, cz);
        Path file = outDir.resolve(filename);

        // Sample both heightmaps in one populateNoise() pass via NoiseChunk.
        float[][][] heightmaps = noise.sampleBothHeightmaps(cx, cz);
        float[][] surfaceHm   = heightmaps[0];  // WORLD_SURFACE_WG
        float[][] oceanHm     = heightmaps[1];  // OCEAN_FLOOR_WG

        // Sample biomes at surface level (chunk-free via BiomeSource.getNoiseBiome())
        String[][] biomeNames = noise.sampleBiomeNames(cx, cz, surfaceHm);

        // ---- Raw NoiseRouter fields ----------------------------------------
        NoiseRouter router = noise.getNoiseRouter();
        final int seaLevel = 63;

        // 2D climate/shape fields (4×4 cell grid, sampled at sea level)
        float[][] continents  = noise.sampleRouterField2D(router.continents(),  cx, cz, seaLevel);
        float[][] erosion     = noise.sampleRouterField2D(router.erosion(),     cx, cz, seaLevel);
        float[][] ridges      = noise.sampleRouterField2D(router.ridges(),      cx, cz, seaLevel);
        float[][] temperature = noise.sampleRouterField2D(router.temperature(), cx, cz, seaLevel);
        float[][] vegetation  = noise.sampleRouterField2D(router.vegetation(),  cx, cz, seaLevel);
        float[][] depth       = noise.sampleRouterField2D(router.depth(),       cx, cz, seaLevel);

        // 3D density field (4×48×4 cell grid, full overworld Y range -64..320)
        float[][][] finalDensity = noise.sampleRouterField3D(router.finalDensity(), cx, cz);

        // Build JSON
        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");

        // Heightmaps — flat 256 values, x-major (x outer, z inner)
        sb.append("  \"heightmap_surface\": [");
        appendFloatGrid(sb, surfaceHm);
        sb.append("],\n");

        sb.append("  \"heightmap_ocean_floor\": [");
        appendFloatGrid(sb, oceanHm);
        sb.append("],\n");

        // Biome names — flat 256 strings, x-major (block resolution)
        sb.append("  \"biome_names\": [");
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                sb.append('"').append(biomeNames[x][z]).append('"');
            }
        }
        sb.append("],\n");

        // 2D router fields — flat 16 values each (4×4), cx-outer / cz-inner
        sb.append("  \"continents\": [");
        appendCell2D(sb, continents);
        sb.append("],\n");

        sb.append("  \"erosion\": [");
        appendCell2D(sb, erosion);
        sb.append("],\n");

        sb.append("  \"ridges\": [");
        appendCell2D(sb, ridges);
        sb.append("],\n");

        sb.append("  \"temperature\": [");
        appendCell2D(sb, temperature);
        sb.append("],\n");

        sb.append("  \"vegetation\": [");
        appendCell2D(sb, vegetation);
        sb.append("],\n");

        sb.append("  \"depth\": [");
        appendCell2D(sb, depth);
        sb.append("],\n");

        // 3D final_density — flat 768 values (4×48×4), cx-outer / cy-middle / cz-inner
        sb.append("  \"final_density\": [");
        appendCell3D(sb, finalDensity);
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Append a flat 16×16 heightmap as integers (x-major). */
    private static void appendFloatGrid(StringBuilder sb, float[][] grid) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                // Cast to int — heightmaps are whole-block Y values
                sb.append((int) grid[x][z]);
            }
        }
    }

    /**
     * Append a 4×4 float cell grid as a flat JSON array (cx-outer, cz-inner).
     */
    private static void appendCell2D(StringBuilder sb, float[][] grid) {
        boolean first = true;
        for (int cx = 0; cx < 4; cx++) {
            for (int cz = 0; cz < 4; cz++) {
                if (!first) sb.append(',');
                first = false;
                sb.append(String.format("%.6g", grid[cx][cz]));
            }
        }
    }

    /**
     * Append a 4×48×4 float cell grid as a flat JSON array (cx-outer, cy-middle, cz-inner).
     */
    private static void appendCell3D(StringBuilder sb, float[][][] grid) {
        boolean first = true;
        for (int cx = 0; cx < 4; cx++) {
            for (int cy = 0; cy < 48; cy++) {
                for (int cz = 0; cz < 4; cz++) {
                    if (!first) sb.append(',');
                    first = false;
                    sb.append(String.format("%.6g", grid[cx][cy][cz]));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    //  Stage 1 training data — /dumpnoise stage1 <radius>  (WS-4.1)
    // ------------------------------------------------------------------

    /**
     * Names for the 12 Stage 1 NN input features sampled from the density
     * function registry, in order.
     */
    private static final String[][] STAGE1_FIELDS = {
        // { JSON key, registry path (null → special handling) }
        {"offset",             "overworld/offset"},
        {"factor",             "overworld/factor"},
        {"jaggedness",         "overworld/jaggedness"},
        {"depth",              null},                                   // router.depth() direct
        {"sloped_cheese",      "overworld/sloped_cheese"},
        {"y",                  null},                                   // cell centre Y
        {"entrances",          "overworld/caves/entrances"},
        {"cheese_caves",       "overworld/caves/pillars"},
        {"spaghetti_2d",       "overworld/caves/spaghetti_2d"},
        {"roughness",          "overworld/caves/spaghetti_roughness_function"},
        {"noodle",             "overworld/caves/noodle"},
        {"base_3d_noise",      "overworld/base_3d_noise"},
    };

    /**
     * Execute {@code /dumpnoise stage1 <radius>}.
     */
    private static int executeStage1(CommandContext<CommandSourceStack> ctx,
                                      int radius) {
        CommandSourceStack source = ctx.getSource();
        ServerLevel world = source.getLevel();

        Path outDir = Path.of("stage1_dumps");
        try {
            Files.createDirectories(outDir);
        } catch (IOException e) {
            source.sendFailure(Component.literal("[Stage1Dump] Cannot create output dir: " + e.getMessage()));
            return 0;
        }

        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendFailure(Component.literal(
                    "[Stage1Dump] Failed to initialise noise pipeline."));
            return 0;
        }

        long seed = world.getSeed();
        int[] centre = {0, 0};
        try {
            BlockPos bOrigin = BlockPos.containing(source.getPosition());
            centre[0] = bOrigin.getX() >> 4;
            centre[1] = bOrigin.getZ() >> 4;
        } catch (UnsupportedOperationException e) {
            // keep (0,0)
        }
        final int centerCx = centre[0];
        final int centerCz = centre[1];

        int totalChunks = (2 * radius + 1) * (2 * radius + 1);
        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[Stage1Dump] Dumping %d chunks r=%d centred (%d,%d) → %s",
                        totalChunks, radius, centerCx, centerCz,
                        outDir.toAbsolutePath())),
                false);

        // Resolve all DensityFunction objects once (avoids per-chunk registry lookups)
        NoiseRouter router = noise.getNoiseRouter();
        DensityFunction[] dfs = resolveStageDensityFunctions(noise, router);
        if (dfs == null) {
            source.sendFailure(Component.literal(
                    "[Stage1Dump] Failed to resolve density functions — check server log."));
            return 0;
        }

        int threadCount = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threadCount, r -> {
            Thread t = new Thread(r, "Stage1Dumper-Worker");
            t.setDaemon(true);
            return t;
        });

        AtomicInteger dumped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        long startTime = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>(totalChunks);

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                final int fcx = centerCx + dx;
                final int fcz = centerCz + dz;
                futures.add(pool.submit(() -> {
                    try {
                        dumpChunkNoiseStage1(noise, dfs, fcx, fcz, seed, outDir);
                        dumped.incrementAndGet();
                    } catch (Exception e) {
                        LOG.warn("[Stage1Dump] Failed chunk ({},{}): {}", fcx, fcz, e.getMessage());
                        failed.incrementAndGet();
                    }
                    int done = dumped.get() + failed.get();
                    if (done % 100 == 0 || done == totalChunks) {
                        double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                        double rate = elapsed > 0 ? done / elapsed : 0;
                        source.sendSuccess(
                                () -> Component.literal(String.format(
                                        "[Stage1Dump] %d/%d (%.1f/s)", done, totalChunks, rate)),
                                false);
                    }
                }));
            }
        }

        Thread coordinator = new Thread(() -> {
            for (Future<?> f : futures) {
                try { f.get(); } catch (Exception e) { LOG.warn("[Stage1Dump] {}", e.getMessage()); }
            }
            pool.shutdown();
            double totalSec = (System.currentTimeMillis() - startTime) / 1000.0;
            final int d = dumped.get(), fl = failed.get();
            source.sendSuccess(
                    () -> Component.literal(String.format(
                            "[Stage1Dump] Done. %d dumped, %d failed in %.1fs",
                            d, fl, totalSec)),
                    false);
        }, "Stage1Dumper-Coordinator");
        coordinator.setDaemon(true);
        coordinator.start();

        return 1;
    }

    /**
     * Resolve all 12 Stage 1 density functions from the registry.
     */
    private static DensityFunction[] resolveStageDensityFunctions(WorldNoiseAccess noise,
                                                                     NoiseRouter router) {
        DensityFunction[] dfs = new DensityFunction[STAGE1_FIELDS.length];
        for (int i = 0; i < STAGE1_FIELDS.length; i++) {
            String path = STAGE1_FIELDS[i][1];
            if (path == null) {
                if ("depth".equals(STAGE1_FIELDS[i][0])) {
                    dfs[i] = router.depth();  // direct router field
                }
                // "y" field: dfs[i] stays null → handled by cell centre Y in dump loop
                continue;
            }
            DensityFunction df = noise.lookupDensityFunction(path);
            if (df == null) {
                LOG.warn("[Stage1Dump] Could not resolve '{}' — will use 0.0", path);
                df = DensityFunctions.zero();
            }
            dfs[i] = df;
        }
        return dfs;
    }

    /**
     * Dump Stage 1 training data for a single chunk to JSON.
     */
    static void dumpChunkNoiseStage1(WorldNoiseAccess noise,
                                      DensityFunction[] dfs,
                                      int cx, int cz, long seed,
                                      Path outDir) throws IOException {
        String filename = String.format("chunk_%d_%d.json", cx, cz);
        Path file = outDir.resolve(filename);

        // Pre-compute all 3D fields
        float[][][] finalDensity = noise.sampleRouterField3D(
                noise.getNoiseRouter().finalDensity(), cx, cz);

        float[][][][] fieldValues = new float[STAGE1_FIELDS.length][4][48][4];
        float[][][] yValues = new float[4][48][4];

        // Precompute y (cell centre)
        for (int cy = 0; cy < 48; cy++) {
            float yCentre = -64f + cy * 8f + 4f;
            for (int ccx = 0; ccx < 4; ccx++)
                for (int ccz = 0; ccz < 4; ccz++)
                    yValues[ccx][cy][ccz] = yCentre;
        }

        for (int fieldIdx = 0; fieldIdx < STAGE1_FIELDS.length; fieldIdx++) {
            String fieldName = STAGE1_FIELDS[fieldIdx][0];
            DensityFunction df = dfs[fieldIdx];

            if ("y".equals(fieldName)) {
                fieldValues[fieldIdx] = yValues;
            } else if (df != null) {
                fieldValues[fieldIdx] = noise.sampleRouterField3D(df, cx, cz);
            }
        }

        // --- Write JSON
        StringBuilder sb = new StringBuilder(32 * 1024);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");
        sb.append("  \"cell_resolution\": \"4x48x4\",\n");
        sb.append("  \"note\": \"flat arrays indexed [cx*48*4 + cy*4 + cz], cell centres\",\n");

        for (int fieldIdx = 0; fieldIdx < STAGE1_FIELDS.length; fieldIdx++) {
            sb.append("  \"").append(STAGE1_FIELDS[fieldIdx][0]).append("\": [");
            appendCell3D(sb, fieldValues[fieldIdx]);
            sb.append(fieldIdx < STAGE1_FIELDS.length - 1 ? "],\n" : "],\n");
        }

        sb.append("  \"final_density\": [");
        appendCell3D(sb, finalDensity);
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    //  [DEPRECATED] SparseRoot training data — /dumpnoise sparse_root
    //  Superseded by /dumpnoise v7 (15 RouterField channels, 4×4×4).
    //  Retained for backward compatibility with legacy training data.
    // ------------------------------------------------------------------

    /**
     * Names for the 13 SparseRoot noise_3d channels.
     *
     * @deprecated Use {@code V7_FIELD_NAMES} and {@code /dumpnoise v7} instead.
     */
    private static final String[][] SPARSE_ROOT_NOISE_FIELDS = {
        {"offset",             "overworld/offset"},
        {"factor",             "overworld/factor"},
        {"jaggedness",         "overworld/jaggedness"},
        {"depth",              null},                                   // router.depth()
        {"sloped_cheese",      "overworld/sloped_cheese"},
        {"y",                  null},                                   // cell centre Y
        {"entrances",          "overworld/caves/entrances"},
        {"pillars",            "overworld/caves/pillars"},
        {"spaghetti_2d",       "overworld/caves/spaghetti_2d"},
        {"spaghetti_roughness", "overworld/caves/spaghetti_roughness_function"},
        {"noodle",             "overworld/caves/noodle"},
        {"base_3d_noise",      "overworld/base_3d_noise"},
        {"final_density",      null},                                   // router.finalDensity()
    };

    /**
     * Execute {@code /dumpnoise sparse_root <radius>}.
     */
    private static int executeSparseRoot(CommandContext<CommandSourceStack> ctx,
                                         int radius) {
        CommandSourceStack source = ctx.getSource();
        ServerLevel world = source.getLevel();

        Path outDir = Path.of("sparse_root_dumps");
        try {
            Files.createDirectories(outDir);
        } catch (IOException e) {
            source.sendFailure(Component.literal("[SparseRootDump] Cannot create output dir: " + e.getMessage()));
            return 0;
        }

        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendFailure(Component.literal(
                    "[SparseRootDump] Failed to initialise noise pipeline."));
            return 0;
        }

        long seed = world.getSeed();
        int[] centre = {0, 0};
        try {
            BlockPos bOrigin = BlockPos.containing(source.getPosition());
            centre[0] = bOrigin.getX() >> 4;
            centre[1] = bOrigin.getZ() >> 4;
        } catch (UnsupportedOperationException e) {
            // keep (0,0)
        }
        final int centerCx = centre[0];
        final int centerCz = centre[1];

        int totalSections = (2 * radius + 1) * (2 * radius + 1) * 24;
        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[SparseRootDump] Dumping %d sections r=%d centred (%d,%d) → %s",
                        totalSections, radius, centerCx, centerCz,
                        outDir.toAbsolutePath())),
                false);

        // Resolve all 13 DensityFunction objects once
        NoiseRouter router = noise.getNoiseRouter();
        DensityFunction[] dfs = resolveSparseRootDensityFunctions(noise, router);
        if (dfs == null) {
            source.sendFailure(Component.literal(
                    "[SparseRootDump] Failed to resolve density functions — check server log."));
            return 0;
        }

        int threadCount = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threadCount, r -> {
            Thread t = new Thread(r, "SparseRootDumper-Worker");
            t.setDaemon(true);
            return t;
        });

        AtomicInteger dumped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        long startTime = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>(totalSections);

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                final int fcx = centerCx + dx;
                final int fcz = centerCz + dz;
                for (int sy = -4; sy <= 19; sy++) {
                    final int sectionY = sy;
                    futures.add(pool.submit(() -> {
                        try {
                            dumpSectionNoiseSparseRoot(noise, dfs, fcx, sectionY, fcz, seed, outDir);
                            dumped.incrementAndGet();
                        } catch (Exception e) {
                            LOG.warn("[SparseRootDump] Failed section ({},{},{}): {}",
                                    fcx, sectionY, fcz, e.getMessage());
                            failed.incrementAndGet();
                        }
                        int done = dumped.get() + failed.get();
                        if (done % 100 == 0 || done == totalSections) {
                            double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                            double rate = elapsed > 0 ? done / elapsed : 0;
                            source.sendSuccess(
                                    () -> Component.literal(String.format(
                                            "[SparseRootDump] %d/%d (%.1f/s)", done, totalSections, rate)),
                                    false);
                        }
                    }));
                }
            }
        }

        Thread coordinator = new Thread(() -> {
            for (Future<?> f : futures) {
                try { f.get(); } catch (Exception e) { LOG.warn("[SparseRootDump] {}", e.getMessage()); }
            }
            pool.shutdown();
            double totalSec = (System.currentTimeMillis() - startTime) / 1000.0;
            final int d = dumped.get(), fl = failed.get();
            source.sendSuccess(
                    () -> Component.literal(String.format(
                            "[SparseRootDump] Done. %d dumped, %d failed in %.1fs",
                            d, fl, totalSec)),
                    false);
        }, "SparseRootDumper-Coordinator");
        coordinator.setDaemon(true);
        coordinator.start();

        return 1;
    }

    /**
     * Resolve all 13 SparseRoot density functions from the registry.
     */
    private static DensityFunction[] resolveSparseRootDensityFunctions(WorldNoiseAccess noise,
                                                                       NoiseRouter router) {
        DensityFunction[] dfs = new DensityFunction[SPARSE_ROOT_NOISE_FIELDS.length];
        for (int i = 0; i < SPARSE_ROOT_NOISE_FIELDS.length; i++) {
            String fieldName = SPARSE_ROOT_NOISE_FIELDS[i][0];
            String path = SPARSE_ROOT_NOISE_FIELDS[i][1];

            if (path == null) {
                if ("depth".equals(fieldName)) {
                    dfs[i] = router.depth();
                } else if ("final_density".equals(fieldName)) {
                    dfs[i] = router.finalDensity();
                }
                continue;
            }

            DensityFunction df = noise.lookupDensityFunction(path);
            if (df == null) {
                LOG.warn("[SparseRootDump] Could not resolve '{}' — will use 0.0", path);
                df = DensityFunctions.zero();
            }
            dfs[i] = df;
        }
        return dfs;
    }

    /**
     * Dump SparseRoot training data for a single section to JSON.
     */
    static void dumpSectionNoiseSparseRoot(WorldNoiseAccess noise,
                                           DensityFunction[] dfs,
                                           int cx, int sy, int cz,
                                           long seed,
                                           Path outDir) throws IOException {
        String filename = String.format("section_%d_%d_%d.json", cx, sy, cz);
        Path file = outDir.resolve(filename);

        // Sample all 13 noise fields + biome IDs for this section
        float[][][][] noiseSamples = new float[13][4][2][4];
        for (int field = 0; field < 13; field++) {
            float[][][] fieldData = noiseSamples[field];
            int cyStart = (sy + 4) * 2;
            int baseX = cx * 16;
            int baseZ = cz * 16;

            DensityFunction df = dfs[field];
            for (int cx_cell = 0; cx_cell < 4; cx_cell++) {
                int x = baseX + cx_cell * 4 + 2;
                for (int localCy = 0; localCy < 2; localCy++) {
                    int cy = cyStart + localCy;
                    int y = -64 + cy * 8 + 4;

                    if (df == null) {
                        for (int cz_cell = 0; cz_cell < 4; cz_cell++) {
                            fieldData[cx_cell][localCy][cz_cell] = (float) y;
                        }
                    } else {
                        for (int cz_cell = 0; cz_cell < 4; cz_cell++) {
                            int z = baseZ + cz_cell * 4 + 2;
                            DensityFunction.SinglePointContext pos =
                                    new DensityFunction.SinglePointContext(x, y, z);
                            fieldData[cx_cell][localCy][cz_cell] = (float) df.compute(pos);
                        }
                    }
                }
            }
        }

        // Sample biome IDs at 4x2x4 resolution
        int[][][] biomeIds = noise.sampleBiomeIdsForSection(cx, sy, cz);

        // --- Write JSON
        StringBuilder sb = new StringBuilder(16 * 1024);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"section_y\": ").append(sy).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");
        sb.append("  \"cell_resolution\": \"4x2x4\",\n");
        sb.append("  \"note\": \"flat arrays indexed [cx*2*4 + localCy*4 + cz]\",\n");

        String[] fieldNames = {
            "offset", "factor", "jaggedness", "depth", "sloped_cheese",
            "y", "entrances", "pillars", "spaghetti_2d", "spaghetti_roughness",
            "noodle", "base_3d_noise", "final_density"
        };
        for (int fieldIdx = 0; fieldIdx < 13; fieldIdx++) {
            sb.append("  \"").append(fieldNames[fieldIdx]).append("\": [");
            boolean first = true;
            for (int cx_cell = 0; cx_cell < 4; cx_cell++) {
                for (int localCy = 0; localCy < 2; localCy++) {
                    for (int cz_cell = 0; cz_cell < 4; cz_cell++) {
                        if (!first) sb.append(',');
                        first = false;
                        sb.append(String.format("%.6g", noiseSamples[fieldIdx][cx_cell][localCy][cz_cell]));
                    }
                }
            }
            sb.append(fieldIdx < 12 ? "],\n" : "],\n");
        }

        // Write biome IDs
        sb.append("  \"biome_ids\": [");
        boolean first = true;
        for (int cx_cell = 0; cx_cell < 4; cx_cell++) {
            for (int localCy = 0; localCy < 2; localCy++) {
                for (int cz_cell = 0; cz_cell < 4; cz_cell++) {
                    if (!first) sb.append(',');
                    first = false;
                    sb.append(biomeIds[cx_cell][localCy][cz_cell]);
                }
            }
        }
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    //  V7 RouterField dump — /dumpnoise v7 <radius>
    // ------------------------------------------------------------------

    /**
     * The 15 JSON field names for the v7 RouterField channels, in index order.
     * Must match {@code router_field.py} / Java {@code RouterField} ordinals.
     */
    private static final String[] V7_FIELD_NAMES = {
        "temperature",               //  0
        "vegetation",                //  1
        "continents",                //  2
        "erosion",                   //  3
        "depth",                     //  4
        "ridges",                    //  5
        "preliminary_surface_level", //  6
        "final_density",             //  7
        "barrier",                   //  8
        "fluid_level_floodedness",   //  9
        "fluid_level_spread",        // 10
        "lava",                      // 11
        "vein_toggle",               // 12
        "vein_ridged",               // 13
        "vein_gap",                  // 14
    };

    /**
     * Execute {@code /dumpnoise v7 <radius>}.
     *
     * <p>Dumps all 15 RouterField channels at 4×4×4 quart resolution plus
     * 4×4×4 biome IDs and per-column heightmaps for every section in the
     * overworld column range.  Output: {@code v7_dumps/section_<cx>_<sy>_<cz>.json}.
     */
    private static int executeV7(CommandContext<CommandSourceStack> ctx,
                                 int radius) {
        CommandSourceStack source = ctx.getSource();
        ServerLevel world = source.getLevel();

        Path outDir = Path.of("v7_dumps");
        try {
            Files.createDirectories(outDir);
        } catch (IOException e) {
            source.sendFailure(Component.literal("[V7Dump] Cannot create output dir: " + e.getMessage()));
            return 0;
        }

        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendFailure(Component.literal("[V7Dump] Failed to initialise noise pipeline."));
            return 0;
        }

        long seed = world.getSeed();
        int[] centre = {0, 0};
        try {
            BlockPos bOrigin = BlockPos.containing(source.getPosition());
            centre[0] = bOrigin.getX() >> 4;
            centre[1] = bOrigin.getZ() >> 4;
        } catch (UnsupportedOperationException e) {
            // keep (0,0)
        }
        final int centerCx = centre[0];
        final int centerCz = centre[1];

        // 24 sections per column: sectionY -4..19
        int totalSections = (2 * radius + 1) * (2 * radius + 1) * 24;
        source.sendSuccess(
                () -> Component.literal(String.format(
                        "[V7Dump] Dumping %d sections (15ch 4×4×4) r=%d centred (%d,%d) → %s",
                        totalSections, radius, centerCx, centerCz,
                        outDir.toAbsolutePath())),
                false);

        int threadCount = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threadCount, r -> {
            Thread t = new Thread(r, "V7Dumper-Worker");
            t.setDaemon(true);
            return t;
        });

        AtomicInteger dumped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        long startTime = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>(totalSections);

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                final int fcx = centerCx + dx;
                final int fcz = centerCz + dz;

                // Pre-sample the per-column heightmaps once (shared by all 24 sections)
                float[][][] colHeightmaps;
                try {
                    colHeightmaps = noise.sampleBothHeightmaps(fcx, fcz);
                } catch (Exception e) {
                    LOG.warn("[V7Dump] Failed heightmaps for column ({},{}): {}", fcx, fcz, e.getMessage());
                    colHeightmaps = new float[2][16][16]; // zeros fallback
                }
                final float[][] surfaceHm = colHeightmaps[0];
                final float[][] oceanFloorHm = colHeightmaps[1];

                for (int sy = -4; sy <= 19; sy++) {
                    final int sectionY = sy;
                    futures.add(pool.submit(() -> {
                        try {
                            dumpSectionNoiseV7(noise, fcx, sectionY, fcz, seed,
                                    surfaceHm, oceanFloorHm, outDir);
                            dumped.incrementAndGet();
                        } catch (Exception e) {
                            LOG.warn("[V7Dump] Failed section ({},{},{}): {}",
                                    fcx, sectionY, fcz, e.getMessage());
                            failed.incrementAndGet();
                        }
                        int done = dumped.get() + failed.get();
                        if (done % 200 == 0 || done == totalSections) {
                            double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                            double rate = elapsed > 0 ? done / elapsed : 0;
                            source.sendSuccess(
                                    () -> Component.literal(String.format(
                                            "[V7Dump] %d/%d (%.1f/s)", done, totalSections, rate)),
                                    false);
                        }
                    }));
                }
            }
        }

        Thread coordinator = new Thread(() -> {
            for (Future<?> f : futures) {
                try { f.get(); } catch (Exception e) { LOG.warn("[V7Dump] {}", e.getMessage()); }
            }
            pool.shutdown();
            double totalSec = (System.currentTimeMillis() - startTime) / 1000.0;
            final int d = dumped.get(), fl = failed.get();
            source.sendSuccess(
                    () -> Component.literal(String.format(
                            "[V7Dump] Done. %d dumped, %d failed in %.1fs (%.1f sections/s)",
                            d, fl, totalSec, d / totalSec)),
                    false);
        }, "V7Dumper-Coordinator");
        coordinator.setDaemon(true);
        coordinator.start();

        return 1;
    }

    /**
     * Dump v7 training data for a single section to JSON.
     *
     * <p>Each JSON file contains:
     * <ul>
     *   <li>15 RouterField channels as flat 64-element arrays (4×4×4, channel-first)</li>
     *   <li>{@code biome_ids}: flat 64-element int array (4×4×4)</li>
     *   <li>{@code heightmap_surface}: flat 256-element int array (16×16, x-major)</li>
     *   <li>{@code heightmap_ocean_floor}: flat 256-element int array (16×16, x-major)</li>
     * </ul>
     *
     * <p>Flat ordering is {@code [qx][qy][qz]} = {@code qx * 16 + qy * 4 + qz}.
     */
    static void dumpSectionNoiseV7(WorldNoiseAccess noise,
                                   int cx, int sy, int cz,
                                   long seed,
                                   float[][] surfaceHm,
                                   float[][] oceanFloorHm,
                                   Path outDir) throws IOException {
        String filename = String.format("section_%d_%d_%d.json", cx, sy, cz);
        Path file = outDir.resolve(filename);

        // 15-channel flat array [field * 64 + qx * 16 + qy * 4 + qz]
        float[] routerFlat = noise.sampleRouterFieldsForSection(cx, sy, cz);

        // 4×4×4 biome IDs
        int[][][] biomeIds = noise.sampleBiomeIdsForSectionV7(cx, sy, cz);

        // --- Write JSON
        StringBuilder sb = new StringBuilder(24 * 1024);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"section_y\": ").append(sy).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");
        sb.append("  \"version\": 7,\n");
        sb.append("  \"cell_resolution\": \"4x4x4\",\n");
        sb.append("  \"note\": \"flat arrays indexed [qx*16 + qy*4 + qz]; channel-first for router fields\",\n");

        // Write each of the 15 router fields as a separate JSON key,
        // each with 64 values (4×4×4).
        for (int field = 0; field < WorldNoiseAccess.N_ROUTER_FIELDS; field++) {
            sb.append("  \"").append(V7_FIELD_NAMES[field]).append("\": [");
            int base = field * 64;
            for (int i = 0; i < 64; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format("%.6g", routerFlat[base + i]));
            }
            sb.append("],\n");
        }

        // Biome IDs — flat 64 values [qx][qy][qz]
        sb.append("  \"biome_ids\": [");
        boolean first = true;
        for (int qx = 0; qx < 4; qx++) {
            for (int qy = 0; qy < 4; qy++) {
                for (int qz = 0; qz < 4; qz++) {
                    if (!first) sb.append(',');
                    first = false;
                    sb.append(biomeIds[qx][qy][qz]);
                }
            }
        }
        sb.append("],\n");

        // Per-column heightmaps (shared across all sections in same column)
        sb.append("  \"heightmap_surface\": [");
        appendFloatGrid(sb, surfaceHm);
        sb.append("],\n");

        sb.append("  \"heightmap_ocean_floor\": [");
        appendFloatGrid(sb, oceanFloorHm);
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }
}
