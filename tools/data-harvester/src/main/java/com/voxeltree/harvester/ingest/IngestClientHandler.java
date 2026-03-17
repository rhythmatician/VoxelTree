package com.voxeltree.harvester.ingest;

import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.Holder;
import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.network.RegistryFriendlyByteBuf;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.DataLayer;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.chunk.PalettedContainer;
import net.minecraft.world.level.chunk.PalettedContainerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

/**
 * Client-side handler for {@link IngestPayload} packets.
 *
 * <p>When the server's {@code /ingestall} command sends a chunk's sections,
 * this handler deserialises them back into vanilla {@link LevelChunkSection}
 * objects and calls Voxy's {@code VoxelIngestService.rawIngest()} via
 * reflection — <em>the exact same code path</em> as VoxyWorldGen v2's
 * {@code NetworkClientHandler.handleLODData()}.
 *
 * <p>This guarantees <strong>absolute parity</strong> with Voxy's
 * ingestion algorithm:
 * <ol>
 *   <li>{@code WorldConversionFactory.convert()} — BlockState/Biome/Light → packed long[]</li>
 *   <li>{@code WorldConversionFactory.mipSection()} — build L0–L4 mip levels via {@code Mipper}</li>
 *   <li>{@code WorldUpdater.insertUpdate()} — write 32³ WorldSections + cascade up LOD tree</li>
 *   <li>{@code SectionSavingService} → ZSTD-compressed RocksDB persistence</li>
 * </ol>
 */
public class IngestClientHandler {

    private static final Logger LOGGER = LoggerFactory.getLogger("dataharvester");

    // ---- Voxy reflection bridge (initialised on first payload) ----
    private static boolean bridgeInitialized;
    private static MethodHandle rawIngestMethod;
    private static MethodHandle worldIdOfMethod;
    private static int sectionsIngested;

    /**
     * Register the client-side payload handler.
     * Call from {@link com.voxeltree.harvester.DataHarvesterClient#onInitializeClient()}.
     */
    public static void init() {
        ClientPlayNetworking.registerGlobalReceiver(IngestPayload.TYPE,
                (payload, context) -> context.client().execute(
                        () -> handlePayload(payload)));
        LOGGER.info("[DataHarvester] Client ingest handler registered.");
    }

    // ==================== payload processing ====================

    /**
     * Deserialise each section from the payload and ingest it into Voxy.
     *
     * <p>This mirrors VoxyWorldGen v2's
     * {@code NetworkClientHandler.handleLODData()} line-for-line:
     * <ol>
     *   <li>Create an empty {@link LevelChunkSection} from a {@link PalettedContainerFactory}</li>
     *   <li>Read the serialised block-state palette into the section</li>
     *   <li>Read the serialised biome palette into the section</li>
     *   <li>Reconstruct light {@link DataLayer} objects</li>
     *   <li>Call Voxy's {@code VoxelIngestService.rawIngest()} via reflection</li>
     * </ol>
     */
    @SuppressWarnings("unchecked")
    private static void handlePayload(IngestPayload payload) {
        ClientLevel level = Minecraft.getInstance().level;
        if (level == null) return;

        if (!bridgeInitialized) initVoxyBridge();
        if (rawIngestMethod == null) return; // Voxy not available

        for (IngestPayload.SectionData sd : payload.sections()) {
            io.netty.buffer.ByteBuf statesRaw =
                    io.netty.buffer.Unpooled.wrappedBuffer(sd.states());
            io.netty.buffer.ByteBuf biomesRaw =
                    io.netty.buffer.Unpooled.wrappedBuffer(sd.biomes());
            try {
                // 1. Create empty section with correct registry backing
                PalettedContainerFactory factory =
                        PalettedContainerFactory.create(level.registryAccess());
                LevelChunkSection section = new LevelChunkSection(factory);

                // 2. Deserialise block-state palette
                RegistryFriendlyByteBuf statesBuf = new RegistryFriendlyByteBuf(
                        new FriendlyByteBuf(statesRaw), level.registryAccess());
                ((PalettedContainer<BlockState>) section.getStates()).read(statesBuf);

                // 3. Deserialise biome palette
                RegistryFriendlyByteBuf biomesBuf = new RegistryFriendlyByteBuf(
                        new FriendlyByteBuf(biomesRaw), level.registryAccess());
                ((PalettedContainer<Holder<Biome>>) section.getBiomes()).read(biomesBuf);

                // 4. Reconstruct light layers
                DataLayer bl = sd.blockLight() != null
                        ? new DataLayer(sd.blockLight()) : null;
                DataLayer sl = sd.skyLight() != null
                        ? new DataLayer(sd.skyLight()) : null;

                // 5. Feed to Voxy's ingestion pipeline
                invokeRawIngest(level, section,
                        payload.pos().x, sd.y(), payload.pos().z, bl, sl);

                sectionsIngested++;
                if (sectionsIngested % 5000 == 0) {
                    LOGGER.info("[DataHarvester] Client: {} sections ingested into Voxy",
                            sectionsIngested);
                }

            } catch (Exception e) {
                LOGGER.error("[DataHarvester] Section ({}, {}, {}) ingest failed",
                        payload.pos().x, sd.y(), payload.pos().z, e);
            } finally {
                statesRaw.release();
                biomesRaw.release();
            }
        }
    }

    // ==================== Voxy reflection bridge ====================

    /**
     * One-time reflection setup to find Voxy's ingest API.
     *
     * <p>Mirrors VoxyWorldGen v2's {@code VoxyIntegration} approach:
     * <ul>
     *   <li>{@code WorldIdentifier.of(Level)} — get dimension identity</li>
     *   <li>{@code VoxelIngestService.rawIngest(WorldIdentifier, LevelChunkSection,
     *       cx, cy, cz, DataLayer, DataLayer)} — static ingest entry point</li>
     * </ul>
     */
    private static void initVoxyBridge() {
        bridgeInitialized = true;
        try {
            Class<?> ingestClass = Class.forName(
                    "me.cortex.voxy.common.world.service.VoxelIngestService");
            Class<?> worldIdClass = Class.forName(
                    "me.cortex.voxy.commonImpl.WorldIdentifier");

            MethodHandles.Lookup lookup = MethodHandles.lookup();

            // VoxelIngestService.rawIngest(WorldIdentifier, LevelChunkSection,
            //                              int cx, int cy, int cz,
            //                              DataLayer blockLight, DataLayer skyLight)
            Method raw = ingestClass.getMethod("rawIngest",
                    worldIdClass,
                    LevelChunkSection.class,
                    int.class, int.class, int.class,
                    DataLayer.class, DataLayer.class);
            rawIngestMethod = lookup.unreflect(raw);

            // WorldIdentifier.of(Level)
            Method of = worldIdClass.getMethod("of", Level.class);
            worldIdOfMethod = lookup.unreflect(of);

            LOGGER.info("[DataHarvester] Voxy ingest bridge initialised (rawIngest found).");

        } catch (ClassNotFoundException e) {
            LOGGER.warn("[DataHarvester] Voxy not found — ingest bridge disabled. "
                    + "Install Voxy to enable /ingestall client-side ingestion.");
        } catch (Exception e) {
            LOGGER.error("[DataHarvester] Failed to initialise Voxy bridge", e);
        }
    }

    /** Call Voxy's rawIngest via reflection. */
    private static void invokeRawIngest(Level level, LevelChunkSection section,
                                        int cx, int cy, int cz,
                                        DataLayer blockLight, DataLayer skyLight) {
        try {
            Object worldId = worldIdOfMethod.invoke(level);
            if (worldId == null) return;
            rawIngestMethod.invoke(worldId, section, cx, cy, cz, blockLight, skyLight);
        } catch (Throwable e) {
            LOGGER.error("[DataHarvester] rawIngest call failed for ({}, {}, {})",
                    cx, cy, cz, e);
        }
    }
}
