package com.voxeltree.harvester;

import com.voxeltree.harvester.ingest.IngestAllCommand;
import com.voxeltree.harvester.ingest.IngestPayload;
import com.voxeltree.harvester.noise.NoiseDumperCommand;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.networking.v1.PayloadTypeRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Server-side (and shared) mod initializer for Data Harvester.
 *
 * <p>Registers server commands and network payload types:
 * <ul>
 *   <li>{@code /dumpnoise} — extract vanilla noise signals to JSON</li>
 *   <li>{@code /ingestall} — send pre-generated chunks to client for
 *       Voxy ingestion (no teleporting required)</li>
 * </ul>
 *
 * <p>The client-side auto-connect and ingest handling lives in
 * {@link DataHarvesterClient} and is registered separately.
 */
public class DataHarvesterMod implements ModInitializer {

    public static final Logger LOGGER = LoggerFactory.getLogger("dataharvester");

    @Override
    public void onInitialize() {
        LOGGER.info("[DataHarvester] Registering server-side commands...");

        // Register S2C payload type (both client and server must know the type)
        PayloadTypeRegistry.playS2C().register(IngestPayload.TYPE, IngestPayload.CODEC);

        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
            NoiseDumperCommand.register(dispatcher);
            LOGGER.info("[DataHarvester] /dumpnoise command registered.");

            IngestAllCommand.register(dispatcher);
            LOGGER.info("[DataHarvester] /ingestall command registered.");
        });
    }
}
