package com.voxeltree.harvester;

import com.voxeltree.harvester.noise.NoiseDumperCommand;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Server-side (and shared) mod initializer for Data Harvester.
 *
 * <p>Registers the {@code /dumpnoise} server command for extracting
 * vanilla noise signals to JSON files.  This command runs on both
 * integrated and dedicated servers.
 *
 * <p>The client-side auto-connect logic lives in
 * {@link DataHarvesterClient} and is registered separately.
 */
public class DataHarvesterMod implements ModInitializer {

    private static final Logger LOGGER = LoggerFactory.getLogger("dataharvester");

    @Override
    public void onInitialize() {
        LOGGER.info("[DataHarvester] Registering server-side commands...");

        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
            NoiseDumperCommand.register(dispatcher);
            LOGGER.info("[DataHarvester] /dumpnoise command registered.");
        });
    }
}
