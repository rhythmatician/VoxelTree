package com.voxeltree.harvester;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Data Harvester — headless-friendly Fabric client mod for automated
 * Voxy LOD data extraction.
 *
 * <p>On game start, auto-connects to a configured server so that
 * VoxyWorldGen v2 can generate chunks and stream LOD data into the
 * client-side Voxy database.  No manual player interaction required.
 *
 * <h3>How it works</h3>
 * <ol>
 *   <li>Reads {@code config/dataharvester.json} for the target server.</li>
 *   <li>After the title screen loads, automatically initiates a connection.</li>
 *   <li>The player spawns in spectator mode (set in server.properties).</li>
 *   <li>VoxyWorldGen detects the player and begins generating chunks
 *       around it, ingesting them into the client's Voxy database.</li>
 *   <li>An external script teleports the player via RCON to cover the
 *       desired area.</li>
 * </ol>
 *
 * <h3>Configuration</h3>
 * Place {@code config/dataharvester.json} in the Minecraft profile directory:
 * <pre>{@code
 * {
 *   "serverAddress": "localhost:25565",
 *   "autoConnect": true,
 *   "autoConnectDelaySec": 5,
 *   "reconnectOnDisconnect": true,
 *   "reconnectDelaySec": 10
 * }
 * }</pre>
 */
public class DataHarvesterClient implements ClientModInitializer {

    public static final String MOD_ID = "dataharvester";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private static HarvesterConfig config;
    private static AutoConnectHandler connectHandler;

    @Override
    public void onInitializeClient() {
        LOGGER.info("[DataHarvester] Initializing...");
        config = loadConfig();
        LOGGER.info("[DataHarvester] Config: server={}, autoConnect={}, delay={}s",
                config.serverAddress, config.autoConnect, config.autoConnectDelaySec);

        if (!config.autoConnect) {
            LOGGER.info("[DataHarvester] Auto-connect disabled in config. Standing by.");
            return;
        }

        connectHandler = new AutoConnectHandler(config);
        ClientTickEvents.END_CLIENT_TICK.register(connectHandler::onClientTick);

        LOGGER.info("[DataHarvester] Auto-connect armed. Will connect to {} in {}s after title screen.",
                config.serverAddress, config.autoConnectDelaySec);
    }

    public static HarvesterConfig getConfig() {
        return config;
    }

    // ----- config I/O -----

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static HarvesterConfig loadConfig() {
        Path configPath = Path.of("config", "dataharvester.json");
        if (Files.exists(configPath)) {
            try {
                String json = Files.readString(configPath);
                HarvesterConfig cfg = GSON.fromJson(json, HarvesterConfig.class);
                if (cfg != null) return cfg;
            } catch (IOException e) {
                LOGGER.warn("[DataHarvester] Failed to read config, using defaults.", e);
            }
        }
        // Write defaults so the user can edit
        HarvesterConfig defaults = new HarvesterConfig();
        try {
            Files.createDirectories(configPath.getParent());
            Files.writeString(configPath, GSON.toJson(defaults));
            LOGGER.info("[DataHarvester] Wrote default config to {}", configPath);
        } catch (IOException e) {
            LOGGER.warn("[DataHarvester] Could not write default config.", e);
        }
        return defaults;
    }
}
