package com.voxeltree.harvester;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.ConnectScreen;
import net.minecraft.client.gui.screens.DisconnectedScreen;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.multiplayer.ServerData;
import net.minecraft.client.multiplayer.resolver.ServerAddress;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles automatic connection and reconnection to the target server.
 *
 * <p>Registered as a {@link net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents}
 * handler. On each client tick it checks the current screen state and decides
 * whether to initiate or retry a server connection.
 */
public class AutoConnectHandler {

    private static final Logger LOGGER = LoggerFactory.getLogger("dataharvester");

    private final HarvesterConfig config;

    /** Have we initiated the first connection attempt? */
    private boolean firstConnectTriggered = false;

    /** Tick counter for delayed connection (gives the title screen time to load). */
    private int titleScreenTickCount = 0;

    /** Whether we are currently connected / connecting. */
    private boolean connecting = false;

    /** Tick counter for reconnect delay after a disconnect. */
    private int reconnectTickCounter = 0;

    /** Whether we have ever successfully connected at least once. */
    private boolean hasConnectedOnce = false;

    public AutoConnectHandler(HarvesterConfig config) {
        this.config = config;
    }

    /**
     * Called every client tick via the Fabric event bus.
     */
    public void onClientTick(Minecraft client) {
        if (client.screen == null) {
            // In-game (no screen overlay) — we're connected. Reset state.
            if (!hasConnectedOnce) {
                hasConnectedOnce = true;
                LOGGER.info("[DataHarvester] Successfully connected to {}!", config.serverAddress);
            }
            connecting = false;
            reconnectTickCounter = 0;
            return;
        }

        // ── First auto-connect from the title screen ──────────────────────
        if (!firstConnectTriggered && client.screen instanceof TitleScreen) {
            titleScreenTickCount++;
            int delayTicks = config.autoConnectDelaySec * 20; // 20 ticks/sec
            if (titleScreenTickCount >= delayTicks) {
                LOGGER.info("[DataHarvester] Title screen ready. Connecting to {}...",
                        config.serverAddress);
                connect(client);
                firstConnectTriggered = true;
            }
            return;
        }

        // ── Auto-reconnect on disconnect ──────────────────────────────────
        if (config.reconnectOnDisconnect
                && firstConnectTriggered
                && !connecting
                && client.screen instanceof DisconnectedScreen) {

            reconnectTickCounter++;
            int delayTicks = config.reconnectDelaySec * 20;
            if (reconnectTickCounter >= delayTicks) {
                LOGGER.info("[DataHarvester] Disconnected. Reconnecting to {}...",
                        config.serverAddress);
                reconnectTickCounter = 0;
                // Return to title screen first, then connect
                client.disconnect();
                connect(client);
            } else if (reconnectTickCounter == 1) {
                LOGGER.info("[DataHarvester] Disconnected. Will reconnect in {}s.",
                        config.reconnectDelaySec);
            }
        }
    }

    private void connect(Minecraft client) {
        connecting = true;
        ServerAddress address = ServerAddress.parseString(config.serverAddress);
        ServerData serverData = new ServerData(
                "DataHarvester Target",
                config.serverAddress,
                ServerData.Type.OTHER
        );
        ConnectScreen.startConnecting(
                new TitleScreen(),  // parent screen (returned to on cancel)
                client,
                address,
                serverData,
                false,              // isQuickPlay
                null                // transferState
        );
    }
}
