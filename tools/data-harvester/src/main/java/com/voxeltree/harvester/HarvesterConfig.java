package com.voxeltree.harvester;

/**
 * Configuration POJO — deserialized from {@code config/dataharvester.json}.
 */
public class HarvesterConfig {

    /** Server address in {@code host:port} format. */
    public String serverAddress = "localhost:25565";

    /** Whether to auto-connect on game start. */
    public boolean autoConnect = true;

    /** Seconds to wait after the title screen appears before connecting. */
    public int autoConnectDelaySec = 5;

    /** Whether to auto-reconnect if disconnected. */
    public boolean reconnectOnDisconnect = true;

    /** Seconds to wait before attempting reconnection. */
    public int reconnectDelaySec = 10;
}
