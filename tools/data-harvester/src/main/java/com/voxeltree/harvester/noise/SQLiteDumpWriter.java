package com.voxeltree.harvester.noise;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes v7 noise-dump data directly into a SQLite database instead of
 * millions of individual JSON files.
 *
 * <p>Schema is 100% compatible with {@code consolidate_dumps.py} and
 * {@code build_sparse_octree_pairs.py --db}:
 * <pre>{@code
 * sections(chunk_x, section_y, chunk_z,
 *          noise_data BLOB,   -- float32[480] = 15 fields × 32 quart cells
 *          biome_ids  BLOB)   -- int32[32]    = 4×2×4 biome indices
 *
 * heightmaps(chunk_x, chunk_z,
 *            surface     BLOB, -- int32[256] = 16×16
 *            ocean_floor BLOB) -- int32[256] = 16×16
 *
 * metadata(key TEXT, value TEXT)
 * }</pre>
 *
 * <p>Thread-safety: All public methods are synchronized because SQLite in WAL
 * mode supports concurrent reads but only one writer at a time. The thread
 * pool in {@code executeV7} must funnel writes through a single instance.
 */
public class SQLiteDumpWriter implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(SQLiteDumpWriter.class);

    private static final String CREATE_TABLES = """
        CREATE TABLE IF NOT EXISTS sections (
            chunk_x   INTEGER NOT NULL,
            section_y INTEGER NOT NULL,
            chunk_z   INTEGER NOT NULL,
            noise_data BLOB NOT NULL,
            biome_ids  BLOB NOT NULL,
            PRIMARY KEY (chunk_x, section_y, chunk_z)
        );

        CREATE TABLE IF NOT EXISTS heightmaps (
            chunk_x INTEGER NOT NULL,
            chunk_z INTEGER NOT NULL,
            surface     BLOB NOT NULL,
            ocean_floor BLOB NOT NULL,
            PRIMARY KEY (chunk_x, chunk_z)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """;

    private static final int BATCH_SIZE = 2000;

    private final Connection conn;
    private final PreparedStatement insertSection;
    private final PreparedStatement insertHeightmap;
    private final Set<Long> heightmapSeen = new HashSet<>();

    private int pendingSections = 0;
    private int totalSections = 0;
    private int totalHeightmaps = 0;

    /**
     * Open (or create) the database at the given path.
     *
     * @param dbPath path to the SQLite file (will be created if missing)
     * @throws SQLException if the database cannot be opened
     */
    public SQLiteDumpWriter(Path dbPath) throws SQLException {
        // Ensure parent directory exists (if there is one)
        try {
            Path parent = dbPath.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
        } catch (IOException e) {
            throw new SQLException("Cannot create parent dirs for " + dbPath, e);
        }

        String url = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        this.conn = DriverManager.getConnection(url);

        try (Statement stmt = conn.createStatement()) {
            stmt.execute("PRAGMA journal_mode=WAL");
            stmt.execute("PRAGMA synchronous=NORMAL");
            stmt.execute("PRAGMA cache_size=-512000");   // 512 MB page cache
            stmt.execute("PRAGMA busy_timeout=30000");   // 30s busy wait
            String[] ddl = CREATE_TABLES.split(";");
            for (String sql : ddl) {
                sql = sql.trim();
                if (!sql.isEmpty()) {
                    stmt.execute(sql);
                }
            }
        }

        conn.setAutoCommit(false);

        this.insertSection = conn.prepareStatement(
                "INSERT OR REPLACE INTO sections (chunk_x, section_y, chunk_z, noise_data, biome_ids) "
                        + "VALUES (?, ?, ?, ?, ?)");

        this.insertHeightmap = conn.prepareStatement(
                "INSERT OR IGNORE INTO heightmaps (chunk_x, chunk_z, surface, ocean_floor) "
                        + "VALUES (?, ?, ?, ?)");
    }

    /**
     * Write a single section's data to the database.
     *
     * @param cx         chunk X
     * @param sy         section Y (-4..19)
     * @param cz         chunk Z
     * @param routerFlat float[480] — 15 fields × 32 (4×2×4)
     * @param biomeIds   int[4][2][4] biome registry IDs
     * @param surfaceHm  float[16][16] WORLD_SURFACE_WG heightmap
     * @param oceanFloorHm float[16][16] OCEAN_FLOOR_WG heightmap
     */
    public synchronized void writeSection(
            int cx, int sy, int cz,
            float[] routerFlat,
            int[][][] biomeIds,
            float[][] surfaceHm,
            float[][] oceanFloorHm) throws SQLException {

        // Pack noise_data: float32[480]
        byte[] noiseBlob = packFloats(routerFlat);

        // Pack biome_ids: int32[32] in [qx][qy][qz] order
        int[] biomeFlat = new int[32];
        int idx = 0;
        for (int qx = 0; qx < 4; qx++) {
            for (int qy = 0; qy < 2; qy++) {
                for (int qz = 0; qz < 4; qz++) {
                    biomeFlat[idx++] = biomeIds[qx][qy][qz];
                }
            }
        }
        byte[] biomeBlob = packInts(biomeFlat);

        insertSection.setInt(1, cx);
        insertSection.setInt(2, sy);
        insertSection.setInt(3, cz);
        insertSection.setBytes(4, noiseBlob);
        insertSection.setBytes(5, biomeBlob);
        insertSection.addBatch();
        pendingSections++;
        totalSections++;

        // Heightmap deduplication: one per column
        long colKey = ((long) cx << 32) | (cz & 0xFFFFFFFFL);
        if (heightmapSeen.add(colKey)) {
            byte[] surfaceBlob = packHeightmap(surfaceHm);
            byte[] oceanBlob = packHeightmap(oceanFloorHm);

            insertHeightmap.setInt(1, cx);
            insertHeightmap.setInt(2, cz);
            insertHeightmap.setBytes(3, surfaceBlob);
            insertHeightmap.setBytes(4, oceanBlob);
            insertHeightmap.addBatch();
            totalHeightmaps++;
        }

        if (pendingSections >= BATCH_SIZE) {
            flush();
        }
    }

    /**
     * Flush pending batch inserts to disk.
     */
    public synchronized void flush() throws SQLException {
        insertSection.executeBatch();
        insertHeightmap.executeBatch();
        conn.commit();
        pendingSections = 0;
    }

    /**
     * Write metadata and create indices, then close the connection.
     *
     * @param radius the dump radius used
     */
    public synchronized void finalizeDb(int radius) throws SQLException {
        // Flush any remaining
        flush();

        // Metadata
        try (PreparedStatement ps = conn.prepareStatement(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)")) {
            ps.setString(1, "radius");
            ps.setString(2, String.valueOf(radius));
            ps.execute();

            ps.setString(1, "created");
            ps.setString(2, LocalDateTime.now().format(
                    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            ps.execute();

            ps.setString(1, "sections_count");
            ps.setString(2, String.valueOf(totalSections));
            ps.execute();
        }

        // Create indices
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_section_y ON sections(section_y)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_section_xz ON sections(chunk_x, chunk_z)");
        }
        conn.commit();

        LOG.info("[SQLiteDump] Finalized: {} sections, {} heightmaps",
                totalSections, totalHeightmaps);
    }

    @Override
    public synchronized void close() {
        try {
            insertSection.close();
        } catch (SQLException ignored) {}
        try {
            insertHeightmap.close();
        } catch (SQLException ignored) {}
        try {
            conn.close();
        } catch (SQLException ignored) {}
    }

    // -----------------------------------------------------------------------
    //  Binary packing — little-endian, matching Python struct.pack("<...")
    // -----------------------------------------------------------------------

    private static byte[] packFloats(float[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            buf.putFloat(v);
        }
        return buf.array();
    }

    private static byte[] packInts(int[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            buf.putInt(v);
        }
        return buf.array();
    }

    /**
     * Pack a 16×16 heightmap (stored as float[][] in WorldNoiseAccess) into
     * int32[256] BLOB. The float values are rounded to int (they're actually
     * integer-valued heights).
     */
    private static byte[] packHeightmap(float[][] hm) {
        ByteBuffer buf = ByteBuffer.allocate(256 * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                buf.putInt(Math.round(hm[x][z]));
            }
        }
        return buf.array();
    }

    // -----------------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------------

    public int getTotalSections() { return totalSections; }
    public int getTotalHeightmaps() { return totalHeightmaps; }
}
