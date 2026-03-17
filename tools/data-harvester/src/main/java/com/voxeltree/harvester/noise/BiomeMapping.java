package com.voxeltree.harvester.noise;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import net.minecraft.core.Holder;
import net.minecraft.world.level.biome.Biome;

/**
 * Canonical biome name → integer mapping for MC 1.21 overworld.
 *
 * <p>This mapping is shared between:
 * <ul>
 *   <li>Java inference ({@link WorldNoiseAccess})</li>
 *   <li>Python training pipeline ({@code scripts/biome_mapping.py})</li>
 * </ul>
 *
 * <p>Biomes are sorted alphabetically by their full registry key
 * (e.g. {@code minecraft:badlands}).  Indices 0–53 are assigned to the 54
 * overworld biomes.  Index 255 is reserved for unknown/unmapped biomes.
 *
 * <p><strong>IMPORTANT:</strong> This list must stay in sync with
 * {@code scripts/biome_mapping.py}.  Both files use identical alphabetical
 * ordering.
 */
public final class BiomeMapping {

    /** ID returned for biomes not in the canonical mapping. */
    public static final int UNKNOWN_BIOME_ID = 255;

    /**
     * Alphabetically sorted overworld biomes (MC 1.21.1).
     * Index in this array = canonical biome ID.
     */
    private static final String[] OVERWORLD_BIOMES = {
        "minecraft:badlands",
        "minecraft:bamboo_jungle",
        "minecraft:beach",
        "minecraft:birch_forest",
        "minecraft:cherry_grove",
        "minecraft:cold_ocean",
        "minecraft:dark_forest",
        "minecraft:deep_cold_ocean",
        "minecraft:deep_dark",
        "minecraft:deep_frozen_ocean",
        "minecraft:deep_lukewarm_ocean",
        "minecraft:deep_ocean",
        "minecraft:desert",
        "minecraft:dripstone_caves",
        "minecraft:eroded_badlands",
        "minecraft:flower_forest",
        "minecraft:forest",
        "minecraft:frozen_ocean",
        "minecraft:frozen_peaks",
        "minecraft:frozen_river",
        "minecraft:grove",
        "minecraft:ice_spikes",
        "minecraft:jagged_peaks",
        "minecraft:jungle",
        "minecraft:lukewarm_ocean",
        "minecraft:lush_caves",
        "minecraft:mangrove_swamp",
        "minecraft:meadow",
        "minecraft:mushroom_fields",
        "minecraft:ocean",
        "minecraft:old_growth_birch_forest",
        "minecraft:old_growth_pine_taiga",
        "minecraft:old_growth_spruce_taiga",
        "minecraft:pale_garden",
        "minecraft:plains",
        "minecraft:river",
        "minecraft:savanna",
        "minecraft:savanna_plateau",
        "minecraft:snowy_beach",
        "minecraft:snowy_plains",
        "minecraft:snowy_slopes",
        "minecraft:snowy_taiga",
        "minecraft:sparse_jungle",
        "minecraft:stony_peaks",
        "minecraft:stony_shore",
        "minecraft:sunflower_plains",
        "minecraft:swamp",
        "minecraft:taiga",
        "minecraft:warm_ocean",
        "minecraft:windswept_forest",
        "minecraft:windswept_gravelly_hills",
        "minecraft:windswept_hills",
        "minecraft:windswept_savanna",
        "minecraft:wooded_badlands",
    };

    /** Immutable map: biome name → canonical int ID. */
    private static final Map<String, Integer> NAME_TO_ID;
    static {
        Map<String, Integer> map = new HashMap<>(OVERWORLD_BIOMES.length * 2);
        for (int i = 0; i < OVERWORLD_BIOMES.length; i++) {
            map.put(OVERWORLD_BIOMES[i], i);
        }
        NAME_TO_ID = Collections.unmodifiableMap(map);
    }

    private BiomeMapping() {}

    /**
     * Map a biome's registry key name to its canonical integer ID.
     *
     * @param name full registry key name (e.g. {@code "minecraft:plains"})
     * @return canonical ID (0–53) or {@link #UNKNOWN_BIOME_ID} (255)
     */
    public static int toCanonicalId(String name) {
        return NAME_TO_ID.getOrDefault(name, UNKNOWN_BIOME_ID);
    }

    /**
     * Map a {@link Holder} to its canonical integer ID.
     *
     * @param entry biome holder entry
     * @return canonical ID (0–53) or {@link #UNKNOWN_BIOME_ID} (255)
     */
    public static int toCanonicalId(Holder<Biome> entry) {
        return entry.unwrapKey()
                .map(key -> toCanonicalId(key.identifier().toString()))
                .orElse(UNKNOWN_BIOME_ID);
    }

    /** Number of canonical overworld biomes. */
    public static int size() {
        return OVERWORLD_BIOMES.length;
    }

    /**
     * Get the registry-key name for a canonical biome ID.
     *
     * @param id canonical ID (0–53)
     * @return the biome name (e.g. {@code "minecraft:plains"}), or {@code null} if out of range
     */
    public static String getCanonicalName(int id) {
        if (id < 0 || id >= OVERWORLD_BIOMES.length) return null;
        return OVERWORLD_BIOMES[id];
    }
}
