"""Canonical biome name → integer mapping for MC 1.21 overworld.

This mapping is shared between:
  - ``scripts/add_column_heights.py``  (training data preparation)
  - Java ``AnchorSampler`` / ``WorldNoiseAccess`` (inference)

Biomes are sorted alphabetically by their full registry key
(e.g. ``minecraft:badlands``).  Index 0–53 are assigned to the 54
overworld biomes.  Index 255 is reserved for unknown/unmapped biomes.
Cave biomes (deep_dark, dripstone_caves, lush_caves) are included since
they can appear at surface level in some world-gen configurations.

The model's ``biome_vocab_size = 256`` accommodates this range easily.
"""

from __future__ import annotations

# Alphabetically sorted overworld biomes (MC 1.21.1)
OVERWORLD_BIOMES: list[str] = [
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
]

UNKNOWN_BIOME_ID = 255

# name → canonical int
BIOME_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(OVERWORLD_BIOMES)}

# int → name (for debugging)
BIOME_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(OVERWORLD_BIOMES)}
BIOME_ID_TO_NAME[UNKNOWN_BIOME_ID] = "unknown"


def biome_name_to_id(name: str) -> int:
    """Map a biome registry key name to its canonical integer ID.

    Returns ``UNKNOWN_BIOME_ID`` (255) for unrecognised biomes.
    """
    return BIOME_NAME_TO_ID.get(name, UNKNOWN_BIOME_ID)
