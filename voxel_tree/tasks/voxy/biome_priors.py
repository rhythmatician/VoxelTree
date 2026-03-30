"""Biome surface-rule priors for the sparse octree model.

Provides a deterministic mapping from canonical biome ID (0–53) to a
*SurfaceType* category (8 classes).  This mirrors the ``surfaceTypeForBiome()``
switch in ``HeightmapFallbackGenerator.java`` exactly, ensuring the Python
training pipeline and the Java runtime share the same prior knowledge.

The mapping is injected into ``_NoiseEncoder`` as a **non-persistent buffer**
(biome_id → surface_type lookup table) plus a **learnable** ``nn.Embedding``
for each surface type.  The lookup is deterministic and frozen; the embedding
is learned — so biomes sharing a surface type share gradients, providing
strong inductive bias without eliminating flexibility.

Surface types encode the biome-dependent block material at the terrain
surface (top 3 solid blocks).  The depth-dependent block selection logic
(top vs filler vs underground) is not modelled here — the model already
has Y-position awareness from ``noise_3d`` and ``heightmap5``.
"""

from __future__ import annotations

import enum

import torch

# ────────────────────────────────────────────────────────────────────────
# Surface-type categories  (must match HeightmapFallbackGenerator.SurfaceType)
# ────────────────────────────────────────────────────────────────────────


class SurfaceType(enum.IntEnum):
    """Block-material category for the top 3 solid blocks of a biome.

    Ordinals match ``HeightmapFallbackGenerator.SurfaceType`` in Java.
    """

    GRASS = 0  # grass_block / dirt
    SAND = 1  # sand (deserts, beaches, warm oceans)
    RED_SAND = 2  # red_sand (badlands)
    GRAVEL = 3  # gravel (cold/regular oceans, stony biomes)
    STONE = 4  # stone (windswept hills)
    SNOW = 5  # snow_layer / snowy_grass_block / dirt
    PODZOL = 6  # podzol / dirt
    MYCELIUM = 7  # mycelium / dirt


NUM_SURFACE_TYPES: int = len(SurfaceType)  # 8

# ────────────────────────────────────────────────────────────────────────
# Biome → SurfaceType mapping  (indices from biome_mapping.py / BiomeMapping.java)
# ────────────────────────────────────────────────────────────────────────
#
# Canonical biome indices (alphabetical):
#   0  badlands              →  RED_SAND
#   1  bamboo_jungle         →  GRASS
#   2  beach                 →  SAND
#   3  birch_forest          →  GRASS
#   4  cherry_grove          →  GRASS
#   5  cold_ocean            →  GRAVEL
#   6  dark_forest           →  GRASS
#   7  deep_cold_ocean       →  GRAVEL
#   8  deep_dark             →  GRASS
#   9  deep_frozen_ocean     →  GRAVEL
#  10  deep_lukewarm_ocean   →  SAND
#  11  deep_ocean            →  GRAVEL
#  12  desert                →  SAND
#  13  dripstone_caves       →  GRASS
#  14  eroded_badlands       →  RED_SAND
#  15  flower_forest         →  GRASS
#  16  forest                →  GRASS
#  17  frozen_ocean          →  GRAVEL
#  18  frozen_peaks          →  SNOW
#  19  frozen_river          →  SNOW
#  20  grove                 →  SNOW
#  21  ice_spikes            →  SNOW
#  22  jagged_peaks          →  SNOW
#  23  jungle                →  GRASS
#  24  lukewarm_ocean        →  SAND
#  25  lush_caves            →  GRASS
#  26  mangrove_swamp        →  GRASS
#  27  meadow                →  GRASS
#  28  mushroom_fields       →  MYCELIUM
#  29  ocean                 →  GRAVEL
#  30  old_growth_birch_forest → GRASS
#  31  old_growth_pine_taiga →  PODZOL
#  32  old_growth_spruce_taiga → PODZOL
#  33  pale_garden           →  GRASS
#  34  plains                →  GRASS
#  35  river                 →  GRASS
#  36  savanna               →  GRASS
#  37  savanna_plateau       →  GRASS
#  38  snowy_beach           →  SAND
#  39  snowy_plains          →  SNOW
#  40  snowy_slopes          →  SNOW
#  41  snowy_taiga           →  SNOW
#  42  sparse_jungle         →  GRASS
#  43  stony_peaks           →  GRAVEL
#  44  stony_shore           →  GRAVEL
#  45  sunflower_plains      →  GRASS
#  46  swamp                 →  GRASS
#  47  taiga                 →  GRASS
#  48  warm_ocean            →  SAND
#  49  windswept_forest      →  GRASS
#  50  windswept_gravelly_hills → GRAVEL
#  51  windswept_hills       →  STONE
#  52  windswept_savanna     →  GRASS
#  53  wooded_badlands       →  RED_SAND
# 255  unknown               →  GRASS (safe default)

_BIOME_SURFACE_TYPE: dict[int, SurfaceType] = {
    # SAND — deserts, beaches, warm/lukewarm ocean floors
    2: SurfaceType.SAND,
    10: SurfaceType.SAND,
    12: SurfaceType.SAND,
    24: SurfaceType.SAND,
    38: SurfaceType.SAND,
    48: SurfaceType.SAND,
    # RED_SAND — badlands variants
    0: SurfaceType.RED_SAND,
    14: SurfaceType.RED_SAND,
    53: SurfaceType.RED_SAND,
    # GRAVEL — cold/regular ocean floors, stony shores, gravelly hills
    5: SurfaceType.GRAVEL,
    7: SurfaceType.GRAVEL,
    9: SurfaceType.GRAVEL,
    11: SurfaceType.GRAVEL,
    17: SurfaceType.GRAVEL,
    29: SurfaceType.GRAVEL,
    43: SurfaceType.GRAVEL,
    44: SurfaceType.GRAVEL,
    50: SurfaceType.GRAVEL,
    # STONE — rocky windswept hills
    51: SurfaceType.STONE,
    # SNOW — frozen and snowy biomes
    18: SurfaceType.SNOW,
    19: SurfaceType.SNOW,
    20: SurfaceType.SNOW,
    21: SurfaceType.SNOW,
    22: SurfaceType.SNOW,
    39: SurfaceType.SNOW,
    40: SurfaceType.SNOW,
    41: SurfaceType.SNOW,
    # PODZOL — old-growth taigas
    31: SurfaceType.PODZOL,
    32: SurfaceType.PODZOL,
    # MYCELIUM — mushroom fields
    28: SurfaceType.MYCELIUM,
    # Everything else defaults to GRASS
}

# Total number of overworld biomes in the canonical mapping
_NUM_OVERWORLD_BIOMES = 54


def biome_to_surface_type_table(vocab_size: int = 256) -> torch.LongTensor:
    """Build a (vocab_size,) int64 lookup: biome_id → SurfaceType ordinal.

    Unmapped biome IDs (including 255 = unknown) default to GRASS (0).
    """
    table = torch.full((vocab_size,), SurfaceType.GRASS, dtype=torch.long)
    for biome_id, surface_type in _BIOME_SURFACE_TYPE.items():
        if biome_id < vocab_size:
            table[biome_id] = int(surface_type)
    return table


__all__ = [
    "SurfaceType",
    "NUM_SURFACE_TYPES",
    "biome_to_surface_type_table",
]
