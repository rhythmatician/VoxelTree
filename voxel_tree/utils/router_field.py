"""Canonical RouterField definition — Python mirror of Java's RouterField enum.

This file is the **single source of truth** for the 15 NoiseRouter field
definitions used throughout the VoxelTree training pipeline.  It must stay
in exact sync with the Java enum:

    ``com.rhythmatician.lodiffusion.world.noise.RouterField``

Any change to field names, ordinals, or COUNT must be reflected in both files.

Usage
-----
>>> from voxel_tree.utils.router_field import RouterField, CLIMATE_FIELDS
>>> RouterField.TEMPERATURE.index
0
>>> RouterField.by_name("final_density")
<RouterField.FINAL_DENSITY: 7>
>>> [RouterField.by_index(i).name for i in CLIMATE_FIELDS]
['TEMPERATURE', 'VEGETATION', 'CONTINENTS', 'EROSION', 'DEPTH', 'RIDGES']
"""

from __future__ import annotations

import enum
from typing import FrozenSet, List, Sequence


class RouterField(enum.Enum):
    """The 15 vanilla NoiseRouter fields sampled at 4×4×4 quart resolution.

    Ordinals 0–14 match the Java ``RouterField`` enum exactly.
    Each member's value is its index (ordinal).
    """

    TEMPERATURE = 0
    VEGETATION = 1
    CONTINENTS = 2
    EROSION = 3
    DEPTH = 4
    RIDGES = 5
    PRELIMINARY_SURFACE_LEVEL = 6
    FINAL_DENSITY = 7
    BARRIER = 8
    FLUID_LEVEL_FLOODEDNESS = 9
    FLUID_LEVEL_SPREAD = 10
    LAVA = 11
    VEIN_TOGGLE = 12
    VEIN_RIDGED = 13
    VEIN_GAP = 14

    # -- convenience properties --

    @property
    def index(self) -> int:
        """Zero-based channel index (same as .value)."""
        return self.value

    @property
    def lower_name(self) -> str:
        """Lowercase name matching JSON keys (e.g. ``'temperature'``)."""
        return self.name.lower()

    # -- lookup helpers --

    @classmethod
    def by_index(cls, idx: int) -> "RouterField":
        """Look up a field by its ordinal index."""
        return _BY_INDEX[idx]

    @classmethod
    def by_name(cls, name: str) -> "RouterField":
        """Look up a field by its lowercase name (case-insensitive)."""
        return cls[name.upper()]

    @classmethod
    def names(cls) -> List[str]:
        """Return all 15 lowercase field names in index order."""
        return [f.lower_name for f in cls]

    @classmethod
    def indices(cls) -> List[int]:
        """Return ``[0, 1, ..., 14]``."""
        return list(range(COUNT))


# Total number of RouterField channels.
COUNT: int = len(RouterField)
assert COUNT == 15, f"Expected 15 RouterField channels, got {COUNT}"

# Quick index→field lookup (avoid linear scan on every call).
_BY_INDEX: dict[int, RouterField] = {f.value: f for f in RouterField}

# ── Semantic field groups ──────────────────────────────────────────────────
# These mirror the Java-side ``RouterField.isClimate()`` etc. helpers.

CLIMATE_FIELDS: FrozenSet[int] = frozenset({
    RouterField.TEMPERATURE.index,
    RouterField.VEGETATION.index,
    RouterField.CONTINENTS.index,
    RouterField.EROSION.index,
    RouterField.DEPTH.index,
    RouterField.RIDGES.index,
})
"""Indices of the 6 climate/shape fields used by the biome classifier and
density predictor.  These are exactly the inputs to vanilla's
``MultiNoiseBiomeSource.getBiome()``."""

DENSITY_FIELDS: FrozenSet[int] = frozenset({
    RouterField.PRELIMINARY_SURFACE_LEVEL.index,
    RouterField.FINAL_DENSITY.index,
})
"""Indices of the 2 density-related fields (expensive to compute)."""

AQUIFER_FIELDS: FrozenSet[int] = frozenset({
    RouterField.BARRIER.index,
    RouterField.FLUID_LEVEL_FLOODEDNESS.index,
    RouterField.FLUID_LEVEL_SPREAD.index,
    RouterField.LAVA.index,
})
"""Indices of the 4 aquifer noise fields."""

ORE_FIELDS: FrozenSet[int] = frozenset({
    RouterField.VEIN_TOGGLE.index,
    RouterField.VEIN_RIDGED.index,
    RouterField.VEIN_GAP.index,
})
"""Indices of the 3 ore-vein noise fields."""

# ── Ordered lists for slicing convenience ──────────────────────────────────

CLIMATE_INDICES: List[int] = sorted(CLIMATE_FIELDS)
"""Climate field indices in ascending order: [0, 1, 2, 3, 4, 5]."""

DENSITY_INDICES: List[int] = sorted(DENSITY_FIELDS)
"""Density field indices in ascending order: [6, 7]."""

AQUIFER_INDICES: List[int] = sorted(AQUIFER_FIELDS)
"""Aquifer field indices in ascending order: [8, 9, 10, 11]."""

ORE_INDICES: List[int] = sorted(ORE_FIELDS)
"""Ore vein field indices in ascending order: [12, 13, 14]."""

# ── NoiseRouter accessor names (for Java DensityFunction binding) ──────────

ROUTER_ACCESSOR_NAMES: List[str] = [
    "temperature",
    "vegetation",
    "continents",
    "erosion",
    "depth",
    "ridges",
    "initialDensityWithoutJaggedness",
    "finalDensity",
    "barrierNoise",
    "fluidLevelFloodednessNoise",
    "fluidLevelSpreadNoise",
    "lavaNoise",
    "veinToggle",
    "veinRidged",
    "veinGap",
]
"""Java ``NoiseRouter`` accessor method names, in RouterField index order.

Used by the data-harvester's ``sampleRouterFieldsForSection()`` to bind
each field index to its vanilla ``DensityFunction`` accessor."""

assert len(ROUTER_ACCESSOR_NAMES) == COUNT


def extract_climate(all_fields: "Sequence[float]") -> "List[float]":
    """Extract the 6 climate values from a flat 15-element field array."""
    return [all_fields[i] for i in CLIMATE_INDICES]


__all__ = [
    "RouterField",
    "COUNT",
    "CLIMATE_FIELDS",
    "CLIMATE_INDICES",
    "DENSITY_FIELDS",
    "DENSITY_INDICES",
    "AQUIFER_FIELDS",
    "AQUIFER_INDICES",
    "ORE_FIELDS",
    "ORE_INDICES",
    "ROUTER_ACCESSOR_NAMES",
    "extract_climate",
]
