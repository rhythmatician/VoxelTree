"""
Configuration for Progressive LOD Models

This module holds the shared ``SimpleFlexibleConfig`` dataclass used by
all four progressive LOD models (Init→LOD4, LOD4→3, LOD3→2, LOD2→1).
"""

from dataclasses import dataclass


@dataclass
class SimpleFlexibleConfig:
    """Configuration for progressive multi-LOD models with anchor conditioning."""

    # Core architecture
    base_channels: int = 32

    # Anchor conditioning channels
    height_channels: int = 5  # surface, ocean_floor, slope_x, slope_z, curvature
    y_embed_dim: int = 16  # Y-slab embedding dimension

    # Input conditioning
    biome_vocab_size: int = 256
    biome_embed_dim: int = 32

    # Output
    block_vocab_size: int = 1104
