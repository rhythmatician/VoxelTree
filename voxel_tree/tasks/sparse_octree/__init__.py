"""Sparse octree / Voxy-native model utilities.

This package contains tools for training and exporting per-level Voxy models
used in the LODiffusion pipeline.
"""

from .sparse_octree_targets import (
    SparseOctreeLevel,
    SparseOctreeNode,
    build_sparse_octree_targets,
    child_occupancy_mask,
    iter_sparse_octree_nodes,
)
from .voxy_models import (
    BIOME_SHAPES,
    HEIGHTMAP_SHAPES,
    LEVEL_MODEL_CLASSES,
    NOISE_SHAPES,
    VoxyL0Model,
    VoxyL1Model,
    VoxyL2Model,
    VoxyL3Model,
    VoxyL4Model,
    VoxyModelConfig,
    create_model,
)

__all__ = [
    # Sparse octree targets (still used for training data)
    "SparseOctreeLevel",
    "SparseOctreeNode",
    "build_sparse_octree_targets",
    "child_occupancy_mask",
    "iter_sparse_octree_nodes",
    # Per-level Voxy models
    "VoxyModelConfig",
    "VoxyL0Model",
    "VoxyL1Model",
    "VoxyL2Model",
    "VoxyL3Model",
    "VoxyL4Model",
    "create_model",
    "LEVEL_MODEL_CLASSES",
    "NOISE_SHAPES",
    "BIOME_SHAPES",
    "HEIGHTMAP_SHAPES",
]
