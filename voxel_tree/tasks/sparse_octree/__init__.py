"""Sparse octree utilities.

This package contains tools for training and exporting sparse octree models used in
the LODiffusion pipeline.
"""

from .sparse_octree_targets import (
    SparseOctreeLevel,
    SparseOctreeNode,
    build_sparse_octree_targets,
    child_occupancy_mask,
    iter_sparse_octree_nodes,
)

__all__ = [
    "SparseOctreeLevel",
    "SparseOctreeNode",
    "build_sparse_octree_targets",
    "child_occupancy_mask",
    "iter_sparse_octree_nodes",
]
