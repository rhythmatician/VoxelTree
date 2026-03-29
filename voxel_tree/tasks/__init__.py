"""VoxelTree.tasks — Data-processing utility modules.

Importable as a proper package; key symbols are re-exported here.

Note: VoxyReader (requires rocksdict) is not re-exported here to avoid
importing a heavy optional dependency at package load time. Import it
directly: ``from voxel_tree.tasks.voxy_reader import VoxyReader``
"""

from .sparse_octree import (
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
