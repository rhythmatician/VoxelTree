"""VoxelTree.tasks — Data-processing utility modules.

Importable as a proper package; key symbols are re-exported here.

Note: VoxyReader (requires rocksdict) is not re-exported here to avoid
importing a heavy optional dependency at package load time. Import it
directly: ``from voxel_tree.tasks.voxy_reader import VoxyReader``
"""

from voxel_tree.tasks.octree.build_pairs import (
    build_section_index,
    child_coords_from_parent,
    extract_octant_and_upsample,
    parent_coords_and_octant,
)
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
    "build_section_index",
    "build_sparse_octree_targets",
    "child_coords_from_parent",
    "child_occupancy_mask",
    "extract_octant_and_upsample",
    "iter_sparse_octree_nodes",
    "parent_coords_and_octant",
]
