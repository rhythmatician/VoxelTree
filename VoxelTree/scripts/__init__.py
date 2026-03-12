"""VoxelTree.scripts — Data-processing utility modules.

Importable as a proper package; key symbols are re-exported here.

Note: VoxyReader (requires rocksdict) is not re-exported here to avoid
importing a heavy optional dependency at package load time. Import it
directly: ``from VoxelTree.scripts.voxy_reader import VoxyReader``
"""

from VoxelTree.scripts.build_octree_pairs import (
    build_section_index,
    child_coords_from_parent,
    extract_octant_and_upsample,
    parent_coords_and_octant,
)

__all__ = [
    "build_section_index",
    "child_coords_from_parent",
    "extract_octant_and_upsample",
    "parent_coords_and_octant",
]
