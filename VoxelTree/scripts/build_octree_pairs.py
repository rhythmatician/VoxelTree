"""Compatibility shim for legacy imports.

Historically this module lived at ``VoxelTree/scripts/build_octree_pairs.py``.
It has since been moved into the ``scripts/octree`` subpackage, but some
code (including tests) still imports it from the original location.

This shim re-exports the original implementation.
"""

from __future__ import annotations

from VoxelTree.scripts.octree.build_pairs import *  # noqa: F401,F403

__all__ = [
    # core helpers
    "build_section_index",
    "child_coords_from_parent",
    "extract_octant",
    "extract_octant_and_upsample",
    "parent_coords_and_octant",
    # pair building
    "build_pairs_for_level",
    "stack_and_save",
    "stack_and_save_sparse_root",
    "build",
    "main",
]
