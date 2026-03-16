"""Compatibility shim to preserve legacy import paths.

The old script lived at ``scripts/octree/build_octree_pairs.py``.  The code was
refactored/renamed to ``scripts/octree/build_pairs.py``, but some internal tests
and external callers still refer to the old module name.

This module re-exports the public API from the new location.
"""

from __future__ import annotations

from VoxelTree.scripts.octree.build_pairs import *  # noqa: F401,F403
