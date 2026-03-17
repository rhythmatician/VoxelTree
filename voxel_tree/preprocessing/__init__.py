"""Compatibility shim for moved preprocessing code.

The real implementation now lives in :mod:`voxel_tree.tasks.preprocessing`.

This file exists so imports like ``from voxel_tree.preprocessing import harvest``
continue to work even though the module was moved.
"""

from __future__ import annotations

import pathlib

# Point the package path to the new location.
_this_dir = pathlib.Path(__file__).resolve().parent
_new_path = _this_dir.parent / "tasks" / "preprocessing"
__path__ = [str(_new_path)]

