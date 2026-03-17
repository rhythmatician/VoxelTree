from __future__ import annotations

import sys
from pathlib import Path

# Ensure the top-level `VoxelTree/` folder is on sys.path for tests.
# The real Python package lives under `VoxelTree/voxel_tree`, so we patch the
# imported package's __path__ to include that inner directory.
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = ROOT / "voxel_tree"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import voxel_tree as _VT  # noqa: E402

if hasattr(_VT, "__path__"):
    pkg_path = str(PACKAGE_PATH)
    if pkg_path not in _VT.__path__:
        _VT.__path__ = list(_VT.__path__) + [pkg_path]
