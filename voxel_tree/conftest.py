from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path when running tests.
# This allows imports like `import voxel_tree` to work regardless of where
# pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# If voxel_tree is installed as a namespace package that points to the outer
# folder, ensure the real package directory is included so imports work.
import voxel_tree as _VT  # noqa: E402

if hasattr(_VT, "__path__"):
    pkg_path = str(Path(__file__).resolve().parent)
    if pkg_path not in _VT.__path__:
        _VT.__path__ = list(_VT.__path__) + [pkg_path]
