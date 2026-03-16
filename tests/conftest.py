from __future__ import annotations

import sys
from pathlib import Path

# Ensure the inner `VoxelTree/VoxelTree` package is on sys.path for tests.
# The repository layout has a top-level `VoxelTree/` directory that is a
# namespace package (no __init__.py), so pytest may not automatically find the
# real package nested at `VoxelTree/VoxelTree`.
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = ROOT / "VoxelTree"
if str(PACKAGE_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PATH))
