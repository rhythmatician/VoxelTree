import os
import sys

# Ensure the project root is on sys.path when running tests from this subpackage.
# This allows imports like `import voxel_tree` to work even when pytest is invoked
# with a path to the test file.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
