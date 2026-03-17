"""Deploy the `leaf` octree model to LODiffusion.

This is a thin wrapper around :mod:`voxel_tree.tasks.deploy` that always
deploys the `leaf` submodel.
"""

from __future__ import annotations

import sys
from typing import List

from .deploy import main as _deploy_main


def main(argv: List[str] | None = None) -> None:
    """Deploy the leaf model."""
    if argv is None:
        argv = sys.argv[1:]

    if "--models" not in argv:
        argv = ["--models", "leaf"] + argv

    _deploy_main(argv)


if __name__ == "__main__":
    main()
