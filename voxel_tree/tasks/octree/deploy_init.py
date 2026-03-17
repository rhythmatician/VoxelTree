"""Deploy the `init` octree model to LODiffusion.

This is a thin wrapper around :mod:`voxel_tree.tasks.deploy` that always
deploys the `init` submodel.
"""

from __future__ import annotations

import sys
from typing import List

from .octree.deploy import main as _deploy_main


def main(argv: List[str] | None = None) -> None:
    """Deploy the init model.

    Args:
        argv: CLI args (same as :func:`voxel_tree.tasks.deploy.main`).
    """
    if argv is None:
        argv = sys.argv[1:]

    # Ensure the deploy helper only deploys `init`.
    if "--models" not in argv:
        argv = ["--models", "init"] + argv

    _deploy_main(argv)


if __name__ == "__main__":
    main()
