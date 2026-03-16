"""Compatibility shim for legacy imports.

Historically this module was called ``train_octree.py``. It has since been
renamed to ``train.py`` to reflect the canonical entrypoint for the octree
training pipeline.

This module re-exports the public API of ``VoxelTree.train.train`` so that
existing imports continue to work.
"""

from __future__ import annotations

from VoxelTree.train.train import (
    OctreeLoss,
    _bitmask_to_binary,
    _prepare_targets,
    compute_octree_metrics,
    main,
)  # noqa: F401
