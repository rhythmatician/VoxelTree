"""VoxelTree Training Module — Octree Pipeline.

Contains PyTorch dataset and model implementations for the
3-model octree training pipeline.

Exports
-------
OctreeConfig
    Configuration dataclass for the 3-model architecture.
OctreeInitModel
    Root model for L4 WorldSections (no parent context).
OctreeRefineModel
    Shared refinement model for L3/L2/L1 WorldSections.
OctreeLeafModel
    Leaf model for L0 block-level WorldSections.
OctreeDataset
    Dataset for octree training from pair-cache NPZ files.
collate_octree_batch
    Collate function grouping samples by model type.
"""

from train.octree_models import (  # noqa: F401
    OctreeConfig,
    OctreeInitModel,
    OctreeLeafModel,
    OctreeRefineModel,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)
from train.octree_dataset import (  # noqa: F401
    OctreeDataset,
    collate_octree_batch,
)

__all__ = [
    "OctreeConfig",
    "OctreeInitModel",
    "OctreeRefineModel",
    "OctreeLeafModel",
    "create_init_model",
    "create_refine_model",
    "create_leaf_model",
    "OctreeDataset",
    "collate_octree_batch",
]
