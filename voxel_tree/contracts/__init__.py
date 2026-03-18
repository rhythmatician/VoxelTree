"""voxel_tree.contracts — Model I/O contract definitions with revision tracking.

Every trainable model declares a ``ModelContract`` that pins down its exact
input and output tensor shapes, names, and semantics.  When the I/O changes,
you bump the **revision** and update the affected pipeline stages in one place.

Quick start
-----------
>>> from voxel_tree.contracts import CONTRACTS, get_contract
>>> c = get_contract("density", revision=2)
>>> c.inputs
[TensorSpec(name='climate_input', shape=('batch', 6), dtype='float32', ...)]

See Also
--------
- ``voxel_tree.contracts.registry``  — the ``CONTRACTS`` dict and lookup helpers
- ``voxel_tree.contracts.spec``      — ``TensorSpec``, ``ModelContract``, validation
- ``voxel_tree.contracts.catalog``   — all contract revisions, one per model family
"""

from voxel_tree.contracts.spec import ModelContract, TensorSpec
from voxel_tree.contracts.registry import (
    AlignmentIssue,
    CONTRACTS,
    check_track_alignment,
    get_contract,
    latest_revision,
    list_models,
    register,
    validate_checkpoint_contract,
)

__all__ = [
    "AlignmentIssue",
    "CONTRACTS",
    "ModelContract",
    "TensorSpec",
    "check_track_alignment",
    "get_contract",
    "latest_revision",
    "list_models",
    "register",
    "validate_checkpoint_contract",
]
