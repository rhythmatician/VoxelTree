"""Node registry for the GUI pipeline.

This module exposes the available pipeline nodes and their dependency graph.
The GUI can import this module to discover available nodes and auto-wire
connections based on declared prerequisites.
"""

from .distill_node import DistillNode
from .pregen_node import PregenNode
from .sparse_root_train_node import SparseRootTrainNode

# A registry of node name -> node class.
AVAILABLE_NODES = {
    "pregen": PregenNode,
    "distill": DistillNode,
    "sparse_root_train": SparseRootTrainNode,
}

# A lightweight node graph definition used by the GUI pipeline runner.
# Each entry describes a node and the nodes it depends on.
NODE_GRAPH = [
    {
        "name": "pregen",
        "prerequisites": [],
        "description": "Chunky pregeneration with real-time progress tracking via Chunky's percentage output.",
    },
    {
        "name": "build_octree_pairs",
        "prerequisites": [],
        "description": "Build training data caches for the octree models.",
    },
    {
        "name": "sparse_root_train",
        "prerequisites": ["build_octree_pairs"],
        "description": "Train the sparse-root octree model using the sparse_root_pairs cache.",
    },
    {
        "name": "distill",
        "prerequisites": ["build_octree_pairs"],
        "description": "Distill a density model (existing node).",
    },
]
