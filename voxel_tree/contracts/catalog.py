"""Contract catalog — all model I/O contract revisions in one place.

Each model family has a section below.  When you change the I/O of a model:

  1. Copy the latest revision block.
  2. Bump ``revision`` by 1.
  3. Update the ``TensorSpec`` entries.
  4. Write a ``changelog`` string explaining what changed.
  5. Update ``build_pairs_fn`` / ``train_fn`` / ``export_fn`` if the entry
     points moved.

The training, export, and build-pairs scripts import ``get_contract(...)``
and use it to:
  - build their sidecar JSON     (``contract.to_sidecar(...)``)
  - embed metadata in checkpoints (``contract.to_checkpoint_meta()``)
  - validate tensor shapes        (``contract.validate_forward(...)``)

Revision numbering
------------------
Revisions are **per model family**, starting at 1.  There is no requirement
that revisions across different models stay in sync.  A "v7 pipeline" model
that already shipped with a sidecar marked ``version: "7.0.0"`` is mapped
to revision 1 here (the first contract-tracked revision).
"""

from __future__ import annotations

from voxel_tree.contracts.spec import ModelContract, TensorSpec

# We import register locally to avoid circular import issues.
# It's fine — this module is loaded by registry._ensure_catalog_loaded().
from voxel_tree.contracts.registry import register as _register


# ══════════════════════════════════════════════════════════════════════════
#  DENSITY
# ══════════════════════════════════════════════════════════════════════════

# ── revision 1 (6 climate → 2 density) ──────────────────────────────────
_register(
    ModelContract(
        model_name="density",
        revision=1,
        contract_id="lodiffusion.v7.density",
        inputs=(
            TensorSpec(
                name="climate_input",
                shape=("batch", 6),
                dtype="float32",
                channels=(
                    "temperature",
                    "vegetation",
                    "continents",
                    "erosion",
                    "depth",
                    "ridges",
                ),
                channel_indices=(0, 1, 2, 3, 4, 5),
                description="6 climate RouterField values per quart cell",
            ),
        ),
        outputs=(
            TensorSpec(
                name="density_output",
                shape=("batch", 2),
                dtype="float32",
                channels=("preliminary_surface_level", "final_density"),
                channel_indices=(6, 7),
                description="PSL + final_density per quart cell",
            ),
        ),
        onnx_opset=18,
        description="Density MLP: 6 climate → (preliminary_surface_level, final_density)",
        changelog="6 raw climate fields → 2 density outputs. "
        "Replaced legacy 12-feature terrain-shaper approach.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.density.train_density:main",
        export_fn="voxel_tree.tasks.density.export_density:main",
    )
)


# ══════════════════════════════════════════════════════════════════════════
#  BIOME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════

_register(
    ModelContract(
        model_name="biome",
        revision=1,
        contract_id="lodiffusion.v7.biome_classifier",
        inputs=(
            TensorSpec(
                name="climate_input",
                shape=("batch", 6),
                dtype="float32",
                channels=(
                    "temperature",
                    "vegetation",
                    "continents",
                    "erosion",
                    "depth",
                    "ridges",
                ),
                channel_indices=(0, 1, 2, 3, 4, 5),
                description="6 climate RouterField values per quart cell",
            ),
        ),
        outputs=(
            TensorSpec(
                name="biome_logits",
                shape=("batch", 54),
                dtype="float32",
                description="54-class biome logits (apply argmax at inference)",
            ),
        ),
        onnx_opset=18,
        description="BiomeClassifier: 6 climate → 54-class biome logits",
        changelog="Initial tracked revision.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.biome.train_biome_classifier:main",
        export_fn="voxel_tree.tasks.biome.export_biome:main",
        extra={"num_classes": 54},
    )
)


# ══════════════════════════════════════════════════════════════════════════
#  HEIGHTMAP PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

_register(
    ModelContract(
        model_name="heightmap",
        revision=1,
        contract_id="lodiffusion.v7.heightmap_predictor",
        inputs=(
            TensorSpec(
                name="climate_grid",
                shape=("batch", 96),
                dtype="float32",
                channels=(
                    "temperature",
                    "vegetation",
                    "continents",
                    "erosion",
                    "depth",
                    "ridges",
                ),
                description="6 climate fields × 4×4 quart grid, flattened to 96",
            ),
        ),
        outputs=(
            TensorSpec(
                name="heightmap_output",
                shape=("batch", 32),
                dtype="float32",
                description="2 heightmap types × 4×4 quart grid, flattened to 32",
            ),
        ),
        onnx_opset=18,
        description="HeightmapPredictor: 96 climate grid → 32 height values",
        changelog="Initial tracked revision.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.heightmap.train_heightmap:main",
        export_fn="voxel_tree.tasks.heightmap.export_heightmap:main",
    )
)


# ══════════════════════════════════════════════════════════════════════════
#  SPARSE OCTREE
# ══════════════════════════════════════════════════════════════════════════

# ── revision 0 (legacy 13ch / 4×2×4 spatial) ─────────────────────────────
_register(
    ModelContract(
        model_name="sparse_octree",
        revision=0,
        contract_id="lodiffusion.v6.sparse_octree",
        inputs=(
            TensorSpec(
                name="noise_3d",
                shape=(1, 13, 4, 2, 4),
                dtype="float32",
                description="13-channel noise, 4×2×4 spatial (legacy v6 layout)",
            ),
        ),
        outputs=tuple(
            spec
            for lvl in range(4, -1, -1)
            for spec in (
                TensorSpec(
                    name=f"split_L{lvl}",
                    shape=(1, 8 ** (4 - lvl)),
                    dtype="float32",
                ),
                TensorSpec(
                    name=f"label_L{lvl}",
                    shape=(1, 8 ** (4 - lvl), "num_classes"),
                    dtype="float32",
                ),
            )
        ),
        onnx_opset=18,
        description="Sparse octree v6: 13ch/4×2×4 → 5-level block hierarchy",
        changelog="Initial tracked revision (retroactive).",
        build_pairs_fn="voxel_tree.tasks.octree.build_pairs:main",
        train_fn="voxel_tree.tasks.sparse_octree.train:train_sparse_octree",
        export_fn="voxel_tree.tasks.sparse_octree.export_sparse_octree:export_sparse_octree",
    )
)

# ── revision 1 (v7: 15ch / 4×4×4 spatial) ────────────────────────────────
_register(
    ModelContract(
        model_name="sparse_octree",
        revision=1,
        contract_id="lodiffusion.v7.sparse_octree",
        inputs=(
            TensorSpec(
                name="noise_3d",
                shape=(1, 15, 4, 4, 4),
                dtype="float32",
                channels=(
                    "temperature",
                    "vegetation",
                    "continents",
                    "erosion",
                    "depth",
                    "ridges",
                    "preliminary_surface_level",
                    "final_density",
                    "barrier",
                    "fluid_level_floodedness",
                    "fluid_level_spread",
                    "lava",
                    "vein_toggle",
                    "vein_ridged",
                    "vein_gap",
                ),
                channel_indices=tuple(range(15)),
                description="15 RouterField channels at 4×4×4 quart resolution",
            ),
        ),
        outputs=tuple(
            spec
            for lvl in range(4, -1, -1)
            for spec in (
                TensorSpec(
                    name=f"split_L{lvl}",
                    shape=(1, 8 ** (4 - lvl)),
                    dtype="float32",
                ),
                TensorSpec(
                    name=f"label_L{lvl}",
                    shape=(1, 8 ** (4 - lvl), "num_classes"),
                    dtype="float32",
                ),
            )
        ),
        onnx_opset=18,
        description="Sparse octree v7: 15ch/4×4×4 → 5-level block hierarchy",
        changelog="Expanded from 13 to 15 RouterField channels. "
        "Spatial Y dimension expanded from 2 to 4.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.sparse_octree.train:train_sparse_octree",
        export_fn="voxel_tree.tasks.sparse_octree.export_sparse_octree:export_sparse_octree",
    )
)

# ── revision 2 (v7 actual: 13 cave noise ch / 4×2×4 spatial) ─────────────
# Revision 1 was a speculative pre-release spec (15ch/4×4×4) that was never
# matched by real DataHarvester output.  Revision 2 documents what the
# extract_octree + build_v7_pairs pipeline actually produces.
_register(
    ModelContract(
        model_name="sparse_octree",
        revision=2,
        contract_id="lodiffusion.v7.sparse_octree_v2",
        inputs=(
            TensorSpec(
                name="noise_3d",
                shape=(1, 13, 4, 2, 4),
                dtype="float32",
                channels=(
                    "offset",
                    "factor",
                    "jaggedness",
                    "depth",
                    "sloped_cheese",
                    "y",
                    "entrances",
                    "pillars",
                    "spaghetti_2d",
                    "spaghetti_roughness",
                    "noodle",
                    "base_3d_noise",
                    "final_density",
                ),
                channel_indices=tuple(range(13)),
                description="13 cave noise channels at 4×2×4 quart resolution",
            ),
        ),
        outputs=tuple(
            spec
            for lvl in range(4, -1, -1)
            for spec in (
                TensorSpec(
                    name=f"split_L{lvl}",
                    shape=(1, 8 ** (4 - lvl)),
                    dtype="float32",
                ),
                TensorSpec(
                    name=f"label_L{lvl}",
                    shape=(1, 8 ** (4 - lvl), "num_classes"),
                    dtype="float32",
                ),
            )
        ),
        onnx_opset=18,
        description="Sparse octree v7: 13ch/4×2×4 cave noise → 5-level block hierarchy",
        changelog="Corrected channel count from 15 to 13 (actual DataHarvester output). "
        "Corrected spatial_y from 4 to 2 (actual NPZ shape from build_v7_pairs). "
        "Rev 1 was a speculative spec never matched by real data.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.sparse_octree.train:train_sparse_octree",
        export_fn="voxel_tree.tasks.sparse_octree.export_sparse_octree:export_sparse_octree",
    )
)

# ── revision 3 (v7 + 17-bit position encoding) ──────────────────────────
# Adds optional binary-encoded chunk position as an auxiliary input:
#   4 bits (chunk_x % 16) + 4 bits (chunk_z % 16) + 9 bits (block_y + 64)
# When position_bits are present the encoder receives 17 extra boolean
# features concatenated after the flattened noise_3d.
_register(
    ModelContract(
        model_name="sparse_octree",
        revision=3,
        contract_id="lodiffusion.v7.sparse_octree_v3",
        inputs=(
            TensorSpec(
                name="noise_3d",
                shape=(1, 13, 4, 2, 4),
                dtype="float32",
                channels=(
                    "offset",
                    "factor",
                    "jaggedness",
                    "depth",
                    "sloped_cheese",
                    "y",
                    "entrances",
                    "pillars",
                    "spaghetti_2d",
                    "spaghetti_roughness",
                    "noodle",
                    "base_3d_noise",
                    "final_density",
                ),
                channel_indices=tuple(range(13)),
                description="13 cave noise channels at 4×2×4 quart resolution",
            ),
            TensorSpec(
                name="position_bits",
                shape=(1, 17),
                dtype="float32",
                channels=(
                    "cx_b3", "cx_b2", "cx_b1", "cx_b0",
                    "cz_b3", "cz_b2", "cz_b1", "cz_b0",
                    "y_b8", "y_b7", "y_b6", "y_b5", "y_b4",
                    "y_b3", "y_b2", "y_b1", "y_b0",
                ),
                channel_indices=tuple(range(17)),
                description="Binary-encoded chunk position: "
                "4 bits (chunk_x %% 16) + 4 bits (chunk_z %% 16) "
                "+ 9 bits (block_y + 64, range 0-383)",
            ),
        ),
        outputs=tuple(
            spec
            for lvl in range(4, -1, -1)
            for spec in (
                TensorSpec(
                    name=f"split_L{lvl}",
                    shape=(1, 8 ** (4 - lvl)),
                    dtype="float32",
                ),
                TensorSpec(
                    name=f"label_L{lvl}",
                    shape=(1, 8 ** (4 - lvl), "num_classes"),
                    dtype="float32",
                ),
            )
        ),
        onnx_opset=18,
        description="Sparse octree v7: 13ch/4×2×4 cave noise + 17-bit position "
        "→ 5-level block hierarchy",
        changelog="Added 17-bit binary position encoding (chunk_x, chunk_z, block_y) "
        "as auxiliary input to the noise encoder. build_sparse_octree_pairs now "
        "saves chunk_x, chunk_z, section_y in the NPZ; train reads them and "
        "calls encode_chunk_position_bits() when use_position_bits=True.",
        build_pairs_fn="voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs:main",
        train_fn="voxel_tree.tasks.sparse_octree.train:train_sparse_octree",
        export_fn="voxel_tree.tasks.sparse_octree.export_sparse_octree:export_sparse_octree",
        extra={"position_bits": 17},
    )
)
