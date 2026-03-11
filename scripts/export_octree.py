#!/usr/bin/env python3
"""Octree ONNX Export — 3 Separate ONNX Models for the Octree Pipeline.

Exports three 3D U-Net models to ONNX for the LODiffusion octree runtime:

  octree_init.onnx      — L4 root (conditioning only, no parent)
  octree_refine.onnx    — L3/L2/L1 shared (parent context + level embedding)
  octree_leaf.onnx      — L0 block-level (parent context, no occ head)

All models operate on 32³ WorldSection grids. Exported with **dynamic batch
dimensions** so the Java runtime can batch multiple sections per call.

ONNX I/O Contracts::

    octree_init.onnx:
      Inputs:  heightmap float32[N,5,32,32], biome int64[N,32,32], y_position int64[N]
      Outputs: block_logits float32[N,V,32,32,32], occ_logits float32[N,8]

    octree_refine.onnx:
      Inputs:  parent_blocks int64[N,32,32,32],
               heightmap float32[N,5,32,32], biome int64[N,32,32],
               y_position int64[N], level int64[N]
      Outputs: block_logits float32[N,V,32,32,32], occ_logits float32[N,8]

    octree_leaf.onnx:
      Inputs:  parent_blocks int64[N,32,32,32],
               heightmap float32[N,5,32,32], biome int64[N,32,32],
               y_position int64[N]
      Outputs: block_logits float32[N,V,32,32,32]

Usage::

    python scripts/export_octree.py \\
        --checkpoint octree_training/best_model.pt \\
        --out-dir production

"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.octree_models import (  # noqa: E402
    OctreeConfig,
    OctreeInitModel,
    OctreeLeafModel,
    OctreeRefineModel,
    create_init_model,
    create_leaf_model,
    create_refine_model,
)

LOGGER = logging.getLogger("export_octree")


# ─── Provenance helpers ─────────────────────────────────────────────


def collect_export_provenance() -> Dict[str, Any]:
    """Collect git commit, branch, and working-tree status."""
    prov: Dict[str, Any] = {}
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        prov["git_commit"] = sha
    except Exception:
        pass
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        prov["git_branch"] = branch
    except Exception:
        pass
    try:
        subprocess.check_output(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        prov["git_clean"] = True
    except subprocess.CalledProcessError:
        prov["git_clean"] = False
    except Exception:
        prov["git_clean"] = None
    return prov


def embed_block_mapping(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Embed Voxy vocabulary into a config dict for self-contained export."""
    try:
        vocab_path = Path("config/voxy_vocab.json")
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                block_mapping = json.load(f)
            config_dict["block_mapping"] = block_mapping
            config_dict["block_id_to_name"] = {v: k for k, v in block_mapping.items()}
            LOGGER.info("Embedded %d block mappings from %s", len(block_mapping), vocab_path)
        else:
            LOGGER.warning("No block vocabulary found at %s", vocab_path)
    except Exception as e:
        LOGGER.warning("Failed to embed block mapping: %s", e)
    return config_dict


# ─── ONNX adapter wrappers ──────────────────────────────────────────


class OctreeInitAdapter(torch.nn.Module):
    """ONNX adapter for OctreeInitModel (L4, no parent).

    Inputs (N = dynamic batch):
      heightmap  : float32[N, 5, 32, 32]
      biome      : int64[N, 32, 32]
      y_position : int64[N]

    Outputs:
      block_logits : float32[N, V, 32, 32, 32]
      occ_logits   : float32[N, 8]
    """

    def __init__(self, model: OctreeInitModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(heightmap=heightmap, biome=biome, y_position=y_position)
        return out["block_type_logits"], out["occ_logits"]


class OctreeRefineAdapter(torch.nn.Module):
    """ONNX adapter for OctreeRefineModel (L3/L2/L1).

    Accepts raw int64 block IDs — the ParentEncoder embedding is baked
    into the ONNX graph so the Java runtime passes block IDs directly.

    Inputs (N = dynamic batch):
      parent_blocks  : int64[N, 32, 32, 32]
      heightmap      : float32[N, 5, 32, 32]
      biome          : int64[N, 32, 32]
      y_position     : int64[N]
      level          : int64[N]

    Outputs:
      block_logits : float32[N, V, 32, 32, 32]
      occ_logits   : float32[N, 8]
    """

    def __init__(self, model: OctreeRefineModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        parent_blocks: torch.Tensor,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
        level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            heightmap=heightmap,
            biome=biome,
            y_position=y_position,
            level=level,
            parent_blocks=parent_blocks,
        )
        return out["block_type_logits"], out["occ_logits"]


class OctreeLeafAdapter(torch.nn.Module):
    """ONNX adapter for OctreeLeafModel (L0).

    Accepts raw int64 block IDs — the ParentEncoder embedding is baked
    into the ONNX graph.

    Inputs (N = dynamic batch):
      parent_blocks  : int64[N, 32, 32, 32]
      heightmap      : float32[N, 5, 32, 32]
      biome          : int64[N, 32, 32]
      y_position     : int64[N]

    Outputs:
      block_logits : float32[N, V, 32, 32, 32]
    """

    def __init__(self, model: OctreeLeafModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        parent_blocks: torch.Tensor,
        heightmap: torch.Tensor,
        biome: torch.Tensor,
        y_position: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            heightmap=heightmap,
            biome=biome,
            y_position=y_position,
            parent_blocks=parent_blocks,
        )
        return out["block_type_logits"]


# ─── Export logic ────────────────────────────────────────────────────


def _export_init(
    model: OctreeInitModel,
    config: OctreeConfig,
    out_dir: Path,
) -> Path:
    """Export octree_init.onnx with sidecar config and test vectors."""
    adapter = OctreeInitAdapter(model)
    adapter.eval()

    onnx_filename = "octree_init.onnx"
    onnx_path = out_dir / onnx_filename

    # Dummy inputs
    dummy_height = torch.rand(1, 5, 32, 32)
    dummy_biome = torch.randint(0, config.biome_vocab_size, (1, 32, 32), dtype=torch.long)
    dummy_y = torch.tensor([12], dtype=torch.long)
    dummy = (dummy_height, dummy_biome, dummy_y)

    input_names = ["heightmap", "biome", "y_position"]
    output_names = ["block_logits", "occ_logits"]

    dynamic_axes = {
        "heightmap": {0: "batch"},
        "biome": {0: "batch"},
        "y_position": {0: "batch"},
        "block_logits": {0: "batch"},
        "occ_logits": {0: "batch"},
    }

    torch.onnx.export(
        adapter,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    LOGGER.info("Exported octree_init → %s", onnx_path)

    # Sidecar config
    V = config.block_vocab_size
    model_config: Dict[str, Any] = {
        "version": "5.0.0",
        "contract": "lodiffusion.v5.octree",
        "model": "octree_init",
        "level": 4,
        "inputs": {
            "heightmap": [1, 5, 32, 32],
            "biome": [1, 32, 32],
            "y_position": [1],
        },
        "input_dtypes": {
            "heightmap": "float32",
            "biome": "int64",
            "y_position": "int64",
        },
        "outputs": {
            "block_logits": [1, V, 32, 32, 32],
            "occ_logits": [1, 8],
        },
        "output_resolution": 32,
        "dynamic_batch": True,
        "biome_vocab_size": config.biome_vocab_size,
        "block_vocab_size": V,
        "y_vocab_size": config.y_vocab_size,
        "channels": list(config.init_channels),
        "assumptions": {
            "y_position_range": [0, config.y_vocab_size - 1],
            "height_planes_normalized": True,
        },
        "provenance": collect_export_provenance(),
    }
    model_config = embed_block_mapping(model_config)

    config_path = out_dir / "octree_init_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Test vectors
    with torch.no_grad():
        block_logits, occ_logits = adapter(*dummy)

    vectors: Dict[str, np.ndarray] = {
        "heightmap": dummy_height.numpy(),
        "biome": dummy_biome.numpy(),
        "y_position": dummy_y.numpy(),
        "block_logits": block_logits.numpy(),
        "occ_logits": occ_logits.numpy(),
    }
    vectors_path = out_dir / "octree_init_test_vectors.npz"
    np.savez(str(vectors_path), **vectors)  # type: ignore[arg-type]
    LOGGER.info("Test vectors: %s", vectors_path)

    return onnx_path


def _export_refine(
    model: OctreeRefineModel,
    config: OctreeConfig,
    out_dir: Path,
) -> Path:
    """Export octree_refine.onnx with sidecar config and test vectors."""
    adapter = OctreeRefineAdapter(model)
    adapter.eval()

    onnx_filename = "octree_refine.onnx"
    onnx_path = out_dir / onnx_filename

    # Dummy inputs
    dummy_parent = torch.randint(0, config.block_vocab_size, (1, 32, 32, 32), dtype=torch.long)
    dummy_height = torch.rand(1, 5, 32, 32)
    dummy_biome = torch.randint(0, config.biome_vocab_size, (1, 32, 32), dtype=torch.long)
    dummy_y = torch.tensor([12], dtype=torch.long)
    dummy_level = torch.tensor([2], dtype=torch.long)
    dummy = (dummy_parent, dummy_height, dummy_biome, dummy_y, dummy_level)

    input_names = ["parent_blocks", "heightmap", "biome", "y_position", "level"]
    output_names = ["block_logits", "occ_logits"]

    dynamic_axes = {
        "parent_blocks": {0: "batch"},
        "heightmap": {0: "batch"},
        "biome": {0: "batch"},
        "y_position": {0: "batch"},
        "level": {0: "batch"},
        "block_logits": {0: "batch"},
        "occ_logits": {0: "batch"},
    }

    torch.onnx.export(
        adapter,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    LOGGER.info("Exported octree_refine → %s", onnx_path)

    # Sidecar config
    V = config.block_vocab_size
    model_config: Dict[str, Any] = {
        "version": "5.0.0",
        "contract": "lodiffusion.v5.octree",
        "model": "octree_refine",
        "levels": [1, 2, 3],
        "inputs": {
            "parent_blocks": [1, 32, 32, 32],
            "heightmap": [1, 5, 32, 32],
            "biome": [1, 32, 32],
            "y_position": [1],
            "level": [1],
        },
        "input_dtypes": {
            "parent_blocks": "int64",
            "heightmap": "float32",
            "biome": "int64",
            "y_position": "int64",
            "level": "int64",
        },
        "outputs": {
            "block_logits": [1, V, 32, 32, 32],
            "occ_logits": [1, 8],
        },
        "output_resolution": 32,
        "dynamic_batch": True,
        "biome_vocab_size": config.biome_vocab_size,
        "block_vocab_size": V,
        "y_vocab_size": config.y_vocab_size,
        "level_vocab_size": config.level_vocab_size,
        "channels": list(config.refine_channels),
        "assumptions": {
            "y_position_range": [0, config.y_vocab_size - 1],
            "level_range": [1, 3],
            "height_planes_normalized": True,
            "parent_blocks": "int64 block IDs; embedding is baked into ONNX graph",
        },
        "provenance": collect_export_provenance(),
    }
    model_config = embed_block_mapping(model_config)

    config_path = out_dir / "octree_refine_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Test vectors
    with torch.no_grad():
        block_logits, occ_logits = adapter(*dummy)

    vectors: Dict[str, np.ndarray] = {
        "parent_blocks": dummy_parent.numpy(),
        "heightmap": dummy_height.numpy(),
        "biome": dummy_biome.numpy(),
        "y_position": dummy_y.numpy(),
        "level": dummy_level.numpy(),
        "block_logits": block_logits.numpy(),
        "occ_logits": occ_logits.numpy(),
    }
    vectors_path = out_dir / "octree_refine_test_vectors.npz"
    np.savez(str(vectors_path), **vectors)  # type: ignore[arg-type]
    LOGGER.info("Test vectors: %s", vectors_path)

    return onnx_path


def _export_leaf(
    model: OctreeLeafModel,
    config: OctreeConfig,
    out_dir: Path,
) -> Path:
    """Export octree_leaf.onnx with sidecar config and test vectors."""
    adapter = OctreeLeafAdapter(model)
    adapter.eval()

    onnx_filename = "octree_leaf.onnx"
    onnx_path = out_dir / onnx_filename

    # Dummy inputs
    dummy_parent = torch.randint(0, config.block_vocab_size, (1, 32, 32, 32), dtype=torch.long)
    dummy_height = torch.rand(1, 5, 32, 32)
    dummy_biome = torch.randint(0, config.biome_vocab_size, (1, 32, 32), dtype=torch.long)
    dummy_y = torch.tensor([12], dtype=torch.long)
    dummy = (dummy_parent, dummy_height, dummy_biome, dummy_y)

    input_names = ["parent_blocks", "heightmap", "biome", "y_position"]
    output_names = ["block_logits"]

    dynamic_axes = {
        "parent_blocks": {0: "batch"},
        "heightmap": {0: "batch"},
        "biome": {0: "batch"},
        "y_position": {0: "batch"},
        "block_logits": {0: "batch"},
    }

    torch.onnx.export(
        adapter,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    LOGGER.info("Exported octree_leaf → %s", onnx_path)

    # Sidecar config
    V = config.block_vocab_size
    model_config: Dict[str, Any] = {
        "version": "5.0.0",
        "contract": "lodiffusion.v5.octree",
        "model": "octree_leaf",
        "level": 0,
        "inputs": {
            "parent_blocks": [1, 32, 32, 32],
            "heightmap": [1, 5, 32, 32],
            "biome": [1, 32, 32],
            "y_position": [1],
        },
        "input_dtypes": {
            "parent_blocks": "int64",
            "heightmap": "float32",
            "biome": "int64",
            "y_position": "int64",
        },
        "outputs": {
            "block_logits": [1, V, 32, 32, 32],
        },
        "output_resolution": 32,
        "dynamic_batch": True,
        "biome_vocab_size": config.biome_vocab_size,
        "block_vocab_size": V,
        "y_vocab_size": config.y_vocab_size,
        "channels": list(config.leaf_channels),
        "assumptions": {
            "y_position_range": [0, config.y_vocab_size - 1],
            "height_planes_normalized": True,
            "parent_blocks": "int64 block IDs; embedding is baked into ONNX graph",
        },
        "provenance": collect_export_provenance(),
    }
    model_config = embed_block_mapping(model_config)

    config_path = out_dir / "octree_leaf_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Test vectors
    with torch.no_grad():
        block_logits = adapter(*dummy)

    vectors: Dict[str, np.ndarray] = {
        "parent_blocks": dummy_parent.numpy(),
        "heightmap": dummy_height.numpy(),
        "biome": dummy_biome.numpy(),
        "y_position": dummy_y.numpy(),
        "block_logits": block_logits.numpy(),
    }
    vectors_path = out_dir / "octree_leaf_test_vectors.npz"
    np.savez(str(vectors_path), **vectors)  # type: ignore[arg-type]
    LOGGER.info("Test vectors: %s", vectors_path)

    return onnx_path


def _validate_onnx(onnx_path: Path, test_vectors_path: Path) -> bool:
    """Validate ONNX model round-trip accuracy against test vectors.

    Returns True if all outputs match within tolerance.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        LOGGER.warning("onnxruntime not installed — skipping ONNX validation")
        return True

    vectors = np.load(test_vectors_path)
    session = ort.InferenceSession(str(onnx_path))

    # Build input dict from session's expected inputs
    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if name in vectors:
            feed[name] = vectors[name]
        else:
            LOGGER.warning("Missing test vector for ONNX input '%s'", name)
            return False

    # Run inference
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, feed)

    # Compare outputs
    all_ok = True
    for oname, result in zip(output_names, results):
        if oname in vectors:
            expected = vectors[oname]
            max_diff = np.max(np.abs(result - expected))
            if max_diff > 1e-3:
                LOGGER.error(
                    "ONNX validation FAILED for %s output '%s': max_diff=%.6f",
                    onnx_path.name,
                    oname,
                    max_diff,
                )
                all_ok = False
            else:
                LOGGER.info(
                    "ONNX validation OK for %s output '%s': max_diff=%.6f",
                    onnx_path.name,
                    oname,
                    max_diff,
                )
    return all_ok


# ─── Checkpoint loading ─────────────────────────────────────────────


def load_octree_checkpoint(
    checkpoint_path: Path,
) -> Tuple[OctreeConfig, Dict[str, torch.nn.Module]]:
    """Load a 3-model octree checkpoint.

    Returns ``(config, {"init": model, "refine": model, "leaf": model})``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            f"Available keys: {list(ckpt.keys())}"
        )
    config: OctreeConfig = ckpt["config"]
    LOGGER.info("Config from checkpoint: %s", config)

    if "model_state_dicts" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'model_state_dicts'. "
            f"Available keys: {list(ckpt.keys())}"
        )
    state_dicts = ckpt["model_state_dicts"]
    LOGGER.info("Checkpoint model keys: %s", list(state_dicts.keys()))

    model_factories = {
        "init": create_init_model,
        "refine": create_refine_model,
        "leaf": create_leaf_model,
    }

    models: Dict[str, torch.nn.Module] = {}
    for key, factory in model_factories.items():
        model = factory(config)
        if key in state_dicts:
            result = model.load_state_dict(state_dicts[key], strict=False)
            if result.unexpected_keys:
                LOGGER.warning("Unexpected keys in %s: %s", key, result.unexpected_keys)
            if result.missing_keys:
                LOGGER.warning("Missing keys in %s: %s", key, result.missing_keys)
            LOGGER.info("Loaded state dict for '%s'", key)
        else:
            LOGGER.warning("No state dict for '%s' — using random weights", key)
        models[key] = model

    return config, models


# ─── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 3 octree models to ONNX with sidecar configs"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Octree checkpoint .pt from train_octree.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("production"),
        help="Output directory for ONNX files (default: production/)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate ONNX round-trip accuracy (requires onnxruntime)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX validation",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    config, models = load_octree_checkpoint(args.checkpoint)

    LOGGER.info("Exporting 3 octree models to %s", out_dir)

    # Export each model
    exported = []

    init_path = _export_init(models["init"], config, out_dir)  # type: ignore[arg-type]
    exported.append(init_path)

    refine_path = _export_refine(models["refine"], config, out_dir)  # type: ignore[arg-type]
    exported.append(refine_path)

    leaf_path = _export_leaf(models["leaf"], config, out_dir)  # type: ignore[arg-type]
    exported.append(leaf_path)

    # Validate ONNX round-trip
    if args.validate and not args.no_validate:
        for onnx_path in exported:
            vectors_path = out_dir / onnx_path.name.replace(".onnx", "_test_vectors.npz")
            if vectors_path.exists():
                _validate_onnx(onnx_path, vectors_path)

    # Pipeline manifest
    required_files = [
        "pipeline_manifest.json",
        "octree_init.onnx",
        "octree_init_config.json",
        "octree_refine.onnx",
        "octree_refine_config.json",
        "octree_leaf.onnx",
        "octree_leaf_config.json",
    ]

    manifest: Dict[str, Any] = {
        "version": "5.0.0",
        "contract": "lodiffusion.v5.octree",
        "required_files": required_files,
        "pipeline": [
            {
                "model": "octree_init",
                "onnx": "octree_init.onnx",
                "level": 4,
                "has_parent": False,
                "has_occ_head": True,
                "output_resolution": 32,
            },
            {
                "model": "octree_refine",
                "onnx": "octree_refine.onnx",
                "levels": [3, 2, 1],
                "has_parent": True,
                "has_occ_head": True,
                "output_resolution": 32,
            },
            {
                "model": "octree_leaf",
                "onnx": "octree_leaf.onnx",
                "level": 0,
                "has_parent": True,
                "has_occ_head": False,
                "output_resolution": 32,
            },
        ],
        "block_vocab_size": config.block_vocab_size,
        "biome_vocab_size": config.biome_vocab_size,
        "y_vocab_size": config.y_vocab_size,
        "provenance": collect_export_provenance(),
    }
    manifest = embed_block_mapping(manifest)

    manifest_path = out_dir / "pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info("Pipeline manifest: %s", manifest_path)
    LOGGER.info("Export complete.")
    for p in exported:
        LOGGER.info("  %s", p)


if __name__ == "__main__":
    main()
