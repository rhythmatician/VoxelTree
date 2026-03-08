#!/usr/bin/env python3
"""LODiffusion Progressive Export — 4 Separate ONNX Models

Exports 4 separate progressive LOD models to ONNX for the LODiffusion runtime:

  init_to_lod4.onnx        — Conditioning → 1×1×1  (tiny MLP)
  refine_lod4_to_lod3.onnx — 1³ parent → 2×2×2     (small Conv3D)
  refine_lod3_to_lod2.onnx — 2³ parent → 4×4×4     (medium Conv3D)
  refine_lod2_to_lod1.onnx — 4³ parent → 8×8×8     (medium-large Conv3D)

LOD0 is NOT exported — vanilla terrain handles full resolution.

Each model receives anchor conditioning inputs:
  x_height_planes : [1, 5, 16, 16]   float32
  x_biome         : [1, 16, 16]      int64
  x_y_index       : [1]              int64

Refinement models additionally receive:
  x_parent        : [1, 1, P, P, P]  float32   (P = output_size // 2)

All models output:
  block_logits    : [1, N_blocks, D, D, D]
  air_mask        : [1, 1, D, D, D]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Ensure the repo root (VoxelTree/) is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.progressive_lod_models import (  # noqa: E402
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
    create_init_model,
    create_lod2_to_lod1_model,
    create_lod3_to_lod2_model,
    create_lod4_to_lod3_model,
)
from train.unet3d import SimpleFlexibleConfig  # noqa: E402

LOGGER = logging.getLogger("export_lod")

# ─── Model step definitions ─────────────────────────────────────────────
# Each entry: (model_key, factory_func, output_size, parent_size, onnx_filename)
MODEL_STEPS = [
    ("init_to_lod4", create_init_model, 1, 0, "init_to_lod4.onnx"),
    ("lod4to3", create_lod4_to_lod3_model, 2, 1, "refine_lod4_to_lod3.onnx"),
    ("lod3to2", create_lod3_to_lod2_model, 4, 2, "refine_lod3_to_lod2.onnx"),
    ("lod2to1", create_lod2_to_lod1_model, 8, 4, "refine_lod2_to_lod1.onnx"),
]


# ─── Reusable helpers ────────────────────────────────────────────────────


def collect_export_provenance() -> Dict:
    """Collect comprehensive provenance for the exported model."""
    prov: Dict[str, Any] = {}
    # Git commit SHA
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        prov["git_commit"] = sha
    except Exception:  # pragma: no cover
        pass
    # Git branch
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        prov["git_branch"] = branch
    except Exception:  # pragma: no cover
        pass
    # Working directory status
    try:
        subprocess.check_output(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        prov["git_clean"] = True
    except subprocess.CalledProcessError:
        prov["git_clean"] = False
    except Exception:  # pragma: no cover
        prov["git_clean"] = None
    return prov


def embed_block_mapping(model_config: Dict) -> Dict:
    """Embed Voxy vocabulary into model config for self-contained export."""
    try:
        vocab_path = Path("config/voxy_vocab.json")
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                block_mapping = json.load(f)
            model_config["block_mapping"] = block_mapping
            id_to_name = {v: k for k, v in block_mapping.items()}
            model_config["block_id_to_name"] = id_to_name
            LOGGER.info(f"Embedded {len(block_mapping)} block mappings from {vocab_path}")
        else:
            LOGGER.warning("No block vocabulary found")
    except Exception as e:  # pragma: no cover
        LOGGER.warning(f"Failed to embed block mapping: {e}")
    return model_config


# ─── ONNX adapter wrappers ──────────────────────────────────────────────


class InitModelAdapter(torch.nn.Module):
    """Adapter for Init→LOD4 model (no parent).

    ONNX inputs:
      x_height_planes : [1, 5, 16, 16]  float32
      x_biome         : [1, 16, 16]     int64
      x_y_index       : [1]             int64

    ONNX outputs:
      block_logits : [1, N_blocks, 1, 1, 1]
      air_mask     : [1, 1, 1, 1, 1]
    """

    def __init__(self, model: ProgressiveLODModel0_Initial):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_height_planes: torch.Tensor,
        x_biome: torch.Tensor,
        x_y_index: torch.Tensor,
    ):
        out = self.model(
            height_planes=x_height_planes,
            biome_indices=x_biome,
            y_index=x_y_index,
        )
        return out["block_type_logits"], out["air_mask_logits"]


class RefinementModelAdapter(torch.nn.Module):
    """Adapter for refinement models (LOD4→3, LOD3→2, LOD2→1).

    ONNX inputs:
      x_height_planes : [1, 5, 16, 16]   float32
      x_biome         : [1, 16, 16]      int64
      x_y_index       : [1]              int64
      x_parent        : [1, 1, P, P, P]  float32

    ONNX outputs:
      block_logits : [1, N_blocks, D, D, D]
      air_mask     : [1, 1, D, D, D]
    """

    def __init__(self, model: ProgressiveLODModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_height_planes: torch.Tensor,
        x_biome: torch.Tensor,
        x_y_index: torch.Tensor,
        x_parent: torch.Tensor,
    ):
        out = self.model(
            height_planes=x_height_planes,
            biome_indices=x_biome,
            y_index=x_y_index,
            x_parent=x_parent,
        )
        return out["block_type_logits"], out["air_mask_logits"]


# ─── Export logic ────────────────────────────────────────────────────────


def export_step(
    adapter: torch.nn.Module,
    step_name: str,
    output_size: int,
    parent_size: int,
    onnx_filename: str,
    config: SimpleFlexibleConfig,
    out_dir: Path,
) -> Path:
    """Export a single progressive step to ONNX with sidecar config + test vectors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter.eval()

    biome_vocab = config.biome_vocab_size
    block_vocab = config.block_vocab_size

    # Build dummy inputs
    dummy_height = torch.rand(1, 5, 16, 16)
    dummy_biome = torch.randint(0, biome_vocab, (1, 16, 16), dtype=torch.long)
    dummy_y = torch.tensor([12], dtype=torch.long)

    if parent_size == 0:
        # Init model — no parent
        dummy = (dummy_height, dummy_biome, dummy_y)
        input_names = ["x_height_planes", "x_biome", "x_y_index"]
        inputs_spec = {
            "x_height_planes": [1, 5, 16, 16],
            "x_biome": [1, 16, 16],
            "x_y_index": [1],
        }
        input_dtypes = {
            "x_height_planes": "float32",
            "x_biome": "int64",
            "x_y_index": "int64",
        }
    else:
        # Refinement model — with parent
        dummy_parent = torch.rand(1, 1, parent_size, parent_size, parent_size)
        dummy = (dummy_height, dummy_biome, dummy_y, dummy_parent)
        input_names = ["x_height_planes", "x_biome", "x_y_index", "x_parent"]
        inputs_spec = {
            "x_height_planes": [1, 5, 16, 16],
            "x_biome": [1, 16, 16],
            "x_y_index": [1],
            "x_parent": [1, 1, parent_size, parent_size, parent_size],
        }
        input_dtypes = {
            "x_height_planes": "float32",
            "x_biome": "int64",
            "x_y_index": "int64",
            "x_parent": "float32",
        }

    onnx_path = out_dir / onnx_filename
    torch.onnx.export(
        adapter,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["block_logits", "air_mask"],
        dynamic_axes=None,
        dynamo=False,
    )
    LOGGER.info("Exported %s → %s", step_name, onnx_path)

    # ── Per-model sidecar config ─────────────────────────────────────
    D = output_size
    model_config: Dict[str, Any] = {
        "version": "3.0.0",
        "contract": "lodiffusion.v3.progressive",
        "step": step_name,
        "inputs": inputs_spec,
        "input_dtypes": input_dtypes,
        "outputs": {
            "block_logits": [1, block_vocab, D, D, D],
            "air_mask": [1, 1, D, D, D],
        },
        "parent_resolution": parent_size,
        "output_resolution": D,
        "assumptions": {
            "y_index_range": [0, 23],
            "height_planes_normalized": True,
        },
        "biome_vocab_size": biome_vocab,
        "block_vocab_size": block_vocab,
        "provenance": collect_export_provenance(),
    }
    model_config = embed_block_mapping(model_config)

    config_name = onnx_filename.replace(".onnx", "_config.json")
    with open(out_dir / config_name, "w") as f:
        json.dump(model_config, f, indent=2)

    # ── Test vectors ──────────────────────────────────────────────────
    with torch.no_grad():
        block_logits, air_mask = adapter(*dummy)

    vectors = {
        "x_height_planes": dummy_height.cpu().numpy(),
        "x_biome": dummy_biome.cpu().numpy(),
        "x_y_index": dummy_y.cpu().numpy(),
        "block_logits": block_logits.cpu().numpy(),
        "air_mask": air_mask.cpu().numpy(),
    }
    if parent_size > 0:
        vectors["x_parent"] = dummy[-1].cpu().numpy()

    vectors_name = onnx_filename.replace(".onnx", "_test_vectors.npz")
    np.savez(out_dir / vectors_name, **vectors)
    LOGGER.info("Wrote test vectors: %s", out_dir / vectors_name)

    return onnx_path


def load_progressive_checkpoint(
    checkpoint_path: Path,
) -> tuple[SimpleFlexibleConfig, Dict[str, torch.nn.Module]]:
    """Load a multi-model checkpoint and return config + per-step models.

    Reads the ``SimpleFlexibleConfig`` stored in the checkpoint by
    ``train.py`` so no external YAML config is needed.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            "Expected a checkpoint from train.py with an embedded "
            f"SimpleFlexibleConfig. Available keys: {list(ckpt.keys())}"
        )
    config: SimpleFlexibleConfig = ckpt["config"]
    LOGGER.info("Loaded config from checkpoint: %s", config)

    if "model_state_dicts" in ckpt:
        state_dicts = ckpt["model_state_dicts"]
        LOGGER.info("Loaded per-step checkpoint with keys: %s", list(state_dicts.keys()))
    else:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'model_state_dicts' key. "
            "Expected a multi-model checkpoint from train.py. "
            f"Available keys: {list(ckpt.keys())}"
        )

    models: Dict[str, torch.nn.Module] = {}
    for key, factory, output_size, parent_size, _ in MODEL_STEPS:
        model = factory(config)
        if key in state_dicts:
            model.load_state_dict(state_dicts[key], strict=False)
            LOGGER.info("Loaded state dict for %s (output %d³)", key, output_size)
        else:
            LOGGER.warning("No state dict for %s — using random weights", key)
        models[key] = model

    return config, models


def main():
    parser = argparse.ArgumentParser(
        description="Export progressive LOD models to 4 separate ONNX files"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Multi-model .pt checkpoint from train.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("production"),
        help="Output directory for ONNX files",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Load config + per-step models from checkpoint (no external YAML needed)
    config, models = load_progressive_checkpoint(args.checkpoint)

    LOGGER.info("Exporting 4 progressive LOD models to %s", args.out_dir)

    exported = []
    for key, factory, output_size, parent_size, onnx_filename in MODEL_STEPS:
        model = models[key]
        if parent_size == 0:
            adapter = InitModelAdapter(model)
        else:
            adapter = RefinementModelAdapter(model)

        path = export_step(
            adapter=adapter,
            step_name=key,
            output_size=output_size,
            parent_size=parent_size,
            onnx_filename=onnx_filename,
            config=config,
            out_dir=args.out_dir,
        )
        exported.append(path)

    # Write a pipeline manifest listing all exported models
    manifest = {
        "version": "3.0.0",
        "contract": "lodiffusion.v3.progressive",
        "pipeline": [
            {
                "step": key,
                "onnx": fn,
                "output_resolution": out,
                "parent_resolution": par,
            }
            for key, _, out, par, fn in MODEL_STEPS
        ],
        "provenance": collect_export_provenance(),
    }
    manifest = embed_block_mapping(manifest)
    with open(args.out_dir / "pipeline_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info("Export complete. Manifest: %s", args.out_dir / "pipeline_manifest.json")
    for p in exported:
        LOGGER.info("  %s", p)


if __name__ == "__main__":
    main()
