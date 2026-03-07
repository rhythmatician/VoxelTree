#!/usr/bin/env python3
"""LODiffusion Contract Export

Exports a trained VoxelTree model to the strict LODiffusion runtime contract:
Inputs (static batch=1):
  x_parent : [1,1,8,8,8]          float32 (0/1 occupancy)
  x_biome  : [1,N_biomes,16,16,1] float32 (one-hot, 16x16 chunk grid)
  x_height : [1,1,16,16,1]        float32 (normalized 0..1, 16x16 chunk grid)
  x_lod    : [1,1]                float32 (LOD scalar in [1,4])

Outputs:
  block_logits : [1,N_blocks,16,16,16]
  air_mask     : [1,1,16,16,16]

The adapter squeezes the trailing dim from 2D inputs (biome, height), converts
one-hot biome to integer IDs, and passes everything to SimpleFlexibleUNet3D with
fixed y_index=12 and integer lod.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from train.unet3d import SimpleFlexibleConfig, SimpleFlexibleUNet3D

LOGGER = logging.getLogger("export_lod")


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
        # Prefer Voxy-native vocabulary
        vocab_path = Path("config/voxy_vocab.json")
        if not vocab_path.exists():
            # Fallback to legacy VoxelTree mapping
            vocab_path = Path("scripts/extraction/complete_block_mapping.json")

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


class LODiffusionAdapter(torch.nn.Module):
    """Wraps SimpleFlexibleUNet3D to expose the LODiffusion input contract.

    Input contract (static batch=1):
      x_parent : (1,1,8,8,8)          float32  0/1 occupancy
      x_biome  : (1,N_biomes,16,16,1) float32  one-hot biome per cell
      x_height : (1,1,16,16,1)        float32  normalized 0..1
      x_lod    : (1,1)                float32  LOD scalar 1..4

    Output contract:
      block_logits : (1,N_blocks,16,16,16)
      air_mask     : (1,1,16,16,16)
    """

    def __init__(self, model: SimpleFlexibleUNet3D, biome_vocab_size: int):
        super().__init__()
        self.model = model
        self.biome_vocab_size = biome_vocab_size

    def forward(self, x_parent, x_biome, x_height, x_lod):  # type: ignore
        # x_parent: (1,1,8,8,8)
        # x_biome:  (1,N_biomes,16,16,1) one-hot  → squeeze → (1,N_biomes,16,16)
        # x_height: (1,1,16,16,1)                 → squeeze → (1,1,16,16)
        # x_lod:    (1,1)

        x_biome = x_biome.squeeze(-1)  # (1, N_biomes, 16, 16)
        x_height = x_height.squeeze(-1)  # (1, 1, 16, 16)

        # One-hot → integer biome IDs, already at 16×16 resolution
        biome_patch = torch.argmax(x_biome, dim=1).long()  # (1, 16, 16)

        # y_index fixed to midpoint slab 12
        batch = x_parent.shape[0]
        y_index = torch.full((batch,), 12, device=x_parent.device, dtype=torch.long)

        # LOD float → integer index, clamped 1..4
        lod = torch.clamp(torch.round(x_lod.squeeze(-1)).long(), 1, 4)  # (B,)

        outputs = self.model(
            parent_voxel=x_parent,
            biome_patch=biome_patch,
            heightmap_patch=x_height,
            y_index=y_index,
            lod=lod,
        )
        # Return (block_logits, air_mask) to match contract output ordering
        return outputs["block_type_logits"], outputs["air_mask_logits"]


class LODiffusionAdapterV2(torch.nn.Module):
    """Wraps SimpleFlexibleUNet3D to expose the LODiffusion v2 input contract.

    v2 drops the legacy one-hot biome and scalar heightmap in favour of:
      x_parent        : (1, 1,  8,  8,  8)  float32  binary occupancy
      x_height_planes : (1, 5, 16, 16)      float32  5-plane heightmap
      x_router6       : (1, 6, 16, 16)      float32  normalised router6
      x_biome         : (1, 16, 16)         int64    biome integer index
      x_y_index       : (1,)                int64    y-slab index 0..23
      x_lod           : (1,)                int64    LOD token 1..4

    Outputs (same as v1):
      block_logits : (1, N_blocks, 16, 16, 16)
      air_mask     : (1, 1, 16, 16, 16)
    """

    def __init__(self, model: SimpleFlexibleUNet3D):
        super().__init__()
        self.model = model

    def forward(  # type: ignore
        self,
        x_parent: torch.Tensor,  # (1,1,8,8,8) float32
        x_height_planes: torch.Tensor,  # (1,5,16,16) float32
        x_router6: torch.Tensor,  # (1,6,16,16) float32
        x_biome: torch.Tensor,  # (1,16,16)   int64
        x_y_index: torch.Tensor,  # (1,)        int64
        x_lod: torch.Tensor,  # (1,)        int64
    ):
        # Build a legacy heightmap_patch [1,1,16,16] from height_planes[0] (surface plane)
        # so the model's fallback biome/height tensors are properly shaped.
        heightmap_patch = x_height_planes[:, 0:1, :, :]  # (1,1,16,16)

        outputs = self.model(
            parent_voxel=x_parent,
            biome_patch=x_biome,  # integer indices expected by forward()
            heightmap_patch=heightmap_patch,
            y_index=x_y_index.squeeze(-1),  # (B,)
            lod=x_lod.squeeze(-1),  # (B,)
            height_planes=x_height_planes,
            router6=x_router6,
        )
        return outputs["block_type_logits"], outputs["air_mask_logits"]


def export_contract_v2(adapter: LODiffusionAdapterV2, cfg: Dict, out_dir: Path):
    """Export ONNX model+sidecar for the lodiffusion.v2 contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter.eval()

    biome_vocab = cfg["model"].get("biome_vocab_size", 256)
    block_vocab = cfg["model"].get("block_type_channels", 1102)

    dummy = (
        torch.rand(1, 1, 8, 8, 8),  # x_parent
        torch.rand(1, 5, 16, 16),  # x_height_planes
        torch.rand(1, 6, 16, 16),  # x_router6
        torch.randint(0, biome_vocab, (1, 16, 16), dtype=torch.long),  # x_biome
        torch.tensor([12], dtype=torch.long),  # x_y_index
        torch.tensor([2], dtype=torch.long),  # x_lod
    )

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        adapter,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_parent", "x_height_planes", "x_router6", "x_biome", "x_y_index", "x_lod"],
        output_names=["block_logits", "air_mask"],
        dynamic_axes=None,
        dynamo=False,  # Force legacy TorchScript-based exporter
    )
    LOGGER.info("Exported v2 ONNX model: %s", onnx_path)

    # Sidecar model_config.json (v2)
    model_config: Dict[str, Any] = {
        "version": "2.0.0",
        "contract": "lodiffusion.v2",
        "inputs": {
            "x_parent": [1, 1, 8, 8, 8],
            "x_height_planes": [1, 5, 16, 16],
            "x_router6": [1, 6, 16, 16],
            "x_biome": [1, 16, 16],
            "x_y_index": [1],
            "x_lod": [1],
        },
        "input_dtypes": {
            "x_parent": "float32",
            "x_height_planes": "float32",
            "x_router6": "float32",
            "x_biome": "int64",
            "x_y_index": "int64",
            "x_lod": "int64",
        },
        "outputs": {
            "block_logits": [1, block_vocab, 16, 16, 16],
            "air_mask": [1, 1, 16, 16, 16],
        },
        "assumptions": {
            "lod_range": [1, 4],
            "y_index_range": [0, 23],
            "height_planes_normalized": True,
            "router6_normalized": True,
        },
        "normalization": {
            "router6": {
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        },
        "biome_vocab_size": biome_vocab,
        "block_vocab_size": block_vocab,
        "provenance": collect_export_provenance(),
    }
    model_config = embed_block_mapping(model_config)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Test vectors
    with torch.no_grad():
        block_logits, air_mask = adapter(*dummy)
    np.savez(
        out_dir / "test_vectors.npz",
        x_parent=dummy[0].cpu().numpy(),
        x_height_planes=dummy[1].cpu().numpy(),
        x_router6=dummy[2].cpu().numpy(),
        x_biome=dummy[3].cpu().numpy(),
        x_y_index=dummy[4].cpu().numpy(),
        x_lod=dummy[5].cpu().numpy(),
        block_logits=block_logits.cpu().numpy(),
        air_mask=air_mask.cpu().numpy(),
    )
    LOGGER.info("Wrote test vectors: %s", out_dir / "test_vectors.npz")
    return onnx_path


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict) -> SimpleFlexibleUNet3D:
    """Build model from config, passing only fields known to SimpleFlexibleConfig."""
    mcfg = cfg.get("model", {})
    valid_fields = set(inspect.signature(SimpleFlexibleConfig).parameters.keys())
    filtered = {k: v for k, v in mcfg.items() if k in valid_fields}
    model_cfg = SimpleFlexibleConfig(**filtered)
    return SimpleFlexibleUNet3D(model_cfg)


def load_checkpoint(model: SimpleFlexibleUNet3D, checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state_dict, strict=False)
    LOGGER.info("Loaded checkpoint: %s", checkpoint)


def export_contract(adapter: LODiffusionAdapter, cfg: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = adapter
    model.eval()

    biome_vocab = cfg["model"].get("biome_vocab_size", 256)
    block_vocab = cfg["model"].get("block_type_channels", 1102)

    dummy = (
        torch.rand(1, 1, 8, 8, 8),  # x_parent (already 0..1)
        F.one_hot(torch.randint(0, biome_vocab, (1, 16, 16)), num_classes=biome_vocab)
        .permute(0, 3, 1, 2)
        .unsqueeze(-1)
        .float(),  # x_biome: 16x16 chunk size
        torch.rand(1, 1, 16, 16, 1),  # x_height: 16x16 chunk size
        torch.randint(1, 5, (1, 1)).float(),  # x_lod
    )

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_parent", "x_biome", "x_height", "x_lod"],
        output_names=["block_logits", "air_mask"],
        dynamic_axes=None,
    )
    LOGGER.info("Exported ONNX model: %s", onnx_path)

    # Sidecar model_config.json
    model_config = {
        "version": "1.0.0",
        "contract": "lodiffusion.v1",
        "inputs": {
            "x_parent": [1, 1, 8, 8, 8],
            "x_biome": [1, biome_vocab, 16, 16, 1],  # 16x16 chunk size
            "x_height": [1, 1, 16, 16, 1],  # 16x16 chunk size
            "x_lod": [1, 1],
        },
        "outputs": {
            "block_logits": [1, block_vocab, 16, 16, 16],
            "air_mask": [1, 1, 16, 16, 16],
        },
        "assumptions": {
            "y_index_fixed": 12,
            "river_patch": "zeros",
            "lod_range": [1, 4],
            "height_normalized": True,
        },
        "biome_vocab_size": biome_vocab,
        "block_vocab_size": block_vocab,
        "provenance": collect_export_provenance(),
    }
    # Embed complete block mapping for self-contained exports
    model_config = embed_block_mapping(model_config)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Test vectors (inputs + forward outputs)
    with torch.no_grad():
        block_logits, air_mask = model(*dummy)
    np.savez(
        out_dir / "test_vectors.npz",
        x_parent=dummy[0].cpu().numpy(),
        x_biome=dummy[1].cpu().numpy(),
        x_height=dummy[2].cpu().numpy(),
        x_lod=dummy[3].cpu().numpy(),
        block_logits=block_logits.cpu().numpy(),
        air_mask=air_mask.cpu().numpy(),
    )
    LOGGER.info("Wrote test vectors: %s", out_dir / "test_vectors.npz")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export VoxelTree model to LODiffusion contract")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("production"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    cfg = load_config(args.config)
    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)

    # Detect contract version from config (export.contract or v2 marker keys)
    export_cfg = cfg.get("export", {})
    contract = export_cfg.get("contract", "lodiffusion.v1")

    if contract == "lodiffusion.v2":
        LOGGER.info("Exporting lodiffusion.v2 contract (anchor channels)")
        adapter_v2 = LODiffusionAdapterV2(model)
        export_contract_v2(adapter_v2, cfg, args.out_dir)
    else:
        LOGGER.info("Exporting lodiffusion.v1 contract (legacy one-hot)")
        adapter_v1 = LODiffusionAdapter(model, cfg["model"].get("biome_vocab_size", 256))
        export_contract(adapter_v1, cfg, args.out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
