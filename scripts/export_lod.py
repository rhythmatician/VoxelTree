#!/usr/bin/env python3
"""LODiffusion Contract Export

Exports a trained VoxelTree model to the strict LODiffusion runtime contract:
Inputs (static batch=1):
  x_parent : [1,1,8,8,8] float32 (0/1 occupancy)
  x_biome  : [1,N_biomes,8,8,1] float32 (one-hot)
  x_height : [1,1,8,8,1] float32 (normalized 0..1)
  x_lod    : [1,1] float32 (LOD scalar in [1,4])

Outputs:
  block_logits : [1,N_blocks,16,16,16]
  air_mask     : [1,1,16,16,16]

The adapter internally converts these tensors to the model's current internal
conditioning format (biome ids @16x16, heightmap @16x16, river zeros, fixed y_index=12).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from train.unet3d import UNet3DConfig, VoxelUNet3D

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
    """Embed complete block mapping into model config for self-contained export."""
    try:
        mapping_path = Path("scripts/extraction/complete_block_mapping.json")
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                block_mapping = json.load(f)
            # Embed as both name->id and id->name for convenience
            model_config["block_mapping"] = block_mapping
            id_to_name = {v: k for k, v in block_mapping.items()}
            model_config["block_id_to_name"] = id_to_name
            LOGGER.info(f"Embedded {len(block_mapping)} block mappings into model config")
        else:
            LOGGER.warning(f"Block mapping not found at {mapping_path}")
    except Exception as e:  # pragma: no cover
        LOGGER.warning(f"Failed to embed block mapping: {e}")
    return model_config


class LODiffusionAdapter(torch.nn.Module):
    """Wraps VoxelUNet3D to expose the LODiffusion input contract."""

    def __init__(self, model: VoxelUNet3D, biome_vocab_size: int):
        super().__init__()
        self.model = model
        self.biome_vocab_size = biome_vocab_size

    def forward(self, x_parent, x_biome, x_height, x_lod):  # type: ignore
        # Shapes:
        # x_parent: (1,1,8,8,8)
        # x_biome:  (1,N_biomes,8,8,1) one-hot
        # x_height: (1,1,8,8,1)
        # x_lod:    (1,1)
        # Remove trailing singleton dim from spatial 2D inputs
        x_biome = x_biome.squeeze(-1)  # (1,N_biomes,8,8)
        x_height = x_height.squeeze(-1)  # (1,1,8,8)

        # Convert one-hot biome to indices then upscale to 16x16
        biome_ids = torch.argmax(x_biome, dim=1)  # (1,8,8)
        biome_ids_up = (
            F.interpolate(biome_ids.unsqueeze(1).float(), size=(16, 16), mode="nearest")
            .squeeze(1)
            .long()
        )  # (1,16,16)

        # Upscale heightmap to 16x16 bilinear
        height_up = F.interpolate(x_height, size=(16, 16), mode="bilinear", align_corners=False)
        # River patch zeros (not provided in contract yet)
        river_patch = torch.zeros_like(height_up)

        # y_index fixed midpoint (12) for now; could be parameterized later
        batch = x_parent.shape[0]
        y_index = torch.full((batch,), 12, device=x_parent.device, dtype=torch.long)

        # LOD -> integer index (round & clamp 1..4)
        lod = torch.clamp(torch.round(x_lod.squeeze(-1)).long(), 1, 4)

        outputs = self.model(
            x_parent,
            biome_ids_up,
            height_up,
            river_patch,
            y_index,
            lod,
        )
        # Reorder to contract (block_logits, air_mask)
        return outputs["block_type_logits"], outputs["air_mask_logits"]


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict) -> VoxelUNet3D:
    mcfg = cfg["model"]
    model = VoxelUNet3D(
        UNet3DConfig(
            input_channels=mcfg.get("input_channels", 1),
            output_channels=2,
            base_channels=mcfg.get("base_channels", 64),
            depth=mcfg.get("depth", 2),
            biome_vocab_size=mcfg.get("biome_vocab_size", 256),
            biome_embed_dim=mcfg.get("biome_embed_dim", 16),
            heightmap_channels=mcfg.get("heightmap_channels", 1),
            river_channels=mcfg.get("river_channels", 1),
            y_embed_dim=mcfg.get("y_embed_dim", 8),
            lod_embed_dim=mcfg.get("lod_embed_dim", 32),
            dropout_rate=mcfg.get("dropout_rate", 0.1),
            use_batch_norm=mcfg.get("use_batch_norm", True),
            activation=mcfg.get("activation", "gelu"),
            air_mask_channels=1,
            block_type_channels=mcfg.get("block_type_channels", 1104),
        )
    )
    return model


def load_checkpoint(model: VoxelUNet3D, checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)
    LOGGER.info("Loaded checkpoint: %s", checkpoint)


def export_contract(adapter: LODiffusionAdapter, cfg: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = adapter
    model.eval()

    biome_vocab = cfg["model"].get("biome_vocab_size", 256)
    block_vocab = cfg["model"].get("block_type_channels", 1104)

    dummy = (
        torch.rand(1, 1, 8, 8, 8),  # x_parent (already 0..1)
        F.one_hot(torch.randint(0, biome_vocab, (1, 8, 8)), num_classes=biome_vocab)
        .permute(0, 3, 1, 2)
        .unsqueeze(-1)
        .float(),  # x_biome
        torch.rand(1, 1, 8, 8, 1),  # x_height
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
            "x_biome": [1, biome_vocab, 8, 8, 1],
            "x_height": [1, 1, 8, 8, 1],
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
    adapter = LODiffusionAdapter(model, cfg["model"].get("biome_vocab_size", 256))
    export_contract(adapter, cfg, args.out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
