"""Export trained Voxy per-level models to ONNX with sidecar configs.

Usage
-----
    # Export all 5 levels
    python -m voxel_tree.tasks.sparse_octree.voxy_export \\
        --checkpoint-dir checkpoints --out-dir production

    # Export just L4 and L3
    python -m voxel_tree.tasks.sparse_octree.voxy_export \\
        --checkpoint-dir checkpoints --out-dir production --levels 4 3

Each level produces three artefacts:
    voxy_l{N}.onnx              – the ONNX model (opset 17, dynamic batch)
    voxy_l{N}_config.json       – sidecar describing I/O contract + vocab
    voxy_l{N}_test_vectors.npz  – deterministic dummy I/O for round-trip tests
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from voxel_tree.tasks.sparse_octree.voxy_models import (
    BIOME_SHAPES,
    L2_NOISE_CHANNELS,
    L3_NOISE_CHANNELS,
    L4_NOISE_CHANNELS,
    NOISE_SHAPES,
    VoxyL0Model,
    VoxyL1Model,
    VoxyL2Model,
    VoxyL3Model,
    VoxyL4Model,
    VoxyModelConfig,
    create_model,
)

LOGGER = logging.getLogger("voxy_export")

# RouterField names matching Java enum ordinals 0–14
ROUTER_FIELD_NAMES = [
    "TEMPERATURE",
    "VEGETATION",
    "CONTINENTS",
    "EROSION",
    "DEPTH",
    "RIDGES",
    "PRELIMINARY_SURFACE_LEVEL",
    "FINAL_DENSITY",
    "BARRIER",
    "FLUID_LEVEL_FLOODEDNESS",
    "FLUID_LEVEL_SPREAD",
    "LAVA",
    "VEIN_TOGGLE",
    "VEIN_RIDGED",
    "VEIN_GAP",
]

# Per-level noise channel selection (RouterField ordinals)
LEVEL_NOISE_CHANNELS: Dict[int, List[int]] = {
    0: list(range(15)),  # all 15
    1: list(range(15)),  # all 15
    2: L2_NOISE_CHANNELS,  # [0,1,2,3,4,5,7]
    3: L3_NOISE_CHANNELS,  # [0,1,2,3,4,5]
    4: L4_NOISE_CHANNELS,  # [0,1,2,3,4,5]
}


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
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
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
    vocab_path = Path(__file__).resolve().parent.parent.parent / "config" / "voxy_vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            block_mapping = json.load(f)
        config_dict["block_mapping"] = block_mapping
        config_dict["block_id_to_name"] = {v: k for k, v in block_mapping.items()}
        LOGGER.info("Embedded %d block mappings from %s", len(block_mapping), vocab_path)
    else:
        LOGGER.warning("No block vocabulary found at %s", vocab_path)
    return config_dict


# ─── ONNX adapter wrappers ──────────────────────────────────────────
#
# Each adapter converts the model's Dict output into a flat tuple of
# tensors that torch.onnx.export can serialise.


class VoxyL0Adapter(nn.Module):
    """ONNX adapter for L0 (finest level, no occupancy head).

    Inputs:  noise_3d[B,15,8,4,8], biome_3d[B,8,4,8],
             y_position[B], parent_blocks[B,32,32,32]
    Outputs: block_logits[B,V,32,32,32]
    """

    def __init__(self, model: VoxyL0Model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noise_3d: torch.Tensor,
        biome_3d: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(noise_3d, biome_3d, y_position, parent_blocks)
        return out["block_logits"]


class VoxyL1Adapter(nn.Module):
    """ONNX adapter for L1 (3D noise, with occupancy).

    Inputs:  noise_3d[B,15,16,8,16], biome_3d[B,16,8,16],
             y_position[B], parent_blocks[B,32,32,32]
    Outputs: block_logits[B,V,32,32,32], occ_logits[B,8]
    """

    def __init__(self, model: VoxyL1Model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noise_3d: torch.Tensor,
        biome_3d: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(noise_3d, biome_3d, y_position, parent_blocks)
        return out["block_logits"], out["occ_logits"]


class VoxyL2Adapter(nn.Module):
    """ONNX adapter for L2 (2D climate, 7ch, with occupancy).

    Inputs:  climate_2d[B,7,8,8], biome_2d[B,8,8],
             y_position[B], parent_blocks[B,32,32,32]
    Outputs: block_logits[B,V,32,32,32], occ_logits[B,8]
    """

    def __init__(self, model: VoxyL2Model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(climate_2d, biome_2d, y_position, parent_blocks)
        return out["block_logits"], out["occ_logits"]


class VoxyL3Adapter(nn.Module):
    """ONNX adapter for L3 (2D climate, 6ch, with occupancy).

    Inputs:  climate_2d[B,6,8,8], biome_2d[B,8,8],
             y_position[B], parent_blocks[B,32,32,32]
    Outputs: block_logits[B,V,32,32,32], occ_logits[B,8]
    """

    def __init__(self, model: VoxyL3Model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        y_position: torch.Tensor,
        parent_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(climate_2d, biome_2d, y_position, parent_blocks)
        return out["block_logits"], out["occ_logits"]


class VoxyL4Adapter(nn.Module):
    """ONNX adapter for L4 (root level, 2D climate, 6ch, no parent).

    Inputs:  climate_2d[B,6,8,8], biome_2d[B,8,8], y_position[B]
    Outputs: block_logits[B,V,24,32,32], occ_logits[B,8]
    """

    def __init__(self, model: VoxyL4Model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        climate_2d: torch.Tensor,
        biome_2d: torch.Tensor,
        y_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(climate_2d, biome_2d, y_position)
        return out["block_logits"], out["occ_logits"]


ADAPTER_CLASSES = {
    0: VoxyL0Adapter,
    1: VoxyL1Adapter,
    2: VoxyL2Adapter,
    3: VoxyL3Adapter,
    4: VoxyL4Adapter,
}


# ─── Dummy input generation ─────────────────────────────────────────


def _make_dummy_inputs(level: int, cfg: VoxyModelConfig) -> Tuple[tuple, List[str], Dict[str, str]]:
    """Build dummy inputs, input names, and dtype map for a level.

    Returns (dummy_tuple, input_names, input_dtypes).
    """
    noise_shape = NOISE_SHAPES[level]
    biome_shape = BIOME_SHAPES[level]

    if level <= 1:
        # 3D noise + biome
        noise = torch.randn(1, *noise_shape)
        biome = torch.randint(0, cfg.biome_vocab_size, (1, *biome_shape), dtype=torch.long)
        y_pos = torch.tensor([12], dtype=torch.long)
        parent = torch.randint(0, cfg.block_vocab_size, (1, 32, 32, 32), dtype=torch.long)
        return (
            (noise, biome, y_pos, parent),
            ["noise_3d", "biome_3d", "y_position", "parent_blocks"],
            {
                "noise_3d": "float32",
                "biome_3d": "int64",
                "y_position": "int64",
                "parent_blocks": "int64",
            },
        )
    elif level == 4:
        # 2D climate, no parent
        climate = torch.randn(1, *noise_shape)
        biome = torch.randint(0, cfg.biome_vocab_size, (1, *biome_shape), dtype=torch.long)
        y_pos = torch.tensor([12], dtype=torch.long)
        return (
            (climate, biome, y_pos),
            ["climate_2d", "biome_2d", "y_position"],
            {
                "climate_2d": "float32",
                "biome_2d": "int64",
                "y_position": "int64",
            },
        )
    else:
        # L2, L3: 2D climate + parent
        climate = torch.randn(1, *noise_shape)
        biome = torch.randint(0, cfg.biome_vocab_size, (1, *biome_shape), dtype=torch.long)
        y_pos = torch.tensor([12], dtype=torch.long)
        parent = torch.randint(0, cfg.block_vocab_size, (1, 32, 32, 32), dtype=torch.long)
        return (
            (climate, biome, y_pos, parent),
            ["climate_2d", "biome_2d", "y_position", "parent_blocks"],
            {
                "climate_2d": "float32",
                "biome_2d": "int64",
                "y_position": "int64",
                "parent_blocks": "int64",
            },
        )


# ─── Generic per-level export ───────────────────────────────────────


def export_level(
    level: int,
    model: nn.Module,
    cfg: VoxyModelConfig,
    out_dir: Path,
    *,
    validate: bool = True,
) -> Path:
    """Export one Voxy level to ONNX + sidecar config + test vectors.

    Returns the path to the exported .onnx file.
    """
    adapter_cls = ADAPTER_CLASSES[level]
    adapter = adapter_cls(model)
    adapter.eval()

    onnx_filename = f"voxy_l{level}.onnx"
    onnx_path = out_dir / onnx_filename

    # Build dummy inputs
    dummy, input_names, input_dtypes = _make_dummy_inputs(level, cfg)

    # Determine output names
    has_occ = level > 0
    output_names = ["block_logits", "occ_logits"] if has_occ else ["block_logits"]

    # Dynamic batch axes
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    # Export
    with torch.no_grad():
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
    LOGGER.info("Exported voxy_l%d → %s", level, onnx_path)

    # Run adapter to get reference outputs for test vectors and shape discovery
    with torch.no_grad():
        raw_out = adapter(*dummy)

    if has_occ:
        block_logits, occ_logits = raw_out
    else:
        block_logits = raw_out
        occ_logits = None

    # ── Sidecar config ────────────────────────────────────────────
    V = cfg.block_vocab_size
    noise_channels = LEVEL_NOISE_CHANNELS[level]
    is_3d = level <= 1

    input_shapes: Dict[str, List[int]] = {}
    for name, tensor in zip(input_names, dummy):
        input_shapes[name] = list(tensor.shape)

    output_shapes: Dict[str, List[int]] = {
        "block_logits": list(block_logits.shape),
    }
    if occ_logits is not None:
        output_shapes["occ_logits"] = list(occ_logits.shape)

    model_config: Dict[str, Any] = {
        "version": "6.0.0",
        "contract": "lodiffusion.v6.voxy",
        "model": f"voxy_l{level}",
        "level": level,
        # Noise routing info for Java side
        "noise_encoding": "3d_native" if is_3d else "2d_climate",
        "noise_channels": noise_channels,
        "noise_channel_names": [ROUTER_FIELD_NAMES[i] for i in noise_channels],
        "has_parent": level < 4,
        "has_occupancy": has_occ,
        # I/O contract
        "inputs": input_shapes,
        "input_dtypes": input_dtypes,
        "outputs": output_shapes,
        "output_resolution": 32,
        "dynamic_batch": True,
        # Vocab sizes
        "block_vocab_size": V,
        "biome_vocab_size": cfg.biome_vocab_size,
        "y_vocab_size": cfg.y_vocab_size,
        # Architecture metadata
        "channels": list(getattr(cfg, f"l{level}_channels")),
        "bottleneck_extra": getattr(cfg, f"l{level}_bottleneck_extra"),
        "assumptions": {
            "y_position_range": [0, cfg.y_vocab_size - 1],
            "parent_blocks": (
                ("int64 block IDs [0, block_vocab_size); " "embedding is baked into ONNX graph")
                if level < 4
                else None
            ),
            "noise_format": ("float32, raw noise router values (not normalized)"),
            "biome_format": "int64, Minecraft biome registry IDs",
        },
        "provenance": collect_export_provenance(),
    }

    # Add 2D-specific hints for Java noise preparation
    if not is_3d:
        model_config["noise_2d_preparation"] = {
            "description": (
                "Select noise_channels from 15-field router output, "
                "collapse Y dimension (mean), subsample XZ to 8×8"
            ),
            "y_collapse": "mean",
            "xz_subsample_to": [8, 8],
        }

    model_config = embed_block_mapping(model_config)

    config_path = out_dir / f"voxy_l{level}_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    LOGGER.info("Sidecar config: %s", config_path)

    # ── Test vectors ──────────────────────────────────────────────
    vectors: Dict[str, np.ndarray] = {}
    for name, tensor in zip(input_names, dummy):
        vectors[name] = tensor.numpy()
    vectors["block_logits"] = block_logits.numpy()
    if occ_logits is not None:
        vectors["occ_logits"] = occ_logits.numpy()

    vectors_path = out_dir / f"voxy_l{level}_test_vectors.npz"
    np.savez(str(vectors_path), **vectors)
    LOGGER.info("Test vectors: %s", vectors_path)

    # ── Validate ──────────────────────────────────────────────────
    if validate:
        ok = _validate_onnx(onnx_path, vectors_path)
        if not ok:
            LOGGER.error("ONNX validation FAILED for L%d!", level)
        else:
            LOGGER.info("ONNX validation OK for L%d", level)

    return onnx_path


# ─── ONNX round-trip validation ──────────────────────────────────────


def _validate_onnx(onnx_path: Path, test_vectors_path: Path) -> bool:
    """Validate ONNX model round-trip accuracy against test vectors."""
    try:
        import onnxruntime as ort
    except ImportError:
        LOGGER.warning("onnxruntime not installed — skipping ONNX validation")
        return True

    vectors = np.load(test_vectors_path)
    session = ort.InferenceSession(str(onnx_path))

    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if name in vectors:
            feed[name] = vectors[name]
        else:
            LOGGER.warning("Missing test vector for ONNX input '%s'", name)
            return False

    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, feed)

    all_ok = True
    for oname, result in zip(output_names, results):
        if oname in vectors:
            expected = vectors[oname]
            max_diff = float(np.max(np.abs(result - expected)))
            if max_diff > 1e-3:
                LOGGER.error(
                    "  %s output '%s': FAILED max_diff=%.6f",
                    onnx_path.name,
                    oname,
                    max_diff,
                )
                all_ok = False
            else:
                LOGGER.info(
                    "  %s output '%s': OK max_diff=%.6f",
                    onnx_path.name,
                    oname,
                    max_diff,
                )
    return all_ok


# ─── Checkpoint loading ──────────────────────────────────────────────


def load_voxy_checkpoint(
    checkpoint_path: Path,
) -> Tuple[int, VoxyModelConfig, nn.Module]:
    """Load a per-level Voxy checkpoint.

    Returns ``(level, config, model)`` with weights loaded.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    level: int = ckpt["level"]
    cfg: VoxyModelConfig = ckpt["config"]
    LOGGER.info("Checkpoint L%d: %s", level, checkpoint_path)

    model = create_model(level, cfg)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if result.unexpected_keys:
        LOGGER.warning("Unexpected keys: %s", result.unexpected_keys)

    return level, cfg, model


# ─── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Voxy models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing voxy_L{N}.pt files (default: checkpoints)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("production"),
        help="Output directory for ONNX + configs (default: production)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Which levels to export (default: all 5)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip onnxruntime round-trip validation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in sorted(args.levels):
        ckpt_path = args.checkpoint_dir / f"voxy_L{level}.pt"
        if not ckpt_path.exists():
            LOGGER.error("Checkpoint not found: %s — skipping L%d", ckpt_path, level)
            continue

        lvl, cfg, model = load_voxy_checkpoint(ckpt_path)
        assert lvl == level, f"Checkpoint level {lvl} != requested {level}"

        export_level(
            level,
            model,
            cfg,
            out_dir,
            validate=not args.no_validate,
        )
        LOGGER.info("")  # blank line between levels

    LOGGER.info("Done. Artefacts in %s", out_dir)


if __name__ == "__main__":
    main()
