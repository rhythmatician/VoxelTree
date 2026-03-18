"""Export a trained SparseOctreeModel checkpoint to ONNX + sidecar config.

Usage
-----
  python voxel_tree/tasks/sparse_octree/export_sparse_octree.py \\
      --checkpoint path/to/sparse_octree_best.pt \\
      --out-dir    LODiffusion/run/models \\
      [--n2d 0] [--n3d 15] [--hidden 128] [--num-classes 1040] [--spatial-y 4]

The script writes:
  <out-dir>/sparse_octree.onnx          (ONNX model)
  <out-dir>/sparse_octree_config.json   (sidecar for SparseOctreeModelRunner)

ONNX contract: ``lodiffusion.v7.sparse_octree``
---------------------------------------------
Input:
  noise_3d   float32[1, n3d, 4, spatial_y, 4]   (n2d=0 → no noise_2d input)

Outputs (10 tensors, levels 4 down to 0):
  split_L4   float32[1,    1]     split logit at L4 root
  label_L4   float32[1,    1, C]  block-class logits
  split_L3   float32[1,    8]
  label_L3   float32[1,    8, C]
  split_L2   float32[1,   64]
  label_L2   float32[1,   64, C]
  split_L1   float32[1,  512]
  label_L1   float32[1,  512, C]
  split_L0   float32[1, 4096]
  label_L0   float32[1, 4096, C]

The Java decoder (SparseOctreeModelRunner) identifies tensors by shape, so
the output name order is informational only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the voxel_tree package root is importable when run as a standalone script
# ---------------------------------------------------------------------------
_VT_ROOT = Path(__file__).resolve().parents[3]  # …/VoxelTree
if str(_VT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VT_ROOT))

from voxel_tree.tasks.sparse_octree.sparse_octree import (
    SparseOctreeFastModel,
    SparseOctreeModel,
)  # noqa: E402


# ---------------------------------------------------------------------------
# Channel name registry  (must stay in sync with WorldNoiseAccess.NOISE_3D_PATHS)
# ---------------------------------------------------------------------------

# Canonical names for the 15 v7 RouterField channels, in index order.
# Must stay in sync with ``router_field.py`` and Java ``RouterField`` enum.
_V7_NOISE_3D_CHANNELS: list[str] = [
    "temperature",               # 0
    "vegetation",                # 1
    "continents",                # 2
    "erosion",                   # 3
    "depth",                     # 4
    "ridges",                    # 5
    "preliminary_surface_level",  # 6
    "final_density",             # 7
    "barrier",                   # 8
    "fluid_level_floodedness",   # 9
    "fluid_level_spread",        # 10
    "lava",                      # 11
    "vein_toggle",               # 12
    "vein_ridged",               # 13
    "vein_gap",                  # 14
]

# Legacy 13-channel names for backward compatibility.
# @deprecated: Use _V7_NOISE_3D_CHANNELS (15 fields) for new models.
_LEGACY_NOISE_3D_CHANNELS: list[str] = [
    "offset",            # 0  overworld/offset
    "factor",            # 1  overworld/factor
    "jaggedness",        # 2  overworld/jaggedness
    "depth",             # 3  router.depth()
    "sloped_cheese",     # 4  overworld/sloped_cheese
    "y",                 # 5  cell-centre Y in blocks
    "entrances",         # 6  overworld/caves/entrances
    "cheese_caves",      # 7  overworld/caves/pillars
    "spaghetti_2d",      # 8  overworld/caves/spaghetti_2d
    "roughness",         # 9  overworld/caves/spaghetti_roughness_function
    "noodle",            # 10 overworld/caves/noodle
    "base_3d_noise",     # 11 overworld/base_3d_noise
    "final_density",     # 12 router.finalDensity()
]


# ---------------------------------------------------------------------------
# ONNX-compatible wrapper
# ---------------------------------------------------------------------------


def _flatten_outputs(
    out: dict[int, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    """Flatten model outputs in the expected deterministic order.

    The sparse root model produces a dict mapping level → {'split', 'label'}.
    ONNX export expects a flat tuple of tensors in the order:
    split_L4, label_L4, ..., split_L0, label_L0.
    """

    result: list[torch.Tensor] = []
    for lvl in range(4, -1, -1):  # L4 → L0
        result.append(out[lvl]["split"])
        result.append(out[lvl]["label"])
    return tuple(result)


class _OnnxWrapperWith2d(nn.Module):
    """Wraps SparseOctreeModel for export when noise_2d is present."""

    def __init__(self, model: nn.Module, spatial_y: int = 4) -> None:
        super().__init__()
        self.model = model
        self.spatial_y = spatial_y

    def forward(
        self, noise_2d: torch.Tensor, noise_3d: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        B = noise_2d.shape[0]
        biome_ids = torch.zeros((B, 4, self.spatial_y, 4), dtype=torch.long, device=noise_2d.device)
        out = self.model(noise_2d, noise_3d, biome_ids)
        return _flatten_outputs(out)


class _OnnxWrapperNo2d(nn.Module):
    """Wraps SparseOctreeModel for export when no noise_2d input exists."""

    def __init__(self, model: nn.Module, spatial_y: int = 4) -> None:
        super().__init__()
        self.model = model
        self.spatial_y = spatial_y

    def forward(self, noise_3d: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Create an empty noise_2d tensor to satisfy the underlying model signature.
        # The model treats zero-element noise_2d as a no-op.
        B = noise_3d.shape[0]
        noise_2d = torch.zeros(
            (B, 0, 4, 4), dtype=noise_3d.dtype, device=noise_3d.device
        )
        biome_ids = torch.zeros((B, 4, self.spatial_y, 4), dtype=torch.long, device=noise_3d.device)
        out = self.model(noise_2d, noise_3d, biome_ids)
        return _flatten_outputs(out)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _infer_num_classes(state_dict: dict[str, Any]) -> int:
    """Infer num_classes from label_head.bias shape in a state dict."""
    key = "label_head.bias"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    key_fast = "label_head.out_proj.bias"
    if key_fast in state_dict:
        return int(state_dict[key_fast].shape[0])
    # Fallback: search for label_head.weight
    key2 = "label_head.weight"
    if key2 in state_dict:
        return int(state_dict[key2].shape[0])
    key2_fast = "label_head.out_proj.weight"
    if key2_fast in state_dict:
        return int(state_dict[key2_fast].shape[0])
    return -1


def _infer_model_variant(state_dict: dict[str, Any]) -> str:
    """Infer baseline vs fast sparse-root checkpoint layout."""
    if any(k.startswith("level_mod.") for k in state_dict):
        return "fast"
    if (
        "label_head.out_proj.weight" in state_dict
        or "child_proj.out_proj.weight" in state_dict
    ):
        return "fast"
    return "baseline"


# Default block vocabulary shipped alongside this script.
# Prefer the Voxy canonical vocabulary (alphabetical) used during training.
# Fall back to standard_minecraft_blocks.json if voxy_vocab.json is absent.
_VOXY_VOCAB = _VT_ROOT / "voxel_tree" / "config" / "voxy_vocab.json"
_STANDARD_BLOCK_VOCAB = (
    Path(__file__).resolve().parent.parent / "standard_minecraft_blocks.json"
)
_DEFAULT_BLOCK_VOCAB = _VOXY_VOCAB if _VOXY_VOCAB.exists() else _STANDARD_BLOCK_VOCAB


def _load_block_vocab(path: "Path | None") -> dict[str, int]:
    """Load a block-name→index mapping from *path*.  Returns {} on failure."""
    p = path if path is not None else _DEFAULT_BLOCK_VOCAB
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                print(f"[export] Loaded block vocab from {p} ({len(data)} entries)")
                return data
        except Exception as exc:  # noqa: BLE001
            print(f"[export] Warning: could not load block vocab from {p}: {exc}")
    else:
        print(
            f"[export] Warning: block vocab not found at {p} — blockMapping will be empty"
        )
    return {}


def export_sparse_octree(
    checkpoint: Path,
    out_dir: Path,
    *,
    n2d: int = 0,
    n3d: int = 15,
    hidden: int = 72,
    num_classes: int = -1,
    model_variant: str | None = None,
    split_threshold: float = 0.6,
    softmax_temperature: float = 0.8,
    block_vocab: Path | None = None,
    spatial_y: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "sparse_octree.onnx"
    config_path = out_dir / "sparse_octree_config.json"

    # ── Load checkpoint ────────────────────────────────────────────────
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # Unwrap if saved as {"model": state_dict, ...}
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    # Auto-detect num_classes from the saved weights
    if num_classes <= 0:
        num_classes = _infer_num_classes(state)
    if num_classes <= 0:
        raise ValueError(
            "Cannot determine num_classes from checkpoint; "
            "pass --num-classes explicitly."
        )

    if model_variant is None:
        model_variant = _infer_model_variant(state)

    print(
        f"[export] variant={model_variant}  n2d={n2d}  n3d={n3d}  "
        f"hidden={hidden}  num_classes={num_classes}  spatial_y={spatial_y}"
    )

    # ── Build model ────────────────────────────────────────────────────────
    if model_variant == "fast":
        model: nn.Module = SparseOctreeFastModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
            spatial_y=spatial_y,
        )
    else:
        model = SparseOctreeModel(
            n2d=n2d,
            n3d=n3d,
            hidden=hidden,
            num_classes=num_classes,
            spatial_y=spatial_y,
        )
    # ── Compat-patch pre-biome checkpoints ────────────────────────────
    # Old checkpoints have mlp.0.weight shape [H*2, flat_noise] without the
    # biome columns.  Pad with zeros so biome contributes nothing to the output
    # until a fresh retrain populates the weights properly.
    _m: Any = model  # local Any alias — model IS a concrete subclass at runtime
    _mlp_w_key = "noise_enc.mlp.0.weight"
    _biome_key = "noise_enc.biome_embed.weight"
    if _mlp_w_key in state:
        expected_in = int(_m.noise_enc.mlp[0].in_features)
        actual_in = state[_mlp_w_key].shape[1]
        if actual_in < expected_in:
            pad_cols = expected_in - actual_in
            print(
                f"[export] Pre-biome checkpoint detected -- padding {_mlp_w_key} "
                f"({actual_in} -> {expected_in} input cols with zeros)"
            )
            state[_mlp_w_key] = torch.cat(
                [state[_mlp_w_key], torch.zeros(state[_mlp_w_key].shape[0], pad_cols)],
                dim=1,
            )
    if _biome_key not in state:
        state[_biome_key] = torch.zeros(
            int(_m.noise_enc.biome_embed.num_embeddings),
            int(_m.noise_enc.biome_embed.embedding_dim),
        )

    missing, unexpected = model.load_state_dict(
        state, strict=True
    )  # now strict is safe
    model.eval()

    # Choose the appropriate ONNX wrapper based on whether noise_2d is used.
    if n2d > 0:
        wrapper: nn.Module = _OnnxWrapperWith2d(model, spatial_y=spatial_y).eval()
    else:
        wrapper = _OnnxWrapperNo2d(model, spatial_y=spatial_y).eval()

    # ── Dummy inputs ───────────────────────────────────────────────────────
    dummy_3d = torch.zeros(1, n3d, 4, spatial_y, 4)
    if n2d > 0:
        dummy_2d = torch.zeros(1, n2d, 4, 4)

    output_names = [
        "split_L4",
        "label_L4",
        "split_L3",
        "label_L3",
        "split_L2",
        "label_L2",
        "split_L1",
        "label_L1",
        "split_L0",
        "label_L0",
    ]

    if n2d > 0:
        input_names = ["noise_2d", "noise_3d"]
    else:
        input_names = ["noise_3d"]

    # Dynamic shapes: batch dimension is dynamic on inputs.
    # (Current torch.export only allows arg-name keys, not outputs.)
    dynamic_shapes: dict[str, dict[int, str]] = {
        "noise_3d": {0: "batch"},
    }
    if n2d > 0:
        dynamic_shapes["noise_2d"] = {0: "batch"}

    # ── ONNX export ────────────────────────────────────────────────────
    # Tracing the model first avoids onnxscript type-hint issues in torch 2.x.
    print(f"[export] Exporting to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_2d, dummy_3d) if n2d > 0 else (dummy_3d,),
            str(onnx_path),
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            do_constant_folding=True,
            export_params=True,
        )
    print(f"[export] ONNX saved → {onnx_path}")

    # ── Consolidate into a single self-contained .onnx file ────────────
    # torch.onnx.export may split large weights into a sidecar .data file.
    # Re-load and re-save with save_as_external_data=False so the runtime
    # only needs the one .onnx file (avoids ORT file_size errors on deploy).
    model_proto = onnx.load(str(onnx_path))
    onnx.save(model_proto, str(onnx_path), save_as_external_data=False)
    data_path = onnx_path.with_name(onnx_path.name + ".data")
    if data_path.exists():
        data_path.unlink()
        print(f"[export] Removed external data file: {data_path.name}")
    print(
        f"[export] Model consolidated to single file ({onnx_path.stat().st_size // 1024} KB)"
    )

    # ── Sidecar config ─────────────────────────────────────────────────
    # ModelConfig fields understood by SparseOctreeModelRunner / ConfigLoader:
    #   version, blockVocabSize, blockMapping (can be empty)

    # Determine contract version and channel names based on spatial_y
    if spatial_y == 4 and n3d == 15:
        contract = "lodiffusion.v7.sparse_octree"
        channel_names = _V7_NOISE_3D_CHANNELS
    elif n3d == len(_LEGACY_NOISE_3D_CHANNELS):
        contract = "lodiffusion.v6.sparse_octree"
        channel_names = _LEGACY_NOISE_3D_CHANNELS
    else:
        contract = f"lodiffusion.custom.sparse_octree_n3d{n3d}_sy{spatial_y}"
        channel_names = [f"ch_{i}" for i in range(n3d)]

    config = {
        "modelName": "sparse_octree",
        "version": contract,
        "contract": contract,
        "blockVocabSize": num_classes,
        "blockMapping": _load_block_vocab(block_vocab),
        "noise3dChannels": channel_names,
        "spatialY": spatial_y,
        "splitThreshold": split_threshold,
        "softmaxTemperature": softmax_temperature,
        "inputs": {
            **({"noise_2d": [1, n2d, 4, 4]} if n2d > 0 else {}),
            "noise_3d": [1, n3d, 4, spatial_y, 4],
        },
        "outputs": {
            "split_L4": [1, 1],
            "label_L4": [1, 1, num_classes],
            "split_L3": [1, 8],
            "label_L3": [1, 8, num_classes],
            "split_L2": [1, 64],
            "label_L2": [1, 64, num_classes],
            "split_L1": [1, 512],
            "label_L1": [1, 512, num_classes],
            "split_L0": [1, 4096],
            "label_L0": [1, 4096, num_classes],
        },
    }
    # If we are exporting a model with n2d==0, ensure the sidecar does not
    # declare a noise_2d input (some runtimes cannot create zero-length tensors).
    if n2d == 0:
        inputs = config.get("inputs")
        if isinstance(inputs, dict):
            inputs.pop("noise_2d", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[export] Config saved → {config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export SparseOctreeModel checkpoint to ONNX",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to .pt file saved by train_sparse_octree()",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory (will be created).  "
        "Writes sparse_octree.onnx and sparse_octree_config.json here.",
    )
    p.add_argument(
        "--n2d", type=int, default=0, help="Number of 2-D noise channels (default: 0)"
    )
    p.add_argument(
        "--n3d", type=int, default=15, help="Number of 3-D noise channels (default: 15)"
    )
    p.add_argument(
        "--hidden", type=int, default=72, help="Model hidden size (default: 72)"
    )
    p.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Model variant to export: baseline or fast (default: auto-detect from checkpoint)",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=-1,
        help="Block vocab size (auto-detected from checkpoint if omitted)",
    )
    p.add_argument(
        "--split-threshold",
        type=float,
        default=0.6,
        help="sigmoid(split_logit) > threshold → expand node in octree (default: 0.6)",
    )
    p.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.8,
        help="Divide label logits by this before argmax; <1.0 sharpens selection (default: 0.8)",
    )
    p.add_argument(
        "--block-vocab",
        type=Path,
        default=None,
        help="Path to a JSON block-name→index mapping (default: auto-detect "
        "standard_minecraft_blocks.json next to this script)",
    )
    p.add_argument(
        "--spatial-y",
        type=int,
        default=4,
        help="Y-axis quart cells per section (4 for v7, 2 for legacy; default: 4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_sparse_octree(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        n2d=args.n2d,
        n3d=args.n3d,
        hidden=args.hidden,
        num_classes=args.num_classes,
        model_variant=args.model_variant,
        split_threshold=args.split_threshold,
        softmax_temperature=args.softmax_temperature,
        block_vocab=args.block_vocab,
        spatial_y=args.spatial_y,
    )
