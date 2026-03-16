"""step_definitions.py — Pipeline step metadata and CLI command factories.

Each StepDef knows:
  - its ID, display label, and which steps must succeed before it can run
  - how to build the CLI command list given a loaded profile dict
  - whether it is currently enabled (loopback stubs start disabled)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class StepDef:
    id: str
    label: str
    prereqs: list[str]
    cmd_factory: Callable[[dict], list[str]]
    enabled: bool = True
    """False → rendered as a faded stub in the dashboard (future loopback steps)."""
    server_required: bool = False
    """True → step needs the Fabric server running (sends RCON commands)."""


# ---------------------------------------------------------------------------
# Helper — resolve the Python interpreter inside the current venv (if any)
# ---------------------------------------------------------------------------


def _python() -> str:
    return sys.executable


def _vt_root() -> Path:
    """Absolute path to the VoxelTree repo root (3 levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Command factories
# Each receives the *full* profile dict and returns a list[str] suitable for
# subprocess.Popen.  Commands are run with cwd=VoxelTree root.
# ---------------------------------------------------------------------------


def _pregen_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "pregen",
        "--radius",
        str(world.get("radius", 2048)),
        "--password",
        str(rcon["password"]),
        "--host",
        str(rcon["host"]),
        "--port",
        str(rcon["port"]),
    ]
    return cmd


def _voxy_import_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
    # timeout is still profile-level (it's not a server property)
    rcon_timeout = p.get("rcon", {}).get("timeout", 300)
    return [
        _python(),
        "-m",
        "VoxelTree",
        "voxy-import",
        "--world-name",
        str(world.get("save_name", "New World")),
        "--password",
        str(rcon["password"]),
        "--host",
        str(rcon["host"]),
        "--port",
        str(rcon["port"]),
        "--timeout",
        str(rcon_timeout),
    ]


def _dumpnoise_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
    return [
        _python(),
        "-m",
        "VoxelTree",
        "dumpnoise",
        "--radius",
        str(world.get("radius", 2048)),
        "--password",
        str(rcon["password"]),
        "--host",
        str(rcon["host"]),
        "--port",
        str(rcon["port"]),
    ]


def _extract_octree_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "dataprep",
        "--from-step",
        "extract-octree",
        "--voxy-dir",
        str(data.get("voxy_dir", "../LODiffusion/run/saves")),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if data.get("max_sections"):
        cmd += ["--max-sections", str(data["max_sections"])]
    if data.get("min_solid"):
        cmd += ["--min-solid", str(data["min_solid"])]
    return cmd


def _column_heights_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "dataprep",
        "--from-step",
        "column-heights-octree",
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if data.get("noise_dump_dir"):
        cmd += ["--noise-dump-dir", str(data["noise_dump_dir"])]
    return cmd


def _build_pairs_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "dataprep",
        "--from-step",
        "build-octree-pairs",
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    cmd += ["--val-split", str(data.get("val_split", 0.1))]
    return cmd


def _build_pairs_model_cmd(p: dict, model_type: str) -> list[str]:
    """Build the pair cache for a specific model type (init/refine/leaf)."""
    data = p.get("data", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "build-pairs",
        "--model-type",
        model_type,
        "--val-split",
        str(data.get("val_split", 0.1)),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    return cmd


def _build_pairs_init_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "init")


def _build_pairs_refine_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "refine")


def _build_pairs_leaf_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "leaf")


def _train_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "pipeline",
        "train",
        "--epochs",
        str(train.get("epochs", 20)),
        "--batch-size",
        str(train.get("batch_size", 4)),
        "--lr",
        str(train.get("lr", 1e-4)),
        "--device",
        str(train.get("device", "auto")),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if train.get("output_dir"):
        cmd += ["--model-dir", str(train["output_dir"])]
    return cmd


def _train_model_cmd(p: dict, model_type: str) -> list[str]:
    """Train a specific model type independently."""
    data = p.get("data", {})
    train = p.get("train", {})
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "train",
        "--models",
        model_type,
        "--epochs",
        str(train.get("epochs", 20)),
        "--batch-size",
        str(train.get("batch_size", 4)),
        "--lr",
        str(train.get("lr", 1e-4)),
        "--device",
        str(train.get("device", "auto")),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if train.get("output_dir"):
        cmd += ["--output-dir", str(train["output_dir"])]
    return cmd


def _train_init_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "init")


def _train_refine_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "refine")


def _train_leaf_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "leaf")


def _export_cmd(p: dict) -> list[str]:
    """Generic export command (all models)."""
    train = p.get("train", {})
    export = p.get("export", {})
    model_dir = train.get("output_dir", "models/voxy_octree")
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "pipeline",
        "export",
        "--checkpoint-dir",
        str(model_dir),
    ]
    if export.get("output_dir"):
        cmd += ["--export-dir", str(export["output_dir"])]
    # the CLI now supports --models to restrict which submodels are exported
    models = export.get("models")
    if models:
        # expect a list of strings like ["init","refine"]
        cmd += ["--models"] + list(models)
    return cmd


def _deploy_cmd(p: dict) -> list[str]:
    """Generic deploy command (all models)."""
    export = p.get("export", {})
    deploy = p.get("deploy", {})
    export_dir = export.get("output_dir", "production")
    cmd = [_python(), "-m", "VoxelTree", "pipeline", "deploy", str(export_dir)]
    if deploy.get("target_dir"):
        cmd += ["--dest", str(deploy["target_dir"])]
    # optional filtering of models
    models = deploy.get("models")
    if models:
        cmd += ["--models"] + list(models)
    return cmd


# convenience wrappers for tests and step definitions


def _export_init_cmd(p: dict) -> list[str]:
    return _export_cmd({**p, "export": {**p.get("export", {}), "models": ["init"]}})


def _export_refine_cmd(p: dict) -> list[str]:
    return _export_cmd({**p, "export": {**p.get("export", {}), "models": ["refine"]}})


def _export_leaf_cmd(p: dict) -> list[str]:
    return _export_cmd({**p, "export": {**p.get("export", {}), "models": ["leaf"]}})


def _deploy_init_cmd(p: dict) -> list[str]:
    return _deploy_cmd({**p, "deploy": {**p.get("deploy", {}), "models": ["init"]}})


def _deploy_refine_cmd(p: dict) -> list[str]:
    return _deploy_cmd({**p, "deploy": {**p.get("deploy", {}), "models": ["refine"]}})


def _deploy_leaf_cmd(p: dict) -> list[str]:
    return _deploy_cmd({**p, "deploy": {**p.get("deploy", {}), "models": ["leaf"]}})


# ---------------------------------------------------------------------------
# Stage 1 Density (tiny NN) helpers
# ---------------------------------------------------------------------------


def _train_stage1_density_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    cmd = [_python(), "tools/train_stage1_density.py"]
    if data.get("stage1_dump_dir"):
        cmd += ["--data-dir", str(data["stage1_dump_dir"])]
    if train.get("output_dir"):
        cmd += ["--out-dir", str(train["output_dir"])]
    if train.get("epochs") is not None:
        cmd += ["--epochs", str(train["epochs"])]
    if train.get("batch_size") is not None:
        cmd += ["--batch-size", str(train["batch_size"])]
    if train.get("lr") is not None:
        cmd += ["--lr", str(train["lr"])]
    if train.get("target_mse") is not None:
        cmd += ["--target-mse", str(train["target_mse"])]
    return cmd


def _train_terrain_shaper_cmd(_: dict) -> list[str]:
    """Run the terrain shaper trainer script (uses fixed internal settings)."""
    return [_python(), "tools/train_terrain_shaper.py"]


def _extract_stage1_weights_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    extract = p.get("extract", {})
    cmd = [_python(), "tools/extract_stage1_weights.py"]
    # model directory generally matches train.output_dir
    model_dir = train.get("output_dir")
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    # output dir can be overridden
    if extract.get("output_dir"):
        cmd += ["--out-dir", str(extract["output_dir"])]
    return cmd


def _distill_density_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    distill = p.get("distill", {})
    cmd = [
        _python(),
        "scripts/distill_density_nn.py",
        "--teacher",
        str(distill.get("teacher", "unet")),
        "--student",
        str(distill.get("student", "sep")),
        "--epochs",
        str(distill.get("epochs", 120)),
        "--alpha",
        str(distill.get("alpha", 0.5)),
        "--lr",
        str(distill.get("lr", 2e-3)),
        "--device",
        str(train.get("device", "auto")),
    ]
    return cmd


# Future loopback stubs (enabled=False → rendered faded, not clickable)
def _reset_data_cmd(p: dict) -> list[str]:
    return ["echo", "[loopback] reset_data not yet implemented"]


def _new_seed_cmd(p: dict) -> list[str]:
    return ["echo", "[loopback] new_seed not yet implemented"]


# ---------------------------------------------------------------------------
# Ordered step list
# ---------------------------------------------------------------------------

PIPELINE_STEPS: list[StepDef] = [
    StepDef(
        id="pregen",
        label="Pregen",
        prereqs=[],
        cmd_factory=_pregen_cmd,
        server_required=True,
    ),
    StepDef(
        id="voxy_import",
        label="Voxy",
        prereqs=["pregen"],
        cmd_factory=_voxy_import_cmd,
        server_required=True,
    ),
    StepDef(
        id="dumpnoise",
        label="Noise",
        # Dumpnoise only requires the server to be running; no pregen needed.
        prereqs=[],
        cmd_factory=_dumpnoise_cmd,
        server_required=True,
    ),
    # ── Tiny NN (Stage 1 density) pipeline ──
    StepDef(
        id="train_stage1_density",
        label="T1 Density",
        prereqs=["dumpnoise"],
        cmd_factory=_train_stage1_density_cmd,
    ),
    StepDef(
        id="extract_stage1_weights",
        label="E1 Weights",
        prereqs=["train_stage1_density"],
        cmd_factory=_extract_stage1_weights_cmd,
    ),
    StepDef(
        id="distill_density",
        label="Distill",
        prereqs=["train_stage1_density"],
        cmd_factory=_distill_density_cmd,
    ),
    StepDef(
        id="train_terrain_shaper",
        label="T Shaper",
        prereqs=["distill_density"],
        cmd_factory=_train_terrain_shaper_cmd,
    ),
    StepDef(
        id="extract_octree",
        label="Extract",
        prereqs=["voxy_import"],
        cmd_factory=_extract_octree_cmd,
    ),
    StepDef(
        id="column_heights",
        label="Heights",
        prereqs=["extract_octree", "dumpnoise"],
        cmd_factory=_column_heights_cmd,
    ),
    # ── Build training pairs — one per model, all depend on column_heights ──
    StepDef(
        id="build_pairs_init",
        label="P·Init",
        prereqs=["column_heights"],
        cmd_factory=_build_pairs_init_cmd,
    ),
    StepDef(
        id="build_pairs_refine",
        label="P·Refine",
        prereqs=["column_heights"],
        cmd_factory=_build_pairs_refine_cmd,
    ),
    StepDef(
        id="build_pairs_leaf",
        label="P·Leaf",
        prereqs=["column_heights"],
        cmd_factory=_build_pairs_leaf_cmd,
    ),
    # ── Train — one per model, each depends on its own pair cache ──
    StepDef(
        id="train_init",
        label="T·Init",
        prereqs=["build_pairs_init"],
        cmd_factory=_train_init_cmd,
    ),
    StepDef(
        id="train_refine",
        label="T·Refine",
        prereqs=["build_pairs_refine"],
        cmd_factory=_train_refine_cmd,
    ),
    StepDef(
        id="train_leaf",
        label="T·Leaf",
        prereqs=["build_pairs_leaf"],
        cmd_factory=_train_leaf_cmd,
    ),
    # ── Export per-model, parallelised
    StepDef(
        id="export_init",
        label="E·Init",
        prereqs=["train_init"],
        cmd_factory=_export_init_cmd,
    ),
    StepDef(
        id="export_refine",
        label="E·Refine",
        prereqs=["train_refine"],
        cmd_factory=_export_refine_cmd,
    ),
    StepDef(
        id="export_leaf",
        label="E·Leaf",
        prereqs=["train_leaf"],
        cmd_factory=_export_leaf_cmd,
    ),
    # ── Deploy per-model, each depends only on its export
    StepDef(
        id="deploy_init",
        label="D·Init",
        prereqs=["export_init"],
        cmd_factory=_deploy_init_cmd,
    ),
    StepDef(
        id="deploy_refine",
        label="D·Refine",
        prereqs=["export_refine"],
        cmd_factory=_deploy_refine_cmd,
    ),
    StepDef(
        id="deploy_leaf",
        label="D·Leaf",
        prereqs=["export_leaf"],
        cmd_factory=_deploy_leaf_cmd,
    ),
    # ──── Future loopback stubs (leave door open) ────
    StepDef(
        id="reset_data",
        label="Reset",
        prereqs=["deploy_leaf"],
        cmd_factory=_reset_data_cmd,
        enabled=False,
    ),
    StepDef(
        id="new_seed",
        label="New Seed",
        prereqs=["reset_data"],
        cmd_factory=_new_seed_cmd,
        enabled=False,
    ),
]

# Convenience lookup
STEP_BY_ID: dict[str, StepDef] = {s.id: s for s in PIPELINE_STEPS}

# Registry alias — preferred name when treating PIPELINE_STEPS as a type catalogue
# rather than a fixed pipeline order.  New code should import STEP_REGISTRY; the
# ``PIPELINE_STEPS`` name is kept for backward compatibility with existing callers.
STEP_REGISTRY: list[StepDef] = PIPELINE_STEPS

# Only the enabled steps shown as live nodes; disabled ones are stubs
ACTIVE_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if s.enabled]
STUB_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if not s.enabled]
