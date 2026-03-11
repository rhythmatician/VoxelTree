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


# ---------------------------------------------------------------------------
# Helper — resolve the Python interpreter inside the current venv (if any)
# ---------------------------------------------------------------------------

def _python() -> str:
    return sys.executable


def _vt_root() -> Path:
    """Absolute path to the VoxelTree project root (parent of this file)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Command factories
# Each receives the *full* profile dict and returns a list[str] suitable for
# subprocess.Popen.  Commands are run with cwd=VoxelTree root.
# ---------------------------------------------------------------------------

def _pregen_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    rcon = p.get("rcon", {})
    cmd = [
        _python(), "data-cli.py", "dataprep",
        "--from-step", "pregen",
        "--radius", str(world.get("radius", 2048)),
        "--password", str(rcon.get("password", "")),
        "--host", str(rcon.get("host", "localhost")),
        "--port", str(rcon.get("port", 25575)),
    ]
    if world.get("seed"):
        # seed is used by the server config, not data-cli; included for reference
        pass
    return cmd


def _voxy_import_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    rcon = p.get("rcon", {})
    return [
        _python(), "data-cli.py", "voxy-import",
        "--world-name", str(world.get("save_name", "New World")),
        "--password", str(rcon.get("password", "")),
        "--host", str(rcon.get("host", "localhost")),
        "--port", str(rcon.get("port", 25575)),
        "--timeout", str(rcon.get("timeout", 300)),
    ]


def _dumpnoise_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    rcon = p.get("rcon", {})
    return [
        _python(), "data-cli.py", "dumpnoise",
        "--radius", str(world.get("radius", 2048)),
        "--password", str(rcon.get("password", "")),
        "--host", str(rcon.get("host", "localhost")),
        "--port", str(rcon.get("port", 25575)),
    ]


def _extract_octree_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(), "data-cli.py", "dataprep",
        "--from-step", "extract-octree",
        "--voxy-dir", str(data.get("voxy_dir", "../LODiffusion/run/saves")),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if data.get("max_sections"):
        cmd += ["--max-sections", str(data["max_sections"])]
    if data.get("min_solid"):
        cmd += ["--min-solid", str(data["min_solid"])]
    # Stop after this step; the remaining steps are separate nodes
    cmd += ["--to-step", "extract-octree"] if False else []  # data-cli runs through; that's ok
    return cmd


def _column_heights_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(), "data-cli.py", "dataprep",
        "--from-step", "column-heights-octree",
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if data.get("noise_dump_dir"):
        cmd += ["--noise-dump-dir", str(data["noise_dump_dir"])]
    return cmd


def _build_pairs_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(), "data-cli.py", "dataprep",
        "--from-step", "build-octree-pairs",
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    cmd += ["--val-split", str(data.get("val_split", 0.1))]
    return cmd


def _train_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    cmd = [
        _python(), "pipeline.py", "train",
        "--epochs", str(train.get("epochs", 20)),
        "--batch-size", str(train.get("batch_size", 4)),
        "--lr", str(train.get("lr", 1e-4)),
        "--device", str(train.get("device", "auto")),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    if train.get("output_dir"):
        cmd += ["--model-dir", str(train["output_dir"])]
    return cmd


def _export_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    export = p.get("export", {})
    model_dir = train.get("output_dir", "models/voxy_octree")
    checkpoint = str(Path(model_dir) / "best_model.pt")
    cmd = [
        _python(), "pipeline.py", "export",
        "--checkpoint", checkpoint,
    ]
    if export.get("output_dir"):
        cmd += ["--export-dir", str(export["output_dir"])]
    return cmd


def _deploy_cmd(p: dict) -> list[str]:
    export = p.get("export", {})
    deploy = p.get("deploy", {})
    export_dir = export.get("output_dir", "production")
    cmd = [_python(), "pipeline.py", "deploy", str(export_dir)]
    if deploy.get("target_dir"):
        cmd += ["--dest", str(deploy["target_dir"])]
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
    ),
    StepDef(
        id="voxy_import",
        label="Voxy",
        prereqs=["pregen"],
        cmd_factory=_voxy_import_cmd,
    ),
    StepDef(
        id="dumpnoise",
        label="Noise",
        prereqs=["pregen"],
        cmd_factory=_dumpnoise_cmd,
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
    StepDef(
        id="build_pairs",
        label="Pairs",
        prereqs=["column_heights"],
        cmd_factory=_build_pairs_cmd,
    ),
    StepDef(
        id="train",
        label="Train",
        prereqs=["build_pairs"],
        cmd_factory=_train_cmd,
    ),
    StepDef(
        id="export",
        label="Export",
        prereqs=["train"],
        cmd_factory=_export_cmd,
    ),
    StepDef(
        id="deploy",
        label="Deploy",
        prereqs=["export"],
        cmd_factory=_deploy_cmd,
    ),
    # ---- Future loopback stubs (leave door open) ----
    StepDef(
        id="reset_data",
        label="Reset",
        prereqs=["deploy"],
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

# Only the enabled steps shown as live nodes; disabled ones are stubs
ACTIVE_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if s.enabled]
STUB_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if not s.enabled]
