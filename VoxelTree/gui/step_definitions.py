"""step_definitions.py — Pipeline step metadata and CLI command factories.

# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE: MODEL TRACKS
# ═══════════════════════════════════════════════════════════════════════════
#
# Every trainable model is declared as a ``ModelTrack``.  One entry in
# ``MODEL_TRACKS`` (at the bottom of this file) auto-generates four canonical
# StepDef entries for that model:
#
#   build_pairs → train → export → deploy
#
# Step IDs follow the convention ``{phase}_{track_id}``
# (e.g. ``train_sparse_root``, ``deploy_init``).
#
# ## Adding a new model — 3-step recipe
#
#   1. Write cmd-factory functions for each phase.
#      Leave ``None`` for any phase not yet implemented — the step is created
#      as an ``enabled=False`` stub so it appears greyed-out in the GUI.
#
#   2. Append one ``ModelTrack(...)`` entry to ``MODEL_TRACKS`` below.
#
#   3. Done.  Swim-lane rows, detail-panel groups, and DAG nodes all appear
#      automatically on the next GUI launch.  No other files need editing.
#
# ## Phase vocabulary
#
#   build_pairs — generate/download training pair files (input to training)
#   train       — train the model from pairs
#   export      — export checkpoint to ONNX / binary weights
#   deploy      — copy exported artefacts to the LODiffusion mod directory
#
# ## Track IDs in use (add new tracks to this list when you create them)
#
#   "init"         — Octree Init model (L4 → L3)
#   "refine"       — Octree Refine model (L3 → L1)
#   "leaf"         — Octree Leaf model (L0 block labels)
#   "sparse_root"  — Sparse-Root hierarchy classifier (5-level octree)
#   "stage1"       — Stage-1 density MLP (tiny NN distilled from Minecraft)
#
# ═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# StepDef — one node in the pipeline DAG
# ---------------------------------------------------------------------------


@dataclass
class StepDef:
    """Metadata for a single pipeline step / GUI node.

    Fields
    ------
    id           : Stable string key used in RunRegistry JSON files.
                   Changing this invalidates saved run states.
    label        : Short text shown inside the 52 px circle in the DAG.
    prereqs      : IDs of steps that must succeed before this one can run.
    cmd_factory  : Callable(profile_dict) → list[str] passed to subprocess.
    enabled      : False → faded stub node; not runnable.
    server_required : True → step sends RCON commands; needs Fabric server up.
    track        : Which ModelTrack owns this step (e.g. "init", "sparse_root").
                   None for data-acquisition and loopback steps.
    phase        : Which canonical phase this is within its track.
                   One of: "build_pairs", "train", "export", "deploy",
                   or a custom string for non-standard phases.
                   None for data-acquisition steps.
    """

    id: str
    label: str
    prereqs: list[str]
    cmd_factory: Callable[[dict], list[str]]
    enabled: bool = True
    server_required: bool = False
    track: str | None = None
    phase: str | None = None


# ---------------------------------------------------------------------------
# ModelTrack — declarative model registration
# ---------------------------------------------------------------------------


@dataclass
class ModelTrack:
    """Declares one model's pipeline track.

    Call ``ModelTrack.to_steps()`` to receive the list of StepDef entries
    to add to PIPELINE_STEPS.

    Parameters
    ----------
    track_id   : Snake-case identifier, e.g. "sparse_root".
                 Auto-generates step IDs like ``build_pairs_sparse_root``.
    label      : Human-readable name shown in swim-lane headers.
    swim_lane_color : Hex background tint for the model's DAG row.
    build_pairs_factory / train_factory / export_factory / deploy_factory :
                 CMD factory for each phase.  ``None`` → the step is generated
                 as an ``enabled=False`` stub (renders greyed-out).
    id_overrides : Maps phase → step_id when the model predates ModelTrack and
                 must preserve legacy run-state JSON keys for continuity.
                 Leave empty for new models.
    build_pairs_prereqs : Full prereq list for the build_pairs step.
                 Defaults to ``["column_heights"]``.
    extra_steps : Additional StepDef entries appended after the 4 canonical
                 phases (e.g. distillation, calibration).
    """

    track_id: str
    label: str
    swim_lane_color: str

    build_pairs_factory: Callable[[dict], list[str]] | None = None
    train_factory: Callable[[dict], list[str]] | None = None
    export_factory: Callable[[dict], list[str]] | None = None
    deploy_factory: Callable[[dict], list[str]] | None = None

    # Override step IDs for backward compat with pre-existing run-state JSONs.
    # Leave empty for new tracks — default IDs are ``{phase}_{track_id}``.
    id_overrides: dict[str, str] = field(default_factory=dict)

    # Full prereq list for build_pairs.  Defaults to ["column_heights"].
    build_pairs_prereqs: list[str] | None = None

    # Non-standard extra steps appended after the 4 canonical phases.
    extra_steps: list[StepDef] = field(default_factory=list)

    def step_id(self, phase: str) -> str:
        return self.id_overrides.get(phase, f"{phase}_{self.track_id}")

    def to_steps(self) -> list[StepDef]:
        """Generate the canonical 4-phase StepDef list for this track."""
        bp = self.step_id("build_pairs")
        tr = self.step_id("train")
        ex = self.step_id("export")
        dp = self.step_id("deploy")

        short = self.label[:6]

        steps: list[StepDef] = [
            StepDef(
                id=bp,
                label=f"P·{short}",
                prereqs=(
                    self.build_pairs_prereqs
                    if self.build_pairs_prereqs is not None
                    else ["column_heights"]
                ),
                cmd_factory=self.build_pairs_factory or _stub_cmd,
                enabled=self.build_pairs_factory is not None,
                track=self.track_id,
                phase="build_pairs",
            ),
            StepDef(
                id=tr,
                label=f"T·{short}",
                prereqs=[bp],
                cmd_factory=self.train_factory or _stub_cmd,
                enabled=self.train_factory is not None,
                track=self.track_id,
                phase="train",
            ),
            StepDef(
                id=ex,
                label=f"E·{short}",
                prereqs=[tr],
                cmd_factory=self.export_factory or _stub_cmd,
                enabled=self.export_factory is not None,
                track=self.track_id,
                phase="export",
            ),
            StepDef(
                id=dp,
                label=f"D·{short}",
                prereqs=[ex],
                cmd_factory=self.deploy_factory or _stub_cmd,
                enabled=self.deploy_factory is not None,
                track=self.track_id,
                phase="deploy",
            ),
        ]
        steps.extend(self.extra_steps)
        return steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python() -> str:
    return sys.executable


def _vt_root() -> Path:
    """Absolute path to the VoxelTree repo root (3 levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _stub_cmd(_p: dict) -> list[str]:
    """Placeholder factory for steps not yet implemented."""
    return [_python(), "-c", "raise NotImplementedError('step not yet implemented')"]


# ---------------------------------------------------------------------------
# Command factories — Data Acquisition
# ---------------------------------------------------------------------------


def _pregen_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
    return [
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


def _voxy_import_cmd(p: dict) -> list[str]:
    world = p.get("world", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
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
    # Use profile rcon.timeout so large-radius dumps do not time out.
    rcon_timeout = p.get("rcon", {}).get("timeout", 3600)
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
        "--timeout",
        str(rcon_timeout),
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


# ---------------------------------------------------------------------------
# Command factories — Octree models (init / refine / leaf)
# ---------------------------------------------------------------------------


def _build_pairs_model_cmd(p: dict, model_type: str) -> list[str]:
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


def _train_model_cmd(p: dict, model_type: str) -> list[str]:
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


def _export_model_cmd(p: dict, model_type: str) -> list[str]:
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
        "--models",
        model_type,
    ]
    if export.get("output_dir"):
        cmd += ["--export-dir", str(export["output_dir"])]
    return cmd


def _deploy_model_cmd(p: dict, model_type: str) -> list[str]:
    export = p.get("export", {})
    deploy = p.get("deploy", {})
    export_dir = export.get("output_dir", "production")
    cmd = [
        _python(),
        "-m",
        "VoxelTree",
        "pipeline",
        "deploy",
        str(export_dir),
        "--models",
        model_type,
    ]
    if deploy.get("target_dir"):
        cmd += ["--dest", str(deploy["target_dir"])]
    return cmd


def _build_pairs_init_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "init")


def _train_init_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "init")


def _export_init_cmd(p: dict) -> list[str]:
    return _export_model_cmd(p, "init")


def _deploy_init_cmd(p: dict) -> list[str]:
    return _deploy_model_cmd(p, "init")


def _build_pairs_refine_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "refine")


def _train_refine_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "refine")


def _export_refine_cmd(p: dict) -> list[str]:
    return _export_model_cmd(p, "refine")


def _deploy_refine_cmd(p: dict) -> list[str]:
    return _deploy_model_cmd(p, "refine")


def _build_pairs_leaf_cmd(p: dict) -> list[str]:
    return _build_pairs_model_cmd(p, "leaf")


def _train_leaf_cmd(p: dict) -> list[str]:
    return _train_model_cmd(p, "leaf")


def _export_leaf_cmd(p: dict) -> list[str]:
    return _export_model_cmd(p, "leaf")


def _deploy_leaf_cmd(p: dict) -> list[str]:
    return _deploy_model_cmd(p, "leaf")


# ---------------------------------------------------------------------------
# Command factories — Stage-1 density (tiny NN)
# ---------------------------------------------------------------------------


def _train_stage1_density_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    cmd = [_python(), "scripts/stage1/train_density.py"]
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


def _extract_stage1_weights_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    extract = p.get("extract", {})
    cmd = [_python(), "scripts/stage1/extract_density_weights.py"]
    model_dir = train.get("output_dir")
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    if extract.get("output_dir"):
        cmd += ["--out-dir", str(extract["output_dir"])]
    return cmd


def _distill_density_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    distill = p.get("distill", {})
    return [
        _python(),
        "scripts/stage1/distill_density.py",
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


def _train_terrain_shaper_cmd(_: dict) -> list[str]:
    """Train the TerrainShaper spline-approximation MLP (fixed internal settings)."""
    return [_python(), "scripts/stage1/train_terrain_shaper.py"]


# ---------------------------------------------------------------------------
# Command factories — Sparse Root
# ---------------------------------------------------------------------------


def _build_pairs_sparse_root_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    cmd = [
        _python(),
        str(_vt_root() / "VoxelTree" / "scripts" / "build_octree_pairs.py"),
        "--sparse-root",
        "--val-split",
        str(data.get("val_split", 0.1)),
    ]
    if data.get("data_dir"):
        cmd += ["--data-dir", str(data["data_dir"])]
    return cmd


def _train_sparse_root_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    data_dir = data.get("data_dir", "noise_training_data")
    cmd = [
        _python(),
        "scripts/sparse_root/train.py",
        "--data",
        str(Path(data_dir) / "sparse_root_pairs.npz"),
        "--model-variant",
        str(train.get("sparse_root_variant", "fast")),
        "--hidden",
        str(train.get("sparse_root_hidden", 80)),
        "--epochs",
        str(train.get("epochs", 20)),
        "--batch-size",
        str(train.get("batch_size", 4)),
        "--lr",
        str(train.get("lr", 1e-4)),
        "--device",
        str(train.get("device", "auto")),
    ]
    if train.get("output_dir"):
        cmd += ["--out", str(Path(train["output_dir"]) / "sparse_root_model.pt")]
    return cmd


def _distill_sparse_root_cmd(p: dict) -> list[str]:
    data = p.get("data", {})
    train = p.get("train", {})
    data_dir = data.get("data_dir", "noise_training_data")
    teacher_dir = train.get("output_dir", ".")
    return [
        _python(),
        "scripts/sparse_root/distill.py",
        "--teacher-checkpoint",
        str(Path(teacher_dir) / "sparse_root_model.pt"),
        "--data",
        str(Path(data_dir) / "sparse_root_pairs.npz"),
        "--student-variant",
        str(train.get("sparse_root_variant", "fast")),
        "--student-hidden",
        str(train.get("sparse_root_hidden", 80)),
    ]


# ---------------------------------------------------------------------------
# Command factories — Loopback stubs (future iteration)
# ---------------------------------------------------------------------------


def _reset_data_cmd(_p: dict) -> list[str]:
    return ["echo", "[loopback] reset_data not yet implemented"]


def _new_seed_cmd(_p: dict) -> list[str]:
    return ["echo", "[loopback] new_seed not yet implemented"]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRACKS
# ═══════════════════════════════════════════════════════════════════════════
#
# Register models here.  Each entry auto-generates StepDef objects via
# ModelTrack.to_steps() and controls swim-lane layout and detail-panel groups.
#
# swim_lane_color : subtle dark background tint for the DAG row
# id_overrides    : use only for models with pre-existing run-state JSON keys
# extra_steps     : non-standard steps beyond the 4 canonical phases
#
# ═══════════════════════════════════════════════════════════════════════════

MODEL_TRACKS: list[ModelTrack] = [
    # ── Octree: Init ──────────────────────────────────────────────────────
    ModelTrack(
        track_id="init",
        label="Init",
        swim_lane_color="#0b1a2e",
        build_pairs_factory=_build_pairs_init_cmd,
        train_factory=_train_init_cmd,
        export_factory=_export_init_cmd,
        deploy_factory=_deploy_init_cmd,
    ),
    # ── Octree: Refine ────────────────────────────────────────────────────
    ModelTrack(
        track_id="refine",
        label="Refine",
        swim_lane_color="#170b2e",
        build_pairs_factory=_build_pairs_refine_cmd,
        train_factory=_train_refine_cmd,
        export_factory=_export_refine_cmd,
        deploy_factory=_deploy_refine_cmd,
    ),
    # ── Octree: Leaf ──────────────────────────────────────────────────────
    ModelTrack(
        track_id="leaf",
        label="Leaf",
        swim_lane_color="#0b2514",
        build_pairs_factory=_build_pairs_leaf_cmd,
        train_factory=_train_leaf_cmd,
        export_factory=_export_leaf_cmd,
        deploy_factory=_deploy_leaf_cmd,
    ),
    # ── Sparse Root ───────────────────────────────────────────────────────
    # export and deploy are stubs until scripts exist:
    #   TODO export: write scripts/sparse_root/export_sparse_root.py
    #   TODO deploy: extend VoxelTree/scripts/deploy_models.py for sparse_root
    ModelTrack(
        track_id="sparse_root",
        label="SparseRoot",
        swim_lane_color="#2a1500",
        build_pairs_factory=_build_pairs_sparse_root_cmd,
        train_factory=_train_sparse_root_cmd,
        export_factory=None,
        deploy_factory=None,
        extra_steps=[
            StepDef(
                id="distill_sparse_root",
                label="Distill",
                prereqs=["train_sparse_root"],
                cmd_factory=_distill_sparse_root_cmd,
                track="sparse_root",
                phase="distill",
            ),
        ],
    ),
    # ── Stage-1 Density (tiny NN) ─────────────────────────────────────────
    # Predates ModelTrack; id_overrides preserve legacy run-state JSON keys.
    # build_pairs_prereqs=[dumpnoise] because the training script reads the
    # noise-dump JSONs directly — no explicit pair-building step is needed.
    # deploy is baked into extract_stage1_weights (pass --no-deploy to skip).
    ModelTrack(
        track_id="stage1",
        label="Stage 1",
        swim_lane_color="#00202a",
        build_pairs_factory=None,
        train_factory=_train_stage1_density_cmd,
        export_factory=_extract_stage1_weights_cmd,
        deploy_factory=None,
        id_overrides={
            "build_pairs": "build_pairs_stage1",
            "train": "train_stage1_density",
            "export": "extract_stage1_weights",
            "deploy": "deploy_stage1",
        },
        build_pairs_prereqs=["dumpnoise"],
        extra_steps=[
            StepDef(
                id="distill_density",
                label="Distill",
                prereqs=["train_stage1_density"],
                cmd_factory=_distill_density_cmd,
                track="stage1",
                phase="distill",
            ),
            StepDef(
                id="train_terrain_shaper",
                label="T·Shaper",
                prereqs=["distill_density"],
                cmd_factory=_train_terrain_shaper_cmd,
                track="stage1",
                phase="train_terrain_shaper",
            ),
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE_STEPS — canonical ordered list consumed by RunRegistry, DAG, etc.
# ═══════════════════════════════════════════════════════════════════════════

# Data-acquisition steps (no model track — they feed all tracks)
_DATA_ACQ_STEPS: list[StepDef] = [
    StepDef(
        id="pregen",
        label="Pregen",
        prereqs=[],
        cmd_factory=_pregen_cmd,
        server_required=True,
        phase="data_acq",
    ),
    StepDef(
        id="voxy_import",
        label="Voxy",
        prereqs=["pregen"],
        cmd_factory=_voxy_import_cmd,
        server_required=True,
        phase="data_acq",
    ),
    StepDef(
        id="dumpnoise",
        label="Noise",
        prereqs=[],
        cmd_factory=_dumpnoise_cmd,
        server_required=True,
        phase="data_acq",
    ),
    StepDef(
        id="extract_octree",
        label="Extract",
        prereqs=["voxy_import"],
        cmd_factory=_extract_octree_cmd,
        phase="data_acq",
    ),
    StepDef(
        id="column_heights",
        label="Heights",
        prereqs=["extract_octree", "dumpnoise"],
        cmd_factory=_column_heights_cmd,
        phase="data_acq",
    ),
]

# Future iteration stubs
_LOOPBACK_STEPS: list[StepDef] = [
    StepDef(
        id="reset_data",
        label="Reset",
        prereqs=["deploy_leaf"],
        cmd_factory=_reset_data_cmd,
        enabled=False,
        phase="loopback",
    ),
    StepDef(
        id="new_seed",
        label="New Seed",
        prereqs=["reset_data"],
        cmd_factory=_new_seed_cmd,
        enabled=False,
        phase="loopback",
    ),
]

# Assemble: data acq → all model tracks (in MODEL_TRACKS order) → loopback
PIPELINE_STEPS: list[StepDef] = (
    _DATA_ACQ_STEPS
    + [step for track in MODEL_TRACKS for step in track.to_steps()]
    + _LOOPBACK_STEPS
)

# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

STEP_BY_ID: dict[str, StepDef] = {s.id: s for s in PIPELINE_STEPS}

# Registry alias — preferred name when treating PIPELINE_STEPS as a type
# catalogue rather than a fixed order.  PIPELINE_STEPS kept for backward compat.
STEP_REGISTRY: list[StepDef] = PIPELINE_STEPS

ACTIVE_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if s.enabled]
STUB_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if not s.enabled]

# Track order as declared (data_acq first, then model tracks, then loopback)
TRACK_ORDER: list[str] = ["data_acq"] + [t.track_id for t in MODEL_TRACKS] + ["loopback"]

# Map track_id → ModelTrack for quick lookup
TRACK_BY_ID: dict[str, ModelTrack] = {t.track_id: t for t in MODEL_TRACKS}
