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
# ## Artifact-based prerequisite wiring
#
# Each step declares what it ``produces`` (artifacts created) and
# ``consumes`` (artifacts required).  At module load time,
# ``_wire_prereqs()`` auto-computes every step's ``prereqs`` list from
# the artifact graph.  Manual prereq lists are no longer needed.
#
# ## Adding a new model — 3-step recipe
#
#   1. Write cmd-factory functions for each phase.
#      Leave ``None`` for any phase not yet implemented — the step is created
#      as an ``enabled=False`` stub so it appears greyed-out in the GUI.
#
#   2. Append one ``ModelTrack(...)`` entry to ``MODEL_TRACKS`` below.
#      Standard artifact names (``{track_id}_pairs``, ``{track_id}_checkpoint``,
#      etc.) are generated automatically.  Set ``build_pairs_consumes`` if
#      the track reads something other than ``octree_with_heights``.
#
#   3. Done.  Swim-lane rows, detail-panel groups, DAG nodes, **and prereq
#      edges** all appear automatically on the next GUI launch.
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
#   "sparse_root"  — Sparse-Root hierarchy classifier (5-level octree)
#   "stage1"       — Stage-1 density MLP + Terrain Shaper NN
#
# ═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
                   **Auto-computed** by ``_wire_prereqs()`` from the
                   produces / consumes artifact graph.  Do not set manually.
    cmd_factory  : Callable(profile_dict) → list[str] passed to subprocess.
    enabled      : False → faded stub node; not runnable.
    server_required : True → step sends RCON commands; needs Fabric server up.
    track        : Which ModelTrack owns this step (e.g. "init", "sparse_root").
                   None for data-acquisition and loopback steps.
    phase        : Which canonical phase this is within its track.
                   One of: "build_pairs", "train", "export", "deploy",
                   or a custom string for non-standard phases.
                   None for data-acquisition steps.
    produces     : Logical artifact names this step creates.
    consumes     : Logical artifact names this step requires as input.
    """

    id: str
    label: str
    prereqs: list[str]
    cmd_factory: Callable[[dict], list[str]]
    enabled: bool = True
    server_required: bool = False
    track: str | None = None
    phase: str | None = None
    produces: frozenset[str] = frozenset()
    consumes: frozenset[str] = frozenset()


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
    build_pairs_consumes : Artifacts consumed by the build_pairs step.
                 Defaults to ``{"octree_with_heights"}``.
    extra_steps : Additional StepDef entries appended after the 4 canonical
                 phases (e.g. distillation, calibration).  Must include
                 ``produces`` and ``consumes`` for auto-wiring.
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

    # Artifacts consumed by build_pairs.  Defaults to {"octree_with_heights"}.
    build_pairs_consumes: frozenset[str] = frozenset({"octree_with_heights"})

    # Non-standard extra steps appended after the 4 canonical phases.
    extra_steps: list[StepDef] = field(default_factory=list)

    # When False, this track's steps are excluded from ProfileDag.default().
    # They remain in the global step registry so advanced profiles can still
    # include them explicitly.
    in_default_dag: bool = True

    def step_id(self, phase: str) -> str:
        return self.id_overrides.get(phase, f"{phase}_{self.track_id}")

    def to_steps(self) -> list[StepDef]:
        """Generate the canonical 4-phase StepDef list for this track.

        Each phase auto-generates standard artifact names:

        * build_pairs → produces ``{track_id}_pairs``
        * train       → produces ``{track_id}_checkpoint``
        * export      → produces ``{track_id}_exported``
        * deploy      → produces ``{track_id}_deployed``

        Prerequisites are auto-wired later by ``_wire_prereqs()``.
        """
        bp = self.step_id("build_pairs")
        tr = self.step_id("train")
        ex = self.step_id("export")
        dp = self.step_id("deploy")

        tid = self.track_id
        short = self.label[:6]

        steps: list[StepDef] = [
            StepDef(
                id=bp,
                label=f"P·{short}",
                prereqs=[],
                cmd_factory=self.build_pairs_factory or _stub_cmd,
                enabled=self.build_pairs_factory is not None,
                track=tid,
                phase="build_pairs",
                produces=frozenset({f"{tid}_pairs"}),
                consumes=self.build_pairs_consumes,
            ),
            StepDef(
                id=tr,
                label=f"T·{short}",
                prereqs=[],
                cmd_factory=self.train_factory or _stub_cmd,
                enabled=self.train_factory is not None,
                track=tid,
                phase="train",
                produces=frozenset({f"{tid}_checkpoint"}),
                consumes=frozenset({f"{tid}_pairs"}),
            ),
            StepDef(
                id=ex,
                label=f"E·{short}",
                prereqs=[],
                cmd_factory=self.export_factory or _stub_cmd,
                enabled=self.export_factory is not None,
                track=tid,
                phase="export",
                produces=frozenset({f"{tid}_exported"}),
                consumes=frozenset({f"{tid}_checkpoint"}),
            ),
            StepDef(
                id=dp,
                label=f"D·{short}",
                prereqs=[],
                cmd_factory=self.deploy_factory or _stub_cmd,
                enabled=self.deploy_factory is not None,
                track=tid,
                phase="deploy",
                produces=frozenset({f"{tid}_deployed"}),
                consumes=frozenset({f"{tid}_exported"}),
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


def _wire_prereqs(steps: list[StepDef]) -> None:
    """Auto-compute each step's ``prereqs`` from the artifact graph.

    1. Build a map of artifact → producer step.
    2. For each step, set ``prereqs`` to the list of steps that produce
       its consumed artifacts.
    3. Perform transitive reduction so that only the *minimal* set of
       direct prerequisites remains (removes edges implied by other
       prerequisite chains).
    """
    # ── 1. Producer map ───────────────────────────────────────────────
    producers: dict[str, str] = {}
    for step in steps:
        for art in step.produces:
            if art in producers:
                raise ValueError(
                    f"Artifact '{art}' produced by both " f"'{producers[art]}' and '{step.id}'"
                )
            producers[art] = step.id

    # ── 2. Direct prereqs from consumed artifacts ─────────────────────
    step_map: dict[str, StepDef] = {s.id: s for s in steps}
    for step in steps:
        computed: list[str] = []
        for art in sorted(step.consumes):  # sorted for determinism
            pid = producers.get(art)
            if pid is None:
                raise ValueError(f"Step '{step.id}' consumes '{art}' " f"but no step produces it")
            if pid != step.id and pid not in computed:
                computed.append(pid)
        step.prereqs = computed

    # ── 3. Transitive reduction ───────────────────────────────────────
    #
    # For each prereq P of step S, check whether P is already a
    # transitive ancestor of some *other* prereq Q of S.  If so, Q
    # already implies P and P is redundant.
    for step in steps:
        if len(step.prereqs) <= 1:
            continue
        # Collect transitive ancestors of each prereq.
        anc: dict[str, set[str]] = {}
        for p in step.prereqs:
            visited: set[str] = set()
            stack = [p]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node in step_map:
                    stack.extend(step_map[node].prereqs)
            visited.discard(p)
            anc[p] = visited

        redundant = {p for p in step.prereqs if any(p in anc[q] for q in step.prereqs if q != p)}
        if redundant:
            step.prereqs = [p for p in step.prereqs if p not in redundant]


# ---------------------------------------------------------------------------
# Command factories — Data Acquisition
# ---------------------------------------------------------------------------


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


def _harvest_cmd(p: dict) -> list[str]:
    """Automated Voxy data harvest via DataHarvester spiral + RocksDB monitoring.

    Replaces the old manual pregen → voxy-import two-step with a single
    orchestrated command that handles chunk pregeneration, bot teleportation,
    and blocks until the Voxy RocksDB stabilises.
    """
    world = p.get("world", {})
    data = p.get("data", {})
    from VoxelTree.gui.server_manager import get_rcon_settings  # noqa: PLC0415

    rcon = get_rcon_settings()
    rcon_timeout = p.get("rcon", {}).get("timeout", 3600)
    cmd = [
        _python(),
        "-m",
        "VoxelTree.preprocessing.harvest",
        "--radius",
        str(world.get("radius", 2048)),
        "--password",
        str(rcon["password"]),
        "--rcon-port",
        str(rcon["port"]),
        "--voxy-timeout",
        str(rcon_timeout),
    ]
    if data.get("voxy_dir"):
        cmd += ["--voxy-dir", str(data["voxy_dir"])]
    return cmd


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
    # Export weights only (do not deploy).  The deploy phase runs the same
    # script without --no-deploy to push into LODiffusion resources.
    train = p.get("train", {})
    extract = p.get("extract", {})
    cmd = [_python(), "scripts/stage1/extract_density_weights.py", "--no-deploy"]
    model_dir = train.get("output_dir")
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    if extract.get("output_dir"):
        cmd += ["--out-dir", str(extract["output_dir"])]
    return cmd


def _build_pairs_stage1_cmd(p: dict) -> list[str]:
    """Validate that Stage1 noise dumps exist (stage1_dumps)."""
    data = p.get("data", {})
    dump_dir = data.get("stage1_dump_dir", "stage1_dumps")
    # Fail fast if there are no Stage1 dumps.
    snippet = (
        "import pathlib,sys; "
        "p=pathlib.Path(%r); "
        "files=list(p.glob('chunk_*.json')); "
        "sys.exit(0 if files else (print(f'No Stage1 dumps found in {p}', file=sys.stderr), 1)[1])"
        % str(dump_dir)
    )
    return [_python(), "-c", snippet]


def _deploy_stage1_cmd(p: dict) -> list[str]:
    train = p.get("train", {})
    extract = p.get("extract", {})
    cmd = [_python(), "scripts/stage1/extract_density_weights.py"]
    model_dir = train.get("output_dir")
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    if extract.get("output_dir"):
        cmd += ["--out-dir", str(extract["output_dir"])]
    return cmd


def _distill_density_cmd(p: dict[str, Any]) -> list[str]:
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


def _export_sparse_root_cmd(p: dict[str, Any]) -> list[str]:
    train = p.get("train", {})
    export = p.get("export", {})
    checkpoint = Path(train.get("output_dir", ".")) / "sparse_root_model.pt"
    out_dir = export.get("output_dir") or "LODiffusion/run/models"
    return [
        _python(),
        "LODiffusion/models/export_sparse_root.py",
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
    ]


def _deploy_sparse_root_cmd(p: dict[str, Any]) -> list[str]:
    deploy = p.get("deploy", {})
    checkpoint = Path(p.get("train", {}).get("output_dir", ".")) / "sparse_root_model.pt"
    out_dir = (
        deploy.get("target_dir")
        or p.get("export", {}).get("output_dir")
        or "LODiffusion/run/models"
    )
    return [
        _python(),
        "LODiffusion/models/export_sparse_root.py",
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
    ]


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
        export_factory=_export_sparse_root_cmd,
        deploy_factory=_deploy_sparse_root_cmd,
        build_pairs_consumes=frozenset({"voxy_db", "noise_dumps"}),
        extra_steps=[
            StepDef(
                id="distill_sparse_root",
                label="Distill",
                prereqs=[],
                cmd_factory=_distill_sparse_root_cmd,
                track="sparse_root",
                phase="distill",
                produces=frozenset({"sparse_root_distilled"}),
                consumes=frozenset({"sparse_root_checkpoint", "sparse_root_pairs"}),
            ),
        ],
    ),
    # ── Stage-1 Density (tiny NN) ─────────────────────────────────────────
    # Predates ModelTrack; id_overrides preserve legacy run-state JSON keys.
    # build_pairs_consumes={"noise_dumps"} because the training script reads
    # the noise-dump JSONs directly — no explicit pair-building step is needed.
    # export is via extract_stage1_weights (uses --no-deploy).
    # deploy runs extract_stage1_weights without --no-deploy to push into LODiffusion resources.
    ModelTrack(
        track_id="stage1",
        label="Stage 1",
        swim_lane_color="#00202a",
        build_pairs_factory=_build_pairs_stage1_cmd,
        train_factory=_train_stage1_density_cmd,
        export_factory=_extract_stage1_weights_cmd,
        deploy_factory=_deploy_stage1_cmd,
        id_overrides={
            "build_pairs": "build_pairs_stage1",
            "train": "train_stage1_density",
            "export": "extract_stage1_weights",
            "deploy": "deploy_stage1",
        },
        build_pairs_consumes=frozenset({"noise_dumps"}),
        extra_steps=[
            StepDef(
                id="distill_density",
                label="Distill",
                prereqs=[],
                cmd_factory=_distill_density_cmd,
                track="stage1",
                phase="distill",
                produces=frozenset({"stage1_distilled"}),
                consumes=frozenset({"stage1_checkpoint"}),
            ),
            StepDef(
                id="train_terrain_shaper",
                label="T·Shaper",
                prereqs=[],
                cmd_factory=_train_terrain_shaper_cmd,
                track="stage1",
                phase="train_terrain_shaper",
                produces=frozenset({"terrain_shaper_checkpoint"}),
                consumes=frozenset({"noise_dumps"}),
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
        id="harvest",
        label="Harvest",
        prereqs=[],
        cmd_factory=_harvest_cmd,
        server_required=True,
        phase="data_acq",
        produces=frozenset({"voxy_db", "noise_dumps"}),
        consumes=frozenset(),
    ),
    StepDef(
        id="extract_octree",
        label="Extract",
        prereqs=[],
        cmd_factory=_extract_octree_cmd,
        phase="data_acq",
        produces=frozenset({"octree_npz"}),
        consumes=frozenset({"voxy_db"}),
    ),
    StepDef(
        id="column_heights",
        label="Heights",
        prereqs=[],
        cmd_factory=_column_heights_cmd,
        phase="data_acq",
        produces=frozenset({"octree_with_heights"}),
        consumes=frozenset({"octree_npz", "noise_dumps"}),
    ),
]

PIPELINE_STEPS: list[StepDef] = _DATA_ACQ_STEPS + [
    step for track in MODEL_TRACKS for step in track.to_steps()
]

# Auto-wire prereqs from the produces/consumes artifact graph.
_wire_prereqs(PIPELINE_STEPS)

# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

STEP_BY_ID: dict[str, StepDef] = {s.id: s for s in PIPELINE_STEPS}

# Registry alias — preferred name when treating PIPELINE_STEPS as a type
# catalogue rather than a fixed order.  PIPELINE_STEPS kept for backward compat.
STEP_REGISTRY: list[StepDef] = PIPELINE_STEPS

ACTIVE_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if s.enabled]
STUB_STEPS: list[StepDef] = [s for s in PIPELINE_STEPS if not s.enabled]

TRACK_ORDER: list[str] = ["data_acq"] + [t.track_id for t in MODEL_TRACKS]

# Map track_id → ModelTrack for quick lookup
TRACK_BY_ID: dict[str, ModelTrack] = {t.track_id: t for t in MODEL_TRACKS}
