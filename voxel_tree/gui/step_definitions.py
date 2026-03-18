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
# (e.g. ``train_sparse_octree``, ``deploy_init``).
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
#   "sparse_octree"       — Sparse Octree hierarchy classifier (5-level octree, legacy 13ch/4×2×4)
#   "terrain_shaper"      — Terrain Shaper + Density NN (legacy)
#   "density_v2"          — v7 Density predictor (6 climate → 2 density fields)
#   "biome_classifier"    — v7 Biome classifier (6 climate → 54 biome classes)
#   "heightmap_predictor" — v7 Heightmap predictor (96 climate → 32 height values)
#   "sparse_octree_v7"    — v7 Sparse Octree (15ch/4×4×4 → block hierarchy)
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
    run_fn       : Callable(profile_dict) → None; runs the step directly.
    enabled      : False → faded stub node; not runnable.
    server_required : True → step sends RCON commands; needs Fabric server up.
    client_required : True → step needs a Minecraft client (DataHarvester bot).
    track        : Which ModelTrack owns this step (e.g. "init", "sparse_octree").
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
    run_fn: Callable[[dict[str, Any]], None]
    enabled: bool = True
    server_required: bool = False
    client_required: bool = False
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
    track_id   : Snake-case identifier, e.g. "sparse_octree".
                 Auto-generates step IDs like ``build_pairs_sparse_octree``.
    label      : Human-readable name shown in swim-lane headers.
    swim_lane_color : Hex background tint for the model's DAG row.
    build_pairs_factory / train_factory / export_factory / deploy_factory :
                 Step runner for each phase.  ``None`` → the step is generated
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

    build_pairs_factory: Callable[[dict[str, Any]], None] | None = None
    train_factory: Callable[[dict[str, Any]], None] | None = None
    export_factory: Callable[[dict[str, Any]], None] | None = None
    deploy_factory: Callable[[dict[str, Any]], None] | None = None

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

    # ── Contract binding ─────────────────────────────────────────────────
    # Links this track to a model I/O contract in voxel_tree.contracts.
    # When set, the contracts system can detect when a contract revision
    # has been bumped but the pipeline scripts haven't been updated yet.
    contract_name: str | None = None
    contract_revision: int | None = None

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
                run_fn=self.build_pairs_factory or _stub_run,
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
                run_fn=self.train_factory or _stub_run,
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
                run_fn=self.export_factory or _stub_run,
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
                run_fn=self.deploy_factory or _stub_run,
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


def _stub_run(_p: dict[str, Any]) -> None:
    """Placeholder for steps not yet implemented."""
    raise NotImplementedError("step not yet implemented")


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
# Step runners — Data Acquisition
# ---------------------------------------------------------------------------


def _dumpnoise_run(p: dict[str, Any]) -> None:
    from voxel_tree.gui.server_manager import get_rcon_settings  # noqa: PLC0415
    from voxel_tree.preprocessing.cli import main as cli_main  # noqa: PLC0415

    world = p.get("world", {})
    rcon = get_rcon_settings()
    rcon_timeout = p.get("rcon", {}).get("timeout", 3600)
    try:
        cli_main(
            [
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
        )
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Dumpnoise failed with exit code {e.code}") from e


def _extract_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.preprocessing.cli import main as cli_main  # noqa: PLC0415

    data = p.get("data", {})
    argv = [
        "dataprep",
        "--from-step",
        "extract-octree",
        "--voxy-dir",
        str(data.get("voxy_dir", "../LODiffusion/run/saves")),
    ]
    if data.get("data_dir"):
        argv += ["--data-dir", str(data["data_dir"])]
    if data.get("max_sections"):
        argv += ["--max-sections", str(data["max_sections"])]
    if data.get("min_solid"):
        argv += ["--min-solid", str(data["min_solid"])]
    try:
        cli_main(argv)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Extract octree failed with exit code {e.code}") from e


def _column_heights_run(p: dict[str, Any]) -> None:
    from voxel_tree.preprocessing.cli import main as cli_main  # noqa: PLC0415

    data = p.get("data", {})
    argv = ["dataprep", "--from-step", "column-heights-octree"]
    if data.get("data_dir"):
        argv += ["--data-dir", str(data["data_dir"])]
    if data.get("noise_dump_dir"):
        argv += ["--noise-dump-dir", str(data["noise_dump_dir"])]
    try:
        cli_main(argv)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Column heights failed with exit code {e.code}") from e


def _pregen_run(p: dict[str, Any]) -> None:
    """Chunky pregeneration — generates chunks on the server."""
    from voxel_tree.gui.server_manager import get_rcon_settings  # noqa: PLC0415
    from voxel_tree.preprocessing.harvest import main as harvest_main  # noqa: PLC0415

    world = p.get("world", {})
    rcon = get_rcon_settings()
    rcon_timeout = p.get("rcon", {}).get("timeout", 7200)
    harvest_main(
        [
            "--pregen-only",
            "--radius",
            str(world.get("radius", 2048)),
            "--password",
            str(rcon["password"]),
            "--rcon-port",
            str(rcon["port"]),
            "--chunky-timeout",
            str(rcon_timeout),
        ]
    )


def _harvest_run(p: dict[str, Any]) -> None:
    """Voxy data harvest — ingests pre-generated chunks into Voxy.

    The voxy-dir is resolved in priority order:
      1. Explicit ``data.voxy_dir`` in the profile.
      2. Derived from the active server port in ``server.properties``
         (set by :meth:`ServerManager.configure_for_role` before launch).
      3. harvest.py's own default when neither is set.
    """
    from voxel_tree.gui.server_manager import get_rcon_settings, read_server_property  # noqa: PLC0415
    from voxel_tree.preprocessing.harvest import main as harvest_main  # noqa: PLC0415

    world = p.get("world", {})
    data = p.get("data", {})
    rcon = get_rcon_settings()
    rcon_timeout = p.get("rcon", {}).get("timeout", 3600)
    argv = [
        "--skip-pregen",
        "--radius",
        str(world.get("radius", 2048)),
        "--password",
        str(rcon["password"]),
        "--rcon-port",
        str(rcon["port"]),
        "--voxy-timeout",
        str(rcon_timeout),
    ]

    # Resolve voxy-dir: explicit profile value → active server port → harvest default.
    voxy_dir = data.get("voxy_dir")
    if not voxy_dir:
        server_port = read_server_property("server-port", "25565")
        if server_port:
            from voxel_tree.preprocessing.harvest import MODRINTH_VOXY_SAVES  # noqa: PLC0415

            voxy_dir = str(MODRINTH_VOXY_SAVES / f"localhost_{server_port}")

    if voxy_dir:
        argv += ["--voxy-dir", str(voxy_dir)]
    harvest_main(argv)


# ---------------------------------------------------------------------------
# Step runners — Terrain Shaper + Density NN
# ---------------------------------------------------------------------------


def _train_terrain_shaper_density_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.density.train_density import main as train_main  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    argv: list[str] = []
    if data.get("terrain_shaper_dump_dir"):
        argv += ["--data-dir", str(data["terrain_shaper_dump_dir"])]
    if train.get("output_dir"):
        argv += ["--out-dir", str(train["output_dir"])]
    if train.get("epochs") is not None:
        argv += ["--epochs", str(train["epochs"])]
    if train.get("batch_size") is not None:
        argv += ["--batch-size", str(train["batch_size"])]
    if train.get("lr") is not None:
        argv += ["--lr", str(train["lr"])]
    if train.get("target_mse") is not None:
        argv += ["--target-mse", str(train["target_mse"])]
    train_main(argv)


def _extract_terrain_shaper_weights_run(p: dict[str, Any]) -> None:
    """Export weights only (do not deploy)."""
    from voxel_tree.tasks.density.extract_density_weights import (
        main as extract_main,
    )  # noqa: PLC0415

    train = p.get("train", {})
    extract = p.get("extract", {})
    argv = ["--no-deploy"]
    model_dir = train.get("output_dir")
    if model_dir:
        argv += ["--model-dir", str(model_dir)]
    if extract.get("output_dir"):
        argv += ["--out-dir", str(extract["output_dir"])]
    extract_main(argv)


def _build_pairs_terrain_shaper_run(p: dict[str, Any]) -> None:
    """Validate that terrain shaper noise dumps exist."""
    data = p.get("data", {})
    dump_dir = Path(data.get("terrain_shaper_dump_dir", "terrain_shaper_dumps"))
    files = list(dump_dir.glob("chunk_*.json"))
    if not files:
        print(f"No terrain shaper dumps found in {dump_dir}", file=sys.stderr)
        sys.exit(1)


def _deploy_terrain_shaper_run(p: dict[str, Any]) -> None:
    """Deploy weights to LODiffusion resources."""
    from voxel_tree.tasks.density.extract_density_weights import (
        main as extract_main,
    )  # noqa: PLC0415

    train = p.get("train", {})
    extract = p.get("extract", {})
    argv: list[str] = []
    model_dir = train.get("output_dir")
    if model_dir:
        argv += ["--model-dir", str(model_dir)]
    if extract.get("output_dir"):
        argv += ["--out-dir", str(extract["output_dir"])]
    extract_main(argv)


# ---------------------------------------------------------------------------
# Step runners — Octree export/deploy (init/refine/leaf)
# ---------------------------------------------------------------------------


def _export_octree_run(p: dict[str, Any], model: str) -> None:
    """Export a single octree submodel via the octree export script."""
    from voxel_tree.preprocessing.pipeline import phase3_export  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    checkpoint_dir = Path(train.get("output_dir")) if train.get("output_dir") else None
    export_dir = Path(export.get("output_dir", "production"))
    phase3_export(None, export_dir, checkpoint_dir=checkpoint_dir, models=[model])


def _deploy_octree_run(p: dict[str, Any], model: str) -> None:
    """Deploy a single octree submodel via the deploy_models script."""
    from voxel_tree.preprocessing.pipeline import phase4_deploy  # noqa: PLC0415

    export = p.get("export", {})
    deploy = p.get("deploy", {})
    export_dir = Path(export.get("output_dir", "production"))
    dest = Path(deploy.get("target_dir")) if deploy.get("target_dir") else None
    phase4_deploy(export_dir, dest, models=[model])


def _export_init_run(p: dict[str, Any]) -> None:
    _export_octree_run(p, "init")


def _deploy_init_run(p: dict[str, Any]) -> None:
    _deploy_octree_run(p, "init")


def _export_refine_run(p: dict[str, Any]) -> None:
    _export_octree_run(p, "refine")


def _deploy_refine_run(p: dict[str, Any]) -> None:
    _deploy_octree_run(p, "refine")


def _export_leaf_run(p: dict[str, Any]) -> None:
    _export_octree_run(p, "leaf")


def _deploy_leaf_run(p: dict[str, Any]) -> None:
    _deploy_octree_run(p, "leaf")


def _export_init_cmd(profile: dict[str, Any]) -> list[str]:
    args: list[str] = ["--models", "init"]
    train = profile.get("train", {})
    if train.get("output_dir"):
        args += ["--checkpoint-dir", str(train["output_dir"])]
    return args


def _export_refine_cmd(profile: dict[str, Any]) -> list[str]:
    args: list[str] = ["--models", "refine"]
    train = profile.get("train", {})
    if train.get("output_dir"):
        args += ["--checkpoint-dir", str(train["output_dir"])]
    return args


def _export_leaf_cmd(profile: dict[str, Any]) -> list[str]:
    args: list[str] = ["--models", "leaf"]
    train = profile.get("train", {})
    if train.get("output_dir"):
        args += ["--checkpoint-dir", str(train["output_dir"])]
    return args


def _deploy_init_cmd(profile: dict[str, Any]) -> list[str]:
    return ["--models", "init"]


def _deploy_refine_cmd(profile: dict[str, Any]) -> list[str]:
    return ["--models", "refine"]


def _deploy_leaf_cmd(profile: dict[str, Any]) -> list[str]:
    return ["--models", "leaf"]


def _distill_density_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.density.distill_density import distill_student  # noqa: PLC0415

    train = p.get("train", {})
    distill = p.get("distill", {})
    distill_student(
        teacher_name=distill.get("teacher", "unet"),
        student_name=distill.get("student", "sep"),
        epochs=distill.get("epochs", 120),
        alpha=distill.get("alpha", 0.5),
        lr=distill.get("lr", 2e-3),
        device=train.get("device", "auto"),
    )


def _train_terrain_shaper_run(_: dict[str, Any]) -> None:
    """Train the TerrainShaper spline-approximation MLP."""
    from voxel_tree.tasks.terrain_shaper.train_terrain_shaper import (
        main as shaper_main,
    )  # noqa: PLC0415

    shaper_main()


# ---------------------------------------------------------------------------
# Step runners — Sparse Octree
# ---------------------------------------------------------------------------


def _build_pairs_sparse_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.octree.build_pairs import (
        main as pairs_main,
    )  # noqa: PLC0415

    data = p.get("data", {})
    argv = ["--sparse-octree", "--val-split", str(data.get("val_split", 0.1))]
    if data.get("data_dir"):
        argv += ["--data-dir", str(data["data_dir"])]
    pairs_main(argv)


def _train_sparse_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.train import train_sparse_octree  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    data_dir = data.get("data_dir", "noise_training_data")
    out_path = Path(train.get("output_dir", ".")) / "sparse_octree_model.pt"
    train_sparse_octree(
        data_path=Path(data_dir) / "sparse_octree_pairs.npz",
        out_path=out_path,
        model_variant=train.get("sparse_octree_variant", "fast"),
        hidden=train.get("sparse_octree_hidden", 80),
        epochs=train.get("epochs", 20),
        batch_size=train.get("batch_size", 4),
        lr=train.get("lr", 1e-4),
        device=train.get("device", "auto"),
    )


def _export_sparse_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.export_sparse_octree import (
        export_sparse_octree,
    )  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    export_sparse_octree(
        checkpoint=Path(train.get("output_dir", ".")) / "sparse_octree_model.pt",
        out_dir=Path(export.get("output_dir") or "LODiffusion/run/models"),
    )


def _deploy_sparse_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.export_sparse_octree import (
        export_sparse_octree,
    )  # noqa: PLC0415

    deploy = p.get("deploy", {})
    out_dir = (
        deploy.get("target_dir")
        or p.get("export", {}).get("output_dir")
        or "LODiffusion/run/models"
    )
    export_sparse_octree(
        checkpoint=Path(p.get("train", {}).get("output_dir", ".")) / "sparse_octree_model.pt",
        out_dir=Path(out_dir),
    )


def _distill_sparse_octree_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.distill import distill_sparse_octree  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    data_dir = data.get("data_dir", "noise_training_data")
    teacher_dir = train.get("output_dir", ".")
    distill_sparse_octree(
        teacher_checkpoint=Path(teacher_dir) / "sparse_octree_model.pt",
        data_path=Path(data_dir) / "sparse_octree_pairs.npz",
        out_path=Path(teacher_dir) / "sparse_octree_distilled.pt",
        student_variant=train.get("sparse_octree_variant", "fast"),
        student_hidden=train.get("sparse_octree_hidden", 80),
    )


# ---------------------------------------------------------------------------
# Step runners — v7 Data Acquisition
# ---------------------------------------------------------------------------


def _dumpnoise_v7_run(p: dict[str, Any]) -> None:
    """Dump v7 noise fields (15 RouterField channels, 4×4×4) via RCON."""
    from voxel_tree.gui.server_manager import get_rcon_settings  # noqa: PLC0415
    from voxel_tree.preprocessing.cli import main as cli_main  # noqa: PLC0415

    world = p.get("world", {})
    rcon = get_rcon_settings()
    rcon_timeout = p.get("rcon", {}).get("timeout", 3600)
    try:
        cli_main(
            [
                "dumpnoise",
                "--v7",
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
        )
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Dumpnoise v7 failed with exit code {e.code}") from e


def _build_v7_pairs_run(p: dict[str, Any]) -> None:
    """Build v7 training pairs NPZ from v7 section dumps."""
    from voxel_tree.tasks.sparse_octree.build_sparse_octree_pairs import (  # noqa: PLC0415
        main as pairs_main,
    )

    data = p.get("data", {})
    argv: list[str] = []
    dumps_dir = data.get("v7_dumps_dir", "v7_dumps")
    argv += ["--dumps", str(dumps_dir)]
    if data.get("v7_pairs_output"):
        argv += ["--output", str(data["v7_pairs_output"])]
    pairs_main(argv)


# ---------------------------------------------------------------------------
# Step runners — v7 Density V2
# ---------------------------------------------------------------------------


def _build_pairs_density_v2_run(p: dict[str, Any]) -> None:
    """Validate v7 pairs NPZ exists (pairs built by shared build_v7_pairs)."""
    data = p.get("data", {})
    npz = Path(data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz"))
    if not npz.exists():
        raise FileNotFoundError(f"v7 pairs NPZ not found: {npz}")
    print(f"[density_v2] v7 pairs validated: {npz}")


def _train_density_v2_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.density_v2.train_density_v2 import main as train_main  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    argv: list[str] = []
    npz = data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz")
    argv += ["--data", str(npz)]
    if train.get("output_dir"):
        argv += ["--out-dir", str(train["output_dir"])]
    if train.get("epochs") is not None:
        argv += ["--epochs", str(train["epochs"])]
    if train.get("batch_size") is not None:
        argv += ["--batch-size", str(train["batch_size"])]
    if train.get("lr") is not None:
        argv += ["--lr", str(train["lr"])]
    train_main(argv)


def _export_density_v2_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.density_v2.export_density_v2 import main as export_main  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    checkpoint = Path(train.get("output_dir", ".")) / "density_v2_best.pt"
    out_dir = Path(export.get("output_dir") or "LODiffusion/run/models")
    export_main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])


def _deploy_density_v2_run(p: dict[str, Any]) -> None:
    """Deploy density_v2 (re-export to deploy target dir)."""
    _export_density_v2_run(p)


# ---------------------------------------------------------------------------
# Step runners — v7 Biome Classifier
# ---------------------------------------------------------------------------


def _build_pairs_biome_run(p: dict[str, Any]) -> None:
    """Validate v7 pairs NPZ exists (pairs built by shared build_v7_pairs)."""
    data = p.get("data", {})
    npz = Path(data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz"))
    if not npz.exists():
        raise FileNotFoundError(f"v7 pairs NPZ not found: {npz}")
    print(f"[biome_classifier] v7 pairs validated: {npz}")


def _train_biome_classifier_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.biome.train_biome_classifier import main as train_main  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    argv: list[str] = []
    npz = data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz")
    argv += ["--data", str(npz)]
    if train.get("output_dir"):
        argv += ["--out-dir", str(train["output_dir"])]
    if train.get("epochs") is not None:
        argv += ["--epochs", str(train["epochs"])]
    if train.get("batch_size") is not None:
        argv += ["--batch-size", str(train["batch_size"])]
    if train.get("lr") is not None:
        argv += ["--lr", str(train["lr"])]
    train_main(argv)


def _export_biome_classifier_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.biome.export_biome import main as export_main  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    checkpoint = Path(train.get("output_dir", ".")) / "biome_classifier_best.pt"
    out_dir = Path(export.get("output_dir") or "LODiffusion/run/models")
    export_main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])


def _deploy_biome_classifier_run(p: dict[str, Any]) -> None:
    """Deploy biome_classifier (re-export to deploy target dir)."""
    _export_biome_classifier_run(p)


# ---------------------------------------------------------------------------
# Step runners — v7 Heightmap Predictor
# ---------------------------------------------------------------------------


def _build_pairs_heightmap_run(p: dict[str, Any]) -> None:
    """Validate v7 pairs NPZ exists (pairs built by shared build_v7_pairs)."""
    data = p.get("data", {})
    npz = Path(data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz"))
    if not npz.exists():
        raise FileNotFoundError(f"v7 pairs NPZ not found: {npz}")
    print(f"[heightmap_predictor] v7 pairs validated: {npz}")


def _train_heightmap_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.heightmap.train_heightmap import main as train_main  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    argv: list[str] = []
    npz = data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz")
    argv += ["--data", str(npz)]
    if train.get("output_dir"):
        argv += ["--out-dir", str(train["output_dir"])]
    if train.get("epochs") is not None:
        argv += ["--epochs", str(train["epochs"])]
    if train.get("batch_size") is not None:
        argv += ["--batch-size", str(train["batch_size"])]
    if train.get("lr") is not None:
        argv += ["--lr", str(train["lr"])]
    train_main(argv)


def _export_heightmap_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.heightmap.export_heightmap import main as export_main  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    checkpoint = Path(train.get("output_dir", ".")) / "heightmap_predictor_best.pt"
    out_dir = Path(export.get("output_dir") or "LODiffusion/run/models")
    export_main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])


def _deploy_heightmap_run(p: dict[str, Any]) -> None:
    """Deploy heightmap_predictor (re-export to deploy target dir)."""
    _export_heightmap_run(p)


# ---------------------------------------------------------------------------
# Step runners — v7 Sparse Octree
# ---------------------------------------------------------------------------


def _build_pairs_sparse_octree_v7_run(p: dict[str, Any]) -> None:
    """Validate v7 pairs NPZ exists (pairs built by shared build_v7_pairs)."""
    data = p.get("data", {})
    npz = Path(data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz"))
    if not npz.exists():
        raise FileNotFoundError(f"v7 pairs NPZ not found: {npz}")
    print(f"[sparse_octree_v7] v7 pairs validated: {npz}")


def _train_sparse_octree_v7_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.train import train_sparse_octree  # noqa: PLC0415

    data = p.get("data", {})
    train = p.get("train", {})
    npz = data.get("v7_pairs_npz", "sparse_octree_pairs_v7.npz")
    out_path = Path(train.get("output_dir", ".")) / "sparse_octree_v7_model.pt"
    train_sparse_octree(
        data_path=Path(npz),
        out_path=out_path,
        model_variant=train.get("sparse_octree_variant", "fast"),
        hidden=train.get("sparse_octree_hidden", 80),
        epochs=train.get("epochs", 20),
        batch_size=train.get("batch_size", 4),
        lr=train.get("lr", 1e-4),
        device=train.get("device", "auto"),
    )


def _export_sparse_octree_v7_run(p: dict[str, Any]) -> None:
    from voxel_tree.tasks.sparse_octree.export_sparse_octree import (
        export_sparse_octree,
    )  # noqa: PLC0415

    train = p.get("train", {})
    export = p.get("export", {})
    export_sparse_octree(
        checkpoint=Path(train.get("output_dir", ".")) / "sparse_octree_v7_model.pt",
        out_dir=Path(export.get("output_dir") or "LODiffusion/run/models"),
        n3d=15,
        spatial_y=4,
    )


def _deploy_sparse_octree_v7_run(p: dict[str, Any]) -> None:
    """Deploy sparse_octree_v7 (re-export to deploy target dir)."""
    _export_sparse_octree_v7_run(p)


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
    ModelTrack(
        track_id="sparse_octree",
        label="SparseOctree",
        swim_lane_color="#2a1500",
        build_pairs_factory=_build_pairs_sparse_octree_run,
        train_factory=_train_sparse_octree_run,
        export_factory=_export_sparse_octree_run,
        deploy_factory=_deploy_sparse_octree_run,
        build_pairs_consumes=frozenset({"voxy_db", "noise_dumps"}),
        contract_name="sparse_octree",
        contract_revision=0,
        extra_steps=[
            StepDef(
                id="distill_sparse_octree",
                label="Distill",
                prereqs=[],
                run_fn=_distill_sparse_octree_run,
                track="sparse_octree",
                phase="distill",
                produces=frozenset({"sparse_octree_distilled"}),
                consumes=frozenset({"sparse_octree_checkpoint", "sparse_octree_pairs"}),
            ),
        ],
    ),
    # ── Terrain Shaper + Density NN ───────────────────────────
    # Predates ModelTrack; id_overrides preserve legacy run-state JSON keys.
    # build_pairs_consumes={"noise_dumps"} because the training script reads
    # the noise-dump JSONs directly — no explicit pair-building step is needed.
    # export is via extract_terrain_shaper_weights (uses --no-deploy).
    # deploy runs extract_terrain_shaper_weights without --no-deploy to push into LODiffusion resources.
    ModelTrack(
        track_id="terrain_shaper",
        label="TerrainShaper",
        swim_lane_color="#00202a",
        build_pairs_factory=_build_pairs_terrain_shaper_run,
        train_factory=_train_terrain_shaper_density_run,
        export_factory=_extract_terrain_shaper_weights_run,
        deploy_factory=_deploy_terrain_shaper_run,
        contract_name="terrain_shaper",
        contract_revision=0,
        id_overrides={
            "build_pairs": "build_pairs_terrain_shaper",
            "train": "train_terrain_shaper_density",
            "export": "extract_terrain_shaper_weights",
            "deploy": "deploy_terrain_shaper",
        },
        build_pairs_consumes=frozenset({"noise_dumps"}),
        extra_steps=[
            StepDef(
                id="distill_density",
                label="Distill",
                prereqs=[],
                run_fn=_distill_density_run,
                track="terrain_shaper",
                phase="distill",
                produces=frozenset({"terrain_shaper_distilled"}),
                consumes=frozenset({"terrain_shaper_checkpoint"}),
            ),
            StepDef(
                id="train_terrain_shaper",
                label="T·Shaper",
                prereqs=[],
                run_fn=_train_terrain_shaper_run,
                track="terrain_shaper",
                phase="train_terrain_shaper",
                produces=frozenset({"terrain_shaper_spline_checkpoint"}),
                consumes=frozenset({"noise_dumps"}),
            ),
        ],
    ),
    # ── v7 Density V2 (climate → density prediction) ─────────────────
    ModelTrack(
        track_id="density_v2",
        label="DensityV2",
        swim_lane_color="#1a2a00",
        build_pairs_factory=_build_pairs_density_v2_run,
        train_factory=_train_density_v2_run,
        export_factory=_export_density_v2_run,
        deploy_factory=_deploy_density_v2_run,
        build_pairs_consumes=frozenset({"v7_pairs_npz"}),
        contract_name="density",
        contract_revision=1,
    ),
    # ── v7 Biome Classifier (climate → biome class) ──────────────────
    ModelTrack(
        track_id="biome_classifier",
        label="BiomeClass",
        swim_lane_color="#002a1a",
        build_pairs_factory=_build_pairs_biome_run,
        train_factory=_train_biome_classifier_run,
        export_factory=_export_biome_classifier_run,
        deploy_factory=_deploy_biome_classifier_run,
        build_pairs_consumes=frozenset({"v7_pairs_npz"}),
        contract_name="biome",
        contract_revision=1,
    ),
    # ── v7 Heightmap Predictor (climate → heightmaps) ────────────────
    ModelTrack(
        track_id="heightmap_predictor",
        label="Heightmap",
        swim_lane_color="#0a0a2a",
        build_pairs_factory=_build_pairs_heightmap_run,
        train_factory=_train_heightmap_run,
        export_factory=_export_heightmap_run,
        deploy_factory=_deploy_heightmap_run,
        build_pairs_consumes=frozenset({"v7_pairs_npz"}),
        contract_name="heightmap",
        contract_revision=1,
    ),
    # ── v7 Sparse Octree (15ch/4×4×4 noise → block hierarchy) ───────
    ModelTrack(
        track_id="sparse_octree_v7",
        label="OctreeV7",
        swim_lane_color="#2a0a00",
        build_pairs_factory=_build_pairs_sparse_octree_v7_run,
        train_factory=_train_sparse_octree_v7_run,
        export_factory=_export_sparse_octree_v7_run,
        deploy_factory=_deploy_sparse_octree_v7_run,
        build_pairs_consumes=frozenset({"v7_pairs_npz"}),
        contract_name="sparse_octree",
        contract_revision=1,
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
        run_fn=_pregen_run,
        server_required=True,
        phase="data_acq",
        produces=frozenset({"pregenerated_chunks"}),
        consumes=frozenset(),
    ),
    StepDef(
        id="harvest",
        label="Harvest",
        prereqs=[],
        run_fn=_harvest_run,
        server_required=True,
        client_required=True,
        phase="data_acq",
        produces=frozenset({"voxy_db"}),
        consumes=frozenset({"pregenerated_chunks"}),
    ),
    StepDef(
        id="dumpnoise",
        label="Noise",
        prereqs=[],
        run_fn=_dumpnoise_run,
        server_required=True,
        phase="data_acq",
        produces=frozenset({"noise_dumps"}),
        consumes=frozenset(),
    ),
    StepDef(
        id="extract_octree",
        label="Extract",
        prereqs=[],
        run_fn=_extract_octree_run,
        phase="data_acq",
        produces=frozenset({"octree_npz"}),
        consumes=frozenset({"voxy_db"}),
    ),
    StepDef(
        id="column_heights",
        label="Heights",
        prereqs=[],
        run_fn=_column_heights_run,
        phase="data_acq",
        produces=frozenset({"octree_with_heights"}),
        consumes=frozenset({"octree_npz", "noise_dumps"}),
    ),
    # ── v7 data acquisition ───────────────────────────────────────────
    StepDef(
        id="dumpnoise_v7",
        label="Noise·v7",
        prereqs=[],
        run_fn=_dumpnoise_v7_run,
        server_required=True,
        phase="data_acq",
        produces=frozenset({"v7_noise_dumps"}),
        consumes=frozenset(),
    ),
    StepDef(
        id="build_v7_pairs",
        label="Pairs·v7",
        prereqs=[],
        run_fn=_build_v7_pairs_run,
        phase="data_acq",
        produces=frozenset({"v7_pairs_npz"}),
        consumes=frozenset({"v7_noise_dumps"}),
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
