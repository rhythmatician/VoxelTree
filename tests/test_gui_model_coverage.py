"""Test that all model tracks have required GUI integration methods.

This test ensures that whenever a new model track is added, the corresponding
cmd factories and (if applicable) model implementations have the required methods
for GUI execution, cancellation, and progress reporting.

It also validates prerequisite wiring via a lightweight artifact manifest:
each step declares what it *produces* and *consumes*, and the test verifies
that every consumed artifact has a producer among the step's transitive
prerequisite chain.  This catches broken prereqs (drift) without trying to
auto-discover edges from code.
"""

from __future__ import annotations

import pytest

from VoxelTree.gui.step_definitions import (
    MODEL_TRACKS,
    PIPELINE_STEPS,
    STEP_BY_ID,
)


class TestModelTrackCoverage:
    """Validate all model tracks are properly configured for GUI integration."""

    def test_all_model_tracks_registered(self):
        """Verify each model track is in the global registry."""
        assert len(MODEL_TRACKS) > 0, "No MODEL_TRACKS registered"

        track_ids = [t.track_id for t in MODEL_TRACKS]
        assert len(track_ids) == len(set(track_ids)), "Duplicate track IDs found"

        # Known tracks that should exist
        required_tracks = {"init", "refine", "leaf", "sparse_root", "stage1"}
        found_tracks = set(track_ids)
        assert required_tracks.issubset(
            found_tracks
        ), f"Missing required tracks. Expected {required_tracks}, got {found_tracks}"

    def test_all_tracks_have_label(self):
        """Each track must have a descriptive label."""
        for track in MODEL_TRACKS:
            assert track.label, f"Track '{track.track_id}' missing label"
            assert isinstance(
                track.label, str
            ), f"Track '{track.track_id}' label must be string, got {type(track.label)}"

    def test_all_tracks_have_swim_lane_color(self):
        """Each track must have a swim_lane_color for GUI rendering."""
        for track in MODEL_TRACKS:
            assert track.swim_lane_color, f"Track '{track.track_id}' missing swim_lane_color"
            # Validate hex color format
            color = track.swim_lane_color.lstrip("#")
            assert len(color) == 6, (
                f"Track '{track.track_id}' color '{track.swim_lane_color}' "
                "not valid hex (expected #RRGGBB)"
            )
            assert all(c in "0123456789abcdefABCDEF" for c in color), (
                f"Track '{track.track_id}' color '{track.swim_lane_color}' "
                "contains invalid hex characters"
            )

    def test_all_tracks_generate_steps(self):
        """Each track must generate at least one step via to_steps()."""
        for track in MODEL_TRACKS:
            steps = track.to_steps()
            assert len(steps) > 0, f"Track '{track.track_id}' to_steps() returned empty list"

            # All steps should be in PIPELINE_STEPS
            step_ids = [s.id for s in steps]
            for step_id in step_ids:
                assert step_id in STEP_BY_ID, (
                    f"Step '{step_id}' from track '{track.track_id}' " "not found in STEP_BY_ID"
                )

    def test_all_steps_have_callable_factories(self):
        """Every step must have a callable cmd_factory."""
        for step in PIPELINE_STEPS:
            assert callable(step.cmd_factory), (
                f"Step '{step.id}' cmd_factory is not callable: " f"{step.cmd_factory}"
            )

    def test_all_steps_factory_returns_list(self):
        """Verify each factory can be called and returns a list of strings."""
        test_profile = {
            "name": "test_profile",
            "data": {
                "data_dir": "test_data",
                "noise_dump_dir": "test_noise",
                "stage1_dump_dir": "test_stage1",
                "val_split": 0.1,
            },
            "train": {
                "output_dir": "test_output",
                "epochs": 5,
                "batch_size": 2,
                "lr": 1e-4,
                "device": "cpu",
                "sparse_root_variant": "fast",
                "sparse_root_hidden": 80,
            },
            "extract": {"output_dir": "test_extract"},
            "distill": {
                "teacher": "unet",
                "student": "sep",
                "epochs": 10,
                "alpha": 0.5,
                "lr": 2e-3,
            },
            "export": {"output_dir": "test_export"},
            "deploy": {"target_dir": "test_deploy"},
        }

        for step in PIPELINE_STEPS:
            try:
                cmd = step.cmd_factory(test_profile)
                assert isinstance(cmd, list), (
                    f"Step '{step.id}' factory returned {type(cmd)}, " "expected list"
                )
                assert len(cmd) > 0, f"Step '{step.id}' factory returned empty list"
                assert all(isinstance(c, str) for c in cmd), (
                    f"Step '{step.id}' factory returned non-string elements: "
                    f"{[type(c) for c in cmd]}"
                )
            except Exception as e:
                pytest.fail(f"Step '{step.id}' factory raised {type(e).__name__}: {e}")

    def test_track_factories_defined_for_phases(self):
        """Validate that tracks have required factories for their model phases.

        Not all tracks have all phases:
        - octree (init/refine/leaf) use the shared CLI interface
        - stage1 is special (no data pairing phase)
        - sparse_root may have stubs for export/deploy
        """
        # Tracks that MUST have train
        required_to_have_train = {"init", "refine", "leaf", "sparse_root", "stage1"}
        for track in MODEL_TRACKS:
            if track.track_id in required_to_have_train:
                assert (
                    track.train_factory is not None
                ), f"Track '{track.track_id}' missing required train_factory"

    def test_no_orphaned_steps(self):
        """Ensure all steps in PIPELINE_STEPS correspond to registered tracks."""
        track_ids = {t.track_id for t in MODEL_TRACKS}
        valid_special_tracks = {"data_acq", "loopback", None}

        for step in PIPELINE_STEPS:
            is_valid = (step.track in track_ids) or (step.track in valid_special_tracks)
            assert is_valid, f"Step '{step.id}' references unknown track '{step.track}'"

    def test_step_prerequisites_exist(self):
        """Verify all step prerequisites are actually defined steps."""
        step_ids = {s.id for s in PIPELINE_STEPS}

        for step in PIPELINE_STEPS:
            for prereq in step.prereqs:
                assert prereq in step_ids, (
                    f"Step '{step.id}' prereq '{prereq}' not found in " f"PIPELINE_STEPS"
                )


class TestGUIRequiredMethods:
    """Test that model implementations have required GUI methods.

    This is informational/aspirational - the test documents what we expect
    future model code to implement. It allows stubs/pass implementations.
    """

    REQUIRED_METHODS = {
        "run",  # Execute training/export/etc
        "cancel",  # Stop execution gracefully
    }

    REQUIRED_ATTRIBUTES = {
        "pbar",  # Progress bar support
    }

    def test_sparse_root_model_has_stubs(self):
        """Check sparse_root model has required method stubs."""
        try:
            from LODiffusion.models.sparse_root import SparseRootModel

            self._check_class_for_methods(SparseRootModel)
        except ImportError:
            pytest.skip("LODiffusion sparse_root model not available")

    def test_sparse_root_train_has_stubs(self):
        """Check sparse_root training module has required method stubs."""
        try:
            # Just check the module is importable
            import VoxelTree.core.sparse_root_train  # noqa: F401

            # Doesn't need methods - just a data class
            pass
        except ImportError:
            pytest.skip("VoxelTree sparse_root_train module not available")

    def _check_class_for_methods(self, cls):
        """Helper: verify class has required methods/attributes (stubs OK)."""
        for method_name in self.REQUIRED_METHODS:
            # Allow either real implementation or stub (pass)
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                if callable(method):
                    # Just check it exists and is callable
                    pass
                else:
                    pytest.skip(f"{cls.__name__}.{method_name} is not callable")
            else:
                # Document missing methods as a warning, but don't fail
                pytest.skip(f"{cls.__name__} missing {method_name} (stub required)")

    def test_gui_runner_integration(self):
        """Verify RunRegistry can handle all model tracks."""
        # This is a sanity check that the GUI state management
        # integration points exist
        try:
            from VoxelTree.gui.run_registry import RunRegistry

            # RunRegistry should expose status and lifecycle methods
            registry = RunRegistry("test_profile")
            required_methods = ["get_status", "mark_started", "mark_success", "get_runnable_steps"]
            for method_name in required_methods:
                assert hasattr(registry, method_name), f"RunRegistry missing {method_name} method"
        except ImportError:
            pytest.skip("RunRegistry not available")


# ═══════════════════════════════════════════════════════════════════════════
# Artifact-manifest prereq validation
# ═══════════════════════════════════════════════════════════════════════════
#
# Each entry maps  step_id → {"produces": {artifacts}, "consumes": {artifacts}}
#
# Artifacts are short logical names (not file paths).  The test checks that
# every consumed artifact is produced by at least one transitive prereq.
# This table is intentionally maintained by hand — the whole point is to
# catch drift when future changes forget to update prereqs.
#
# To add a new step:
#   1. Add an entry here with produces / consumes.
#   2. If the test fails, you forgot to update prereqs in step_definitions.py.
#
# ═══════════════════════════════════════════════════════════════════════════

ARTIFACT_MANIFEST: dict[str, dict[str, set[str]]] = {
    # ── Data acquisition ──────────────────────────────────────────────
    "pregen": {
        "produces": {"mc_world"},
        "consumes": set(),
    },
    "voxy_import": {
        "produces": {"voxy_db"},
        "consumes": {"mc_world"},
    },
    "dumpnoise": {
        "produces": {"noise_dumps"},
        "consumes": set(),
    },
    "extract_octree": {
        "produces": {"octree_npz"},
        "consumes": {"voxy_db"},
    },
    "column_heights": {
        "produces": {"octree_with_heights"},
        "consumes": {"octree_npz", "noise_dumps"},
    },
    # ── Octree: Init ──────────────────────────────────────────────────
    "build_pairs_init": {
        "produces": {"init_pairs"},
        "consumes": {"octree_with_heights"},
    },
    "train_init": {
        "produces": {"init_checkpoint"},
        "consumes": {"init_pairs"},
    },
    "export_init": {
        "produces": {"init_onnx"},
        "consumes": {"init_checkpoint"},
    },
    "deploy_init": {
        "produces": {"init_deployed"},
        "consumes": {"init_onnx"},
    },
    # ── Octree: Refine ────────────────────────────────────────────────
    "build_pairs_refine": {
        "produces": {"refine_pairs"},
        "consumes": {"octree_with_heights"},
    },
    "train_refine": {
        "produces": {"refine_checkpoint"},
        "consumes": {"refine_pairs"},
    },
    "export_refine": {
        "produces": {"refine_onnx"},
        "consumes": {"refine_checkpoint"},
    },
    "deploy_refine": {
        "produces": {"refine_deployed"},
        "consumes": {"refine_onnx"},
    },
    # ── Octree: Leaf ──────────────────────────────────────────────────
    "build_pairs_leaf": {
        "produces": {"leaf_pairs"},
        "consumes": {"octree_with_heights"},
    },
    "train_leaf": {
        "produces": {"leaf_checkpoint"},
        "consumes": {"leaf_pairs"},
    },
    "export_leaf": {
        "produces": {"leaf_onnx"},
        "consumes": {"leaf_checkpoint"},
    },
    "deploy_leaf": {
        "produces": {"leaf_deployed"},
        "consumes": {"leaf_onnx"},
    },
    # ── Sparse Root ───────────────────────────────────────────────────
    "build_pairs_sparse_root": {
        "produces": {"sparse_root_pairs"},
        "consumes": {"octree_with_heights"},
    },
    "train_sparse_root": {
        "produces": {"sparse_root_checkpoint"},
        "consumes": {"sparse_root_pairs"},
    },
    "distill_sparse_root": {
        "produces": {"sparse_root_distilled"},
        "consumes": {"sparse_root_checkpoint", "sparse_root_pairs"},
    },
    "export_sparse_root": {
        "produces": {"sparse_root_exported"},
        "consumes": {"sparse_root_checkpoint"},
    },
    "deploy_sparse_root": {
        "produces": {"sparse_root_deployed"},
        "consumes": {"sparse_root_exported"},
    },
    # ── Stage-1 Density ───────────────────────────────────────────────
    "build_pairs_stage1": {
        "produces": set(),  # stage1 reads noise dumps directly
        "consumes": {"noise_dumps"},
    },
    "train_stage1_density": {
        "produces": {"stage1_checkpoint"},
        "consumes": {"noise_dumps"},
    },
    "extract_stage1_weights": {
        "produces": {"stage1_weights_bin"},
        "consumes": {"stage1_checkpoint"},
    },
    "distill_density": {
        "produces": {"stage1_distilled"},
        "consumes": {"stage1_checkpoint"},
    },
    "train_terrain_shaper": {
        "produces": {"terrain_shaper_checkpoint"},
        "consumes": {"stage1_distilled"},
    },
    "deploy_stage1": {
        "produces": {"stage1_deployed"},
        "consumes": {"stage1_weights_bin"},
    },
    # ── Loopback stubs ────────────────────────────────────────────────
    "reset_data": {
        "produces": set(),
        "consumes": {"leaf_deployed"},
    },
    "new_seed": {
        "produces": set(),
        "consumes": set(),
    },
}


def _transitive_prereqs(step_id: str) -> set[str]:
    """Walk the prereq DAG and return all transitive ancestors of *step_id*."""
    visited: set[str] = set()
    stack = [step_id]
    while stack:
        sid = stack.pop()
        if sid in visited:
            continue
        visited.add(sid)
        step = STEP_BY_ID.get(sid)
        if step:
            stack.extend(step.prereqs)
    visited.discard(step_id)  # exclude self
    return visited


def _all_artifacts_produced_by(step_ids: set[str]) -> set[str]:
    """Union of all artifacts produced by a set of steps."""
    arts: set[str] = set()
    for sid in step_ids:
        entry = ARTIFACT_MANIFEST.get(sid)
        if entry:
            arts |= entry["produces"]
    return arts


class TestPrereqDataFlow:
    """Validate that prereq wiring matches the artifact manifest.

    If this test fails, either:
      - A prereqs list in step_definitions.py is wrong (fix the prereqs), OR
      - The ARTIFACT_MANIFEST above is stale (update the manifest).
    Either way, it caught drift.
    """

    def test_manifest_covers_all_pipeline_steps(self):
        """Every PIPELINE_STEPS entry should appear in ARTIFACT_MANIFEST."""
        missing = [s.id for s in PIPELINE_STEPS if s.id not in ARTIFACT_MANIFEST]
        assert (
            not missing
        ), f"Steps missing from ARTIFACT_MANIFEST (add entries for them): {missing}"

    def test_manifest_has_no_stale_entries(self):
        """Every ARTIFACT_MANIFEST key should correspond to a real step."""
        step_ids = {s.id for s in PIPELINE_STEPS}
        stale = [k for k in ARTIFACT_MANIFEST if k not in step_ids]
        assert not stale, f"ARTIFACT_MANIFEST has entries for non-existent steps: {stale}"

    def test_consumed_artifacts_have_producers(self):
        """For each step, every consumed artifact must be produced by a transitive prereq."""
        failures: list[str] = []

        for step in PIPELINE_STEPS:
            entry = ARTIFACT_MANIFEST.get(step.id)
            if not entry:
                continue

            consumed = entry["consumes"]
            if not consumed:
                continue

            ancestors = _transitive_prereqs(step.id)
            available = _all_artifacts_produced_by(ancestors)

            missing = consumed - available
            if missing:
                failures.append(
                    f"Step '{step.id}' consumes {missing} but no transitive "
                    f"prereq produces them.  Ancestors: {sorted(ancestors)}"
                )

        assert not failures, "Prerequisite wiring drift detected:\n" + "\n".join(failures)

    def test_no_unreachable_producers(self):
        """Every produced artifact should be consumed by at least one downstream step.

        This is a soft check — warns about dead-end artifacts that nobody uses.
        Stubs (enabled=False) are excluded from this check.
        """
        # Collect all consumed artifacts
        all_consumed: set[str] = set()
        for entry in ARTIFACT_MANIFEST.values():
            all_consumed |= entry["consumes"]

        # Find artifacts produced but never consumed
        orphaned: list[str] = []
        for step_id, entry in ARTIFACT_MANIFEST.items():
            step = STEP_BY_ID.get(step_id)
            if step and not step.enabled:
                continue  # stubs get a pass
            for art in entry["produces"]:
                if art not in all_consumed:
                    orphaned.append(f"{step_id} → {art}")

        # Terminal artifacts (deployed models) are acceptable, filter those
        terminal_suffixes = ("_deployed",)
        real_orphans = [o for o in orphaned if not any(o.endswith(s) for s in terminal_suffixes)]

        # This is advisory — use a warning rather than a hard fail,
        # since some artifacts may truly be terminal outputs.
        if real_orphans:
            import warnings

            warnings.warn(
                f"Artifacts produced but never consumed (possible dead ends): " f"{real_orphans}",
                stacklevel=1,
            )

    def test_no_circular_prerequisites(self):
        """DAG must be acyclic — detect cycles in the prereq graph."""
        # Kahn's algorithm: topological sort and check all nodes visited.
        from collections import deque

        in_degree: dict[str, int] = {s.id: 0 for s in PIPELINE_STEPS}
        children: dict[str, list[str]] = {s.id: [] for s in PIPELINE_STEPS}

        for step in PIPELINE_STEPS:
            for prereq in step.prereqs:
                if prereq in children:
                    children[prereq].append(step.id)
                    in_degree[step.id] += 1

        queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        assert visited == len(in_degree), (
            f"Cycle detected in prereq graph!  "
            f"Visited {visited}/{len(in_degree)} steps.  "
            f"Stuck nodes: {[s for s, d in in_degree.items() if d > 0]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
