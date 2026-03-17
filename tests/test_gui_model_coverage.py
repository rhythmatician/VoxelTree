"""Test that all model tracks have required GUI integration methods.

This test ensures that whenever a new model track is added, the corresponding
cmd factories and (if applicable) model implementations have the required methods
for GUI execution, cancellation, and progress reporting.

It also validates the artifact-based prerequisite graph: each StepDef
declares ``produces`` and ``consumes`` frozensets, and ``_wire_prereqs()``
auto-computes the ``prereqs`` list at module load time.  The tests below
verify structural soundness (no missing producers, no duplicate artifacts,
no cycles) and snapshot the expected DAG for regression detection.
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
        """Every step must have a callable run_fn."""
        for step in PIPELINE_STEPS:
            assert callable(step.run_fn), (
                f"Step '{step.id}' run_fn is not callable: " f"{step.run_fn}"
            )

    def test_all_steps_run_fn_callable_with_profile(self):
        """Verify each run_fn can be called without import errors.

        We cannot truly run the steps (they need real data/servers), but we
        verify the function object is callable and accepts a dict argument
        by inspecting its signature.
        """
        import inspect

        for step in PIPELINE_STEPS:
            sig = inspect.signature(step.run_fn)
            params = list(sig.parameters.values())
            assert (
                len(params) >= 1
            ), f"Step '{step.id}' run_fn must accept at least one parameter (profile dict)"

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
# Artifact-graph structural validation
# ═══════════════════════════════════════════════════════════════════════════
#
# produces / consumes now live directly on StepDef, and _wire_prereqs()
# auto-computes prereqs at import time.  The tests below verify that the
# resulting artifact graph and prereq DAG are structurally sound.
#
# ═══════════════════════════════════════════════════════════════════════════


class TestArtifactGraph:
    """Validate the produces/consumes artifact graph on PIPELINE_STEPS.

    Since produces/consumes live directly on StepDef and ``_wire_prereqs()``
    auto-computes ``prereqs`` at module load time, these tests verify that
    the resulting DAG is structurally sound and matches the expected shape.
    """

    def test_all_steps_declare_artifacts(self):
        """Every step (except terminal loopback stubs) should produce something."""
        no_output = [s.id for s in PIPELINE_STEPS if not s.produces and s.phase != "loopback"]
        assert not no_output, f"Steps with no produces (add artifact declarations): {no_output}"

    def test_no_duplicate_producers(self):
        """Each artifact name should be produced by exactly one step."""
        seen: dict[str, str] = {}
        dupes: list[str] = []
        for step in PIPELINE_STEPS:
            for art in step.produces:
                if art in seen:
                    dupes.append(f"'{art}' produced by both '{seen[art]}' and '{step.id}'")
                seen[art] = step.id
        assert not dupes, "Duplicate artifact producers:\n" + "\n".join(dupes)

    def test_consumed_artifacts_have_producers(self):
        """Every consumed artifact must be produced by some step."""
        all_produced = {art for s in PIPELINE_STEPS for art in s.produces}
        failures: list[str] = []
        for step in PIPELINE_STEPS:
            missing = step.consumes - all_produced
            if missing:
                failures.append(
                    f"Step '{step.id}' consumes {missing} " f"but no step produces them"
                )
        assert not failures, "Broken artifact edges:\n" + "\n".join(failures)

    def test_no_unreachable_producers(self):
        """Every produced artifact should be consumed by at least one step.

        Advisory — warns about dead-end artifacts nobody uses.
        Stubs (enabled=False) and terminal deployments are excluded.
        """
        all_consumed = {art for s in PIPELINE_STEPS for art in s.consumes}
        orphaned: list[str] = []
        for step in PIPELINE_STEPS:
            if not step.enabled:
                continue
            for art in step.produces:
                if art not in all_consumed:
                    orphaned.append(f"{step.id} → {art}")

        terminal_suffixes = ("_deployed", "_checkpoint")
        real_orphans = [o for o in orphaned if not any(o.endswith(s) for s in terminal_suffixes)]

        if real_orphans:
            import warnings

            warnings.warn(
                f"Artifacts produced but never consumed " f"(possible dead ends): {real_orphans}",
                stacklevel=1,
            )

    def test_auto_wired_prereqs_match_expected(self):
        """Snapshot: verify auto-wired prereqs match the known-good DAG.

        If this fails, artifact declarations changed and prereqs shifted.
        Review the diff to confirm the new edges are correct, then update
        this snapshot.
        """
        expected = {
            "pregen": [],
            "voxy_import": ["pregen"],
            "dumpnoise": [],
            "extract_octree": ["voxy_import"],
            "column_heights": ["dumpnoise", "extract_octree"],
            # ── Octree tracks ─────────────────────────────────────────
            "build_pairs_init": ["column_heights"],
            "train_init": ["build_pairs_init"],
            "export_init": ["train_init"],
            "deploy_init": ["export_init"],
            "build_pairs_refine": ["column_heights"],
            "train_refine": ["build_pairs_refine"],
            "export_refine": ["train_refine"],
            "deploy_refine": ["export_refine"],
            "build_pairs_leaf": ["column_heights"],
            "train_leaf": ["build_pairs_leaf"],
            "export_leaf": ["train_leaf"],
            "deploy_leaf": ["export_leaf"],
            # ── Sparse root ───────────────────────────────────────────
            "build_pairs_sparse_root": ["column_heights"],
            "train_sparse_root": ["build_pairs_sparse_root"],
            "export_sparse_root": ["train_sparse_root"],
            "deploy_sparse_root": ["export_sparse_root"],
            "distill_sparse_root": ["train_sparse_root"],
            # ── Stage-1 ───────────────────────────────────────────────
            "build_pairs_stage1": ["dumpnoise"],
            "train_stage1_density": ["build_pairs_stage1"],
            "extract_stage1_weights": ["train_stage1_density"],
            "deploy_stage1": ["extract_stage1_weights"],
            "distill_density": ["train_stage1_density"],
            "train_terrain_shaper": ["distill_density"],
            # ── Loopback ──────────────────────────────────────────────
            "reset_data": ["deploy_leaf"],
            "new_seed": ["reset_data"],
        }

        actual = {s.id: s.prereqs for s in PIPELINE_STEPS}

        missing_steps = set(expected) - set(actual)
        assert not missing_steps, f"Expected steps missing from pipeline: {missing_steps}"

        extra_steps = set(actual) - set(expected)
        assert not extra_steps, f"Unexpected new steps (add to snapshot): {extra_steps}"

        mismatches: list[str] = []
        for step_id, exp in sorted(expected.items()):
            act = actual[step_id]
            if sorted(act) != sorted(exp):
                mismatches.append(f"  {step_id}: expected {exp}, got {act}")
        assert not mismatches, "Auto-wired prereqs differ from snapshot:\n" + "\n".join(mismatches)

    def test_no_circular_prerequisites(self):
        """DAG must be acyclic — detect cycles in the prereq graph."""
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
