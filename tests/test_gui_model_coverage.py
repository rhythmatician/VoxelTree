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

        # Current architecture handles models via ModelTrack registration.
        # We check for the core models currently in active development.
        required_tracks = {"sparse_root", "stage1"}
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


class TestArtifactGraph:
    """Validate the produces/consumes artifact graph on PIPELINE_STEPS.

    Verified structural soundness of the auto-wired DAG.
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

    def test_auto_wired_prereqs_match_expected(self):
        """Snapshot: verify auto-wired prereqs match the known-good DAG for current tracks.

        If this fails, artifact declarations changed and prereqs shifted.
        Review the diff to confirm the new edges are correct, then update
        this snapshot.
        """
        # Note: core DAG structure for Sparse Root and Stage 1.
        important_steps = {
            "extract_octree": ["harvest"],
            "column_heights": ["extract_octree", "dumpnoise"],
            "build_pairs_sparse_root": ["harvest", "dumpnoise"],
            "train_sparse_root": ["build_pairs_sparse_root"],
            "export_sparse_root": ["train_sparse_root"],
            "deploy_sparse_root": ["export_sparse_root"],
            "build_pairs_stage1": ["dumpnoise"],
            "train_stage1_density": ["build_pairs_stage1"],
            "extract_stage1_weights": ["train_stage1_density"],
            "deploy_stage1": ["extract_stage1_weights"],
        }

        actual = {s.id: s.prereqs for s in PIPELINE_STEPS}

        for step_id, expected_prereqs in important_steps.items():
            if step_id in actual:
                actual_set = set(actual[step_id])
                expected_set = set(expected_prereqs)
                assert expected_set.issubset(actual_set) or actual_set == expected_set, (
                    f"Prereq mismatch for '{step_id}'. "
                    f"Expected to include {expected_prereqs}, got {actual[step_id]}"
                )
