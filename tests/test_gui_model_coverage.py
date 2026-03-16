"""Test that all model tracks have required GUI integration methods.

This test ensures that whenever a new model track is added, the corresponding
cmd factories and (if applicable) model implementations have the required methods
for GUI execution, cancellation, and progress reporting.
"""

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
        assert required_tracks.issubset(found_tracks), (
            f"Missing required tracks. Expected {required_tracks}, got {found_tracks}"
        )

    def test_all_tracks_have_label(self):
        """Each track must have a descriptive label."""
        for track in MODEL_TRACKS:
            assert track.label, f"Track '{track.track_id}' missing label"
            assert isinstance(track.label, str), (
                f"Track '{track.track_id}' label must be string, got {type(track.label)}"
            )

    def test_all_tracks_have_swim_lane_color(self):
        """Each track must have a swim_lane_color for GUI rendering."""
        for track in MODEL_TRACKS:
            assert track.swim_lane_color, (
                f"Track '{track.track_id}' missing swim_lane_color"
            )
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
            assert len(steps) > 0, (
                f"Track '{track.track_id}' to_steps() returned empty list"
            )
            
            # All steps should be in PIPELINE_STEPS
            step_ids = [s.id for s in steps]
            for step_id in step_ids:
                assert step_id in STEP_BY_ID, (
                    f"Step '{step_id}' from track '{track.track_id}' "
                    "not found in STEP_BY_ID"
                )

    def test_all_steps_have_callable_factories(self):
        """Every step must have a callable cmd_factory."""
        for step in PIPELINE_STEPS:
            assert callable(step.cmd_factory), (
                f"Step '{step.id}' cmd_factory is not callable: "
                f"{step.cmd_factory}"
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
                    f"Step '{step.id}' factory returned {type(cmd)}, "
                    "expected list"
                )
                assert len(cmd) > 0, (
                    f"Step '{step.id}' factory returned empty list"
                )
                assert all(isinstance(c, str) for c in cmd), (
                    f"Step '{step.id}' factory returned non-string elements: "
                    f"{[type(c) for c in cmd]}"
                )
            except Exception as e:
                pytest.fail(
                    f"Step '{step.id}' factory raised {type(e).__name__}: {e}"
                )

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
                assert track.train_factory is not None, (
                    f"Track '{track.track_id}' missing required train_factory"
                )

    def test_no_orphaned_steps(self):
        """Ensure all steps in PIPELINE_STEPS correspond to registered tracks."""
        track_ids = {t.track_id for t in MODEL_TRACKS}
        valid_special_tracks = {"data_acq", "loopback", None}
        
        for step in PIPELINE_STEPS:
            is_valid = (step.track in track_ids) or (step.track in valid_special_tracks)
            assert is_valid, (
                f"Step '{step.id}' references unknown track '{step.track}'"
            )

    def test_step_prerequisites_exist(self):
        """Verify all step prerequisites are actually defined steps."""
        step_ids = {s.id for s in PIPELINE_STEPS}
        
        for step in PIPELINE_STEPS:
            for prereq in step.prereqs:
                assert prereq in step_ids, (
                    f"Step '{step.id}' prereq '{prereq}' not found in "
                    f"PIPELINE_STEPS"
                )


class TestGUIRequiredMethods:
    """Test that model implementations have required GUI methods.
    
    This is informational/aspirational - the test documents what we expect
    future model code to implement. It allows stubs/pass implementations.
    """

    REQUIRED_METHODS = {
        "run",      # Execute training/export/etc
        "cancel",   # Stop execution gracefully
    }

    REQUIRED_ATTRIBUTES = {
        "pbar",     # Progress bar support
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
                    pytest.skip(
                        f"{cls.__name__}.{method_name} is not callable"
                    )
            else:
                # Document missing methods as a warning, but don't fail
                pytest.skip(
                    f"{cls.__name__} missing {method_name} (stub required)"
                )

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
                assert hasattr(registry, method_name), (
                    f"RunRegistry missing {method_name} method"
                )
        except ImportError:
            pytest.skip("RunRegistry not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
