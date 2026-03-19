"""Tests for the GUI run registry persistence and loading.

These were added after a refactor busted the hardcoded runs root path.  The
registry must read and write JSON state files under the project root
``runs/`` directory and not the package subdirectory.
"""

from __future__ import annotations

import json
from pathlib import Path


import pytest  # noqa: E402

from voxel_tree.gui import run_registry

# we only need the constant for the server‑session test
from voxel_tree.gui.main_window import _SERVER_SESSION_STEPS
from voxel_tree.gui.run_registry import RunRegistry


def test_run_registry_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry should load existing state and persist changes.

    We monkeypatch ``_RUNS_ROOT`` to a temporary directory so the test does not
    interfere with the real workspace.  Creating a registry writes the file on
    demand and loading a new instance should read the previously saved values.
    """
    monkeypatch.setattr(run_registry, "_RUNS_ROOT", tmp_path)

    profile = "foo"
    # start with a clean directory
    assert not (tmp_path / profile / "run_state.json").exists()

    reg = RunRegistry(profile)
    # default status for every active step must be not_run
    for step in run_registry.PIPELINE_STEPS:
        assert reg.get_status(step.id) == "not_run"
    # sanity: ensure core track steps are present
    assert "extract_octree" in run_registry.STEP_BY_ID
    assert "build_pairs_sparse_octree" in run_registry.STEP_BY_ID
    assert "train_sparse_octree" in run_registry.STEP_BY_ID
    assert "export_sparse_octree" in run_registry.STEP_BY_ID
    assert "deploy_sparse_octree" in run_registry.STEP_BY_ID

    # mark a couple of steps and check persistence
    reg.mark_success("pregen")
    reg.mark_failed("dumpnoise", exit_code=42)
    # the file should now exist and contain our updates
    path = tmp_path / profile / "run_state.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["pregen"]["status"] == "success"
    assert data["dumpnoise"]["status"] == "failed"
    assert data["dumpnoise"]["exit_code"] == 42

    # a new registry instance should read the same state
    reg2 = RunRegistry(profile)
    assert reg2.get_status("pregen") == "success"
    assert reg2.get_status("dumpnoise") == "failed"


def test_phase_export_and_deploy_args(monkeypatch, tmp_path):
    """phase3_export/phase4_deploy should forward the models argument."""
    called = {}

    def fake_export_main(arglist):
        called["export"] = arglist

    def fake_deploy_main(arglist):
        called["deploy"] = arglist

    # patch the underlying script entrypoints that phase3_export/phase4_deploy
    # import dynamically
    monkeypatch.setattr("voxel_tree.tasks.octree.export.main", fake_export_main)
    monkeypatch.setattr("voxel_tree.tasks.octree.deploy_refine.main", fake_deploy_main)

    phase3 = __import__(
        "voxel_tree.preprocessing.pipeline", fromlist=["phase3_export"]
    ).phase3_export
    phase4 = __import__(
        "voxel_tree.preprocessing.pipeline", fromlist=["phase4_deploy"]
    ).phase4_deploy

    # call export with filtering
    chk = tmp_path / "best.pt"
    chk.write_text("x")
    phase3(chk, tmp_path, models=["init", "leaf"])
    assert "--models" in called["export"]
    assert "init" in called["export"] and "leaf" in called["export"]

    # ensure checkpoint-dir is passed when provided
    called.clear()
    dirpath = tmp_path / "ckdir"
    dirpath.mkdir()
    phase3(None, tmp_path, checkpoint_dir=dirpath, models=["refine"])
    assert "--checkpoint-dir" in called["export"]
    assert str(dirpath) in called["export"]
    assert "refine" in called["export"]

    # call deploy with filtering
    phase4(tmp_path, tmp_path / "dest", models=["refine"])
    assert called["deploy"][0] == str(tmp_path)
    assert "--dest" in called["deploy"]


def test_reconcile_marks_early_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If output data exists, the registry should mark the corresponding steps
    successful even if the JSON file previously said otherwise.
    """
    # create fake structure with one NPZ containing heightmap32
    data_dir = tmp_path / "data" / "voxy_octree" / "level_0"
    data_dir.mkdir(parents=True)
    npz_path = data_dir / "voxy_L0_x0_y0_z0.npz"
    import numpy as np

    np.savez_compressed(
        npz_path,
        labels32=np.zeros((32, 32, 32), dtype=np.int32),
        heightmap32=np.zeros((5, 32, 32), dtype=np.float32),
        biome32=np.zeros((32, 32), dtype=np.int32),
        section_y=np.int64(0),
        non_empty_children=np.uint8(0),
    )

    dump_dir = tmp_path / "noise_dumps"
    dump_dir.mkdir()
    (dump_dir / "dummy.txt").write_text("x")

    profile = {
        "data": {
            "data_dir": str(tmp_path / "data" / "voxy_octree"),
            "noise_dump_dir": str(dump_dir),
        }
    }
    # make sure the registry writes into our temp tree
    monkeypatch.setattr(run_registry, "_RUNS_ROOT", tmp_path)
    reg = RunRegistry("foo")
    # simulate previous failures
    reg.mark_failed("pregen")
    reg.mark_failed("dumpnoise")
    reg.mark_failed("column_heights")

    # If a step is currently running, we should not overwrite it even if the
    # expected output files exist.
    reg.mark_started("extract_octree")
    assert reg.get_status("extract_octree") == "running"

    reg.reconcile_with_profile(profile)
    assert reg.get_status("pregen") == "success"
    assert reg.get_status("dumpnoise") == "success"
    # heightmap present should also mark column_heights
    assert reg.get_status("column_heights") == "success"
    # running state should not be overwritten by artifact presence
    assert reg.get_status("extract_octree") == "running"

    # metadata setter should work and persist
    reg.set_metadata("pregen", "foo", 123)
    assert reg.get_metadata("pregen", "foo") == 123
    # reload the same profile by name
    reg2 = RunRegistry("foo")
    assert reg2.get_metadata("pregen", "foo") == 123
    # clearing the key should remove it from persisted file
    reg.set_metadata("pregen", "foo", None)
    assert reg.get_metadata("pregen", "foo") is None
    data = json.loads((tmp_path / "foo" / "run_state.json").read_text())
    assert "foo" not in data["pregen"].get("metadata", {})

    # convenience set_progress should merely set a metadata entry
    reg.set_progress("dumpnoise", 0.5)
    assert reg.get_metadata("dumpnoise", "progress") == 0.5
    reg.set_progress("dumpnoise", None)
    assert reg.get_metadata("dumpnoise", "progress") is None


def test_runs_root_defaults_to_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level ``_RUNS_ROOT`` should point at the project root\runs.

    This guard ensures we don't accidentally regress back to the package
    subdirectory when refactoring later.
    """
    # Temporarily compute what the project root would be in a normal workspace
    # Derive expectation using the same logic the module uses; this
    # avoids assumptions about where the tests are placed relative to the
    # project root.
    expected = Path(run_registry.__file__).resolve().parents[2] / "runs"
    assert run_registry._RUNS_ROOT == expected


def test_stale_detection_and_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A step is reported stale when a prerequisite is no longer successful.

    Stale steps used to appear in ``get_runnable_steps`` so the user could
    re-run them; after the refactor they are no longer returned by that
    helper and must be treated as pending by the caller.
    """
    monkeypatch.setattr(run_registry, "_RUNS_ROOT", tmp_path)
    reg = RunRegistry("p1")

    # Pick a step that has at least one prerequisite
    step_with_prereq = None
    for step in run_registry.PIPELINE_STEPS:
        if step.prereqs:
            step_with_prereq = step.id
            break
    assert step_with_prereq is not None, "pipeline must have at least one dependent step"

    # create the artificial state: prereq not success but step marked success
    prereq = run_registry.STEP_BY_ID[step_with_prereq].prereqs[0]
    # direct state manipulation to avoid timestamp noise
    reg._state[prereq]["status"] = "not_run"
    reg._state[step_with_prereq]["status"] = "success"
    # stale detection
    assert reg.is_stale(step_with_prereq)
    # the step is *not* considered runnable any more; callers need to ask
    # ``is_stale`` themselves if they want to requeue it.
    r = reg.get_runnable_steps()
    assert step_with_prereq not in r

    # also check propagation: mark a child-of-child successful and then make
    # the original prereq stale – the downstream descendant should also stale.
    # this is easiest if we can find a two-step chain.
    # find any step that depends on step_with_prereq
    desc = None
    for step in run_registry.PIPELINE_STEPS:
        if step_with_prereq in step.prereqs:
            desc = step.id
            break
    if desc:
        # mark descendant success, then revert prereq to not_run again
        reg._state[desc]["status"] = "success"
        assert reg.is_stale(desc)

    # if prereq becomes success, stale should clear
    reg._state[prereq]["status"] = "success"
    assert not reg.is_stale(step_with_prereq)
    if desc:
        assert not reg.is_stale(desc)


def test_server_session_queues_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Server sessions must treat stale steps as pending.

    The GUI looks at ``_SERVER_SESSION_STEPS`` and queues anything that is not
    currently in the ``success`` state **or** is considered stale.  A simple
    helper check reproduces that logic here to guard against regressions.
    """
    monkeypatch.setattr(run_registry, "_RUNS_ROOT", tmp_path)
    reg = RunRegistry("p1")

    # pick a server step that actually has prerequisites; otherwise it can
    # never become stale.  ``pregen`` is listed first but has no prereqs, so
    # we skip it if necessary.
    server_step = None
    for s in _SERVER_SESSION_STEPS:
        if run_registry.STEP_BY_ID[s].prereqs:
            server_step = s
            break
    assert server_step is not None, "no server step with prerequisites?"

    # ensure prereqs are marked failed so the step becomes stale
    for prereq in run_registry.STEP_BY_ID[server_step].prereqs:
        reg._state[prereq]["status"] = "not_run"
    reg._state[server_step]["status"] = "success"

    assert reg.is_stale(server_step)
    pending = [
        s for s in _SERVER_SESSION_STEPS if reg.get_status(s) != "success" or reg.is_stale(s)
    ]
    assert server_step in pending
    assert server_step in pending
    assert server_step in pending
