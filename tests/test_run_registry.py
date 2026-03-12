"""Tests for the GUI run registry persistence and loading.

These were added after a refactor busted the hardcoded runs root path.  The
registry must read and write JSON state files under the project root
``runs/`` directory and not the package subdirectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from VoxelTree.gui import run_registry
from VoxelTree.gui.run_registry import RunRegistry


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
    # default status for every step must be not_run
    assert reg.get_status("pregen") == "not_run"
    assert reg.get_status("export") == "not_run"

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
