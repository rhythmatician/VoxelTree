"""Tests to ensure GUI subprocesses run from the correct working directory.

The root cause of the current issue was that ``RunWorker._VT_ROOT`` pointed to
``VoxelTree/VoxelTree`` (package root) instead of the workspace root.  As a
result commands that took relative ``--data-dir`` paths ended up looking in
the wrong location.  This test guards against regressions.
"""

from __future__ import annotations

from pathlib import Path

from VoxelTree.gui import run_worker


def test_vt_root_points_to_repo_root() -> None:
    """Ensure the VT_ROOT is two levels above this module, not one.

    This mirrors the logic used in other GUI helpers (profile_editor, run_registry).
    """
    expected = Path(run_worker.__file__).resolve().parents[2]
    assert run_worker.RunWorker._VT_ROOT == expected
    # also verify that the repo root contains the expected top-level markers
    assert (expected / "pyproject.toml").exists(), "repo root missing pyproject.toml"
    assert (expected / "VoxelTree").is_dir(), "repo root missing VoxelTree package directory"


def test_progress_signal_parsed(monkeypatch):
    """RunWorker should emit a ``progress`` signal when stdout lines start with a percent.

    We replace ``subprocess.Popen`` with a fake object that yields a few lines
    containing progress markers and verify that the signal is fired with the
    correct fractions.
    """
    lines = [
        "10% doing stuff\n",
        "irrelevant line\n",
        "Train:  50%|#####\r\n",  # prefix + carriage return typical of tqdm
        "100% complete\n",
    ]

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(lines)
            self.returncode = 0

        def wait(self):
            return 0

        def poll(self):
            return None

        def kill(self):
            pass

    monkeypatch.setattr(run_worker.subprocess, "Popen", FakePopen)
    worker = run_worker.RunWorker("step1", ["dummy"])
    received: list[tuple[str, float]] = []
    worker.progress.connect(lambda sid, frac: received.append((sid, frac)))
    worker.log_line.connect(lambda _l: None)
    worker.step_finished.connect(lambda _sid, _code: None)

    worker.run()  # call directly instead of start() to stay in same thread

    assert received == [("step1", 0.1), ("step1", 0.5), ("step1", 1.0)]
