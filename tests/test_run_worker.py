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
        "Train:  2.3%|~\r\n",  # decimal percentage
        "100% complete\n",
    ]

    class FakePopen:
        def __init__(self, *args, **kwargs):
            # simulate a file object with an iterator
            class FakeStdout:
                def __init__(self, lines):
                    self.lines = lines

                def __iter__(self):
                    return iter(self.lines)

            self.stdout = FakeStdout(lines)
            self.stdin = type(
                "FakeStdin", (), {"write": lambda self, x: None, "close": lambda self: None}
            )()
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
    worker.log_line.connect(lambda _sid, _l: None)
    worker.step_finished.connect(lambda _sid, _code: None)

    # Mock signals manually to avoid PySide6 timing/event-loop issues in tests.
    class MockSignal:
        def __init__(self):
            self.handlers = []

        def connect(self, h):
            self.handlers.append(h)

        def emit(self, *args):
            for h in self.handlers:
                h(*args)

    monkeypatch.setattr(worker, "progress", MockSignal())
    monkeypatch.setattr(worker, "log_line", MockSignal())
    monkeypatch.setattr(worker, "step_started", MockSignal())
    monkeypatch.setattr(worker, "step_finished", MockSignal())

    worker.progress.connect(lambda sid, frac: received.append((sid, frac)))

    worker.run()  # call directly instead of start() to stay in same thread

    assert received == [("step1", 0.1), ("step1", 0.023), ("step1", 1.0)]
