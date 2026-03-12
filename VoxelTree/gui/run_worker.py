"""run_worker.py — QThread-based subprocess runner for pipeline steps."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class RunWorker(QThread):
    """Runs a single pipeline step as a subprocess.

    Signals
    -------
    log_line(str)
        Emitted for every line of combined stdout/stderr output.
    step_started(str)
        Emitted once, immediately before the subprocess is launched.
        Carries the step_id.
    step_finished(str, int)
        Emitted when the subprocess exits.
        Carries (step_id, exit_code).  exit_code == -2 means cancelled.
    """

    log_line: Signal = Signal(str)
    step_started: Signal = Signal(str)
    step_finished: Signal = Signal(str, int)

    #: Directory where all pipeline scripts live (VoxelTree repo root).
    #
    # After the refactor the package tree gained an extra nesting level
    # (``VoxelTree/VoxelTree``).  ``parent.parent`` would therefore still point
    # inside the installed package, causing subprocesses to run with cwd
    # ``.../VoxelTree/VoxelTree``.  Relative paths such as ``data/voxy_octree``
    # ended up being resolved below the package instead of the workspace root,
    # so steps like build-pairs saw empty directories and reported "No sections
    # found".  Compute the repo root the same way `profile_editor` and
    # `run_registry` do: two parents up from this file.
    _VT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, step_id: str, cmd: list[str]) -> None:
        super().__init__()
        self.step_id = step_id
        self.cmd = cmd
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    # ------------------------------------------------------------------

    def run(self) -> None:  # called by QThread.start()
        self.step_started.emit(self.step_id)
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self._VT_ROOT),
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.log_line.emit(line.rstrip("\n"))
            self._proc.wait()
            exit_code = -2 if self._cancelled else self._proc.returncode
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(f"[RunWorker error] {exc}")
            exit_code = -1

        self.step_finished.emit(self.step_id, exit_code)

    def cancel(self) -> None:
        """Request cancellation — kills the subprocess if running."""
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except OSError:
                pass
