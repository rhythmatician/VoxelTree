"""run_worker.py — QThread-based subprocess runner for pipeline steps.

Steps are executed via the ``voxel_tree.step_runner`` module in a child
process.  The step's ``run_fn`` is called directly — no CLI arg-building,
no argparse round-trip.  The profile dict is passed as JSON on stdin.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class RunWorker(QThread):
    """Runs a single pipeline step as a subprocess.

    The child process executes ``python -m voxel_tree.step_runner <step_id>``
    and receives the profile dict as JSON on stdin.  stdout/stderr are
    captured and forwarded as ``log_line`` signals.

    Signals
    -------
    log_line(str, str)
        Emitted for every line of combined stdout/stderr output.
    step_started(str)
        Emitted once, immediately before the subprocess is launched.
    step_finished(str, int)
        Emitted when the subprocess exits.
        exit_code == -2 means cancelled.
    progress(str, float)
        Progress fraction 0.0–1.0 parsed from child output.
    """

    log_line: Signal = Signal(str, str)
    step_started: Signal = Signal(str)
    step_finished: Signal = Signal(str, int)
    progress: Signal = Signal(str, float)

    #: VoxelTree repo root — used as cwd for child processes.
    _VT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, step_id: str, profile: dict) -> None:  # type: ignore[type-arg]
        super().__init__()
        self.step_id = step_id
        self.profile = profile
        self._proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._cancelled = False

    # ------------------------------------------------------------------

    _PROGRESS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")

    def run(self) -> None:  # called by QThread.start()
        self.step_started.emit(self.step_id)
        try:
            cmd = [sys.executable, "-m", "voxel_tree.step_runner", self.step_id]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self._VT_ROOT),
            )
            # Send profile as JSON on stdin, then close to unblock the child.
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(self.profile))
            self._proc.stdin.close()

            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                stripped = line.rstrip("\r\n")
                self.log_line.emit(self.step_id, stripped)
                m = self._PROGRESS_RE.search(stripped)
                if m:
                    try:
                        pct = float(m.group(1)) / 100.0
                        pct = max(0.0, min(1.0, pct))
                        self.progress.emit(self.step_id, pct)
                    except ValueError:
                        pass
            self._proc.wait()
            exit_code = -2 if self._cancelled else self._proc.returncode
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(self.step_id, f"[RunWorker error] {exc}")
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
