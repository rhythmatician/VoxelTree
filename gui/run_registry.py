"""run_registry.py — Per-profile run state persistence.

Each profile gets a JSON file at:
    runs/<profile_name>/run_state.json

Schema:
    {
      "step_id": {
        "status": "not_run" | "running" | "success" | "failed",
        "started_at": "ISO8601 or null",
        "completed_at": "ISO8601 or null",
        "exit_code": int | null,
        "metadata": {optional dict of extra data, e.g. {"epochs_completed": 5}}
      },
      ...
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gui.step_definitions import ACTIVE_STEPS, PIPELINE_STEPS, STEP_BY_ID

StepStatus = Literal["not_run", "running", "success", "failed"]

_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRegistry:
    """Manages the run_state.json for a single named profile."""

    def __init__(self, profile_name: str) -> None:
        self.profile_name = profile_name
        self._path = _RUNS_ROOT / profile_name / "run_state.json"
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._state = {}
        # Ensure every step has an entry
        for step in PIPELINE_STEPS:
            if step.id not in self._state:
                self._state[step.id] = {
                    "status": "not_run",
                    "started_at": None,
                    "completed_at": None,
                    "exit_code": None,
                }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def reload(self) -> None:
        """Re-read from disk (useful when switching profiles)."""
        self._load()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_status(self, step_id: str) -> StepStatus:
        entry = self._state.get(step_id, {})
        return entry.get("status", "not_run")  # type: ignore[return-value]

    def get_entry(self, step_id: str) -> dict[str, Any]:
        return dict(self._state.get(step_id, {}))

    def get_metadata(self, step_id: str, key: str) -> Any:
        """Get a metadata value for a step (returns None if not present)."""
        entry = self._state.get(step_id, {})
        metadata = entry.get("metadata", {})
        return metadata.get(key)

    def can_run(self, step_id: str) -> bool:
        """Return True if all prereqs for step_id are 'success'."""
        step = STEP_BY_ID.get(step_id)
        if step is None or not step.enabled:
            return False
        for prereq in step.prereqs:
            if self.get_status(prereq) != "success":
                return False
        return True

    def any_running(self) -> bool:
        return any(v.get("status") == "running" for v in self._state.values())

    def get_runnable_steps(self) -> list[str]:
        """Return all step_ids that are eligible to start right now.

        A step is eligible when:
          - all its prereqs have status 'success'
          - its own status is not 'running' or 'success'
          - it is enabled

        Unlike the old linear ``get_next_runnable_step``, this is DAG-aware
        and can return multiple steps simultaneously (e.g. the three parallel
        training steps once their respective pair caches are done).
        """
        runnable: list[str] = []
        for step in ACTIVE_STEPS:
            status = self.get_status(step.id)
            if status in ("running", "success"):
                continue
            if self.can_run(step.id):
                runnable.append(step.id)
        return runnable

    def get_next_runnable_step(self) -> str | None:
        """Return the highest-priority single runnable step, or None.

        Returns the first step from ``get_runnable_steps()`` that is not
        currently running.  This is kept for components that only show one
        highlighted node at a time (e.g. the compact dashboard row).
        Returns None if any step is currently running (prefer
        ``get_runnable_steps()`` + separate running check for parallel UI).
        """
        if self.any_running():
            return None
        runnable = self.get_runnable_steps()
        return runnable[0] if runnable else None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def mark_started(self, step_id: str) -> None:
        self._state[step_id] = {
            "status": "running",
            "started_at": _now_iso(),
            "completed_at": None,
            "exit_code": None,
        }
        self.save()

    def mark_success(self, step_id: str) -> None:
        entry = self._state.get(step_id, {})
        entry["status"] = "success"
        entry["completed_at"] = _now_iso()
        entry["exit_code"] = 0
        self._state[step_id] = entry
        self.save()

    def mark_failed(self, step_id: str, exit_code: int = -1) -> None:
        entry = self._state.get(step_id, {})
        entry["status"] = "failed"
        entry["completed_at"] = _now_iso()
        entry["exit_code"] = exit_code
        self._state[step_id] = entry
        self.save()

    def reset_step(self, step_id: str) -> None:
        self._state[step_id] = {
            "status": "not_run",
            "started_at": None,
            "completed_at": None,
            "exit_code": None,
        }
        self.save()

    def reset_from(self, step_id: str) -> None:
        """Reset step_id and all steps that depend on it (downstream)."""
        # Build downstream set via topological scan
        to_reset = {step_id}
        changed = True
        while changed:
            changed = False
            for step in PIPELINE_STEPS:
                if step.id not in to_reset:
                    if any(p in to_reset for p in step.prereqs):
                        to_reset.add(step.id)
                        changed = True
        for sid in to_reset:
            self.reset_step(sid)
