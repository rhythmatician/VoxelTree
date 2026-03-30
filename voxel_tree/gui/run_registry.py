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
import sys
import types
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from voxel_tree.gui.step_definitions import ACTIVE_STEPS, PIPELINE_STEPS, STEP_BY_ID

StepStatus = Literal["not_run", "running", "success", "failed"]

# When running from the repo tree the code lives in ``VoxelTree/VoxelTree``.
# When installed, the package could be in a different location.  We therefore
# search upward from the current module until we find a repo marker (pyproject
# or .git) and treat that directory as the project root.
#
# This keeps the GUI in sync with the CLI and other tools which all expect
# run state to live under ``<repo_root>/runs``.


def _find_project_root(start: Path) -> Path:
    for ancestor in [start] + list(start.parents):
        if (ancestor / "pyproject.toml").exists() or (ancestor / ".git").exists():
            return ancestor
    return start


_RUNS_ROOT = _find_project_root(Path(__file__).resolve().parent) / "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_checkpoint_epoch(checkpoint_path: Path) -> int | None:
    """Best-effort checkpoint epoch extraction.

    Returns None if torch is unavailable, the file is unreadable, or the
    checkpoint does not contain an integer-like ``epoch`` field.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None

    # Backward-compat: older checkpoints were pickled with
    # voxel_tree.tasks.sparse_octree.* module paths.
    _install_legacy_sparse_octree_aliases()

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch")
        try:
            return int(epoch) if epoch is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _install_legacy_sparse_octree_aliases() -> None:
    """Install sys.modules aliases for pre-rename sparse_octree paths."""
    pkg_name = "voxel_tree.tasks.sparse_octree"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # mark as package-like for import machinery
        sys.modules[pkg_name] = pkg

    alias_targets = {
        f"{pkg_name}.voxy_models": "voxel_tree.tasks.voxy.voxy_models",
        f"{pkg_name}.voxy_train": "voxel_tree.tasks.voxy.voxy_train",
        f"{pkg_name}.voxy_dataset": "voxel_tree.tasks.voxy.voxy_dataset",
        f"{pkg_name}.voxy_targets": "voxel_tree.tasks.voxy.voxy_targets",
    }
    for old_name, new_name in alias_targets.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except Exception:
            # Best-effort only; unresolved aliases simply won't be used.
            continue


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
        # Ensure every step has an entry (this will only add missing steps; existing
        # statuses are preserved).  Fixing the runs root path may cause a new file to
        # be created when the old one lived elsewhere, so we populate defaults here
        # to avoid KeyError later.
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
        # Also reconcile against the profile YAML so that externally-generated
        # outputs (e.g. noise_dumps) can be detected and reflected in the UI.
        profile = self._load_profile_yaml()
        if profile:
            self.reconcile_with_profile(profile)

    def _load_profile_yaml(self) -> dict:
        """Load the profile YAML for this registry (returns empty dict if missing)."""
        try:
            import yaml
        except ImportError:
            return {}

        from pathlib import Path

        profiles_dir = Path(__file__).resolve().parents[2] / "profiles"
        path = profiles_dir / f"{self.profile_name}.yaml"
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # State reconciliation helpers
    # ------------------------------------------------------------------

    def reconcile_with_profile(self, profile: dict) -> None:
        """Attempt to repair registry state by inspecting output files.

        This is a heuristic used by the GUI to avoid misleading "failed" status
        markers when the underlying data is still present.  It only nudges the
        early pipeline steps (pregen/voxy_import/dumpnoise/extract) which are
        easy to detect, plus column_heights if ``heightmap32`` planes exist.

        For Voxy per-level training, existing checkpoint files are also treated
        as evidence that corresponding train steps completed successfully; when
        possible, ``epochs_completed`` is restored from checkpoint metadata.

        The method is idempotent and will save the state file if any changes are
        made.
        """
        changed = False

        # helper to update a step once
        def _set_success(step_id: str) -> None:
            nonlocal changed
            # Do not override an in-progress run.  Running status should be
            # authoritative even if output artifacts appear early.
            current = self.get_status(step_id)
            if current == "running":
                return
            if current != "success":
                self.mark_success(step_id)
                changed = True

        data = profile.get("data", {})
        data_dir = Path(data.get("data_dir", ""))

        # If extraction output exists, assume the early pipeline steps completed.
        any_npz = False
        if data_dir.is_dir():
            any_npz = any(data_dir.glob("level_*/*.npz"))

        if any_npz:
            _set_success("pregen")
            _set_success("voxy_import")
            _set_success("dumpnoise")
            _set_success("extract_octree")

        # Dumpnoise can be considered successful if the noise-dump directory exists
        # and contains any files.  This allows users to run dumpnoise independently
        # from pregen.
        dump_dir = Path(data.get("noise_dump_dir", "tools/fabric-server/runtime/noise_dumps"))
        if dump_dir.is_dir() and any(dump_dir.iterdir()):
            _set_success("dumpnoise")
            # If we have dumpnoise output, the server was clearly reachable.
            # In that case, we can also consider pregen/voxy_import as done.
            _set_success("pregen")
            _set_success("voxy_import")

            # column heights if any file already contains the feature
            # (avoid loading many large npz files on the UI thread)
            max_checks = 20
            checked = 0
            for npz_path in data_dir.glob("level_*/*.npz"):
                if checked >= max_checks:
                    break
                checked += 1
                try:
                    # Use mmap_mode to avoid reading large arrays into memory.
                    with np.load(npz_path, mmap_mode="r") as arr:
                        if "heightmap32" in getattr(arr, "files", []):
                            _set_success("column_heights")
                            break
                except Exception:
                    continue

        if changed:
            # ``mark_success`` already saved, but ensure overall registry persists
            self.save()

        # Voxy train checkpoint recovery: restore historical train statuses and
        # epoch counters so the dashboard reflects prior runs.
        train_cfg = profile.get("train", {})
        output_dir = Path(train_cfg.get("output_dir", "checkpoints"))
        found_voxy_checkpoint = False
        for level in range(4, -1, -1):
            step_id = f"train_voxy_l{level}"
            ckpt_path = output_dir / f"voxy_L{level}.pt"
            if not ckpt_path.exists():
                continue

            found_voxy_checkpoint = True
            _set_success(step_id)

            epoch = _read_checkpoint_epoch(ckpt_path)
            if epoch is not None and self.get_metadata(step_id, "epochs_completed") != epoch:
                self.set_metadata(step_id, "epochs_completed", epoch)

        if found_voxy_checkpoint:
            # If we can see Voxy train checkpoints, upstream data-acquisition
            # must have completed at least once. Mark these done to avoid
            # stale propagation that would otherwise hide green train nodes.
            for sid in ("pregen", "harvest", "extract_octree", "dumpnoise", "import_voxy"):
                _set_success(sid)

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
        """Return True if all prereqs for step_id are 'success'.

        This is used to decide whether a step is eligible to start.  Note that a
        step may have ``status == 'success'`` in the registry while one of its
        prerequisites has since been reset or failed; we consider such a step
        *stale* (see :meth:`is_stale`) but still mark it runnable so the user can
        re-execute it.
        """
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
          - its own status is neither 'running' nor 'success'
          - it is enabled

        Previously we treated steps that had succeeded but whose prerequisites
        later failed as "stale" and listed them here so the user could manually
        re‑execute them.  That behaviour has been removed; stale state is still
        detectable via :meth:`is_stale` but the registry no longer exposes
        stale steps in the runnable list.  The GUI is responsible for handling
        staleness in other contexts (e.g. server sessions).

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
        prev = self._state.get(step_id, {})
        entry: dict[str, Any] = {
            "status": "running",
            "started_at": _now_iso(),
            "completed_at": None,
            "exit_code": None,
        }
        metadata = prev.get("metadata")
        if isinstance(metadata, dict) and metadata:
            entry["metadata"] = dict(metadata)
        self._state[step_id] = entry
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
        prev = self._state.get(step_id, {})
        entry: dict[str, Any] = {
            "status": "not_run",
            "started_at": None,
            "completed_at": None,
            "exit_code": None,
        }
        # Keep configured epoch targets visible in the dashboard after reset.
        metadata = prev.get("metadata")
        if isinstance(metadata, dict):
            target = metadata.get("epochs_target")
            if target is not None:
                entry["metadata"] = {"epochs_target": target}
        self._state[step_id] = entry
        self.save()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_metadata(self, step_id: str, key: str, value: Any) -> None:
        """Write an arbitrary metadata key for *step_id*.

        ``value`` may be any JSON-serializable object.  Passing ``None`` clears
        the key from the entry.  A metadata dict is created on demand if one
        does not already exist.  Changes are immediately persisted.
        """
        entry = self._state.setdefault(step_id, {})
        metadata = entry.get("metadata", {})
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
        if metadata:
            entry["metadata"] = metadata
        elif "metadata" in entry:
            entry.pop("metadata")
        # ensure the entry is stored back and save
        self._state[step_id] = entry
        self.save()

    def set_progress(self, step_id: str, fraction: float | None) -> None:
        """Convenience wrapper for :meth:`set_metadata` using key ``'progress'``."""
        self.set_metadata(step_id, "progress", fraction)

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

    # ------------------------------------------------------------------
    # Staleness support
    # ------------------------------------------------------------------

    def is_stale(self, step_id: str) -> bool:
        """Return True if a step currently marked ``success`` is *stale*.

        A step becomes stale when at least one of its prerequisites is no longer
        in the ``success`` state.  We do **not** persist this state, since it is
        derivable from the existing registry entries; the GUI queries this
        helper when deciding how to colour nodes.
        """
        if self.get_status(step_id) != "success":
            return False
        step = STEP_BY_ID.get(step_id)
        if step is None:
            return False
        # A step is stale if any of its prerequisites is not currently a
        # successful run, or if any of its prerequisites themselves are
        # stale.  That ensures the “stale” flag propagates downstream.
        for prereq in step.prereqs:
            if self.get_status(prereq) != "success" or self.is_stale(prereq):
                return True
        return False
