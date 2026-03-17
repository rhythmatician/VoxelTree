"""dag_definition.py — Per-profile pipeline DAG configuration.

A ProfileDag records which steps are active for a given profile and allows
optional per-step prereq overrides.  It is serialized into a ``dag:`` section
in the profile YAML so that the topology is preserved alongside other settings.

Format in profile YAML
----------------------
::

    dag:
      steps:                       # ordered list; omit = use full PIPELINE_STEPS
        - id: pregen
        - id: dumpnoise
        - id: train_stage1_density
          prereqs: [dumpnoise]     # optional override; absent → registry default

Persistence note
----------------
``ProfileDag.from_profile_dict`` returns *None* when the profile has no
``dag:`` key so that callers can distinguish "no override" from "empty DAG".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DagStepEntry:
    """One step entry in a ProfileDag.

    Parameters
    ----------
    id:
        Must match a ``StepDef.id`` in the step registry.
    prereqs:
        When *None* the registry default prereqs are used (with any entries
        not in this DAG's active set automatically stripped).  Set to an
        explicit list to hard-wire specific connections.
    """

    id: str
    prereqs: list[str] | None = None

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {"id": self.id}
        if self.prereqs is not None:
            d["prereqs"] = list(self.prereqs)
        return d

    @classmethod
    def from_dict(cls, d) -> DagStepEntry:
        if isinstance(d, str):
            # shorthand: just a bare step id
            return cls(id=d)
        return cls(
            id=d["id"],
            prereqs=list(d["prereqs"]) if "prereqs" in d else None,
        )


@dataclass
class ProfileDag:
    """Per-profile pipeline DAG.

    Parameters
    ----------
    entries:
        Ordered list of active step entries.  The order follows the logical
        pipeline order (pregen first, deploy last) and is used to decide row
        assignment within a topological column when there are ties.
    """

    entries: list[DagStepEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def active_ids(self) -> set[str]:
        return {e.id for e in self.entries}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dag_dict(self) -> dict:
        """Return a dict suitable for embedding as the ``dag:`` value in a profile YAML."""
        return {"steps": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dag_dict(cls, d: dict) -> ProfileDag:
        entries = [DagStepEntry.from_dict(e) for e in d.get("steps", [])]
        return cls(entries=entries)

    @classmethod
    def from_profile_dict(cls, profile: dict) -> ProfileDag | None:
        """Return a ProfileDag if the profile YAML contains a ``dag:`` section.

        Returns *None* when the key is absent so the caller can fall back to
        the global default.
        """
        dag_section = profile.get("dag")
        if not dag_section:
            return None
        return cls.from_dag_dict(dag_section)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> ProfileDag:
        """Return a ProfileDag containing only default-active steps.

        Steps belonging to model tracks where ``in_default_dag=False`` are
        excluded — they remain in the full step registry so advanced profiles
        can include them explicitly, but they do not appear in freshly-created
        or reset profiles.
        """
        from VoxelTree.gui.step_definitions import PIPELINE_STEPS, TRACK_BY_ID  # late import

        excluded_tracks: set[str] = {
            tid for tid, track in TRACK_BY_ID.items() if not track.in_default_dag
        }
        entries = [
            DagStepEntry(id=s.id)
            for s in PIPELINE_STEPS
            if s.enabled and (s.track is None or s.track not in excluded_tracks)
        ]
        return cls(entries=entries)

    # ------------------------------------------------------------------
    # Resolve to StepDef instances
    # ------------------------------------------------------------------

    def resolve_steps(self) -> list:
        """Return a ``list[StepDef]`` for the active entries.

        Steps not present in the registry are silently skipped (forward-compat
        with older profiles that reference step ids not yet implemented).
        Prereqs that reference a step not in *this* DAG's active set are also
        stripped so the rendered graph is always self-consistent.
        """
        from dataclasses import replace

        from VoxelTree.gui.step_definitions import STEP_BY_ID  # late import

        ids = self.active_ids
        result = []
        for entry in self.entries:
            template = STEP_BY_ID.get(entry.id)
            if template is None:
                # Unknown step — could be a future step type added in code but
                # not yet in the registry.  Skip silently.
                continue

            # Determine effective prereqs
            if entry.prereqs is not None:
                prereqs = [p for p in entry.prereqs if p in ids]
            else:
                prereqs = [p for p in template.prereqs if p in ids]

            step = replace(template, prereqs=prereqs, enabled=True)
            result.append(step)
        return result
