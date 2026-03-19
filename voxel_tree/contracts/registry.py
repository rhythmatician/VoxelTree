"""Contract registry — global dict of all model contracts, keyed by (name, revision).

Registration happens at import time when ``voxel_tree.contracts.catalog`` is
loaded.  The registry is the single source of truth for "what does model X
revision Y expect as inputs/outputs?"

Usage
-----
>>> from voxel_tree.contracts import get_contract, latest_revision, list_models
>>> get_contract("density", revision=2)
ModelContract(model_name='density', revision=2, ...)
>>> latest_revision("density")
2
>>> list_models()
['biome', 'density', 'heightmap', 'sparse_octree']
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from voxel_tree.contracts.spec import ContractViolation, ModelContract, TensorSpec, compare_specs


# ---------------------------------------------------------------------------
# Protocol for anything that carries contract binding info (e.g. ModelTrack)
# ---------------------------------------------------------------------------


@runtime_checkable
class HasContractBinding(Protocol):
    """Any object with optional contract_name / contract_revision attrs."""

    track_id: str
    contract_name: str | None
    contract_revision: int | None


# ---------------------------------------------------------------------------
# AlignmentIssue — one detected problem between a track and its contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentIssue:
    """One problem detected by ``check_track_alignment()``.

    Attributes
    ----------
    track_id : The pipeline track that has the problem.
    severity : ``"error"`` (contract missing), ``"stale"`` (newer revision
               exists), or ``"incompatible"`` (build_pairs OUTPUT_SPEC
               doesn't match the contract inputs).
    message  : Human-readable description.
    current_revision  : What the track is pinned to (may be None).
    latest_revision_  : The newest revision available in the catalog (may be None).
    io_mismatches     : Specific I/O mismatches from ``compare_specs()``.
    """

    track_id: str
    severity: str  # "error" | "stale" | "incompatible"
    message: str
    current_revision: int | None = None
    latest_revision_: int | None = None
    io_mismatches: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

# (model_name, revision) → ModelContract
CONTRACTS: dict[tuple[str, int], ModelContract] = {}

# model_name → highest registered revision
_LATEST: dict[str, int] = {}


def register(contract: ModelContract) -> ModelContract:
    """Add a contract to the global registry.  Returns the contract for chaining."""
    key = (contract.model_name, contract.revision)
    if key in CONTRACTS:
        existing = CONTRACTS[key]
        if existing.fingerprint != contract.fingerprint:
            raise ValueError(
                f"Conflicting registration for {key}: "
                f"fingerprints {existing.fingerprint} vs {contract.fingerprint}"
            )
        return contract  # idempotent re-registration is fine
    CONTRACTS[key] = contract
    if contract.revision > _LATEST.get(contract.model_name, -1):
        _LATEST[contract.model_name] = contract.revision
    return contract


def get_contract(model_name: str, *, revision: int | None = None) -> ModelContract:
    """Look up a contract.  ``revision=None`` → latest.

    Raises ``KeyError`` if not found.
    """
    if revision is None:
        revision = latest_revision(model_name)
    key = (model_name, revision)
    if key not in CONTRACTS:
        available = [r for (n, r) in CONTRACTS if n == model_name]
        raise KeyError(
            f"No contract for model={model_name!r} revision={revision}. "
            f"Available revisions: {sorted(available) or 'none'}"
        )
    return CONTRACTS[key]


def latest_revision(model_name: str) -> int:
    """Return the highest registered revision for *model_name*.

    Raises ``KeyError`` if the model has never been registered.
    """
    if model_name not in _LATEST:
        raise KeyError(f"No contracts registered for model={model_name!r}")
    return _LATEST[model_name]


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_LATEST.keys())


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    expected: ModelContract,
    *,
    strict: bool = False,
) -> None:
    """Verify a loaded checkpoint was trained under *expected* (or a compatible) contract.

    Checks ``checkpoint["contract_meta"]`` against *expected*.  In non-strict
    mode, only the model name must match and the revision must be ≤ expected.
    In strict mode, the fingerprint must match exactly.
    """
    meta = checkpoint.get("contract_meta")
    if meta is None:
        if strict:
            raise ContractViolation(
                f"Checkpoint has no contract_meta — cannot verify against "
                f"{expected.contract_id}"
            )
        warnings.warn(
            f"Checkpoint has no contract_meta — skipping contract validation "
            f"(expected {expected.contract_id})",
            stacklevel=2,
        )
        return

    ckpt_name = meta.get("model_name", "")
    ckpt_rev = meta.get("revision", -1)
    ckpt_fp = meta.get("fingerprint", "")

    if ckpt_name != expected.model_name:
        raise ContractViolation(
            f"Checkpoint model '{ckpt_name}' != expected '{expected.model_name}'"
        )

    if strict:
        if ckpt_fp != expected.fingerprint:
            raise ContractViolation(
                f"Checkpoint fingerprint {ckpt_fp} != expected {expected.fingerprint} "
                f"(revision {ckpt_rev} vs {expected.revision})"
            )
    else:
        if ckpt_rev > expected.revision:
            raise ContractViolation(
                f"Checkpoint was trained under revision {ckpt_rev}, but "
                f"expected ≤ {expected.revision}. Re-export with the matching "
                f"contract revision."
            )


# ---------------------------------------------------------------------------
# Build-pairs OUTPUT_SPEC loader
# ---------------------------------------------------------------------------


def _load_output_spec(dotted_fn: str) -> tuple[TensorSpec, ...] | None:
    """Import a build_pairs module and return its OUTPUT_SPEC, or None.

    Parameters
    ----------
    dotted_fn : Module path in ``"package.module:callable"`` form.
                Only the module part (before ``:``) is imported.

    Returns
    -------
    The module's ``OUTPUT_SPEC`` attribute, or ``None`` if the module
    doesn't define one or can't be imported.
    """
    if not dotted_fn:
        return None
    module_path = dotted_fn.split(":")[0]
    try:
        import importlib

        mod = importlib.import_module(module_path)
    except Exception:
        return None
    spec = getattr(mod, "OUTPUT_SPEC", None)
    if spec is not None and isinstance(spec, tuple):
        return spec
    return None


# ---------------------------------------------------------------------------
# Track↔Contract alignment check
# ---------------------------------------------------------------------------


def check_track_alignment(
    tracks: list[HasContractBinding] | None = None,
) -> list[AlignmentIssue]:
    """Compare pipeline tracks against the contract catalog and report issues.

    For each track that carries a ``contract_name``:
      1. Verify the contract actually exists in the catalog.
      2. Verify the track's ``contract_revision`` matches the latest catalog
         revision — if not, the track is **stale** (the contract was bumped
         but the pipeline wasn't updated to match).
      3. Load the build_pairs module's ``OUTPUT_SPEC`` and compare it against
         the contract's ``inputs``.  If the specs don't match, the track is
         **incompatible** — the script hasn't been updated to produce the
         data the model now expects.

    Parameters
    ----------
    tracks : Iterable of objects with ``track_id``, ``contract_name``, and
             ``contract_revision`` attributes.  If *None*, imports
             ``MODEL_TRACKS`` from ``voxel_tree.gui.step_definitions``.

    Returns
    -------
    List of ``AlignmentIssue`` objects (empty ⇒ everything aligned).
    """
    if tracks is None:
        from voxel_tree.gui.step_definitions import MODEL_TRACKS

        tracks = MODEL_TRACKS  # type: ignore[assignment]

    assert tracks is not None  # satisfied by the fallback above
    issues: list[AlignmentIssue] = []

    for track in tracks:
        cname = track.contract_name
        if cname is None:
            continue  # track not bound to any contract — nothing to check

        crev = track.contract_revision

        # --- Resolve None → latest ("always track latest" semantics) ---
        if crev is None:
            try:
                crev = latest_revision(cname)
            except KeyError:
                issues.append(
                    AlignmentIssue(
                        track_id=track.track_id,
                        severity="error",
                        message=(
                            f"Track '{track.track_id}' has contract_name='{cname}' "
                            f"but no revisions are registered for that model."
                        ),
                    )
                )
                continue

        # --- Does the contract even exist? ---
        key = (cname, crev)
        if key not in CONTRACTS:
            issues.append(
                AlignmentIssue(
                    track_id=track.track_id,
                    severity="error",
                    message=(
                        f"Track '{track.track_id}' references contract "
                        f"'{cname}' rev {crev}, but that revision does "
                        f"not exist in the catalog."
                    ),
                    current_revision=crev,
                )
            )
            continue

        # --- Is it up to date? ---
        try:
            latest = latest_revision(cname)
        except KeyError:
            issues.append(
                AlignmentIssue(
                    track_id=track.track_id,
                    severity="error",
                    message=(
                        f"Track '{track.track_id}' references contract "
                        f"'{cname}', but no revisions are registered."
                    ),
                    current_revision=crev,
                )
            )
            continue

        if crev < latest:
            # Show what changed so the user sees exactly what needs updating
            new_contract = CONTRACTS[(cname, latest)]

            # Preview: if upgraded, would build_pairs be incompatible?
            upgrade_mismatches: tuple[str, ...] = ()
            output_spec = _load_output_spec(new_contract.build_pairs_fn)
            if output_spec is not None:
                diffs = compare_specs(output_spec, new_contract.inputs)
                upgrade_mismatches = tuple(diffs)

            msg = (
                f"Track '{track.track_id}' is pinned to "
                f"'{cname}' rev {crev}, but rev {latest} exists. "
                f"Changelog: {new_contract.changelog or '(none)'}"
            )
            if upgrade_mismatches:
                msg += (
                    f"\n  ⚠ build_pairs would be INCOMPATIBLE with rev {latest}:"
                )
                for m in upgrade_mismatches:
                    msg += f"\n    • {m}"

            issues.append(
                AlignmentIssue(
                    track_id=track.track_id,
                    severity="stale",
                    message=msg,
                    current_revision=crev,
                    latest_revision_=latest,
                    io_mismatches=upgrade_mismatches,
                )
            )

        # --- Does the build_pairs script match the pinned contract? ---
        contract = CONTRACTS[(cname, crev)]
        output_spec = _load_output_spec(contract.build_pairs_fn)
        if output_spec is not None:
            diffs = compare_specs(output_spec, contract.inputs)
            if diffs:
                issues.append(
                    AlignmentIssue(
                        track_id=track.track_id,
                        severity="incompatible",
                        message=(
                            f"Track '{track.track_id}' build_pairs OUTPUT_SPEC "
                            f"doesn't match '{cname}' rev {crev} inputs:\n"
                            + "\n".join(f"  • {d}" for d in diffs)
                        ),
                        current_revision=crev,
                        io_mismatches=tuple(diffs),
                    )
                )

    return issues


# Auto-import catalog to populate the registry at first access
# ---------------------------------------------------------------------------


def _ensure_catalog_loaded() -> None:
    """Import the catalog module to trigger registration."""
    import voxel_tree.contracts.catalog  # noqa: F401


# Eagerly load on module import so CONTRACTS is always populated.
_ensure_catalog_loaded()
