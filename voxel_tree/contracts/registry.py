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
from typing import Any

from voxel_tree.contracts.spec import ContractViolation, ModelContract

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
# Auto-import catalog to populate the registry at first access
# ---------------------------------------------------------------------------


def _ensure_catalog_loaded() -> None:
    """Import the catalog module to trigger registration."""
    import voxel_tree.contracts.catalog  # noqa: F401


# Eagerly load on module import so CONTRACTS is always populated.
_ensure_catalog_loaded()
