"""Validates the shared router_field_contract.yaml at training init time.

This module enforces the normalization and ordering contract between
VoxelTree (training) and LODiffusion (runtime).  It MUST be called
before any training run that produces models consumed by LODiffusion.

Failure is hard — a mismatch raises ``ContractViolation``, not a warning.

Usage
-----
>>> from voxel_tree.utils.contract_validator import validate_router_contract
>>> validate_router_contract()  # raises on mismatch
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import yaml

from voxel_tree.utils.router_field import COUNT, ROUTER_ACCESSOR_NAMES, RouterField


class ContractViolation(Exception):
    """Raised when the shared contract does not match the Python codebase."""


# Default contract location: repo root (shared between LODiffusion & VoxelTree)
_DEFAULT_CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[3] / "router_field_contract.yaml"


def _load_contract(path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load and parse the YAML contract file."""
    p = path or _DEFAULT_CONTRACT_PATH
    if not p.exists():
        # Also try the MC workspace root
        alt = pathlib.Path(__file__).resolve().parents[4] / "router_field_contract.yaml"
        if alt.exists():
            p = alt
        else:
            raise ContractViolation(
                f"Contract file not found at {p} or {alt}. "
                "This file is required for training/runtime alignment."
            )
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_router_contract(contract_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Validate the contract against the Python RouterField enum.

    Checks:
    1. Field count matches (must be 15)
    2. Field ordering matches (index, name, accessor)
    3. Contract version is present

    Returns the parsed contract dict on success.

    Raises
    ------
    ContractViolation
        If any check fails.
    """
    contract = _load_contract(contract_path)

    # Version check
    version = contract.get("version")
    if not version:
        raise ContractViolation("Contract missing 'version' field")

    fields: List[Dict[str, Any]] = contract.get("fields", [])
    if len(fields) != COUNT:
        raise ContractViolation(
            f"Contract declares {len(fields)} fields, Python has {COUNT}"
        )

    for field_def in fields:
        idx = field_def["index"]
        name = field_def["name"]
        accessor = field_def["accessor"]

        # Check index range
        if idx < 0 or idx >= COUNT:
            raise ContractViolation(f"Field index {idx} out of range [0, {COUNT})")

        # Check name matches Python RouterField
        py_field = RouterField.by_index(idx)
        if py_field.lower_name != name:
            raise ContractViolation(
                f"Field {idx}: contract name '{name}' != Python name '{py_field.lower_name}'"
            )

        # Check accessor matches ROUTER_ACCESSOR_NAMES
        expected_accessor = ROUTER_ACCESSOR_NAMES[idx]
        if accessor != expected_accessor:
            raise ContractViolation(
                f"Field {idx} ({name}): contract accessor '{accessor}' "
                f"!= Python accessor '{expected_accessor}'"
            )

    # Spatial layout check
    spatial = contract.get("spatial", {})
    expected_quarts = [4, 2, 4]
    if spatial.get("quarts_per_section") != expected_quarts:
        raise ContractViolation(
            f"Spatial quarts_per_section {spatial.get('quarts_per_section')} "
            f"!= expected {expected_quarts}"
        )

    return contract


def validate_normalization_method(contract: Dict[str, Any]) -> None:
    """Verify all fields use 'none' normalization (raw pass-through).

    This is the current contract: router fields are passed through raw
    from the Java data harvester to the training pipeline with no
    normalization applied.  If we ever add normalization, both sides
    must agree on the method.
    """
    for field_def in contract.get("fields", []):
        norm = field_def.get("normalization", {})
        method = norm.get("method", "none")
        if method != "none":
            raise ContractViolation(
                f"Field {field_def['index']} ({field_def['name']}): "
                f"normalization method '{method}' is not 'none'. "
                "If normalization is added, both training and runtime must agree."
            )


if __name__ == "__main__":
    print(f"Validating contract at: {_DEFAULT_CONTRACT_PATH}")
    c = validate_router_contract()
    validate_normalization_method(c)
    print(f"✓ Contract v{c['version']} validated — {len(c['fields'])} fields, all checks passed.")
