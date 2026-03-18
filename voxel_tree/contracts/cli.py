"""CLI sub-command: inspect and diff model contracts.

Usage
-----
  # List all registered models and their latest revision
  voxel-tree contracts

  # Show full detail for a specific model
  voxel-tree contracts density

  # Show a specific revision
  voxel-tree contracts density --revision 0

  # Show what changed between revisions
  voxel-tree contracts density --diff 0 1

  # Dump the sidecar JSON that export would produce
  voxel-tree contracts density --sidecar
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _fmt_shape(shape: tuple) -> str:
    return "(" + ", ".join(str(s) for s in shape) + ")"


def _print_table(models: list[dict]) -> None:
    """Pretty-print a model summary table."""
    headers = ["Model", "Rev", "Contract ID", "Inputs", "Outputs", "Fingerprint"]
    rows = []
    for m in models:
        rows.append(
            [
                m["model_name"],
                str(m["revision"]),
                m["contract_id"],
                ", ".join(f"{s['name']}{_fmt_shape(tuple(s['shape']))}" for s in m["inputs"]),
                ", ".join(f"{s['name']}{_fmt_shape(tuple(s['shape']))}" for s in m["outputs"]),
                m["fingerprint"][:8],
            ]
        )

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _print_contract_detail(contract) -> None:  # noqa: ANN001
    """Print detailed view of one contract."""
    print(f"\n{'═' * 62}")
    print(f"  {contract.model_name} revision {contract.revision}")
    print(f"  Contract ID : {contract.contract_id}")
    print(f"  Fingerprint : {contract.fingerprint}")
    print(f"  ONNX opset  : {contract.onnx_opset}")
    print(f"  Description : {contract.description}")
    if contract.changelog:
        print(f"  Changelog   : {contract.changelog}")
    print(f"{'═' * 62}")

    print("\n  INPUTS:")
    for spec in contract.inputs:
        print(f"    {spec.name}: {_fmt_shape(spec.shape)}  dtype={spec.dtype}")
        if spec.channels:
            print(f"      channels: {list(spec.channels)}")
        if spec.channel_indices:
            print(f"      indices : {list(spec.channel_indices)}")
        if spec.description:
            print(f"      note    : {spec.description}")

    print("\n  OUTPUTS:")
    for spec in contract.outputs:
        print(f"    {spec.name}: {_fmt_shape(spec.shape)}  dtype={spec.dtype}")
        if spec.channels:
            print(f"      channels: {list(spec.channels)}")
        if spec.description:
            print(f"      note    : {spec.description}")

    if contract.build_pairs_fn or contract.train_fn or contract.export_fn:
        print("\n  PIPELINE ENTRY POINTS:")
        if contract.build_pairs_fn:
            print(f"    build_pairs : {contract.build_pairs_fn}")
        if contract.train_fn:
            print(f"    train       : {contract.train_fn}")
        if contract.export_fn:
            print(f"    export      : {contract.export_fn}")

    if contract.extra:
        print(f"\n  EXTRA: {json.dumps(contract.extra, indent=4)}")
    print()


def _diff_revisions(contract_a, contract_b) -> None:  # noqa: ANN001
    """Print a human-readable diff between two contract revisions."""
    print(f"\n{'═' * 62}")
    print(f"  DIFF: {contract_a.model_name} rev {contract_a.revision} → rev {contract_b.revision}")
    print(f"{'═' * 62}")

    if contract_b.changelog:
        print(f"\n  Changelog: {contract_b.changelog}")

    # Compare inputs
    a_inputs = {s.name: s for s in contract_a.inputs}
    b_inputs = {s.name: s for s in contract_b.inputs}

    removed_inputs = set(a_inputs) - set(b_inputs)
    added_inputs = set(b_inputs) - set(a_inputs)
    common_inputs = set(a_inputs) & set(b_inputs)

    if removed_inputs or added_inputs or any(a_inputs[n] != b_inputs[n] for n in common_inputs):
        print("\n  INPUT CHANGES:")
        for name in sorted(removed_inputs):
            print(f"    - REMOVED: {name} {_fmt_shape(a_inputs[name].shape)}")
        for name in sorted(added_inputs):
            print(f"    + ADDED:   {name} {_fmt_shape(b_inputs[name].shape)}")
        for name in sorted(common_inputs):
            if a_inputs[name] != b_inputs[name]:
                print(f"    ~ CHANGED: {name}")
                if a_inputs[name].shape != b_inputs[name].shape:
                    print(
                        f"        shape: {_fmt_shape(a_inputs[name].shape)} → {_fmt_shape(b_inputs[name].shape)}"
                    )
                if a_inputs[name].channels != b_inputs[name].channels:
                    print(
                        f"        channels: {list(a_inputs[name].channels or ())} → {list(b_inputs[name].channels or ())}"
                    )
    else:
        print("\n  INPUTS: unchanged")

    # Compare outputs
    a_outputs = {s.name: s for s in contract_a.outputs}
    b_outputs = {s.name: s for s in contract_b.outputs}

    removed_outputs = set(a_outputs) - set(b_outputs)
    added_outputs = set(b_outputs) - set(a_outputs)
    common_outputs = set(a_outputs) & set(b_outputs)

    if (
        removed_outputs
        or added_outputs
        or any(a_outputs[n] != b_outputs[n] for n in common_outputs)
    ):
        print("\n  OUTPUT CHANGES:")
        for name in sorted(removed_outputs):
            print(f"    - REMOVED: {name} {_fmt_shape(a_outputs[name].shape)}")
        for name in sorted(added_outputs):
            print(f"    + ADDED:   {name} {_fmt_shape(b_outputs[name].shape)}")
        for name in sorted(common_outputs):
            if a_outputs[name] != b_outputs[name]:
                print(f"    ~ CHANGED: {name}")
                if a_outputs[name].shape != b_outputs[name].shape:
                    print(
                        f"        shape: {_fmt_shape(a_outputs[name].shape)} → {_fmt_shape(b_outputs[name].shape)}"
                    )
    else:
        print("\n  OUTPUTS: unchanged")

    # Pipeline entry points
    changes = []
    for attr in ("build_pairs_fn", "train_fn", "export_fn"):
        va = getattr(contract_a, attr, "")
        vb = getattr(contract_b, attr, "")
        if va != vb:
            label = attr.replace("_fn", "")
            changes.append(f"    {label}: {va or '(none)'} → {vb or '(none)'}")
    if changes:
        print("\n  PIPELINE CHANGES:")
        for c in changes:
            print(c)

    print(f"\n  Fingerprint: {contract_a.fingerprint[:8]} → {contract_b.fingerprint[:8]}")
    print()


def run_contracts_cli(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``voxel-tree contracts`` sub-command."""
    from voxel_tree.contracts import CONTRACTS, get_contract, latest_revision, list_models

    parser = argparse.ArgumentParser(
        prog="voxel-tree contracts",
        description="Inspect and diff model I/O contracts",
    )
    parser.add_argument("model", nargs="?", help="Model family name (e.g. 'density')")
    parser.add_argument(
        "--revision",
        "-r",
        type=int,
        default=None,
        help="Show a specific revision (default: latest)",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        type=int,
        metavar=("FROM", "TO"),
        help="Diff two revisions (e.g. --diff 0 1)",
    )
    parser.add_argument("--sidecar", action="store_true", help="Dump sidecar JSON to stdout")
    parser.add_argument(
        "--all-revisions", "-a", action="store_true", help="Show all revisions for a model"
    )
    args = parser.parse_args(argv)

    if args.model is None:
        # List all models
        models = list_models()
        if not models:
            print("No contracts registered.")
            return
        summaries = []
        for name in models:
            rev = latest_revision(name)
            c = get_contract(name, revision=rev)
            summaries.append(
                {
                    "model_name": c.model_name,
                    "revision": c.revision,
                    "contract_id": c.contract_id,
                    "inputs": [s.to_dict() for s in c.inputs],
                    "outputs": [s.to_dict() for s in c.outputs],
                    "fingerprint": c.fingerprint,
                }
            )
        _print_table(summaries)

        # Also show if older revisions exist
        for name in models:
            revisions = sorted(r for (n, r) in CONTRACTS if n == name)
            if len(revisions) > 1:
                print(f"  ↳ {name} has {len(revisions)} revisions: {revisions}")
        return

    # Model specified
    if args.diff:
        from_rev, to_rev = args.diff
        try:
            ca = get_contract(args.model, revision=from_rev)
            cb = get_contract(args.model, revision=to_rev)
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        _diff_revisions(ca, cb)
        return

    if args.all_revisions:
        revisions = sorted(r for (n, r) in CONTRACTS if n == args.model)
        if not revisions:
            print(f"No contracts registered for model={args.model!r}")
            sys.exit(1)
        for rev in revisions:
            c = get_contract(args.model, revision=rev)
            _print_contract_detail(c)
        return

    try:
        c = get_contract(args.model, revision=args.revision)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.sidecar:
        print(json.dumps(c.to_sidecar(), indent=2))
    else:
        _print_contract_detail(c)
