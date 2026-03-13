#!/usr/bin/env python3
"""Deploy exported ONNX models to the LODiffusion runtime directory.

Reads ``required_files`` from ``pipeline_manifest.json`` and copies exactly
those files — nothing more, nothing less.  Fails fast if any are missing.

Usage::

    python scripts/deploy_models.py production/v7

    # Custom destination:
    python scripts/deploy_models.py production/v7 --dest path/to/config/lodiffusion
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deploy ONNX models to LODiffusion")
    parser.add_argument("source", type=Path, help="Production directory (e.g. production/v7)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: ../LODiffusion/run/config/lodiffusion)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["init", "refine", "leaf"],
        help="Only deploy the specified submodels; others will be left untouched",
    )
    args = parser.parse_args(argv)

    source: Path = args.source.resolve()
    if args.dest:
        dest = args.dest.resolve()
    else:
        # Default destination: locate the LODiffusion repo as a sibling of the
        # VoxelTree repo inside the workspace.
        # Layout on disk:
        #   MC/
        #     VoxelTree/          ← repo root (3 levels up from __file__)
        #       VoxelTree/        ← Python package
        #         scripts/        ← this file's directory
        #     LODiffusion/        ← target repo (sibling of VoxelTree repo)
        #
        # __file__ → .parent → scripts/
        #           → .parent.parent → VoxelTree/ (pkg)
        #           → .parent.parent.parent → VoxelTree/ (repo, 3rd parent)
        #           → .parent.parent.parent.parent → MC/ (workspace root)
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        candidate = workspace_root / "LODiffusion"
        dest = candidate / "run" / "config" / "lodiffusion"

    # ── Read manifest ────────────────────────────────────────────────
    manifest_path = source / "pipeline_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No pipeline_manifest.json in {source}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    required = manifest.get("required_files")
    if not required:
        print("ERROR: Manifest has no 'required_files' list", file=sys.stderr)
        sys.exit(1)

    # filter if models argument provided
    if args.models:
        # keep only filenames that mention the requested submodels
        filtered: list[str] = []
        for m in args.models:
            if m == "init":
                filtered += [n for n in required if "init" in n]
            elif m == "refine":
                filtered += [n for n in required if "refine" in n]
            elif m == "leaf":
                filtered += [n for n in required if "leaf" in n]
        # always keep manifest itself so we can read later if needed
        if "pipeline_manifest.json" in required and "pipeline_manifest.json" not in filtered:
            filtered.insert(0, "pipeline_manifest.json")
        required = filtered

    # ── Validate all files exist in source ───────────────────────────
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        print(f"ERROR: {len(missing)} required file(s) missing from {source}:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)

    # ── Deploy ───────────────────────────────────────────────────────
    dest.mkdir(parents=True, exist_ok=True)

    # Clear old files only if we're deploying everything
    if not args.models:
        for old in dest.iterdir():
            if old.is_file():
                old.unlink()

    copied = []
    for name in required:
        shutil.copy2(source / name, dest / name)
        copied.append(name)

    print(f"Deployed {len(copied)} files: {source} -> {dest}")
    for name in copied:
        size_kb = (dest / name).stat().st_size / 1024
        print(f"  {name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
