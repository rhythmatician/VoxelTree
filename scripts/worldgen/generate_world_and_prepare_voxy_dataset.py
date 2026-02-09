"""
Generate a world via FabricWorldGenBootstrap and prepare artifacts for Voxy datasets.

This script:
1) Generates terrain via Chunky.
2) Optionally freezes ticks (Carpet).
3) Optionally attempts Voxy import commands.
4) Writes a run manifest alongside generated .mca files.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.worldgen.bootstrap import FabricWorldGenBootstrap

logger = logging.getLogger(__name__)


def _parse_mc_version(jar_path: str) -> Optional[str]:
    try:
        filename = Path(jar_path).name
    except Exception:
        return None
    if "mc." not in filename:
        return None
    parts = filename.split("mc.")
    if len(parts) < 2:
        return None
    version = parts[1].split("-")[0]
    return version or None


def _mod_installed(mod_paths: list[Path], keyword: str) -> bool:
    keyword = keyword.lower()
    return any(keyword in mod_path.name.lower() for mod_path in mod_paths)


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate terrain and prepare Voxy dataset artifacts.")
    parser.add_argument("--center-x", type=int, default=0)
    parser.add_argument("--center-z", type=int, default=0)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--world-name", type=str, default="voxeltree_world")
    args = parser.parse_args()

    bootstrap = FabricWorldGenBootstrap()
    worldgen_config = bootstrap.config.get("worldgen", {})

    configured_mods = worldgen_config.get("mods", [])
    resolved_mods = bootstrap._resolve_mod_paths(configured_mods)
    has_carpet = _mod_installed(resolved_mods, "carpet")
    has_voxy = _mod_installed(resolved_mods, "voxy")

    post_commands: list[str] = []
    if has_carpet:
        post_commands.append("tick freeze")
    if has_voxy:
        post_commands.append("voxy import world")

    run_start = datetime.now(timezone.utc).isoformat()
    world_path = bootstrap.generate_chunks(
        center_x=args.center_x,
        center_z=args.center_z,
        radius=args.radius,
        world_name=args.world_name,
        post_generation_commands=post_commands,
    )
    run_end = datetime.now(timezone.utc).isoformat()

    if world_path is None:
        logger.error("World generation failed.")
        return 1

    if has_voxy and bootstrap.was_command_unknown("voxy import world"):
        logger.warning(
            "/voxy commands are not available on this dedicated server; you likely need a "
            "client/integrated-server environment for Voxy import/export. Proceeding with only "
            ".mca output."
        )

    manifest = {
        "seed": worldgen_config.get("seed"),
        "mc_version": _parse_mc_version(worldgen_config.get("java_tools", {}).get("primary", "")),
        "world_name": args.world_name,
        "center": {"x": args.center_x, "z": args.center_z},
        "radius": args.radius,
        "generate_structures": worldgen_config.get("generate_structures", False),
        "mods_copied": bootstrap.copied_mods,
        "timestamps": {"start": run_start, "end": run_end},
    }

    manifest_path = Path("data") / "test_world" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote manifest to {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
