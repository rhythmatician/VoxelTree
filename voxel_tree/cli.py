"""VoxelTree CLI — step-driven pipeline runner.

No args → launches the GUI.

Usage
-----
  voxel-tree                                        Launch the GUI
  voxel-tree --list-steps                          Print all registered step IDs
  voxel-tree --step STEP_ID                        Show step info
  voxel-tree --step STEP_ID --run                  Run step (default profile)
  voxel-tree --step STEP_ID --run --profile NAME   Run step with a named profile
  voxel-tree --server start [--role NAME]          Start the Fabric server
  voxel-tree --server stop                         Stop the Fabric server via RCON

All step settings (RCON credentials, data paths, training params, etc.) are
read from the profile YAML.  There are no per-step CLI flags.

Profiles live in the ``profiles/`` directory next to this package.
The default profile is the first YAML file found there alphabetically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _profiles_dir() -> Path:
    """Locate the ``profiles/`` directory by walking up from this module."""
    cur = Path(__file__).resolve()
    for _ in range(6):
        candidate = cur.parent / "profiles"
        if candidate.exists():
            return candidate
        cur = cur.parent
    return Path(__file__).resolve().parents[2] / "profiles"


def _repo_root() -> Path:
    """Best-effort repo root (parent of the ``profiles/`` directory)."""
    return _profiles_dir().parent


def _load_profile(name: str) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    profiles = _profiles_dir()
    # Try ``profiles/<name>.yaml`` first, then treat *name* as a raw path.
    path = profiles / f"{name}.yaml"
    if not path.exists():
        path = Path(name)
    if not path.exists():
        print(f"error: profile {name!r} not found in {profiles}", file=sys.stderr)
        sys.exit(1)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _default_profile_name() -> str | None:
    """Return the stem of the first YAML in ``profiles/``, or None."""
    yamls = sorted(_profiles_dir().glob("*.yaml"))
    return yamls[0].stem if yamls else None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _launch_gui() -> None:
    from voxel_tree.gui.app import create_app  # noqa: PLC0415
    from voxel_tree.gui.main_window import MainWindow  # noqa: PLC0415

    app = create_app()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def _cmd_list_steps() -> None:
    """Print all registered step IDs with their flags."""
    from voxel_tree.gui.step_definitions import PIPELINE_STEPS  # noqa: PLC0415

    print("Registered pipeline steps:")
    print(f"  {'Step ID':<32} {'Label':<12} {'Flags':<18} Notes")
    print("  " + "-" * 72)
    for step in PIPELINE_STEPS:
        flags: list[str] = []
        if step.server_required:
            flags.append("server")
        if step.client_required:
            flags.append("client")
        notes = "[stub]" if not step.enabled else ""
        prereqs = ", ".join(step.prereqs) if step.prereqs else ""
        flag_str = ", ".join(flags)
        print(f"  {step.id:<32} {step.label:<12} {flag_str:<18} {notes}")
        if prereqs:
            print(f"  {'':32} {'':12} after: {prereqs}")


def _cmd_show_step(step_id: str) -> None:
    """Print metadata for one step."""
    from voxel_tree.gui.step_definitions import STEP_BY_ID  # noqa: PLC0415

    step = STEP_BY_ID.get(step_id)
    if step is None:
        known = sorted(STEP_BY_ID.keys())
        print(f"error: unknown step {step_id!r}", file=sys.stderr)
        print(f"  known steps: {', '.join(known)}", file=sys.stderr)
        sys.exit(1)

    print(f"Step:     {step.id}")
    print(f"  Label:   {step.label}")
    print(f"  Phase:   {step.phase}")
    print(f"  Track:   {step.track or '(none)'}")
    print(f"  Prereqs: {', '.join(step.prereqs) or '(none)'}")
    print(f"  Server:  {'yes' if step.server_required else 'no'}")
    print(f"  Client:  {'yes' if step.client_required else 'no'}")
    print(f"  Enabled: {'yes' if step.enabled else 'no (stub)'}")
    print()
    print(f"Run with:  voxel-tree --step {step.id} --run [--profile NAME]")


def _cmd_run_step(step_id: str, profile_name: str | None) -> None:
    """Run a pipeline step directly by calling its run_fn."""
    import os  # noqa: PLC0415

    from voxel_tree.gui.step_definitions import STEP_BY_ID  # noqa: PLC0415

    step = STEP_BY_ID.get(step_id)
    if step is None:
        known = sorted(STEP_BY_ID.keys())
        print(f"error: unknown step {step_id!r}", file=sys.stderr)
        print(f"  known steps: {', '.join(known)}", file=sys.stderr)
        sys.exit(1)

    name = profile_name or _default_profile_name()
    if name is None:
        print(
            "error: no profiles found — create one via the GUI first",
            file=sys.stderr,
        )
        sys.exit(1)

    profile = _load_profile(name)

    print(f"[voxel-tree] step={step_id!r}  profile={name!r}")

    os.chdir(str(_repo_root()))
    step.run_fn(profile)


def _cmd_server(action: str, role: str | None) -> None:
    """Start or stop the Fabric server from the command line (no Qt required)."""
    import subprocess  # noqa: PLC0415

    from voxel_tree.gui.server_config import get_role  # noqa: PLC0415
    from voxel_tree.gui.server_manager import (  # noqa: PLC0415
        _JAR_PATH,
        _RUNTIME_DIR,
        _patch_server_properties,
        get_rcon_settings,
    )

    if action == "start":
        role_cfg = get_role(role)
        print(f"[server] Configuring role {role_cfg.name!r}:")
        print(f"  level-name : {role_cfg.level_name}")
        print(f"  seed       : {role_cfg.seed}")
        print(f"  server-port: {role_cfg.server_port}")
        print(f"  rcon-port  : {role_cfg.rcon_port}")
        _patch_server_properties(
            {
                "level-name": role_cfg.level_name,
                "level-seed": str(role_cfg.seed),
                "server-port": str(role_cfg.server_port),
                "rcon.port": str(role_cfg.rcon_port),
                "rcon.password": role_cfg.rcon_password,
                "enable-rcon": "true",
            }
        )
        if not _JAR_PATH or not _JAR_PATH.exists():
            print(
                "error: server JAR not found — expected in tools/fabric-server/",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[server] java -jar {_JAR_PATH.name} --nogui  (cwd: {_RUNTIME_DIR})")
        # Run in foreground so Ctrl-C cleanly stops the server.
        result = subprocess.run(
            ["java", "-jar", str(_JAR_PATH), "--nogui"],
            cwd=str(_RUNTIME_DIR),
        )
        sys.exit(result.returncode)

    elif action == "stop":
        from voxel_tree.utils.rcon import RconClient  # noqa: PLC0415

        rcon = get_rcon_settings()
        print(f"[server] Sending /stop via RCON ({rcon['host']}:{rcon['port']})…")
        try:
            with RconClient(str(rcon["host"]), int(rcon["port"]), str(rcon["password"])) as rc:
                rc.command("stop")
            print("[server] /stop sent.")
        except Exception as exc:  # noqa: BLE001
            print(f"error: RCON /stop failed: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"error: unknown server action {action!r} (use start|stop)", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch; no args → GUI."""
    import argparse  # noqa: PLC0415

    if argv is None:
        argv = sys.argv[1:]

    # Fast path: no args → GUI.
    if not argv:
        _launch_gui()
        return

    parser = argparse.ArgumentParser(
        prog="voxel-tree",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Primary mode flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-steps",
        action="store_true",
        help="Print all registered step IDs and exit",
    )
    mode.add_argument(
        "--server",
        metavar="ACTION",
        choices=["start", "stop"],
        help="Start or stop the Fabric server (start | stop)",
    )

    # Step selection + execution
    parser.add_argument("--step", metavar="STEP_ID", help="Step to inspect or run")
    parser.add_argument("--run", action="store_true", help="Execute the step")
    parser.add_argument(
        "--profile",
        metavar="NAME",
        default=None,
        help="Profile name (default: first YAML in profiles/)",
    )

    # Server role selection
    parser.add_argument(
        "--role",
        metavar="ROLE",
        default=None,
        help="Server role from servers.yaml (e.g. train, validate)",
    )

    args = parser.parse_args(argv)

    if args.list_steps:
        _cmd_list_steps()
    elif args.server:
        _cmd_server(args.server, args.role)
    elif args.step and args.run:
        _cmd_run_step(args.step, args.profile)
    elif args.step:
        _cmd_show_step(args.step)
    else:
        _launch_gui()


if __name__ == "__main__":
    main()
