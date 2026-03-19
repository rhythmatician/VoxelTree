"""step_runner — subprocess bridge for pipeline steps.

Invoked by ``RunWorker`` as::

    python -m voxel_tree.step_runner <step_id>

The profile dict is read as JSON from **stdin**.  The step's ``run_fn``
is looked up from the step registry and called directly — no argparse
round-trip, no CLI indirection.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m voxel_tree.step_runner <step_id>", file=sys.stderr)
        sys.exit(2)

    step_id = sys.argv[1]

    # Read profile JSON from stdin (avoids command-line escaping issues).
    profile: dict[str, object] = json.loads(sys.stdin.read())

    # Sanitise sys.argv so scripts that call argparse don't see our args.
    sys.argv = [f"step_runner:{step_id}"]

    # Ensure cwd is the VoxelTree repo root (same as old RunWorker cwd).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(repo_root)

    from voxel_tree.gui.step_definitions import STEP_BY_ID  # noqa: PLC0415

    step = STEP_BY_ID.get(step_id)
    if step is None:
        print(f"error: unknown step {step_id!r}", file=sys.stderr)
        sys.exit(1)

    try:
        step.run_fn(profile)
    except SystemExit as exc:
        sys.exit(exc.code if exc.code is not None else 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
