"""VoxelTree unified CLI — ``voxel-tree [command] [args]``

No args → launches the GUI.

Subcommands
-----------
  gui            Launch the GUI (same as no args)
  train          Train the 3-model octree pipeline
  pipeline       Orchestrate train → export → deploy
  dataprep       Data preparation steps (extract, heights, pairs)
  pregen         RCON: Chunky chunk pre-generation
  voxy-import    RCON: /voxy import world <name>
  dumpnoise      RCON: /dumpnoise <radius>
  status         RCON: Server status
  freeze         RCON: Freeze world for deterministic data collection
  unfreeze       RCON: Unfreeze world
  rcon           Send a single RCON command
  extract        Extract octree data from Voxy RocksDB
  build-pairs    Build octree parent/child training pair caches
  export-onnx    Export checkpoint to 3 ONNX models
  deploy-models  Copy ONNX models to LODiffusion config directory
"""

from __future__ import annotations

import sys


def _launch_gui() -> None:
    from VoxelTree.gui.app import create_app
    from VoxelTree.gui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the requested subcommand, or launch GUI if no args given."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _launch_gui()
        return

    cmd, rest = argv[0], argv[1:]

    if cmd == "gui":
        _launch_gui()

    elif cmd == "train":
        from VoxelTree.train.train import main as _main

        _main(rest)

    elif cmd == "pipeline":
        from VoxelTree.preprocessing.pipeline import main as _main

        _main(rest)

    # Top-level aliases for pipeline subcommands
    elif cmd in ("export", "deploy", "run"):
        from VoxelTree.preprocessing.pipeline import main as _main

        _main([cmd] + rest)

    # data-cli subcommands
    elif cmd in (
        "dataprep",
        "pregen",
        "voxy-import",
        "dumpnoise",
        "status",
        "freeze",
        "unfreeze",
    ):
        from VoxelTree.preprocessing.cli import main as _main

        _main([cmd] + rest)

    elif cmd == "rcon":
        from VoxelTree.preprocessing.rcon import main as _main

        _main(rest)

    elif cmd in ("extract-octree", "extract"):
        from VoxelTree.scripts.extract_octree_data import main as _main

        _main(rest)

    elif cmd == "build-pairs":
        from VoxelTree.scripts.build_octree_pairs import main as _main

        _main(rest)

    elif cmd == "export-onnx":
        from VoxelTree.scripts.octree.export import main as _main

        _main(rest)

    elif cmd == "deploy-models":
        from VoxelTree.scripts.deploy import main as _main

        _main(rest)

    else:
        print(__doc__)
        print(f"error: unknown command {cmd!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
