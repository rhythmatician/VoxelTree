"""Entry point: python -m VoxelTree.gui  (or invoked via voxel-tree gui)"""

import sys
import traceback


def main() -> None:
    try:
        from voxel_tree.gui.app import create_app
        from voxel_tree.gui.main_window import MainWindow

        app = create_app()
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"[ERROR] Exception in main(): {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
