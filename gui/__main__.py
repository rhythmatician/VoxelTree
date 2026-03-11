"""Entry point: python -m gui  (run from VoxelTree/ directory)"""

import sys
import traceback
from pathlib import Path

# Ensure the VoxelTree root is on sys.path so `gui.*` imports resolve
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    print("[1] Importing create_app...", flush=True)
    from gui.app import create_app

    print("[2] Importing MainWindow...", flush=True)
    from gui.main_window import MainWindow

    def main() -> None:
        try:
            app = create_app()
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
        except Exception as e:
            print(f"[ERROR] Exception in main(): {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)

    if __name__ == "__main__":
        try:
            print("[0] Starting GUI...", flush=True)
            main()
        except Exception as e:
            print(f"[FATAL] Uncaught exception: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)

except ImportError as e:
    print(f"[IMPORT ERROR] {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


if __name__ == "__main__":
    main()
