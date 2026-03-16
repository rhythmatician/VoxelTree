from pathlib import Path

root = Path(__file__).resolve().parent.parent / "data" / "voxy_octree"
for level in range(5, -1, -1):
    p = root / f"level_{level}"
    if not p.exists():
        print(level, "missing")
        continue
    files = list(p.glob(f"voxy_L{level}_*.npz"))
    print(level, len(files), "files", [f.name for f in files[:5]])
