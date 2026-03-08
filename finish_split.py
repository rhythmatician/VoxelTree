"""Move remaining NPZ files from root to train directory."""

from pathlib import Path

src = Path("data/voxy_training")
train_dir = src / "train"

remaining = sorted(src.glob("*.npz"))
print(f"Moving {len(remaining)} remaining files to train/")

for i, f in enumerate(remaining):
    f.rename(train_dir / f.name)
    if (i + 1) % 10000 == 0:
        print(f"  moved {i + 1}/{len(remaining)}")

print(f"Done! Train: {len(list(train_dir.glob('*.npz')))} files")
