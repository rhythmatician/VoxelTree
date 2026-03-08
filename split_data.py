"""Split extracted NPZ files into train/val directories."""

import random
from pathlib import Path

src = Path("data/voxy_training")
train_dir = src / "train"
val_dir = src / "val"
train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

files = sorted(src.glob("*.npz"))
print(f"Total NPZ files: {len(files)}")

random.seed(42)
random.shuffle(files)

val_count = int(len(files) * 0.1)  # 10% validation
val_files = files[:val_count]
train_files = files[val_count:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}")

for f in val_files:
    f.rename(val_dir / f.name)
for f in train_files:
    f.rename(train_dir / f.name)

print("Done!")
print(f"  Train dir: {len(list(train_dir.glob('*.npz')))} files")
print(f"  Val dir: {len(list(val_dir.glob('*.npz')))} files")
