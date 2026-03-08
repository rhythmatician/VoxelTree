"""Create a smaller training subset for practical CPU training."""

import random
import shutil
from pathlib import Path

src_train = Path("data/voxy_training/train")
subset_dir = Path("data/voxy_subset/train")
val_src = Path("data/voxy_training/val")
val_dst = Path("data/voxy_subset/val")

TRAIN_SIZE = 20000
VAL_SIZE = 2000

# Create directories
subset_dir.mkdir(parents=True, exist_ok=True)
val_dst.mkdir(parents=True, exist_ok=True)

# Clean existing subset
for f in subset_dir.glob("*.npz"):
    f.unlink()
for f in val_dst.glob("*.npz"):
    f.unlink()

# Sample training files
train_files = sorted(src_train.glob("*.npz"))
print(f"Source train files: {len(train_files)}")
random.seed(42)
train_sample = random.sample(train_files, min(TRAIN_SIZE, len(train_files)))
print(f"Sampled {len(train_sample)} train files")

for f in train_sample:
    shutil.copy2(f, subset_dir / f.name)

# Sample validation files
val_files = sorted(val_src.glob("*.npz"))
print(f"Source val files: {len(val_files)}")
val_sample = random.sample(val_files, min(VAL_SIZE, len(val_files)))
print(f"Sampled {len(val_sample)} val files")

for f in val_sample:
    shutil.copy2(f, val_dst / f.name)

print("Done!")
print(f"  Train subset: {len(list(subset_dir.glob('*.npz')))} files")
print(f"  Val subset: {len(list(val_dst.glob('*.npz')))} files")
