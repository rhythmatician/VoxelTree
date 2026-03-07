"""Quick real-data training trial: 3 epochs, then eval metrics."""

import sys
import time
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))


from train.multi_lod_dataset import MultiLODDataset, collate_multi_lod_batch
from train.trainer import VoxelTrainer

start_time = time.localtime()

CONFIG_PATH = "config.yaml"
DATA_TRAIN = "data/processed/train"
DATA_VAL = "data/processed/val"
EPOCHS = 3
BATCH_SIZE = 8

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

cfg["training"] = cfg.get("training", {})
cfg["training"]["learning_rate"] = 1e-3
cfg["training"]["num_epochs"] = EPOCHS

trainer = VoxelTrainer(cfg)
train_ds = MultiLODDataset(DATA_TRAIN, split="train")
val_ds = MultiLODDataset(DATA_VAL, split="val")
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, collate_fn=collate_multi_lod_batch, shuffle=True
)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_multi_lod_batch)

print(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")
print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
print()

for epoch in range(EPOCHS):
    train_m = trainer.train_one_epoch(train_dl)
    val_m = trainer.validate_one_epoch(val_dl)
    print(
        f"Epoch {epoch}: train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}"
        f"  lr={train_m.get('lr', '?')}"
    )

print("\nTraining complete.")
duration = time.mktime(time.localtime()) - time.mktime(start_time)
print(f"Duration: {duration} seconds")
