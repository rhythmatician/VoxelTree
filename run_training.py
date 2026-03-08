"""Wrapper script to run training with file-based logging."""

import subprocess
import sys
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "training.log"

cmd = [
    sys.executable,
    "-u",
    "train_multi_lod.py",
    "--data-dir",
    "data/voxy_training",
    "--output-dir",
    "models/voxy",
    "--epochs",
    "20",
    "--batch-size",
    "16",
    "--save-every",
    "5",
    "--validate-every",
    "5",
    "--lr",
    "3e-4",
]

print(f"Starting training, logging to {log_file}")
print(f"Command: {' '.join(cmd)}")

with open(log_file, "w") as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
    )
    proc.wait()
    print(f"Training finished with exit code {proc.returncode}")
