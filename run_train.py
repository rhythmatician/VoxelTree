"""Wrapper to launch training with line-buffered dual output (console + log file)."""

import subprocess
import sys
from pathlib import Path

log_file = Path("logs/training_v2.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-u",
    "train_multi_lod.py",
    "--data-dir",
    "data/voxy_subset",
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
    "--num-workers",
    "0",
]

print(f"Launching: {' '.join(cmd)}", flush=True)
print(f"Log file: {log_file.resolve()}", flush=True)

with open(log_file, "w", encoding="utf-8") as flog:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        flog.write(line)
        flog.flush()
    proc.wait()

print(f"\nProcess exited with code {proc.returncode}", flush=True)
sys.exit(proc.returncode)
