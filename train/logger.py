"""
Training logging utilities for VoxelTree.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import SummaryWriter at module level for easier testing
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore


class TrainingLogger:
    """Handles CSV and TensorBoard logging for training metrics."""

    def __init__(self, log_dir: Path, use_tensorboard: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_log.csv"
        self.json_path = self.log_dir / "training_log.jsonl"

        self.use_tensorboard = use_tensorboard
        self.writer = None

        if use_tensorboard:
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            else:
                print("Warning: TensorBoard not available, falling back to CSV only")
                self.use_tensorboard = False

        # Initialize CSV file with headers
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "loss", "lr", "epoch_time", "global_step"])

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to CSV and optionally TensorBoard."""
        timestamp = time.time()

        # Add timestamp to metrics
        metrics_with_time = {"timestamp": timestamp, **metrics}

        # Log to CSV
        self._log_to_csv(metrics_with_time)

        # Log to JSONL
        self._log_to_jsonl(metrics_with_time)

        # Log to TensorBoard if available
        if self.use_tensorboard and self.writer is not None:
            self._log_to_tensorboard(metrics, step)

    def _log_to_csv(self, metrics: Dict[str, Any]):
        """Log metrics to CSV file."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                metrics.get("timestamp", ""),
                metrics.get("epoch", ""),
                metrics.get("loss", ""),
                metrics.get("lr", ""),
                metrics.get("epoch_time", ""),
                metrics.get("global_step", ""),
            ]
            writer.writerow(row)

    def _log_to_jsonl(self, metrics: Dict[str, Any]):
        """Log metrics to JSONL file."""
        with open(self.json_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _log_to_tensorboard(self, metrics: Dict[str, Any], step: Optional[int]):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        if step is None:
            step = metrics.get("global_step", 0)

        for key, value in metrics.items():
            # Skip timestamp and global_step (global_step is used as step parameter, not logged as metric)
            if isinstance(value, (int, float)) and key not in ["timestamp", "global_step"]:
                self.writer.add_scalar(key, value, step)

        self.writer.flush()

    def close(self):
        """Close logging resources."""
        if self.writer is not None:
            self.writer.close()
