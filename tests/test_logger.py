import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from train.logger import TrainingLogger


class TestTrainingLogger:
    """Test suite for TrainingLogger class."""

    def test_logger_init_creates_directories(self):
        """Test that logger creates log directory and initializes files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)

            assert log_dir.exists()
            assert logger.csv_path.exists()
            assert logger.log_dir == log_dir

    def test_csv_initialization_creates_headers(self):
        """Test that CSV file is initialized with correct headers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)

            with open(logger.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                headers = next(reader)
                expected_headers = ["timestamp", "epoch", "loss", "lr", "epoch_time", "global_step"]
                assert headers == expected_headers

    def test_tensorboard_init_with_available_tensorboard(self):
        """Test TensorBoard initialization when available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            # Create a mock SummaryWriter class
            mock_writer_class = Mock()
            mock_writer_instance = Mock()
            mock_writer_class.return_value = mock_writer_instance

            # Patch the module-level import to simulate TensorBoard being available
            with patch("train.logger.SummaryWriter", mock_writer_class):
                logger = TrainingLogger(log_dir, use_tensorboard=True)
                assert logger.use_tensorboard is True
                mock_writer_class.assert_called_once()
                assert logger.writer is mock_writer_instance

    def test_tensorboard_init_with_unavailable_tensorboard(self):
        """Test TensorBoard initialization when not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            # Mock SummaryWriter to be None (as if import failed)
            with patch("train.logger.SummaryWriter", None):
                logger = TrainingLogger(log_dir, use_tensorboard=True)

                assert logger.use_tensorboard is False
                assert logger.writer is None

    def test_log_metrics_to_csv(self):
        """Test logging metrics to CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)

            metrics = {"epoch": 1, "loss": 0.5, "lr": 0.001, "epoch_time": 10.5, "global_step": 100}
            logger.log_metrics(metrics)

            with open(logger.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # Skip headers
                row = next(reader)

                assert row[1] == "1"  # epoch
                assert row[2] == "0.5"  # loss
                assert row[3] == "0.001"  # lr
                assert row[4] == "10.5"  # epoch_time
                assert row[5] == "100"  # global_step

    def test_log_metrics_to_jsonl(self):
        """Test logging metrics to JSONL file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)

            metrics = {"epoch": 1, "loss": 0.5, "lr": 0.001}
            logger.log_metrics(metrics)

            with open(logger.json_path, "r") as f:
                line = f.readline().strip()
                logged_data = json.loads(line)

                assert logged_data["epoch"] == 1
                assert logged_data["loss"] == 0.5
                assert logged_data["lr"] == 0.001
                assert "timestamp" in logged_data

    def test_log_to_tensorboard_with_writer(self):
        """Test logging to TensorBoard when writer is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            mock_writer = Mock()
            logger = TrainingLogger(log_dir)
            logger.writer = mock_writer
            logger.use_tensorboard = True

            metrics = {"epoch": 1, "loss": 0.5, "lr": 0.001}
            step = 100

            logger._log_to_tensorboard(metrics, step)

            # Verify add_scalar was called for each numeric metric
            expected_calls = [("epoch", 1, 100), ("loss", 0.5, 100), ("lr", 0.001, 100)]

            assert mock_writer.add_scalar.call_count == 3
            mock_writer.flush.assert_called_once()

    def test_log_to_tensorboard_without_writer(self):
        """Test logging to TensorBoard when writer is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)
            logger.writer = None

            metrics = {"epoch": 1, "loss": 0.5}
            logger._log_to_tensorboard(metrics, 100)

            # Should return early without error

    def test_log_to_tensorboard_with_default_step(self):
        """Test TensorBoard logging uses global_step when step is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            mock_writer = Mock()
            logger = TrainingLogger(log_dir)
            logger.writer = mock_writer

            metrics = {"loss": 0.5, "global_step": 50}
            logger._log_to_tensorboard(metrics, None)

            # Should use global_step value as step
            mock_writer.add_scalar.assert_called_with("loss", 0.5, 50)

    def test_log_to_tensorboard_filters_non_numeric(self):
        """Test TensorBoard logging filters out non-numeric values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            mock_writer = Mock()
            logger = TrainingLogger(log_dir)
            logger.writer = mock_writer

            metrics = {"loss": 0.5, "model_name": "unet", "timestamp": 123456789}
            logger._log_to_tensorboard(metrics, 100)

            # Should only log numeric values, excluding timestamp
            mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 100)

    def test_close_logger(self):
        """Test closing logger resources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            mock_writer = Mock()
            logger = TrainingLogger(log_dir)
            logger.writer = mock_writer

            logger.close()

            mock_writer.close.assert_called_once()

    def test_close_logger_without_writer(self):
        """Test closing logger when writer is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            logger = TrainingLogger(log_dir)
            logger.writer = None

            # Should not raise error
            logger.close()
