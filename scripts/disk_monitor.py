#!/usr/bin/env python3
"""
VoxelTree Disk Monitor

This script provides utilities for tracking and managing disk usage during
large-scale dataset generation and training. It can be used as a standalone
CLI tool or imported as a utility module.

Usage:
    # As a CLI tool
    python scripts/disk_monitor.py --threshold 10 --path data/

    # Imported as a module
    from scripts.disk_monitor import DiskMonitor
    monitor = DiskMonitor(threshold_gb=10)

    # Monitor disk usage with a callback
    monitor.start_monitoring(path="data/", interval=60, callback=my_callback_fn)
"""

import argparse
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("DiskMonitor")


class DiskMonitor:
    """
    Monitor disk usage during dataset generation and training.

    This class provides utilities to:
    - Track disk usage of specific directories
    - Enforce disk usage limits
    - Clean up temporary files when needed
    - Alert when disk space is running low
    """

    def __init__(
        self,
        threshold_gb: float = 10.0,
        critical_threshold_gb: float = 2.0,
        temp_dirs: Optional[List[str]] = None,
    ):
        """
        Initialize the disk monitor.

        Args:
            threshold_gb: Warning threshold in GB
            critical_threshold_gb: Critical threshold in GB (emergency cleanup)
            temp_dirs: List of temporary directories that can be cleaned up
        """
        self.threshold_gb = threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self.temp_dirs = temp_dirs or []
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.monitoring_paths: Set[str] = set()

        logger.info(
            f"DiskMonitor initialized with threshold={threshold_gb}GB, "
            f"critical_threshold={critical_threshold_gb}GB"
        )

    def check_disk_space(self, path: str = ".") -> Tuple[float, bool, bool]:
        """
        Check available disk space for the given path.

        Args:
            path: Directory path to check

        Returns:
            Tuple of (available_space_gb, warning_triggered, critical_triggered)
        """
        disk_usage = psutil.disk_usage(path)
        available_gb = disk_usage.free / (1024**3)  # Convert bytes to GB

        warning_triggered = available_gb < self.threshold_gb
        critical_triggered = available_gb < self.critical_threshold_gb

        if warning_triggered:
            logger.warning(
                f"Low disk space on {path}: {available_gb:.2f}GB available, "
                f"threshold: {self.threshold_gb}GB"
            )

        if critical_triggered:
            logger.critical(
                f"CRITICAL: Extremely low disk space on {path}: {available_gb:.2f}GB available, "
                f"critical threshold: {self.critical_threshold_gb}GB"
            )

        return available_gb, warning_triggered, critical_triggered

    def get_directory_size(self, path: str) -> Tuple[float, int]:
        """
        Get the size of a directory and the number of files in it.

        Args:
            path: Directory path to analyze

        Returns:
            Tuple of (size_in_gb, file_count)
        """
        path_obj = Path(path)

        if not path_obj.exists():
            return 0.0, 0

        # Check if it's a file
        if path_obj.is_file():
            size_bytes = path_obj.stat().st_size
            return size_bytes / (1024**3), 1

        # It's a directory, calculate size recursively
        total_size = 0
        file_count = 0

        try:
            for item in path_obj.glob("**/*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        except (PermissionError, OSError) as e:
            logger.warning(f"Error accessing {path}: {e}")

        return total_size / (1024**3), file_count

    def cleanup_temp_dirs(self) -> float:
        """
        Clean up temporary directories to free disk space.

        Returns:
            Space freed in GB
        """
        if not self.temp_dirs:
            logger.warning("No temporary directories configured for cleanup")
            return 0.0

        total_freed = 0.0

        for temp_dir in self.temp_dirs:
            if not Path(temp_dir).exists():
                continue

            # Check size before deletion
            size_gb, file_count = self.get_directory_size(temp_dir)

            try:
                logger.info(f"Cleaning up {temp_dir}: {size_gb:.2f}GB, {file_count} files")

                # Delete directory contents
                for item in Path(temp_dir).glob("*"):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

                total_freed += size_gb
                logger.info(f"Cleaned up {temp_dir}, freed {size_gb:.2f}GB")
            except Exception as e:
                logger.error(f"Error cleaning up {temp_dir}: {e}")

        return total_freed

    def analyze_disk_usage(self, base_path: str = ".") -> Dict:
        """
        Analyze disk usage for the given path.

        Args:
            base_path: Directory path to analyze

        Returns:
            Dictionary with usage statistics
        """
        results = {}

        # Get overall disk information
        disk_usage = psutil.disk_usage(base_path)
        results["total_gb"] = disk_usage.total / (1024**3)
        results["used_gb"] = disk_usage.used / (1024**3)
        results["free_gb"] = disk_usage.free / (1024**3)
        results["percent_used"] = disk_usage.percent

        # Get directory breakdown (top-level only)
        dir_stats = {}
        try:
            base_path_obj = Path(base_path)
            for item in base_path_obj.glob("*"):
                if item.is_dir():
                    size_gb, file_count = self.get_directory_size(str(item))
                    if size_gb > 0.01:  # Only include dirs larger than 10MB
                        dir_stats[item.name] = {"size_gb": size_gb, "file_count": file_count}
        except Exception as e:
            logger.error(f"Error analyzing directory structure: {e}")

        results["directories"] = dir_stats

        # Get largest files (optional)
        results["timestamp"] = datetime.now().isoformat()

        return results

    def monitor_disk_space(
        self, path: str = ".", interval: int = 60, callback: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """
        Monitor disk space and trigger callback when thresholds are reached.

        Args:
            path: Path to monitor
            interval: Check interval in seconds
            callback: Function to call when thresholds are reached
        """
        self.monitoring_paths.add(path)

        while not self.stop_event.is_set():
            available_gb, warning, critical = self.check_disk_space(path)

            results = {
                "path": path,
                "available_gb": available_gb,
                "warning_triggered": warning,
                "critical_triggered": critical,
                "threshold_gb": self.threshold_gb,
                "critical_threshold_gb": self.critical_threshold_gb,
                "timestamp": datetime.now().isoformat(),
            }

            # If critically low, try emergency cleanup
            if critical:
                freed = self.cleanup_temp_dirs()
                results["emergency_cleanup"] = True
                results["freed_gb"] = freed

                # Re-check space after cleanup
                available_gb, warning, critical = self.check_disk_space(path)
                results["available_gb_after_cleanup"] = available_gb
                results["still_critical"] = critical

            # Call the callback if provided
            if callback and (warning or critical):
                try:
                    callback(results)
                except Exception as e:
                    logger.error(f"Error in disk space callback: {e}")

            # Sleep for the specified interval
            self.stop_event.wait(interval)

    def start_monitoring(
        self, path: str = ".", interval: int = 60, callback: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """
        Start monitoring disk space in a background thread.

        Args:
            path: Path to monitor
            interval: Check interval in seconds
            callback: Function to call when thresholds are reached
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return

        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self.monitor_disk_space,
            kwargs={"path": path, "interval": interval, "callback": callback},
            daemon=True,
        )
        self.monitoring_thread.start()
        logger.info(f"Started monitoring disk space for {path} (interval: {interval}s)")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_thread:
            logger.warning("No monitoring thread is running")
            return

        self.stop_event.set()
        self.monitoring_thread.join(timeout=5)
        logger.info("Stopped disk space monitoring")


def default_alert_callback(results: Dict) -> None:
    """Default callback function for disk alerts."""
    if results.get("critical_triggered", False):
        logger.critical(
            f"CRITICAL DISK SPACE ALERT: {results['available_gb']:.2f}GB available on {results['path']}"
        )

        if results.get("emergency_cleanup", False):
            logger.warning(
                f"Emergency cleanup freed {results.get('freed_gb', 0):.2f}GB, "
                f"now {results.get('available_gb_after_cleanup', 0):.2f}GB available"
            )
    elif results.get("warning_triggered", False):
        logger.warning(
            f"LOW DISK SPACE WARNING: {results['available_gb']:.2f}GB available on {results['path']}"
        )


def main():
    """CLI entry point for disk monitor."""
    parser = argparse.ArgumentParser(description="VoxelTree Disk Monitor")
    parser.add_argument("--threshold", type=float, default=10.0, help="Warning threshold in GB")
    parser.add_argument("--critical", type=float, default=2.0, help="Critical threshold in GB")
    parser.add_argument("--path", type=str, default=".", help="Path to monitor")
    parser.add_argument(
        "--temp-dirs", type=str, nargs="+", help="Temporary directories that can be cleaned up"
    )
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument(
        "--analyze", action="store_true", help="Run analysis of disk usage and exit"
    )

    args = parser.parse_args()

    # Configure logging to file
    log_file = Path("disk_monitor.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Create monitor
    monitor = DiskMonitor(
        threshold_gb=args.threshold, critical_threshold_gb=args.critical, temp_dirs=args.temp_dirs
    )

    if args.analyze:
        # Run analysis and exit
        logger.info(f"Analyzing disk usage for {args.path}")
        analysis = monitor.analyze_disk_usage(args.path)

        print("\n===== DISK USAGE ANALYSIS =====")
        print(f"Total: {analysis['total_gb']:.2f} GB")
        print(f"Used:  {analysis['used_gb']:.2f} GB ({analysis['percent_used']}%)")
        print(f"Free:  {analysis['free_gb']:.2f} GB")
        print("\nLargest directories:")

        # Sort directories by size
        sorted_dirs = sorted(
            analysis["directories"].items(), key=lambda x: x[1]["size_gb"], reverse=True
        )

        for dir_name, stats in sorted_dirs[:10]:  # Show top 10
            print(f"  {dir_name}: {stats['size_gb']:.2f} GB ({stats['file_count']} files)")

        print(f"\nAnalysis completed at {analysis['timestamp']}")
        return 0

    try:
        # Start monitoring
        logger.info(
            f"Starting disk monitoring for {args.path} "
            f"(threshold: {args.threshold}GB, critical: {args.critical}GB)"
        )

        monitor.start_monitoring(
            path=args.path, interval=args.interval, callback=default_alert_callback
        )

        # Run until interrupted
        print(f"Monitoring disk space... (Press Ctrl+C to stop)")
        print(f"Logs are being saved to {log_file}")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        monitor.stop_monitoring()

    except Exception as e:
        logger.error(f"Error in disk monitor: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
