"""
PatchValidator - Phase 2.3 GREEN Implementation

Minimal implementation to make the RED tests pass.
Validates training example format and detects malformed data.
"""

import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PatchValidator:
    """
    Validates the format of training examples (linked LOD pairs with seed inputs).

    Ensures that all training examples have correct shapes, dtypes, and required keys
    before being fed to the dataset loader.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize PatchValidator with configuration."""
        self.config_path = config_path if config_path else Path("config.yaml")
        self._load_config()
        self._define_format_specs()

        logger.info("PatchValidator initialized")

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # Extract validation configuration
        validation_config = config.get("validation", {})
        self.strict_validation = validation_config.get("strict_validation", True)
        self.max_errors_per_file = validation_config.get("max_errors_per_file", 10)

    def _define_format_specs(self):
        """Define format specifications for training examples."""
        # Required keys for training examples
        self.required_keys = [
            "parent_voxel",
            "target_mask",
            "target_types",
            "biome_patch",
            "heightmap_patch",
            "river_patch",
            "y_index",
            "chunk_x",
            "chunk_z",
            "lod",
        ]

        # Expected shapes for each key
        self.shape_specs = {
            "parent_voxel": (8, 8, 8),
            "target_mask": (16, 16, 16),
            "target_types": (16, 16, 16),
            "biome_patch": (16, 16),
            "heightmap_patch": (16, 16),
            "river_patch": (16, 16),
        }

        # Expected dtypes for each key
        self.dtype_specs = {
            "parent_voxel": [np.bool_, bool],
            "target_mask": [np.bool_, bool],
            "target_types": [np.uint8],
            "biome_patch": [np.uint8],
            "heightmap_patch": [np.uint16],
            "river_patch": [np.float32],
            "y_index": [int, np.integer],
            "chunk_x": [int, np.integer],
            "chunk_z": [int, np.integer],
            "lod": [int, np.integer],
        }

        # Valid ranges for scalar values
        self.range_specs = {
            "y_index": (0, 23),  # 384 blocks / 16 = 24 subchunks (0-23)
            "lod": (1, 4),  # Valid LOD levels
            "target_types": (0, 255),  # Block type IDs
            "biome_patch": (0, 255),  # Biome IDs
            "river_patch": (-10.0, 10.0),  # River noise range (with some tolerance)
        }

    def validate_training_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate a single training example.

        Args:
            example: Training example dictionary

        Returns:
            True if valid, False otherwise
        """
        errors = self.get_validation_errors(example)
        return len(errors) == 0

    def get_validation_errors(self, example: Dict[str, Any]) -> List[str]:
        """
        Get list of validation errors for a training example.

        Args:
            example: Training example dictionary

        Returns:
            List of error messages
        """
        errors = []

        # Check for missing keys
        missing_keys = set(self.required_keys) - set(example.keys())
        for key in missing_keys:
            errors.append(f"Missing required key: {key}")

        # Check shapes and dtypes for present keys
        for key, value in example.items():
            if key in self.shape_specs:
                expected_shape = self.shape_specs[key]
                if hasattr(value, "shape") and value.shape != expected_shape:
                    errors.append(
                        f"Wrong shape for {key}: expected {expected_shape}, got {value.shape}"
                    )

            if key in self.dtype_specs:
                expected_dtypes = self.dtype_specs[key]
                if hasattr(value, "dtype"):
                    if not any(np.issubdtype(value.dtype, dt) for dt in expected_dtypes):
                        errors.append(
                            f"Wrong dtype for {key}: expected one of {expected_dtypes}, got {value.dtype}"
                        )
                else:
                    if not any(isinstance(value, dt) for dt in expected_dtypes):
                        errors.append(
                            f"Wrong type for {key}: expected one of {expected_dtypes}, got {type(value)}"
                        )

            # Check ranges for specific keys
            if key in self.range_specs:
                min_val, max_val = self.range_specs[key]
                if hasattr(value, "min") and hasattr(value, "max"):
                    # Array values
                    if np.any(value < min_val) or np.any(value > max_val):
                        errors.append(
                            f"Values out of range for {key}: expected [{min_val}, {max_val}]"
                        )
                else:
                    # Scalar values
                    if value < min_val or value > max_val:
                        errors.append(
                            f"Value out of range for {key}: expected [{min_val}, {max_val}], got {value}"
                        )

        return errors

    def validate_file(self, file_path: Path) -> bool:
        """
        Validate a single NPZ file.

        Args:
            file_path: Path to NPZ file

        Returns:
            True if valid, False otherwise
        """
        try:
            if not file_path.exists():
                return False

            data = np.load(file_path)
            example = {key: data[key] for key in data.keys()}
            return self.validate_training_example(example)

        except Exception:
            return False

    def get_file_errors(self, file_path: Path) -> List[str]:
        """
        Get validation errors for a file.

        Args:
            file_path: Path to NPZ file

        Returns:
            List of error messages
        """
        try:
            if not file_path.exists():
                return [f"File not found: {file_path}"]

            data = np.load(file_path)
            example = {key: data[key] for key in data.keys()}
            return self.get_validation_errors(example)

        except Exception as e:
            return [f"Corrupted or invalid file: {e}"]

    def validate_batch_directory(self, directory: Path) -> Dict[str, Any]:
        """
        Validate all NPZ files in a directory.

        Args:
            directory: Directory containing NPZ files

        Returns:
            Validation report dictionary
        """
        npz_files = list(directory.glob("*.npz"))

        valid_count = 0
        invalid_count = 0
        all_errors = []

        for file_path in npz_files:
            if self.validate_file(file_path):
                valid_count += 1
            else:
                invalid_count += 1
                errors = self.get_file_errors(file_path)
                all_errors.extend([f"{file_path.name}: {error}" for error in errors])

        return {
            "total_files": len(npz_files),
            "valid_files": valid_count,
            "invalid_files": invalid_count,
            "errors": all_errors,
        }

    def get_valid_training_files(self, directory: Path) -> List[Path]:
        """
        Get list of valid training files in directory.

        Args:
            directory: Directory containing NPZ files

        Returns:
            List of valid file paths
        """
        npz_files = list(directory.glob("*.npz"))
        valid_files = []

        for file_path in npz_files:
            if self.validate_file(file_path):
                valid_files.append(file_path)

        return valid_files

    def generate_validation_report(self, directory: Path) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for directory.

        Args:
            directory: Directory containing NPZ files

        Returns:
            Detailed validation report
        """
        batch_report = self.validate_batch_directory(directory)

        # Analyze error patterns
        error_categories = self.categorize_errors(batch_report["errors"])

        # Generate file details
        file_details = []
        for file_path in directory.glob("*.npz"):
            is_valid = self.validate_file(file_path)
            errors = self.get_file_errors(file_path) if not is_valid else []

            file_details.append(
                {
                    "filename": file_path.name,
                    "valid": is_valid,
                    "errors": errors,
                }
            )

        return {
            "summary": {
                "total_files": batch_report["total_files"],
                "valid_files": batch_report["valid_files"],
                "invalid_files": batch_report["invalid_files"],
                "success_rate": batch_report["valid_files"] / max(1, batch_report["total_files"]),
            },
            "file_details": file_details,
            "common_errors": error_categories,
        }

    def categorize_errors(self, errors: List[str]) -> Dict[str, List[str]]:
        """
        Categorize errors by type.

        Args:
            errors: List of error messages        Returns:
            Dictionary of error categories
        """
        categories: Dict[str, List[str]] = {
            "shape_errors": [],
            "dtype_errors": [],
            "missing_key_errors": [],
            "range_errors": [],
            "other_errors": [],
        }

        for error in errors:
            error_lower = error.lower()
            if "shape" in error_lower:
                categories["shape_errors"].append(error)
            elif "dtype" in error_lower or "type" in error_lower:
                categories["dtype_errors"].append(error)
            elif "missing" in error_lower:
                categories["missing_key_errors"].append(error)
            elif "range" in error_lower:
                categories["range_errors"].append(error)
            else:
                categories["other_errors"].append(error)

        return categories

    def compute_validation_statistics(
        self, validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute validation statistics from results.

        Args:
            validation_results: List of validation result dictionaries

        Returns:
            Statistics dictionary
        """
        total = len(validation_results)
        valid = sum(1 for result in validation_results if result["valid"])
        invalid = total - valid

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "success_rate": valid / max(1, total),
        }
