"""
PatchPairer - Phase 2.1 GREEN Implementation

Minimal implementation to make the RED tests pass.
Assembles parent-child LOD pairs from extracted chunk data.
"""

import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)


class PatchPairer:
    """
    Assembles parent-child LOD pairs from extracted chunk data.

    Takes 16x16x384 chunk data and creates parent-child pairs where:
    - Parent: downsampled 8x8x8 voxel representation
    - Child: full resolution 16x16x16 subchunk target
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize PatchPairer with configuration."""
        self.config_path = config_path if config_path else Path("config.yaml")
        self._load_config()

        logger.info(f"PatchPairer initialized with {self.lod_levels} LOD levels")

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # Extract pairing configuration
        pairing_config = config.get("pairing", {})
        self.extracted_data_dir = Path(pairing_config.get("extracted_data_dir", "data/chunks"))
        self.seed_inputs_dir = Path(pairing_config.get("seed_inputs_dir", "data/seed_inputs"))
        self.output_dir = Path(pairing_config.get("output_dir", "data/pairs"))
        self.lod_levels = pairing_config.get("lod_levels", 4)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_parent_child_pairs(
        self, chunk_file: Path, lod_level: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract parent-child pairs from a single chunk file.

        Args:
            chunk_file: Path to .npz chunk file
            lod_level: LOD level for this extraction

        Returns:
            List of parent-child pair dictionaries
        """
        try:
            chunk_data = np.load(chunk_file)
        except Exception as e:
            logger.error(f"Failed to load chunk file {chunk_file}: {e}")
            raise

        # Validate required keys
        required_keys = [
            "block_types",
            "air_mask",
            "biomes",
            "heightmap",
            "chunk_x",
            "chunk_z",
        ]
        for key in required_keys:
            if key not in chunk_data:
                raise KeyError(f"Missing required key '{key}' in chunk file {chunk_file}")

        # Validate dimensions
        expected_shape = (16, 16, 384)
        if chunk_data["block_types"].shape != expected_shape:
            raise ValueError(
                f"Invalid block_types shape: {chunk_data['block_types'].shape}, expected {expected_shape}"
            )

        # Slice chunk into subchunks and create pairs
        subchunks = self.slice_chunk_into_subchunks(chunk_file)
        pairs = []

        for subchunk in subchunks:
            pair = self.create_training_pair(subchunk, lod_level)
            pairs.append(pair)

        return pairs

    def slice_chunk_into_subchunks(self, chunk_file: Path) -> List[Dict[str, Any]]:
        """
        Slice a chunk into 16x16x16 subchunks.

        Args:
            chunk_file: Path to .npz chunk file

        Returns:
            List of subchunk dictionaries
        """
        chunk_data = np.load(chunk_file)

        block_types = chunk_data["block_types"]  # (16, 16, 384)
        air_mask = chunk_data["air_mask"]  # (16, 16, 384)
        chunk_x = int(chunk_data["chunk_x"])
        chunk_z = int(chunk_data["chunk_z"])

        subchunks = []

        # Create 24 vertical slices (384 / 16 = 24)
        for y_index in range(24):
            y_start = y_index * 16
            y_end = y_start + 16

            subchunk = {
                "target_mask": air_mask[:, :, y_start:y_end],  # (16, 16, 16)
                "target_types": block_types[:, :, y_start:y_end],  # (16, 16, 16)
                "y_index": y_index,
                "chunk_x": chunk_x,
                "chunk_z": chunk_z,
            }
            subchunks.append(subchunk)

        return subchunks

    def downsample_to_parent(self, target_data: np.ndarray) -> np.ndarray:
        """
        Downsample 16x16x16 target data to 8x8x8 parent voxel using max pooling.

        Args:
            target_data: (16, 16, 16) array to downsample

        Returns:
            (8, 8, 8) downsampled array
        """
        # Reshape to enable 2x2x2 pooling
        reshaped = target_data.reshape(8, 2, 8, 2, 8, 2)

        # Use max pooling (most common value in each 2x2x2 block)
        if target_data.dtype == bool:
            # For boolean data, use logical OR (any True becomes True)
            pooled = np.any(reshaped, axis=(1, 3, 5))
        else:
            # For other data types, use actual max
            pooled = np.max(reshaped, axis=(1, 3, 5))

        return pooled

    def create_training_pair(self, subchunk: Dict[str, Any], lod_level: int) -> Dict[str, Any]:
        """
        Create a training pair from a subchunk.

        Args:
            subchunk: Subchunk data dictionary
            lod_level: LOD level for this pair

        Returns:
            Training pair dictionary
        """
        target_mask = subchunk["target_mask"]
        target_types = subchunk["target_types"]

        # Create parent voxel by downsampling the target
        parent_voxel = self.downsample_to_parent(target_mask)

        pair = {
            "parent_voxel": parent_voxel,
            "target_mask": target_mask,
            "target_types": target_types,
            "y_index": subchunk["y_index"],
            "chunk_x": subchunk["chunk_x"],
            "chunk_z": subchunk["chunk_z"],
            "lod": lod_level,
        }

        return pair

    def save_pair_npz(self, pair_data: Dict[str, Any], output_path: Path) -> Path:
        """
        Save a training pair to compressed NPZ format.

        Args:
            pair_data: Training pair dictionary
            output_path: Output file path

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with compression
        np.savez_compressed(output_path, **pair_data)

        return output_path

    def process_batch(self, chunks_dir: Path, output_dir: Path) -> int:
        """
        Process a batch of chunk files into training pairs.

        Args:
            chunks_dir: Directory containing chunk .npz files
            output_dir: Directory to save training pairs

        Returns:
            Number of pairs created
        """
        chunk_files = list(chunks_dir.glob("*.npz"))
        total_pairs = 0

        for chunk_file in chunk_files:
            pairs = self.extract_parent_child_pairs(chunk_file)

            for i, pair in enumerate(pairs):
                # Create unique filename for each pair
                filename = f"pair_{chunk_file.stem}_y{pair['y_index']}.npz"
                output_path = output_dir / filename

                self.save_pair_npz(pair, output_path)
                total_pairs += 1

        logger.info(f"Processed {len(chunk_files)} chunks into {total_pairs} training pairs")
        return total_pairs

    def process_batch_parallel(
        self, chunks_dir: Path, output_dir: Path, num_workers: Optional[int] = None
    ) -> int:
        """
        Process chunk files in parallel using multiprocessing.

        Args:
            chunks_dir: Directory containing chunk .npz files
            output_dir: Directory to save training pairs
            num_workers: Number of worker processes (default: CPU count)

        Returns:
            Number of pairs created
        """
        if num_workers is None:
            num_workers = min(cpu_count(), 4)  # Reasonable default

        list(chunks_dir.glob("*.npz"))

        # For this minimal implementation, we'll just call the sequential version
        # In a full implementation, we'd use multiprocessing.Pool
        return self.process_batch(chunks_dir, output_dir)


class LODValidator:
    """
    Validates LOD alignment and detects mismatches in parent-child pairs.
    """

    def validate_lod_alignment(
        self, parent: np.ndarray, target: np.ndarray, lod_level: int
    ) -> bool:
        """
        Validate that parent and target are properly aligned for the given LOD level.

        Args:
            parent: Parent voxel array
            target: Target voxel array
            lod_level: LOD level being validated

        Returns:
            True if alignment is valid, False otherwise
        """
        # For LOD level 1, parent should be 8x8x8 and target should be 16x16x16
        if lod_level == 1:
            expected_parent_shape = (8, 8, 8)
            expected_target_shape = (16, 16, 16)

            return parent.shape == expected_parent_shape and target.shape == expected_target_shape

        # For other LOD levels, implement additional validation logic as needed
        return True

    def detect_lod_mismatch(self, parent: np.ndarray, target: np.ndarray) -> bool:
        """
        Detect if there's a mismatch between parent and target dimensions.

        Args:
            parent: Parent voxel array
            target: Target voxel array

        Returns:
            True if mismatch detected, False if properly aligned
        """
        # Check if target is exactly 2x larger than parent in each dimension
        expected_target_shape = tuple(dim * 2 for dim in parent.shape)

        return target.shape != expected_target_shape
