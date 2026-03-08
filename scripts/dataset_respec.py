#!/usr/bin/env python3
"""
Dataset Respec Pipeline

Implements the dataset respec specification from docs/DATASET-RESPEC.md.
Transforms vanilla-generated Minecraft worlds into training data matching
the 5-model LOD hierarchy (Init, LOD4→3, LOD3→2, LOD2→1, LOD1→0).

This pipeline:
1. Reads vanilla .mca files or FeatureBundle caches
2. Generates full LOD pyramids (1³/2³/4³/8³/16³)
3. Creates training samples for all 5 models
4. Applies normalization according to model_config.json
5. Generates dataset manifest with provenance
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import mode

logger = logging.getLogger(__name__)


class LODPyramidBuilder:
    """
    Builds complete LOD pyramid from 16³ target blocks.

    Generates all LOD levels: 16³ → 8³ → 4³ → 2³ → 1³
    """

    def __init__(self, air_id: int = 0):
        """
        Initialize pyramid builder.

        Args:
            air_id: Block ID for air blocks
        """
        self.air_id = air_id

    def build_pyramid(self, target_blocks: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Build complete LOD pyramid from 16³ target.

        Args:
            target_blocks: [16, 16, 16] array of block IDs

        Returns:
            Dict mapping LOD level to {"blocks": [...], "occupancy": [...]}
        """
        if target_blocks.shape != (16, 16, 16):
            raise ValueError(f"Expected [16,16,16], got {target_blocks.shape}")

        pyramid = {}

        # LOD0: Full resolution (16×16×16)
        occupancy_16 = self._blocks_to_occupancy(target_blocks)
        pyramid[0] = {
            "blocks": target_blocks.copy(),
            "occupancy": occupancy_16,
            "size": 16,
        }

        # Generate progressively coarser LODs
        current_blocks = target_blocks
        current_occupancy = occupancy_16

        for lod_level in [1, 2, 3, 4]:
            factor = 2
            size = 16 // (2**lod_level)  # 8, 4, 2, 1

            # Downsample blocks (majority vote)
            current_blocks = self._downsample_blocks(current_blocks, factor)

            # Downsample occupancy (mean)
            current_occupancy = self._downsample_occupancy(current_occupancy, factor)

            pyramid[lod_level] = {
                "blocks": current_blocks.copy(),
                "occupancy": current_occupancy.copy(),
                "size": size,
            }

        return pyramid

    def _blocks_to_occupancy(self, blocks: np.ndarray) -> np.ndarray:
        """Convert block IDs to occupancy mask (0=air, 1=solid)."""
        return (blocks != self.air_id).astype(np.float32)

    def _downsample_blocks(self, blocks: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample blocks using majority vote in factor×factor×factor regions.

        Args:
            blocks: [D, D, D] block IDs
            factor: Downsampling factor (must divide D)

        Returns:
            [D//factor, D//factor, D//factor] downsampled blocks
        """
        d, h, w = blocks.shape
        assert d % factor == 0 and h % factor == 0 and w % factor == 0

        new_d, new_h, new_w = d // factor, h // factor, w // factor

        # Reshape into blocks of factor×factor×factor
        reshaped = blocks.reshape(new_d, factor, new_h, factor, new_w, factor)

        # Take mode (most common) along factor dimensions
        # Flatten factor dimensions, then find mode
        flattened = reshaped.transpose(0, 2, 4, 1, 3, 5).reshape(new_d, new_h, new_w, factor**3)

        # Use scipy.stats.mode for majority vote
        downsampled = np.zeros((new_d, new_h, new_w), dtype=blocks.dtype)
        for i in range(new_d):
            for j in range(new_h):
                for k in range(new_w):
                    mode_result = mode(flattened[i, j, k], keepdims=True)
                    downsampled[i, j, k] = (
                        mode_result.mode[0] if mode_result.count[0] > 0 else self.air_id
                    )

        return downsampled

    def _downsample_occupancy(self, occupancy: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample occupancy using mean in factor×factor×factor regions.

        Args:
            occupancy: [D, D, D] occupancy (0.0=air, 1.0=solid)
            factor: Downsampling factor

        Returns:
            [D//factor, D//factor, D//factor] downsampled occupancy
        """
        d, h, w = occupancy.shape
        assert d % factor == 0 and h % factor == 0 and w % factor == 0

        new_d, new_h, new_w = d // factor, h // factor, w // factor

        # Reshape and take mean
        reshaped = occupancy.reshape(new_d, factor, new_h, factor, new_w, factor)
        downsampled = reshaped.mean(axis=(1, 3, 5))

        return downsampled.astype(np.float32)


class ModelConfigLoader:
    """Loads and validates model_config.json files."""

    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        """Load model config from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_input_shapes(config: Dict[str, Any]) -> Dict[str, List[int]]:
        """Extract input shapes from model config."""
        shapes = {}
        for name, shape in config.get("inputs", {}).items():
            shapes[name] = shape
        return shapes

    @staticmethod
    def get_output_shapes(config: Dict[str, Any]) -> Dict[str, List[int]]:
        """Extract output shapes from model config."""
        shapes = {}
        for name, shape in config.get("outputs", {}).items():
            shapes[name] = shape
        return shapes


class InputNormalizer:
    """Applies normalization according to model_config.json."""

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize normalizer with model config."""
        self.config = model_config
        self.norm_config = model_config.get("normalization", {})

    def normalize(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize all inputs in sample according to config.

        Args:
            sample: Dict of input arrays

        Returns:
            Normalized sample
        """
        normalized = sample.copy()

        # Normalize height planes
        if "x_height_planes" in normalized and "heights" in self.norm_config:
            normalized["x_height_planes"] = self._normalize_heights(normalized["x_height_planes"])

        # Normalize router6
        if "x_router6" in normalized and "router6" in self.norm_config:
            normalized["x_router6"] = self._normalize_router6(normalized["x_router6"])

        # Normalize coords
        if "x_chunk_pos" in normalized and "coords" in self.norm_config:
            normalized["x_chunk_pos"] = self._normalize_coords(normalized["x_chunk_pos"])

        return normalized

    def _normalize_heights(self, heights: np.ndarray) -> np.ndarray:
        """Normalize heights using min-max by world limits."""
        norm_config = self.norm_config["heights"]
        bottom_y = norm_config["bottomY"]  # -64
        height = norm_config["height"]  # 384
        top_y = bottom_y + height  # 320

        # Clamp and normalize to [0, 1]
        heights_clamped = np.clip(heights, bottom_y, top_y)
        heights_norm = (heights_clamped - bottom_y) / height

        return heights_norm.astype(np.float32)

    def _normalize_router6(self, router6: np.ndarray) -> np.ndarray:
        """Normalize router6 using z-score."""
        norm_config = self.norm_config["router6"]
        mean = np.array(norm_config["mean"], dtype=np.float32)
        std = np.array(norm_config["std"], dtype=np.float32)

        # Apply z-score: (x - mean) / std
        router6_norm = (router6 - mean[None, :, None, None]) / std[None, :, None, None]

        return router6_norm.astype(np.float32)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coords using tanh scaling."""
        norm_config = self.norm_config["coords"]
        scale = norm_config["scale"]  # 1000.0

        # Apply tanh: tanh(x / scale)
        coords_norm = np.tanh(coords / scale)

        return coords_norm.astype(np.float32)


class DatasetRespecPipeline:
    """
    Main pipeline for dataset respec.

    Transforms vanilla worlds into training data for 5-model LOD hierarchy.
    """

    def __init__(
        self,
        schema_dir: Path,
        output_dir: Path,
        air_id: int = 0,
    ):
        """
        Initialize pipeline.

        Args:
            schema_dir: Directory containing model_config.json files
            output_dir: Output directory for processed samples
            air_id: Block ID for air
        """
        self.schema_dir = Path(schema_dir)
        self.output_dir = Path(output_dir)
        self.air_id = air_id

        self.pyramid_builder = LODPyramidBuilder(air_id=air_id)
        self.config_loader = ModelConfigLoader()

        # Load all 5 model configs
        self.model_configs = {}
        for model_idx in range(5):
            config_path = (
                self.schema_dir / f"model{model_idx}{self._get_model_suffix(model_idx)}.json"
            )
            if config_path.exists():
                self.model_configs[model_idx] = self.config_loader.load(config_path)
            else:
                logger.warning(f"Model config not found: {config_path}")

        # Create output directories
        for model_idx in range(5):
            (self.output_dir / f"model{model_idx}").mkdir(parents=True, exist_ok=True)

    def _get_model_suffix(self, model_idx: int) -> str:
        """Get filename suffix for model config."""
        suffixes = {
            0: "initial",
            1: "lod4to3",
            2: "lod3to2",
            3: "lod2to1",
            4: "lod1to0",
        }
        return suffixes.get(model_idx, f"model{model_idx}")

    def process_chunk(
        self,
        target_blocks: np.ndarray,
        feature_bundle: Optional[Dict[str, np.ndarray]] = None,
        chunk_x: int = 0,
        chunk_z: int = 0,
        y_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Process a single 16³ chunk into training samples for all 5 models.

        Args:
            target_blocks: [16, 16, 16] block IDs
            feature_bundle: Optional FeatureBundle dict with anchor channels
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate
            y_index: Y-level index

        Returns:
            List of training samples (one per model)
        """
        # Build LOD pyramid
        pyramid = self.pyramid_builder.build_pyramid(target_blocks)

        # Generate samples for each model
        samples = []

        # Model 0: Init (Noise → LOD4)
        samples.append(self._create_init_sample(pyramid, feature_bundle, chunk_x, chunk_z))

        # Model 1: LOD4 → LOD3
        samples.append(self._create_refinement_sample(1, pyramid, feature_bundle, chunk_x, chunk_z))

        # Model 2: LOD3 → LOD2
        samples.append(self._create_refinement_sample(2, pyramid, feature_bundle, chunk_x, chunk_z))

        # Model 3: LOD2 → LOD1
        samples.append(self._create_refinement_sample(3, pyramid, feature_bundle, chunk_x, chunk_z))

        # Model 4: LOD1 → LOD0
        samples.append(self._create_refinement_sample(4, pyramid, feature_bundle, chunk_x, chunk_z))

        return samples

    def _create_init_sample(
        self,
        pyramid: Dict[int, Dict[str, np.ndarray]],
        feature_bundle: Optional[Dict[str, np.ndarray]],
        chunk_x: int,
        chunk_z: int,
    ) -> Dict[str, Any]:
        """Create training sample for Init model (Noise → LOD4)."""
        model_config = self.model_configs.get(0, {})
        normalizer = InputNormalizer(model_config)

        # Parent is zeros for Init model
        x_parent_prev = np.zeros([1, 1, 1, 1, 1], dtype=np.float32)

        # Get LOD4 target
        lod4_data = pyramid[4]
        target_blocks = lod4_data["blocks"]  # [1, 1, 1]
        target_occupancy = lod4_data["occupancy"]  # [1, 1, 1]

        # Build inputs (use feature_bundle or generate placeholders)
        inputs = self._build_shared_inputs(feature_bundle, chunk_x, chunk_z, lod_level=4)

        inputs["x_parent_prev"] = x_parent_prev
        inputs["x_lod"] = np.array([[4]], dtype=np.float32)

        # Normalize inputs
        inputs = normalizer.normalize(inputs)

        # Build targets
        # Expand target_blocks to [1, N_blocks, 1, 1, 1] one-hot
        # For now, use simple encoding (will be expanded during training)
        target_block_logits = np.zeros([1, 1104, 1, 1, 1], dtype=np.float32)
        if target_blocks.size > 0:
            block_id = int(target_blocks.flat[0])
            if 0 <= block_id < 1104:
                target_block_logits[0, block_id, 0, 0, 0] = 1.0

        target_air_mask = target_occupancy.reshape(1, 1, 1, 1, 1)

        return {
            "model_idx": 0,
            "inputs": inputs,
            "targets": {
                "block_logits": target_block_logits,
                "air_mask": target_air_mask,
            },
            "metadata": {
                "chunk_x": chunk_x,
                "chunk_z": chunk_z,
                "lod_transition": "init_to_lod4",
            },
        }

    def _create_refinement_sample(
        self,
        model_idx: int,
        pyramid: Dict[int, Dict[str, np.ndarray]],
        feature_bundle: Optional[Dict[str, np.ndarray]],
        chunk_x: int,
        chunk_z: int,
    ) -> Dict[str, Any]:
        """Create training sample for refinement model (LOD N → LOD N-1)."""
        model_config = self.model_configs.get(model_idx, {})
        normalizer = InputNormalizer(model_config)

        # Map model_idx to LOD levels
        lod_mapping = {1: (4, 3), 2: (3, 2), 3: (2, 1), 4: (1, 0)}
        parent_lod, target_lod = lod_mapping[model_idx]

        # Get parent and target data
        parent_data = pyramid[parent_lod]
        target_data = pyramid[target_lod]

        # Parent occupancy: [1, 1, D, D, D] where D is parent size
        parent_size = parent_data["size"]
        x_parent_prev = parent_data["occupancy"].reshape(
            1, 1, parent_size, parent_size, parent_size
        )

        # Target
        target_blocks = target_data["blocks"]  # [D, D, D]
        target_occupancy = target_data["occupancy"]  # [D, D, D]
        target_size = target_data["size"]

        # Build inputs
        inputs = self._build_shared_inputs(feature_bundle, chunk_x, chunk_z, lod_level=target_lod)

        inputs["x_parent_prev"] = x_parent_prev.astype(np.float32)
        inputs["x_lod"] = np.array([[target_lod]], dtype=np.float32)

        # Normalize inputs
        inputs = normalizer.normalize(inputs)

        # Build targets
        # Expand target_blocks to [1, N_blocks, D, D, D] one-hot
        target_block_logits = np.zeros(
            [1, 1104, target_size, target_size, target_size], dtype=np.float32
        )
        for i in range(target_size):
            for j in range(target_size):
                for k in range(target_size):
                    block_id = int(target_blocks[i, j, k])
                    if 0 <= block_id < 1104:
                        target_block_logits[0, block_id, i, j, k] = 1.0

        target_air_mask = target_occupancy.reshape(1, 1, target_size, target_size, target_size)

        return {
            "model_idx": model_idx,
            "inputs": inputs,
            "targets": {
                "block_logits": target_block_logits,
                "air_mask": target_air_mask,
            },
            "metadata": {
                "chunk_x": chunk_x,
                "chunk_z": chunk_z,
                "lod_transition": f"lod{parent_lod}_to_lod{target_lod}",
            },
        }

    def _build_shared_inputs(
        self,
        feature_bundle: Optional[Dict[str, np.ndarray]],
        chunk_x: int,
        chunk_z: int,
        lod_level: int,
    ) -> Dict[str, np.ndarray]:
        """
        Build shared input channels (height_planes, biome_quart, router6, chunk_pos).

        If feature_bundle is provided, use it. Otherwise, generate placeholders.
        """
        inputs = {}

        if feature_bundle:
            # Use cached FeatureBundle
            inputs["x_height_planes"] = feature_bundle.get(
                "height_planes", self._placeholder_height_planes()
            )
            inputs["x_biome_quart"] = feature_bundle.get(
                "biome_quart", self._placeholder_biome_quart()
            )
            inputs["x_router6"] = feature_bundle.get("router6", self._placeholder_router6())
        else:
            # Generate placeholders (will be replaced during actual extraction)
            inputs["x_height_planes"] = self._placeholder_height_planes()
            inputs["x_biome_quart"] = self._placeholder_biome_quart()
            inputs["x_router6"] = self._placeholder_router6()

        inputs["x_chunk_pos"] = np.array([[chunk_x, chunk_z]], dtype=np.float32)

        return inputs

    def _placeholder_height_planes(self) -> np.ndarray:
        """Generate placeholder height planes [1, 5, 1, 16, 16]."""
        return np.zeros([1, 5, 1, 16, 16], dtype=np.float32)

    def _placeholder_biome_quart(self) -> np.ndarray:
        """Generate placeholder biome quart [1, 6, 4, 4, 4]."""
        return np.zeros([1, 6, 4, 4, 4], dtype=np.float32)

    def _placeholder_router6(self) -> np.ndarray:
        """Generate placeholder router6 [1, 6, 1, 16, 16]."""
        return np.zeros([1, 6, 1, 16, 16], dtype=np.float32)

    def save_sample(self, sample: Dict[str, Any], sample_idx: int) -> Path:
        """
        Save training sample to NPZ file.

        Args:
            sample: Training sample dict
            sample_idx: Sample index

        Returns:
            Path to saved file
        """
        model_idx = sample["model_idx"]
        output_path = self.output_dir / f"model{model_idx}" / f"sample_{sample_idx:06d}.npz"

        # Flatten nested dicts for NPZ
        save_dict = {}
        for key, value in sample["inputs"].items():
            save_dict[f"input_{key}"] = value
        for key, value in sample["targets"].items():
            save_dict[f"target_{key}"] = value
        for key, value in sample["metadata"].items():
            save_dict[f"meta_{key}"] = value

        np.savez_compressed(output_path, **save_dict)
        return output_path


def main():
    """Example usage of dataset respec pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Respec Pipeline")
    parser.add_argument(
        "--schema-dir", type=Path, default=Path("schema"), help="Model config directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="Output directory"
    )
    parser.add_argument("--chunks-dir", type=Path, help="Directory with extracted chunk NPZ files")
    args = parser.parse_args()

    # Initialize pipeline
    _pipeline = DatasetRespecPipeline(  # noqa: F841
        schema_dir=args.schema_dir,
        output_dir=args.output_dir,
    )

    logger.info("Dataset respec pipeline initialized")
    logger.info(f"Schema dir: {args.schema_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # TODO: Process chunks from chunks_dir
    # For now, this is a framework that can be extended


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
