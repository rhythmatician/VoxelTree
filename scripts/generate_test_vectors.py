#!/usr/bin/env python3
"""
Generate Test Vectors for DJL Parity Testing

Creates test_vectors.npz files for each model to validate PyTorch ↔ DJL parity.
These vectors contain sample inputs and expected outputs for ONNX model validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_test_vectors_for_model(
    model_config_path: Path,
    output_path: Path,
    num_samples: int = 5,
    seed: int = 42,
) -> None:
    """
    Generate test vectors for a single model.

    Args:
        model_config_path: Path to model_config.json
        output_path: Path to save test_vectors.npz
        num_samples: Number of test samples to generate
        seed: Random seed for reproducibility
    """
    # Load model config
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get input/output shapes
    input_shapes = config.get("inputs", {})
    output_shapes = config.get("outputs", {})

    # Generate test samples
    test_vectors: Dict[str, Any] = {}

    for sample_idx in range(num_samples):
        sample_inputs: Dict[str, Any] = {}
        sample_outputs: Dict[str, Any] = {}

        # Generate inputs according to config
        for input_name, shape in input_shapes.items():
            if input_name == "x_parent_prev":
                # Occupancy: random 0/1
                sample_inputs[input_name] = np.random.randint(0, 2, size=shape).astype(np.float32)
            elif input_name == "x_height_planes":
                # Heights: random in reasonable range (e.g., 64-200)
                sample_inputs[input_name] = np.random.uniform(64, 200, size=shape).astype(
                    np.float32
                )
            elif input_name == "x_biome_quart":
                # Biome features: random in [0, 1]
                sample_inputs[input_name] = np.random.uniform(0, 1, size=shape).astype(np.float32)
            elif input_name == "x_router6":
                # Router values: random in reasonable range
                sample_inputs[input_name] = np.random.uniform(-2, 2, size=shape).astype(np.float32)
            elif input_name == "x_chunk_pos":
                # Chunk coords: random integers
                sample_inputs[input_name] = np.random.randint(-100, 100, size=shape).astype(
                    np.float32
                )
            elif input_name == "x_lod":
                # LOD level: from config or reasonable default
                lod_value = (
                    config.get("description", "").split()[-1] if "description" in config else 0
                )
                if isinstance(lod_value, str) and lod_value.isdigit():
                    lod_value = int(lod_value)
                else:
                    lod_value = 0
                sample_inputs[input_name] = np.array([[lod_value]], dtype=np.float32)
            else:
                # Generic: random in [0, 1]
                sample_inputs[input_name] = np.random.uniform(0, 1, size=shape).astype(np.float32)

        # Generate placeholder outputs (will be replaced with actual model outputs)
        for output_name, shape in output_shapes.items():
            if output_name == "block_logits":
                # Block logits: random logits
                sample_outputs[output_name] = np.random.randn(*shape).astype(np.float32)
            elif output_name == "air_mask":
                # Air mask: random probabilities in [0, 1]
                sample_outputs[output_name] = np.random.uniform(0, 1, size=shape).astype(np.float32)
            else:
                # Generic: random
                sample_outputs[output_name] = np.random.randn(*shape).astype(np.float32)

        # Store sample
        test_vectors[f"sample_{sample_idx}_inputs"] = sample_inputs
        test_vectors[f"sample_{sample_idx}_outputs"] = sample_outputs

    # Save test vectors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **test_vectors)

    logger.info(f"Generated {num_samples} test vectors for {model_config_path.name}")
    logger.info(f"Saved to {output_path}")


def generate_all_test_vectors(
    schema_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
    seed: int = 42,
) -> None:
    """
    Generate test vectors for all 5 models.

    Args:
        schema_dir: Directory containing model_config.json files
        output_dir: Output directory for test vectors
        num_samples: Number of test samples per model
        seed: Random seed
    """
    schema_dir = Path(schema_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [
        ("model0initial.json", "model0"),
        ("model1lod4to3.json", "model1"),
        ("model2lod3to2.json", "model2"),
        ("model3lod2to1.json", "model3"),
        ("model4lod1to0.json", "model4"),
    ]

    for config_filename, model_name in model_names:
        config_path = schema_dir / config_filename
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, skipping")
            continue

        output_path = output_dir / f"{model_name}_test_vectors.npz"
        generate_test_vectors_for_model(
            model_config_path=config_path,
            output_path=output_path,
            num_samples=num_samples,
            seed=seed,
        )

    logger.info(f"Generated test vectors for all models in {output_dir}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate test vectors for DJL parity")
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("schema"),
        help="Directory with model_config.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("production"),
        help="Output directory for test vectors",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    generate_all_test_vectors(
        schema_dir=args.schema_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
