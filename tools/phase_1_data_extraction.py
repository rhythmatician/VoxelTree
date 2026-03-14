"""
Phase 1 Data Extraction Framework

Generates training data for Phase 1A/B/C networks by interfacing with a
Minecraft server to capture real noise router output during terrain generation.

Coordinates with Fabric server mod to extract:
- Phase 1A: Continents, Erosion, Ridges noise fields
- Phase 1B: Temperature, Humidity climate values
- Phase 1C: Cave carving, Aquifer carving probability fields
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


class Phase1DataExtractor:
    """Interface to Minecraft server for terrain noise factor extraction."""

    def __init__(self, server_host: str = "localhost", server_port: int = 25565):
        """Initialize connection to Minecraft server."""
        self.server_host = server_host
        self.server_port = server_port
        self.connected = False
        print(f"Phase1DataExtractor initialized (server={server_host}:{server_port})")

    def extract_macro_shape_data(
        self,
        num_chunks: int = 10000,
        seed: int = 12345,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Phase 1A macro-shape data: coordinates → noise values.

        Args:
            num_chunks: Number of chunks to sample
            seed: World seed

        Returns:
            (inputs, targets) where:
            - inputs: [N, 3] = (chunk_x, chunk_z, seed_id)
            - targets: [N, 3] = (continents, erosion, ridges)
        """
        print(f"\nPhase 1A: Extracting macro-shape data for {num_chunks} chunks...")
        print("Generating synthetic sampled coordinates...")

        # Synthetic data generation (placeholder until server integration)
        chunk_xs = np.random.randint(-1000, 1000, num_chunks)
        chunk_zs = np.random.randint(-1000, 1000, num_chunks)

        # Normalized coordinates
        norm_xs = np.clip(chunk_xs / 2000.0, -1, 1)
        norm_zs = np.clip(chunk_zs / 2000.0, -1, 1)

        # Simulate noise outputs (placeholder)
        continents = np.random.randn(num_chunks) * 0.5
        erosion = np.random.randn(num_chunks) * 0.3
        ridges = np.random.randn(num_chunks) * 0.4

        inputs = np.column_stack([norm_xs, norm_zs, np.full(num_chunks, seed)])
        targets = np.column_stack([continents, erosion, ridges])

        print(f"  Generated {num_chunks} samples")
        print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")

        return inputs, targets

    def extract_climate_data(
        self,
        num_chunks: int = 10000,
        seed: int = 12345,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Phase 1B climate data: coordinates → temperature/humidity.

        Args:
            num_chunks: Number of chunks to sample
            seed: World seed

        Returns:
            (inputs, targets) where:
            - inputs: [N, 3] = (chunk_x, chunk_z, seed_id)
            - targets: [N, 2] = (temperature, humidity)
        """
        print(f"\nPhase 1B: Extracting climate data for {num_chunks} chunks...")
        print("Generating synthetic sampled coordinates...")

        # Synthetic data
        chunk_xs = np.random.randint(-1000, 1000, num_chunks)
        chunk_zs = np.random.randint(-1000, 1000, num_chunks)

        norm_xs = np.clip(chunk_xs / 2000.0, -1, 1)
        norm_zs = np.clip(chunk_zs / 2000.0, -1, 1)

        # Simulate climate outputs
        temperature = np.random.uniform(0, 2, num_chunks)
        humidity = np.random.uniform(0, 1, num_chunks)

        inputs = np.column_stack([norm_xs, norm_zs, np.full(num_chunks, seed)])
        targets = np.column_stack([temperature, humidity])

        print(f"  Generated {num_chunks} samples")
        print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")

        return inputs, targets

    def extract_carving_data(
        self,
        num_chunks: int = 10000,
        seed: int = 12345,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Phase 1C carving/aquifer data: coordinates → carving probability.

        Args:
            num_chunks: Number of chunks to sample
            seed: World seed

        Returns:
            (inputs, targets) where:
            - inputs: [N, 3] = (chunk_x, chunk_z, seed_id)
            - targets: [N, 2] = (cave_carving, aquifer_carving)
        """
        print(f"\nPhase 1C: Extracting carving data for {num_chunks} chunks...")
        print("Generating synthetic sampled coordinates...")

        # Synthetic data
        chunk_xs = np.random.randint(-1000, 1000, num_chunks)
        chunk_zs = np.random.randint(-1000, 1000, num_chunks)

        norm_xs = np.clip(chunk_xs / 2000.0, -1, 1)
        norm_zs = np.clip(chunk_zs / 2000.0, -1, 1)

        # Simulate carving outputs
        cave_carving = np.random.uniform(0, 1, num_chunks)
        aquifer_carving = np.random.uniform(0, 1, num_chunks)

        inputs = np.column_stack([norm_xs, norm_zs, np.full(num_chunks, seed)])
        targets = np.column_stack([cave_carving, aquifer_carving])

        print(f"  Generated {num_chunks} samples")
        print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")

        return inputs, targets

    def save_dataset(self, inputs: np.ndarray, targets: np.ndarray, output_path: Union[str, Path]):
        """Save extracted data as PyTorch tensors."""
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        inputs_tensor = torch.from_numpy(inputs).float()
        targets_tensor = torch.from_numpy(targets).float()

        torch.save(
            {
                "inputs": inputs_tensor,
                "targets": targets_tensor,
                "num_samples": len(inputs),
            },
            save_path,
        )
        print(f"✅ Saved dataset to {save_path}")


def main():
    """Extract all Phase 1A/B/C datasets."""
    print("=" * 70)
    print("Phase 1 Data Extraction Framework")
    print("=" * 70)
    print()
    print("This script generates training data for Phase 1A/B/C networks.")
    print()
    print("Status: Framework ready, awaiting server integration")
    print()
    print("When Minecraft server mod is ready, this will:")
    print("  1. Extract Phase 1A macro-shape (continents/erosion/ridges)")
    print("  2. Extract Phase 1B climate (temperature/humidity)")
    print("  3. Extract Phase 1C carving (cave/aquifer probabilities)")
    print()

    # Placeholder demonstration
    extractor = Phase1DataExtractor()

    print("\nGenerating synthetic demo data...")
    print()

    # Phase 1A
    inputs_1a, targets_1a = extractor.extract_macro_shape_data(num_chunks=1000)

    # Phase 1B
    inputs_1b, targets_1b = extractor.extract_climate_data(num_chunks=1000)

    # Phase 1C
    inputs_1c, targets_1c = extractor.extract_carving_data(num_chunks=1000)

    print()
    print("Demo complete. Replace synthetic data with real server output when ready.")
    print()
    print("Next: Run phase_1_train_macro_shape.py with extracted data")


if __name__ == "__main__":
    main()
