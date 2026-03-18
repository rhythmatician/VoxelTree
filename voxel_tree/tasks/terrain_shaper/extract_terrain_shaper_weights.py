#!/usr/bin/env python3
"""
Extract weights from the trained TerrainShaperMLP model and save them in SSBO format.

This script loads the PyTorch checkpoint and exports the weights as a flat binary file
that can be directly loaded into an OpenGL SSBO for the terrain_compute.comp shader.

Post-extraction, the binary weights are automatically copied to the LODiffusion JAR
assets directory so they're available at runtime.
"""

import torch
import numpy as np
import shutil
from pathlib import Path


class TerrainShaperMLP(torch.nn.Module):
    """4-input, 3-output MLP for terrain shaper splines."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def extract_weights(checkpoint_path, output_path_bin, output_path_cpp):
    """
    Load the PyTorch checkpoint and extract weights into SSBO binary format and C++ header.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Recreate the model
    model = TerrainShaperMLP(hidden_size=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract weights and biases in the exact SSBO layout expected by the shader:
    #   [0..127]    = W1[32,4] row-major
    #   [128..159]  = b1[32]
    #   [160..1183] = W2[32,32] row-major
    #   [1184..1215] = b2[32]
    #   [1216..1248] = W3[3,32] row-major
    #   [1249..1251] = b3[3]

    weights_flat: list[float] = []

    # W1 [32,4] -> flatten to [128] in row-major order
    W1 = model.fc1.weight.data.cpu().numpy()  # Shape: (32, 4)
    print(f"W1 shape: {W1.shape}, flattened: {W1.size}")
    weights_flat.extend(W1.flatten())

    # b1 [32]
    b1 = model.fc1.bias.data.cpu().numpy()  # Shape: (32,)
    print(f"b1 shape: {b1.shape}")
    weights_flat.extend(b1)

    # W2 [32,32] -> flatten to [1024] in row-major order
    W2 = model.fc2.weight.data.cpu().numpy()  # Shape: (32, 32)
    print(f"W2 shape: {W2.shape}, flattened: {W2.size}")
    weights_flat.extend(W2.flatten())

    # b2 [32]
    b2 = model.fc2.bias.data.cpu().numpy()  # Shape: (32,)
    print(f"b2 shape: {b2.shape}")
    weights_flat.extend(b2)

    # W3 [3,32] -> flatten to [96] in row-major order
    W3 = model.fc3.weight.data.cpu().numpy()  # Shape: (3, 32)
    print(f"W3 shape: {W3.shape}, flattened: {W3.size}")
    weights_flat.extend(W3.flatten())

    # b3 [3]
    b3 = model.fc3.bias.data.cpu().numpy()  # Shape: (3,)
    print(f"b3 shape: {b3.shape}")
    weights_flat.extend(b3)

    weights_array = np.array(weights_flat, dtype=np.float32)
    print(f"\nTotal weight count: {len(weights_array)}")
    print("Expected: 128 + 32 + 1024 + 32 + 96 + 3 = 1315")

    # Verify offsets
    assert len(weights_array) == 1315, f"Weight count mismatch: {len(weights_array)} != 1315"

    # Save as binary file (float32 native byte order)
    print(f"\nSaving binary weights to {output_path_bin}...")
    weights_array.tofile(output_path_bin)

    # Save as C++ header file
    print(f"Saving C++ header to {output_path_cpp}...")
    with open(output_path_cpp, "w") as f:
        f.write("// Auto-generated TerrainShaperMLP weights\n")
        f.write("// Generated from terrain_shaper.pth\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <array>\n\n")
        f.write("namespace lodiffusion {\n\n")
        f.write("constexpr std::array<float, 1315> TERRAIN_SHAPER_MLP_WEIGHTS = {{\n")

        # Write weights in chunks of 16 per line for readability
        for i, w in enumerate(weights_array):
            if i % 16 == 0:
                f.write("    ")
            f.write(f"{w:.10f}f, " if i < len(weights_array) - 1 else f"{w:.10f}f\n")
            if (i + 1) % 16 == 0 and i < len(weights_array) - 1:
                f.write("\n")

        f.write("};\n\n")
        f.write("} // namespace lodiffusion\n")

    print("\nWeights extraction complete!")
    print(f"  Binary: {output_path_bin}")
    print(f"  C++ header: {output_path_cpp}")

    # Verify with a test forward pass
    print("\nVerifying model with test input...")
    test_input = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        output = model(test_input)
    print(f"Test input: {test_input.numpy()[0]}")
    print(f"Test output: {output.numpy()[0]}")

    return Path(output_path_bin)


def main():
    base_path = Path(__file__).parent / "model"
    base_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = base_path / "terrain_shaper.pth"
    output_bin = base_path / "terrain_shaper_weights.bin"
    output_cpp = base_path / "terrain_shaper_weights.h"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    weights_bin = extract_weights(str(checkpoint_path), str(output_bin), str(output_cpp))

    # Copy weights to LODiffusion assets for runtime loading
    lodiffusion_assets = (
        Path(__file__).parent.parent.parent
        / "LODiffusion"
        / "src"
        / "main"
        / "resources"
        / "assets"
        / "lodiffusion"
        / "models"
    )
    if lodiffusion_assets.exists():
        try:
            lodiffusion_assets.mkdir(parents=True, exist_ok=True)
            dest = lodiffusion_assets / "terrain_shaper_weights.bin"
            shutil.copy2(weights_bin, dest)
            print(f"\n✅ Copied weights to LODiffusion assets: {dest}")
        except Exception as e:
            print(f"\n⚠️  Could not copy to LODiffusion assets: {e}")
            print(f"   (You may need to manually copy {weights_bin} to {lodiffusion_assets})")
    else:
        print(f"\n⚠️  LODiffusion assets directory not found: {lodiffusion_assets}")
        print(f"   Manually copy {weights_bin} to the LODiffusion mod resources")


if __name__ == "__main__":
    main()
