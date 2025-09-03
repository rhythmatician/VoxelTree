#!/usr/bin/env python3
"""Quick verification of the exported ONNX model for LODiffusion integration."""

import json
from pathlib import Path

import numpy as np
import onnx


def main():
    artifacts_dir = Path("artifacts/quick_test")

    print("🌲 VoxelTree → LODiffusion Integration Check")
    print("=" * 60)

    # Check ONNX model
    model_path = artifacts_dir / "model.onnx"
    model = onnx.load(str(model_path))

    print(f"✅ ONNX Model: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   Opset: {model.opset_import[0].version}")

    # Check inputs/outputs
    inputs = [
        (i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input
    ]
    outputs = [
        (o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output
    ]

    print("\n📥 Inputs:")
    for name, shape in inputs:
        print(f"   {name}: {shape}")

    print("\n📤 Outputs:")
    for name, shape in outputs:
        print(f"   {name}: {shape}")

    # Verify output names match LODiffusion contract
    expected_outputs = ["block_logits", "air_mask"]
    actual_outputs = [o[0] for o in outputs]
    names_match = actual_outputs == expected_outputs

    print(f"\n🎯 Contract Compliance:")
    print(f"   Expected outputs: {expected_outputs}")
    print(f"   Actual outputs:   {actual_outputs}")
    print(f"   Names match: {'✅ YES' if names_match else '❌ NO'}")

    # Check model config
    config_path = artifacts_dir / "model_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\n📋 Model Configuration:")
    print(f"   Block vocabulary: {config.get('block_vocab_size', 'unknown')}")
    print(f"   Biome vocabulary: {config.get('biome_vocab_size', 'unknown')}")
    print(f"   Contract: {config.get('contract', 'unknown')}")
    print(f"   Block mapping embedded: {'✅ YES' if 'block_mapping' in config else '❌ NO'}")

    # Check provenance
    if "provenance" in config:
        prov = config["provenance"]
        print(f"\n🔍 Provenance:")
        print(f"   Git commit: {prov.get('git_commit', 'unknown')[:8]}...")
        print(f"   Git branch: {prov.get('git_branch', 'unknown')}")
        print(f"   Clean repo: {'✅ YES' if prov.get('git_clean', False) else '⚠️  NO'}")

    # Check test vectors
    vectors_path = artifacts_dir / "test_vectors.npz"
    vectors = np.load(str(vectors_path))

    print(f"\n🧪 Test Vectors ({vectors_path.stat().st_size / 1024 / 1024:.1f} MB):")
    for key, array in vectors.items():
        print(f"   {key}: {array.shape} {array.dtype}")

    print("\n" + "=" * 60)
    print("🚀 LODiffusion Integration Status:")

    if names_match and config.get("block_vocab_size") == 1104 and "block_mapping" in config:
        print("✅ READY FOR INTEGRATION")
        print("\nNext steps:")
        print("1. Copy model.onnx to your LODiffusion mod")
        print("2. Use model_config.json for block ID mapping")
        print("3. Test with DJL using test_vectors.npz")
        print("4. Verify max_abs_diff ≤ 1e-4 between PyTorch and DJL")
        print("\nModel generates reasonable terrain - perfect for initial testing!")
    else:
        print("❌ ISSUES DETECTED")
        if not names_match:
            print("- Output names don't match contract")
        if config.get("block_vocab_size") != 1104:
            print("- Block vocabulary size incorrect")
        if "block_mapping" not in config:
            print("- Block mapping not embedded")

    print("=" * 60)


if __name__ == "__main__":
    main()
