#!/usr/bin/env python3
"""
Runtime Performance Evaluation for Progressive LOD Models

This script measures:
- Inference time (forward pass)
- Memory usage during inference
- Model size (parameters and disk space)
- Throughput (samples per second)
"""

import gc
import os
import sys
import time
from pathlib import Path

import psutil
import torch

# Add train directory to path
sys.path.append(str(Path(__file__).parent / "train"))

from train.progressive_lod_models import (  # noqa: E402
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
)
from train.unet3d import SimpleFlexibleConfig  # noqa: E402


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage():
    """Get GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0


def create_test_inputs(batch_size=1, output_size=16):
    """Create test inputs for performance evaluation."""
    return {
        "x_height_planes": torch.randn(batch_size, 5, 1, 16, 16),
        "x_biome_quart": torch.randn(batch_size, 6, 4, 4, 4),
        "x_router6": torch.randn(batch_size, 6, 1, 16, 16),
        "x_chunk_pos": torch.randn(batch_size, 2),
        "x_lod": torch.tensor([[0]], dtype=torch.long),
        "x_parent_prev": torch.randn(
            batch_size, 1, output_size // 2, output_size // 2, output_size // 2
        ),
    }


def benchmark_model(
    model, model_name, device="cpu", num_warmup=10, num_iterations=100, batch_size=1
):
    """Benchmark a single model."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*50}")

    model = model.to(device)
    model.eval()

    # Model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    print("📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")

    # Determine output size from model name/type
    if "0_Initial" in model_name:
        output_size = 2
        has_parent = False
    elif "1_LOD4to3" in model_name:
        output_size = 4
        has_parent = True
    elif "2_LOD3to2" in model_name:
        output_size = 8
        has_parent = True
    elif "3_LOD2to1" in model_name:
        output_size = 16
        has_parent = True
    else:
        output_size = 16
        has_parent = True

    # Create inputs
    inputs = create_test_inputs(batch_size, output_size)

    # Move inputs to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    print("🔧 Test Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output size: {output_size}³")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iterations}")

    # Memory before inference
    memory_before = get_memory_usage()
    gpu_memory_before = get_gpu_memory_usage()

    # Warmup
    print("🔥 Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                if has_parent:
                    _ = model(
                        inputs["x_height_planes"],
                        inputs["x_biome_quart"],
                        inputs["x_router6"],
                        inputs["x_chunk_pos"],
                        inputs["x_lod"],
                        inputs["x_parent_prev"],
                    )
                else:
                    _ = model(
                        inputs["x_height_planes"],
                        inputs["x_biome_quart"],
                        inputs["x_router6"],
                        inputs["x_chunk_pos"],
                        inputs["x_lod"],
                    )
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                return None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print("⚡ Running benchmark...")
    times = []

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.perf_counter()

            try:
                if has_parent:
                    outputs = model(
                        inputs["x_height_planes"],
                        inputs["x_biome_quart"],
                        inputs["x_router6"],
                        inputs["x_chunk_pos"],
                        inputs["x_lod"],
                        inputs["x_parent_prev"],
                    )
                else:
                    outputs = model(
                        inputs["x_height_planes"],
                        inputs["x_biome_quart"],
                        inputs["x_router6"],
                        inputs["x_chunk_pos"],
                        inputs["x_lod"],
                    )
            except Exception as e:
                print(f"❌ Benchmark iteration {i} failed: {e}")
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Memory after inference
    memory_after = get_memory_usage()
    gpu_memory_after = get_gpu_memory_usage()

    # Calculate statistics
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        p95_time = sorted(times)[int(0.95 * len(times))]
        throughput = batch_size * 1000 / avg_time  # samples per second

        print("📈 Performance Results:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Min inference time: {min_time:.2f} ms")
        print(f"  Max inference time: {max_time:.2f} ms")
        print(f"  95th percentile: {p95_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")

        print("💾 Memory Usage:")
        print(f"  CPU memory increase: {memory_after - memory_before:.1f} MB")
        if torch.cuda.is_available():
            print(f"  GPU memory increase: {gpu_memory_after - gpu_memory_before:.1f} MB")

        # Output tensor sizes
        if outputs:
            print("📦 Output Tensors:")
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
                    print(f"  {key}: {list(tensor.shape)} ({size_mb:.2f} MB)")

        return {
            "model_name": model_name,
            "total_params": total_params,
            "model_size_mb": model_size_mb,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "p95_time_ms": p95_time,
            "throughput_sps": throughput,
            "memory_increase_mb": memory_after - memory_before,
            "gpu_memory_increase_mb": gpu_memory_after - gpu_memory_before,
            "output_size": output_size,
        }
    else:
        print("❌ No successful iterations recorded")
        return None


def load_model(model_path, device="cpu"):
    """Load a trained model from disk."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint.get("model_name", "Unknown")

        # Create appropriate model based on name
        config = SimpleFlexibleConfig()

        if "0_Initial" in model_name:
            model = ProgressiveLODModel0_Initial(config, output_size=2)
        elif "1_LOD4to3" in model_name:
            model = ProgressiveLODModel(config, output_size=4)
        elif "2_LOD3to2" in model_name:
            model = ProgressiveLODModel(config, output_size=8)
        elif "3_LOD2to1" in model_name:
            model = ProgressiveLODModel(config, output_size=16)
        else:
            print(f"❌ Unknown model type: {model_name}")
            return None, model_name

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded model: {model_name}")
        return model, model_name

    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None, "Unknown"


def main():
    """Main performance evaluation script."""
    print("🚀 Progressive LOD Models - Runtime Performance Evaluation")
    print("=" * 70)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return

    # Find all model files
    model_files = list(models_dir.glob("quick_*.pt"))
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        return

    print(f"📂 Found {len(model_files)} models to evaluate")

    results = []

    # Benchmark each model
    for model_file in sorted(model_files):
        print(f"\n🔄 Loading {model_file.name}...")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, model_name = load_model(model_file, device)
        if model is not None:
            result = benchmark_model(model, model_name, device)
            if result:
                results.append(result)

        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary report
    print(f"\n{'='*70}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*70}")

    if results:
        print(
            f"{'Model':<20} {'Params':<12} {'Size(MB)':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'Throughput':<12}"  # noqa: E501
        )
        print("-" * 70)

        total_params = 0
        total_size = 0

        for result in results:
            name = result["model_name"].replace("Model_", "").replace("_", "→")
            params = f"{result['total_params']/1e6:.1f}M"
            size = f"{result['model_size_mb']:.1f}"
            avg_time = f"{result['avg_time_ms']:.1f}"
            p95_time = f"{result['p95_time_ms']:.1f}"
            throughput = f"{result['throughput_sps']:.1f} sps"

            print(
                f"{name:<20} {params:<12} {size:<10} {avg_time:<10} {p95_time:<10} {throughput:<12}"
            )

            total_params += result["total_params"]
            total_size += result["model_size_mb"]

        print("-" * 70)
        print(f"{'TOTAL':<20} {total_params/1e6:.1f}M{'':<7} {total_size:.1f}{'':<6}")

        # Performance insights
        print("\n🎯 Performance Insights:")
        fastest = min(results, key=lambda x: x["avg_time_ms"])
        slowest = max(results, key=lambda x: x["avg_time_ms"])
        print(f"  Fastest: {fastest['model_name']} ({fastest['avg_time_ms']:.1f} ms)")
        print(f"  Slowest: {slowest['model_name']} ({slowest['avg_time_ms']:.1f} ms)")

        largest = max(results, key=lambda x: x["total_params"])
        smallest = min(results, key=lambda x: x["total_params"])
        print(f"  Largest: {largest['model_name']} ({largest['total_params']/1e6:.1f}M params)")
        print(f"  Smallest: {smallest['model_name']} ({smallest['total_params']/1e6:.1f}M params)")

        # LODiffusion integration readiness
        print("\n🎮 LODiffusion Integration Readiness:")
        for result in results:
            avg_time = result["avg_time_ms"]
            p95_time = result["p95_time_ms"]

            # Target: ≤100ms median, ≤150ms p95
            median_ok = avg_time <= 100
            p95_ok = p95_time <= 150

            status = (
                "✅ READY"
                if median_ok and p95_ok
                else "⚠️ SLOW" if avg_time <= 200 else "❌ TOO_SLOW"
            )
            print(
                f"  {result['model_name']}: {status} (avg: {avg_time:.1f}ms, p95: {p95_time:.1f}ms)"
            )

    else:
        print("❌ No successful benchmarks completed")


if __name__ == "__main__":
    main()
