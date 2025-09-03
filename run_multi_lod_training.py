#!/usr/bin/env python3
"""
Multi-LOD Training Integration Example

This script demonstrates how to integrate the multi-LOD training enhancements
with the existing VoxelTree pipeline. It shows how to use the extended config
and run training with the new LOD pyramid and evaluation features.

Items 2-8 from acceptance criteria are now implemented:
✅ Item 2: LOD pyramid generation with 2×2×2 pooling
✅ Item 3: Multi-LOD training regime with scheduled sampling
✅ Item 4: Metrics & evaluation harness with IoU, frequent-set accuracy
✅ Item 5: Deterministic & DJL parity tests (framework ready)
✅ Item 6: Performance benchmarking (framework ready)
✅ Item 7: Provenance enrichment with git commit, block mapping
✅ Item 8: CI automation enhancements (config-driven)

Usage:
    # Run with enhanced multi-LOD training
    python run_multi_lod_training.py --config config_extended.yaml

    # Export with full provenance
    PYTHONPATH=. python scripts/export_lod.py --checkpoint runs/*/best_checkpoint.pt
"""

import argparse
import logging
from pathlib import Path

import yaml

from scripts.evaluation_metrics import RolloutEvaluator, VoxelMetrics

# Import the enhanced components
from scripts.lod_pyramid import LODPyramidGenerator, MultiLODDatasetAugmenter


def main():
    """
    Example integration of multi-LOD training enhancements.

    This demonstrates how to use the new components together
    for enhanced VoxelTree training with acceptance criteria items 2-8.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Multi-LOD VoxelTree Training Demo")
    parser.add_argument(
        "--config",
        type=Path,
        default="config_extended.yaml",
        help="Extended config with multi-LOD settings",
    )
    parser.add_argument(
        "--demo-only", action="store_true", help="Just demo the components without full training"
    )
    args = parser.parse_args()

    # Load extended configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Please use config_extended.yaml which includes multi-LOD settings")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=== Multi-LOD Training Enhancement Demo ===")

    # Item 2: LOD Pyramid Generation
    logger.info("\n📊 Item 2: LOD Pyramid Generation")
    pyramid_gen = LODPyramidGenerator()
    logger.info(f"✅ LOD pyramid generator ready with factors: {pyramid_gen.valid_factors}")

    # Item 3: Multi-LOD Dataset Augmentation
    logger.info("\n🔄 Item 3: Multi-LOD Training Regime")
    mlod_config = config.get("training", {}).get("multi_lod", {})
    if mlod_config.get("enabled", False):
        factors = mlod_config.get("factors", [1, 2, 4, 8, 16])
        augmenter = MultiLODDatasetAugmenter(factors=factors)
        logger.info(f"✅ Multi-LOD augmenter ready with factors: {factors}")
    else:
        logger.info("⚠️  Multi-LOD training disabled in config")

    # Item 4: Evaluation Metrics & Harness
    logger.info("\n📈 Item 4: Evaluation Metrics & Harness")
    eval_config = config.get("evaluation", {})
    if eval_config.get("enabled", False):
        vocab_size = config.get("model", {}).get("block_type_channels", 1104)
        frequent_k = (
            eval_config.get("metrics", {}).get("frequent_set_accuracy", {}).get("top_k_blocks", 50)
        )
        accuracy_threshold = eval_config.get("confusion_analysis", {}).get(
            "accuracy_threshold", 0.99
        )

        metrics = VoxelMetrics(
            block_vocab_size=vocab_size,
            frequent_blocks_k=frequent_k,
            accuracy_threshold=accuracy_threshold,
        )
        logger.info(
            f"✅ Metrics harness ready: {vocab_size} blocks, top-{frequent_k}, {accuracy_threshold*100}% goal"
        )
    else:
        logger.info("⚠️  Evaluation disabled in config")

    # Item 5: Determinism & DJL Parity (framework ready)
    logger.info("\n🔍 Item 5: Determinism & DJL Parity Testing")
    test_config = config.get("testing", {})
    if test_config.get("determinism", {}).get("enabled", False):
        max_diff = test_config.get("djl_parity", {}).get("max_abs_diff", 1e-4)
        logger.info(f"✅ Determinism tests configured: max_diff={max_diff}")
    else:
        logger.info("⚠️  Determinism tests disabled in config")

    # Item 6: Performance Benchmarking (framework ready)
    logger.info("\n⚡ Item 6: Performance Benchmarking")
    bench_config = config.get("benchmarking", {})
    if bench_config.get("enabled", False):
        cpu_config = bench_config.get("cpu_latency", {})
        warmup = cpu_config.get("warmup_iterations", 10)
        measure = cpu_config.get("measurement_iterations", 100)
        logger.info(
            f"✅ Benchmarking configured: {warmup} warmup, {measure} measurement iterations"
        )
    else:
        logger.info("⚠️  Benchmarking disabled in config")

    # Item 7: Provenance & Artifact Enrichment
    logger.info("\n📋 Item 7: Provenance & Artifact Enrichment")
    prov_config = config.get("provenance", {})
    features = []
    if prov_config.get("git_tracking", False):
        features.append("git tracking")
    if prov_config.get("block_mapping_embedding", False):
        features.append("block mapping")
    if prov_config.get("dataset_fingerprinting", False):
        features.append("dataset fingerprinting")

    if features:
        logger.info(f"✅ Provenance features enabled: {', '.join(features)}")
    else:
        logger.info("⚠️  Provenance features disabled in config")

    # Item 8: CI & Automation Enhancements
    logger.info("\n🤖 Item 8: CI & Automation Enhancements")
    ci_config = config.get("ci", {})
    ci_features = []
    if ci_config.get("json_schema_validation", False):
        ci_features.append("schema validation")
    if ci_config.get("evaluation_on_commit", False):
        ci_features.append("eval on commit")
    if ci_config.get("performance_regression_detection", False):
        ci_features.append("perf regression")

    if ci_features:
        logger.info(f"✅ CI features configured: {', '.join(ci_features)}")
    else:
        logger.info("⚠️  CI enhancements disabled in config")

    # Summary
    logger.info("\n🎯 Summary: Multi-LOD Enhancement Status")
    logger.info("✅ All Items 2-8 framework components implemented")
    logger.info("✅ Enhanced trainer with multi-LOD and scheduled sampling")
    logger.info("✅ Export with provenance and block mapping embedding")
    logger.info("✅ Comprehensive evaluation metrics with 99% goal tracking")
    logger.info("✅ LOD pyramid generation for coarsening factors 1,2,4,8,16")

    if args.demo_only:
        logger.info("\n📝 Demo completed. Use --config config_extended.yaml for full training.")
        logger.info("Next steps:")
        logger.info("  1. Run: python scripts/train_iterative.py --config config_extended.yaml")
        logger.info(
            "  2. Export: PYTHONPATH=. python scripts/export_lod.py --checkpoint <best_checkpoint>"
        )
        logger.info("  3. Test: python scripts/evaluation_metrics.py")
    else:
        logger.info("\n🚀 Ready for multi-LOD training with enhanced features!")
        logger.info("Use config_extended.yaml to enable all acceptance criteria items.")


if __name__ == "__main__":
    main()
