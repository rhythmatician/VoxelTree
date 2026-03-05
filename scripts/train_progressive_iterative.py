#!/usr/bin/env python3
"""
Iterative training orchestrator for Progressive LOD models (region-by-region).

Pipeline per iteration (one region per batch):
  1) Worldgen (vanilla terrain; structures=false) for exactly one region (32x32 chunks)
  2) Extract chunks to NPZ
  3) Pair/link into training examples (linked inputs)
  4) Train all five progressive models (0→4) for a few epochs on this batch
This aligns with Phase‑1 disk hygiene: keep only what’s needed per iteration.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "train"))

# Direct imports to avoid legacy trainer module
from scripts.extraction.chunk_extractor import ChunkExtractor  # noqa: E402
from scripts.pairing.patch_pairer import PatchPairer  # noqa: E402
from scripts.pairing.seed_input_linker import SeedInputLinker  # noqa: E402
from scripts.seed_inputs.generator import SeedInputGenerator  # noqa: E402
from scripts.worldgen.bootstrap import WorldGenBootstrap  # noqa: E402
from train.multi_lod_dataset import (  # noqa: E402
    MultiLODDataset,
    collate_multi_lod_batch,
)
from train.progressive_lod_models import (  # noqa: E402
    ProgressiveLODModel,
    ProgressiveLODModel0_Initial,
)
from train.unet3d import SimpleFlexibleConfig  # noqa: E402

# Reuse training loop from quick progressive trainer
from train_progressive_quick import train_model as train_progressive_batch  # noqa: E402


def get_current_seed_and_region(base_seed: int, iteration: int, regions_per_seed: int = 16):
    """Map iteration index to (seed, region_x, region_z)."""
    grid_size = int(regions_per_seed**0.5)
    seed_index = iteration // regions_per_seed
    region_index = iteration % regions_per_seed
    seed = base_seed + seed_index
    x = region_index % grid_size - grid_size // 2
    z = region_index // grid_size - grid_size // 2
    return seed, x, z


def generate_single_region_batch(
    config: dict,
    seed: int,
    region_x: int,
    region_z: int,
    temp_dir: Path,
) -> Path:
    """Worldgen one region (32x32 chunks), extract to NPZ, return batch dir."""
    logger = logging.getLogger(__name__)
    batch_dir = temp_dir / f"batch_s{seed}_r{region_x}_{region_z}_{int(time.time())}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        bootstrap = WorldGenBootstrap(
            seed=str(seed),
            java_heap=config.get("worldgen", {}).get("java_heap", "4G"),
            defer_cleanup=True,
        )
        # Generate exactly the requested region using Chunky's rectangle selection
        logger.info(f"Generating exact region r.{region_x}.{region_z}")
        world_dir = bootstrap.generate_exact_region(region_x, region_z)
        if world_dir is None:
            raise RuntimeError("World generation failed: no world_dir returned")

        chunk_output_dir = batch_dir / "chunks"
        chunk_output_dir.mkdir(exist_ok=True)

        # Minimal temp extraction config
        temp_cfg_path = batch_dir / "temp_extract.yaml"
        extraction_cfg = {
            "extraction": {
                "output_dir": str(chunk_output_dir),
                "num_workers": 4,
                "batch_size": 64,
            }
        }
        with open(temp_cfg_path, "w") as f:
            yaml.safe_dump(extraction_cfg, f)

        extractor = ChunkExtractor(temp_cfg_path)
        region_dir = world_dir / "region"
        if not region_dir.exists():
            raise RuntimeError(f"Region directory not found: {region_dir}")

        # Strict one-region policy: require the exact region file and process only it
        expected_mca = region_dir / f"r.{region_x}.{region_z}.mca"
        if not expected_mca.exists():
            raise RuntimeError(
                (
                    f"Expected region file not generated: {expected_mca} "
                    f"(seed={seed} rx={region_x} rz={region_z})"
                )
            )

        # Extract exactly this region
        extractor.extract_region_batch(expected_mca)

        # Verify completeness: 32x32 = 1024 chunk npz outputs for this region
        region_id = expected_mca.stem  # e.g., r.-1.0
        produced = list(chunk_output_dir.glob(f"chunk_{region_id}_*.npz"))
        if len(produced) != 1024:
            raise RuntimeError(
                f"Incomplete extraction for {region_id}: {len(produced)}/1024 chunks present"
            )

        # Clean up temp config and world dir immediately
        try:
            if temp_cfg_path.exists():
                temp_cfg_path.unlink()
        finally:
            if world_dir.exists():
                shutil.rmtree(world_dir)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Region generation failed (seed={seed} rx={region_x} rz={region_z}): {e}")
        return batch_dir

    return batch_dir


def _generate_missing_seed_inputs(pairs_dir: Path, seed_inputs_dir: Path, seed: int) -> int:
    """Generate seed input NPZ files for all chunk_x/z pairs referenced in pairs_dir.

        Writes files named seed_patch_{chunk_x}_{chunk_z}.npz containing keys:
            - biomes (16,16) uint8
            - heightmap (16,16) uint16
            - chunk_x (scalar)
            - chunk_z (scalar)
    Returns the number of files generated (skips existing).
    """
    generated = 0
    gen = SeedInputGenerator(seed=seed)

    pairs = list(pairs_dir.glob("*.npz"))
    if not pairs:
        return 0

    # Collect unique chunk coords from pair metadata
    # FIXME: This is extremely inefficient. We're looping through all pairs rather than the chunks
    coords = set()
    for pf in tqdm(pairs, desc="Scanning pairs for seed inputs", unit="pair"):
        try:
            with np.load(pf) as data:  # type: ignore[name-defined]
                cx = int(data["chunk_x"]) if "chunk_x" in data else None
                cz = int(data["chunk_z"]) if "chunk_z" in data else None
            if cx is not None and cz is not None:
                coords.add((cx, cz))
        except Exception:
            # Fallback: try parse from filename like 'pair_chunk_r.-1.0_21_7_y13.npz'
            name = pf.stem
            try:
                parts = name.split("_")
                # last chunk coords appear before '_y*'
                # e.g., ['pair', 'chunk', 'r.-1.0', '21', '7', 'y13']
                cx = int(parts[-3])
                cz = int(parts[-2])
                coords.add((cx, cz))
            except Exception:
                continue

    seed_inputs_dir.mkdir(parents=True, exist_ok=True)

    # Generate missing files (Slow ~ 3s per patch)
    for cx, cz in tqdm(sorted(coords), desc="Generating seed input patches", unit="patch"):
        out_path = seed_inputs_dir / f"seed_patch_{cx}_{cz}.npz"
        if out_path.exists():
            continue
        # Convert chunk coords to world-block coords for a 16x16 patch
        wx, wz = cx * 16, cz * 16
        # No fallback: require real seed-derived inputs
        patch = gen.get_patch(wx, wz, 16)
        # Validate and cast to expected types/shapes for the linker
        biomes = np.asarray(patch["biomes"], dtype=np.int64)
        heightmap = np.asarray(patch["heightmap"], dtype=np.uint16)

        # Shape checks (16x16)
        if biomes.shape != (16, 16) or heightmap.shape != (16, 16):
            raise ValueError(
                f"Seed patch shape error @({cx},{cz}): "
                f"biomes={biomes.shape} heightmap={heightmap.shape}"
            )

        # Map keys to expected names and include coords
        save_dict = {
            "biomes": biomes,  # int64
            "heightmap": heightmap,  # uint16 absolute surface heights
            "chunk_x": np.int32(cx),
            "chunk_z": np.int32(cz),
        }

        np.savez_compressed(out_path, **save_dict)
        generated += 1

    return generated


def create_training_pairs(config: dict, chunk_dir: Path, temp_dir: Path, seed: int) -> Path:
    """Pair and link extracted chunks into training examples; return linked_dir.

    Also generates any missing seed input patches required for the pairs.
    """
    logger = logging.getLogger(__name__)
    pairs_dir = temp_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    pairer = PatchPairer()
    n_pairs = pairer.process_batch(chunk_dir, pairs_dir)
    logger.info(f"Created {n_pairs} training pairs from {chunk_dir}")

    linked_dir = temp_dir / "linked"
    linked_dir.mkdir(exist_ok=True)
    seed_inputs_dir = Path("data/seed_inputs")
    seed_inputs_dir.mkdir(parents=True, exist_ok=True)

    # Generate seed inputs as needed for this batch
    try:
        made = _generate_missing_seed_inputs(pairs_dir, seed_inputs_dir, seed)
        if made:
            logger.info(f"Generated {made} seed input patches in {seed_inputs_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate seed inputs: {e}")

    linker = SeedInputLinker()
    n_linked = linker.process_batch_linking(pairs_dir, seed_inputs_dir, linked_dir)
    logger.info(f"Linked {n_linked} examples into {linked_dir}")
    return linked_dir


def cleanup_batch(path: Path) -> None:
    """Delete a batch directory if it exists."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to cleanup {path}: {e}")


def build_dataloaders(data_dir: Path, batch_size: int):
    from torch.utils.data import DataLoader

    dataset = MultiLODDataset(data_dir=data_dir, split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found in {data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multi_lod_batch,
        num_workers=0,
    )
    return loader, loader


def load_or_init_models(config: SimpleFlexibleConfig):
    models = [
        (
            "Model_0_Initial",
            ProgressiveLODModel0_Initial(config, output_size=1),
            Path("models/quick_model_0_initial.pt"),
        ),
        (
            "Model_1_LOD4to3",
            ProgressiveLODModel(config, output_size=2),
            Path("models/quick_model_1_lod4to3.pt"),
        ),
        (
            "Model_2_LOD3to2",
            ProgressiveLODModel(config, output_size=4),
            Path("models/quick_model_2_lod3to2.pt"),
        ),
        (
            "Model_3_LOD2to1",
            ProgressiveLODModel(config, output_size=8),
            Path("models/quick_model_3_lod2to1.pt"),
        ),
        (
            "Model_4_LOD1to0",
            ProgressiveLODModel(config, output_size=16),
            Path("models/quick_model_4_lod1to0.pt"),
        ),
    ]

    loaded = []
    for name, model, ckpt_path in models:
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state = ckpt.get("model_state_dict")
                if state:
                    model.load_state_dict(state, strict=True)
                print(f"[resume] Loaded {name} from {ckpt_path}")
            except Exception as e:
                print(f"[warn] Failed to load {ckpt_path}: {e} — training from scratch")
        loaded.append((name, model, ckpt_path))
    return loaded


def save_model_checkpoint(model, model_name: str, output_path: Path, save_config: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": save_config,
            "model_name": model_name,
        },
        output_path,
    )
    print(f"  [save] {model_name} -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Iterative training for progressive LOD models")
    ap.add_argument("--config", default="config.yaml", help="Global pipeline config (YAML)")
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--regions-per-seed", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1, help="Epochs per iteration per model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    print(f"Using device: {device}")

    run_root = Path("runs") / f"progressive_iter_{int(time.time())}"
    temp_dir = run_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = SimpleFlexibleConfig()
    models = load_or_init_models(model_cfg)

    # Training hyperparams for the reused train loop
    train_hparams = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "air_loss_weight": 0.25,
        "quick_epochs": args.epochs,
        "max_batches_per_epoch": 10_000,  # effectively unlimited per iteration
        "batch_size": args.batch_size,
    }

    for it in range(args.max_iterations):
        print(f"\n=== Iteration {it+1}/{args.max_iterations} ===")
        # Pick seed+region
        seed, rx, rz = get_current_seed_and_region(args.base_seed, it, args.regions_per_seed)
        print(f"Seed={seed} Region=({rx},{rz})")

        # 1) Worldgen + extraction (single-region batch)
        batch_dir = generate_single_region_batch(
            config={
                "worldgen": {"java_heap": "4G"},
                "extraction": {"num_workers": 4, "batch_size": 64},
            },
            seed=seed,
            region_x=rx,
            region_z=rz,
            temp_dir=temp_dir,
        )
        chunks_dir = batch_dir / "chunks"

        # 2) Pair/link to build training inputs
        linked_dir = create_training_pairs(
            config={}, chunk_dir=chunks_dir, temp_dir=temp_dir, seed=seed
        )

        # 3) Dataloaders
        train_loader, val_loader = build_dataloaders(linked_dir, args.batch_size)

        # 4) Train each model for the requested epochs and save
        for model_name, model, ckpt_path in models:
            try:
                trained = train_progressive_batch(
                    model, train_loader, val_loader, train_hparams, model_name, device
                )
                save_model_checkpoint(trained, model_name, ckpt_path, train_hparams)
            except Exception as e:
                print(f"[fail] {model_name} iteration training failed: {e}")

        # 5) Cleanup batch data to free disk
        cleanup_batch(batch_dir)
        cleanup_batch(linked_dir)

    print("\nAll iterations complete.")


if __name__ == "__main__":
    main()
