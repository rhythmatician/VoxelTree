#!/usr/bin/env python3
"""Ablation experiment runner for octree training pipeline.

Generates and optionally executes a set of training runs with different
hyperparameter configurations.  Each run gets its own output directory
under ``--base-output-dir``.

Usage (dry-run — print commands only)::

    python scripts/run_ablations.py --data-dir data/octree_pairs --dry-run

Usage (execute sequentially)::

    python scripts/run_ablations.py --data-dir data/octree_pairs

Usage (generate shell script)::

    python scripts/run_ablations.py --data-dir data/octree_pairs --shell > ablations.sh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationConfig:
    """One ablation experiment configuration."""

    name: str
    description: str
    extra_args: List[str] = field(default_factory=list)


# ── Ablation suite definition ────────────────────────────────────────


def build_ablation_suite() -> List[AblationConfig]:
    """Return the full suite of ablation experiments."""
    suite: List[AblationConfig] = []

    # ── 0. Baseline ──────────────────────────────────────────────────
    suite.append(
        AblationConfig(
            name="baseline",
            description="Default settings (parent_embed_dim=16, focal_gamma=2.0, occ_weight=1.0)",
            extra_args=[],
        )
    )

    # ── 1. Parent-embed-dim sweep ────────────────────────────────────
    for dim in [4, 8, 32]:
        suite.append(
            AblationConfig(
                name=f"parent_dim_{dim}",
                description=f"Parent embedding dimension = {dim}",
                extra_args=[f"--parent-embed-dim={dim}"],
            )
        )

    # ── 2. Parent-context ablation ───────────────────────────────────
    suite.append(
        AblationConfig(
            name="parent_zeros",
            description="Zero-filled parent channels (same architecture, no learned content)",
            extra_args=["--parent-context-mode=zeros"],
        )
    )
    suite.append(
        AblationConfig(
            name="parent_disabled",
            description="Remove parent channels entirely (smaller U-Net input)",
            extra_args=["--parent-context-mode=disabled"],
        )
    )

    # ── 3. Focal-gamma sweep ─────────────────────────────────────────
    for gamma in [0.0, 1.0, 3.0, 5.0]:
        suite.append(
            AblationConfig(
                name=f"focal_gamma_{gamma:.0f}",
                description=f"Focal loss gamma = {gamma} (0 = plain BCE)",
                extra_args=[f"--focal-gamma={gamma}"],
            )
        )

    # ── 4. Focal-alpha sweep ─────────────────────────────────────────
    for alpha in [0.25, 0.5, 0.9]:
        suite.append(
            AblationConfig(
                name=f"focal_alpha_{alpha:.2f}",
                description=f"Focal loss alpha = {alpha}",
                extra_args=[f"--focal-alpha={alpha}"],
            )
        )

    # ── 5. Occ-weight sweep ──────────────────────────────────────────
    for w in [0.1, 0.5, 2.0, 5.0]:
        suite.append(
            AblationConfig(
                name=f"occ_weight_{w:.1f}",
                description=f"Global occupancy weight = {w}",
                extra_args=[f"--occ-weight={w}"],
            )
        )

    # ── 6. Per-level occ weights ─────────────────────────────────────
    suite.append(
        AblationConfig(
            name="level_occ_top_heavy",
            description="Higher occ weight at coarser levels (L4=3.0, L3=2.0, L2=1.0, L1=0.5)",
            extra_args=["--level-occ-weights=4:3.0,3:2.0,2:1.0,1:0.5"],
        )
    )
    suite.append(
        AblationConfig(
            name="level_occ_bottom_heavy",
            description="Higher occ weight at finer levels (L4=0.5, L3=1.0, L2=2.0, L1=3.0)",
            extra_args=["--level-occ-weights=4:0.5,3:1.0,2:2.0,1:3.0"],
        )
    )

    # ── 7. Occ warmup ───────────────────────────────────────────────
    for warmup in [5, 10, 20]:
        suite.append(
            AblationConfig(
                name=f"occ_warmup_{warmup}",
                description=f"Occupancy warmup for {warmup} epochs",
                extra_args=[f"--occ-warmup-epochs={warmup}"],
            )
        )

    # ── 8. Channel width experiments ─────────────────────────────────
    suite.append(
        AblationConfig(
            name="channels_narrow",
            description="Narrow channels (init=16,32,64  refine=16,32,64  leaf=24,48,96)",
            extra_args=[
                "--init-channels=16,32,64",
                "--refine-channels=16,32,64",
                "--leaf-channels=24,48,96",
            ],
        )
    )
    suite.append(
        AblationConfig(
            name="channels_wide",
            description="Wide channels (init=32,64,128  refine=48,96,192  leaf=64,128,256)",
            extra_args=[
                "--init-channels=32,64,128",
                "--refine-channels=48,96,192",
                "--leaf-channels=64,128,256",
            ],
        )
    )

    return suite


# ── CLI + execution ──────────────────────────────────────────────────


def build_command(
    data_dir: str,
    output_dir: str,
    config: AblationConfig,
    base_args: List[str],
) -> List[str]:
    """Build the full command for one ablation run."""
    cmd = [
        sys.executable,
        "train_octree.py",
        f"--data-dir={data_dir}",
        f"--output-dir={output_dir}",
        *base_args,
        *config.extra_args,
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for octree training")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="./ablation_runs",
        help="Base output directory (each run gets a subdirectory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per ablation run (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per run (for quick testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Output as shell script (to stdout)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run ablations whose name contains this substring",
    )
    args = parser.parse_args()

    suite = build_ablation_suite()

    # Filter
    if args.filter:
        suite = [c for c in suite if args.filter in c.name]
        if not suite:
            print(f"No ablations match filter '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    # Base args shared by all runs
    base_args: List[str] = [
        f"--epochs={args.epochs}",
        f"--batch-size={args.batch_size}",
        "--validate-every=5",
        "--save-every=10",
    ]
    if args.max_samples is not None:
        base_args.append(f"--max-samples={args.max_samples}")

    print(f"Ablation suite: {len(suite)} experiments\n")

    if args.shell:
        # Output as shell script
        print("#!/bin/bash")
        print("# Auto-generated ablation runner")
        print(f"# {len(suite)} experiments\n")
        for config in suite:
            out_dir = str(Path(args.base_output_dir) / config.name)
            cmd = build_command(args.data_dir, out_dir, config, base_args)
            print(f"# {config.description}")
            print(" ".join(cmd))
            print()
        return

    for i, config in enumerate(suite, 1):
        out_dir = str(Path(args.base_output_dir) / config.name)
        cmd = build_command(args.data_dir, out_dir, config, base_args)

        print(f"[{i}/{len(suite)}] {config.name}")
        print(f"  {config.description}")
        print(f"  CMD: {' '.join(cmd)}")

        if args.dry_run:
            print("  (dry-run, skipping)\n")
            continue

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})\n")
        else:
            print(f"  DONE\n")


if __name__ == "__main__":
    main()
