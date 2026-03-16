#!/usr/bin/env python3
"""Distill a sparse-root student checkpoint from a teacher checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from VoxelTree.core.sparse_root_distill import distill_sparse_root
except ModuleNotFoundError:
    import sys

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    voxel_tree_pkg_root = script_path.parents[1]
    for p in (voxel_tree_pkg_root, repo_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from VoxelTree.core.sparse_root_distill import distill_sparse_root


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Distill a sparse-root student checkpoint from a teacher checkpoint."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to sparse_root_pairs.npz")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
        help="Teacher sparse-root checkpoint to distill from.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("distilled_sparse_root.pt"),
        help="Student checkpoint output path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults beside the output checkpoint.",
    )
    parser.add_argument(
        "--student-variant",
        type=str,
        default="fast",
        choices=("fast", "baseline"),
        help="Student model variant.",
    )
    parser.add_argument(
        "--student-hidden",
        type=int,
        default=80,
        help="Student hidden size.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Distillation epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=0.5,
        help="Weight for the real supervised sparse-root loss.",
    )
    parser.add_argument(
        "--split-distill-weight",
        type=float,
        default=0.15,
        help="Weight for teacher-guided split distillation.",
    )
    parser.add_argument(
        "--label-distill-weight",
        type=float,
        default=0.35,
        help="Weight for teacher-guided label distillation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Softmax temperature for distillation.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help="Label smoothing for the hard supervised term.",
    )
    args = parser.parse_args(argv)

    def _progress(epoch, total, metrics):
        print(
            f"[{epoch}/{total}] loss={metrics['loss']:.6f} "
            f"hard={metrics['hard_loss']:.6f} "
            f"split_kd={metrics['split_distill_loss']:.6f} "
            f"label_kd={metrics['label_distill_loss']:.6f} "
            f"leaf_acc={metrics['leaf_acc']:.6f}"
        )

    result = distill_sparse_root(
        data_path=args.data,
        teacher_checkpoint=args.teacher_checkpoint,
        out_path=args.out,
        summary_path=args.summary,
        student_variant=args.student_variant,
        student_hidden=args.student_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        hard_weight=args.hard_weight,
        split_distill_weight=args.split_distill_weight,
        label_distill_weight=args.label_distill_weight,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
        progress_callback=_progress,
    )

    print("Distillation complete")
    print(result)


if __name__ == "__main__":
    main()
