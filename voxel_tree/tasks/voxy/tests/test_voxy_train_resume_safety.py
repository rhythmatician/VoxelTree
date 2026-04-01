from __future__ import annotations

from pathlib import Path

import pytest

from voxel_tree.tasks.voxy.voxy_train import train_voxy_level


def test_train_voxy_level_fails_if_resume_checkpoint_missing(tmp_path: Path) -> None:
    missing_resume = tmp_path / "missing_resume.pt"

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        train_voxy_level(
            db_path=tmp_path / "does_not_matter.db",
            out_path=tmp_path / "voxy_L2.pt",
            level=2,
            epochs=1,
            resume_from=missing_resume,
            device="cpu",
            num_workers=0,
        )
