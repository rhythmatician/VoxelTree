"""Tests that the GUI ModelTrack export/deploy runners invoke the correct scripts.

These tests primarily guard against regressions where the ModelTrack
"export"/"deploy" factories are stubs or not wired up.
"""

from __future__ import annotations

from pathlib import Path


def test_sparse_octree_export_and_deploy_call_export_sparse_octree(
    monkeypatch, tmp_path: Path
) -> None:
    """Export + deploy should call the LODiffusion sparse_octree exporter."""

    called: dict[str, tuple[Path, Path]] = {}

    def fake_export(checkpoint: Path, out_dir: Path) -> None:
        called["export"] = (checkpoint, out_dir)

    # LODiffusion is not installed into the test environment, so create a minimal
    # fake module in sys.modules so the import inside the runner succeeds.
    import sys
    import types

    lodiffusion_mod = types.ModuleType("LODiffusion")
    sys.modules["LODiffusion"] = lodiffusion_mod
    models_mod = types.ModuleType("LODiffusion.models")
    sys.modules["LODiffusion.models"] = models_mod
    export_mod = types.ModuleType("LODiffusion.models.export_sparse_octree")
    setattr(export_mod, "export_sparse_octree", fake_export)
    sys.modules["LODiffusion.models.export_sparse_octree"] = export_mod

    from voxel_tree.gui.step_definitions import (
        _deploy_sparse_octree_run,
        _export_sparse_octree_run,
    )

    profile = {
        "train": {"output_dir": str(tmp_path / "models")},
        "export": {"output_dir": str(tmp_path / "exported")},
        "deploy": {"target_dir": str(tmp_path / "deployed")},
    }

    _export_sparse_octree_run(profile)
    assert "export" in called
    checkpoint, out_dir = called["export"]
    assert checkpoint.name == "sparse_octree_model.pt"
    assert out_dir == tmp_path / "exported"

    # Reset and test deploy runner
    called.clear()
    _deploy_sparse_octree_run(profile)
    assert "export" in called
    checkpoint, out_dir = called["export"]
    assert checkpoint.name == "sparse_octree_model.pt"
    assert out_dir == tmp_path / "deployed"


def test_terrain_shaper_extract_and_deploy_calls_extract_density_weights(
    monkeypatch, tmp_path: Path
) -> None:
    """Export/deploy for Stage-1 should call the extract_density_weights script."""

    called: dict[str, list[list[str]]] = {}

    def fake_main(argv: list[str]) -> None:
        called.setdefault("args", []).append(list(argv))

    monkeypatch.setattr(
        "voxel_tree.tasks.density.extract_density_weights.main",
        fake_main,
    )

    from voxel_tree.gui.step_definitions import (
        _deploy_terrain_shaper_run,
        _extract_terrain_shaper_weights_run,
    )

    profile = {
        "train": {"output_dir": str(tmp_path / "terrain_shaper_model")},
        "extract": {"output_dir": str(tmp_path / "terrain_shaper_export")},
    }

    _extract_terrain_shaper_weights_run(profile)
    # should include --no-deploy and match our profile directories
    assert any("--no-deploy" in args for args in called.get("args", []))
    assert any(
        "--model-dir" in args and str(tmp_path / "terrain_shaper_model") in args
        for args in called.get("args", [])
    )
    assert any(
        "--out-dir" in args and str(tmp_path / "terrain_shaper_export") in args
        for args in called.get("args", [])
    )

    called.clear()
    _deploy_terrain_shaper_run(profile)
    assert any("--no-deploy" not in args for args in called.get("args", []))
    assert any(
        "--model-dir" in args and str(tmp_path / "terrain_shaper_model") in args
        for args in called.get("args", [])
    )
    assert any(
        "--out-dir" in args and str(tmp_path / "terrain_shaper_export") in args
        for args in called.get("args", [])
    )
