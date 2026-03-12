"""Tests for the export_octree script's model filtering logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def fake_exporters(monkeypatch, tmp_path):
    """Replace the heavy ONNX export functions with stubs that touch files.

    The real export functions use torch and produce valid ONNX graphs, which we
    don't want to execute in a unit test.  By stubbing them we can verify that
    the CLI honors ``--models`` and correctly builds the manifest.
    """

    def _stub_export(model, config, out_dir):
        # simply create the expected file and return its Path
        name = f"octree_{model}.onnx" if isinstance(model, str) else "unknown.onnx"
        if "init" in name:
            fname = "octree_init.onnx"
        elif "refine" in name:
            fname = "octree_refine.onnx"
        elif "leaf" in name:
            fname = "octree_leaf.onnx"
        else:
            fname = "octree_generic.onnx"
        path = out_dir / fname
        path.write_text("dummy")
        return path

    # monkeypatch the three helpers in the export script
    import VoxelTree.scripts.export_octree as expmod

    monkeypatch.setattr(expmod, "_export_init", lambda m, c, o: _stub_export("init", c, o))
    monkeypatch.setattr(expmod, "_export_refine", lambda m, c, o: _stub_export("refine", c, o))
    monkeypatch.setattr(expmod, "_export_leaf", lambda m, c, o: _stub_export("leaf", c, o))

    # patch checkpoint loaders to return dummy config and models map
    # config needs attribute access; use a simple namespace
    from types import SimpleNamespace

    def _fake_load(ckpt):
        cfg = SimpleNamespace(biome_vocab_size=1, block_vocab_size=1, y_vocab_size=1)
        return cfg, {"init": None, "refine": None, "leaf": None}

    monkeypatch.setattr(expmod, "load_octree_checkpoint", _fake_load)
    monkeypatch.setattr(expmod, "load_octree_checkpoints_from_dir", lambda d: _fake_load(d))

    return tmp_path


def run_export(args: list[str], tmp_path: Path) -> Path:
    """Invoke the CLI and return the export directory path."""
    from VoxelTree.scripts import export_octree

    export_dir = tmp_path / "out"
    argv = ["--checkpoint", str(tmp_path / "dummy.pt"), "--out-dir", str(export_dir)] + args
    # create dummy checkpoint file
    (tmp_path / "dummy.pt").write_text("x")
    export_octree.main(argv)
    return export_dir


def test_export_filters_models(tmp_path: Path):
    # export only init and leaf
    out = run_export(["--models", "init", "leaf"], tmp_path)
    manifest = json.loads((out / "pipeline_manifest.json").read_text())
    names = manifest["required_files"]
    assert "octree_init.onnx" in names
    assert "octree_leaf.onnx" in names
    assert "octree_refine.onnx" not in names
    # pipeline entries count should match
    entry_models = [e["model"] for e in manifest["pipeline"]]
    assert "octree_refine" not in entry_models


def test_export_all_by_default(tmp_path: Path):
    out = run_export([], tmp_path)
    manifest = json.loads((out / "pipeline_manifest.json").read_text())
    assert set(manifest["required_files"]) >= {
        "octree_init.onnx",
        "octree_refine.onnx",
        "octree_leaf.onnx",
    }


def test_load_checkpoint_with_old_module(tmp_path: Path):
    """Ensure our compatibility shim allows loading pickles referencing
    ``train.unet3d``.

    We create a dummy object whose ``__module__`` is set to the old path and
    include it in the checkpoint.  ``torch.load`` will then attempt to import
    that module; the shim in ``export_octree`` should provide it.
    """
    import importlib

    import torch

    import VoxelTree.scripts.export_octree as expmod

    # reload module to undo the fake_exporters monkeypatch
    expmod = importlib.reload(expmod)
    load_octree_checkpoint = expmod.load_octree_checkpoint

    # define a fake class and instance that pretends to live in train.unet3d
    FakeLegacy = type("FakeLegacy", (object,), {})
    FakeLegacy.__module__ = "train.unet3d"
    # ensure the fake module exists in sys.modules and expose the class there
    import sys
    import types
    if "train.unet3d" not in sys.modules:
        sys.modules["train.unet3d"] = types.ModuleType("train.unet3d")
    setattr(sys.modules["train.unet3d"], "FakeLegacy", FakeLegacy)
    legacy_obj = FakeLegacy()

    # build a minimal checkpoint containing the fake object
    cfg = object()
    ckpt = {
        "config": cfg,
        "model_state_dicts": {"init": {}},
        "legacy": legacy_obj,
    }
    path = tmp_path / "old.ckpt"
    torch.save(ckpt, path)

    # loading should not raise ImportError
    loaded_cfg, models = load_octree_checkpoint(path)
    assert loaded_cfg is cfg
    # loading should not raise ImportError
    loaded_cfg, models = load_octree_checkpoint(path)
    assert loaded_cfg is cfg
