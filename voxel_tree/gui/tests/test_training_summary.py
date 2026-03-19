"""Unit tests for GUI training completion summaries."""

from __future__ import annotations

from voxel_tree.gui.step_definitions import StepDef
from voxel_tree.gui.training_summary import summarize_training_run


def _step(track: str) -> StepDef:
    return StepDef(
        id=f"train_{track}",
        label="Train",
        prereqs=[],
        run_fn=lambda _profile: None,
        track=track,
        phase="train",
    )


def test_biome_summary_formats_accuracy() -> None:
    summary = summarize_training_run(
        _step("biome_classifier"),
        [
            "Epoch 1/20",
            "  Training complete in 12.3s — best val_ce=1.2345 acc=0.678 @ epoch 7",
        ],
        profile_name="demo",
    )

    assert summary is not None
    assert summary["title"] == "Training complete — Biome Classifier"
    assert "Validation accuracy: 67.8%" in summary["text"]
    assert "Best epoch: 7" in summary["text"]


def test_heightmap_summary_includes_rmse() -> None:
    summary = summarize_training_run(
        _step("heightmap_predictor"),
        [
            "  Training complete in 33.1s — best val_mse=4.00 (rmse=2.00 blocks) @ epoch 12",
        ],
    )

    assert summary is not None
    assert "Validation RMSE: 2.00 blocks" in summary["text"]
    assert "Best validation MSE: 4.00" in summary["text"]


def test_sparse_octree_summary_uses_last_history_row() -> None:
    summary = summarize_training_run(
        _step("sparse_octree"),
        [
            "Training complete",
            "{'checkpoint': 'models/sparse_octree_model.pt', 'best_loss': 0.1234, 'history': [{'epoch': 1.0, 'loss': 0.5, 'split_f1': 0.4, 'leaf_acc': 0.5, 'leaf_node_ratio': 0.9}, {'epoch': 2.0, 'loss': 0.3, 'split_f1': 0.75, 'leaf_acc': 0.8, 'leaf_node_ratio': 1.1}], 'model_variant': 'fast', 'hidden': 80}",
        ],
    )

    assert summary is not None
    assert "Best loss: 0.1234" in summary["text"]
    assert "Latest split F1: 0.7500" in summary["text"]
    assert "Latest leaf accuracy: 80.0%" in summary["text"]
    assert "Final epoch: 2" in summary["text"]


def test_sparse_octree_summary_parses_step_result_marker() -> None:
    summary = summarize_training_run(
        _step("sparse_octree"),
        [
            "[PROGRESS] 50%",
            '[STEP_RESULT]{"best_loss": 0.2222, "checkpoint": "models/sparse.pt", "hidden": 80, "history": [{"epoch": 2.0, "leaf_acc": 0.91, "leaf_node_ratio": 1.03, "loss": 0.3, "split_f1": 0.88}], "model_variant": "fast"}',
        ],
    )

    assert summary is not None
    assert "Best loss: 0.2222" in summary["text"]
    assert "Latest split F1: 0.8800" in summary["text"]
    assert "Latest leaf accuracy: 91.0%" in summary["text"]


def test_octree_style_summary_parses_validation_metrics() -> None:
    summary = summarize_training_run(
        StepDef(
            id="train_legacy_octree",
            label="Train",
            prereqs=[],
            run_fn=lambda _profile: None,
            track="legacy_octree",
            phase="train",
        ),
        [
            "  Val   — Loss: 0.3210 (Block: 0.1234, Occ: 0.1976)",
            "  Val   — Acc: 0.875 (Air: 0.950, Block: 0.800)",
            "  Val   — Occ F1: 0.700  Recall: 0.800  FNR: 0.200  Recall@0.3: 0.820",
            "  ** New best model saved (val_loss: 0.3210)",
            "Training completed in 1.50 hours",
        ],
    )

    assert summary is not None
    assert "Best validation loss: 0.3210" in summary["text"]
    assert "Validation accuracy: 87.5%" in summary["text"]
    assert "Occupancy recall: 80.0%" in summary["text"]
    assert "Duration: 1.50 hours" in summary["text"]
