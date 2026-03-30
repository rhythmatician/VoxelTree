"""continue_training_dialog.py — Dialog for continuing training from an existing checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _load_checkpoint_info(path: Path) -> dict[str, Any] | None:
    """Load a .pt checkpoint and extract summary info. Returns None if not found."""
    if not path.exists():
        return None
    try:
        import torch  # noqa: PLC0415

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        history: list[dict[str, Any]] = ckpt.get("history", [])
        epochs_trained: int = ckpt.get("epoch", len(history))
        best_loss: float = ckpt.get("best_loss", float("nan"))

        # Average block_acc over the last min(5, n) epochs
        recent = history[-5:] if len(history) >= 5 else history
        avg_acc = sum(r.get("block_acc", 0.0) for r in recent) / len(recent) if recent else float("nan")

        # Average elapsed_seconds → minutes/epoch
        times = [r["elapsed_seconds"] for r in recent if "elapsed_seconds" in r]
        avg_min = (sum(times) / len(times) / 60.0) if times else None

        return {
            "epochs_trained": epochs_trained,
            "best_loss": best_loss,
            "avg_acc": avg_acc,
            "avg_min_per_epoch": avg_min,
        }
    except Exception:  # noqa: BLE001
        return None


class ContinueTrainingDialog(QDialog):
    """Dialog shown when the user right-clicks a train node and chooses
    "Continue training...".

    Displays per-level checkpoint statistics and lets the user choose which
    levels to continue and how many additional epochs to run.
    """

    def __init__(self, profile_dict: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Continue Training")
        self.setMinimumWidth(560)

        train_cfg = profile_dict.get("train", {})
        self._out_dir = Path(train_cfg.get("output_dir", "."))

        # Load checkpoint info for all 5 levels
        self._infos: list[dict[str, Any] | None] = [
            _load_checkpoint_info(self._out_dir / f"voxy_L{lv}.pt") for lv in range(5)
        ]

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Header ──
        hdr = QLabel(f"<b>Checkpoint directory:</b> {self._out_dir}")
        hdr.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        # ── Table ──
        self._table = QTableWidget(5, 5)
        self._table.setHorizontalHeaderLabels(
            ["Level", "Epochs Trained", "Best Loss", "Avg Acc (last 5)", "Avg min/epoch"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background: #1e1e1e; color: #cccccc; gridline-color: #333; "
            "alternate-background-color: #252525; }"
            "QHeaderView::section { background: #2a2a2a; color: #aaaaaa; "
            "border: 1px solid #333; padding: 4px; }"
            "QTableWidget::item:selected { background: #2a4a6e; color: #cce0ff; }"
            "QTableWidget::item:disabled { color: #555555; }"
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 5):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        self._selectable_levels: list[int] = []
        for lv in range(5):
            info = self._infos[lv]
            has_ckpt = info is not None

            lv_item = QTableWidgetItem(f"L{lv}")
            lv_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if has_ckpt:
                epochs_str = str(info["epochs_trained"])  # type: ignore[index]
                loss_str = f"{info['best_loss']:.4f}"  # type: ignore[index]
                acc_str = f"{info['avg_acc']:.3f}"  # type: ignore[index]
                min_val = info.get("avg_min_per_epoch")  # type: ignore[index]
                time_str = f"{min_val:.1f}" if min_val is not None else "—"
                self._selectable_levels.append(lv)
            else:
                epochs_str = "—"
                loss_str = "—"
                acc_str = "—"
                time_str = "—"

            cols = [lv_item, QTableWidgetItem(epochs_str), QTableWidgetItem(loss_str),
                    QTableWidgetItem(acc_str), QTableWidgetItem(time_str)]
            for col, item in enumerate(cols):
                if col > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if not has_ckpt:
                    item.setForeground(Qt.GlobalColor.darkGray)
                self._table.setItem(lv, col, item)

            if not has_ckpt:
                # Grey out the row so the user knows these aren't selectable
                for col in range(5):
                    self._table.item(lv, col).setFlags(Qt.ItemFlag.ItemIsEnabled)

        # Pre-select levels that have checkpoints
        for lv in self._selectable_levels:
            self._table.selectRow(lv)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        # ── Epochs spinbox ──
        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Additional epochs:"))
        self._epoch_spin = QSpinBox()
        self._epoch_spin.setRange(1, 500)
        self._epoch_spin.setValue(5)
        self._epoch_spin.setFixedWidth(80)
        self._epoch_spin.setStyleSheet(
            "QSpinBox { background: #2a2a2a; color: #cccccc; border: 1px solid #444; "
            "border-radius: 3px; padding: 3px 6px; }"
        )
        epoch_row.addWidget(self._epoch_spin)
        epoch_row.addStretch()
        layout.addLayout(epoch_row)

        # ── Buttons ──
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._on_selection_changed()  # set initial OK-button state

    def _on_selection_changed(self) -> None:
        selected = self.selected_levels()
        ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(len(selected) > 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def selected_levels(self) -> list[int]:
        """Returns sorted list of selected level indices."""
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        return sorted(lv for lv in rows if lv in self._selectable_levels)

    def additional_epochs(self) -> int:
        return self._epoch_spin.value()
