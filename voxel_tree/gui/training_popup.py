"""Custom training summary popup with embedded matplotlib graph."""

from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class TrainingResultsPopup(QDialog):
    """Custom dialog showing training results with optional metrics graph."""

    def __init__(
        self,
        title: str,
        summary_text: str,
        history: list[dict[str, Any]] | None = None,
        metric_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize the training results popup.

        Parameters
        ----------
        title : str
            Window title
        summary_text : str
            Plain text summary to display
        history : list[dict] | None
            Per-epoch history records (for plotting)
        metric_key : str | None
            Primary metric to plot (e.g., 'loss', 'split_f1')
        parent : QWidget | None
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1000, 600)
        self._build_layout(summary_text, history, metric_key)

    def _build_layout(
        self,
        summary_text: str,
        history: list[dict[str, Any]] | None = None,
        metric_key: str | None = None,
    ) -> None:
        """Build the main layout with summary and optional graph."""
        main_layout = QHBoxLayout()

        # Left panel: Summary text
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        summary_label = QLabel(summary_text)
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet(
            """
            QLabel {
                font-family: monospace;
                font-size: 10pt;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 4px;
            }
        """
        )

        left_layout.addWidget(summary_label)
        left_layout.addStretch()
        left_widget.setLayout(left_layout)

        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(350)

        main_layout.addWidget(scroll)

        # Right panel: Graph (if history available)
        if history:
            graph_widget = self._build_graph_panel(history, metric_key)
            main_layout.addWidget(graph_widget, stretch=1)

        main_layout.setSpacing(10)
        self.setLayout(main_layout)

    def _build_graph_panel(
        self,
        history: list[dict[str, Any]],
        metric_key: str | None = None,
    ) -> QWidget:
        """Build the right-side graph panel."""
        widget = QWidget()
        layout = QVBoxLayout()

        try:
            df = pd.DataFrame(history)
            fig = self._create_figure(df, metric_key)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
        except Exception as e:
            error_label = QLabel(f"Could not plot graph:\n{str(e)}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)

        widget.setLayout(layout)
        return widget

    def _create_figure(
        self,
        df: pd.DataFrame,
        metric_key: str | None = None,
    ) -> Figure:
        """Create matplotlib figure with training metrics."""
        fig = Figure(figsize=(6, 5), dpi=100)
        axes = fig.subplots(2, 2)

        # Determine available metrics
        has_loss = "loss" in df.columns
        has_split_f1 = "split_f1" in df.columns
        has_leaf_acc = "leaf_acc" in df.columns
        has_leaf_ratio = "leaf_node_ratio" in df.columns

        # 1. Loss
        if has_loss:
            ax = axes[0, 0]
            ax.plot(df["epoch"], df["loss"], "o-", linewidth=1.5, markersize=3, color="#d62728")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Loss", fontsize=9)
            ax.set_title("Training Loss", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")

        # 2. Split F1
        if has_split_f1:
            ax = axes[0, 1]
            ax.plot(df["epoch"], df["split_f1"], "o-", linewidth=1.5, markersize=3, color="#2ca02c")
            ax.axhline(
                0.90, color="green", linestyle="--", alpha=0.4, linewidth=1, label="Target: 0.90"
            )
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Split F1", fontsize=9)
            ax.set_title("Split Decision F1", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(fontsize=8)
            ax.set_ylim([min(0.4, df["split_f1"].min() - 0.05), 1.0])

        # 3. Leaf Accuracy
        if has_leaf_acc:
            ax = axes[1, 0]
            ax.plot(df["epoch"], df["leaf_acc"], "s-", linewidth=1.5, markersize=3, color="#1f77b4")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Leaf Accuracy", fontsize=9)
            ax.set_title("Leaf Classification Accuracy", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")

        # 4. Leaf Node Ratio
        if has_leaf_ratio:
            ax = axes[1, 1]
            ax.plot(
                df["epoch"],
                df["leaf_node_ratio"],
                "^-",
                linewidth=1.5,
                markersize=3,
                color="#ff7f0e",
            )
            ax.axhline(
                1.0, color="green", linestyle="--", alpha=0.4, linewidth=1, label="Target: 1.0"
            )
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Leaf Node Ratio", fontsize=9)
            ax.set_title("Tree Node Ratio", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(fontsize=8)

        fig.tight_layout()
        return fig
