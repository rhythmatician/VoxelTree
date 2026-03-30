"""Helpers for extracting user-facing training summaries from step logs."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from voxel_tree.gui.step_definitions import StepDef

_DISPLAY_NAMES = {
    "biome_classifier": "Biome Classifier",
    "density": "Density MLP",
    "heightmap_predictor": "Heightmap Predictor",
    "voxy": "Voxy",
}


def summarize_training_run(
    step: StepDef,
    log_lines: Sequence[str],
    profile_name: str | None = None,
) -> dict[str, Any] | None:
    """Build a popup summary for a completed training step."""
    summary_lines = _summary_lines_for_step(step, log_lines)
    if not summary_lines:
        return None

    title = f"Training complete — {_display_name(step)}"
    text_lines: list[str] = []
    if profile_name:
        text_lines.append(f"Profile: {profile_name}")
        text_lines.append("")
    text_lines.extend(summary_lines)

    result: dict[str, Any] = {
        "title": title,
        "text": "\n".join(text_lines),
    }

    # Attach history for voxy and similar models
    track = step.track or ""
    if track == "voxy":
        parsed_result = _parse_last_voxy_result(log_lines)
        if parsed_result and isinstance(parsed_result.get("history"), list):
            result["history"] = parsed_result["history"]
            result["metric_key"] = "loss"

    return result


def summarize_build_pairs_run(
    log_lines: Sequence[str],
) -> dict[str, Any] | None:
    """Build a popup summary for a completed build_pairs step.

    Parses the ``[STEP_RESULT]`` JSON emitted by build_voxy_pairs.main()
    and returns a dict with ``title`` and ``text`` keys (same shape as the
    dict returned by :func:`summarize_training_run`).
    """
    result = _parse_last_build_pairs_result(log_lines)
    if not result:
        return None

    pairs_saved = int(result.get("pairs_saved", 0))
    matched = int(result.get("matched_sections", 0))
    total_dumps = int(result.get("total_dump_files", 0))
    total_voxy = int(result.get("total_voxy_sections", 0))
    skipped_no_voxy = int(result.get("skipped_no_voxy", 0))
    skipped_no_dump = int(result.get("skipped_no_dump", 0))
    total_skipped = int(result.get("total_skipped", 0))

    lines: list[str] = [
        f"Pairs saved:           {pairs_saved:,}  ({matched:,} sections \u00d7 8 octants)",
        f"Dump files found:      {total_dumps:,}",
        f"Voxy sections found:   {total_voxy:,}",
        f"Matched sections:      {matched:,}",
    ]

    if total_skipped > 0:
        lines.append("")
        lines.append(f"Failures ({total_skipped:,} total):")
        if skipped_no_voxy > 0:
            lines.append(
                f"  Missing Voxy section:  {skipped_no_voxy:,}"
                "  (dump exists, no Voxy NPZ)"
            )
        if skipped_no_dump > 0:
            lines.append(
                f"  Missing dump file:     {skipped_no_dump:,}"
                "  (Voxy exists, no noise dump)"
            )
    else:
        lines.append("")
        lines.append("No failures \u2014 all sections matched.")

    return {
        "title": "Build pairs complete",
        "text": "\n".join(lines),
    }


def _parse_last_build_pairs_result(lines: Sequence[str]) -> dict[str, Any] | None:
    """Return the last ``[STEP_RESULT]`` payload from a build_pairs log, or None."""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("[STEP_RESULT]"):
            payload = stripped[len("[STEP_RESULT]"):]
            try:
                parsed = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and "pairs_saved" in parsed:
                return parsed
    return None


def _summary_lines_for_step(step: StepDef, log_lines: Sequence[str]) -> list[str]:
    track = step.track or ""
    if track == "biome_classifier":
        return _summarize_biome(log_lines)
    if track == "density":
        return _summarize_density(log_lines)
    if track == "heightmap_predictor":
        return _summarize_heightmap(log_lines)
    if track == "voxy":
        return _summarize_voxy(log_lines)
    return _summarize_octree_style(log_lines)


def _display_name(step: StepDef) -> str:
    if step.track:
        return _DISPLAY_NAMES.get(step.track, step.track.replace("_", " ").title())
    return step.label


def _find_last_match(lines: Sequence[str], pattern: str) -> re.Match[str] | None:
    regex = re.compile(pattern)
    for line in reversed(lines):
        match = regex.search(line)
        if match:
            return match
    return None


def _format_percent(value: str | float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _summarize_biome(lines: Sequence[str]) -> list[str]:
    match = _find_last_match(
        lines,
        r"Training complete in (?P<duration>[0-9.]+s) — best val_ce=(?P<val_ce>[0-9.]+) acc=(?P<acc>[0-9.]+) @ epoch (?P<epoch>\d+)",
    )
    if not match:
        return []
    return [
        f"Best validation CE: {match.group('val_ce')}",
        f"Validation accuracy: {_format_percent(match.group('acc'))}",
        f"Best epoch: {match.group('epoch')}",
        f"Duration: {match.group('duration')}",
    ]


def _summarize_density(lines: Sequence[str]) -> list[str]:
    match = _find_last_match(
        lines,
        r"Training complete in (?P<duration>[0-9.]+s) — best val_mse=(?P<val_mse>[0-9.]+) @ epoch (?P<epoch>\d+)",
    )
    if not match:
        return []
    return [
        f"Best validation MSE: {match.group('val_mse')}",
        f"Best epoch: {match.group('epoch')}",
        f"Duration: {match.group('duration')}",
    ]


def _summarize_heightmap(lines: Sequence[str]) -> list[str]:
    match = _find_last_match(
        lines,
        r"Training complete in (?P<duration>[0-9.]+s) — best val_mse=(?P<val_mse>[0-9.]+) \(rmse=(?P<rmse>[0-9.]+) blocks\) @ epoch (?P<epoch>\d+)",
    )
    if not match:
        return []
    return [
        f"Best validation MSE: {match.group('val_mse')}",
        f"Validation RMSE: {match.group('rmse')} blocks",
        f"Best epoch: {match.group('epoch')}",
        f"Duration: {match.group('duration')}",
    ]


def _summarize_voxy(lines: Sequence[str]) -> list[str]:
    result = _parse_last_voxy_result(lines)
    if not result:
        return []

    history = result.get("history")
    latest = history[-1] if isinstance(history, list) and history else {}
    if not isinstance(latest, dict):
        latest = {}

    summary = [f"Best loss: {_format_float(result.get('best_loss'))}"]
    if latest.get("split_f1") is not None:
        summary.append(f"Latest split F1: {_format_float(latest.get('split_f1'), 4)}")
    if latest.get("leaf_acc") is not None:
        summary.append(f"Latest leaf accuracy: {_format_percent(latest.get('leaf_acc'))}")
    if latest.get("leaf_node_ratio") is not None:
        summary.append(f"Latest leaf ratio: {_format_float(latest.get('leaf_node_ratio'), 3)}")
    if latest.get("epoch") is not None:
        summary.append(f"Final epoch: {int(float(latest['epoch']))}")
    return summary


def _parse_last_voxy_result(lines: Sequence[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        stripped = line.strip()
        try:
            if stripped.startswith("[STEP_RESULT]"):
                payload = stripped[len("[STEP_RESULT]") :]
                parsed = json.loads(payload)
            elif stripped.startswith("{") and "best_loss" in stripped and "history" in stripped:
                parsed = ast.literal_eval(stripped)
            else:
                continue
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _summarize_octree_style(lines: Sequence[str]) -> list[str]:
    duration = _find_last_match(lines, r"Training completed in (?P<hours>[0-9.]+) hours")
    best_loss = _find_last_match(lines, r"New best model saved \(val_loss: (?P<val_loss>[0-9.]+)\)")
    val_loss = _find_last_match(lines, r"Val\s+— Loss: (?P<loss>[0-9.]+)")
    val_acc = _find_last_match(
        lines,
        r"Val\s+— Acc: (?P<overall>[0-9.]+) \(Air: (?P<air>[0-9.]+), Block: (?P<block>[0-9.]+)\)",
    )
    occ_f1 = _find_last_match(
        lines,
        r"Val\s+— Occ F1: (?P<f1>[0-9.]+)\s+Recall: (?P<recall>[0-9.]+)\s+FNR: (?P<fnr>[0-9.]+)\s+Recall@0.3: (?P<recall_rt>[0-9.]+)",
    )

    summary: list[str] = []
    if best_loss:
        summary.append(f"Best validation loss: {best_loss.group('val_loss')}")
    elif val_loss:
        summary.append(f"Latest validation loss: {val_loss.group('loss')}")
    if val_acc:
        summary.append(f"Validation accuracy: {_format_percent(val_acc.group('overall'))}")
        summary.append(
            "Air / block accuracy: "
            f"{_format_percent(val_acc.group('air'))} / {_format_percent(val_acc.group('block'))}"
        )
    if occ_f1:
        summary.append(f"Occupancy F1: {occ_f1.group('f1')}")
        summary.append(f"Occupancy recall: {_format_percent(occ_f1.group('recall'))}")
    if duration:
        summary.append(f"Duration: {duration.group('hours')} hours")
    return summary


def _format_float(value: Any, precision: int = 4) -> str:
    return f"{float(value):.{precision}f}"

