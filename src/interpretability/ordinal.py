"""Ordinal score-to-class helpers for interpretability artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def summary_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Return the run-summary path associated with a model checkpoint."""
    stem = checkpoint_path.stem
    if not stem.startswith("model-"):
        return None

    timestamp = stem.removeprefix("model-")
    experiment_name = checkpoint_path.parent.name
    summary_path = Path("artifacts") / experiment_name / f"run-summary-{timestamp}.json"
    if not summary_path.exists():
        return None
    return summary_path


def load_ordinal_calibration(
    *,
    eval_features_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load stored ordinal thresholds from eval sidecar or checkpoint summary."""
    if eval_features_path is not None:
        sidecar_path = eval_features_path.with_name("ordinal_thresholds.json")
        if sidecar_path.exists():
            payload = json.loads(sidecar_path.read_text())
            if isinstance(payload, dict) and payload.get("thresholds"):
                return payload

    if checkpoint_path is None:
        return None

    summary_path = summary_for_checkpoint(checkpoint_path)
    if summary_path is None:
        return None

    summary_payload = json.loads(summary_path.read_text())
    calibration = summary_payload.get("ordinal_calibration")
    if isinstance(calibration, dict) and calibration.get("thresholds"):
        return calibration
    return None


def qwk_metric_label(ordinal_calibration: dict[str, Any] | None) -> str:
    """Return the active ordinal class-mapping label."""
    thresholds = None if ordinal_calibration is None else ordinal_calibration.get("thresholds")
    if thresholds is None or len(thresholds) == 0:
        return "rounded_scores"
    return str(ordinal_calibration.get("method", "optimized_thresholds"))


def classes_from_scores(
    scores: np.ndarray | list[float],
    ordinal_calibration: dict[str, Any] | None,
    *,
    num_classes: int = 8,
) -> np.ndarray:
    """Map continuous KAN scores to ordinal classes using the stored contract."""
    score_array = np.asarray(scores, dtype=float)
    thresholds = None if ordinal_calibration is None else ordinal_calibration.get("thresholds")
    if thresholds is not None and len(thresholds) > 0:
        from src.metrics.qwk import _apply_thresholds

        classes = _apply_thresholds(score_array, np.asarray(thresholds, dtype=float))
    else:
        classes = np.round(score_array)
    return np.clip(classes, 1, num_classes).astype(int)


def class_from_score(score: float, ordinal_calibration: dict[str, Any] | None) -> int:
    """Map one continuous score to an ordinal class."""
    return int(classes_from_scores([score], ordinal_calibration)[0])


def attach_ordinal_calibration(
    model_wrapper: Any,
    ordinal_calibration: dict[str, Any] | None,
) -> None:
    """Attach stored thresholds to a TabKAN-like wrapper when available."""
    if ordinal_calibration is None:
        return

    thresholds = ordinal_calibration.get("thresholds")
    if thresholds is None or len(thresholds) == 0:
        return

    model_wrapper.thresholds = np.asarray(thresholds, dtype=float)
    if hasattr(model_wrapper, "threshold_source_split"):
        model_wrapper.threshold_source_split = ordinal_calibration.get("source_split")
    if hasattr(model_wrapper, "threshold_optimization_qwk"):
        optimized_qwk = ordinal_calibration.get("optimized_qwk_on_source_split")
        model_wrapper.threshold_optimization_qwk = (
            None if optimized_qwk is None else float(optimized_qwk)
        )
