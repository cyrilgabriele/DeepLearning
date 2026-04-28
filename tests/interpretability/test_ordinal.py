import json
from pathlib import Path

import numpy as np

from src.interpretability.ordinal import (
    classes_from_scores,
    load_ordinal_calibration,
    qwk_metric_label,
)


def test_classes_from_scores_uses_stored_thresholds():
    calibration = {
        "method": "optimized_thresholds",
        "thresholds": [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75],
    }

    classes = classes_from_scores(np.array([1.6, 1.8, 7.9]), calibration)

    assert classes.tolist() == [1, 2, 8]
    assert qwk_metric_label(calibration) == "optimized_thresholds"


def test_classes_from_scores_falls_back_to_rounding_without_thresholds():
    classes = classes_from_scores(np.array([1.4, 1.6, 8.6]), None)

    assert classes.tolist() == [1, 2, 8]
    assert qwk_metric_label(None) == "rounded_scores"


def test_load_ordinal_calibration_prefers_eval_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eval_path = tmp_path / "outputs" / "eval" / "kan_paper" / "run" / "X_eval.parquet"
    eval_path.parent.mkdir(parents=True)
    sidecar_payload = {
        "method": "optimized_thresholds",
        "thresholds": [1, 2, 3, 4, 5, 6, 7],
    }
    (eval_path.parent / "ordinal_thresholds.json").write_text(json.dumps(sidecar_payload))

    checkpoint = Path("checkpoints/run/model-20260424-000000.pt")

    assert load_ordinal_calibration(
        eval_features_path=eval_path,
        checkpoint_path=checkpoint,
    ) == sidecar_payload
