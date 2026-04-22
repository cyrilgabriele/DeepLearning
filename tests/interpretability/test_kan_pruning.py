import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
from src.interpretability.kan_pruning import (
    PruningStats,
    _load_ordinal_calibration,
    run,
)
from src.metrics.qwk import _apply_thresholds


def _build_tabkan_config(tmp_path: Path, experiment_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name=experiment_name,
            train_csv=tmp_path / "train.csv",
            test_csv=None,
            seed=42,
        ),
        preprocessing=PreprocessingConfig(recipe="kan_paper"),
        model=ModelConfig(
            name="tabkan-base",
            flavor="chebykan",
            hidden_widths=(4, 2),
            degree=3,
            params={
                "max_epochs": 1,
                "lr": 0.001,
                "weight_decay": 0.0,
                "batch_size": 8,
                "sparsity_lambda": 0.0,
                "l1_weight": 1.0,
                "entropy_weight": 1.0,
            },
        ),
    )


def test_load_ordinal_calibration_falls_back_to_run_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checkpoint_path = tmp_path / "checkpoints" / "pruning-exp" / "model-20260422-082502.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({}, checkpoint_path)

    eval_features_path = tmp_path / "outputs" / "eval" / "kan_paper" / "pruning-exp" / "X_eval.parquet"
    eval_features_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feat_a": [0.0]}).to_parquet(eval_features_path)

    payload = {
        "method": "optimized_thresholds",
        "source_split": "inner_validation",
        "thresholds": [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75],
    }
    summary_path = tmp_path / "artifacts" / "pruning-exp" / "run-summary-20260422-082502.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"ordinal_calibration": payload}))

    calibration = _load_ordinal_calibration(
        eval_features_path=eval_features_path,
        checkpoint_path=checkpoint_path,
    )

    assert calibration == payload


def test_run_uses_stored_threshold_sidecar_for_pruning_qwk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    eval_dir = tmp_path / "outputs" / "eval" / "kan_paper" / "pruning-exp"
    eval_dir.mkdir(parents=True, exist_ok=True)

    X_eval = pd.DataFrame({"feat_a": [0.0, 1.0, 2.0, 3.0]})
    y_eval = pd.DataFrame({"Response": [1, 1, 2, 2]})
    X_eval.to_parquet(eval_dir / "X_eval.parquet")
    y_eval.to_parquet(eval_dir / "y_eval.parquet")

    thresholds = [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75]
    (eval_dir / "ordinal_thresholds.json").write_text(
        json.dumps(
            {
                "method": "optimized_thresholds",
                "source_split": "inner_validation",
                "thresholds": thresholds,
            }
        )
    )

    checkpoint_path = tmp_path / "checkpoints" / "pruning-exp" / "model-20260422-082502.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({}, checkpoint_path)

    config = _build_tabkan_config(tmp_path, "pruning-exp")
    interpret_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / "pruning-exp"

    class _FakeModule:
        def __init__(self, *args, **kwargs):
            self.loaded_state = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def state_dict(self):
            return self.loaded_state or {}

        def eval(self):
            return self

    class _FakeWrapper:
        def __init__(self, *args, **kwargs):
            self.widths = [4, 2]
            self.degree = 3
            self.grid_size = 4
            self.spline_order = 3
            self.lr = 0.001
            self.weight_decay = 0.0
            self.sparsity_lambda = 0.0
            self.l1_weight = 1.0
            self.entropy_weight = 1.0
            self.use_layernorm = False
            self.module = None
            self.thresholds = None
            self.threshold_source_split = None
            self.threshold_optimization_qwk = None

        def predict_scores(self, X):
            return np.array([1.49, 1.51, 2.49, 2.51], dtype=float)

        def predict(self, X):
            scores = self.predict_scores(X)
            if self.thresholds is not None:
                return np.clip(_apply_thresholds(scores, self.thresholds), 1, 8).astype(int)
            return np.clip(np.round(scores), 1, 8).astype(int)

    import src.interpretability.kan_pruning as kan_pruning
    import src.models.tabkan as tabkan

    monkeypatch.setattr(tabkan, "TabKANClassifier", _FakeWrapper)
    monkeypatch.setattr(tabkan, "TabKAN", _FakeModule)
    monkeypatch.setattr(
        kan_pruning,
        "prune_kan",
        lambda model, threshold: (
            model,
            PruningStats(
                threshold=threshold,
                edges_before=8,
                edges_after=8,
                sparsity_ratio=0.0,
            ),
            [],
        ),
    )

    result = run(
        checkpoint_path=checkpoint_path,
        config=config,
        flavor="chebykan",
        eval_features_path=eval_dir / "X_eval.parquet",
        eval_labels_path=eval_dir / "y_eval.parquet",
        threshold=0.01,
        output_dir=interpret_dir,
    )

    summary_path = interpret_dir / "reports" / "chebykan_pruning_summary.json"
    summary_payload = json.loads(summary_path.read_text())

    assert result["qwk_metric"] == "optimized_thresholds"
    assert result["qwk_metric_source_split"] == "inner_validation"
    assert result["qwk_before"] == 1.0
    assert result["qwk_after"] == 1.0
    assert summary_payload["qwk_metric"] == "optimized_thresholds"
    assert summary_payload["qwk_metric_source_split"] == "inner_validation"
