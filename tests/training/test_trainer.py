import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainerConfig,
    load_experiment_config,
    set_global_seed,
)
from src.training.trainer import Trainer


def _write_mock_training_csv(tmp_path: Path, n_rows: int = 80) -> Path:
    rng = np.random.default_rng(7)
    repeats = math.ceil(n_rows / 8)
    response = np.tile(np.arange(1, 9), repeats)[:n_rows]
    df = pd.DataFrame(
        {
            "Id": np.arange(n_rows),
            "Response": response,
            "BMI": rng.normal(30, 5, size=n_rows),
            "Product_Info_2": rng.choice(list("ABCDEFG"), size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
            "Product_Info_3": rng.integers(1, 4, size=n_rows),
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def _write_mock_test_csv(tmp_path: Path, n_rows: int = 24) -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Id": np.arange(1000, 1000 + n_rows),
            "BMI": rng.normal(30, 5, size=n_rows),
            "Product_Info_2": rng.choice(list("ABCDEFG"), size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
            "Product_Info_3": rng.integers(1, 4, size=n_rows),
        }
    )
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_trainer_runs_on_mock_data(tmp_path):
    train_csv = _write_mock_training_csv(tmp_path)
    test_csv = _write_mock_test_csv(tmp_path)
    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="test",
            train_csv=train_csv,
            test_csv=test_csv,
            seed=123,
        ),
        preprocessing=PreprocessingConfig(
            recipe="xgboost_paper",
        ),
        model=ModelConfig(
            name="tabkan-tiny",
            flavor="chebykan",
            depth=2,
            width=32,
            degree=3,
            params={},
        ),
    )

    seed = set_global_seed(config.trainer.seed)
    trainer = Trainer(config, device="cpu", random_seed=seed)
    artifacts = trainer.run()
    assert set(artifacts.metrics) == {"mae", "accuracy", "f1_macro", "qwk"}
    assert artifacts.device in {"cpu", "cuda", "mps"}
    assert artifacts.test_predictions_path is not None
    assert artifacts.test_predictions_path.exists()


def test_trainer_runs_xgboost_paper_model(tmp_path):
    train_csv = _write_mock_training_csv(tmp_path)
    test_csv = _write_mock_test_csv(tmp_path)
    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="xgb-paper",
            train_csv=train_csv,
            test_csv=test_csv,
            seed=21,
        ),
        preprocessing=PreprocessingConfig(
            recipe="xgboost_paper",
        ),
        model=ModelConfig(
            name="xgboost-paper",
            flavor=None,
            depth=1,
            width=1,
            degree=1,
            params={
                "n_estimators": 3,
                "max_depth": 3,
                "learning_rate": 0.3,
            },
        ),
    )

    seed = set_global_seed(config.trainer.seed)
    trainer = Trainer(config, device="cpu", random_seed=seed)
    artifacts = trainer.run()
    assert artifacts.metrics["accuracy"] is not None
    assert artifacts.test_predictions_path is not None
    assert artifacts.test_predictions_path.exists()


def test_trainer_exports_eval_artifacts_under_recipe_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_csv = _write_mock_training_csv(tmp_path)
    test_csv = _write_mock_test_csv(tmp_path)
    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="xgb-paper",
            train_csv=train_csv,
            test_csv=test_csv,
            seed=21,
        ),
        preprocessing=PreprocessingConfig(
            recipe="xgboost_paper",
        ),
        model=ModelConfig(
            name="xgboost-paper",
            flavor=None,
            depth=1,
            width=1,
            degree=1,
            params={
                "n_estimators": 3,
                "max_depth": 3,
                "learning_rate": 0.3,
            },
        ),
    )

    seed = set_global_seed(config.trainer.seed)
    trainer = Trainer(config, device="cpu", random_seed=seed)
    trainer.run()

    eval_dir = tmp_path / "outputs" / "eval" / "xgboost_paper" / "xgb-paper"
    assert (eval_dir / "X_eval.parquet").exists()
    assert (eval_dir / "y_eval.parquet").exists()
    assert (eval_dir / "X_eval_raw.parquet").exists()
    assert (eval_dir / "feature_names.json").exists()
    assert (eval_dir / "feature_types.json").exists()


def test_trainer_persists_preprocessing_payload_without_fingerprint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_csv = _write_mock_training_csv(tmp_path)
    test_csv = _write_mock_test_csv(tmp_path)
    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="xgb-artifacts",
            train_csv=train_csv,
            test_csv=test_csv,
            seed=21,
        ),
        preprocessing=PreprocessingConfig(
            recipe="xgboost_paper",
        ),
        model=ModelConfig(
            name="xgboost-paper",
            flavor=None,
            depth=1,
            width=1,
            degree=1,
            params={
                "n_estimators": 3,
                "max_depth": 3,
                "learning_rate": 0.3,
            },
        ),
    )

    seed = set_global_seed(config.trainer.seed)
    trainer = Trainer(config, device="cpu", random_seed=seed)
    artifacts = trainer.run()

    summary_payload = json.loads(artifacts.summary_path.read_text())
    checkpoint_manifest = json.loads(artifacts.checkpoint_path.with_suffix(".manifest.json").read_text())

    assert "expected_feature_fingerprint" not in summary_payload["preprocessing"]
    assert "feature_space_fingerprint" not in summary_payload["preprocessing"]
    assert "expected_feature_fingerprint" not in checkpoint_manifest["preprocessing"]
    assert "feature_space_fingerprint" not in checkpoint_manifest["preprocessing"]


def test_config_loader_reads_yaml(tmp_path):
    cfg_text = """
trainer:
  experiment_name: base
  train_csv: placeholder.csv
  seed: 11
preprocessing:
  recipe: kan_paper
model:
  name: tabkan-base
  flavor: chebykan
  depth: 2
  width: 64
  degree: 3
  params: {}
tune:
  name: smoke-study
  storage: sweeps/smoke-study.db
  n_trials: 7
  timeout: 120
  sampler: random
  search_space:
    degree:
      type: int
      low: 2
      high: 6
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(cfg_text)

    config = load_experiment_config(config_path)
    assert config.preprocessing.recipe == "kan_paper"
    params = config.model.registry_kwargs()
    assert params["flavor"] == "chebykan"
    assert params["depth"] == 2
    assert params["width"] == 64
    assert config.tune is not None
    assert config.tune.n_trials == 7
    assert config.tune.search_space["degree"].type == "int"


def test_config_loader_rejects_stale_keys(tmp_path):
    cfg_text = """
trainer:
  experiment_name: base
  train_csv: placeholder.csv
  seed: 11
  eval_size: 0.2
preprocessing:
  recipe: kan_paper
  expected_feature_fingerprint: abc123
  missing_threshold: 0.5
model:
  name: tabkan-base
  flavor: chebykan
  depth: 2
  width: 64
  degree: 3
  params: {}
"""
    config_path = tmp_path / "bad-config.yaml"
    config_path.write_text(cfg_text)

    with pytest.raises(ValueError, match="extra"):
        load_experiment_config(config_path)
