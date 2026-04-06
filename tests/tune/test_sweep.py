from pathlib import Path

import optuna
import pytest

from configs import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
from configs.tune.tune_config import SearchParamConfig, TuneConfig
from src.tune.sweep import _build_trial_config, _sample_trial_params, run_tune


def _base_cheby_config(tmp_path: Path, *, include_tune: bool) -> ExperimentConfig:
    tune = None
    if include_tune:
        tune = TuneConfig(
            name="cheby-smoke",
            storage=tmp_path / "sweeps" / "cheby-smoke.db",
            n_trials=5,
            sampler="random",
            search_space={
                "depth": SearchParamConfig(type="int", low=1, high=4),
                "width": SearchParamConfig(type="categorical", choices=[32, 64, 128]),
                "degree": SearchParamConfig(type="int", low=2, high=8),
                "max_epochs": SearchParamConfig(type="int", low=50, high=200, step=25),
                "sparsity_lambda": SearchParamConfig(
                    type="categorical",
                    choices=[0.0, 1e-4, 1e-3],
                ),
            },
        )
    return ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="cheby-base",
            train_csv=tmp_path / "train.csv",
            test_csv=tmp_path / "test.csv",
            seed=11,
        ),
        preprocessing=PreprocessingConfig(recipe="kan_paper"),
        model=ModelConfig(
            name="tabkan-base",
            flavor="chebykan",
            depth=2,
            width=128,
            degree=3,
            params={},
        ),
        tune=tune,
    )


def test_sample_trial_params_uses_configured_search_space():
    trial = optuna.trial.FixedTrial(
        {
            "depth": 4,
            "width": 64,
            "degree": 6,
            "max_epochs": 125,
            "sparsity_lambda": 1e-4,
        }
    )
    search_space = {
        "depth": SearchParamConfig(type="int", low=1, high=4),
        "width": SearchParamConfig(type="categorical", choices=[32, 64, 128]),
        "degree": SearchParamConfig(type="int", low=2, high=8),
        "max_epochs": SearchParamConfig(type="int", low=50, high=200, step=25),
        "sparsity_lambda": SearchParamConfig(type="categorical", choices=[0.0, 1e-4, 1e-3]),
    }

    sampled = _sample_trial_params(trial, search_space=search_space)

    assert sampled == {
        "depth": 4,
        "width": 64,
        "degree": 6,
        "max_epochs": 125,
        "sparsity_lambda": 1e-4,
    }


def test_build_trial_config_applies_sampled_params(tmp_path):
    base_config = _base_cheby_config(tmp_path, include_tune=True)

    trial_config = _build_trial_config(
        base_config,
        {
            "depth": 4,
            "width": 64,
            "degree": 5,
            "max_epochs": 125,
            "sparsity_lambda": 1e-4,
        },
        experiment_name="cheby-trial-001",
        include_test_csv=False,
    )

    assert trial_config.trainer.experiment_name == "cheby-trial-001"
    assert trial_config.trainer.test_csv is None
    assert trial_config.model.depth == 4
    assert trial_config.model.width == 64
    assert trial_config.model.degree == 5
    assert trial_config.model.params["max_epochs"] == 125
    assert trial_config.model.params["sparsity_lambda"] == 1e-4
    assert trial_config.model.params["l1_weight"] == 1.0
    assert trial_config.model.params["entropy_weight"] == 1.0


def test_run_tune_requires_tune_block(tmp_path):
    config = _base_cheby_config(tmp_path, include_tune=False)

    with pytest.raises(ValueError, match="requires a `tune:` block"):
        run_tune(config, device="cpu")
