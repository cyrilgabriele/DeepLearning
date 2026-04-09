import json
from pathlib import Path

import optuna
import pytest

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
from src.config.tune.tune_config import SearchParamConfig, TuneConfig
from src.models import TrainingArtifacts
from src.tune.sweep import _build_trial_config, _sample_trial_params, run_tune


def _base_cheby_config(tmp_path: Path, *, include_tune: bool) -> ExperimentConfig:
    tune = None
    if include_tune:
        tune = TuneConfig(
            name="cheby-smoke",
            storage=tmp_path / "sweeps" / "cheby-smoke.db",
            n_trials=5,
            sampler="random",
            top_k_candidates=3,
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


def test_run_tune_exports_candidate_manifest(tmp_path, monkeypatch):
    config = _base_cheby_config(tmp_path, include_tune=True)

    def fake_run_train(trial_config, *, device):
        trial_number = int(trial_config.trainer.experiment_name.rsplit("-", 1)[-1])
        qwk = 0.7 + (trial_number * 0.01)
        summary_path = tmp_path / f"summary-{trial_number}.json"
        checkpoint_path = tmp_path / f"checkpoint-{trial_number}.pt"
        summary_path.write_text("{}")
        checkpoint_path.write_text("checkpoint")
        return TrainingArtifacts(
            model=object(),
            metrics={"mae": 1.0, "accuracy": 0.5, "f1_macro": 0.4, "qwk": qwk},
            device=device,
            config=trial_config,
            random_seed=trial_config.trainer.seed,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.tune.sweep.run_train", fake_run_train)

    run_tune(config, device="cpu", n_trials_override=3)

    manifest_path = tmp_path / "sweeps" / "cheby-smoke_candidates.json"
    payload = json.loads(manifest_path.read_text())

    assert payload["study_name"] == "cheby-smoke"
    assert payload["model_family"] == "chebykan"
    assert payload["preprocessing_recipe"] == "kan_paper"
    assert payload["top_k"] == 3
    assert len(payload["candidates"]) == 3
    best_candidate = payload["candidates"][0]
    assert best_candidate["candidate_id"] == "cheby-smoke-trial-002"
    assert best_candidate["rank"] == 1
    assert best_candidate["trial_number"] == 2
    assert best_candidate["qwk"] == pytest.approx(0.72)
    assert best_candidate["model_config"]["hidden_widths"] == [128, 128]
    assert best_candidate["trainer_config"]["experiment_name"] == "cheby-base-candidate-002"
    assert best_candidate["preprocessing_config"]["recipe"] == "kan_paper"
    assert best_candidate["selection_metadata"]["architecture"]["hidden_widths"] == [128, 128]
    assert best_candidate["summary_path"].endswith("summary-2.json")
