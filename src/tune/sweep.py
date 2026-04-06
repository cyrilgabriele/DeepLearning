"""Optuna tuning runner driven through the main entrypoint and Trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml

from src.configs import ExperimentConfig
from src.training.trainer import run_train


_SWEEP_DIR = Path("sweeps")
DEFAULT_N_TRIALS = 50
ModelFamily = Literal["glm", "xgboost-paper", "chebykan", "fourierkan", "bsplinekan"]


def run_tune(
    base_config: ExperimentConfig,
    *,
    device: str,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
) -> optuna.Study:
    """Run an Optuna sweep by materialising trial configs and calling Trainer.run()."""

    model_family = _resolve_model_family(base_config)
    n_trials = n_trials_override if n_trials_override is not None else DEFAULT_N_TRIALS

    _SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    study_name = f"{base_config.trainer.experiment_name}-{model_family}-tune"
    db_path = (_SWEEP_DIR / f"{study_name}.db").resolve()
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base_config.trainer.seed),
    )

    print(f"\nStarting tune stage for '{base_config.trainer.experiment_name}'")
    print(f"Model family: {model_family}")
    print(f"Trials: {n_trials}")
    if timeout_override is not None:
        print(f"Timeout: {timeout_override}s")
    print(f"Storage: {db_path}")
    print("=" * 70)

    def objective(trial: optuna.Trial) -> float:
        sampled_params = _sample_trial_params(trial, model_family=model_family)
        trial_config = _build_trial_config(
            base_config,
            sampled_params,
            model_family=model_family,
            experiment_name=f"{base_config.trainer.experiment_name}-trial-{trial.number:03d}",
            include_test_csv=False,
        )

        artifacts = run_train(trial_config, device=device)
        qwk = artifacts.metrics.get("qwk")
        if qwk is None:
            raise RuntimeError("Trainer returned no QWK metric for tune stage.")

        trial.set_user_attr("experiment_name", trial_config.trainer.experiment_name)
        trial.set_user_attr(
            "metrics",
            {key: value for key, value in artifacts.metrics.items() if value is not None},
        )
        if artifacts.summary_path is not None:
            trial.set_user_attr("summary_path", str(artifacts.summary_path))
        if artifacts.checkpoint_path is not None:
            trial.set_user_attr("checkpoint_path", str(artifacts.checkpoint_path))

        print(f"Trial {trial.number}: qwk={qwk:.4f} | params={sampled_params}")
        return qwk

    study.optimize(objective, n_trials=n_trials, timeout=timeout_override)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Tune stage finished without any completed trials.")

    best_config = _build_trial_config(
        base_config,
        study.best_params,
        model_family=model_family,
        experiment_name=f"{base_config.trainer.experiment_name}-tuned",
        include_test_csv=True,
    )

    results_payload = {
        "study_name": study.study_name,
        "model_family": model_family,
        "preprocessing_recipe": base_config.preprocessing.recipe,
        "best_qwk": round(float(study.best_value), 6),
        "best_params": study.best_params,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": dict(study.best_trial.user_attrs),
    }
    results_path = _SWEEP_DIR / f"{study_name}_best.json"
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True))

    config_path = _SWEEP_DIR / f"{study_name}_best.yaml"
    config_path.write_text(
        yaml.safe_dump(
            best_config.model_dump(mode="json"),
            sort_keys=False,
        )
    )

    print(f"\n{'=' * 70}")
    print(f"TUNE COMPLETE — best QWK: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved results: {results_path}")
    print(f"Saved config: {config_path}")
    print(f"Run with: python main.py --stage train --config {config_path}")

    top_trials = sorted(
        completed_trials,
        key=lambda trial: trial.value if trial.value is not None else float("-inf"),
        reverse=True,
    )[:10]
    print("\nTop trials:")
    for rank, trial in enumerate(top_trials, start=1):
        print(f"  {rank}. qwk={trial.value:.4f} | params={trial.params}")

    return study


def _resolve_model_family(config: ExperimentConfig) -> ModelFamily:
    if config.model.name == "glm":
        return "glm"
    if config.model.name == "xgboost-paper":
        return "xgboost-paper"
    if not config.model.name.startswith("tabkan"):
        raise ValueError(
            f"Tune stage does not support model '{config.model.name}'. "
            "Use a registry-backed trainer model."
        )
    if config.model.flavor is None:
        raise ValueError("TabKAN tune runs require `model.flavor` in the experiment config.")
    return config.model.flavor


def _sample_trial_params(trial: optuna.Trial, *, model_family: ModelFamily) -> dict[str, Any]:
    if model_family == "glm":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
        }

    if model_family == "xgboost-paper":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    params: dict[str, Any] = {
        "depth": trial.suggest_int("depth", 1, 4),
        "width": trial.suggest_categorical("width", [32, 64, 128, 256]),
        "max_epochs": trial.suggest_int("max_epochs", 50, 200, step=25),
        "sparsity_lambda": trial.suggest_categorical(
            "sparsity_lambda",
            [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        ),
    }

    if model_family == "chebykan":
        params["degree"] = trial.suggest_int("degree", 2, 8)
        return params

    params["grid_size"] = trial.suggest_int("grid_size", 3, 10)
    if model_family == "bsplinekan":
        params["spline_order"] = trial.suggest_int("spline_order", 2, 5)
    return params


def _build_trial_config(
    base_config: ExperimentConfig,
    sampled_params: dict[str, Any],
    *,
    model_family: ModelFamily,
    experiment_name: str,
    include_test_csv: bool,
) -> ExperimentConfig:
    payload = base_config.model_dump(mode="python")
    payload["trainer"]["experiment_name"] = experiment_name
    if not include_test_csv:
        payload["trainer"]["test_csv"] = None

    model_payload = dict(payload["model"])
    model_params = dict(model_payload.get("params") or {})

    if model_family == "glm":
        model_params["alpha"] = float(sampled_params["alpha"])

    elif model_family == "xgboost-paper":
        model_params.update(
            {
                "n_estimators": int(sampled_params["n_estimators"]),
                "max_depth": int(sampled_params["max_depth"]),
                "min_child_weight": float(sampled_params["min_child_weight"]),
                "learning_rate": float(sampled_params["learning_rate"]),
                "subsample": float(sampled_params["subsample"]),
                "colsample_bytree": float(sampled_params["colsample_bytree"]),
                "reg_alpha": float(sampled_params["reg_alpha"]),
                "reg_lambda": float(sampled_params["reg_lambda"]),
                # Optuna is already searching these params, so disable the model's
                # internal sequential tuner for the tune stage output.
                "auto_tune": False,
            }
        )

    else:
        model_payload["depth"] = int(sampled_params["depth"])
        model_payload["width"] = int(sampled_params["width"])
        model_params["max_epochs"] = int(sampled_params["max_epochs"])
        model_params["sparsity_lambda"] = float(sampled_params["sparsity_lambda"])
        model_params.setdefault("l1_weight", 1.0)
        model_params.setdefault("entropy_weight", 1.0)

        if model_family == "chebykan":
            model_payload["degree"] = int(sampled_params["degree"])
        else:
            model_payload["degree"] = None
            model_params["grid_size"] = int(sampled_params["grid_size"])
            if model_family == "bsplinekan":
                model_params["spline_order"] = int(sampled_params["spline_order"])

    model_payload["params"] = model_params
    payload["model"] = model_payload
    return ExperimentConfig.model_validate(payload)
