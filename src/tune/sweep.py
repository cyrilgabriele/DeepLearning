"""Optuna tuning runner driven by the experiment config's ``tune`` block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml

from configs import ExperimentConfig
from src.training.trainer import run_train


_SWEEP_DIR = Path("sweeps")
ModelFamily = Literal["glm", "xgboost-paper", "chebykan", "fourierkan", "bsplinekan"]


def run_tune(
    base_config: ExperimentConfig,
    *,
    device: str,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
) -> optuna.Study:
    """Run an Optuna sweep by materialising trial configs and calling Trainer.run()."""

    tune_config = _require_tune_config(base_config)
    model_family = _resolve_model_family(base_config)
    study_name = tune_config.name or f"{base_config.trainer.experiment_name}-{model_family}-tune"
    storage_path = tune_config.storage or (_SWEEP_DIR / f"{study_name}.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = storage_path.resolve()
    storage = f"sqlite:///{db_path}"
    n_trials = n_trials_override if n_trials_override is not None else tune_config.n_trials
    timeout = timeout_override if timeout_override is not None else tune_config.timeout

    _SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=_build_sampler(tune_config.sampler, seed=base_config.trainer.seed),
    )

    print(f"\nStarting tune stage for '{base_config.trainer.experiment_name}'")
    print(f"Model family: {model_family}")
    print(f"Trials: {n_trials}")
    if timeout is not None:
        print(f"Timeout: {timeout}s")
    print(f"Storage: {db_path}")
    print("=" * 70)

    def objective(trial: optuna.Trial) -> float:
        sampled_params = _sample_trial_params(trial, search_space=tune_config.search_space)
        trial_config = _build_trial_config(
            base_config,
            sampled_params,
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

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Tune stage finished without any completed trials.")

    best_config = _build_trial_config(
        base_config,
        study.best_params,
        experiment_name=f"{base_config.trainer.experiment_name}-tuned",
        include_test_csv=True,
    )

    results_payload = {
        "study_name": study.study_name,
        "model_family": model_family,
        "preprocessing_recipe": base_config.preprocessing.recipe,
        "best_qwk": round(float(study.best_value), 6),
        "best_params": study.best_params,
        "search_space": {
            name: spec.model_dump(mode="json") for name, spec in tune_config.search_space.items()
        },
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


def _require_tune_config(config: ExperimentConfig):
    if config.tune is None:
        raise ValueError(
            "Stage 'tune' requires a `tune:` block in the experiment config "
            "with study settings and a `search_space` definition."
        )
    return config.tune


def _build_sampler(sampler_name: str, *, seed: int) -> optuna.samplers.BaseSampler:
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


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


def _sample_trial_params(
    trial: optuna.Trial,
    *,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        resolved = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
        spec_type = resolved["type"]
        if spec_type == "log_uniform":
            params[name] = trial.suggest_float(name, resolved["low"], resolved["high"], log=True)
        elif spec_type == "uniform":
            params[name] = trial.suggest_float(name, resolved["low"], resolved["high"])
        elif spec_type == "int":
            kwargs = {}
            if resolved.get("step") is not None:
                kwargs["step"] = resolved["step"]
            params[name] = trial.suggest_int(name, resolved["low"], resolved["high"], **kwargs)
        elif spec_type == "categorical":
            params[name] = trial.suggest_categorical(name, resolved["choices"])
        else:  # pragma: no cover - validated by SearchParamConfig
            raise ValueError(f"Unknown search space type: {spec_type}")
    return params


def _build_trial_config(
    base_config: ExperimentConfig,
    sampled_params: dict[str, Any],
    *,
    experiment_name: str,
    include_test_csv: bool,
) -> ExperimentConfig:
    payload = base_config.model_dump(mode="python")
    payload["trainer"]["experiment_name"] = experiment_name
    if not include_test_csv:
        payload["trainer"]["test_csv"] = None

    model_payload = dict(payload["model"])
    model_params = dict(model_payload.get("params") or {})
    model_family = _resolve_model_family(base_config)

    for key, value in sampled_params.items():
        if key in {"depth", "width", "degree"}:
            model_payload[key] = None if value is None else int(value)
        else:
            model_params[key] = value

    if model_family in {"chebykan", "fourierkan", "bsplinekan"}:
        model_params.setdefault("l1_weight", 1.0)
        model_params.setdefault("entropy_weight", 1.0)

    model_payload["params"] = model_params
    payload["model"] = model_payload
    return ExperimentConfig.model_validate(payload)
