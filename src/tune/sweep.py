"""Optuna tuning runner driven by the experiment config's ``tune`` block."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml

from src.config import ExperimentConfig
from src.training.trainer import run_train


_SWEEP_DIR = Path("sweeps")
ModelFamily = Literal["glm", "xgboost-paper", "chebykan", "fourierkan", "bsplinekan"]


def _compute_sparsity(checkpoint_path: str, config: ExperimentConfig, flavor: str) -> float:
    """Load a KAN checkpoint and return the sparsity ratio after pruning at threshold=0.01."""
    import torch
    from src.models.tabkan import TabKANClassifier, TabKAN
    from src.interpretability.kan_pruning import prune_kan

    manifest_path = Path(checkpoint_path).with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        in_features = manifest.get("in_features", 140)
    else:
        in_features = 140

    wrapper = TabKANClassifier(
        preset=config.model.name,
        flavor=flavor,
        hidden_widths=config.model.resolved_hidden_widths(),
        depth=config.model.depth,
        width=config.model.width,
        degree=config.model.degree or 3,
        grid_size=config.model.params.get("grid_size", 4),
    )
    wrapper.module = TabKAN(
        in_features=in_features,
        widths=wrapper.widths,
        kan_type=flavor,
        degree=wrapper.degree,
        grid_size=wrapper.grid_size,
    )
    wrapper.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    wrapper.module.eval()

    _, stats, _ = prune_kan(wrapper.module, threshold=0.01)
    return stats.sparsity_ratio


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

    multi_objective = tune_config.directions is not None and len(tune_config.directions) > 1

    # Sampler selection rule:
    #   - If the user explicitly chose `grid`, honour it — even for multi-objective.
    #   - Otherwise, if the study is multi-objective, fall back to NSGA-II.
    #     TPE/random are single-objective only and Optuna rejects them for
    #     Pareto studies, so this keeps legacy configs that set sampler=tpe
    #     (the default) with multiple directions working.
    if tune_config.sampler != "grid" and multi_objective:
        sampler = optuna.samplers.NSGAIISampler(seed=base_config.trainer.seed)
    else:
        sampler = _build_sampler(
            tune_config.sampler,
            seed=base_config.trainer.seed,
            search_space=tune_config.search_space,
        )

    if multi_objective:
        study = optuna.create_study(
            study_name=study_name,
            directions=tune_config.directions,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )

    print(f"\nStarting tune stage for '{base_config.trainer.experiment_name}'")
    print(f"Model family: {model_family}")
    if multi_objective:
        print(f"Mode: multi-objective (directions={tune_config.directions})")
    print(f"Trials: {n_trials}")
    if timeout is not None:
        print(f"Timeout: {timeout}s")
    print(f"Storage: {db_path}")
    print("=" * 70)

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
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

        if multi_objective and artifacts.checkpoint_path is not None:
            sparsity = _compute_sparsity(
                str(artifacts.checkpoint_path), trial_config, model_family,
            )
            trial.set_user_attr("sparsity_ratio", sparsity)
            print(
                f"Trial {trial.number}: qwk={qwk:.4f}  sparsity={sparsity:.4f} | "
                f"params={sampled_params}"
            )
            return qwk, sparsity

        print(f"Trial {trial.number}: qwk={qwk:.4f} | params={sampled_params}")
        return qwk

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Tune stage finished without any completed trials.")

    if multi_objective:
        _report_pareto(study, base_config, model_family, study_name, tune_config)
    else:
        _report_single_objective(study, base_config, model_family, study_name, tune_config)

    return study


def _report_pareto(
    study: optuna.Study,
    base_config: ExperimentConfig,
    model_family: ModelFamily,
    study_name: str,
    tune_config,
) -> None:
    """Report results for a multi-objective (Pareto) sweep."""
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pareto_trials = study.best_trials

    pareto_entries = []
    for trial in sorted(pareto_trials, key=lambda t: t.values[0], reverse=True):
        qwk, sparsity = trial.values
        entry = {
            "trial_number": trial.number,
            "qwk": round(qwk, 6),
            "sparsity_ratio": round(sparsity, 4),
            "params": trial.params,
            "checkpoint_path": trial.user_attrs.get("checkpoint_path"),
        }
        pareto_entries.append(entry)

    results_payload = {
        "study_name": study.study_name,
        "model_family": model_family,
        "preprocessing_recipe": base_config.preprocessing.recipe,
        "mode": "pareto",
        "directions": list(tune_config.directions),
        "objective_names": ["qwk", "sparsity_ratio"],
        "search_space": {
            name: spec.model_dump(mode="json") for name, spec in tune_config.search_space.items()
        },
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pareto_front": pareto_entries,
    }
    results_path = _SWEEP_DIR / f"{study_name}_pareto.json"
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True))

    # Save configs for each Pareto trial
    for entry in pareto_entries:
        trial_obj = next(
            t for t in pareto_trials if t.number == entry["trial_number"]
        )
        config = _build_trial_config(
            base_config,
            trial_obj.params,
            experiment_name=(
                f"{base_config.trainer.experiment_name}-pareto-"
                f"q{entry['qwk']:.3f}-s{entry['sparsity_ratio']:.2f}"
            ),
            include_test_csv=True,
        )
        config_path = _SWEEP_DIR / (
            f"{study_name}_pareto_trial{entry['trial_number']:03d}.yaml"
        )
        config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))

    print(f"\n{'=' * 70}")
    print(f"PARETO TUNE COMPLETE — {len(pareto_entries)} Pareto-optimal trials")
    print(f"Saved results: {results_path}")
    print(f"\nPareto front (sorted by QWK):")
    for entry in pareto_entries:
        params_str = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in entry["params"].items())
        print(
            f"  trial {entry['trial_number']:3d}: "
            f"qwk={entry['qwk']:.4f}  sparsity={entry['sparsity_ratio']:.4f}  "
            f"{params_str}"
        )


def _report_single_objective(
    study: optuna.Study,
    base_config: ExperimentConfig,
    model_family: ModelFamily,
    study_name: str,
    tune_config,
) -> None:
    """Report results for a single-objective sweep (original behaviour)."""
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

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

    candidates_path = _SWEEP_DIR / f"{study_name}_candidates.json"
    candidate_manifest = _build_candidate_manifest(
        study=study,
        base_config=base_config,
        top_k=tune_config.top_k_candidates,
        model_family=model_family,
    )
    candidates_path.write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True))

    print(f"\n{'=' * 70}")
    print(f"TUNE COMPLETE — best QWK: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved results: {results_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved candidates: {candidates_path}")
    print(f"Run with: python main.py --stage train --config {config_path}")

    top_trials = sorted(
        completed_trials,
        key=lambda trial: trial.value if trial.value is not None else float("-inf"),
        reverse=True,
    )[:10]
    print("\nTop trials:")
    for rank, trial in enumerate(top_trials, start=1):
        print(f"  {rank}. qwk={trial.value:.4f} | params={trial.params}")


def _build_candidate_manifest(
    *,
    study: optuna.Study,
    base_config: ExperimentConfig,
    top_k: int,
    model_family: ModelFamily,
) -> dict[str, Any]:
    """Export the top completed trials into a retrain-friendly manifest."""

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    ranked_trials = sorted(
        completed_trials,
        key=lambda trial: trial.value if trial.value is not None else float("-inf"),
        reverse=True,
    )[:top_k]

    candidates: list[dict[str, Any]] = []
    for rank, trial in enumerate(ranked_trials, start=1):
        trial_config = _build_trial_config(
            base_config,
            trial.params,
            experiment_name=f"{base_config.trainer.experiment_name}-candidate-{trial.number:03d}",
            include_test_csv=True,
        )
        model_params = dict(trial_config.model.params)
        architecture = trial_config.model.architecture_payload()
        candidates.append(
            {
                "candidate_id": f"{study.study_name}-trial-{trial.number:03d}",
                "rank": rank,
                "trial_number": trial.number,
                "qwk": None if trial.value is None else round(float(trial.value), 6),
                "metrics": dict(trial.user_attrs.get("metrics", {})),
                "model_config": trial_config.model.model_dump(mode="json"),
                "trainer_config": trial_config.trainer.model_dump(mode="json"),
                "preprocessing_config": trial_config.preprocessing.model_dump(mode="json"),
                "summary_path": trial.user_attrs.get("summary_path"),
                "checkpoint_path": trial.user_attrs.get("checkpoint_path"),
                "selection_metadata": {
                    "architecture": architecture,
                    "basis_parameters": _basis_parameters(trial_config),
                    "sparsity_parameters": {
                        key: model_params.get(key)
                        for key in ("sparsity_lambda", "l1_weight", "entropy_weight")
                        if key in model_params
                    },
                    "sampled_params": dict(trial.params),
                    "notes": [
                        f"Source experiment: {trial.user_attrs.get('experiment_name')}",
                    ],
                },
            }
        )

    return {
        "study_name": study.study_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_family": model_family,
        "preprocessing_recipe": base_config.preprocessing.recipe,
        "top_k": top_k,
        "candidates": candidates,
    }


def _basis_parameters(config: ExperimentConfig) -> dict[str, Any]:
    """Extract basis-defining parameters from the effective model config."""

    basis_keys = ("degree", "grid_size", "spline_order")
    model_params = dict(config.model.params)
    basis = {key: model_params[key] for key in basis_keys if key in model_params}
    if config.model.degree is not None:
        basis.setdefault("degree", config.model.degree)
    if config.model.flavor is not None:
        basis["flavor"] = config.model.flavor
    return basis


def _require_tune_config(config: ExperimentConfig):
    if config.tune is None:
        raise ValueError(
            "Stage 'tune' requires a `tune:` block in the experiment config "
            "with study settings and a `search_space` definition."
        )
    return config.tune


def _build_sampler(
    sampler_name: str,
    *,
    seed: int,
    search_space: dict[str, Any] | None = None,
) -> optuna.samplers.BaseSampler:
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        if not search_space:
            raise ValueError("Grid sampler requires a non-empty search_space.")
        grid: dict[str, list[Any]] = {}
        for name, spec in search_space.items():
            resolved = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            if resolved["type"] == "grid":
                grid[name] = list(resolved["values"])
            elif resolved["type"] == "categorical":
                grid[name] = list(resolved["choices"])
            else:
                raise ValueError(
                    f"Grid sampler requires all search-space entries to be type=grid "
                    f"or type=categorical; got type={resolved['type']} for '{name}'."
                )
        return optuna.samplers.GridSampler(grid, seed=seed)
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
        elif spec_type == "grid":
            params[name] = trial.suggest_categorical(name, resolved["values"])
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
            if key in {"depth", "width"}:
                model_payload["hidden_widths"] = None
        else:
            model_params[key] = value

    if model_family in {"chebykan", "fourierkan", "bsplinekan"}:
        model_params.setdefault("l1_weight", 1.0)
        model_params.setdefault("entropy_weight", 1.0)

    model_payload["params"] = model_params
    payload["model"] = model_payload
    return ExperimentConfig.model_validate(payload)
