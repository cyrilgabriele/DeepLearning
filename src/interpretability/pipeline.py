"""Stage-level orchestration for interpretability workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from configs import ExperimentConfig, load_experiment_config
from src.interpretability.utils.paths import (
    data as data_dir,
    eval_run_dir,
    interpret_run_dir,
    models as models_dir,
    reports as reports_dir,
)


def _require_file(path: Path, *, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{purpose} not found at {path}")
    return path


def _resolve_checkpoint(config: ExperimentConfig, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return _require_file(checkpoint_path, purpose="Checkpoint")

    checkpoint_root = Path("checkpoints") / config.trainer.experiment_name
    suffix = ".pt" if config.model.name.startswith("tabkan") else ".joblib"
    candidates = sorted(checkpoint_root.glob(f"model-*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '*{suffix}' found under {checkpoint_root}. "
            "Pass --checkpoint explicitly or run training first."
        )
    return candidates[-1]


def _load_config_from_summary(summary_path: Path) -> ExperimentConfig:
    payload = json.loads(summary_path.read_text())
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Run summary at {summary_path} does not contain a valid config payload.")
    return ExperimentConfig.model_validate(raw_config)


def _summary_for_checkpoint(checkpoint_path: Path) -> Path | None:
    stem = checkpoint_path.stem
    if not stem.startswith("model-"):
        return None

    timestamp = stem.removeprefix("model-")
    experiment_name = checkpoint_path.parent.name
    summary_path = Path("artifacts") / experiment_name / f"run-summary-{timestamp}.json"
    if not summary_path.exists():
        return None
    return summary_path


def _load_checkpoint_config(checkpoint_path: Path) -> ExperimentConfig | None:
    checkpoint = _require_file(checkpoint_path, purpose="Checkpoint")
    summary_path = _summary_for_checkpoint(checkpoint)
    if summary_path is None:
        return None
    return _load_config_from_summary(summary_path)


def _configs_match(left: ExperimentConfig, right: ExperimentConfig) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def resolve_interpret_config(
    *,
    config_path: Path | None,
    checkpoint_path: Path | None,
) -> ExperimentConfig:
    """Resolve interpret config from an explicit YAML path or checkpoint-linked artifacts."""

    explicit_config = load_experiment_config(config_path) if config_path is not None else None
    checkpoint_config = (
        _load_checkpoint_config(checkpoint_path)
        if checkpoint_path is not None
        else None
    )

    if checkpoint_config is not None:
        if explicit_config is not None and not _configs_match(explicit_config, checkpoint_config):
            raise ValueError(
                f"Provided config at {config_path} does not match the saved config for checkpoint "
                f"{checkpoint_path}."
            )
        return checkpoint_config

    if explicit_config is not None:
        return explicit_config

    source_hint = (
        "No run summary matched the checkpoint timestamp."
        if checkpoint_path is not None
        else "No checkpoint was provided."
    )
    raise ValueError(
        "Unable to resolve the interpret config automatically. "
        f"{source_hint} Pass --config explicitly."
    )


def run_interpret(
    config: ExperimentConfig,
    *,
    checkpoint_path: Path | None = None,
    output_root: Path = Path("outputs"),
    pruning_threshold: float = 0.01,
    qwk_tolerance: float = 0.01,
    candidate_library: str = "scipy",
) -> dict[str, object]:
    """Run the model-appropriate interpretability workflow for one experiment."""

    recipe = config.preprocessing.recipe
    experiment_name = config.trainer.experiment_name

    eval_dir = eval_run_dir(output_root, recipe, experiment_name, create=False)
    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Eval artifacts not found at {eval_dir}. "
            "Run `main.py --stage train --config ...` first to export them."
        )

    interpret_dir = interpret_run_dir(output_root, recipe, experiment_name)
    checkpoint = _resolve_checkpoint(config, checkpoint_path)

    eval_features_raw_path = eval_dir / "X_eval_raw.parquet"
    feature_names_path = eval_dir / "feature_names.json"
    feature_types_path = eval_dir / "feature_types.json"

    result: dict[str, object] = {
        "model": config.model.name,
        "recipe": recipe,
        "experiment_name": experiment_name,
        "checkpoint_path": checkpoint,
        "eval_dir": eval_dir,
        "output_dir": interpret_dir,
    }

    if config.model.name == "glm":
        from src.interpretability.glm_coefficients import run as run_glm_coefficients

        coefficients_path = run_glm_coefficients(
            checkpoint,
            _require_file(feature_names_path, purpose="Feature name metadata"),
            output_dir=interpret_dir,
        )
        result["artifacts"] = {
            "coefficients": coefficients_path,
        }
        return result

    if config.model.name == "xgboost-paper":
        from src.interpretability.shap_xgboost import run as run_shap_xgboost

        eval_features_path = _require_file(eval_dir / "X_eval.parquet", purpose="Eval features")
        eval_labels_path = _require_file(eval_dir / "y_eval.parquet", purpose="Eval labels")
        run_shap_xgboost(
            checkpoint,
            eval_features_path,
            eval_labels_path,
            output_dir=interpret_dir,
            eval_features_raw_path=eval_features_raw_path if eval_features_raw_path.exists() else None,
            feature_types_path=feature_types_path if feature_types_path.exists() else None,
        )
        result["artifacts"] = {
            "shap_values": data_dir(interpret_dir) / "shap_xgb_values.parquet",
        }
        return result

    if not config.model.name.startswith("tabkan"):
        raise NotImplementedError(
            f"Interpret stage is not implemented for model '{config.model.name}'."
        )

    flavor = config.model.flavor
    if flavor not in {"chebykan", "fourierkan"}:
        raise NotImplementedError(
            "Interpret stage currently supports only 'chebykan' and 'fourierkan' TabKAN flavors."
        )

    from src.interpretability.kan_pruning import run as run_kan_pruning
    from src.interpretability.kan_symbolic import run as run_kan_symbolic
    from src.interpretability.r2_pipeline import run as run_r2_pipeline

    eval_features_path = _require_file(eval_dir / "X_eval.parquet", purpose="Eval features")
    eval_labels_path = _require_file(eval_dir / "y_eval.parquet", purpose="Eval labels")
    run_kan_pruning(
        checkpoint,
        config,
        flavor,
        eval_features_path=eval_features_path,
        eval_labels_path=eval_labels_path,
        threshold=pruning_threshold,
        qwk_tolerance=qwk_tolerance,
        output_dir=interpret_dir,
    )

    pruning_summary_path = reports_dir(interpret_dir) / f"{flavor}_pruning_summary.json"
    pruned_checkpoint_path = models_dir(interpret_dir) / f"{flavor}_pruned_module.pt"

    feat_types = (
        json.loads(feature_types_path.read_text())
        if feature_types_path.exists()
        else {}
    )
    X_raw = (
        pd.read_parquet(eval_features_raw_path)
        if eval_features_raw_path.exists()
        else None
    )

    run_kan_symbolic(
        pruned_checkpoint_path,
        pruning_summary_path,
        config,
        eval_features_path,
        flavor,
        use_pysr=candidate_library == "pysr",
        output_dir=interpret_dir,
        feat_types=feat_types,
        X_raw=X_raw,
    )
    run_r2_pipeline(
        pruned_checkpoint_path,
        pruning_summary_path,
        config,
        eval_features_path,
        flavor,
        candidate_library=candidate_library,
        output_dir=interpret_dir,
    )

    # ── New paper-ready stages ───────────────────────────────────────────────
    import torch
    from src.models.tabkan import TabKAN

    X_eval = pd.read_parquet(eval_features_path)
    feature_names = list(X_eval.columns)
    in_features = X_eval.shape[1]
    widths = [config.model.width] * config.model.depth
    pruned_module = TabKAN(
        in_features=in_features, widths=widths, kan_type=flavor,
        degree=config.model.degree or 3,
    )
    pruned_module.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
    pruned_module.eval()

    symbolic_fits_path = data_dir(interpret_dir) / f"{flavor}_symbolic_fits.csv"

    # Formula composition (SymPy)
    from src.interpretability.formula_composition import run as run_formula_composition
    run_formula_composition(
        symbolic_fits_path, pruned_module, feature_names,
        interpret_dir, flavor, X_eval=X_eval,
    )

    # R² distribution histogram + quality tier pie chart
    from src.interpretability.quality_figures import plot_r2_distribution
    fits_df = pd.read_csv(symbolic_fits_path)
    if not fits_df.empty:
        plot_r2_distribution(fits_df, flavor, interpret_dir)

    # KAN network diagram with edge functions
    from src.interpretability.kan_network_diagram import draw_kan_diagram
    draw_kan_diagram(
        pruned_module, feature_names, flavor, interpret_dir,
        symbolic_fits=fits_df if not fits_df.empty else None,
    )

    result["artifacts"] = {
        "pruning_summary": pruning_summary_path,
        "pruned_checkpoint": pruned_checkpoint_path,
        "symbolic_fits": symbolic_fits_path,
        "r2_report": reports_dir(interpret_dir) / f"{flavor}_r2_report.json",
        "symbolic_formulas": reports_dir(interpret_dir) / f"{flavor}_symbolic_formulas.json",
        "kan_diagram": Path("figures") / f"{flavor}_kan_diagram.pdf",
        "r2_distribution": Path("figures") / f"{flavor}_r2_distribution.pdf",
    }
    return result
