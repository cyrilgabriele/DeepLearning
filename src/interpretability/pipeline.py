"""Stage-level orchestration for interpretability workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig, load_experiment_config
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
    max_features: int | None = None,
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

    if config.model.name in {"xgboost-paper", "xgb"}:
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
    from src.interpretability.ordinal import (
        classes_from_scores,
        load_ordinal_calibration,
        qwk_metric_label,
    )

    ordinal_calibration = load_ordinal_calibration(
        eval_features_path=eval_features_path,
        checkpoint_path=checkpoint,
    )
    print(f"Ordinal class mapping: {qwk_metric_label(ordinal_calibration)}")

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
    restricted_features: list[str] | None = None

    # ── Optional feature restriction ─────────────────────────────────────────
    if max_features is not None:
        import torch
        from src.models.tabkan import TabKAN
        from src.interpretability.utils.kan_coefficients import (
            coefficient_importance_from_module,
            get_first_kan_layer,
        )

        _X_tmp = pd.read_parquet(eval_features_path)
        _feature_names = list(_X_tmp.columns)
        _widths = config.model.resolved_hidden_widths()
        _mod = TabKAN(
            in_features=_X_tmp.shape[1], widths=_widths, kan_type=flavor,
            degree=config.model.degree or 3,
            grid_size=config.model.params.get("grid_size", 4),
            use_layernorm=config.model.use_layernorm,
        )
        _mod.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
        _mod.eval()

        ranking = coefficient_importance_from_module(_mod, _feature_names)
        top_feats = set(ranking.head(max_features).index.tolist())
        drop_indices = [i for i, f in enumerate(_feature_names) if f not in top_feats]

        first_layer = get_first_kan_layer(_mod)
        with torch.no_grad():
            for in_i in drop_indices:
                if hasattr(first_layer, "cheby_coeffs"):
                    first_layer.cheby_coeffs[:, in_i, :] = 0.0
                elif hasattr(first_layer, "fourier_a"):
                    first_layer.fourier_a[:, in_i, :] = 0.0
                    first_layer.fourier_b[:, in_i, :] = 0.0
                first_layer.base_weight[:, in_i] = 0.0

        torch.save(_mod.state_dict(), pruned_checkpoint_path)
        kept = [f for f in _feature_names if f in top_feats]
        restricted_features = kept
        print(f"Feature restriction: kept top {max_features} features, "
              f"zeroed {len(drop_indices)} input edges.")
        print(f"Top features: {kept[:10]}{'...' if len(kept) > 10 else ''}")
        del _X_tmp, _mod

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
        preprocessing_recipe=recipe,
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
    widths = config.model.resolved_hidden_widths()
    pruned_module = TabKAN(
        in_features=in_features, widths=widths, kan_type=flavor,
        degree=config.model.degree or 3,
        grid_size=config.model.params.get("grid_size", 4),
        use_layernorm=config.model.use_layernorm,
    )
    pruned_module.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
    pruned_module.eval()

    symbolic_fits_path = data_dir(interpret_dir) / f"{flavor}_symbolic_fits.csv"
    ranking_path = data_dir(interpret_dir) / f"{flavor}_feature_ranking.csv"
    top20_path = data_dir(interpret_dir) / f"{flavor}_top20_features.json"
    top12_path = data_dir(interpret_dir) / f"{flavor}_top12_features.json"
    y_eval = pd.read_parquet(eval_labels_path).iloc[:, 0]

    from src.interpretability.utils.kan_coefficients import coefficient_importance_from_module

    ranking = coefficient_importance_from_module(pruned_module, feature_names)
    kan_ranked = ranking.index.tolist() if not ranking.empty else feature_names
    (
        ranking.rename("importance")
        .rename_axis("feature")
        .reset_index()
        .to_csv(ranking_path, index=False)
    )
    top20_path.write_text(json.dumps(kan_ranked[:20], indent=2))
    top12_path.write_text(json.dumps(kan_ranked[:12], indent=2))

    # Formula composition (SymPy)
    from src.interpretability.formula_composition import run as run_formula_composition
    exact_report = run_formula_composition(
        symbolic_fits_path, pruned_module, feature_names,
        interpret_dir, flavor, X_eval=X_eval,
    )

    # Local case explanations for the active feature set.
    from src.interpretability.local_case_explanations import run as run_local_case_explanations

    case_features = restricted_features if restricted_features is not None else kan_ranked[: min(20, len(kan_ranked))]
    case_artifacts = run_local_case_explanations(
        pruned_module,
        X_eval,
        output_dir=interpret_dir,
        flavor=flavor,
        feature_types=feat_types,
        X_eval_raw=X_raw,
        candidate_features=case_features,
        row_position=0,
        ordinal_calibration=ordinal_calibration,
    )

    surrogate_artifacts = None
    if not exact_report.get("exact_available", False):
        from src.interpretability.closed_form_surrogate import run as run_closed_form_surrogate

        surrogate_features = restricted_features if restricted_features is not None else kan_ranked[:20]
        surrogate_artifacts = run_closed_form_surrogate(
            pruned_module,
            X_eval,
            output_dir=interpret_dir,
            feature_names=surrogate_features,
            y_eval=y_eval,
            flavor=flavor,
            ordinal_calibration=ordinal_calibration,
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

    # Partial dependence plots (input → output effect)
    from src.interpretability.partial_dependence import plot_partial_dependence
    from src.interpretability.utils.kan_coefficients import (
        top_features_by_coefficients,
    )
    pdp_n = max_features or 20
    pdp_feats = top_features_by_coefficients(pruned_module, feature_names, top_n=pdp_n)
    plot_partial_dependence(
        pruned_module, X_eval, pdp_feats, interpret_dir, flavor,
        X_raw=X_raw, feat_types=feat_types,
        preprocessing_recipe=recipe,
    )

    # Feature subset validation (TabKAN Section 5.7)
    from src.interpretability.feature_validation import (
        compute_feature_validation_curves,
        plot_feature_validation_curves,
    )

    import torch
    import numpy as np

    def _kan_predict(X_df):
        X_t = torch.tensor(X_df.values, dtype=torch.float32)
        with torch.no_grad():
            preds = pruned_module(X_t).cpu().numpy().flatten()
        return classes_from_scores(preds, ordinal_calibration)

    curves = compute_feature_validation_curves(
        {flavor: kan_ranked},
        {flavor: _kan_predict},
        X_eval, y_eval,
    )
    plot_feature_validation_curves(curves, interpret_dir)
    curves_path = data_dir(interpret_dir) / f"{flavor}_feature_validation_curves.json"
    import json as _json
    curves_path.write_text(_json.dumps(curves, indent=2))
    print(f"Saved feature validation curves -> {curves_path}")

    result["artifacts"] = {
        "pruning_summary": pruning_summary_path,
        "pruned_checkpoint": pruned_checkpoint_path,
        "symbolic_fits": symbolic_fits_path,
        "r2_report": reports_dir(interpret_dir) / f"{flavor}_r2_report.json",
        "symbolic_formulas": reports_dir(interpret_dir) / f"{flavor}_symbolic_formulas.json",
        "exact_closed_form": reports_dir(interpret_dir) / f"{flavor}_exact_closed_form.json",
        "feature_ranking": ranking_path,
        "top20_features": top20_path,
        "top12_features": top12_path,
        "case_summary": case_artifacts["case_summary"],
        "local_sensitivities": case_artifacts["local_sensitivities"],
        "case_what_if": case_artifacts["what_if"],
        "kan_diagram": Path("figures") / f"{flavor}_kan_diagram.pdf",
        "r2_distribution": Path("figures") / f"{flavor}_r2_distribution.pdf",
        "feature_validation": curves_path,
    }
    if surrogate_artifacts is not None:
        result["artifacts"]["closed_form_surrogate"] = surrogate_artifacts["json_path"]
    return result
