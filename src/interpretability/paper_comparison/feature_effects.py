"""Cross-run feature ranking and effect comparison for the paper.

This module compares XGBoost predicted-class SHAP values against ChebyKAN and
FourierKAN native partial-dependence curves for the same selected features.
It is intentionally separate from ``src.interpretability.pipeline`` because it
requires artifacts from multiple already-interpreted runs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.interpretability.utils.paths import data as data_dir
from src.interpretability.utils.paths import figures as figures_dir
from src.interpretability.utils.paths import reports as reports_dir


DEFAULT_XGB_DIR = Path("outputs/interpretability/xgboost_paper/stage-c-xgboost-best")
DEFAULT_CHEBY_DIR = Path(
    "outputs/interpretability/kan_paper/"
    "stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94"
)
DEFAULT_FOURIER_DIR = Path(
    "outputs/interpretability/kan_paper/"
    "stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76"
)
DEFAULT_XGB_EVAL_DIR = Path("outputs/eval/xgboost_paper/stage-c-xgboost-best")
DEFAULT_CHEBY_EVAL_DIR = Path(
    "outputs/eval/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94"
)
DEFAULT_FOURIER_EVAL_DIR = Path(
    "outputs/eval/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76"
)
DEFAULT_OUTPUT_DIR = Path("outputs/interpretability/comparison/pareto_kan_vs_xgboost")

MODEL_LABELS = {
    "xgboost": "XGBoost SHAP",
    "chebykan": "ChebyKAN",
    "fourierkan": "FourierKAN",
}


@dataclass(frozen=True)
class RunPaths:
    """Resolved artifact directories for one comparison run."""

    xgb_dir: Path
    cheby_dir: Path
    fourier_dir: Path
    xgb_eval_dir: Path
    cheby_eval_dir: Path
    fourier_eval_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class KanArtifacts:
    """Loaded KAN module and matching evaluation data."""

    flavor: str
    interpret_dir: Path
    eval_dir: Path
    module: Any
    X_eval: pd.DataFrame
    X_raw: pd.DataFrame | None
    feature_types: dict[str, str]
    ranking: pd.Series
    pruning_summary: dict[str, Any]
    r2_report: dict[str, Any]
    run_summary: dict[str, Any]


def _require(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{purpose} not found at {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_require(path, "JSON artifact").read_text())


def _latest_run_summary(experiment_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((Path("artifacts") / experiment_name).glob("run-summary-*.json"))
    if not candidates:
        raise FileNotFoundError(f"No run summary found for experiment '{experiment_name}'.")
    path = candidates[-1]
    return path, json.loads(path.read_text())


def _load_eval_frame(eval_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_parquet(_require(eval_dir / filename, f"{filename} eval artifact"))


def _load_feature_types(*eval_dirs: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    for eval_dir in eval_dirs:
        path = eval_dir / "feature_types.json"
        if path.exists():
            merged.update(json.loads(path.read_text()))
    return merged


def _load_xgb_ranking(xgb_dir: Path) -> pd.Series:
    shap_path = _require(xgb_dir / "data" / "shap_xgb_values.parquet", "XGBoost SHAP values")
    shap_df = pd.read_parquet(shap_path)
    return shap_df.abs().mean().sort_values(ascending=False).rename("importance")


def _load_kan_ranking(interpret_dir: Path, flavor: str) -> pd.Series:
    path = _require(
        interpret_dir / "data" / f"{flavor}_feature_ranking.csv",
        f"{flavor} feature ranking",
    )
    df = pd.read_csv(path)
    if not {"feature", "importance"}.issubset(df.columns):
        raise ValueError(f"Unexpected ranking schema in {path}.")
    return df.set_index("feature")["importance"].sort_values(ascending=False)


def _load_pruned_kan(
    *,
    interpret_dir: Path,
    eval_dir: Path,
    flavor: str,
) -> KanArtifacts:
    import torch
    from src.models.tabkan import TabKAN

    experiment_name = interpret_dir.name
    _, run_summary = _latest_run_summary(experiment_name)
    config = ExperimentConfig.model_validate(run_summary["config"])

    X_eval = _load_eval_frame(eval_dir, "X_eval.parquet")
    X_raw = (
        pd.read_parquet(eval_dir / "X_eval_raw.parquet")
        if (eval_dir / "X_eval_raw.parquet").exists()
        else None
    )
    feature_types = (
        json.loads((eval_dir / "feature_types.json").read_text())
        if (eval_dir / "feature_types.json").exists()
        else {}
    )

    params = config.model.params
    module = TabKAN(
        in_features=X_eval.shape[1],
        widths=config.model.resolved_hidden_widths(),
        kan_type=flavor,
        degree=config.model.degree or 3,
        grid_size=params.get("grid_size", 4),
        spline_order=params.get("spline_order", 3),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-5),
        sparsity_lambda=params.get("sparsity_lambda", 0.0),
        l1_weight=params.get("l1_weight", 1.0),
        entropy_weight=params.get("entropy_weight", 1.0),
        use_layernorm=config.model.use_layernorm,
    )
    state_path = _require(
        interpret_dir / "models" / f"{flavor}_pruned_module.pt",
        f"{flavor} pruned module",
    )
    module.load_state_dict(torch.load(state_path, map_location="cpu"))
    module.eval()

    return KanArtifacts(
        flavor=flavor,
        interpret_dir=interpret_dir,
        eval_dir=eval_dir,
        module=module,
        X_eval=X_eval,
        X_raw=X_raw,
        feature_types=feature_types,
        ranking=_load_kan_ranking(interpret_dir, flavor),
        pruning_summary=_read_json(interpret_dir / "reports" / f"{flavor}_pruning_summary.json"),
        r2_report=_read_json(interpret_dir / "reports" / f"{flavor}_r2_report.json"),
        run_summary=run_summary,
    )


def _rank_map(ranking: pd.Series) -> dict[str, int]:
    return {feature: rank for rank, feature in enumerate(ranking.index.tolist(), start=1)}


def _rank_scaled_overlap(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    top_n: int,
) -> dict[str, Any]:
    baseline_top = baseline.index.tolist()[:top_n]
    candidate_top = candidate.index.tolist()[:top_n]
    baseline_rank = _rank_map(baseline)
    candidate_rank = _rank_map(candidate)
    shared = [feature for feature in baseline_top if feature in set(candidate_top)]
    rows = []
    total_score = 0.0
    for feature in shared:
        score = ((top_n + 1 - baseline_rank[feature]) / top_n) * (
            (top_n + 1 - candidate_rank[feature]) / top_n
        )
        total_score += score
        rows.append(
            {
                "feature": feature,
                "xgb_rank": baseline_rank[feature],
                "kan_rank": candidate_rank[feature],
                "score": round(float(score), 6),
            }
        )
    return {
        "shared_count": len(shared),
        "rank_scaled_score": round(float(total_score), 6),
        "shared_features": rows,
    }


def _kendall_tau_top_union(
    left: pd.Series,
    right: pd.Series,
    *,
    top_n: int,
) -> float:
    """Kendall-like tau over the union of two top-N lists.

    Missing features are tied at rank ``top_n + 1``. This avoids depending on
    scipy and is enough for a compact paper diagnostic.
    """
    left_top = left.index.tolist()[:top_n]
    right_top = right.index.tolist()[:top_n]
    items = list(dict.fromkeys(left_top + right_top))
    left_rank = {feature: rank for rank, feature in enumerate(left_top, start=1)}
    right_rank = {feature: rank for rank, feature in enumerate(right_top, start=1)}
    missing = top_n + 1
    concordant = discordant = ties_left = ties_right = 0

    for i, first in enumerate(items):
        for second in items[i + 1:]:
            left_delta = (left_rank.get(first, missing) > left_rank.get(second, missing)) - (
                left_rank.get(first, missing) < left_rank.get(second, missing)
            )
            right_delta = (right_rank.get(first, missing) > right_rank.get(second, missing)) - (
                right_rank.get(first, missing) < right_rank.get(second, missing)
            )
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                ties_left += 1
            elif right_delta == 0:
                ties_right += 1
            elif left_delta == right_delta:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    if denominator == 0:
        return float("nan")
    return round(float((concordant - discordant) / denominator), 6)


def build_ranking_comparison(
    *,
    xgb_ranking: pd.Series,
    cheby_ranking: pd.Series,
    fourier_ranking: pd.Series,
    feature_types: dict[str, str],
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the ranking table and summary payload."""
    rankings = {
        "xgboost": xgb_ranking,
        "chebykan": cheby_ranking,
        "fourierkan": fourier_ranking,
    }
    rank_maps = {name: _rank_map(ranking) for name, ranking in rankings.items()}
    top_sets = {name: set(ranking.index.tolist()[:top_n]) for name, ranking in rankings.items()}
    union_top = sorted(
        set().union(*top_sets.values()),
        key=lambda feature: (
            -sum(feature in top_set for top_set in top_sets.values()),
            rank_maps["xgboost"].get(feature, 10_000),
            rank_maps["chebykan"].get(feature, 10_000),
            rank_maps["fourierkan"].get(feature, 10_000),
            feature,
        ),
    )

    rows = []
    for feature in union_top:
        row: dict[str, Any] = {
            "feature": feature,
            "feature_type": feature_types.get(feature, "unknown"),
            "shared_top_n_count": sum(feature in top_set for top_set in top_sets.values()),
        }
        for name, ranking in rankings.items():
            row[f"{name}_rank"] = rank_maps[name].get(feature)
            row[f"{name}_importance"] = (
                float(ranking.loc[feature]) if feature in ranking.index else np.nan
            )
            row[f"{name}_in_top_{top_n}"] = feature in top_sets[name]
        rows.append(row)

    all_three = [
        feature
        for feature in xgb_ranking.index.tolist()[:top_n]
        if feature in top_sets["chebykan"] and feature in top_sets["fourierkan"]
    ]
    summary = {
        "top_n": top_n,
        "xgboost_top_features": xgb_ranking.index.tolist()[:top_n],
        "chebykan_top_features": cheby_ranking.index.tolist()[:top_n],
        "fourierkan_top_features": fourier_ranking.index.tolist()[:top_n],
        "shared_all_three_count": len(all_three),
        "shared_all_three_features": all_three,
        "chebykan_vs_xgboost": _rank_scaled_overlap(
            xgb_ranking, cheby_ranking, top_n=top_n
        )
        | {"kendall_tau_top_union": _kendall_tau_top_union(xgb_ranking, cheby_ranking, top_n=top_n)},
        "fourierkan_vs_xgboost": _rank_scaled_overlap(
            xgb_ranking, fourier_ranking, top_n=top_n
        )
        | {
            "kendall_tau_top_union": _kendall_tau_top_union(
                xgb_ranking, fourier_ranking, top_n=top_n
            )
        },
    }
    return pd.DataFrame(rows), summary


def select_features_for_effect_plot(
    *,
    xgb_ranking: pd.Series,
    cheby_ranking: pd.Series,
    fourier_ranking: pd.Series,
    feature_types: dict[str, str],
    available_features: set[str],
    n_features: int,
    pool_n: int = 40,
) -> list[str]:
    """Select high-overlap features with a preference for readable diversity."""
    rankings = [xgb_ranking.index.tolist(), cheby_ranking.index.tolist(), fourier_ranking.index.tolist()]
    candidates = [
        feature
        for feature in rankings[0][:pool_n]
        if feature in available_features
        and feature in set(rankings[1][:pool_n])
        and feature in set(rankings[2][:pool_n])
    ]

    def score(feature: str) -> float:
        value = 0.0
        for ranked in rankings:
            if feature in ranked[:pool_n]:
                value += (pool_n + 1 - (ranked.index(feature) + 1)) / pool_n
        return value

    ranked_candidates = sorted(candidates, key=lambda f: (-score(f), f))
    if len(ranked_candidates) <= n_features:
        return ranked_candidates

    selected: list[str] = []
    continuous = [f for f in ranked_candidates if feature_types.get(f) in {"continuous", "ordinal"}]
    noncontinuous = [f for f in ranked_candidates if f not in continuous]

    for feature in continuous[: max(1, n_features - 1)]:
        selected.append(feature)
        if len(selected) == n_features:
            return selected
    if noncontinuous and len(selected) < n_features:
        selected.append(noncontinuous[0])
    for feature in ranked_candidates:
        if len(selected) == n_features:
            break
        if feature not in selected:
            selected.append(feature)
    return selected


def _feature_label(feature: str, feature_types: dict[str, str]) -> str:
    marker = {
        "continuous": "C",
        "binary": "B",
        "categorical": "K",
        "ordinal": "O",
        "missing_indicator": "M",
    }.get(feature_types.get(feature))
    return f"{feature} [{marker}]" if marker else feature


def _binned_mean_line(x_values: np.ndarray, y_values: np.ndarray, *, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty or frame["x"].nunique() < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    try:
        grouped = frame.groupby(pd.qcut(frame["x"], q=min(bins, frame["x"].nunique()), duplicates="drop"))
    except ValueError:
        grouped = frame.groupby(pd.cut(frame["x"], bins=min(bins, frame["x"].nunique())))
    centers = []
    means = []
    for interval, group in grouped:
        if group.empty:
            continue
        centers.append(float(group["x"].mean()))
        means.append(float(group["y"].mean()))
    return np.asarray(centers, dtype=float), np.asarray(means, dtype=float)


def _plot_shap_effect(
    *,
    ax,
    feature: str,
    shap_df: pd.DataFrame,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame | None,
    feature_types: dict[str, str],
    preprocessing_recipe: str,
    show_ylabel: bool,
) -> None:
    from src.interpretability.utils.style import (
        MODEL_COLORS,
        build_feature_grid,
        discrete_feature_ticks,
        display_feature_values,
        feature_axis_label,
        resolve_feature_display_spec,
    )

    spec = resolve_feature_display_spec(
        feature,
        feat_types=feature_types,
        preprocessing_recipe=preprocessing_recipe,
    )
    if feature not in shap_df.columns or feature not in X_eval.columns:
        ax.text(0.5, 0.5, "Feature unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    shap_values = shap_df[feature].to_numpy(dtype=float, copy=False)
    encoded = X_eval[feature].to_numpy(dtype=float, copy=False)
    x_plot, use_raw_axis = display_feature_values(spec, X_eval, X_raw, encoded)

    if spec.model_input_kind == "discrete":
        states = build_feature_grid(spec, X_eval, percentile_range=None)
        tick_positions, tick_labels = discrete_feature_ticks(spec, X_eval, X_raw)
        rng = np.random.default_rng(42)
        min_gap = np.min(np.diff(states)) if len(states) > 1 else 1.0
        jitter = min(0.08 * float(min_gap), 0.08)
        x_jittered = x_plot + rng.normal(0.0, jitter, size=len(x_plot))
        ax.scatter(
            x_jittered,
            shap_values,
            alpha=0.18,
            s=8,
            color=MODEL_COLORS["XGBoost"],
            rasterized=True,
        )
        means = []
        for state in states:
            mask = np.isclose(encoded, state, atol=1e-8, rtol=0.0)
            means.append(float(np.mean(shap_values[mask])) if mask.any() else np.nan)
        ax.plot(states, means, color="black", lw=1.6, marker="o", label="state mean")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Observed state", fontsize=8)
    else:
        ax.scatter(
            x_plot,
            shap_values,
            alpha=0.12,
            s=7,
            color=MODEL_COLORS["XGBoost"],
            rasterized=True,
        )
        centers, means = _binned_mean_line(x_plot, shap_values)
        if centers.size:
            order = np.argsort(centers)
            ax.plot(centers[order], means[order], color="black", lw=1.8, label="binned mean")
        ax.set_xlabel(feature_axis_label(spec, use_raw_axis=use_raw_axis), fontsize=8)

    ax.axhline(0.0, color="0.45", lw=0.6, ls=":")
    if show_ylabel:
        ax.set_ylabel("SHAP value\n(predicted class)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.text(
        0.02,
        0.95,
        "XGBoost",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=MODEL_COLORS["XGBoost"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def _plot_kan_pdp(
    *,
    ax,
    artifacts: KanArtifacts,
    feature: str,
    show_ylabel: bool,
) -> None:
    from src.interpretability.partial_dependence import compute_partial_dependence
    from src.interpretability.utils.style import (
        MODEL_COLORS,
        build_feature_grid,
        discrete_feature_ticks,
        display_feature_values,
        feature_axis_label,
        resolve_feature_display_spec,
    )

    label = MODEL_LABELS[artifacts.flavor]
    color = MODEL_COLORS["ChebyKAN"] if artifacts.flavor == "chebykan" else MODEL_COLORS["FourierKAN"]
    X_eval = artifacts.X_eval
    if feature not in X_eval.columns:
        ax.text(0.5, 0.5, "Feature unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    preprocessing_recipe = artifacts.run_summary.get("preprocessing", {}).get("recipe")
    spec = resolve_feature_display_spec(
        feature,
        feat_types=artifacts.feature_types,
        preprocessing_recipe=preprocessing_recipe,
    )
    grid_model = build_feature_grid(
        spec,
        X_eval,
        grid_resolution=80,
        percentile_range=(1.0, 99.0),
    )
    if grid_model.size == 0:
        ax.text(0.5, 0.5, "Feature unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    grid_model, avg_pred = compute_partial_dependence(
        artifacts.module,
        X_eval,
        feature,
        grid_values=grid_model,
    )
    grid_plot, use_raw_axis = display_feature_values(spec, X_eval, artifacts.X_raw, grid_model)

    if spec.model_input_kind == "discrete":
        ax.plot(grid_plot, avg_pred, color=color, lw=2.0, marker="o")
        tick_positions, tick_labels = discrete_feature_ticks(spec, X_eval, artifacts.X_raw)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Observed state", fontsize=8)
    else:
        ax.plot(grid_plot, avg_pred, color=color, lw=2.2)
        ax.set_xlabel(feature_axis_label(spec, use_raw_axis=use_raw_axis), fontsize=8)

    if show_ylabel:
        ax.set_ylabel("Avg predicted\nrisk score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=color,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def plot_feature_effect_comparison(
    *,
    features: list[str],
    shap_df: pd.DataFrame,
    xgb_eval: pd.DataFrame,
    xgb_raw: pd.DataFrame | None,
    feature_types: dict[str, str],
    cheby: KanArtifacts,
    fourier: KanArtifacts,
    output_dir: Path,
) -> Path:
    """Render SHAP-vs-KAN PDP panels for selected features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf

    apply_paper_style()
    n_features = len(features)
    fig, axes = plt.subplots(
        3,
        n_features,
        figsize=(3.4 * n_features, 8.2),
        constrained_layout=True,
    )
    if n_features == 1:
        axes = axes.reshape(3, 1)

    for col, feature in enumerate(features):
        axes[0, col].set_title(_feature_label(feature, feature_types), fontsize=9, fontweight="bold")
        _plot_shap_effect(
            ax=axes[0, col],
            feature=feature,
            shap_df=shap_df,
            X_eval=xgb_eval,
            X_raw=xgb_raw,
            feature_types=feature_types,
            preprocessing_recipe="xgboost_paper",
            show_ylabel=col == 0,
        )
        _plot_kan_pdp(ax=axes[1, col], artifacts=cheby, feature=feature, show_ylabel=col == 0)
        _plot_kan_pdp(ax=axes[2, col], artifacts=fourier, feature=feature, show_ylabel=col == 0)

    fig.suptitle(
        "Feature-Level Explanation Comparison: XGBoost SHAP vs. Native KAN PDP",
        fontsize=12,
        fontweight="bold",
    )
    out = figures_dir(output_dir) / "feature_effect_comparison.pdf"
    savefig_pdf(fig, out)
    png_out = figures_dir(output_dir) / "feature_effect_comparison.png"
    fig.savefig(png_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _metrics_from_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "qwk": payload.get("metrics", {}).get("qwk"),
        "accuracy": payload.get("metrics", {}).get("accuracy"),
        "feature_count": payload.get("preprocessing", {}).get("feature_count"),
        "recipe": payload.get("preprocessing", {}).get("recipe"),
        "seed": payload.get("random_seed"),
    }


def _write_report(
    *,
    output_dir: Path,
    selected_features: list[str],
    ranking_summary: dict[str, Any],
    cheby: KanArtifacts,
    fourier: KanArtifacts,
    xgb_summary: dict[str, Any],
    figure_path: Path,
) -> Path:
    report_path = reports_dir(output_dir) / "feature_effect_comparison.md"
    lines = [
        "# Pareto KAN vs XGBoost Interpretability Comparison",
        "",
        "## Scope",
        "",
        "- XGBoost baseline: `stage-c-xgboost-best` with predicted-class SHAP values.",
        f"- ChebyKAN Pareto run: `{cheby.interpret_dir.name}`.",
        f"- FourierKAN Pareto run: `{fourier.interpret_dir.name}`.",
        "",
        "## Model Summary",
        "",
        "| Model | QWK | Active edges | Per-edge R2 | Explanation source |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| XGBoost | {xgb_summary.get('qwk', float('nan')):.6f} | - | - | "
            "Tree SHAP, predicted-class slice |"
        ),
        (
            f"| ChebyKAN Pareto | {cheby.pruning_summary.get('qwk_after', float('nan')):.6f} | "
            f"{cheby.pruning_summary.get('edges_after')} | "
            f"{_r2_mean(cheby.r2_report):.6f} | Native coefficient/PDP, per-edge symbolic recovery |"
        ),
        (
            f"| FourierKAN Pareto | {fourier.pruning_summary.get('qwk_after', float('nan')):.6f} | "
            f"{fourier.pruning_summary.get('edges_after')} | "
            f"{_r2_mean(fourier.r2_report):.6f} | Native coefficient/PDP, per-edge symbolic recovery |"
        ),
        "",
        "## Ranking Overlap",
        "",
        (
            f"- Shared top-{ranking_summary['top_n']} features across all three models: "
            f"{ranking_summary['shared_all_three_count']}."
        ),
        (
            f"- ChebyKAN vs XGBoost shared top-{ranking_summary['top_n']} count: "
            f"{ranking_summary['chebykan_vs_xgboost']['shared_count']}; rank-scaled score: "
            f"{ranking_summary['chebykan_vs_xgboost']['rank_scaled_score']}."
        ),
        (
            f"- FourierKAN vs XGBoost shared top-{ranking_summary['top_n']} count: "
            f"{ranking_summary['fourierkan_vs_xgboost']['shared_count']}; rank-scaled score: "
            f"{ranking_summary['fourierkan_vs_xgboost']['rank_scaled_score']}."
        ),
        "",
        "## Selected Feature-Effect Panels",
        "",
        "- " + ", ".join(f"`{feature}`" for feature in selected_features),
        "",
        f"Figure: `{figure_path.relative_to(output_dir)}`",
        "",
        "Interpretation convention:",
        "",
        "- XGBoost panels show sample-wise SHAP values for each applicant's predicted class.",
        "- KAN panels show partial dependence: the average predicted risk score when one feature is swept and all other applicant features are kept fixed.",
        "- For discrete features, KAN PDPs use observed states only.",
        "- KAN per-edge symbolic recovery is exact in these artifacts, but these LayerNorm/Fourier models are not end-to-end closed-form expressions.",
        "",
    ]
    report_path.write_text("\n".join(lines))
    return report_path


def _r2_mean(report: dict[str, Any]) -> float:
    if "summary" in report and isinstance(report["summary"], dict):
        value = report["summary"].get("mean_r2")
        if value is not None:
            return float(value)
    fits = report.get("symbolic_fits")
    if isinstance(fits, list) and fits:
        return float(np.mean([row.get("r_squared", np.nan) for row in fits]))
    return float("nan")


def run(
    *,
    xgb_dir: Path = DEFAULT_XGB_DIR,
    cheby_dir: Path = DEFAULT_CHEBY_DIR,
    fourier_dir: Path = DEFAULT_FOURIER_DIR,
    xgb_eval_dir: Path = DEFAULT_XGB_EVAL_DIR,
    cheby_eval_dir: Path = DEFAULT_CHEBY_EVAL_DIR,
    fourier_eval_dir: Path = DEFAULT_FOURIER_EVAL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    features: list[str] | None = None,
    n_features: int = 4,
    top_n: int = 20,
) -> dict[str, Path]:
    """Create paper-facing ranking and feature-effect comparison artifacts."""
    paths = RunPaths(
        xgb_dir=xgb_dir,
        cheby_dir=cheby_dir,
        fourier_dir=fourier_dir,
        xgb_eval_dir=xgb_eval_dir,
        cheby_eval_dir=cheby_eval_dir,
        fourier_eval_dir=fourier_eval_dir,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_path = _require(paths.xgb_dir / "data" / "shap_xgb_values.parquet", "XGBoost SHAP values")
    shap_df = pd.read_parquet(shap_path)
    xgb_eval = _load_eval_frame(paths.xgb_eval_dir, "X_eval.parquet")
    xgb_raw = (
        pd.read_parquet(paths.xgb_eval_dir / "X_eval_raw.parquet")
        if (paths.xgb_eval_dir / "X_eval_raw.parquet").exists()
        else None
    )
    feature_types = _load_feature_types(paths.xgb_eval_dir, paths.cheby_eval_dir, paths.fourier_eval_dir)

    xgb_ranking = _load_xgb_ranking(paths.xgb_dir)
    cheby = _load_pruned_kan(
        interpret_dir=paths.cheby_dir,
        eval_dir=paths.cheby_eval_dir,
        flavor="chebykan",
    )
    fourier = _load_pruned_kan(
        interpret_dir=paths.fourier_dir,
        eval_dir=paths.fourier_eval_dir,
        flavor="fourierkan",
    )

    ranking_df, ranking_summary = build_ranking_comparison(
        xgb_ranking=xgb_ranking,
        cheby_ranking=cheby.ranking,
        fourier_ranking=fourier.ranking,
        feature_types=feature_types,
        top_n=top_n,
    )
    ranking_path = data_dir(output_dir) / "feature_ranking_comparison.csv"
    ranking_df.to_csv(ranking_path, index=False)

    overlap_path = data_dir(output_dir) / "feature_overlap_summary.json"
    overlap_path.write_text(json.dumps(ranking_summary, indent=2))

    available = (
        set(shap_df.columns)
        & set(xgb_eval.columns)
        & set(cheby.X_eval.columns)
        & set(fourier.X_eval.columns)
    )
    selected = (
        list(features)
        if features
        else select_features_for_effect_plot(
            xgb_ranking=xgb_ranking,
            cheby_ranking=cheby.ranking,
            fourier_ranking=fourier.ranking,
            feature_types=feature_types,
            available_features=available,
            n_features=n_features,
        )
    )
    if not selected:
        raise ValueError("No shared features are available for the feature-effect comparison.")
    missing = [feature for feature in selected if feature not in available]
    if missing:
        raise ValueError(f"Selected features are not available in all three models: {missing}")
    selected_path = data_dir(output_dir) / "selected_features.json"
    selected_path.write_text(json.dumps(selected, indent=2))

    figure_path = plot_feature_effect_comparison(
        features=selected,
        shap_df=shap_df,
        xgb_eval=xgb_eval,
        xgb_raw=xgb_raw,
        feature_types=feature_types,
        cheby=cheby,
        fourier=fourier,
        output_dir=output_dir,
    )

    xgb_summary = _latest_run_summary(paths.xgb_dir.name)[1]
    metrics_path = data_dir(output_dir) / "model_summary.json"
    metrics_payload = {
        "xgboost": _metrics_from_run_summary(xgb_summary),
        "chebykan": {
            **_metrics_from_run_summary(cheby.run_summary),
            "qwk_after_pruning": cheby.pruning_summary.get("qwk_after"),
            "active_edges": cheby.pruning_summary.get("edges_after"),
            "mean_r2": _r2_mean(cheby.r2_report),
        },
        "fourierkan": {
            **_metrics_from_run_summary(fourier.run_summary),
            "qwk_after_pruning": fourier.pruning_summary.get("qwk_after"),
            "active_edges": fourier.pruning_summary.get("edges_after"),
            "mean_r2": _r2_mean(fourier.r2_report),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    report_path = _write_report(
        output_dir=output_dir,
        selected_features=selected,
        ranking_summary=ranking_summary,
        cheby=cheby,
        fourier=fourier,
        xgb_summary=metrics_payload["xgboost"],
        figure_path=figure_path,
    )

    return {
        "ranking_comparison": ranking_path,
        "overlap_summary": overlap_path,
        "selected_features": selected_path,
        "model_summary": metrics_path,
        "feature_effect_figure": figure_path,
        "report": report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-facing XGBoost SHAP vs Pareto KAN comparison artifacts."
    )
    parser.add_argument("--xgb-dir", type=Path, default=DEFAULT_XGB_DIR)
    parser.add_argument("--cheby-dir", type=Path, default=DEFAULT_CHEBY_DIR)
    parser.add_argument("--fourier-dir", type=Path, default=DEFAULT_FOURIER_DIR)
    parser.add_argument("--xgb-eval-dir", type=Path, default=DEFAULT_XGB_EVAL_DIR)
    parser.add_argument("--cheby-eval-dir", type=Path, default=DEFAULT_CHEBY_EVAL_DIR)
    parser.add_argument("--fourier-eval-dir", type=Path, default=DEFAULT_FOURIER_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = run(
        xgb_dir=args.xgb_dir,
        cheby_dir=args.cheby_dir,
        fourier_dir=args.fourier_dir,
        xgb_eval_dir=args.xgb_eval_dir,
        cheby_eval_dir=args.cheby_eval_dir,
        fourier_eval_dir=args.fourier_eval_dir,
        output_dir=args.output_dir,
        features=args.features,
        n_features=args.n_features,
        top_n=args.top_n,
    )
    print("Created paper comparison artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
