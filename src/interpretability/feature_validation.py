"""Feature subset validation: QWK vs. number of top-ranked features retained.

Implements the validation methodology from TabKAN (arXiv:2504.06559v3,
Figures 6-7): for each model's native feature ranking, evaluate performance
when only the top-k features are retained (others zeroed out).

This validates that the importance rankings are meaningful — a good ranking
should lose minimal performance when dropping low-importance features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_feature_validation_curves(
    rankings: dict[str, list[str]],
    model_predict: dict[str, object],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    retention_steps: list[int] | None = None,
) -> dict[str, list[dict]]:
    """Evaluate each model at multiple feature retention levels.

    Parameters
    ----------
    rankings : dict[str, list[str]]
        Model name -> ordered list of feature names (most important first).
    model_predict : dict[str, callable]
        Model name -> predict(X_df) -> np.ndarray of predictions.
    X_eval : DataFrame
        Evaluation features (encoded).
    y_eval : Series
        Ground truth labels.
    retention_steps : list[int], optional
        Number of features to retain at each step.
        Defaults to [5, 10, 15, 20, 30, 50, all].

    Returns
    -------
    dict[str, list[dict]] with keys per model, each containing:
        [{"n_features": k, "qwk": score, "pct_features": fraction}, ...]
    """
    from sklearn.metrics import cohen_kappa_score

    n_total = X_eval.shape[1]
    if retention_steps is None:
        retention_steps = sorted(set([5, 10, 15, 20, 30, 50, n_total]))
        retention_steps = [k for k in retention_steps if k <= n_total]

    def qwk(y_true, y_pred):
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    def mask_features(X: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
        X_masked = X.copy()
        for col in X_masked.columns:
            if col not in keep:
                X_masked[col] = 0.0
        return X_masked

    curves: dict[str, list[dict]] = {}

    for model_name, ranked in rankings.items():
        predict_fn = model_predict.get(model_name)
        if predict_fn is None:
            continue

        model_curve = []
        for k in retention_steps:
            keep = ranked[:k]
            X_masked = mask_features(X_eval, keep)
            try:
                preds = predict_fn(X_masked)
                score = qwk(y_eval, preds)
            except Exception:
                score = float("nan")

            model_curve.append({
                "n_features": k,
                "pct_features": round(k / n_total, 4),
                "qwk": round(score, 6),
            })
        curves[model_name] = model_curve

    return curves


def plot_feature_validation_curves(
    curves: dict[str, list[dict]],
    output_dir: Path,
    *,
    filename: str = "feature_validation_curves.pdf",
) -> Path:
    """Plot QWK vs. features retained for all models.

    Follows TabKAN Figures 6-7 style: x-axis is number (or %) of features,
    y-axis is QWK, one line per model with native importance ranking.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, points in curves.items():
        n_feats = [p["n_features"] for p in points]
        pcts = [p["pct_features"] * 100 for p in points]
        qwks = [p["qwk"] for p in points]
        color = MODEL_COLORS.get(model_name, "#333333")

        # Left: absolute number of features
        ax1.plot(n_feats, qwks, color=color, lw=2, marker="o", ms=5, label=model_name)
        # Right: percentage
        ax2.plot(pcts, qwks, color=color, lw=2, marker="o", ms=5, label=model_name)

    ax1.set_xlabel("Number of features retained", fontsize=9)
    ax1.set_ylabel("QWK", fontsize=9)
    ax1.set_title("QWK vs. Features Retained (absolute)", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")

    ax2.set_xlabel("Features retained (%)", fontsize=9)
    ax2.set_ylabel("QWK", fontsize=9)
    ax2.set_title("QWK vs. Features Retained (percentage)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    for ax in [ax1, ax2]:
        ax.text(0.5, -0.12,
                "Note: unretained features set to 0.0 (median in encoded space), not removed.",
                ha="center", transform=ax.transAxes, fontsize=6, color="gray")

    plt.suptitle("Feature Importance Validation (TabKAN §5.7 methodology)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = fig_dir(output_dir) / filename
    savefig_pdf(fig, out)
    print(f"Saved -> {out}")
    plt.close()
    return out


def plot_ranking_comparison(
    rankings: dict[str, list[str]],
    output_dir: Path,
    *,
    top_n: int = 20,
    filename: str = "feature_ranking_comparison.pdf",
) -> Path:
    """Side-by-side comparison of top-N feature rankings across models.

    Shows which features appear in multiple models' top rankings,
    highlighting consensus features.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()

    # Collect all features appearing in any model's top_n
    all_top = set()
    for ranked in rankings.values():
        all_top.update(ranked[:top_n])

    # Count how many models include each feature in their top_n
    consensus_count = {}
    for feat in all_top:
        count = sum(1 for ranked in rankings.values() if feat in ranked[:top_n])
        consensus_count[feat] = count

    n_models = len(rankings)
    model_names = list(rankings.keys())

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        top_feats = rankings[model_name][:top_n]
        color = MODEL_COLORS.get(model_name, "#333333")

        # Color bars by consensus
        bar_colors = []
        for f in top_feats:
            c = consensus_count.get(f, 1)
            if c == n_models:
                bar_colors.append("#2ECC71")  # all models agree
            elif c >= n_models - 1:
                bar_colors.append("#F39C12")  # most agree
            else:
                bar_colors.append(color)

        ax.barh(range(len(top_feats)), range(len(top_feats), 0, -1),
                color=bar_colors, alpha=0.8)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([f[:18] for f in top_feats], fontsize=6)
        ax.set_title(model_name, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Rank (higher = more important)", fontsize=7)

    legend_elements = [
        Patch(facecolor="#2ECC71", label=f"In all {n_models} models"),
        Patch(facecolor="#F39C12", label=f"In {n_models-1}+ models"),
        Patch(facecolor="#AAAAAA", label="Model-specific"),
    ]
    fig.legend(handles=legend_elements, fontsize=7, loc="lower center", ncol=3)

    plt.suptitle(f"Top-{top_n} Feature Rankings by Model",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = fig_dir(output_dir) / filename
    savefig_pdf(fig, out)
    print(f"Saved -> {out}")
    plt.close()
    return out


def run(
    rankings: dict[str, list[str]],
    model_predict: dict[str, object],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    output_dir: Path,
    *,
    retention_steps: list[int] | None = None,
) -> dict:
    """Run feature validation pipeline: compute curves, save data, plot figures."""
    from src.interpretability.utils.paths import data as data_dir, reports as rep_dir
    import json

    curves = compute_feature_validation_curves(
        rankings, model_predict, X_eval, y_eval,
        retention_steps=retention_steps,
    )

    # Save raw curve data
    curves_path = data_dir(output_dir) / "feature_validation_curves.json"
    curves_path.write_text(json.dumps(curves, indent=2))
    print(f"Saved curve data -> {curves_path}")

    # Plot curves
    plot_feature_validation_curves(curves, output_dir)

    # Plot ranking comparison
    plot_ranking_comparison(rankings, output_dir)

    # Summary report
    summary = {}
    for model_name, points in curves.items():
        full_qwk = points[-1]["qwk"] if points else float("nan")
        half_idx = len(points) // 2
        half_qwk = points[half_idx]["qwk"] if points else float("nan")
        summary[model_name] = {
            "full_qwk": full_qwk,
            "half_features_qwk": half_qwk,
            "delta": round(full_qwk - half_qwk, 6),
            "auc": round(float(np.trapezoid(
                [p["qwk"] for p in points],
                [p["pct_features"] for p in points],
            )), 6) if len(points) > 1 else float("nan"),
        }

    report_path = rep_dir(output_dir) / "feature_validation_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {report_path}")

    return {"curves": curves, "summary": summary}
