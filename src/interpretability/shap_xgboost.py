"""Issue 02 — SHAP explanations for XGBoost.

Generates a global beeswarm summary plot, per-risk-level dependence
plots for the top-5 features, and exports raw SHAP values.

Usage:
    uv run python -m src.interpretability.shap_xgboost \
        --checkpoint checkpoints/xgb-baseline/model-<timestamp>.joblib \
        --eval-features outputs/X_eval.parquet \
        --eval-labels  outputs/y_eval.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_model(checkpoint_path: Path):
    import joblib
    wrapper = joblib.load(checkpoint_path)
    return wrapper.model  # XGBBaseline.model → xgb.XGBRegressor


def _top_features(shap_values: np.ndarray, feature_names: list[str], n: int = 5) -> list[str]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return [feature_names[i] for i in idx]


def _axis_label(feat: str, feat_types: dict) -> str:
    """Human-readable x-axis label, annotating encoded features."""
    ftype = feat_types.get(feat, "unknown")
    if ftype == "categorical":
        return f"{feat[:20]} (CatBoost-encoded)"
    if ftype == "continuous":
        return f"{feat[:20]} (original scale)"
    if ftype == "missing_indicator":
        return f"{feat[:20]} (0=present, 1=missing)"
    return feat[:20]


def run(
    checkpoint_path: Path,
    eval_features_path: Path,
    eval_labels_path: Path,
    output_dir: Path = Path("outputs"),
    eval_features_raw_path: Path | None = None,
) -> None:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.interpretability.paths import figures as fig_dir, data as data_dir, reports as rep_dir

    xgb_model = _load_model(checkpoint_path)
    X_eval = pd.read_parquet(eval_features_path)
    y_eval = pd.read_parquet(eval_labels_path)["Response"]

    # Load raw (pre-preprocessing) features for original-scale x-axes
    raw_path = eval_features_raw_path or (data_dir(output_dir) / "X_eval_raw.parquet")
    X_eval_raw: pd.DataFrame | None = None
    if raw_path.exists():
        X_eval_raw = pd.read_parquet(raw_path).reset_index(drop=True)

    # Load feature type metadata if available
    feat_types_path = rep_dir(output_dir) / "feature_types.json"
    feat_types: dict = {}
    if feat_types_path.exists():
        import json
        feat_types = json.loads(feat_types_path.read_text())

    print(f"Computing SHAP values for {len(X_eval)} eval samples …")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_eval)  # (n_samples, n_features)

    # ── Global beeswarm ──────────────────────────────────────────────────────
    # Use raw values for beeswarm color/position when available
    X_for_plot = X_eval_raw[X_eval.columns].copy() if X_eval_raw is not None and all(
        c in X_eval_raw.columns for c in X_eval.columns
    ) else X_eval
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_for_plot, show=False, plot_size=None)
    plt.tight_layout()
    beeswarm_path = fig_dir(output_dir) / "shap_xgb_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved beeswarm plot → {beeswarm_path}")

    # ── Per-risk-level dependence plots ──────────────────────────────────────
    import joblib
    wrapper = joblib.load(checkpoint_path)
    y_pred_ord = wrapper.predict(X_eval)

    top5 = _top_features(shap_values, list(X_eval.columns), n=5)
    feature_idx = {f: i for i, f in enumerate(X_eval.columns)}

    for feat in top5:
        # Use original feature values on x-axis when available
        x_source = X_eval_raw if (X_eval_raw is not None and feat in X_eval_raw.columns) else X_eval
        xlabel = _axis_label(feat, feat_types)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
        axes = axes.flatten()
        for risk_level in range(1, 9):
            ax = axes[risk_level - 1]
            mask = y_pred_ord == risk_level
            if mask.sum() < 5:
                ax.set_title(f"Risk {risk_level} (n={mask.sum()})")
                ax.axis("off")
                continue
            fidx = feature_idx[feat]
            ax.scatter(
                x_source.loc[mask, feat] if hasattr(x_source, "loc") else x_source[mask, fidx],
                shap_values[mask, fidx],
                alpha=0.4,
                s=8,
                c=shap_values[mask, fidx],
                cmap="coolwarm",
            )
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(f"Risk {risk_level} (n={mask.sum()})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("SHAP value" if risk_level in (1, 5) else "")
        fig.suptitle(f"SHAP dependence — {feat}", fontsize=12)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        dep_path = fig_dir(output_dir) / f"shap_xgb_dependence_{safe_name}.png"
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved dependence plot → {dep_path}")

    # ── Export raw SHAP values ────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=X_eval.columns)
    parquet_path = data_dir(output_dir) / "shap_xgb_values.parquet"
    shap_df.to_parquet(parquet_path, index=False)
    print(f"Saved raw SHAP values ({shap_df.shape}) → {parquet_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP explanations for XGBoost")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--eval-labels", type=Path, default=Path("outputs/data/y_eval.parquet"))
    p.add_argument("--eval-features-raw", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.checkpoint, args.eval_features, args.eval_labels, args.output_dir, args.eval_features_raw)
