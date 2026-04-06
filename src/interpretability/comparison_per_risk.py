"""Per-risk-level feature importance comparison.

Produces a panel figure (8 subplots, one per risk class) showing
feature importance from GLM | XGBoost SHAP | ChebyKAN | FourierKAN.

The TabKAN paper defines KAN feature importance globally from first-layer
coefficient magnitudes, not per-risk. We therefore keep KAN series global and
label them as such instead of fabricating conditional KAN scores.

Usage:
    uv run python -m src.interpretability.comparison_per_risk \\
        --glm-coefficients     outputs/data/glm_coefficients.csv \\
        --shap-values          outputs/data/shap_xgb_values.parquet \\
        --chebykan-symbolic    outputs/data/chebykan_symbolic_fits.csv \\
        --fourierkan-symbolic  outputs/data/fourierkan_symbolic_fits.csv \\
        --chebykan-checkpoint  outputs/models/chebykan_pruned_module.pt \\
        --chebykan-config      configs/chebykan_experiment.yaml \\
        --xgb-checkpoint       checkpoints/xgb-baseline/model-<timestamp>.joblib \\
        --eval-features        outputs/data/X_eval.parquet \\
        --eval-labels          outputs/data/y_eval.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────


def _glm_importance(coef_path: Path) -> pd.Series:
    """Absolute coefficient magnitude, indexed by feature name."""
    df = pd.read_csv(coef_path)
    return df.set_index("feature")["abs_magnitude"]


def _shap_importance_by_risk(
    shap_path: Path,
    eval_features_path: Path,
    xgb_checkpoint_path: Path,
) -> dict[int, pd.Series]:
    """Mean |SHAP value| per feature, grouped by predicted risk level (1-8)."""
    import joblib

    shap_df = pd.read_parquet(shap_path)
    X_eval = pd.read_parquet(eval_features_path)
    wrapper = joblib.load(xgb_checkpoint_path)
    y_pred = wrapper.predict(X_eval)

    result = {}
    for risk in range(1, 9):
        mask = y_pred == risk
        if mask.sum() < 5:
            result[risk] = pd.Series(dtype=float)
            continue
        result[risk] = shap_df[mask].abs().mean()
    return result


def _kan_importance_global(
    symbolic_path: Path,
    feature_names: list[str],
    kan_layer,
) -> pd.Series:
    """Global KAN feature importance from layer-0 coefficient magnitudes."""
    if kan_layer is not None:
        try:
            from src.interpretability.utils.kan_coefficients import coefficient_importance_from_layer

            frame = coefficient_importance_from_layer(kan_layer, feature_names)
            if not frame.empty:
                return frame.set_index("feature")["importance"].sort_values(ascending=False)
        except Exception:
            pass
    return _fallback_edge_count(symbolic_path).sort_values(ascending=False)


def _fallback_edge_count(symbolic_path: Path) -> pd.Series:
    """Fallback: count active first-layer edges per input feature."""
    if not symbolic_path.exists():
        return pd.Series(dtype=float)
    sym_df = pd.read_csv(symbolic_path)
    layer0 = sym_df[sym_df["layer"] == 0]
    feat_importance: dict[str, float] = {}
    for _, row in layer0.iterrows():
        feat = row["input_feature"]
        feat_importance[feat] = feat_importance.get(feat, 0.0) + 1.0
    return pd.Series(feat_importance)


def _kan_importance_from_variance(edge_scores, feature_names: list[str]) -> pd.Series:
    """Backwards-compatible helper: sum per-edge scores over outputs by feature."""

    scores = np.asarray(edge_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D edge score matrix, got shape {scores.shape}")
    summed = scores.sum(axis=0)
    n_features = min(len(feature_names), summed.shape[0])
    return pd.Series(
        {feature_names[idx]: float(summed[idx]) for idx in range(n_features)}
    ).sort_values(ascending=False)


# ── Plot ─────────────────────────────────────────────────────────────────────

def _plot_panel(
    glm_imp: pd.Series,
    shap_by_risk: dict[int, pd.Series],
    chebykan_global: pd.Series,
    fourierkan_global: pd.Series,
    output_dir: Path,
    top_n: int = 10,
    fig_dir: Path | None = None,
    feat_types: dict | None = None,
) -> None:
    if fig_dir is None:
        fig_dir = output_dir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS, feature_type_label

    apply_paper_style()
    feat_types = feat_types or {}
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()
    colors = MODEL_COLORS

    for risk in range(1, 9):
        ax = axes[risk - 1]
        shap_risk = shap_by_risk.get(risk, pd.Series(dtype=float))
        rank_frame = pd.DataFrame(
            {
                "glm": glm_imp,
                "shap": shap_risk,
                "cheby_global": chebykan_global,
                "fourier_global": fourierkan_global,
            }
        ).fillna(0.0)
        for col in rank_frame.columns:
            col_max = float(rank_frame[col].max())
            if col_max > 0:
                rank_frame[col] = rank_frame[col] / col_max
        top_feats = rank_frame.sum(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
        labels = [feature_type_label(f, feat_types) for f in top_feats]

        y = np.arange(len(top_feats))
        width = 0.2
        sources = [
            ("GLM", glm_imp),
            ("XGB SHAP", shap_risk),
            ("ChebyKAN (global)", chebykan_global),
            ("FourierKAN (global)", fourierkan_global),
        ]

        color_map = {
            "GLM": colors["GLM"],
            "XGB SHAP": colors["XGBoost"],
            "ChebyKAN (global)": colors["ChebyKAN"],
            "FourierKAN (global)": colors["FourierKAN"],
        }

        for offset, (name, series) in enumerate(sources):
            vals = [series.get(f, 0.0) if not series.empty else 0.0 for f in top_feats]
            max_v = max(vals) if max(vals) > 0 else 1.0
            vals_norm = [v / max_v for v in vals]
            ax.barh(y + offset * width, vals_norm, width - 0.02,
                    label=name, color=color_map[name], alpha=0.85)

        ax.set_yticks(y + width * 1.5)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"Risk level {risk}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Relative importance (normalised)" if risk in (1, 5) else "")
        ax.set_xlim(0, 1.15)
        if risk == 1:
            ax.legend(fontsize=7, loc="upper right")

        # Kendall's τ: SHAP is risk-conditional; KAN is global-by-paper.
        glm_ranks = [float(glm_imp.get(f, 0)) for f in top_feats]
        shap_ranks = [float(shap_risk.get(f, 0)) if not shap_risk.empty else 0.0 for f in top_feats]
        cheby_ranks = [float(chebykan_global.get(f, 0)) if not chebykan_global.empty else 0.0 for f in top_feats]

        tau_gs, _ = kendalltau(glm_ranks, shap_ranks)
        tau_gc, _ = kendalltau(glm_ranks, cheby_ranks)
        tau_sc, _ = kendalltau(shap_ranks, cheby_ranks)
        tau_text = (
            f"τ GLM↔SHAP={tau_gs:.2f}\n"
            f"τ GLM↔Cheby(global)={tau_gc:.2f}\n"
            f"τ SHAP↔Cheby(global)={tau_sc:.2f}"
        )
        ax.text(0.98, 0.02, tau_text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5, color="gray")

    caption = (
        "KAN bars use paper-native global coefficient importance from the first layer. "
        "They are repeated in each risk panel for reference only; only SHAP is risk-conditional."
    )
    fig.text(0.5, -0.01, caption, ha="center", fontsize=6, color="gray", style="italic")
    fig.suptitle("Feature Importance by Risk Level — GLM vs XGB SHAP vs KAN Coefficients",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = fig_dir / "per_risk_level_comparison.pdf"
    savefig_pdf(fig, out_path)
    print(f"Saved → {out_path}")
    plt.close()


def run(
    glm_coef_path: Path,
    shap_path: Path,
    chebykan_symbolic_path: Path,
    fourierkan_symbolic_path: Path,
    xgb_checkpoint_path: Path,
    eval_features_path: Path,
    eval_labels_path: Path,
    output_dir: Path = Path("outputs"),
    chebykan_checkpoint_path: Path | None = None,
    chebykan_config_path: Path | None = None,
    fourierkan_checkpoint_path: Path | None = None,
    fourierkan_config_path: Path | None = None,
) -> None:
    from src.interpretability.utils.paths import figures as fig_dir, data as data_dir, reports as rep_dir

    X_eval = pd.read_parquet(eval_features_path)
    feature_names = list(X_eval.columns)

    # Load KAN layers for paper-native coefficient importance
    chebykan_layer = None
    fourierkan_layer = None

    def _load_layer(ckpt: Path | None, cfg_path: Path | None, flavor: str):
        if not (ckpt and ckpt.exists() and cfg_path and cfg_path.exists()):
            return None
        try:
            import torch
            from src.configs import load_experiment_config
            from src.models.tabkan import TabKAN
            from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

            cfg = load_experiment_config(cfg_path)
            module = TabKAN(
                in_features=X_eval.shape[1],
                widths=[cfg.model.width] * cfg.model.depth,
                kan_type=flavor,
                degree=cfg.model.degree or 3,
            )
            module.load_state_dict(torch.load(ckpt, map_location="cpu"))
            module.eval()
            return next(
                (l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))),
                None,
            )
        except Exception as e:
            print(f"Warning: could not load {flavor} model: {e}")
            return None

    chebykan_layer = _load_layer(chebykan_checkpoint_path, chebykan_config_path, "chebykan")
    fourierkan_layer = _load_layer(fourierkan_checkpoint_path, fourierkan_config_path, "fourierkan")

    feat_types: dict = {}
    ft_path = rep_dir(output_dir) / "feature_types.json"
    if ft_path.exists():
        import json
        feat_types = json.loads(ft_path.read_text())

    glm_imp = _glm_importance(glm_coef_path)
    shap_by_risk = _shap_importance_by_risk(shap_path, eval_features_path, xgb_checkpoint_path)
    chebykan_global = _kan_importance_global(chebykan_symbolic_path, feature_names, chebykan_layer)
    fourierkan_global = _kan_importance_global(fourierkan_symbolic_path, feature_names, fourierkan_layer)

    _plot_panel(glm_imp, shap_by_risk, chebykan_global, fourierkan_global,
                output_dir, fig_dir=fig_dir(output_dir), feat_types=feat_types)

    # ── Export underlying data ────────────────────────────────────────────────
    rows = []
    for risk in range(1, 9):
        for feat in glm_imp.index:
            rows.append({
                "risk_level": risk,
                "feature": feat,
                "glm_importance": glm_imp.get(feat, 0.0),
                "xgb_shap": shap_by_risk.get(risk, pd.Series()).get(feat, 0.0),
                "chebykan_global": chebykan_global.get(feat, 0.0),
                "fourierkan_global": fourierkan_global.get(feat, 0.0),
            })
    out_csv = data_dir(output_dir) / "per_risk_level_data.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved underlying data → {out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-risk-level feature importance comparison")
    p.add_argument("--glm-coefficients", type=Path, default=Path("outputs/data/glm_coefficients.csv"))
    p.add_argument("--shap-values", type=Path, default=Path("outputs/data/shap_xgb_values.parquet"))
    p.add_argument("--chebykan-symbolic", type=Path, default=Path("outputs/data/chebykan_symbolic_fits.csv"))
    p.add_argument("--fourierkan-symbolic", type=Path, default=Path("outputs/data/fourierkan_symbolic_fits.csv"))
    p.add_argument("--chebykan-checkpoint", type=Path, default=None)
    p.add_argument("--chebykan-config", type=Path, default=None)
    p.add_argument("--fourierkan-checkpoint", type=Path, default=None)
    p.add_argument("--fourierkan-config", type=Path, default=None)
    p.add_argument("--xgb-checkpoint", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--eval-labels", type=Path, default=Path("outputs/data/y_eval.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        glm_coef_path=args.glm_coefficients,
        shap_path=args.shap_values,
        chebykan_symbolic_path=args.chebykan_symbolic,
        fourierkan_symbolic_path=args.fourierkan_symbolic,
        xgb_checkpoint_path=args.xgb_checkpoint,
        eval_features_path=args.eval_features,
        eval_labels_path=args.eval_labels,
        output_dir=args.output_dir,
        chebykan_checkpoint_path=args.chebykan_checkpoint,
        chebykan_config_path=args.chebykan_config,
        fourierkan_checkpoint_path=args.fourierkan_checkpoint,
        fourierkan_config_path=args.fourierkan_config,
    )
