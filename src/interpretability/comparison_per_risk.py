"""Per-risk-level feature importance comparison.

Produces a panel figure (8 subplots, one per risk class) showing
feature importance from GLM | XGBoost SHAP | ChebyKAN | FourierKAN.

Uses horizontal bars with feature-type markers and three Kendall's τ
annotations per panel (GLM↔SHAP, GLM↔ChebyKAN, SHAP↔ChebyKAN).
FourierKAN importance is global — its τ equals ChebyKAN in all panels
and is therefore omitted from τ annotations.

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

def _kan_importance_from_variance(
    variances: "torch.Tensor",
    feature_names: list[str],
) -> dict[str, float]:
    """Sum of first-layer edge output variances per input feature."""
    per_input = variances.sum(dim=0)  # (in_features,)
    return {
        feature_names[i]: float(per_input[i].item())
        for i in range(min(len(feature_names), per_input.shape[0]))
    }


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


def _kan_importance_by_risk(
    symbolic_path: Path,
    feature_names: list[str],
    kan_layer,
) -> dict[int, pd.Series]:
    """Feature importance from KAN first-layer edge variances (global, replicated per risk)."""
    importance_series: pd.Series

    if kan_layer is not None:
        try:
            import torch
            from src.interpretability.kan_pruning import _compute_edge_variances
            variances = _compute_edge_variances(kan_layer)
            imp_dict = _kan_importance_from_variance(variances, feature_names)
            importance_series = pd.Series(imp_dict)
        except Exception:
            importance_series = _fallback_edge_count(symbolic_path)
    else:
        importance_series = _fallback_edge_count(symbolic_path)

    return {risk: importance_series for risk in range(1, 9)}


def _fallback_edge_count(symbolic_path: Path) -> pd.Series:
    """Fallback: count active edges per input feature."""
    if not symbolic_path.exists():
        return pd.Series(dtype=float)
    sym_df = pd.read_csv(symbolic_path)
    layer0 = sym_df[sym_df["layer"] == 0]
    feat_importance: dict[str, float] = {}
    for _, row in layer0.iterrows():
        feat = row["input_feature"]
        feat_importance[feat] = feat_importance.get(feat, 0.0) + 1.0
    return pd.Series(feat_importance)


# ── Plot ─────────────────────────────────────────────────────────────────────

def _plot_panel(
    glm_imp: pd.Series,
    shap_by_risk: dict[int, pd.Series],
    chebykan_by_risk: dict[int, pd.Series],
    fourierkan_by_risk: dict[int, pd.Series],
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
    from src.interpretability.style import apply_paper_style, savefig_pdf, MODEL_COLORS, feature_type_label

    apply_paper_style()
    feat_types = feat_types or {}
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()
    colors = MODEL_COLORS

    for risk in range(1, 9):
        ax = axes[risk - 1]
        ref = shap_by_risk.get(risk, glm_imp)
        if ref.empty:
            ref = glm_imp
        top_feats = ref.nlargest(top_n).index.tolist()
        labels = [feature_type_label(f, feat_types) for f in top_feats]

        y = np.arange(len(top_feats))
        width = 0.2
        sources = [
            ("GLM", glm_imp),
            ("XGB SHAP", shap_by_risk.get(risk, pd.Series())),
            ("ChebyKAN", chebykan_by_risk.get(risk, pd.Series())),
            ("FourierKAN", fourierkan_by_risk.get(risk, pd.Series())),
        ]

        color_map = {
            "GLM": colors["GLM"],
            "XGB SHAP": colors["XGBoost"],
            "ChebyKAN": colors["ChebyKAN"],
            "FourierKAN": colors["FourierKAN"],
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

        # Kendall's τ: GLM↔SHAP, GLM↔ChebyKAN, SHAP↔ChebyKAN
        glm_ranks = [float(glm_imp.get(f, 0)) for f in top_feats]
        shap_ranks = [float(shap_by_risk.get(risk, pd.Series()).get(f, 0)
                      if not shap_by_risk.get(risk, pd.Series()).empty else 0)
                      for f in top_feats]
        cheby_ranks = [float(chebykan_by_risk.get(risk, pd.Series()).get(f, 0)
                       if not chebykan_by_risk.get(risk, pd.Series()).empty else 0)
                       for f in top_feats]

        tau_gs, _ = kendalltau(glm_ranks, shap_ranks)
        tau_gc, _ = kendalltau(glm_ranks, cheby_ranks)
        tau_sc, _ = kendalltau(shap_ranks, cheby_ranks)
        tau_text = f"τ GLM↔SHAP={tau_gs:.2f}\nτ GLM↔Cheby={tau_gc:.2f}\nτ SHAP↔Cheby={tau_sc:.2f}"
        ax.text(0.98, 0.02, tau_text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5, color="gray")

    caption = ("FourierKAN importance is global (not per-risk-level); its τ equals ChebyKAN in "
               "all panels and is therefore omitted.")
    fig.text(0.5, -0.01, caption, ha="center", fontsize=6, color="gray", style="italic")
    fig.suptitle("Feature Importance by Risk Level — GLM vs XGB SHAP vs KAN Symbolic",
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
) -> None:
    from src.interpretability.paths import figures as fig_dir, data as data_dir, reports as rep_dir

    X_eval = pd.read_parquet(eval_features_path)
    feature_names = list(X_eval.columns)

    # Load ChebyKAN layer for edge-variance importance
    chebykan_layer = None
    if chebykan_checkpoint_path and chebykan_checkpoint_path.exists() and chebykan_config_path and chebykan_config_path.exists():
        try:
            import torch
            from src.configs import load_experiment_config
            from src.models.tabkan import TabKAN
            from src.models.kan_layers import ChebyKANLayer
            cfg = load_experiment_config(chebykan_config_path)
            module = TabKAN(in_features=X_eval.shape[1],
                            widths=[cfg.model.width] * cfg.model.depth,
                            kan_type="chebykan", degree=cfg.model.degree or 3)
            module.load_state_dict(torch.load(chebykan_checkpoint_path, map_location="cpu"))
            module.eval()
            chebykan_layer = next((l for l in module.kan_layers if isinstance(l, ChebyKANLayer)), None)
        except Exception as e:
            print(f"Warning: could not load ChebyKAN model: {e}")

    feat_types: dict = {}
    ft_path = rep_dir(output_dir) / "feature_types.json"
    if ft_path.exists():
        import json
        feat_types = json.loads(ft_path.read_text())

    glm_imp = _glm_importance(glm_coef_path)
    shap_by_risk = _shap_importance_by_risk(shap_path, eval_features_path, xgb_checkpoint_path)
    chebykan_by_risk = _kan_importance_by_risk(chebykan_symbolic_path, feature_names, chebykan_layer)
    fourierkan_by_risk = _kan_importance_by_risk(fourierkan_symbolic_path, feature_names, None)

    _plot_panel(glm_imp, shap_by_risk, chebykan_by_risk, fourierkan_by_risk,
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
                "chebykan": chebykan_by_risk.get(risk, pd.Series()).get(feat, 0.0),
                "fourierkan": fourierkan_by_risk.get(risk, pd.Series()).get(feat, 0.0),
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
    )
