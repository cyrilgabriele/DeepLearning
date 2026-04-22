"""Side-by-side: GLM coefficient | XGBoost SHAP | KAN symbolic (per flavor).

Run once per KAN flavor:
    uv run python -m src.interpretability.comparison_side_by_side --flavor chebykan ...
    uv run python -m src.interpretability.comparison_side_by_side --flavor fourierkan ...

Produces:
    outputs/figures/side_by_side_chebykan.pdf
    outputs/figures/side_by_side_fourierkan.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _select_top_features(
    glm_coef: pd.DataFrame,
    shap_df: pd.DataFrame,
    kan_rank: pd.Series,
    feat_types: dict,
    n: int = 5,
    min_continuous: int = 2,
    min_binary: int = 2,
) -> list[str]:
    """Union-vote top features ensuring >=min_continuous and >=min_binary."""
    glm_top = set(glm_coef.nlargest(n, "abs_magnitude")["feature"].tolist())
    shap_top = set(shap_df.abs().mean().nlargest(n).index.tolist())
    kan_top = set(kan_rank.head(n).index.tolist()) if not kan_rank.empty else set()

    counter: dict[str, int] = {}
    for s in (glm_top, shap_top, kan_top):
        for f in s:
            counter[f] = counter.get(f, 0) + 1
    ranked = [f for f, _ in sorted(counter.items(), key=lambda t: (-t[1], t[0]))]
    selected = list(ranked[:n])

    glm_ranked_all = glm_coef.sort_values("abs_magnitude", ascending=False)["feature"].tolist()
    cont_pool = [f for f in glm_ranked_all[:30]
                 if feat_types.get(f) in ("continuous", "ordinal") and f not in selected]
    bin_pool = [f for f in glm_ranked_all[:30]
                if feat_types.get(f) in ("binary", "missing_indicator") and f not in selected]

    n_cont = sum(1 for f in selected if feat_types.get(f) in ("continuous", "ordinal"))
    n_bin = sum(1 for f in selected if feat_types.get(f) in ("binary", "missing_indicator"))

    for f in cont_pool:
        if n_cont >= min_continuous:
            break
        selected.append(f)
        n_cont += 1

    for f in bin_pool:
        if n_bin >= min_binary:
            break
        selected.append(f)
        n_bin += 1

    return selected


def _plot_continuous(axes_row, feat, glm_indexed, shap_df, X_eval, X_raw, sym_df, kan_layer, flavor):
    """Continuous: GLM linear effect | SHAP scatter+LOWESS | KAN feature function."""
    from src.interpretability.utils.style import encode_to_raw_lookup, MODEL_COLORS
    from src.interpretability.utils.kan_coefficients import sample_feature_function

    enc = X_eval[feat].values if feat in X_eval.columns else np.linspace(-1, 1, 200)
    has_raw = X_raw is not None and feat in X_raw.columns
    feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1

    # ── Col 0: GLM ──
    ax0 = axes_row[0]
    coef = float(glm_indexed.loc[feat, "coefficient"]) if feat in glm_indexed.index else 0.0
    x_enc_grid = np.linspace(enc.min(), enc.max(), 200)
    x_orig_grid = encode_to_raw_lookup(feat, X_eval, X_raw, x_enc_grid) if has_raw else x_enc_grid
    ax0.plot(x_orig_grid, coef * x_enc_grid, color=MODEL_COLORS["GLM"], lw=2,
             label=f"coef={coef:.4f}")
    ax0.axhline(0, color="gray", lw=0.5, ls=":")
    ax0.set_xlabel(f"{feat} ({'original scale' if has_raw else 'encoded'})", fontsize=8)
    ax0.set_ylabel("Marginal effect", fontsize=8)
    ax0.legend(fontsize=7)

    # ── Col 1: SHAP scatter + LOWESS + std band ──
    ax1 = axes_row[1]
    if feat in shap_df.columns:
        shap_vals = shap_df[feat].values
        x_shap = encode_to_raw_lookup(feat, X_eval, X_raw) if has_raw else enc
        ax1.scatter(x_shap, shap_vals, alpha=0.15, s=4, c=shap_vals, cmap="coolwarm",
                    rasterized=True)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(shap_vals, x_shap, frac=0.3, return_sorted=True)
            bins = pd.cut(pd.Series(x_shap), bins=20)
            grp = pd.Series(shap_vals).groupby(bins)
            means, stds = grp.mean(), grp.std().fillna(0)
            centers = [iv.mid for iv in means.index]
            ax1.fill_between(centers, (means - stds).values, (means + stds).values,
                             alpha=0.2, color=MODEL_COLORS["XGBoost"])
            ax1.plot(smoothed[:, 0], smoothed[:, 1], color=MODEL_COLORS["XGBoost"],
                     lw=2, label="LOWESS ±1σ")
        except Exception:
            pass
        ax1.axhline(0, color="gray", lw=0.5, ls=":")
        ax1.set_xlabel(f"{feat} ({'original scale' if has_raw else 'encoded'})", fontsize=8)
        ax1.set_ylabel("SHAP value", fontsize=8)
        ax1.legend(fontsize=7)

    # ── Col 2: KAN spline + symbolic overlay ──
    ax2 = axes_row[2]
    if kan_layer is not None and feat_idx >= 0:
        x_norm, y_vals, _ = sample_feature_function(kan_layer, feat_idx, n=500, reduction="mean")
        x_plot = encode_to_raw_lookup(feat, X_eval, X_raw, x_norm) if has_raw else x_norm
        kan_color = MODEL_COLORS["ChebyKAN"] if flavor == "chebykan" else MODEL_COLORS["FourierKAN"]
        ax2.plot(x_plot, y_vals, color=kan_color, lw=2, label="Layer-0 aggregated")
        ax2.legend(fontsize=6)
        ax2.set_xlabel(f"{feat} ({'original scale' if has_raw else 'encoded'})", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No active KAN edge", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9, color="gray")
    ax2.set_ylabel("Edge output", fontsize=8)


def _plot_binary(axes_row, feat, glm_indexed, shap_df, X_eval, X_raw, sym_df, kan_layer, flavor):
    """Binary: GLM bars | SHAP violin | KAN aggregated feature function."""
    from src.interpretability.utils.style import MODEL_COLORS
    from src.interpretability.utils.kan_coefficients import sample_feature_function

    enc = X_eval[feat] if feat in X_eval.columns else None
    raw = X_raw[feat].reset_index(drop=True) if (X_raw is not None and feat in X_raw.columns) else None
    feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1

    if raw is not None and enc is not None:
        label_neg = str(raw[enc < -0.5].mode().iloc[0]) if (enc < -0.5).any() else "0"
        label_pos = str(raw[enc > 0.5].mode().iloc[0]) if (enc > 0.5).any() else "1"
    else:
        label_neg, label_pos = "-1 (enc)", "+1 (enc)"

    coef = float(glm_indexed.loc[feat, "coefficient"]) if feat in glm_indexed.index else 0.0

    # ── Col 0: GLM two bars ──
    ax0 = axes_row[0]
    ax0.barh([0, 1], [-coef, coef], color=MODEL_COLORS["GLM"], alpha=0.8)
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels([label_neg, label_pos], fontsize=9)
    ax0.axvline(0, color="gray", lw=0.5)
    ax0.set_xlabel("GLM marginal effect (encoded)", fontsize=8)

    # ── Col 1: SHAP violin by class ──
    ax1 = axes_row[1]
    if feat in shap_df.columns and enc is not None:
        groups = {label_neg: shap_df[feat][enc < -0.5].values,
                  label_pos: shap_df[feat][enc > 0.5].values}
        data = [v for v in groups.values() if len(v) > 0]
        labels = [k for k, v in groups.items() if len(v) > 0]
        if data:
            parts = ax1.violinplot(data, positions=range(len(data)), showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(MODEL_COLORS["XGBoost"])
                pc.set_alpha(0.7)
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("SHAP value", fontsize=8)

    # ── Col 2: KAN full curve on encoded domain with ±1 markers ──
    ax2 = axes_row[2]
    if kan_layer is not None and feat_idx >= 0:
        x_norm, y_vals, _ = sample_feature_function(kan_layer, feat_idx, n=500, reduction="mean")
        kan_color = MODEL_COLORS["ChebyKAN"] if flavor == "chebykan" else MODEL_COLORS["FourierKAN"]
        ax2.plot(x_norm, y_vals, color=kan_color, lw=2, label="Layer-0 aggregated")
        for x_mark, label in [(-1.0, label_neg), (1.0, label_pos)]:
            y_mark = float(np.interp(x_mark, x_norm, y_vals))
            ax2.axvline(x_mark, color="gray", lw=1, ls="--", alpha=0.7)
            ax2.scatter([x_mark], [y_mark], s=60, zorder=5, color="black")
            ax2.annotate(label, (x_mark, y_mark), xytext=(5, 5),
                         textcoords="offset points", fontsize=7)
        delta = float(np.interp(1.0, x_norm, y_vals)) - float(np.interp(-1.0, x_norm, y_vals))
        ax2.set_title(f"Δ = {delta:+.3f}", fontsize=8)
        ax2.legend(fontsize=6)
        ax2.set_xticks([-1.0, 0.0, 1.0])
        ax2.set_xticklabels([label_neg, "0", label_pos], fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No active KAN edge", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9, color="gray")
    ax2.set_xlabel("Encoded domain [-1, 1]", fontsize=8)
    ax2.set_ylabel("Edge output", fontsize=8)


def _plot_categorical(axes_row, feat, glm_indexed, shap_df, X_eval, sym_df, kan_layer, flavor):
    from src.interpretability.utils.style import MODEL_COLORS
    from src.interpretability.utils.kan_coefficients import sample_feature_function

    coef = float(glm_indexed.loc[feat, "coefficient"]) if feat in glm_indexed.index else 0.0
    enc = X_eval[feat] if feat in X_eval.columns else None
    feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1

    ax0 = axes_row[0]
    if enc is not None:
        cats = sorted(enc.unique())[:15]
        ax0.barh(range(len(cats)), [coef * c for c in cats], color=MODEL_COLORS["GLM"], alpha=0.8)
        ax0.set_yticks(range(len(cats)))
        ax0.set_yticklabels([f"{c:.2f}" for c in cats], fontsize=7)
    ax0.set_xlabel("GLM marginal effect", fontsize=8)

    ax1 = axes_row[1]
    if feat in shap_df.columns and enc is not None:
        cats = sorted(enc.unique())[:10]
        data = [shap_df[feat][enc == c].values for c in cats]
        data = [d for d in data if len(d) > 0]
        if data:
            ax1.violinplot(data, positions=range(len(data)), showmedians=True)
    ax1.set_ylabel("SHAP value", fontsize=8)

    ax2 = axes_row[2]
    if kan_layer is not None and feat_idx >= 0:
        x_norm, y_vals, _ = sample_feature_function(kan_layer, feat_idx, n=500, reduction="mean")
        kan_color = MODEL_COLORS["ChebyKAN"] if flavor == "chebykan" else MODEL_COLORS["FourierKAN"]
        ax2.plot(x_norm, y_vals, color=kan_color, lw=2)
        ax2.set_xlabel("CatBoost-encoded scale", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No active KAN edge", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9, color="gray")
    ax2.set_ylabel("Edge output", fontsize=8)


def run(
    glm_coef_path: Path,
    shap_path: Path,
    kan_symbolic_path: Path,
    kan_checkpoint_path: Path,
    kan_config_path: Path,
    kan_pruning_summary_path: Path,
    eval_features_path: Path,
    flavor: str,
    output_dir: Path = Path("outputs"),
    n_features: int = 5,
    eval_features_raw_path: Path | None = None,
) -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, feature_type_label
    from src.interpretability.utils.paths import figures as fig_dir, data as data_dir, reports as rep_dir
    from src.interpretability.utils.kan_coefficients import coefficient_importance_from_module
    from src.config import load_experiment_config
    from src.models.tabkan import TabKAN
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

    apply_paper_style()

    glm_coef = pd.read_csv(glm_coef_path)
    shap_df = pd.read_parquet(shap_path)
    sym_df = pd.read_csv(kan_symbolic_path)
    X_eval = pd.read_parquet(eval_features_path)

    raw_path = eval_features_raw_path or (data_dir(output_dir) / "X_eval_raw.parquet")
    X_raw: pd.DataFrame | None = None
    if raw_path.exists():
        X_raw = pd.read_parquet(raw_path).reset_index(drop=True)

    feat_types_path = rep_dir(output_dir) / "feature_types.json"
    feat_types: dict = {}
    if feat_types_path.exists():
        feat_types = json.loads(feat_types_path.read_text())

    cfg = load_experiment_config(kan_config_path)
    in_features = X_eval.shape[1]
    widths = cfg.model.resolved_hidden_widths()
    kan_type = "chebykan" if flavor == "chebykan" else "fourierkan"
    module = TabKAN(in_features=in_features, widths=widths, kan_type=kan_type,
                    degree=cfg.model.degree or 3,
                    grid_size=cfg.model.params.get("grid_size", 4),
                    use_layernorm=cfg.model.use_layernorm)
    module.load_state_dict(torch.load(kan_checkpoint_path, map_location="cpu"))
    module.eval()

    kan_layer = next(
        (l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))), None
    )
    kan_rank = coefficient_importance_from_module(module, list(X_eval.columns))
    top_features = _select_top_features(glm_coef, shap_df, kan_rank, feat_types, n=n_features)
    print(f"[{flavor}] Top features: {top_features}")

    glm_indexed = glm_coef.set_index("feature")
    n = len(top_features)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for c, title in enumerate(["GLM (coefficient)", "XGBoost SHAP", f"{flavor.title()} feature function"]):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold", pad=10)

    for row, feat in enumerate(top_features):
        ftype = feat_types.get(feat, "unknown")
        axes[row, 0].set_ylabel(feature_type_label(feat, feat_types)[:25], fontsize=8)
        if ftype in ("continuous", "ordinal"):
            _plot_continuous(axes[row], feat, glm_indexed, shap_df, X_eval, X_raw, sym_df, kan_layer, flavor)
        elif ftype in ("binary", "missing_indicator"):
            _plot_binary(axes[row], feat, glm_indexed, shap_df, X_eval, X_raw, sym_df, kan_layer, flavor)
        else:
            _plot_categorical(axes[row], feat, glm_indexed, shap_df, X_eval, sym_df, kan_layer, flavor)

    plt.suptitle(
        f"Side-by-Side Interpretability: GLM | XGBoost SHAP | {flavor.title()} Feature Function",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = fig_dir(output_dir) / f"side_by_side_{flavor}.pdf"
    savefig_pdf(fig, out_path)
    print(f"Saved → {out_path}")
    plt.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flavor", choices=["chebykan", "fourierkan"], required=True)
    p.add_argument("--glm-coefficients", type=Path, default=Path("outputs/data/glm_coefficients.csv"))
    p.add_argument("--shap-values", type=Path, default=Path("outputs/data/shap_xgb_values.parquet"))
    p.add_argument("--kan-symbolic", type=Path, required=True)
    p.add_argument("--kan-checkpoint", type=Path, required=True)
    p.add_argument("--kan-config", type=Path, required=True)
    p.add_argument("--kan-pruning-summary", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--eval-features-raw", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--n-features", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        glm_coef_path=args.glm_coefficients,
        shap_path=args.shap_values,
        kan_symbolic_path=args.kan_symbolic,
        kan_checkpoint_path=args.kan_checkpoint,
        kan_config_path=args.kan_config,
        kan_pruning_summary_path=args.kan_pruning_summary,
        eval_features_path=args.eval_features,
        flavor=args.flavor,
        output_dir=args.output_dir,
        n_features=args.n_features,
        eval_features_raw_path=args.eval_features_raw,
    )
