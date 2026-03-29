"""Feature → Risk-level influence plots.

Produces three panel figures showing how individual features relate to risk level (1–8):

  1. Continuous panel  — scatter + binned-mean + 3rd row 4-model overlay
  2. Binary panel      — paired dot-line plot (4 models × 2 class positions)
  3. Categorical panel — box plots per category sorted by median risk

Usage:
    uv run python -m src.interpretability.feature_risk_influence \\
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shap_rank(shap_path: Path) -> pd.Series:
    """Global mean-|SHAP| importance, descending."""
    if not shap_path.exists():
        return pd.Series(dtype=float)
    return pd.read_parquet(shap_path).abs().mean().sort_values(ascending=False)


def _top_by_type(shap_rank: pd.Series, feat_types: dict, ftype: str, n: int) -> list[str]:
    if shap_rank.empty:
        return [f for f, t in feat_types.items() if t == ftype][:n]
    return [f for f in shap_rank.index if feat_types.get(f) == ftype][:n]


def _binary_dot_values_shap(shap_vals: pd.Series, enc: pd.Series) -> tuple[float, float]:
    """Mean SHAP value for each encoded class group (neg: enc < -0.5, pos: enc > 0.5)."""
    neg_mean = float(shap_vals[enc < -0.5].mean())
    pos_mean = float(shap_vals[enc > 0.5].mean())
    return neg_mean, pos_mean


# ── Continuous panel ──────────────────────────────────────────────────────────

def _plot_continuous(
    feats: list[str],
    X_raw: pd.DataFrame,
    X_eval: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    glm_indexed: pd.DataFrame,
    shap_df: pd.DataFrame,
    chebykan_sym: pd.DataFrame | None,
    fourierkan_sym: pd.DataFrame | None,
    chebykan_layer,
    fourierkan_layer,
    n_bins: int = 20,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS, encode_to_raw_lookup
    from src.interpretability.utils.paths import figures as fig_dir
    from src.interpretability.utils.kan_coefficients import sample_feature_function

    apply_paper_style()
    n = len(feats)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, feat in enumerate(feats):
        if feat not in X_raw.columns:
            continue
        vals = X_raw[feat].reset_index(drop=True)
        risk = y.reset_index(drop=True)
        valid = vals.notna() & risk.notna()
        v, r = vals[valid], risk[valid]

        # ── Row 0: scatter + binned mean ──
        ax = axes[0, col]
        ax.scatter(v, r, alpha=0.05, s=4, color="steelblue", rasterized=True)
        bins = pd.cut(v, bins=n_bins)
        grouped = r.groupby(bins)
        means = grouped.mean()
        sems = grouped.sem()
        bin_centers = [iv.mid for iv in means.index]
        ax.plot(bin_centers, means.values, color="tomato", lw=2, zorder=5, label="Bin mean")
        ax.fill_between(bin_centers, (means - 1.96 * sems).values, (means + 1.96 * sems).values,
                        alpha=0.25, color="tomato", label="95% CI")
        ax.set_xlabel(f"{feat} (original scale)", fontsize=8)
        ax.set_ylabel("Risk level (1–8)", fontsize=8)
        ax.set_ylim(0.5, 8.5)
        ax.set_yticks(range(1, 9))
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        # ── Row 1: KDE per risk level ──
        ax2 = axes[1, col]
        palette = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 8))
        for risk_level, color in zip(range(1, 9), palette):
            subset = v[r == risk_level]
            if len(subset) < 10:
                continue
            subset.plot.kde(ax=ax2, label=f"R{risk_level}", color=color, lw=1.2, bw_method=0.4)
        ax2.set_xlabel(f"{feat} (original scale)", fontsize=8)
        ax2.set_ylabel("Density", fontsize=8)
        ax2.legend(fontsize=6, ncol=2, loc="upper right")
        ax2.set_title(f"{feat} distribution by risk", fontsize=8)

        # ── Row 2: 4-model overlay on original scale ──
        ax3 = axes[2, col]
        x_enc_vals = X_eval[feat].values if feat in X_eval.columns else None
        has_raw = feat in X_raw.columns

        if x_enc_vals is not None:
            x_norm_grid = np.linspace(x_enc_vals.min(), x_enc_vals.max(), 300)
            x_orig_grid = (encode_to_raw_lookup(feat, X_eval, X_raw, x_norm_grid)
                           if has_raw else x_norm_grid)

            # GLM
            if not glm_indexed.empty and feat in glm_indexed.index:
                coef = float(glm_indexed.loc[feat, "coefficient"])
                ax3.plot(x_orig_grid, coef * x_norm_grid, color=MODEL_COLORS["GLM"], lw=2, label="GLM")

            # SHAP LOWESS
            if not shap_df.empty and feat in shap_df.columns:
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    x_shap = (encode_to_raw_lookup(feat, X_eval, X_raw)
                              if has_raw else x_enc_vals)
                    smoothed = lowess(shap_df[feat].values, x_shap, frac=0.3, return_sorted=True)
                    ax3.plot(smoothed[:, 0], smoothed[:, 1], color=MODEL_COLORS["XGBoost"],
                             lw=2, label="XGBoost SHAP")
                except Exception:
                    pass

            # ChebyKAN and FourierKAN
            for layer, model_name in [
                (chebykan_layer, "ChebyKAN"),
                (fourierkan_layer, "FourierKAN"),
            ]:
                if layer is None:
                    continue
                feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1
                if feat_idx < 0:
                    continue
                x_norm_edge, y_edge, _ = sample_feature_function(layer, feat_idx, n=300, reduction="mean")
                x_plot = (encode_to_raw_lookup(feat, X_eval, X_raw, x_norm_edge)
                          if has_raw else x_norm_edge)
                ax3.plot(x_plot, y_edge, color=MODEL_COLORS[model_name], lw=1.5, label=model_name)

            ax3.axhline(0, color="gray", lw=0.5, ls=":")
            ax3.set_xlabel(f"{feat} (original scale)" if has_raw else f"{feat} (encoded)", fontsize=8)
            ax3.set_ylabel("Model attribution", fontsize=8)
            ax3.set_title(f"{feat} — all models", fontsize=8)
            ax3.legend(fontsize=7)

    fig.suptitle("Continuous Features → Risk Level", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = fig_dir(output_dir) / "feature_risk_continuous.pdf"
    savefig_pdf(fig, p)
    print(f"Saved → {p}")
    plt.close()


# ── Categorical panel ─────────────────────────────────────────────────────────

def _plot_categorical(
    feats: list[str],
    X_raw: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    max_cats: int = 15,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        if feat not in X_raw.columns:
            ax.set_visible(False)
            continue
        vals = X_raw[feat].reset_index(drop=True).astype(str)
        risk = y.reset_index(drop=True)
        valid = vals.notna() & risk.notna()
        v, r = vals[valid], risk[valid]

        top_cats = v.value_counts().head(max_cats).index.tolist()
        mask = v.isin(top_cats)
        v, r = v[mask], r[mask]
        cat_order = r.groupby(v).mean().sort_values().index.tolist()

        data_per_cat = [r[v == cat].values for cat in cat_order]
        positions = np.arange(len(cat_order))
        bp = ax.boxplot(data_per_cat, positions=positions, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "lw": 2})
        medians = [np.median(d) if len(d) > 0 else 4 for d in data_per_cat]
        palette = plt.cm.RdYlGn_r(np.interp(medians, [1, 8], [0, 1]))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for i, (cat, d) in enumerate(zip(cat_order, data_per_cat)):
            ax.text(i, 0.3, f"n={len(d)}", ha="center", va="bottom", fontsize=6, color="gray")

        ax.set_xticks(positions)
        ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(1, 9))
        ax.set_ylim(0.5, 8.8)
        ax.set_ylabel("Risk level (1–8)", fontsize=8)
        ax.set_xlabel("Original category code", fontsize=8)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.axhline(r.mean(), color="gray", lw=1, linestyle="--", label=f"Overall mean {r.mean():.2f}")
        ax.legend(fontsize=7)

    fig.suptitle("Categorical Features → Risk Level (sorted by median)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = fig_dir(output_dir) / "feature_risk_categorical.pdf"
    savefig_pdf(fig, p)
    print(f"Saved → {p}")
    plt.close()


# ── Binary panel ──────────────────────────────────────────────────────────────

def _plot_binary(
    feats: list[str],
    X_raw: pd.DataFrame,
    X_eval: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    glm_indexed: pd.DataFrame,
    shap_df: pd.DataFrame,
    chebykan_sym: pd.DataFrame | None,
    fourierkan_sym: pd.DataFrame | None,
    chebykan_layer,
    fourierkan_layer,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir
    from src.interpretability.utils.kan_coefficients import sample_feature_function

    apply_paper_style()

    def max_effect(feat):
        enc = X_eval[feat] if feat in X_eval.columns else None
        effects = []
        if not glm_indexed.empty and feat in glm_indexed.index and enc is not None:
            coef = float(glm_indexed.loc[feat, "coefficient"])
            effects.append(abs(2 * coef))
        if not shap_df.empty and feat in shap_df.columns and enc is not None:
            n_val, p_val = _binary_dot_values_shap(shap_df[feat], enc)
            effects.append(abs(p_val - n_val))
        feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1
        if feat_idx >= 0:
            for layer in (chebykan_layer, fourierkan_layer):
                if layer is None:
                    continue
                x_norm, y_vals, _ = sample_feature_function(layer, feat_idx, n=200, reduction="mean")
                effects.append(abs(float(np.interp(1.0, x_norm, y_vals)) - float(np.interp(-1.0, x_norm, y_vals))))
        return max(effects) if effects else 0.0

    feats = sorted(feats, key=max_effect, reverse=True)

    n = len(feats)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, feat in zip(axes_flat, feats):
        if feat not in X_eval.columns:
            ax.set_visible(False)
            continue
        enc = X_eval[feat].reset_index(drop=True)
        raw = X_raw[feat].reset_index(drop=True) if feat in X_raw.columns else None

        label_neg = str(raw[enc < -0.5].mode().iloc[0]) if (raw is not None and (enc < -0.5).any()) else "0"
        label_pos = str(raw[enc > 0.5].mode().iloc[0]) if (raw is not None and (enc > 0.5).any()) else "1"
        n_neg = int((enc < -0.5).sum())
        n_pos = int((enc > 0.5).sum())

        models_dots: dict[str, tuple[float, float]] = {}

        if not glm_indexed.empty and feat in glm_indexed.index:
            coef = float(glm_indexed.loc[feat, "coefficient"])
            models_dots["GLM"] = (-coef, coef)

        if not shap_df.empty and feat in shap_df.columns:
            neg_v, pos_v = _binary_dot_values_shap(shap_df[feat], enc)
            models_dots["XGBoost"] = (neg_v, pos_v)

        feat_idx = X_eval.columns.get_loc(feat) if feat in X_eval.columns else -1
        for layer, model_name in [
            (chebykan_layer, "ChebyKAN"),
            (fourierkan_layer, "FourierKAN"),
        ]:
            if layer is None or feat_idx < 0:
                continue
            x_norm, y_v, _ = sample_feature_function(layer, feat_idx, n=200, reduction="mean")
            neg_v = float(np.interp(-1.0, x_norm, y_v))
            pos_v = float(np.interp(1.0, x_norm, y_v))
            models_dots[model_name] = (neg_v, pos_v)

        for offset, (model_name, (v0, v1)) in enumerate(models_dots.items()):
            color = MODEL_COLORS[model_name]
            jitter = (offset - len(models_dots) / 2) * 0.06
            ax.plot([0 + jitter, 1 + jitter], [v0, v1], color=color, lw=1.5,
                    marker="o", ms=6, label=model_name)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{label_neg}\n(n={n_neg})", f"{label_pos}\n(n={n_pos})"], fontsize=8)
        ax.set_ylabel("Attribution (model native scale)", fontsize=8)
        ax.set_title(feat[:20], fontsize=9, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        if feat == feats[0]:
            ax.legend(fontsize=7, loc="best")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Binary Features — Model Attribution at Each Class Value", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = fig_dir(output_dir) / "feature_risk_binary.pdf"
    savefig_pdf(fig, p)
    print(f"Saved → {p}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    output_dir: Path = Path("outputs"),
    n_continuous: int = 6,
    n_categorical: int = 6,
    n_binary: int = 12,
    glm_coef_path: Path | None = None,
    shap_path: Path | None = None,
    chebykan_symbolic_path: Path | None = None,
    fourierkan_symbolic_path: Path | None = None,
    chebykan_checkpoint_path: Path | None = None,
    fourierkan_checkpoint_path: Path | None = None,
    chebykan_config_path: Path | None = None,
    fourierkan_config_path: Path | None = None,
    chebykan_pruning_summary_path: Path | None = None,
    fourierkan_pruning_summary_path: Path | None = None,
) -> None:
    import torch
    from src.interpretability.utils.paths import data as data_dir, reports as rep_dir
    from src.interpretability.utils.kan_coefficients import coefficient_importance_from_layer
    from src.models.tabkan import TabKAN
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.configs import load_experiment_config

    _data = data_dir(output_dir)
    _rep = rep_dir(output_dir)
    glm_coef_path = glm_coef_path or _data / "glm_coefficients.csv"
    shap_path = shap_path or _data / "shap_xgb_values.parquet"
    chebykan_symbolic_path = chebykan_symbolic_path or _data / "chebykan_symbolic_fits.csv"
    fourierkan_symbolic_path = fourierkan_symbolic_path or _data / "fourierkan_symbolic_fits.csv"

    X_raw_path = _data / "X_eval_raw.parquet"
    X_eval_path = _data / "X_eval.parquet"
    y_path = _data / "y_eval.parquet"
    feat_types_path = _rep / "feature_types.json"

    if not X_raw_path.exists():
        raise FileNotFoundError(f"Raw eval features not found at {X_raw_path}.")

    X_raw = pd.read_parquet(X_raw_path).reset_index(drop=True)
    X_eval = pd.read_parquet(X_eval_path).reset_index(drop=True)
    y = pd.read_parquet(y_path)["Response"].reset_index(drop=True)
    feat_types: dict = json.loads(feat_types_path.read_text()) if feat_types_path.exists() else {}

    glm_coef = pd.read_csv(glm_coef_path) if glm_coef_path.exists() else pd.DataFrame()
    glm_indexed = glm_coef.set_index("feature") if not glm_coef.empty else pd.DataFrame()
    shap_df = pd.read_parquet(shap_path) if shap_path.exists() else pd.DataFrame()
    shap_rank = shap_df.abs().mean().sort_values(ascending=False) if not shap_df.empty else _shap_rank(shap_path)

    chebykan_sym = pd.read_csv(chebykan_symbolic_path) if (chebykan_symbolic_path and chebykan_symbolic_path.exists()) else None
    fourierkan_sym = pd.read_csv(fourierkan_symbolic_path) if (fourierkan_symbolic_path and fourierkan_symbolic_path.exists()) else None

    def _load_kan_layer(ckpt, cfg_path, flavor):
        if not (ckpt and ckpt.exists() and cfg_path and cfg_path.exists()):
            return None
        try:
            cfg = load_experiment_config(cfg_path)
            import torch
            module = TabKAN(in_features=X_eval.shape[1],
                            widths=[cfg.model.width] * cfg.model.depth,
                            kan_type=flavor, degree=cfg.model.degree or 3)
            module.load_state_dict(torch.load(ckpt, map_location="cpu"))
            module.eval()
            return next((l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))), None)
        except Exception as e:
            print(f"Warning: could not load {flavor} model: {e}")
            return None

    chebykan_layer = _load_kan_layer(chebykan_checkpoint_path, chebykan_config_path, "chebykan")
    fourierkan_layer = _load_kan_layer(fourierkan_checkpoint_path, fourierkan_config_path, "fourierkan")

    kan_rank = pd.Series(dtype=float)
    for layer in (chebykan_layer, fourierkan_layer):
        if layer is None:
            continue
        layer_rank = coefficient_importance_from_layer(layer, list(X_eval.columns)).set_index("feature")["importance"]
        kan_rank = kan_rank.add(layer_rank, fill_value=0.0)
    primary_rank = kan_rank.sort_values(ascending=False) if not kan_rank.empty else shap_rank

    cont_feats = _top_by_type(primary_rank, feat_types, "continuous", n_continuous)
    cat_feats = _top_by_type(primary_rank, feat_types, "categorical", n_categorical)
    bin_feats = _top_by_type(primary_rank, feat_types, "binary", n_binary)
    cont_feats = [f for f in cont_feats if f in X_raw.columns]
    cat_feats = [f for f in cat_feats if f in X_raw.columns]
    bin_feats = [f for f in bin_feats if f in X_raw.columns]

    print(f"Continuous features ({len(cont_feats)}): {cont_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")
    print(f"Binary features ({len(bin_feats)}): {bin_feats}")

    if cont_feats:
        _plot_continuous(cont_feats, X_raw, X_eval, y, output_dir,
                         glm_indexed, shap_df, chebykan_sym, fourierkan_sym,
                         chebykan_layer, fourierkan_layer)
    if cat_feats:
        _plot_categorical(cat_feats, X_raw, y, output_dir)
    if bin_feats:
        _plot_binary(bin_feats, X_raw, X_eval, y, output_dir,
                     glm_indexed, shap_df, chebykan_sym, fourierkan_sym,
                     chebykan_layer, fourierkan_layer)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature → risk-level influence plots")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--n-continuous", type=int, default=6)
    p.add_argument("--n-categorical", type=int, default=6)
    p.add_argument("--n-binary", type=int, default=12)
    p.add_argument("--glm-coefficients", type=Path, default=None)
    p.add_argument("--shap-values", type=Path, default=None)
    p.add_argument("--chebykan-symbolic", type=Path, default=None)
    p.add_argument("--fourierkan-symbolic", type=Path, default=None)
    p.add_argument("--chebykan-checkpoint", type=Path, default=None)
    p.add_argument("--fourierkan-checkpoint", type=Path, default=None)
    p.add_argument("--chebykan-config", type=Path, default=None)
    p.add_argument("--fourierkan-config", type=Path, default=None)
    p.add_argument("--chebykan-pruning-summary", type=Path, default=None)
    p.add_argument("--fourierkan-pruning-summary", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        output_dir=args.output_dir,
        n_continuous=args.n_continuous,
        n_categorical=args.n_categorical,
        n_binary=args.n_binary,
        glm_coef_path=args.glm_coefficients,
        shap_path=args.shap_values,
        chebykan_symbolic_path=args.chebykan_symbolic,
        fourierkan_symbolic_path=args.fourierkan_symbolic,
        chebykan_checkpoint_path=args.chebykan_checkpoint,
        fourierkan_checkpoint_path=args.fourierkan_checkpoint,
        chebykan_config_path=args.chebykan_config,
        fourierkan_config_path=args.fourierkan_config,
        chebykan_pruning_summary_path=args.chebykan_pruning_summary,
        fourierkan_pruning_summary_path=args.fourierkan_pruning_summary,
    )
