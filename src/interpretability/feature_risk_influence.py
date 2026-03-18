"""Feature → Risk-level influence plots.

Produces three panel figures showing how individual features (in their original,
unencoded scale) relate to the actual risk level (1–8):

  1. Continuous panel  — BMI, Ins_Age, etc.: scatter + binned-mean line
  2. Categorical panel — Product_Info_2, Medical_History_* etc.: box plots per
                         original category code, sorted by mean risk
  3. Binary panel      — Medical_Keyword_*, InsuredInfo_* etc.: grouped bar
                         showing mean ± std risk level for value 0 vs. 1

Features are ranked by global mean-|SHAP| importance so the most impactful
features lead each panel.

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


# ── Continuous panel ──────────────────────────────────────────────────────────

def _plot_continuous(
    feats: list[str],
    X_raw: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    n_bins: int = 20,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.paths import figures as fig_dir

    n = len(feats)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, feat in enumerate(feats):
        if feat not in X_raw.columns:
            continue
        vals = X_raw[feat].reset_index(drop=True)
        risk = y.reset_index(drop=True)
        valid = vals.notna() & risk.notna()
        v, r = vals[valid], risk[valid]

        # ── Top row: scatter + binned mean ────────────────────────────────
        ax = axes[0, col]
        ax.scatter(v, r, alpha=0.05, s=4, color="steelblue", rasterized=True)

        # Binned mean and 95 % CI
        bins = pd.cut(v, bins=n_bins)
        grouped = r.groupby(bins)
        means = grouped.mean()
        sems = grouped.sem()
        bin_centers = [iv.mid for iv in means.index]
        ax.plot(bin_centers, means.values, color="tomato", lw=2, zorder=5, label="Bin mean")
        ax.fill_between(
            bin_centers,
            (means - 1.96 * sems).values,
            (means + 1.96 * sems).values,
            alpha=0.25, color="tomato", label="95% CI",
        )
        ax.set_xlabel(f"{feat} (original scale)", fontsize=8)
        ax.set_ylabel("Risk level (1–8)", fontsize=8)
        ax.set_ylim(0.5, 8.5)
        ax.set_yticks(range(1, 9))
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        # ── Bottom row: distribution (KDE / histogram) per risk level ─────
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

    fig.suptitle("Continuous Features → Risk Level", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir(output_dir) / f"feature_risk_continuous.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
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
    from src.interpretability.paths import figures as fig_dir

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

        # Limit to top max_cats categories by count
        top_cats = v.value_counts().head(max_cats).index.tolist()
        mask = v.isin(top_cats)
        v, r = v[mask], r[mask]

        # Sort categories by mean risk level
        cat_order = r.groupby(v).mean().sort_values().index.tolist()

        data_per_cat = [r[v == cat].values for cat in cat_order]
        positions = np.arange(len(cat_order))

        bp = ax.boxplot(
            data_per_cat,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "lw": 2},
        )
        # Color boxes by median risk (low=green, high=red)
        medians = [np.median(d) if len(d) > 0 else 4 for d in data_per_cat]
        palette = plt.cm.RdYlGn_r(np.interp(medians, [1, 8], [0, 1]))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Annotate with sample count
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

    fig.suptitle("Categorical Features → Risk Level  (sorted by median)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir(output_dir) / f"feature_risk_categorical.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close()


# ── Binary panel ──────────────────────────────────────────────────────────────

def _plot_binary(
    feats: list[str],
    X_raw: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.paths import figures as fig_dir

    n = len(feats)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, feats):
        if feat not in X_raw.columns:
            ax.set_visible(False)
            continue
        vals = X_raw[feat].reset_index(drop=True)
        risk = y.reset_index(drop=True)
        valid = vals.notna() & risk.notna()
        v, r = vals[valid], risk[valid]

        groups = {val: r[v == val] for val in sorted(v.unique())}
        positions = list(range(len(groups)))
        labels = [str(int(k)) if float(k) == int(float(k)) else str(k) for k in groups]
        means = [g.mean() for g in groups.values()]
        stds = [g.std() for g in groups.values()]
        counts = [len(g) for g in groups.values()]

        colors = ["#4C72B0", "#DD8452"]
        bars = ax.bar(positions, means, color=colors[:len(positions)], alpha=0.8,
                      yerr=stds, capsize=5, error_kw={"elinewidth": 1.2})

        for i, (pos, mean, count) in enumerate(zip(positions, means, counts)):
            ax.text(pos, mean + max(stds) * 0.15 + 0.05, f"n={count}", ha="center",
                    fontsize=7, color="gray")
            ax.text(pos, 0.6, f"{mean:.2f}", ha="center", fontsize=8, fontweight="bold",
                    color="white")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("Mean risk level", fontsize=8)
        ax.set_ylim(0.5, 8.5)
        ax.set_yticks(range(1, 9))
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.axhline(r.mean(), color="gray", lw=1, linestyle="--", alpha=0.7)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Binary Features → Mean Risk Level  (± 1 std)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir(output_dir) / f"feature_risk_binary.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    output_dir: Path = Path("outputs"),
    n_continuous: int = 6,
    n_categorical: int = 6,
    n_binary: int = 12,
) -> None:
    from src.interpretability.paths import data as data_dir, reports as rep_dir

    X_raw_path = data_dir(output_dir) / "X_eval_raw.parquet"
    y_path = data_dir(output_dir) / "y_eval.parquet"
    shap_path = data_dir(output_dir) / "shap_xgb_values.parquet"
    feat_types_path = rep_dir(output_dir) / "feature_types.json"

    if not X_raw_path.exists():
        raise FileNotFoundError(
            f"Raw eval features not found at {X_raw_path}. "
            "Re-run any experiment training first to export X_eval_raw.parquet."
        )

    X_raw = pd.read_parquet(X_raw_path).reset_index(drop=True)
    y = pd.read_parquet(y_path)["Response"].reset_index(drop=True)
    feat_types: dict = json.loads(feat_types_path.read_text()) if feat_types_path.exists() else {}
    shap_rank = _shap_rank(shap_path)

    cont_feats = _top_by_type(shap_rank, feat_types, "continuous", n_continuous)
    cat_feats = _top_by_type(shap_rank, feat_types, "categorical", n_categorical)
    bin_feats = _top_by_type(shap_rank, feat_types, "binary", n_binary)

    # Filter to features actually present in X_raw
    cont_feats = [f for f in cont_feats if f in X_raw.columns]
    cat_feats = [f for f in cat_feats if f in X_raw.columns]
    bin_feats = [f for f in bin_feats if f in X_raw.columns]

    print(f"Continuous features ({len(cont_feats)}): {cont_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")
    print(f"Binary features ({len(bin_feats)}): {bin_feats}")

    if cont_feats:
        _plot_continuous(cont_feats, X_raw, y, output_dir)
    if cat_feats:
        _plot_categorical(cat_feats, X_raw, y, output_dir)
    if bin_feats:
        _plot_binary(bin_feats, X_raw, y, output_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature → risk-level influence plots")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--n-continuous", type=int, default=6)
    p.add_argument("--n-categorical", type=int, default=6)
    p.add_argument("--n-binary", type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.output_dir, args.n_continuous, args.n_categorical, args.n_binary)
