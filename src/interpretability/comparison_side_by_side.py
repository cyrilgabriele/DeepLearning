"""Issue 10 — Side-by-side: GLM reference | XGBoost SHAP | KAN symbolic spline.

For the top-5 globally important features produces a 5×3 figure:
  Col 1: GLM coefficient (horizontal reference line)
  Col 2: SHAP dependence plot (XGBoost)
  Col 3: KAN learned spline + symbolic fit overlay

Usage:
    uv run python -m src.interpretability.comparison_side_by_side \
        --glm-coefficients    outputs/glm_coefficients.csv \
        --shap-values         outputs/shap_xgb_values.parquet \
        --chebykan-symbolic   outputs/chebykan_symbolic_fits.csv \
        --chebykan-checkpoint outputs/chebykan_pruned_module.pt \
        --chebykan-config     configs/chebykan_experiment.yaml \
        --eval-features       outputs/X_eval.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _select_top_features(
    glm_coef: pd.DataFrame,
    shap_df: pd.DataFrame,
    symbolic_df: pd.DataFrame,
    n: int = 5,
) -> list[str]:
    """Union of top-n from each source, picking the most-frequently-nominated."""
    glm_top = set(glm_coef.nlargest(n, "abs_magnitude")["feature"].tolist())
    shap_top = set(shap_df.abs().mean().nlargest(n).index.tolist())
    # KAN: features from first-layer active edges only
    layer0 = symbolic_df[symbolic_df["layer"] == 0]
    kan_top = set(
        layer0.groupby("input_feature").size().nlargest(n).index.tolist()
    )
    counter: dict[str, int] = {}
    for s in (glm_top, shap_top, kan_top):
        for f in s:
            counter[f] = counter.get(f, 0) + 1
    # Sort by vote count descending, then alphabetically for stability
    ranked = sorted(counter.items(), key=lambda t: (-t[1], t[0]))
    return [f for f, _ in ranked[:n]]


def _get_kan_edge_for_feature(symbolic_df: pd.DataFrame, feature: str):
    """Return the row for the highest-R² active edge from a given input feature."""
    candidates = symbolic_df[
        (symbolic_df["layer"] == 0) & (symbolic_df["input_feature"] == feature)
    ]
    if candidates.empty:
        return None
    return candidates.loc[candidates["r_squared"].idxmax()]


def run(
    glm_coef_path: Path,
    shap_path: Path,
    chebykan_symbolic_path: Path,
    chebykan_checkpoint_path: Path,
    chebykan_config_path: Path,
    eval_features_path: Path,
    output_dir: Path = Path("outputs"),
    n_features: int = 5,
) -> None:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.configs import load_experiment_config
    from src.models.tabkan import TabKAN
    from src.models.kan_layers import ChebyKANLayer
    from src.interpretability.kan_symbolic import sample_edge

    from src.interpretability.paths import figures as fig_dir

    glm_coef = pd.read_csv(glm_coef_path)
    shap_df = pd.read_parquet(shap_path)
    sym_df = pd.read_csv(chebykan_symbolic_path)
    X_eval = pd.read_parquet(eval_features_path)
    feature_names = list(X_eval.columns)

    top_features = _select_top_features(glm_coef, shap_df, sym_df, n=n_features)
    print(f"Top {n_features} features: {top_features}")

    # Load pruned ChebyKAN module
    cfg = load_experiment_config(chebykan_config_path)
    in_features = X_eval.shape[1]
    widths = [cfg.model.width] * cfg.model.depth
    module = TabKAN(in_features=in_features, widths=widths, kan_type="chebykan",
                    degree=cfg.model.degree or 3)
    module.load_state_dict(torch.load(chebykan_checkpoint_path, map_location="cpu"))
    module.eval()

    fig, axes = plt.subplots(n_features, 3, figsize=(15, 4 * n_features))
    col_titles = ["GLM (coefficient)", "XGBoost SHAP", "ChebyKAN symbolic"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold", pad=10)

    glm_indexed = glm_coef.set_index("feature")
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    for row, feat in enumerate(top_features):
        # ── Column 0: GLM coefficient reference ──────────────────────────────
        ax0 = axes[row, 0]
        coef_val = float(glm_indexed.loc[feat, "coefficient"]) if feat in glm_indexed.index else 0.0
        x_range = np.linspace(X_eval[feat].min(), X_eval[feat].max(), 200) if feat in X_eval.columns else np.linspace(-1, 1, 200)
        ax0.axhline(coef_val, color="#4C72B0", lw=2, label=f"coef={coef_val:.4f}")
        ax0.plot(x_range, coef_val * x_range, color="#4C72B0", lw=1.5, alpha=0.6, linestyle="--",
                 label="linear effect")
        ax0.axhline(0, color="gray", lw=0.5, linestyle=":")
        ax0.set_ylabel(feat[:20], fontsize=8)
        ax0.legend(fontsize=7)
        ax0.set_xlabel("Feature value")

        # ── Column 1: XGBoost SHAP dependence ────────────────────────────────
        ax1 = axes[row, 1]
        if feat in shap_df.columns and feat in X_eval.columns:
            ax1.scatter(X_eval[feat], shap_df[feat], alpha=0.2, s=5,
                        c=shap_df[feat], cmap="coolwarm")
            ax1.axhline(0, color="gray", lw=0.5, linestyle=":")
            ax1.set_xlabel("Feature value")
            ax1.set_ylabel("SHAP value")
        else:
            ax1.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax1.transAxes)

        # ── Column 2: KAN spline + symbolic overlay ───────────────────────────
        ax2 = axes[row, 2]
        edge_row = _get_kan_edge_for_feature(sym_df, feat)

        if edge_row is not None:
            out_i = int(edge_row["edge_out"])
            in_i = int(edge_row["edge_in"])

            # Find first ChebyKAN layer
            kan_layer = next(
                (l for l in module.kan_layers if isinstance(l, ChebyKANLayer)), None
            )
            if kan_layer is not None:
                x_vals, y_vals = sample_edge(kan_layer, out_i, in_i, n=500)
                ax2.plot(x_vals, y_vals, lw=1.5, color="steelblue", label="Learned spline")

                # Symbolic fit overlay
                formula = str(edge_row["formula"])
                r2 = float(edge_row["r_squared"])
                try:
                    x = x_vals
                    y_sym = eval(  # noqa: S307
                        formula.replace("^", "**")
                               .replace("sqrt", "np.sqrt")
                               .replace("log", "np.log")
                               .replace("exp", "np.exp")
                               .replace("sin", "np.sin")
                               .replace("cos", "np.cos")
                               .replace("abs", "np.abs")
                    )
                    ax2.plot(x_vals, y_sym, lw=1.5, linestyle="--", color="tomato",
                             label=f"{formula[:30]}\nR²={r2:.3f}")
                except Exception:
                    pass
                ax2.legend(fontsize=6)
            ax2.set_xlabel("Normalised input")
            ax2.set_ylabel("Edge output")
        else:
            ax2.text(0.5, 0.5, "No active KAN edge", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=9, color="gray")

    plt.suptitle("Side-by-Side Interpretability: GLM | XGBoost SHAP | ChebyKAN Symbolic",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out_path = fig_dir(output_dir) / f"side_by_side_comparison.{ext}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side GLM / SHAP / KAN symbolic plots")
    p.add_argument("--glm-coefficients", type=Path, default=Path("outputs/data/glm_coefficients.csv"))
    p.add_argument("--shap-values", type=Path, default=Path("outputs/data/shap_xgb_values.parquet"))
    p.add_argument("--chebykan-symbolic", type=Path, default=Path("outputs/data/chebykan_symbolic_fits.csv"))
    p.add_argument("--chebykan-checkpoint", type=Path, default=Path("outputs/models/chebykan_pruned_module.pt"))
    p.add_argument("--chebykan-config", type=Path, default=Path("configs/chebykan_experiment.yaml"))
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.glm_coefficients,
        args.shap_values,
        args.chebykan_symbolic,
        args.chebykan_checkpoint,
        args.chebykan_config,
        args.eval_features,
        args.output_dir,
    )
