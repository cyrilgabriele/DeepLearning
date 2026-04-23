"""Partial Dependence Plots for KAN models.

Shows the direct input → output effect for each feature: sweep one feature
across its observed range while holding all others at their observed values,
and plot the average model prediction.  This answers the actuary's question:
"if BMI goes up, does predicted risk go up?"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def compute_partial_dependence(
    module,
    X_eval: pd.DataFrame,
    feature: str,
    *,
    grid_resolution: int = 100,
    percentile_range: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PDP for a single feature.

    Returns (grid_values, avg_predictions) where grid_values are the feature
    values swept and avg_predictions are the mean model outputs at each point.
    """
    feat_idx = list(X_eval.columns).index(feature)
    col = X_eval[feature].values
    lo, hi = np.percentile(col, percentile_range)
    grid = np.linspace(lo, hi, grid_resolution)

    X_base = torch.tensor(X_eval.values, dtype=torch.float32)
    avg_preds = np.empty(grid_resolution)

    module.eval()
    with torch.no_grad():
        for i, val in enumerate(grid):
            X_mod = X_base.clone()
            X_mod[:, feat_idx] = val
            preds = module(X_mod).cpu().numpy().flatten()
            avg_preds[i] = preds.mean()

    return grid, avg_preds


def plot_partial_dependence(
    module,
    X_eval: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    flavor: str,
    *,
    X_raw: pd.DataFrame | None = None,
    feat_types: dict | None = None,
    grid_resolution: int = 100,
    ncols: int = 5,
) -> Path:
    """Generate a PDP grid for the given features and save as PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import (
        apply_paper_style, savefig_pdf, MODEL_COLORS,
        FEATURE_TYPE_MARKERS, encode_to_raw_lookup,
    )

    apply_paper_style()
    feat_types = feat_types or {}

    n = len(features)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    color = MODEL_COLORS.get("ChebyKAN" if flavor == "chebykan" else "FourierKAN", "steelblue")

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        grid_enc, avg_pred = compute_partial_dependence(
            module, X_eval, feat, grid_resolution=grid_resolution,
        )

        # Convert to original scale if raw data available
        ftype = feat_types.get(feat, "unknown")
        is_numeric_raw = ftype in ("continuous", "ordinal")
        has_raw = X_raw is not None and feat in X_raw.columns and is_numeric_raw

        if has_raw:
            grid_plot = encode_to_raw_lookup(feat, X_eval, X_raw, grid_enc)
            xlabel = "Original scale"
        else:
            grid_plot = grid_enc
            xlabel = "Encoded [-1,1]" if ftype in ("binary", "missing_indicator") else "Encoded"

        ax.plot(grid_plot, avg_pred, color=color, lw=2)

        # Add rug plot showing data distribution
        col_vals = X_eval[feat].values
        if has_raw:
            rug_vals = encode_to_raw_lookup(feat, X_eval, X_raw, col_vals)
        else:
            rug_vals = col_vals
        ax.plot(
            rug_vals[::max(1, len(rug_vals) // 100)],
            np.full(len(rug_vals[::max(1, len(rug_vals) // 100)]), ax.get_ylim()[0]),
            "|", color="gray", alpha=0.3, markersize=4,
        )

        marker = FEATURE_TYPE_MARKERS.get(ftype, "")
        ax.set_title(f"{feat} {marker}", fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Avg. predicted risk", fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{flavor.title()} — Partial Dependence (Input → Output Effect)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    from src.interpretability.utils.paths import figures as fig_dir
    out = fig_dir(output_dir) / f"{flavor}_partial_dependence.pdf"
    savefig_pdf(fig, out)
    plt.close()
    print(f"Saved PDP → {out}")
    return out
