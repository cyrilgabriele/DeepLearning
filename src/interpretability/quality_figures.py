"""Additional paper figures for symbolic fit quality and pruning analysis.

Generates:
1. R² distribution histogram with quality tier annotations
2. Quality tier pie chart
3. Pruning Pareto curve (QWK vs. sparsity for multiple thresholds)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_r2_distribution(
    fits_df: pd.DataFrame,
    flavor: str,
    output_dir: Path,
) -> Path:
    """Histogram of R² values across all symbolic fits, with tier boundaries.

    Shows the distribution of symbolic fit quality, annotated with the
    three-tier classification from Liu et al. (2024):
    - Clean: R² >= 0.99
    - Acceptable: 0.90 <= R² < 0.99
    - Flagged: R² < 0.90
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()

    r2_values = fits_df["r_squared"].values
    n_clean = int((r2_values >= 0.99).sum())
    n_acceptable = int(((r2_values >= 0.90) & (r2_values < 0.99)).sum())
    n_flagged = int((r2_values < 0.90).sum())
    total = len(r2_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Histogram
    bins = np.linspace(0, 1, 51)
    color = MODEL_COLORS.get("ChebyKAN" if flavor == "chebykan" else "FourierKAN", "steelblue")

    ax1.hist(r2_values, bins=bins, color=color, alpha=0.7, edgecolor="white", lw=0.5)

    # Tier boundary lines
    ax1.axvline(0.99, color="#2ECC71", lw=2, ls="--", label=f"Clean (R²≥0.99): {n_clean}")
    ax1.axvline(0.90, color="#F39C12", lw=2, ls="--", label=f"Acceptable (0.90≤R²<0.99): {n_acceptable}")

    # Shade tier regions
    ylim = ax1.get_ylim()
    ax1.axvspan(0.99, 1.0, alpha=0.1, color="#2ECC71")
    ax1.axvspan(0.90, 0.99, alpha=0.1, color="#F39C12")
    ax1.axvspan(0.0, 0.90, alpha=0.1, color="#E74C3C")

    ax1.set_xlabel("R² (symbolic fit quality)", fontsize=9)
    ax1.set_ylabel("Number of edges", fontsize=9)
    ax1.set_title(f"{flavor} — R² Distribution of Symbolic Fits (n={total})",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper left")

    # Summary stats
    ax1.text(0.02, 0.85,
             f"Mean R² = {np.mean(r2_values):.4f}\n"
             f"Median R² = {np.median(r2_values):.4f}\n"
             f"Flagged (R²<0.90): {n_flagged}",
             transform=ax1.transAxes, fontsize=7,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Pie chart
    sizes = [n_clean, n_acceptable, n_flagged]
    labels = [
        f"Clean\n(≥0.99)\n{n_clean}",
        f"Acceptable\n(0.90–0.99)\n{n_acceptable}",
        f"Flagged\n(<0.90)\n{n_flagged}",
    ]
    pie_colors = ["#2ECC71", "#F39C12", "#E74C3C"]

    # Filter out zero-count slices
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, pie_colors) if s > 0]
    if nonzero:
        sizes_nz, labels_nz, colors_nz = zip(*nonzero)
        ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 7})
    ax2.set_title("Quality Tier Breakdown", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = fig_dir(output_dir) / f"{flavor}_r2_distribution.pdf"
    savefig_pdf(fig, out)
    print(f"Saved -> {out}")
    plt.close()
    return out


def compute_pruning_pareto(
    module,
    config,
    flavor: str,
    eval_features_path: Path,
    eval_labels_path: Path,
    *,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Compute QWK vs. sparsity for multiple pruning thresholds.

    Returns a list of dicts: [{threshold, sparsity, qwk, edges_before, edges_after}, ...]
    """
    import torch
    from sklearn.metrics import cohen_kappa_score
    from src.interpretability.kan_pruning import _compute_edge_l1
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    import copy

    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    X_eval = pd.read_parquet(eval_features_path)
    y_eval = pd.read_parquet(eval_labels_path).iloc[:, 0]
    X_tensor = torch.tensor(X_eval.values, dtype=torch.float32)

    results = []
    for thresh in thresholds:
        pruned = copy.deepcopy(module)
        total_before = 0
        total_after = 0

        for layer in pruned.kan_layers:
            if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
                continue
            scores = _compute_edge_l1(layer)
            total_before += scores.numel()
            mask = scores >= thresh
            total_after += int(mask.sum().item())

            with torch.no_grad():
                if hasattr(layer, "cheby_coeffs"):
                    for out_i in range(layer.out_features):
                        for in_i in range(layer.in_features):
                            if not mask[out_i, in_i]:
                                layer.cheby_coeffs[out_i, in_i, :] = 0.0
                                layer.base_weight[out_i, in_i] = 0.0
                elif hasattr(layer, "fourier_a"):
                    for out_i in range(layer.out_features):
                        for in_i in range(layer.in_features):
                            if not mask[out_i, in_i]:
                                layer.fourier_a[out_i, in_i, :] = 0.0
                                layer.fourier_b[out_i, in_i, :] = 0.0
                                layer.base_weight[out_i, in_i] = 0.0

        pruned.eval()
        with torch.no_grad():
            preds = pruned(X_tensor).cpu().numpy().flatten()
        preds_cls = np.clip(np.round(preds), 1, 8).astype(int)
        qwk = float(cohen_kappa_score(y_eval, preds_cls, weights="quadratic"))

        sparsity = 1.0 - (total_after / total_before) if total_before > 0 else 0.0

        results.append({
            "threshold": thresh,
            "sparsity": round(sparsity, 4),
            "qwk": round(qwk, 6),
            "edges_before": total_before,
            "edges_after": total_after,
        })

    return results


def plot_pruning_pareto(
    pareto_data: dict[str, list[dict]],
    output_dir: Path,
) -> Path:
    """Plot QWK vs. sparsity Pareto curves for ChebyKAN and FourierKAN.

    Parameters
    ----------
    pareto_data : dict[str, list[dict]]
        flavor -> list of {threshold, sparsity, qwk, ...} dicts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for flavor, points in pareto_data.items():
        sparsities = [p["sparsity"] * 100 for p in points]
        qwks = [p["qwk"] for p in points]
        model_name = "ChebyKAN" if flavor == "chebykan" else "FourierKAN"
        color = MODEL_COLORS.get(model_name, "steelblue")

        ax.plot(sparsities, qwks, color=color, lw=2, marker="o", ms=6, label=model_name)

        # Annotate thresholds
        for p in points:
            ax.annotate(
                f"θ={p['threshold']}",
                (p["sparsity"] * 100, p["qwk"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=5, alpha=0.7,
            )

    ax.set_xlabel("Sparsity (% edges pruned)", fontsize=9)
    ax.set_ylabel("QWK", fontsize=9)
    ax.set_title("Pruning Pareto Curve: QWK vs. Sparsity",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Mark the "sweet spot" region
    ax.text(0.5, -0.1,
            "Ideal: high sparsity with minimal QWK loss. "
            "θ = L1 activation magnitude threshold for edge retention.",
            ha="center", transform=ax.transAxes, fontsize=6, color="gray")

    plt.tight_layout()
    out = fig_dir(output_dir) / "pruning_pareto_curve.pdf"
    savefig_pdf(fig, out)
    print(f"Saved -> {out}")
    plt.close()
    return out
