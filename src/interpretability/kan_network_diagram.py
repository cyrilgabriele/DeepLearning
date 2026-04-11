"""Signature KAN network diagram with learned activation functions on edges.

Produces the canonical KAN visualization from Liu et al. (2024) Figure 2.4:
each edge in the network displays a mini-plot of its learned 1D function,
with edge opacity scaled by L1 importance.

This is the key visual differentiator of KANs over MLPs — the learned
functions are directly visible in the network structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def draw_kan_diagram(
    module,
    feature_names: list[str],
    flavor: str,
    output_dir: Path,
    *,
    symbolic_fits: pd.DataFrame | None = None,
    top_n_inputs: int = 10,
    top_n_hidden: int = 8,
    show_formulas: bool = True,
) -> Path | None:
    """Draw a KAN network diagram with mini activation function plots on edges.

    Each edge displays the learned 1D function as a small inset plot.
    Edge opacity is proportional to tanh(3 * normalized_importance),
    following Liu et al. (2024).

    Parameters
    ----------
    module : TabKAN
        Trained KAN module.
    feature_names : list[str]
        Input feature names.
    flavor : str
        "chebykan" or "fourierkan".
    output_dir : Path
        Output directory for figures.
    symbolic_fits : DataFrame, optional
        Symbolic fit results. If provided, locked edges show formula text.
    top_n_inputs : int
        Number of top input features to display.
    top_n_hidden : int
        Number of top hidden nodes to display.
    show_formulas : bool
        Whether to annotate edges with symbolic formula text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from src.interpretability.kan_pruning import _compute_edge_l1
    from src.interpretability.kan_symbolic import sample_edge
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()

    kan_layers = [l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))]
    if len(kan_layers) < 1:
        print("draw_kan_diagram: no KAN layers found. Skipping.")
        return None

    # Use first layer for the main diagram
    layer0 = kan_layers[0]
    l0_scores = np.nan_to_num(_compute_edge_l1(layer0).numpy(), nan=0.0)  # (hidden, inputs)

    # Select top input features
    input_total = l0_scores.sum(axis=0)
    n_in = min(top_n_inputs, len(feature_names), input_total.shape[0])
    top_in_idx = np.argsort(input_total)[::-1][:n_in]
    top_in_labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in top_in_idx]

    # Select top hidden nodes
    hidden_total = l0_scores.sum(axis=1)
    n_hid = min(top_n_hidden, hidden_total.shape[0])
    top_hid_idx = np.argsort(hidden_total)[::-1][:n_hid]

    # Normalize importance for opacity
    max_l0 = l0_scores.max() + 1e-12

    # Build symbolic fit lookup if available
    fit_lookup = {}
    if symbolic_fits is not None:
        for _, row in symbolic_fits[symbolic_fits["layer"] == 0].iterrows():
            key = (int(row["edge_out"]), int(row["edge_in"]))
            fit_lookup[key] = row

    # Layout
    fig_h = max(10, max(n_in, n_hid) * 1.2)
    fig_w = 16
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Positions
    x_in = 0.08
    x_hid = 0.55
    x_out_col = 0.92

    in_ys = np.linspace(0.92, 0.08, n_in)
    hid_ys = np.linspace(0.88, 0.12, n_hid)

    # Draw edges with mini activation plots
    model_color = MODEL_COLORS.get("ChebyKAN" if flavor == "chebykan" else "FourierKAN", "steelblue")

    for ii, i_idx in enumerate(top_in_idx):
        for hi, h_idx in enumerate(top_hid_idx):
            importance = float(l0_scores[h_idx, i_idx])
            w = importance / max_l0
            if not np.isfinite(w) or w < 0.03:
                continue

            alpha = float(np.tanh(3 * w))
            alpha = max(0.15, min(alpha, 0.9))

            # Draw the edge line
            ax.plot(
                [x_in + 0.06, x_hid - 0.06],
                [in_ys[ii], hid_ys[hi]],
                color=model_color,
                alpha=alpha * 0.4,
                lw=w * 2.5,
                zorder=1,
                solid_capstyle="round",
            )

            # Add mini activation function plot if edge is important enough
            if w > 0.15:
                # Position the inset at the midpoint of the edge
                mid_x = (x_in + 0.06 + x_hid - 0.06) / 2
                mid_y = (in_ys[ii] + hid_ys[hi]) / 2

                try:
                    x_vals, y_vals = sample_edge(layer0, h_idx, i_idx, n=100)

                    # Create inset axes for the mini-plot
                    inset_w = 0.04
                    inset_h = 0.035
                    inset_ax = fig.add_axes(
                        [mid_x - inset_w / 2, mid_y - inset_h / 2, inset_w, inset_h],
                        facecolor="white",
                    )
                    inset_ax.plot(x_vals, y_vals, color=model_color, lw=0.8, alpha=alpha)
                    inset_ax.set_xticks([])
                    inset_ax.set_yticks([])
                    for spine in inset_ax.spines.values():
                        spine.set_linewidth(0.3)
                        spine.set_alpha(alpha)

                    # Add formula text if available and requested
                    if show_formulas:
                        fit_row = fit_lookup.get((h_idx, i_idx))
                        if fit_row is not None and fit_row["quality_tier"] in ("clean", "acceptable"):
                            formula = str(fit_row["formula"])
                            inset_ax.set_title(formula, fontsize=2.5, pad=1,
                                               color="darkred", alpha=alpha)
                except Exception:
                    pass

    # Draw input nodes
    for ii, label in enumerate(top_in_labels):
        ax.scatter(x_in, in_ys[ii], s=120, color="#2980B9", zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.text(x_in - 0.01, in_ys[ii], label[:20], ha="right", va="center",
                fontsize=6, fontweight="bold")

    # Draw hidden nodes
    for hi, h_idx in enumerate(top_hid_idx):
        ax.scatter(x_hid, hid_ys[hi], s=100, color="#8E44AD", zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.text(x_hid + 0.01, hid_ys[hi], f"h{h_idx}", ha="left", va="center",
                fontsize=6)

    # Draw second layer edges if present (simplified, no mini-plots)
    if len(kan_layers) >= 2:
        layer1 = kan_layers[1]
        l1_scores = np.nan_to_num(_compute_edge_l1(layer1).numpy(), nan=0.0)
        n_out = l1_scores.shape[0]
        out_ys = np.linspace(0.85, 0.15, n_out)
        max_l1 = l1_scores.max() + 1e-12

        for hi, h_idx in enumerate(top_hid_idx):
            for oi in range(n_out):
                w = float(l1_scores[oi, h_idx]) / max_l1
                if w < 0.05 or not np.isfinite(w):
                    continue
                alpha = float(np.clip(np.tanh(3 * w), 0.15, 0.9))
                ax.plot(
                    [x_hid + 0.03, x_out_col - 0.03],
                    [hid_ys[hi], out_ys[oi]],
                    color="#27AE60",
                    alpha=alpha * 0.5,
                    lw=w * 2.0,
                    zorder=1,
                )

        # Draw output nodes
        for oi in range(n_out):
            ax.scatter(x_out_col, out_ys[oi], s=100, color="#E74C3C", zorder=5,
                       edgecolors="white", linewidths=0.5)
            ax.text(x_out_col + 0.01, out_ys[oi], f"out {oi}", ha="left",
                    va="center", fontsize=6)

    # Column headers
    ax.text(x_in, 0.97, "Input Features", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2980B9")
    ax.text(x_hid, 0.97, "Hidden Layer", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#8E44AD")
    if len(kan_layers) >= 2:
        ax.text(x_out_col, 0.97, "Output", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#E74C3C")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # Stats annotation
    total_edges = l0_scores.size
    active_edges = int((l0_scores > 0.01 * max_l0).sum())
    ax.text(
        0.5, 0.01,
        f"{flavor} | {total_edges} total edges | "
        f"edge opacity = tanh(3 * normalized L1) | "
        f"mini-plots show learned activation functions",
        ha="center", va="bottom", fontsize=6, color="gray",
        transform=ax.transAxes,
    )

    plt.suptitle(
        f"{flavor} — KAN Network Diagram with Learned Edge Functions",
        fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.04)

    out_path = fig_dir(output_dir) / f"{flavor}_kan_diagram.pdf"
    savefig_pdf(fig, out_path)
    print(f"Saved KAN diagram -> {out_path}")
    plt.close()
    return out_path


def draw_before_after_pruning(
    module_before,
    module_after,
    feature_names: list[str],
    flavor: str,
    output_dir: Path,
    *,
    top_n_inputs: int = 10,
) -> Path | None:
    """Side-by-side: dense (before pruning) vs sparse (after pruning) KAN.

    Shows the visual impact of pruning on the network structure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.kan_pruning import _compute_edge_l1
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()

    def _draw_simple_graph(ax, module, title, color):
        kan_layers = [l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))]
        if not kan_layers:
            ax.text(0.5, 0.5, "No KAN layers", ha="center")
            return

        layer0 = kan_layers[0]
        scores = _compute_edge_l1(layer0).numpy()
        max_score = scores.max() + 1e-12

        input_total = scores.sum(axis=0)
        n_in = min(top_n_inputs, len(feature_names), input_total.shape[0])
        top_in_idx = np.argsort(input_total)[::-1][:n_in]

        n_hid = min(8, scores.shape[0])
        hid_total = scores.sum(axis=1)
        top_hid_idx = np.argsort(hid_total)[::-1][:n_hid]

        in_ys = np.linspace(0.9, 0.1, n_in)
        hid_ys = np.linspace(0.85, 0.15, n_hid)

        n_drawn = 0
        for ii, i_idx in enumerate(top_in_idx):
            for hi, h_idx in enumerate(top_hid_idx):
                w = float(scores[h_idx, i_idx]) / max_score
                if w < 0.02:
                    continue
                n_drawn += 1
                alpha = float(np.tanh(3 * w))
                ax.plot([0.1, 0.9], [in_ys[ii], hid_ys[hi]],
                        color=color, alpha=alpha * 0.7, lw=w * 3, zorder=1)

        for ii, i_idx in enumerate(top_in_idx):
            label = feature_names[i_idx][:15] if i_idx < len(feature_names) else f"f{i_idx}"
            ax.scatter(0.1, in_ys[ii], s=60, color=color, zorder=3)
            ax.text(0.08, in_ys[ii], label, ha="right", va="center", fontsize=5)

        for hi, h_idx in enumerate(top_hid_idx):
            ax.scatter(0.9, hid_ys[hi], s=40, color="#8E44AD", zorder=3)

        total = scores.size
        active = int((scores > 0.01 * max_score).sum())
        ax.set_title(f"{title}\n{active}/{total} edges shown", fontsize=9, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        ax.axis("off")

    model_color = MODEL_COLORS.get("ChebyKAN" if flavor == "chebykan" else "FourierKAN", "steelblue")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    _draw_simple_graph(ax1, module_before, "Before Pruning (dense)", "#AAAAAA")
    _draw_simple_graph(ax2, module_after, "After Pruning (sparse)", model_color)

    plt.suptitle(f"{flavor} — Pruning Effect on Network Structure",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = fig_dir(output_dir) / f"{flavor}_before_after_pruning.pdf"
    savefig_pdf(fig, out_path)
    print(f"Saved before/after diagram -> {out_path}")
    plt.close()
    return out_path
