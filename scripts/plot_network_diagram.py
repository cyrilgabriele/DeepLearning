#!/usr/bin/env python3
"""KAN network diagram: input → hidden → output with edge functions and head weights.

Shows the full additive model structure:
- Left column: input features (top-20, ranked by importance)
- Middle column: 8 hidden neurons
- Right: single output node
- Edges input→hidden: mini-plots of learned activation φ_{j,i}(x)
- Edges hidden→output: labeled with linear head weight w_j
- Formulas annotated on strong edges
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_kan_paper import KANPreprocessor
from src.models.tabkan import TabKAN
from src.models.kan_layers import ChebyKANLayer
from src.interpretability.kan_pruning import _compute_edge_l1
from src.interpretability.kan_symbolic import sample_edge, fit_symbolic_edge
from src.metrics.qwk import quadratic_weighted_kappa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_FEATURES = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]

BINARY = {
    "Medical_Keyword_3", "Medical_Keyword_5", "Medical_Keyword_12",
    "Medical_Keyword_13", "Medical_Keyword_14", "Medical_Keyword_29",
    "Medical_Keyword_31", "Medical_Keyword_35", "Medical_Keyword_38",
    "Medical_Keyword_43", "Medical_Keyword_46", "Medical_Keyword_47",
}
CONTINUOUS = {"BMI", "Wt", "Ins_Age", "Product_Info_4", "Employment_Info_6"}


def setup():
    preprocessor = KANPreprocessor()
    data = preprocessor.run_pipeline(
        "data/prudential-life-insurance-assessment/train.csv", random_seed=42
    )
    all_names = data["feature_names"]
    keep_idx = np.array([all_names.index(f) for f in TOP_FEATURES if f in all_names])
    feat_names = [all_names[i] for i in keep_idx]
    X_train = data["X_train_outer"][:, keep_idx]
    X_test = data["X_test_outer"][:, keep_idx]
    y_train, y_test = data["y_train_outer"], data["y_test_outer"]

    L.seed_everything(42)
    module = TabKAN(
        in_features=20, widths=[8], kan_type="chebykan", degree=3,
        lr=5e-3, weight_decay=5e-4, sparsity_lambda=0.005,
        l1_weight=1.0, entropy_weight=1.0, use_layernorm=False,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)), batch_size=2048, shuffle=True
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1)), batch_size=2048, shuffle=False
    )
    trainer = L.Trainer(
        max_epochs=150, accelerator="auto",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    module.eval()

    with torch.no_grad():
        preds = module(X_val_t).cpu().numpy().flatten()
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, np.clip(np.round(preds), 1, 8).astype(int))
    print(f"QWK = {qwk:.4f}")

    return module, feat_names, qwk


def get_feat_color(feat):
    if feat in CONTINUOUS:
        return "#2166AC"
    if feat in BINARY:
        return "#4DAF4A"
    return "#B2182B"


def shorten(feat):
    return (feat
            .replace("Medical_Keyword_", "MedKW_")
            .replace("Medical_History_", "MedHist_")
            .replace("Employment_Info_", "Empl_")
            .replace("Product_Info_", "Prod_"))


def draw_network_diagram(module, feat_names, qwk, output_dir):
    first_layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_w = module.head.weight.detach().squeeze().numpy()  # (8,)
    head_b = module.head.bias.detach().item()
    l1 = _compute_edge_l1(first_layer).numpy()  # (8, 20)
    max_l1 = l1.max() + 1e-12

    n_in = len(feat_names)
    n_hid = first_layer.out_features  # 8

    # Rank inputs by total L1
    input_importance = l1.sum(axis=0)
    in_order = np.argsort(input_importance)[::-1]

    # Rank hidden by total L1
    hidden_importance = l1.sum(axis=1)
    hid_order = np.argsort(hidden_importance)[::-1]

    # Build symbolic fit lookup
    fit_lookup = {}
    for out_j in range(n_hid):
        for in_i in range(n_in):
            if l1[out_j, in_i] < 0.005:
                continue
            xv, yv = sample_edge(first_layer, out_j, in_i, n=200)
            fname, r2 = fit_symbolic_edge(xv, yv)
            fit_lookup[(out_j, in_i)] = (fname, r2, xv, yv)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_h = max(14, n_in * 0.8)
    fig = plt.figure(figsize=(24, fig_h))

    # Main axes for nodes and edges
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")

    # Column positions
    x_in = 0.12
    x_hid = 0.55
    x_out = 0.88

    in_ys = np.linspace(0.95, 0.05, n_in)
    hid_ys = np.linspace(0.90, 0.10, n_hid)
    out_y = 0.50

    # ── Draw edges: input → hidden (with mini-plots) ────────────────────────
    for ii, i_idx in enumerate(in_order):
        for hi, h_idx in enumerate(hid_order):
            importance = l1[h_idx, i_idx]
            w = importance / max_l1
            if w < 0.02:
                continue

            alpha = float(np.clip(np.tanh(3 * w), 0.08, 0.85))
            lw = 0.5 + 2.5 * w

            color = get_feat_color(feat_names[i_idx])

            # Draw edge line
            ax.plot(
                [x_in + 0.04, x_hid - 0.04],
                [in_ys[ii], hid_ys[hi]],
                color=color, alpha=alpha * 0.35, lw=lw, zorder=1,
                solid_capstyle="round",
            )

            # Mini activation plot on strong edges
            if w > 0.08 and (h_idx, i_idx) in fit_lookup:
                fname, r2, xv, yv = fit_lookup[(h_idx, i_idx)]
                mid_x = (x_in + 0.04 + x_hid - 0.04) / 2
                mid_y = (in_ys[ii] + hid_ys[hi]) / 2

                inset_w = 0.038
                inset_h = 0.028
                inset_ax = fig.add_axes(
                    [mid_x - inset_w / 2, mid_y - inset_h / 2, inset_w, inset_h],
                    facecolor="white",
                )
                inset_ax.plot(xv, yv, color=color, lw=0.7, alpha=min(alpha + 0.2, 1.0))
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                for spine in inset_ax.spines.values():
                    spine.set_linewidth(0.3)
                    spine.set_alpha(alpha * 0.7)

                # Formula text on strong edges
                if w > 0.20 and r2 >= 0.95:
                    short_formula = fname.replace("a*", "").replace("+ b", "").replace("+ c", "").replace("+ d", "").strip()
                    if len(short_formula) > 15:
                        short_formula = short_formula[:15] + "..."
                    inset_ax.set_title(short_formula, fontsize=3, pad=0.5,
                                       color="darkred", alpha=min(alpha + 0.3, 1.0))

    # ── Draw edges: hidden → output (linear head weights) ────────────────────
    max_head_w = np.abs(head_w).max() + 1e-12
    for hi, h_idx in enumerate(hid_order):
        w_val = head_w[h_idx]
        w_norm = abs(w_val) / max_head_w
        alpha = float(np.clip(0.3 + 0.6 * w_norm, 0.2, 0.9))
        lw = 1.0 + 3.0 * w_norm
        color = "#2166AC" if w_val > 0 else "#B2182B"

        ax.plot(
            [x_hid + 0.04, x_out - 0.04],
            [hid_ys[hi], out_y],
            color=color, alpha=alpha * 0.6, lw=lw, zorder=1,
            solid_capstyle="round",
        )

        # Label the weight value
        mid_x = (x_hid + 0.04 + x_out - 0.04) / 2
        mid_y = (hid_ys[hi] + out_y) / 2
        ax.text(
            mid_x + 0.02, mid_y,
            f"w={w_val:+.3f}",
            fontsize=6, color=color, alpha=alpha,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # ── Draw input nodes ─────────────────────────────────────────────────────
    for ii, i_idx in enumerate(in_order):
        feat = feat_names[i_idx]
        color = get_feat_color(feat)
        imp = input_importance[i_idx]
        # Size proportional to importance
        s = 80 + 150 * (imp / input_importance.max())
        ax.scatter(x_in, in_ys[ii], s=s, color=color, zorder=5,
                   edgecolors="white", linewidths=0.8)
        label = shorten(feat)
        ax.text(x_in - 0.015, in_ys[ii], label, ha="right", va="center",
                fontsize=7, fontweight="bold", color=color)

    # ── Draw hidden nodes ────────────────────────────────────────────────────
    for hi, h_idx in enumerate(hid_order):
        imp = hidden_importance[h_idx]
        s = 80 + 120 * (imp / hidden_importance.max())
        w_val = head_w[h_idx]
        node_color = "#8E44AD"
        ax.scatter(x_hid, hid_ys[hi], s=s, color=node_color, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(x_hid + 0.015, hid_ys[hi], f"h{h_idx}", ha="left", va="center",
                fontsize=7, fontweight="bold", color=node_color)

        # Count active inputs for this hidden node
        n_active = int((l1[h_idx, :] >= 0.005).sum())
        ax.text(x_hid + 0.015, hid_ys[hi] - 0.015, f"({n_active} inputs)",
                ha="left", va="top", fontsize=5, color="gray")

    # ── Draw output node ─────────────────────────────────────────────────────
    ax.scatter(x_out, out_y, s=250, color="#E74C3C", zorder=5,
               edgecolors="white", linewidths=1.0)
    ax.text(x_out + 0.02, out_y, "Risk\nPrediction", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#E74C3C")
    ax.text(x_out + 0.02, out_y - 0.04, f"bias = {head_b:.3f}",
            ha="left", va="top", fontsize=7, color="gray")

    # ── Column headers ───────────────────────────────────────────────────────
    ax.text(x_in, 1.00, "Input Features (20)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333")
    ax.text(x_hid, 1.00, "Hidden Layer (8)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#8E44AD")
    ax.text(x_out, 1.00, "Output", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#E74C3C")

    # ── Arrows showing flow direction ────────────────────────────────────────
    arrow_y = 0.97
    ax.annotate("", xy=(x_hid - 0.08, arrow_y), xytext=(x_in + 0.08, arrow_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text((x_in + x_hid) / 2, arrow_y + 0.01, "Chebyshev\nactivations φ(x)",
            ha="center", va="bottom", fontsize=7, color="gray", style="italic")

    ax.annotate("", xy=(x_out - 0.06, arrow_y), xytext=(x_hid + 0.06, arrow_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text((x_hid + x_out) / 2, arrow_y + 0.01, "Linear head\nw·h + bias",
            ha="center", va="bottom", fontsize=7, color="gray", style="italic")

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#2166AC", label="Continuous feature"),
        Patch(facecolor="#B2182B", label="Medical History (ordinal)"),
        Patch(facecolor="#4DAF4A", label="Medical Keyword (binary)"),
        Line2D([0], [0], color="#2166AC", lw=2, label="Positive head weight"),
        Line2D([0], [0], color="#B2182B", lw=2, label="Negative head weight"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              frameon=True, bbox_to_anchor=(0.01, 0.01))

    # ── Title ────────────────────────────────────────────────────────────────
    # Count active edges
    n_active = int((l1 >= 0.005).sum())
    n_total = l1.size
    fig.suptitle(
        f"ChebyKAN Additive Model — Network Architecture\n"
        f"prediction = {head_b:.3f} + Σⱼ wⱼ · Σᵢ φⱼᵢ(tanh(xᵢ))   |   "
        f"QWK = {qwk:.4f}   |   {n_active}/{n_total} active edges",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_pdf = output_dir / "kan_network_diagram_full.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_pdf}")

    out_png = output_dir / "kan_network_diagram_full.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")
    plt.close()


def main():
    output_dir = Path("outputs/interpretable_kan_no_layernorm/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up model...")
    module, feat_names, qwk = setup()

    print("\nDrawing network diagram...")
    draw_network_diagram(module, feat_names, qwk, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
