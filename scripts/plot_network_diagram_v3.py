#!/usr/bin/env python3
"""KAN network diagram v3: grid-based layout with one clear function per cell.

Instead of overlapping mini-plots on edges, uses a matrix layout:
rows = input features, columns = hidden neurons.
Each cell shows the learned activation function for that edge.
Right column shows the linear head weights to the output.
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
from matplotlib.gridspec import GridSpec

TOP_FEATURES = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]

CONTINUOUS = {"BMI", "Wt", "Ins_Age", "Product_Info_4", "Employment_Info_6"}
BINARY = {
    "Medical_Keyword_3", "Medical_Keyword_5", "Medical_Keyword_12",
    "Medical_Keyword_13", "Medical_Keyword_14", "Medical_Keyword_29",
    "Medical_Keyword_31", "Medical_Keyword_35", "Medical_Keyword_38",
    "Medical_Keyword_43", "Medical_Keyword_46", "Medical_Keyword_47",
}


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

    L.seed_everything(42)
    module = TabKAN(
        in_features=20, widths=[8], kan_type="chebykan", degree=3,
        lr=5e-3, weight_decay=5e-4, sparsity_lambda=0.005,
        l1_weight=1.0, entropy_weight=1.0, use_layernorm=False,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(data["y_train_outer"], dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)), batch_size=2048, shuffle=True
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(data["y_test_outer"], dtype=torch.float32)
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
    y_true = np.clip(np.round(data["y_test_outer"]), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, np.clip(np.round(preds), 1, 8).astype(int))
    print(f"QWK = {qwk:.4f}")
    return module, feat_names, qwk


def feat_color(feat):
    if feat in CONTINUOUS:
        return "#2166AC"
    if feat in BINARY:
        return "#4DAF4A"
    return "#B2182B"


def shorten(feat):
    return (feat
            .replace("Medical_Keyword_", "KW ")
            .replace("Medical_History_", "MH ")
            .replace("Employment_Info_", "Empl ")
            .replace("Product_Info_", "Prod "))


def draw(module, feat_names, qwk, output_dir):
    layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_w = module.head.weight.detach().squeeze().numpy()
    head_b = module.head.bias.detach().item()
    l1 = _compute_edge_l1(layer).numpy()
    max_l1 = l1.max() + 1e-12

    n_in = len(feat_names)
    n_hid = layer.out_features

    # Rank inputs by importance (total L1)
    input_imp = l1.sum(axis=0)
    in_order = np.argsort(input_imp)[::-1]

    # Rank hidden by importance
    hid_imp = l1.sum(axis=1)
    hid_order = np.argsort(hid_imp)[::-1]

    # Only show top-12 inputs for readability
    show_n_in = 12
    in_show = in_order[:show_n_in]

    # ── Create grid figure ──────────────────────────────────────────────────
    # Layout: rows = inputs, cols = hidden neurons + 1 output column
    # Extra row at top for hidden neuron labels
    # Extra column at right for head weights + output

    n_rows = show_n_in
    n_cols = n_hid + 1  # +1 for head weight column

    fig = plt.figure(figsize=(n_cols * 2.2 + 3, n_rows * 1.6 + 2.5))

    # Use gridspec: leave space for labels
    gs = GridSpec(
        n_rows + 1, n_cols + 1,  # +1 row for header, +1 col for input labels
        figure=fig,
        left=0.10, right=0.95, top=0.90, bottom=0.04,
        wspace=0.15, hspace=0.25,
        width_ratios=[2.5] + [1.0] * n_hid + [1.2],
        height_ratios=[0.4] + [1.0] * n_rows,
    )

    # ── Header row: hidden neuron labels ─────────────────────────────────────
    for hi, h_idx in enumerate(hid_order):
        ax_hdr = fig.add_subplot(gs[0, hi + 1])
        ax_hdr.axis("off")
        w_val = head_w[h_idx]
        sign = "+" if w_val > 0 else ""
        ax_hdr.text(0.5, 0.3, f"h{h_idx}\nw={sign}{w_val:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#8E44AD",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#F0E6FA", edgecolor="#8E44AD", linewidth=1))

    # Output header
    ax_out_hdr = fig.add_subplot(gs[0, n_hid + 1])
    ax_out_hdr.axis("off")
    ax_out_hdr.text(0.5, 0.3, f"Output\nbias={head_b:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#E74C3C",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#FDEDEB", edgecolor="#E74C3C", linewidth=1))

    # ── Grid cells: activation function plots ────────────────────────────────
    for row, i_idx in enumerate(in_show):
        feat = feat_names[i_idx]
        color = feat_color(feat)

        # Input label in first column
        ax_label = fig.add_subplot(gs[row + 1, 0])
        ax_label.axis("off")
        ax_label.text(0.95, 0.5, shorten(feat),
                      ha="right", va="center", fontsize=9, fontweight="bold",
                      color=color)
        # Type indicator
        ftype = "cont." if feat in CONTINUOUS else ("bin." if feat in BINARY else "ord.")
        ax_label.text(0.95, 0.15, ftype, ha="right", va="center",
                      fontsize=6, color="gray")

        # Edge cells
        for col, h_idx in enumerate(hid_order):
            ax_cell = fig.add_subplot(gs[row + 1, col + 1])
            imp = l1[h_idx, i_idx]
            w = imp / max_l1

            if imp < 0.005:
                # Pruned edge — gray X
                ax_cell.text(0.5, 0.5, "—", ha="center", va="center",
                             fontsize=12, color="#DDDDDD")
                ax_cell.set_facecolor("#FAFAFA")
                ax_cell.set_xticks([])
                ax_cell.set_yticks([])
                for spine in ax_cell.spines.values():
                    spine.set_linewidth(0.3)
                    spine.set_color("#EEEEEE")
                continue

            # Active edge — plot the function
            xv, yv = sample_edge(layer, h_idx, i_idx, n=200)
            fname, r2 = fit_symbolic_edge(xv, yv)

            ax_cell.plot(xv, yv, color=color, lw=1.8)
            ax_cell.axhline(0, color="gray", lw=0.3, ls=":", alpha=0.4)
            ax_cell.set_xticks([])
            ax_cell.set_yticks([])

            # Border color by importance
            border_alpha = float(np.clip(0.3 + 0.7 * w, 0.3, 1.0))
            for spine in ax_cell.spines.values():
                spine.set_linewidth(1.0 + 2.0 * w)
                spine.set_color(color)
                spine.set_alpha(border_alpha)

            # Background tint for clean fits
            if r2 >= 0.99:
                ax_cell.set_facecolor("#F0FFF0")  # light green
            elif r2 >= 0.90:
                ax_cell.set_facecolor("#FFFFF0")  # light yellow
            else:
                ax_cell.set_facecolor("#FFF0F0")  # light red

            # Formula label
            short = (fname
                     .replace("a*x^3 + b*x^2 + c*x + d", "x³")
                     .replace("a*x^2 + b*x + c", "x²")
                     .replace("a*x + b", "x")
                     .replace("a*cos(x) + b", "cos")
                     .replace("a*sin(2*x) + b*cos(2*x)", "sin2x")
                     .replace("a*exp(x) + b", "exp")
                     .replace("a*sin(2*x) + b", "sin2x")
                     .replace("a*sin(x) + b*cos(x)", "sincos")
                     .replace("a*sin(x) + b", "sin")
                     .replace("a (constant)", "c")
                     )
            ax_cell.text(0.95, 0.05, short, transform=ax_cell.transAxes,
                         ha="right", va="bottom", fontsize=6, color="black",
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                   alpha=0.8, edgecolor="none"))

        # Composed output cell: f_i(x) = Σ_j w_j * φ_ji(x)
        ax_out = fig.add_subplot(gs[row + 1, n_hid + 1])

        # Compute composite
        x_ref, comp_y = None, None
        for h_idx in range(n_hid):
            if l1[h_idx, i_idx] < 0.005:
                continue
            xv, yv = sample_edge(layer, h_idx, i_idx, n=200)
            w_j = head_w[h_idx]
            if comp_y is None:
                x_ref, comp_y = xv, w_j * yv
            else:
                comp_y += w_j * yv

        if comp_y is not None:
            ax_out.plot(x_ref, comp_y, color=color, lw=2.0)
            ax_out.axhline(0, color="gray", lw=0.3, ls=":", alpha=0.4)
            y_range = comp_y.max() - comp_y.min()
            ax_out.text(0.95, 0.05, f"range\n{y_range:.1f}",
                        transform=ax_out.transAxes, ha="right", va="bottom",
                        fontsize=5.5, color="gray")
            for spine in ax_out.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("#E74C3C")
                spine.set_alpha(0.6)
            ax_out.set_facecolor("#FFF8F6")
        else:
            ax_out.text(0.5, 0.5, "0", ha="center", va="center",
                        fontsize=10, color="#DDDDDD")
            for spine in ax_out.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("#EEEEEE")
            ax_out.set_facecolor("#FAFAFA")

        ax_out.set_xticks([])
        ax_out.set_yticks([])

    # ── Title ────────────────────────────────────────────────────────────────
    n_active = int((l1 >= 0.005).sum())
    fig.suptitle(
        f"ChebyKAN — Learned Activation Functions per Edge\n"
        f"Rows = input features  |  Columns = hidden neurons (h0–h7)  |  "
        f"Last column = composed f(x) to output\n"
        f"QWK = {qwk:.4f}  |  {n_active}/{l1.size} active edges  |  "
        f"Green bg = R²≥0.99  |  Yellow bg = R²≥0.90",
        fontsize=12, fontweight="bold",
    )

    # ── Column label arrows ──────────────────────────────────────────────────
    # Add annotation showing the flow
    fig.text(0.35, 0.925, "← φⱼᵢ(tanh(xᵢ)): each cell = one learned function →",
             ha="center", fontsize=9, color="gray", style="italic")
    fig.text(0.88, 0.925, "← Σⱼ wⱼ·φⱼᵢ →",
             ha="center", fontsize=9, color="#E74C3C", style="italic")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_png = output_dir / "kan_network_grid.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    out_pdf = output_dir / "kan_network_grid.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")
    plt.close()


def main():
    output_dir = Path("outputs/interpretable_kan_no_layernorm/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up model...")
    module, feat_names, qwk = setup()

    print("\nDrawing grid diagram...")
    draw(module, feat_names, qwk, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
