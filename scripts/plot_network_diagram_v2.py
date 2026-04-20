#!/usr/bin/env python3
"""Clean KAN network diagram: large readable activation plots on each edge.

Shows top-10 inputs → 8 hidden → 1 output.
Each active edge has a clear mini-plot of its learned function.
Hidden → output shows the linear head weight.
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
        l1_weight=1.0, entropy_weight=1.0, use_layer_norm=False,
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
            .replace("Medical_Keyword_", "MedKW ")
            .replace("Medical_History_", "MedHist ")
            .replace("Employment_Info_", "Empl ")
            .replace("Product_Info_", "Prod "))


def draw(module, feat_names, qwk, output_dir):
    layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_w = module.head.weight.detach().squeeze().numpy()
    head_b = module.head.bias.detach().item()
    l1 = _compute_edge_l1(layer).numpy()
    max_l1 = l1.max() + 1e-12

    n_in_total = len(feat_names)
    n_hid = layer.out_features  # 8

    # Pick top-10 inputs by total L1
    input_imp = l1.sum(axis=0)
    top_n = 10
    in_order = np.argsort(input_imp)[::-1][:top_n]

    # Rank hidden nodes
    hid_order = np.argsort(l1.sum(axis=1))[::-1]

    # Collect all active edges with their data
    edges = []
    for ii, i_idx in enumerate(in_order):
        for hi, h_idx in enumerate(hid_order):
            imp = l1[h_idx, i_idx]
            if imp < 0.005:
                continue
            xv, yv = sample_edge(layer, h_idx, i_idx, n=200)
            fname, r2 = fit_symbolic_edge(xv, yv)
            edges.append({
                "ii": ii, "hi": hi, "i_idx": i_idx, "h_idx": h_idx,
                "imp": imp, "w": imp / max_l1,
                "xv": xv, "yv": yv, "fname": fname, "r2": r2,
            })

    # ── Layout ──────────────────────────────────────────────────────────────
    fig_w = 28
    fig_h = max(16, top_n * 1.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")

    x_in = 0.08
    x_hid = 0.60
    x_out = 0.92

    in_ys = np.linspace(0.93, 0.07, top_n)
    hid_ys = np.linspace(0.90, 0.10, n_hid)
    out_y = 0.50

    # ── Draw edges input→hidden with mini-plots ─────────────────────────────
    for e in edges:
        alpha = float(np.clip(np.tanh(3 * e["w"]), 0.10, 0.85))
        lw = 0.3 + 2.0 * e["w"]
        color = feat_color(feat_names[e["i_idx"]])

        y1 = in_ys[e["ii"]]
        y2 = hid_ys[e["hi"]]

        # Faint background line
        ax.plot([x_in + 0.03, x_hid - 0.03], [y1, y2],
                color=color, alpha=alpha * 0.25, lw=lw, zorder=1,
                solid_capstyle="round")

        # Mini-plot
        mid_x = x_in + 0.03 + (x_hid - x_in - 0.06) * 0.5
        mid_y = (y1 + y2) / 2

        pw = 0.055   # plot width in figure coords
        ph = 0.038   # plot height

        inset = fig.add_axes(
            [mid_x - pw / 2, mid_y - ph / 2, pw, ph],
            facecolor="white",
        )
        inset.plot(e["xv"], e["yv"], color=color, lw=1.2, alpha=min(alpha + 0.3, 1.0))
        inset.axhline(0, color="gray", lw=0.3, ls=":", alpha=0.4)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color(color)
            spine.set_alpha(alpha * 0.6)

        # Formula label
        short = e["fname"]
        # Simplify display
        short = (short
                 .replace("a*x^3 + b*x^2 + c*x + d", "cubic")
                 .replace("a*x^2 + b*x + c", "quadratic")
                 .replace("a*x + b", "linear")
                 .replace("a*cos(x) + b", "cos")
                 .replace("a*sin(2*x) + b*cos(2*x)", "sin+cos")
                 .replace("a*exp(x) + b", "exp")
                 .replace("a*sin(2*x) + b", "sin(2x)")
                 .replace("a*sin(x) + b", "sin")
                 .replace("a (constant)", "const")
                 )
        tier = "✓" if e["r2"] >= 0.99 else "~"
        inset.set_title(f"{short} {tier}", fontsize=5.5, pad=1,
                        color="black", fontweight="bold")

        # Thin lines connecting inset to nodes
        # Left connector: input node → inset
        ax.plot([x_in + 0.03, mid_x - pw / 2], [y1, mid_y],
                color=color, alpha=alpha * 0.4, lw=lw * 0.7, zorder=0,
                solid_capstyle="round")
        # Right connector: inset → hidden node
        ax.plot([mid_x + pw / 2, x_hid - 0.03], [mid_y, y2],
                color=color, alpha=alpha * 0.4, lw=lw * 0.7, zorder=0,
                solid_capstyle="round")

    # ── Draw edges hidden→output (linear weights) ───────────────────────────
    max_hw = np.abs(head_w).max() + 1e-12
    for hi, h_idx in enumerate(hid_order):
        w_val = head_w[h_idx]
        w_norm = abs(w_val) / max_hw
        alpha = float(np.clip(0.3 + 0.6 * w_norm, 0.2, 0.9))
        lw = 1.5 + 3.5 * w_norm
        color = "#2166AC" if w_val > 0 else "#B2182B"

        ax.plot([x_hid + 0.03, x_out - 0.03], [hid_ys[hi], out_y],
                color=color, alpha=alpha * 0.5, lw=lw, zorder=1,
                solid_capstyle="round")

        # Weight label
        mid_x = x_hid + 0.03 + (x_out - x_hid - 0.06) * 0.45
        mid_y = hid_ys[hi] + (out_y - hid_ys[hi]) * 0.45
        ax.text(mid_x, mid_y, f"×{w_val:+.2f}",
                fontsize=7, color=color, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.85, edgecolor=color, linewidth=0.5))

    # ── Draw input nodes ─────────────────────────────────────────────────────
    for ii, i_idx in enumerate(in_order):
        feat = feat_names[i_idx]
        color = feat_color(feat)
        s = 120
        ax.scatter(x_in, in_ys[ii], s=s, color=color, zorder=5,
                   edgecolors="white", linewidths=1.0)
        label = shorten(feat)
        ax.text(x_in - 0.015, in_ys[ii], label, ha="right", va="center",
                fontsize=9, fontweight="bold", color=color)

    # ── Draw hidden nodes ────────────────────────────────────────────────────
    for hi, h_idx in enumerate(hid_order):
        ax.scatter(x_hid, hid_ys[hi], s=120, color="#8E44AD", zorder=5,
                   edgecolors="white", linewidths=1.0)
        n_active = int((l1[h_idx, :] >= 0.005).sum())
        ax.text(x_hid + 0.015, hid_ys[hi], f"h{h_idx} ({n_active}in)",
                ha="left", va="center", fontsize=8, fontweight="bold", color="#8E44AD")

    # ── Draw output node ─────────────────────────────────────────────────────
    ax.scatter(x_out, out_y, s=300, color="#E74C3C", zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.text(x_out, out_y - 0.045, f"Risk\nbias={head_b:.2f}",
            ha="center", va="top", fontsize=9, fontweight="bold", color="#E74C3C")

    # ── Column labels ────────────────────────────────────────────────────────
    ax.text(x_in, 0.99, "Input Features", ha="center", fontsize=12,
            fontweight="bold", color="#333")
    ax.text(x_hid, 0.99, "Hidden (Σ)", ha="center", fontsize=12,
            fontweight="bold", color="#8E44AD")
    ax.text(x_out, 0.99, "Output", ha="center", fontsize=12,
            fontweight="bold", color="#E74C3C")

    # ── Arrows ───────────────────────────────────────────────────────────────
    ax.annotate("", xy=(0.33, 0.965), xytext=(0.14, 0.965),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))
    ax.text(0.235, 0.975, "φⱼᵢ(tanh(x))  Chebyshev activation",
            ha="center", fontsize=8, color="gray", style="italic")

    ax.annotate("", xy=(x_out - 0.05, 0.965), xytext=(x_hid + 0.05, 0.965),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))
    ax.text((x_hid + x_out) / 2, 0.975, "wⱼ · hⱼ  (linear)",
            ha="center", fontsize=8, color="gray", style="italic")

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Patch(facecolor="#2166AC", label="Continuous"),
        Patch(facecolor="#B2182B", label="Medical History"),
        Patch(facecolor="#4DAF4A", label="Binary Keyword"),
        Line2D([0], [0], color="#2166AC", lw=2.5, label="Positive weight (→ higher risk)"),
        Line2D([0], [0], color="#B2182B", lw=2.5, label="Negative weight (→ lower risk)"),
    ]
    ax.legend(handles=legend_els, loc="lower left", fontsize=9, frameon=True,
              bbox_to_anchor=(0.01, 0.01), ncol=2)

    # ── Remaining inputs note ────────────────────────────────────────────────
    remaining = n_in_total - top_n
    if remaining > 0:
        ax.text(x_in, 0.01, f"+ {remaining} more features\n(lower importance)",
                ha="center", va="bottom", fontsize=8, color="gray", style="italic")

    # ── Title ────────────────────────────────────────────────────────────────
    n_active_total = int((l1 >= 0.005).sum())
    fig.suptitle(
        f"ChebyKAN Network Diagram — Learned Edge Functions (Input → Hidden → Output)\n"
        f"prediction = {head_b:.2f} + Σⱼ wⱼ · Σᵢ φⱼᵢ(tanh(xᵢ))    |    "
        f"QWK = {qwk:.4f}    |    {n_active_total} active edges    |    "
        f"each mini-plot = one learned activation function",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_pdf = output_dir / "kan_network_diagram_v2.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    out_png = output_dir / "kan_network_diagram_v2.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")
    plt.close()


def main():
    output_dir = Path("outputs/interpretable_kan_no_layernorm/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up model...")
    module, feat_names, qwk = setup()

    print("\nDrawing network diagram v2...")
    draw(module, feat_names, qwk, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
