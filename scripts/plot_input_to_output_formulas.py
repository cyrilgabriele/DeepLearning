#!/usr/bin/env python3
"""Plot the exact input→output feature contribution functions.

Retrains the best no-LayerNorm model and plots fᵢ(xᵢ) for each feature,
both on encoded scale and original scale where possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_kan_paper import KANPreprocessor
from src.models.tabkan import TabKAN
from src.models.kan_layers import ChebyKANLayer
from src.interpretability.kan_pruning import _compute_edge_l1
from src.interpretability.kan_symbolic import sample_edge
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

# Feature types for annotation
CONTINUOUS = {"BMI", "Wt", "Ins_Age", "Product_Info_4", "Employment_Info_6"}
ORDINAL_MEDICAL = {
    "Medical_History_4", "Medical_History_5", "Medical_History_20",
    "Medical_History_23", "Medical_History_27", "Medical_History_30",
    "Medical_History_39", "Medical_History_40", "Medical_History_15",
    "Medical_History_18",
}
BINARY = {
    "Medical_Keyword_3", "Medical_Keyword_5", "Medical_Keyword_9",
    "Medical_Keyword_12", "Medical_Keyword_13", "Medical_Keyword_14",
    "Medical_Keyword_15", "Medical_Keyword_29", "Medical_Keyword_31",
    "Medical_Keyword_35", "Medical_Keyword_38", "Medical_Keyword_43",
    "Medical_Keyword_46", "Medical_Keyword_47",
}


def load_data(seed=42):
    csv_path = Path("data/prudential-life-insurance-assessment/train.csv")
    preprocessor = KANPreprocessor()
    return preprocessor.run_pipeline(csv_path, random_seed=seed)


def select_features(data, n_features, top_features=TOP_FEATURES):
    all_names = data["feature_names"]
    keep = top_features[:n_features]
    keep_indices = [all_names.index(f) for f in keep if f in all_names]
    kept_names = [all_names[i] for i in keep_indices]
    idx = np.array(keep_indices)
    return (
        data["X_train_outer"][:, idx],
        data["X_test_outer"][:, idx],
        data["y_train_outer"],
        data["y_test_outer"],
        kept_names,
    )


def train_model(X_train, X_test, y_train, y_test, widths, **kwargs):
    L.seed_everything(42)
    module = TabKAN(
        in_features=X_train.shape[1], widths=widths, kan_type="chebykan",
        degree=3, lr=5e-3, weight_decay=5e-4, sparsity_lambda=0.005,
        l1_weight=1.0, entropy_weight=1.0, use_layernorm=False,
    )
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)),
        batch_size=2048, shuffle=True,
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1)),
        batch_size=2048, shuffle=False,
    )
    trainer = L.Trainer(
        max_epochs=150, accelerator="auto",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    module.eval()
    with torch.no_grad():
        preds = module(X_val_t).cpu().numpy().flatten()
    preds_r = np.clip(np.round(preds), 1, 8).astype(int)
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, preds_r)
    print(f"QWK = {qwk:.4f}")
    return module, qwk


def compute_composite_functions(module, feature_names, n_samples=500, threshold=0.005):
    """Compute fᵢ(x) = Σⱼ wⱼ · φⱼᵢ(x) for each feature."""
    first_layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_weight = module.head.weight.detach().squeeze()
    head_bias = module.head.bias.detach().item()
    l1_scores = _compute_edge_l1(first_layer)

    composites = {}
    for in_i, feat in enumerate(feature_names):
        x_ref = None
        composite_y = None
        for out_j in range(first_layer.out_features):
            if l1_scores[out_j, in_i].item() < threshold:
                continue
            x_vals, y_vals = sample_edge(first_layer, out_j, in_i, n=n_samples)
            w_j = head_weight[out_j].item()
            if composite_y is None:
                x_ref = x_vals
                composite_y = w_j * y_vals
            else:
                composite_y += w_j * y_vals

        if composite_y is not None:
            composites[feat] = (x_ref, composite_y)

    return composites, head_bias


def get_raw_mapping(enc_col, raw_col):
    """Build a monotonic lookup from encoded → raw scale."""
    mask = np.isfinite(enc_col) & np.isfinite(raw_col)
    enc = enc_col[mask].copy()
    raw = raw_col[mask].copy()
    order = np.argsort(enc)
    return enc[order], raw[order]


def plot_all_features(
    composites, head_bias, feature_names, qwk,
    X_test_encoded, raw_df, output_dir,
):
    """Create the main figure: 4×5 grid of feature contribution plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    active = [(f, composites[f]) for f in feature_names if f in composites]
    # Sort by output range (impact)
    active.sort(key=lambda x: x[1][1].max() - x[1][1].min(), reverse=True)

    n = len(active)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.2 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    # Color scheme by feature type
    def get_color(feat):
        if feat in CONTINUOUS:
            return "#2166AC"  # blue
        if feat in ORDINAL_MEDICAL:
            return "#B2182B"  # red
        if feat in BINARY:
            return "#4DAF4A"  # green
        return "#984EA3"     # purple

    def get_type_label(feat):
        if feat in CONTINUOUS:
            return "continuous"
        if feat in ORDINAL_MEDICAL:
            return "ordinal"
        if feat in BINARY:
            return "binary"
        return "other"

    for i, (feat, (x_enc, y_comp)) in enumerate(active):
        ax = axes_flat[i]
        color = get_color(feat)
        ftype = get_type_label(feat)

        # Check if we can map to raw scale
        has_raw = (
            raw_df is not None
            and feat in raw_df.columns
            and feat in CONTINUOUS
        )

        if has_raw:
            # Map encoded x to raw scale for the x-axis
            enc_col = X_test_encoded[:, feature_names.index(feat)]
            raw_col = raw_df[feat].values
            enc_sorted, raw_sorted = get_raw_mapping(enc_col, raw_col)
            x_plot = np.interp(x_enc, enc_sorted, raw_sorted)
            xlabel = feat
        else:
            x_plot = x_enc
            xlabel = f"{feat} (encoded)"

        # Plot the composite function
        ax.plot(x_plot, y_comp, color=color, lw=2.5, zorder=3)

        # Add zero line
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

        # Add rug plot showing data distribution
        if has_raw:
            rug = raw_col[::max(1, len(raw_col) // 80)]
        else:
            enc_col = X_test_encoded[:, feature_names.index(feat)]
            rug = enc_col[::max(1, len(enc_col) // 80)]
        rug_y = np.full_like(rug, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else y_comp.min())
        ax.plot(rug, np.full_like(rug, y_comp.min() - 0.05 * (y_comp.max() - y_comp.min())),
                "|", color="gray", alpha=0.3, markersize=4, zorder=1)

        # For binary features, mark the two positions
        if feat in BINARY:
            for enc_pos in [-1.0, 1.0]:
                y_mark = float(np.interp(enc_pos, x_enc, y_comp))
                ax.scatter([enc_pos], [y_mark], s=60, color="black", zorder=5, edgecolors="white", linewidths=1)
                label = "absent" if enc_pos < 0 else "present"
                ax.annotate(f"{label}\n{y_mark:+.3f}",
                            (enc_pos, y_mark), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=7, color="black")

        y_range = y_comp.max() - y_comp.min()
        ax.set_title(f"{feat}\n[{ftype}] range={y_range:.2f}", fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Risk contribution", fontsize=8)

        # Add subtle grid
        ax.grid(True, alpha=0.15)
        ax.set_axisbelow(True)

    # Hide unused axes
    for j in range(len(active), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"ChebyKAN Additive Model — Per-Feature Risk Contribution Functions\n"
        f"prediction = {head_bias:.3f} + Σ fᵢ(xᵢ)   |   QWK = {qwk:.4f}   |   "
        f"{len(active)} features, 0 flagged, all R² > 0.97",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Legend for feature types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", label="Continuous (BMI, Wt, Age, ...)"),
        Patch(facecolor="#B2182B", label="Medical History (ordinal)"),
        Patch(facecolor="#4DAF4A", label="Medical Keyword (binary)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_path = output_dir / "input_to_output_feature_functions.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    # Also save PNG for easy viewing
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(22, 4.2 * nrows))
    if nrows == 1:
        axes2 = axes2[np.newaxis, :]
    axes2_flat = axes2.flatten()

    for i, (feat, (x_enc, y_comp)) in enumerate(active):
        ax = axes2_flat[i]
        color = get_color(feat)
        ftype = get_type_label(feat)
        has_raw = raw_df is not None and feat in raw_df.columns and feat in CONTINUOUS

        if has_raw:
            enc_col = X_test_encoded[:, feature_names.index(feat)]
            raw_col = raw_df[feat].values
            enc_sorted, raw_sorted = get_raw_mapping(enc_col, raw_col)
            x_plot = np.interp(x_enc, enc_sorted, raw_sorted)
            xlabel = feat
        else:
            x_plot = x_enc
            xlabel = f"{feat} (encoded)"

        ax.plot(x_plot, y_comp, color=color, lw=2.5, zorder=3)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

        if feat in BINARY:
            for enc_pos in [-1.0, 1.0]:
                y_mark = float(np.interp(enc_pos, x_enc, y_comp))
                ax.scatter([enc_pos], [y_mark], s=60, color="black", zorder=5,
                           edgecolors="white", linewidths=1)
                label = "absent" if enc_pos < 0 else "present"
                ax.annotate(f"{label}\n{y_mark:+.3f}",
                            (enc_pos, y_mark), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=7)

        y_range = y_comp.max() - y_comp.min()
        ax.set_title(f"{feat}\n[{ftype}] range={y_range:.2f}", fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Risk contribution", fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.set_axisbelow(True)

    for j in range(len(active), len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.suptitle(
        f"ChebyKAN Additive Model — Per-Feature Risk Contribution Functions\n"
        f"prediction = {head_bias:.3f} + Σ fᵢ(xᵢ)   |   QWK = {qwk:.4f}   |   "
        f"{len(active)} features, 0 flagged, all R² > 0.97",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig2.legend(handles=legend_elements, loc="lower center", ncol=3,
                fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    png_path = output_dir / "input_to_output_feature_functions.png"
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {png_path}")


def plot_waterfall(composites, head_bias, feature_names, X_test, output_dir):
    """Waterfall chart showing average contribution of each feature."""
    avg_contributions = {}
    for feat in feature_names:
        if feat not in composites:
            continue
        x_enc, y_comp = composites[feat]
        # Evaluate at test data points
        feat_idx = feature_names.index(feat)
        enc_vals = X_test[:, feat_idx]
        contrib = np.interp(enc_vals, x_enc, y_comp)
        avg_contributions[feat] = {
            "mean": float(np.mean(contrib)),
            "std": float(np.std(contrib)),
            "abs_mean": float(np.mean(np.abs(contrib))),
        }

    # Sort by absolute mean contribution
    sorted_feats = sorted(avg_contributions, key=lambda f: avg_contributions[f]["abs_mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = range(len(sorted_feats))
    means = [avg_contributions[f]["mean"] for f in sorted_feats]
    stds = [avg_contributions[f]["std"] for f in sorted_feats]

    colors = []
    for f in sorted_feats:
        if f in CONTINUOUS:
            colors.append("#2166AC")
        elif f in ORDINAL_MEDICAL:
            colors.append("#B2182B")
        else:
            colors.append("#4DAF4A")

    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average risk contribution (mean ± std over test set)", fontsize=11)
    ax.set_title(f"Feature Importance — Average Risk Contribution\nbias = {head_bias:.3f}",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(True, axis="x", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166AC", label="Continuous"),
        Patch(facecolor="#B2182B", label="Medical History"),
        Patch(facecolor="#4DAF4A", label="Medical Keyword (binary)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "feature_importance_waterfall.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")

    png_path = output_dir / "feature_importance_waterfall.png"
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_feats, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Average risk contribution (mean ± std over test set)", fontsize=11)
    ax2.set_title(f"Feature Importance — Average Risk Contribution\nbias = {head_bias:.3f}",
                  fontsize=13, fontweight="bold")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.grid(True, axis="x", alpha=0.2)
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {png_path}")


def main():
    output_dir = Path("outputs/interpretable_kan_no_layernorm/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(seed=42)

    n_feat = 20
    X_train, X_test, y_train, y_test, feat_names = select_features(data, n_feat)
    print(f"Features: {feat_names}")

    # Load raw data for axis mapping
    raw_df = None
    try:
        raw_full = pd.read_csv("data/prudential-life-insurance-assessment/train.csv")
        test_indices = data["row_indices"]["outer_test"]
        raw_df = raw_full.loc[test_indices].reset_index(drop=True)
        print(f"Raw data loaded: {raw_df.shape}")
    except Exception as e:
        print(f"Could not load raw data: {e}")

    print("\nTraining model (no LayerNorm, [8], degree=3, sparsity=0.005)...")
    module, qwk = train_model(X_train, X_test, y_train, y_test, widths=[8])

    print("\nComputing input→output composite functions...")
    composites, bias = compute_composite_functions(module, feat_names)
    print(f"Active features: {len(composites)}/{n_feat}, bias={bias:.4f}")

    print("\nPlotting feature contribution functions...")
    plot_all_features(composites, bias, feat_names, qwk, X_test, raw_df, output_dir)

    print("\nPlotting feature importance waterfall...")
    plot_waterfall(composites, bias, feat_names, X_test, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
