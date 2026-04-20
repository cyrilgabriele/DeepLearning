#!/usr/bin/env python3
"""Full interpretability plot suite for the no-LayerNorm KAN additive model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from scipy.optimize import curve_fit

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
import matplotlib.gridspec as gridspec

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

CANDIDATES = {
    "a*x + b":                  (lambda x, a, b: a*x + b, 2),
    "a*x^2 + b*x + c":         (lambda x, a, b, c: a*x**2 + b*x + c, 3),
    "a*x^3 + b*x^2 + c*x + d": (lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, 4),
    "a*exp(x) + b":             (lambda x, a, b: a*np.exp(np.clip(x, -5, 5)) + b, 2),
    "a*cos(x) + b":             (lambda x, a, b: a*np.cos(x) + b, 2),
    "a*sin(x) + b":             (lambda x, a, b: a*np.sin(x) + b, 2),
    "a*sin(2*x) + b":           (lambda x, a, b: a*np.sin(2*x) + b, 2),
    "a*sin(2*x) + b*cos(2*x)":  (lambda x, a, b: a*np.sin(2*x) + b*np.cos(2*x), 2),
    "a*sin(x) + b*cos(x)":     (lambda x, a, b: a*np.sin(x) + b*np.cos(x), 2),
    "a*sin(3*x) + b*cos(3*x)":  (lambda x, a, b: a*np.sin(3*x) + b*np.cos(3*x), 2),
    "a*|x| + b":                (lambda x, a, b: a*np.abs(x) + b, 2),
    "a*sqrt(|x|) + b":          (lambda x, a, b: a*np.sqrt(np.abs(x)) + b, 2),
    "a*log(|x|+1) + b":         (lambda x, a, b: a*np.log(np.abs(x) + 1) + b, 2),
    "a (constant)":             (lambda x, a: np.full_like(x, float(a)), 1),
}


def setup():
    """Load data, train model, return everything needed."""
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
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)

    L.seed_everything(42)
    module = TabKAN(
        in_features=20, widths=[8], kan_type="chebykan", degree=3,
        lr=5e-3, weight_decay=5e-4, sparsity_lambda=0.005,
        l1_weight=1.0, entropy_weight=1.0, use_layer_norm=False,
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
        kan_preds_raw = module(X_val_t).cpu().numpy().flatten()
    kan_preds = np.clip(np.round(kan_preds_raw), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, kan_preds)
    print(f"QWK = {qwk:.4f}")

    first_layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_w = module.head.weight.detach().squeeze().numpy()
    head_b = module.head.bias.detach().item()
    l1 = _compute_edge_l1(first_layer)

    # Build composite functions and symbolic fits
    composites = {}
    sym_fits = {}
    THR = 0.005
    for in_i, feat in enumerate(feat_names):
        x_ref, comp_y = None, None
        for out_j in range(first_layer.out_features):
            if l1[out_j, in_i].item() < THR:
                continue
            xv, yv = sample_edge(first_layer, out_j, in_i, n=1000)
            w = head_w[out_j]
            if comp_y is None:
                x_ref, comp_y = xv, w * yv
            else:
                comp_y += w * yv
        if comp_y is not None:
            composites[feat] = (x_ref, comp_y)
            fname, r2 = fit_symbolic_edge(x_ref, comp_y,
                                            max_poly_degree=getattr(first_layer, "degree", 3))
            popt = None
            if fname in CANDIDATES:
                func, n_p = CANDIDATES[fname]
                try:
                    popt, _ = curve_fit(func, x_ref, comp_y, p0=[1.0] * n_p, maxfev=3000)
                except Exception:
                    pass
            sym_fits[feat] = (fname, r2, popt)

    # Per-sample contributions (using tanh normalization)
    n_test = X_test.shape[0]
    contribs = np.zeros((n_test, len(feat_names)))
    for in_i, feat in enumerate(feat_names):
        if feat not in composites:
            continue
        x_ref, comp_y = composites[feat]
        test_x_norm = np.tanh(X_test[:, in_i])
        contribs[:, in_i] = np.interp(test_x_norm, x_ref, comp_y)

    # Load raw data
    raw_df = None
    try:
        raw_full = pd.read_csv("data/prudential-life-insurance-assessment/train.csv")
        raw_df = raw_full.loc[data["row_indices"]["outer_test"]].reset_index(drop=True)
    except Exception:
        pass

    return {
        "module": module, "feat_names": feat_names, "X_test": X_test,
        "y_true": y_true, "kan_preds": kan_preds, "kan_preds_raw": kan_preds_raw,
        "qwk": qwk, "composites": composites, "sym_fits": sym_fits,
        "head_b": head_b, "contribs": contribs, "raw_df": raw_df,
        "first_layer": first_layer, "head_w": head_w,
    }


# ── Plot 1: Individual prediction decomposition ─────────────────────────────

def plot_individual_decomposition(ctx, output_dir):
    """Waterfall charts showing why specific patients got their risk levels."""
    feat_names = ctx["feat_names"]
    contribs = ctx["contribs"]
    y_true = ctx["y_true"]
    kan_preds = ctx["kan_preds"]
    head_b = ctx["head_b"]
    raw_df = ctx["raw_df"]

    # Pick diverse examples: one from each risk level that exists
    examples = []
    for target_level in [1, 3, 5, 7, 8]:
        mask = y_true == target_level
        if not mask.any():
            continue
        # Pick the one closest to mean prediction for that level
        idxs = np.where(mask)[0]
        pred_vals = kan_preds[idxs]
        best = idxs[np.argmin(np.abs(pred_vals - target_level))]
        examples.append(best)

    n_ex = len(examples)
    fig, axes = plt.subplots(1, n_ex, figsize=(5 * n_ex, 8), sharey=False)
    if n_ex == 1:
        axes = [axes]

    for ax, idx in zip(axes, examples):
        c = contribs[idx]
        # Sort by absolute contribution
        order = np.argsort(np.abs(c))[::-1]
        active = [(feat_names[j], c[j]) for j in order if abs(c[j]) > 0.01]

        names = [x[0] for x in active]
        vals = [x[1] for x in active]

        colors = ["#2166AC" if v > 0 else "#B2182B" for v in vals]
        y_pos = range(len(names))
        ax.barh(y_pos, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(True, axis="x", alpha=0.2)

        total = head_b + c.sum()
        pred = int(np.clip(np.round(total), 1, 8))
        actual = y_true[idx]

        # Show raw feature values if available
        if raw_df is not None:
            for i, (name, val) in enumerate(active):
                if name in raw_df.columns:
                    raw_val = raw_df.iloc[idx][name]
                    if pd.notna(raw_val):
                        ax.annotate(f"({raw_val:.1f})" if isinstance(raw_val, float) else f"({raw_val})",
                                    (0, i), textcoords="offset points",
                                    xytext=(-3, 0), ha="right", fontsize=6, color="gray")

        ax.set_xlabel("Risk contribution", fontsize=9)
        ax.set_title(
            f"Patient #{idx}\nActual: {actual}  Predicted: {pred}\n"
            f"(bias={head_b:.2f}, sum={total:.2f})",
            fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "Individual Prediction Decomposition — Why Each Patient Gets Their Risk Level",
        fontsize=13, fontweight="bold", y=1.02,
    )
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#2166AC", label="Increases risk"),
                 Patch(color="#B2182B", label="Decreases risk")],
        loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout()
    out = output_dir / "individual_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Plot 2: Binary keyword toggle effects ────────────────────────────────────

def plot_binary_toggle_effects(ctx, output_dir):
    """Bar chart: effect of each binary keyword being present vs absent."""
    composites = ctx["composites"]

    toggles = []
    for feat in sorted(BINARY):
        if feat not in composites:
            continue
        x_ref, comp_y = composites[feat]
        # Binary features: encoded as 0 (absent) and 1 (present)
        # After tanh: tanh(0)=0, tanh(1)=0.762
        y_absent = float(np.interp(np.tanh(0.0), x_ref, comp_y))
        y_present = float(np.interp(np.tanh(1.0), x_ref, comp_y))
        delta = y_present - y_absent
        toggles.append({"feature": feat, "absent": y_absent, "present": y_present, "delta": delta})

    toggles.sort(key=lambda t: abs(t["delta"]), reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: delta bar chart
    names = [t["feature"].replace("Medical_Keyword_", "KW_") for t in toggles]
    deltas = [t["delta"] for t in toggles]
    colors = ["#B2182B" if d > 0 else "#2166AC" for d in deltas]
    ax1.barh(range(len(names)), deltas, color=colors, alpha=0.85)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("Risk change when keyword is present", fontsize=11)
    ax1.set_title("Effect of Binary Keywords\n(present vs absent)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.2)

    for i, t in enumerate(toggles):
        ax1.annotate(f"{t['delta']:+.3f}", (t["delta"], i),
                     textcoords="offset points",
                     xytext=(5 if t["delta"] > 0 else -5, 0),
                     ha="left" if t["delta"] > 0 else "right",
                     fontsize=8, fontweight="bold")

    # Right: grouped bar showing absent vs present level
    x = np.arange(len(names))
    width = 0.35
    ax2.barh(x - width / 2, [t["absent"] for t in toggles], width, label="Absent", color="#4DAF4A", alpha=0.7)
    ax2.barh(x + width / 2, [t["present"] for t in toggles], width, label="Present", color="#E41A1C", alpha=0.7)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Risk contribution level", fontsize=11)
    ax2.set_title("Absolute Risk Contribution\n(absent vs present)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    out = output_dir / "binary_toggle_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Plot 3: Symbolic fit fidelity ────────────────────────────────────────────

def plot_symbolic_fidelity(ctx, output_dir):
    """Overlay exact learned curve vs symbolic formula for each feature."""
    composites = ctx["composites"]
    sym_fits = ctx["sym_fits"]
    feat_names = ctx["feat_names"]

    active = [f for f in feat_names if f in composites]
    n = len(active)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for i, feat in enumerate(active):
        ax = axes_flat[i]
        x_ref, comp_y = composites[feat]
        fname, r2, popt = sym_fits[feat]

        # Plot exact learned curve
        ax.plot(x_ref, comp_y, color="#2166AC", lw=2.5, label="Learned (exact)", zorder=3)

        # Plot symbolic fit
        if fname in CANDIDATES and popt is not None:
            func, _ = CANDIDATES[fname]
            sym_y = func(x_ref, *popt)
            ax.plot(x_ref, sym_y, color="#E41A1C", lw=1.5, ls="--", label="Symbolic fit", zorder=4)

            # Shade the residual
            ax.fill_between(x_ref, comp_y, sym_y, alpha=0.15, color="#E41A1C")

        ax.set_title(f"{feat}\nR²={r2:.4f}  [{fname[:25]}]", fontsize=8, fontweight="bold")
        ax.set_xlabel("tanh(x)", fontsize=7)
        ax.set_ylabel("f(x)", fontsize=7)
        ax.grid(True, alpha=0.15)

        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(len(active), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Symbolic Fit Fidelity — Exact Learned Curve vs Closed-Form Formula\n"
        "(blue = exact, red dashed = symbolic, shaded = residual)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = output_dir / "symbolic_fidelity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Plot 4: Prediction distribution by risk class ───────────────────────────

def plot_prediction_distribution(ctx, output_dir):
    """Violin/box plots of raw predictions for each actual risk level."""
    y_true = ctx["y_true"]
    preds_raw = ctx["kan_preds_raw"]
    qwk = ctx["qwk"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: violin plot of raw predictions by actual risk
    risk_levels = sorted(np.unique(y_true))
    data_by_level = [preds_raw[y_true == r] for r in risk_levels]

    parts = ax1.violinplot(data_by_level, positions=risk_levels, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2166AC")
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("#E41A1C")
    parts["cmedians"].set_color("black")

    # Add perfect prediction line
    ax1.plot(risk_levels, risk_levels, "k--", lw=1, alpha=0.5, label="Perfect prediction")
    ax1.set_xlabel("Actual Risk Level", fontsize=11)
    ax1.set_ylabel("Model Raw Prediction", fontsize=11)
    ax1.set_title(f"Prediction Distribution by Risk Level\nQWK = {qwk:.4f}", fontsize=12, fontweight="bold")
    ax1.set_xticks(risk_levels)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: confusion-style heatmap
    from sklearn.metrics import confusion_matrix
    kan_preds = ctx["kan_preds"]
    cm = confusion_matrix(y_true, kan_preds, labels=list(range(1, 9)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax2.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=0.6)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(range(1, 9))
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(range(1, 9))
    ax2.set_xlabel("Predicted Risk", fontsize=11)
    ax2.set_ylabel("Actual Risk", fontsize=11)
    ax2.set_title("Normalized Confusion Matrix", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(8):
        for j in range(8):
            val = cm_norm[i, j]
            if val > 0.01:
                color = "white" if val > 0.3 else "black"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax2, shrink=0.8)
    plt.tight_layout()
    out = output_dir / "prediction_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ── Plot 5: Cumulative feature importance ────────────────────────────────────

def plot_cumulative_importance(ctx, output_dir):
    """QWK achieved using top-1, top-2, ..., top-N features."""
    feat_names = ctx["feat_names"]
    contribs = ctx["contribs"]
    y_true = ctx["y_true"]
    head_b = ctx["head_b"]

    # Rank features by average absolute contribution
    avg_abs = np.mean(np.abs(contribs), axis=0)
    order = np.argsort(avg_abs)[::-1]
    ranked_feats = [feat_names[i] for i in order]

    qwks = []
    for k in range(1, len(ranked_feats) + 1):
        top_k_idx = order[:k]
        preds = head_b + contribs[:, top_k_idx].sum(axis=1)
        preds_r = np.clip(np.round(preds), 1, 8).astype(int)
        qwk_k = quadratic_weighted_kappa(y_true, preds_r)
        qwks.append(qwk_k)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(1, len(qwks) + 1), qwks, "o-", color="#2166AC", lw=2, markersize=6)

    # Annotate each point with feature name
    for k, (feat, qwk_val) in enumerate(zip(ranked_feats, qwks)):
        short = feat.replace("Medical_Keyword_", "KW_").replace("Medical_History_", "MH_")
        ax.annotate(
            f"+{short}",
            (k + 1, qwk_val),
            textcoords="offset points",
            xytext=(5, 8 if k % 2 == 0 else -12),
            fontsize=7, rotation=30,
            ha="left",
        )

    ax.set_xlabel("Number of features (cumulative, ranked by importance)", fontsize=11)
    ax.set_ylabel("QWK", fontsize=11)
    ax.set_title(
        "Cumulative Feature Importance — QWK vs Number of Features\n"
        "(features added in order of average absolute contribution)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(range(1, len(qwks) + 1))
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = output_dir / "cumulative_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main():
    output_dir = Path("outputs/interpretable_kan_no_layernorm/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up model and data...")
    ctx = setup()

    print("\n1. Individual prediction decomposition...")
    plot_individual_decomposition(ctx, output_dir)

    print("\n2. Binary keyword toggle effects...")
    plot_binary_toggle_effects(ctx, output_dir)

    print("\n3. Symbolic fit fidelity...")
    plot_symbolic_fidelity(ctx, output_dir)

    print("\n4. Prediction distribution by risk class...")
    plot_prediction_distribution(ctx, output_dir)

    print("\n5. Cumulative feature importance...")
    plot_cumulative_importance(ctx, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
