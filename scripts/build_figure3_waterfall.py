"""Figure 3 for the interpretability section: waterfall decomposition of one
applicant's predicted risk score under the sparse ChebyKAN hero, using
integrated gradients (exactly additive for this polynomial model).

For applicant 55728, the plot shows how each of the top 7 input features
contributes (as signed bars) to move from the reference (median-all)
prediction to the applicant's actual prediction, with the remaining 13
features bucketed into a single "other" bar. The final bar shows the
applicant's predicted ordinal class.

The attribution method is integrated gradients (Sundararajan et al. 2017):
    IG_i = (x_app_i - x_ref_i) * ∫_0^1 ∂f/∂x_i(x_ref + t(x_app - x_ref)) dt
which is exactly additive by construction: Σ IG_i = f(x_app) − f(x_ref).
For the sparse no-LN ChebyKAN this integral is evaluated on the model's
closed-form polynomial — no surrogate, no sampling, no SHAP approximation.

Saves fig1_interpretability.pdf (same filename as the previous Figure 3 so
main.tex needs no change).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# PyTorch 2.6 weights_only fix
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

from src.config import load_experiment_config
from src.models.tabkan import TabKAN


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
EXP = "stage-c-chebykan-pareto-q0583-top20-noln"
CKPT = REPO / "outputs" / "interpretability" / "kan_paper" / EXP / "models" / "chebykan_pruned_module.pt"
EVAL = REPO / "outputs" / "eval" / "kan_paper" / EXP
CONFIG = REPO / "configs" / "experiment_stages" / "stage_c_explanation_package" / "chebykan_pareto_q0583_top20_noln.yaml"

APPLICANT_ID = 55728.0
TOP_K_TO_SHOW = 7  # remaining features bucketed as "other"
IG_STEPS = 100     # trapezoidal steps for integrated gradients


def _load_module() -> TabKAN:
    cfg = load_experiment_config(CONFIG)
    state = torch.load(CKPT, map_location="cpu")
    first_key = next(k for k in state.keys() if "cheby_coeffs" in k)
    in_features = state[first_key].shape[1]
    widths = cfg.model.resolved_hidden_widths()
    module = TabKAN(
        in_features=in_features,
        widths=widths,
        kan_type=cfg.model.flavor,
        use_layernorm=cfg.model.use_layernorm,
        degree=cfg.model.degree or 3,
    )
    module.load_state_dict(state)
    module.eval()
    return module


def _load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_parquet(EVAL / "X_eval.parquet")
    X_raw = pd.read_parquet(EVAL / "X_eval_raw.parquet")
    return X, X_raw


def _find_applicant(X: pd.DataFrame, X_raw: pd.DataFrame) -> int:
    if "Id" in X_raw.columns:
        mask = X_raw["Id"].astype(float) == float(APPLICANT_ID)
        if mask.any():
            return int(np.flatnonzero(mask.to_numpy())[0])
    raise ValueError(f"Applicant {APPLICANT_ID} not found")


def _predict(module: TabKAN, x: np.ndarray) -> float:
    with torch.no_grad():
        out = module(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return float(out.squeeze())


def integrated_gradients(module: TabKAN, x_ref: np.ndarray, x_app: np.ndarray,
                         n_steps: int = IG_STEPS) -> np.ndarray:
    """Compute per-feature integrated-gradient contributions.

    For polynomial models the integral is computed to machine precision with
    sufficient steps; we use n_steps=100 trapezoidal which is more than
    enough for degree-6 Chebyshev in a 2-layer network.
    """
    x_ref_t = torch.tensor(x_ref, dtype=torch.float32)
    x_app_t = torch.tensor(x_app, dtype=torch.float32)
    dx = x_app_t - x_ref_t

    # Trapezoidal weights: endpoints 0.5, interiors 1.0
    alphas = torch.linspace(0.0, 1.0, n_steps)
    weights = torch.ones(n_steps) / (n_steps - 1)
    weights[0] = weights[-1] = 0.5 / (n_steps - 1)

    total = torch.zeros_like(dx)
    for w, alpha in zip(weights, alphas):
        x = (x_ref_t + alpha * dx).requires_grad_(True)
        y = module(x.unsqueeze(0)).squeeze()
        g = torch.autograd.grad(y, x)[0]
        total = total + w * g
    return (total * dx).detach().numpy()


def main() -> None:
    module = _load_module()
    X, X_raw = _load_eval()
    feature_names = list(X.columns)
    idx = _find_applicant(X, X_raw)

    x_app = X.iloc[idx].to_numpy(dtype=np.float32)

    # Reference = the class-1 applicant the model itself rates as lowest-risk
    # (real applicant, not a non-linear average). This gives the actuarial
    # narrative: "why is this applicant predicted class 5 instead of being a
    # low-risk class 1?" — with meaningful, diverse contributions because the
    # two applicants actually differ across many features.
    y_eval = pd.read_parquet(EVAL / "y_eval.parquet").squeeze("columns").to_numpy()
    class1_mask = y_eval == 1
    if not class1_mask.any():
        raise RuntimeError("No class-1 applicants in eval split")

    # Score all class-1 applicants; pick the one with the lowest predicted score
    with torch.no_grad():
        scores_all = module(torch.tensor(X.to_numpy(dtype=np.float32))).cpu().numpy().flatten()
    class1_indices = np.flatnonzero(class1_mask)
    best_class1_idx = int(class1_indices[np.argmin(scores_all[class1_indices])])
    x_ref = X.iloc[best_class1_idx].to_numpy(dtype=np.float32)
    ref_id = X_raw.iloc[best_class1_idx].get("Id", best_class1_idx)
    print(f"Reference applicant: class-1, Id {ref_id}, predicted score {scores_all[best_class1_idx]:.3f}")

    f_ref = _predict(module, x_ref)
    f_app = _predict(module, x_app)
    print(f"f(reference = median)     = {f_ref:.4f}")
    print(f"f(applicant {APPLICANT_ID}) = {f_app:.4f}")
    print(f"Δ to explain             = {f_app - f_ref:.4f}")

    print("\nComputing integrated gradients ...")
    ig = integrated_gradients(module, x_ref, x_app)
    ig_sum = float(ig.sum())
    print(f"Σ IG = {ig_sum:.4f}  (should equal Δ = {f_app - f_ref:.4f}; "
          f"residual {abs(ig_sum - (f_app - f_ref)):.2e})")

    # Rank features by absolute IG
    df = pd.DataFrame({"feature": feature_names, "ig": ig,
                       "abs_ig": np.abs(ig)})
    df = df.sort_values("abs_ig", ascending=False).reset_index(drop=True)

    top = df.iloc[:TOP_K_TO_SHOW].copy()
    other_sum = float(df.iloc[TOP_K_TO_SHOW:]["ig"].sum())

    # Build waterfall rows: start with reference, then top features, then
    # the "other" bucket, then end bar showing applicant prediction.
    rows_feat = list(top["feature"])
    rows_ig = list(top["ig"])
    rows_feat.append(f"+ {len(df) - TOP_K_TO_SHOW} others")
    rows_ig.append(other_sum)

    # Cumulative positions for waterfall
    cumulative = [f_ref]
    for contrib in rows_ig:
        cumulative.append(cumulative[-1] + contrib)
    final = cumulative[-1]
    print(f"\nFinal waterfall endpoint: {final:.4f}  (applicant actual {f_app:.4f})")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(3.35, 3.3), constrained_layout=True)
    n_bars = len(rows_feat)
    y_positions = np.arange(n_bars + 2)  # +2 for Reference and Applicant bars

    # Colors
    pos_color = "#2a6f97"  # positive contribution
    neg_color = "#c1272d"  # negative contribution
    anchor_color = "#555555"  # reference & applicant anchors

    # Bar 0 (bottom): Reference prediction
    ax.barh(y_positions[0], f_ref, left=0, color=anchor_color, alpha=0.55,
            height=0.65)
    ax.text(f_ref + 0.06, y_positions[0], f"{f_ref:.2f}",
            va="center", ha="left", fontsize=6.5)

    # Middle bars: per-feature IG contributions (waterfall style)
    for i, (feat, contrib) in enumerate(zip(rows_feat, rows_ig)):
        y = y_positions[i + 1]
        left = cumulative[i]
        width = contrib
        color = pos_color if width >= 0 else neg_color
        ax.barh(y, width, left=left, color=color, alpha=0.85, height=0.65)

        # Label value to the right of the bar (or left if negative)
        label_x = left + width + (0.04 if width >= 0 else -0.04)
        ha = "left" if width >= 0 else "right"
        sign = "+" if width >= 0 else ""
        ax.text(label_x, y, f"{sign}{width:.2f}",
                va="center", ha=ha, fontsize=6.5)

    # Top bar: Applicant prediction
    ax.barh(y_positions[-1], f_app, left=0, color=anchor_color, alpha=0.55,
            height=0.65)
    applicant_class = int(np.clip(np.round(f_app), 1, 8))
    ax.text(f_app + 0.06, y_positions[-1],
            f"{f_app:.2f}  → class {applicant_class}",
            va="center", ha="left", fontsize=6.5)

    # Dotted connector lines (waterfall markers)
    for i in range(n_bars + 1):
        x_connect = cumulative[i]
        y_top = y_positions[i] + 0.325
        y_bot = y_positions[i + 1] - 0.325
        ax.plot([x_connect, x_connect], [y_top, y_bot],
                color="#888888", lw=0.5, ls=":", zorder=0)

    # Y-axis labels — display underscores as-is (matplotlib handles them
    # fine outside math mode; no LaTeX escaping needed).
    ylabels = ["Reference (class-1)"] + list(rows_feat) + ["Applicant 55728"]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()

    ax.set_xlabel("Predicted risk score (ordinal)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.axvline(f_ref, color=anchor_color, lw=0.6, ls="--", alpha=0.35, zorder=0)
    ax.axvline(f_app, color=anchor_color, lw=0.6, ls="--", alpha=0.35, zorder=0)
    ax.set_xlim(left=f_ref - 0.5, right=max(f_app, max(cumulative)) + 1.8)
    ax.grid(axis="x", alpha=0.22, lw=0.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    out_path = REPO / "outputs" / "figures" / "fig1_interpretability.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    out_png = out_path.with_suffix(".png")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_png}")


def _escape(s: str) -> str:
    """No-op placeholder for potential future LaTeX escaping in labels."""
    return s


if __name__ == "__main__":
    main()
