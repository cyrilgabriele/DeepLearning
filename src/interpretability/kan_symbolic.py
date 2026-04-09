"""Issues 06 & 07 — Symbolic regression on surviving KAN edges.

For each active (non-pruned) edge in a trained KAN, samples the learned
1-D activation and fits a closed-form symbolic expression.

Usage:
    uv run python -m src.interpretability.kan_symbolic \
        --pruned-checkpoint outputs/chebykan_pruned_module.pt \
        --pruning-summary   outputs/chebykan_pruning_summary.json \
        --config            configs/chebykan_experiment.yaml \
        --eval-features     outputs/X_eval.parquet \
        --flavor            chebykan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from configs import ExperimentConfig


# ── Edge activation sampler ───────────────────────────────────────────────────

def _sample_chebykan_edge(layer, out_idx: int, in_idx: int, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_norm, y_output) for a single ChebyKAN edge.

    x_norm = tanh(linspace(-3, 3, n)), range ≈ [-0.995, +0.995].
    This covers binary feature encoded positions at ±1.0 and is the
    value to pass directly into encode_to_raw_lookup.
    """
    import torch
    x = torch.linspace(-3.0, 3.0, n)        # pre-tanh, extended domain
    x_norm = torch.tanh(x)                   # encoded domain ≈ [-0.995, +0.995]

    coeffs = layer.cheby_coeffs[out_idx, in_idx, :].detach()  # (degree+1,)
    base_w = layer.base_weight[out_idx, in_idx].detach()

    cheby = [torch.ones(n), x_norm]
    for _ in range(2, layer.degree + 1):
        cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
    basis = torch.stack(cheby, dim=-1)  # (n, degree+1)

    y = (basis * coeffs).sum(dim=-1) + base_w * x_norm
    return x_norm.numpy(), y.detach().numpy()   # ← return x_norm, not x


def _sample_fourierkan_edge(layer, out_idx: int, in_idx: int, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_norm, y_output) for a single FourierKAN edge.

    x_norm = tanh(linspace(-3, 3, n)), range ≈ [-0.995, +0.995].
    """
    import torch
    import math

    x = torch.linspace(-3.0, 3.0, n)
    x_norm = torch.tanh(x)
    x_scaled = (x_norm + 1) * math.pi
    k = torch.arange(1, layer.grid_size + 1, dtype=torch.float32)

    a = layer.fourier_a[out_idx, in_idx, :].detach()
    b = layer.fourier_b[out_idx, in_idx, :].detach()
    base_w = layer.base_weight[out_idx, in_idx].detach()

    x_k = x_scaled.unsqueeze(-1) * k
    y = (torch.cos(x_k) * a + torch.sin(x_k) * b).sum(dim=-1) + base_w * x_norm
    return x_norm.numpy(), y.detach().numpy()   # ← return x_norm, not x


def sample_edge(layer, out_idx: int, in_idx: int, n: int = 1000):
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    if isinstance(layer, ChebyKANLayer):
        return _sample_chebykan_edge(layer, out_idx, in_idx, n)
    if isinstance(layer, FourierKANLayer):
        return _sample_fourierkan_edge(layer, out_idx, in_idx, n)
    raise TypeError(f"Unsupported layer: {type(layer)}")


# ── Symbolic fitting ──────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0


def _fit_scipy_candidates(x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
    """Fit a library of candidate formulas using scipy and return (formula, r²).

    Selection uses a BIC-penalised score (Smits & Kotanchek 2005, Springer GPTP-II):
        score = R² - k · log(n) / n
    where k = number of free parameters and n = number of sample points.
    This prefers simpler formulas when the R² gain from added complexity is small,
    producing more interpretable results without sacrificing fit quality.

    Candidate library includes Fourier harmonic pairs (k=1..4) recommended by
    Cranmer (2023) arXiv:2305.01582 and Liu et al. (2024) arXiv:2404.19756 to
    improve symbolic fit quality for FourierKAN edges.
    """
    from scipy.optimize import curve_fit

    # (name, function, p0, n_params)
    candidates: list[tuple[str, object, list, int]] = [
        ("a*x + b",                  lambda x, a, b: a * x + b,                               [1., 0.],         2),
        ("a*x^2 + b*x + c",          lambda x, a, b, c: a * x**2 + b * x + c,                 [1., 0., 0.],     3),
        ("a*x^3 + b*x^2 + c*x + d",  lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,         [1., 0., 0., 0.], 4),
        ("a*|x| + b",                lambda x, a, b: a * np.abs(x) + b,                        [1., 0.],         2),
        ("a*sqrt(|x|) + b",          lambda x, a, b: a * np.sqrt(np.abs(x)) + b,               [1., 0.],         2),
        ("a*log(|x|+1) + b",         lambda x, a, b: a * np.log(np.abs(x) + 1) + b,            [1., 0.],         2),
        ("a*exp(x) + b",             lambda x, a, b: a * np.exp(np.clip(x, -5, 5)) + b,        [1., 0.],         2),
        ("a*sin(x) + b",             lambda x, a, b: a * np.sin(x) + b,                        [1., 0.],         2),
        ("a*sin(2*x) + b",           lambda x, a, b: a * np.sin(2 * x) + b,                    [1., 0.],         2),
        ("a*cos(x) + b",             lambda x, a, b: a * np.cos(x) + b,                        [1., 0.],         2),
        ("a (constant)",             lambda x, a: np.full_like(x, a),                           [0.],             1),
        # Fourier harmonic pairs k=1..4 (Cranmer 2023 / Liu et al. 2024)
        ("a*sin(x) + b*cos(x)",      lambda x, a, b: a * np.sin(x) + b * np.cos(x),            [1., 1.],         2),
        ("a*sin(2*x) + b*cos(2*x)",  lambda x, a, b: a * np.sin(2*x) + b * np.cos(2*x),        [1., 1.],         2),
        ("a*sin(3*x) + b*cos(3*x)",  lambda x, a, b: a * np.sin(3*x) + b * np.cos(3*x),        [1., 1.],         2),
        ("a*sin(4*x) + b*cos(4*x)",  lambda x, a, b: a * np.sin(4*x) + b * np.cos(4*x),        [1., 1.],         2),
    ]

    n = len(x)
    log_n_over_n = np.log(n) / n  # BIC penalty scale factor

    best_formula, best_bic, best_r2 = "a (constant)", -np.inf, 0.0
    for name, func, p0, k in candidates:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=2000)
            y_pred = func(x, *popt)
            r2 = _r2(y, y_pred)
            bic_score = r2 - k * log_n_over_n
            if bic_score > best_bic:
                best_bic = bic_score
                best_r2 = r2
                best_formula = name
        except Exception:
            continue

    return best_formula, best_r2


def _fit_pysr(x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
    """Fit using PySR (requires Julia). Falls back to scipy if unavailable."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        return _fit_scipy_candidates(x, y)

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", "cube", "exp", "log", "sqrt", "abs", "sin", "cos"],
        maxsize=15,
        timeout_in_seconds=60,
        verbosity=0,
        progress=False,
    )
    try:
        model.fit(x.reshape(-1, 1), y)
        best = model.get_best()
        formula = str(best["equation"])
        y_pred = model.predict(x.reshape(-1, 1))
        r2 = _r2(y, y_pred)
        return formula, r2
    except Exception:
        return _fit_scipy_candidates(x, y)


def fit_symbolic_edge(
    x: np.ndarray,
    y: np.ndarray,
    use_pysr: bool = False,
    pysr_fallback_threshold: float = 0.90,
) -> tuple[str, float]:
    """Fit a symbolic formula to edge samples, returning (formula, r²).

    When use_pysr=True, runs scipy first and only invokes PySR on edges where
    scipy achieves R² < pysr_fallback_threshold. This two-stage approach follows
    Cranmer (2023) arXiv:2305.01582: fixed libraries are cost-efficient when the
    functional form is anticipated; PySR is reserved for genuinely complex edges.
    """
    formula, r2 = _fit_scipy_candidates(x, y)
    if use_pysr and r2 < pysr_fallback_threshold:
        pysr_formula, pysr_r2 = _fit_pysr(x, y)
        if pysr_r2 > r2:
            return pysr_formula, pysr_r2
    return formula, r2


# ── Active edge detection ─────────────────────────────────────────────────────

def _quality_tier(r2: float) -> str:
    """Three-tier quality classification based on Liu et al. (2024) arXiv:2404.19756.

    clean      R² ≥ 0.99 — reliable for symbolic lock-in
    acceptable 0.90 ≤ R² < 0.99 — captures shape with residuals
    flagged    R² < 0.90 — formula does not describe the edge
    """
    if r2 >= 0.99:
        return "clean"
    if r2 >= 0.90:
        return "acceptable"
    return "flagged"


def _is_active(layer, out_idx: int, in_idx: int, threshold: float) -> bool:
    """True if the edge L1 norm meets the activity threshold."""
    from src.interpretability.kan_pruning import _compute_edge_l1
    l1 = _compute_edge_l1(layer)
    return bool(l1[out_idx, in_idx].item() >= threshold)


def _top_features_by_l1(
    l1_scores: "torch.Tensor",
    feature_names: list[str],
    top_n: int = 10,
) -> list[str]:
    """Return top_n input features ranked by sum of edge L1 norms.

    Uses L1 norm of activation functions as the importance signal, consistent
    with the pruning criterion from Liu et al. (2024) arXiv:2404.19756 and
    Akazan & Mbingui (2025) arXiv:2509.23366.
    """
    import torch
    per_input = l1_scores.sum(dim=0)
    n = min(top_n, len(feature_names), per_input.shape[0])
    ranked_idx = per_input.argsort(descending=True)[:n]
    return [feature_names[int(i)] for i in ranked_idx]


def _plot_activation_grid(
    module,
    df: pd.DataFrame,
    flavor: str,
    output_dir: Path,
    threshold: float,
    feat_types: dict,
    X_eval: pd.DataFrame | None,
    X_raw: pd.DataFrame | None,
) -> None:
    """TabKAN-style 2×5 grid: one subplot per top-10 input feature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.interpretability.utils.kan_coefficients import top_features_by_coefficients
    from src.interpretability.utils.style import (
        apply_paper_style, savefig_pdf, MODEL_COLORS,
        encode_to_raw_lookup, FEATURE_TYPE_MARKERS,
    )

    apply_paper_style()
    first_layer = next(
        (l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))), None
    )
    if first_layer is None:
        return

    feature_names = list(X_eval.columns) if X_eval is not None else []
    top_feats = top_features_by_coefficients(module, feature_names, top_n=10)

    model_color = MODEL_COLORS.get("ChebyKAN" if flavor == "chebykan" else "FourierKAN", "steelblue")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, top_feats):
        feat_i = feature_names.index(feat) if feat in feature_names else -1
        if feat_i < 0:
            ax.set_visible(False)
            continue

        ftype = feat_types.get(feat, "unknown")
        is_binary = ftype in ("binary", "missing_indicator")
        is_numeric_raw = ftype in ("continuous", "ordinal")
        has_raw = X_raw is not None and feat in X_raw.columns and is_numeric_raw

        active_edges = df[(df["layer"] == 0) & (df["input_feature"] == feat)]
        best_r2_row = None

        for _, erow in active_edges.iterrows():
            out_i = int(erow["edge_out"])
            x_norm, y_vals = sample_edge(first_layer, out_i, feat_i, n=300)
            x_plot = (encode_to_raw_lookup(feat, X_eval, X_raw, x_norm)
                      if (has_raw and not is_binary) else x_norm)
            ax.plot(x_plot, y_vals, color="gray", alpha=0.3, lw=1)
            if best_r2_row is None or erow["r_squared"] > best_r2_row["r_squared"]:
                best_r2_row = erow

        if best_r2_row is not None:
            out_i = int(best_r2_row["edge_out"])
            x_norm, y_vals = sample_edge(first_layer, out_i, feat_i, n=300)
            x_plot = (encode_to_raw_lookup(feat, X_eval, X_raw, x_norm)
                      if (has_raw and not is_binary) else x_norm)
            ax.plot(x_plot, y_vals, color=model_color, lw=2,
                    label=f"{best_r2_row['formula'][:20]}\nR²={best_r2_row['r_squared']:.3f}")
            if is_binary:
                for enc_pos in [-1.0, 1.0]:
                    y_mark = float(np.interp(enc_pos, x_norm, y_vals))
                    ax.axvline(enc_pos, color="gray", lw=1, ls="--", alpha=0.7)
                    ax.scatter([enc_pos], [y_mark], s=40, color="black", zorder=5)

        marker = FEATURE_TYPE_MARKERS.get(ftype, "")
        ax.set_title(f"{feat[:18]} {marker}", fontsize=8, fontweight="bold")
        x_lbl = "Encoded [-1,1]" if is_binary else ("Original scale" if has_raw else "Encoded")
        ax.set_xlabel(x_lbl, fontsize=7)
        ax.set_ylabel("Edge output", fontsize=7)
        if best_r2_row is not None:
            ax.legend(fontsize=5, loc="best")

    for ax in axes_flat[len(top_feats):]:
        ax.set_visible(False)

    plt.suptitle(
        f"{flavor.title()} — Learned Activation Functions (Top-10 Coefficient-Ranked Features)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    from src.interpretability.utils.paths import figures as fig_dir
    out = fig_dir(output_dir) / f"{flavor}_activations.pdf"
    savefig_pdf(fig, out)
    print(f"Saved → {out}")
    plt.close()


def _plot_feature_ranking(
    module,
    flavor: str,
    output_dir: Path,
    feature_names: list[str],
    feat_types: dict,
) -> None:
    """Horizontal bar chart of all features ranked by basis-coefficient magnitude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.interpretability.utils.kan_coefficients import coefficient_importance_from_layer
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.interpretability.utils.style import (
        apply_paper_style, savefig_pdf, FEATURE_TYPE_COLORS,
        FEATURE_TYPE_MARKERS, feature_type_label,
    )

    apply_paper_style()
    first_layer = next(
        (l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))), None
    )
    if first_layer is None:
        return

    coeff_df = coefficient_importance_from_layer(first_layer, feature_names)
    if coeff_df.empty:
        return
    sorted_feats = coeff_df["feature"].tolist()

    colors = [FEATURE_TYPE_COLORS.get(feat_types.get(f, "unknown"), "#AAAAAA") for f in sorted_feats]
    labels = [feature_type_label(f, feat_types) for f in sorted_feats]
    values = coeff_df["importance"].tolist()

    fig_height = max(6, len(sorted_feats) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(range(len(sorted_feats)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient magnitude (sum |coeff| across layer-0 outputs and basis terms)", fontsize=9)
    ax.set_title(f"{flavor.title()} — Feature Importance (Paper-Native Coefficients)",
                 fontsize=11, fontweight="bold")

    present_types = sorted(set(feat_types.get(f, "unknown") for f in sorted_feats))
    legend_elements = [
        Patch(facecolor=FEATURE_TYPE_COLORS.get(t, "#AAAAAA"),
              label=f"{t} {FEATURE_TYPE_MARKERS.get(t, '')}")
        for t in present_types
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    plt.tight_layout()

    from src.interpretability.utils.paths import figures as fig_dir
    out = fig_dir(output_dir) / f"{flavor}_feature_ranking.pdf"
    savefig_pdf(fig, out)
    print(f"Saved → {out}")
    plt.close()


# ── Main per-model runner ─────────────────────────────────────────────────────

def run(
    pruned_checkpoint_path: Path,
    pruning_summary_path: Path,
    config: ExperimentConfig,
    eval_features_path: Path,
    flavor: str,
    use_pysr: bool = False,
    output_dir: Path = Path("outputs"),
    n_samples: int = 1000,
    feat_types: dict | None = None,
    X_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    import torch
    from src.models.tabkan import TabKAN
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

    pruning_summary = json.loads(pruning_summary_path.read_text())
    threshold = pruning_summary["threshold"]

    X_eval = pd.read_parquet(eval_features_path)
    in_features = X_eval.shape[1]

    # Re-create module architecture and load pruned weights
    widths = config.model.resolved_hidden_widths()
    if flavor == "chebykan":
        module = TabKAN(in_features=in_features, widths=widths, kan_type="chebykan",
                        degree=config.model.degree or 3)
    else:
        module = TabKAN(in_features=in_features, widths=widths, kan_type="fourierkan")

    module.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
    module.eval()

    feature_names = list(X_eval.columns)
    from src.interpretability.utils.kan_coefficients import (
        coefficient_importance_from_layer,
        get_first_kan_layer,
    )
    records = []
    layer_idx = 0

    for layer in module.kan_layers:
        if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            continue

        # Precompute all edge L1 norms once per layer (avoids O(n²) recomputation)
        from src.interpretability.kan_pruning import _compute_edge_l1
        variances = _compute_edge_l1(layer)
        n_active = int((variances >= threshold).sum().item())
        print(f"Layer {layer_idx}: {layer.in_features}→{layer.out_features} edges, {n_active} active")
        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                if variances[out_i, in_i].item() < threshold:
                    continue

                x_vals, y_vals = sample_edge(layer, out_i, in_i, n=n_samples)
                formula, r2 = fit_symbolic_edge(x_vals, y_vals, use_pysr=use_pysr)

                input_feat = feature_names[in_i] if layer_idx == 0 and in_i < len(feature_names) else f"h{in_i}"
                records.append({
                    "layer": layer_idx,
                    "edge_in": in_i,
                    "edge_out": out_i,
                    "input_feature": input_feat,
                    "formula": formula,
                    "r_squared": round(r2, 6),
                    "flagged": r2 < 0.90,
                    "quality_tier": _quality_tier(r2),
                })

        layer_idx += 1

    df = pd.DataFrame(records)
    from src.interpretability.utils.paths import data as data_dir, figures as fig_dir
    out_path = data_dir(output_dir) / f"{flavor}_symbolic_fits.csv"
    df.to_csv(out_path, index=False)

    first_layer = get_first_kan_layer(module)
    coeff_frame = coefficient_importance_from_layer(first_layer, feature_names)
    coeff_out_path = data_dir(output_dir) / f"{flavor}_coefficient_importance.csv"
    coeff_frame.to_csv(coeff_out_path, index=False)

    n_flagged = int(df["flagged"].sum())
    print(f"\n{flavor}: {len(df)} active edges, {n_flagged} flagged (R² < 0.90)")
    print(f"Mean R²: {df['r_squared'].mean():.4f}  Median: {df['r_squared'].median():.4f}")
    print(f"Saved → {out_path}")
    print(f"Saved coefficient ranking → {coeff_out_path}")

    # ── Load feat_types and X_raw if not provided ─────────────────────────────
    if not feat_types:
        ft_path = Path(output_dir) / "reports" / "feature_types.json"
        if ft_path.exists():
            feat_types = json.loads(ft_path.read_text())
    if X_raw is None:
        xr_path = Path(output_dir) / "data" / "X_eval_raw.parquet"
        if xr_path.exists():
            X_raw = pd.read_parquet(xr_path)

    # ── TabKAN-style activation grid and feature ranking ──────────────────────
    _plot_activation_grid(module, df, flavor, output_dir, threshold,
                          feat_types or {}, X_eval, X_raw)
    _plot_feature_ranking(module, flavor, output_dir, feature_names, feat_types or {})

    return df


def _plot_example(module, df: pd.DataFrame, flavor: str, output_dir: Path, threshold: float) -> None:
    """Plot one representative edge: raw spline vs. symbolic fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from scipy.optimize import curve_fit

    if df.empty:
        return

    # Pick the edge with the highest R²
    best_row = df.loc[df["r_squared"].idxmax()]
    layer_idx = int(best_row["layer"])
    out_i = int(best_row["edge_out"])
    in_i = int(best_row["edge_in"])

    # Find the correct layer
    current = 0
    target_layer = None
    for layer in module.kan_layers:
        if isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            if current == layer_idx:
                target_layer = layer
                break
            current += 1

    if target_layer is None:
        return

    x_vals, y_vals = sample_edge(target_layer, out_i, in_i, n=1000)

    # Re-fit the winning candidate to get predicted values
    from src.interpretability.kan_symbolic import _fit_scipy_candidates
    _, _ = _fit_scipy_candidates(x_vals, y_vals)  # warm-up

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, y_vals, lw=1.5, label="Learned activation", color="steelblue")

    # Overlay symbolic fit using numpy eval of formula if simple enough
    formula = best_row["formula"]
    try:
        x = x_vals
        y_sym = eval(  # noqa: S307 - controlled formula string
            formula.replace("^", "**")
                   .replace("sqrt", "np.sqrt")
                   .replace("log", "np.log")
                   .replace("exp", "np.exp")
                   .replace("sin", "np.sin")
                   .replace("cos", "np.cos")
                   .replace("abs", "np.abs")
        )
        ax.plot(x_vals, y_sym, lw=1.5, linestyle="--",
                label=f"Symbolic: {formula}\nR²={best_row['r_squared']:.3f}", color="tomato")
    except Exception:
        pass

    ax.set_xlabel("Input (normalized)")
    ax.set_ylabel("Edge output")
    ax.set_title(f"{flavor} — Layer {layer_idx}, edge ({in_i}→{out_i})\nInput: {best_row['input_feature']}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    from src.interpretability.utils import paths as _paths
    fig_path = _paths.figures(output_dir) / f"{flavor}_symbolic_example.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved example plot → {fig_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symbolic regression on KAN edges")
    p.add_argument("--pruned-checkpoint", type=Path, required=True,
                   help="e.g. outputs/models/chebykan_pruned_module.pt")
    p.add_argument("--pruning-summary", type=Path, required=True,
                   help="e.g. outputs/reports/chebykan_pruning_summary.json")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--eval-features-raw", type=Path, default=None)
    p.add_argument("--flavor", choices=["chebykan", "fourierkan"], required=True)
    p.add_argument("--use-pysr", action="store_true", help="Use PySR (requires Julia)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from configs import load_experiment_config

    X_raw_arg = pd.read_parquet(args.eval_features_raw) if args.eval_features_raw else None
    run(
        args.pruned_checkpoint,
        args.pruning_summary,
        load_experiment_config(args.config),
        args.eval_features,
        args.flavor,
        args.use_pysr,
        args.output_dir,
        X_raw=X_raw_arg,
    )
