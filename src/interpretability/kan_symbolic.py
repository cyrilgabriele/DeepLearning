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


# ── Edge activation sampler ───────────────────────────────────────────────────

def _sample_chebykan_edge(layer, out_idx: int, in_idx: int, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_input, y_output) for a single ChebyKAN edge."""
    import torch
    x = torch.linspace(-1.0, 1.0, n)
    x_norm = torch.tanh(x)

    coeffs = layer.cheby_coeffs[out_idx, in_idx, :].detach()  # (degree+1,)
    base_w = layer.base_weight[out_idx, in_idx].detach()

    cheby = [torch.ones(n), x_norm]
    for _ in range(2, layer.degree + 1):
        cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
    basis = torch.stack(cheby, dim=-1)  # (n, degree+1)

    y = (basis * coeffs).sum(dim=-1) + base_w * x
    return x.numpy(), y.detach().numpy()


def _sample_fourierkan_edge(layer, out_idx: int, in_idx: int, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_input, y_output) for a single FourierKAN edge."""
    import torch
    import math

    x = torch.linspace(-1.0, 1.0, n)
    x_scaled = (torch.tanh(x) + 1) * math.pi
    k = torch.arange(1, layer.grid_size + 1, dtype=torch.float32)

    a = layer.fourier_a[out_idx, in_idx, :].detach()
    b = layer.fourier_b[out_idx, in_idx, :].detach()
    base_w = layer.base_weight[out_idx, in_idx].detach()

    x_k = x_scaled.unsqueeze(-1) * k
    y = (torch.cos(x_k) * a + torch.sin(x_k) * b).sum(dim=-1) + base_w * torch.tanh(x)
    return x.numpy(), y.detach().numpy()


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
    """Fit a library of candidate formulas using scipy and return (formula, r²)."""
    from scipy.optimize import curve_fit

    candidates: list[tuple[str, object, list]] = [
        ("a*x + b",            lambda x, a, b: a * x + b,                             [1., 0.]),
        ("a*x^2 + b*x + c",    lambda x, a, b, c: a * x**2 + b * x + c,               [1., 0., 0.]),
        ("a*x^3 + b*x^2 + c*x + d", lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, [1., 0., 0., 0.]),
        ("a*|x| + b",          lambda x, a, b: a * np.abs(x) + b,                      [1., 0.]),
        ("a*sqrt(|x|) + b",    lambda x, a, b: a * np.sqrt(np.abs(x)) + b,             [1., 0.]),
        ("a*log(|x|+1) + b",   lambda x, a, b: a * np.log(np.abs(x) + 1) + b,         [1., 0.]),
        ("a*exp(x) + b",       lambda x, a, b: a * np.exp(np.clip(x, -5, 5)) + b,     [1., 0.]),
        ("a*sin(x) + b",       lambda x, a, b: a * np.sin(x) + b,                     [1., 0.]),
        ("a*sin(2*x) + b",     lambda x, a, b: a * np.sin(2 * x) + b,                 [1., 0.]),
        ("a*cos(x) + b",       lambda x, a, b: a * np.cos(x) + b,                     [1., 0.]),
        ("a (constant)",       lambda x, a: np.full_like(x, a),                        [0.]),
    ]

    best_formula, best_r2 = "a (constant)", -np.inf
    for name, func, p0 in candidates:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=2000)
            y_pred = func(x, *popt)
            r2 = _r2(y, y_pred)
            if r2 > best_r2:
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
) -> tuple[str, float]:
    if use_pysr:
        return _fit_pysr(x, y)
    return _fit_scipy_candidates(x, y)


# ── Active edge detection ─────────────────────────────────────────────────────

def _is_active(layer, out_idx: int, in_idx: int, threshold: float) -> bool:
    """True if the edge has non-negligible output variance."""
    from src.interpretability.kan_pruning import _compute_edge_variances
    variances = _compute_edge_variances(layer)
    return bool(variances[out_idx, in_idx].item() >= threshold)


# ── Main per-model runner ─────────────────────────────────────────────────────

def run(
    pruned_checkpoint_path: Path,
    pruning_summary_path: Path,
    config_path: Path,
    eval_features_path: Path,
    flavor: str,
    use_pysr: bool = False,
    output_dir: Path = Path("outputs"),
    n_samples: int = 1000,
) -> pd.DataFrame:
    import torch
    from src.configs import load_experiment_config
    from src.models.tabkan import TabKAN
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

    pruning_summary = json.loads(pruning_summary_path.read_text())
    threshold = pruning_summary["threshold"]

    cfg = load_experiment_config(config_path)
    X_eval = pd.read_parquet(eval_features_path)
    in_features = X_eval.shape[1]

    # Re-create module architecture and load pruned weights
    if flavor == "chebykan":
        widths = [cfg.model.width] * cfg.model.depth
        module = TabKAN(in_features=in_features, widths=widths, kan_type="chebykan",
                        degree=cfg.model.degree or 3)
    else:
        widths = [cfg.model.width] * cfg.model.depth
        module = TabKAN(in_features=in_features, widths=widths, kan_type="fourierkan")

    module.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
    module.eval()

    feature_names = list(X_eval.columns)
    records = []
    layer_idx = 0

    for layer in module.kan_layers:
        if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            continue

        # Precompute all edge variances once per layer (avoids O(n²) recomputation)
        from src.interpretability.kan_pruning import _compute_edge_variances
        variances = _compute_edge_variances(layer)
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
                })

        layer_idx += 1

    df = pd.DataFrame(records)
    from src.interpretability.paths import data as data_dir, figures as fig_dir
    out_path = data_dir(output_dir) / f"{flavor}_symbolic_fits.csv"
    df.to_csv(out_path, index=False)

    n_flagged = int(df["flagged"].sum())
    print(f"\n{flavor}: {len(df)} active edges, {n_flagged} flagged (R² < 0.90)")
    print(f"Mean R²: {df['r_squared'].mean():.4f}  Median: {df['r_squared'].median():.4f}")
    print(f"Saved → {out_path}")

    # ── Example visualization ─────────────────────────────────────────────────
    _plot_example(module, df, flavor, output_dir, threshold)

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
    from src.interpretability import paths as _paths
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
    p.add_argument("--flavor", choices=["chebykan", "fourierkan"], required=True)
    p.add_argument("--use-pysr", action="store_true", help="Use PySR (requires Julia)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.pruned_checkpoint,
        args.pruning_summary,
        args.config,
        args.eval_features,
        args.flavor,
        args.use_pysr,
        args.output_dir,
    )
