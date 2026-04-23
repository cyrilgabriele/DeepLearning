#!/usr/bin/env python3
"""KAN vs GAM grid search: 63-experiment comparison with 3-level interpretability.

Trains ChebyKAN architectures spanning 1-layer GAM baselines and 2-layer KANs
with varying feature counts, widths, and sparsity levels, then extracts:

  Level 1  Per-edge exact formulas (sample_edge + fit_symbolic_edge)
  Level 2  Per-feature conditional response curves (full model forward pass)
  Level 3  Full model exact composition (Chebyshev coefficients + weights as JSON)

Usage:
    uv run python scripts/kan_vs_gam_search.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from scipy.optimize import curve_fit

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_kan_paper import KANPreprocessor
from src.models.tabkan import TabKAN
from src.models.kan_layers import ChebyKANLayer
from src.interpretability.kan_pruning import _compute_edge_l1
from src.interpretability.kan_symbolic import (
    sample_edge,
    fit_symbolic_edge,
    _quality_tier,
)
from src.metrics.qwk import quadratic_weighted_kappa


# ── Top features from coefficient importance (full [128,64] model) ────────────
TOP_FEATURES_RANKED = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]


# ── Candidate formula library for parameter refitting ─────────────────────────
_REFIT_CANDIDATES: dict[str, tuple] = {
    "a*x + b": (lambda x, a, b: a * x + b, ["a", "b"]),
    "a*x^2 + b*x + c": (lambda x, a, b, c: a * x**2 + b * x + c, ["a", "b", "c"]),
    "a*x^3 + b*x^2 + c*x + d": (
        lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        ["a", "b", "c", "d"],
    ),
    "a*|x| + b": (lambda x, a, b: a * np.abs(x) + b, ["a", "b"]),
    "a*cos(x) + b": (lambda x, a, b: a * np.cos(x) + b, ["a", "b"]),
    "a*sin(x) + b": (lambda x, a, b: a * np.sin(x) + b, ["a", "b"]),
    "a*sin(2*x) + b": (lambda x, a, b: a * np.sin(2 * x) + b, ["a", "b"]),
    "a*sin(2*x) + b*cos(2*x)": (
        lambda x, a, b: a * np.sin(2 * x) + b * np.cos(2 * x),
        ["a", "b"],
    ),
    "a*sin(x) + b*cos(x)": (
        lambda x, a, b: a * np.sin(x) + b * np.cos(x),
        ["a", "b"],
    ),
    "a*sin(3*x) + b*cos(3*x)": (
        lambda x, a, b: a * np.sin(3 * x) + b * np.cos(3 * x),
        ["a", "b"],
    ),
    "a*sin(4*x) + b*cos(4*x)": (
        lambda x, a, b: a * np.sin(4 * x) + b * np.cos(4 * x),
        ["a", "b"],
    ),
    "a*exp(x) + b": (
        lambda x, a, b: a * np.exp(np.clip(x, -5, 5)) + b,
        ["a", "b"],
    ),
    "a*log(|x|+1) + b": (
        lambda x, a, b: a * np.log(np.abs(x) + 1) + b,
        ["a", "b"],
    ),
    "a*sqrt(|x|) + b": (
        lambda x, a, b: a * np.sqrt(np.abs(x)) + b,
        ["a", "b"],
    ),
    "a (constant)": (lambda x, a: np.full_like(x, float(a)), ["a"]),
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    n_features: int
    hidden_widths: list[int]
    degree: int = 3
    sparsity_lambda: float = 0.0
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 150
    batch_size: int = 2048
    pruning_threshold: float = 0.01


@dataclass
class ExperimentResult:
    name: str
    n_features: int
    hidden_widths: list[int]
    degree: int
    sparsity_lambda: float
    qwk: float
    # Level 1: per-edge
    n_active_edges: int
    n_total_edges: int
    n_clean: int
    n_acceptable: int
    n_flagged: int
    mean_r2: float
    median_r2: float
    pct_clean: float
    pct_interpretable: float  # clean + acceptable
    # Level 2: per-feature conditional response
    n_features_interpretable: int
    n_features_total_active: int
    pct_features_interpretable: float
    level2_mean_r2: float
    # Level 3: full model composition (stored separately)
    has_level3: bool
    # Timing
    training_time_s: float
    # Detail data (not in summary CSV)
    edge_fits: Optional[pd.DataFrame] = field(default=None, repr=False)
    feature_responses: Optional[list[dict]] = field(default=None, repr=False)
    level3_composition: Optional[dict] = field(default=None, repr=False)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(seed: int = 42) -> dict:
    """Load and preprocess data using kan_paper pipeline."""
    csv_path = Path("data/prudential-life-insurance-assessment/train.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at {csv_path}")
    preprocessor = KANPreprocessor()
    return preprocessor.run_pipeline(csv_path, random_seed=seed)


def select_features(
    data: dict,
    n_features: int,
    top_features: list[str] = TOP_FEATURES_RANKED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Select top-N features from the preprocessed data."""
    all_names = data["feature_names"]
    keep = top_features[:n_features]

    keep_indices = []
    kept_names = []
    for feat in keep:
        if feat in all_names:
            keep_indices.append(all_names.index(feat))
            kept_names.append(feat)

    idx = np.array(keep_indices)
    return (
        data["X_train_outer"][:, idx],
        data["X_test_outer"][:, idx],
        data["y_train_outer"],
        data["y_test_outer"],
        kept_names,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[TabKAN, float, float]:
    """Train a TabKAN model (no LayerNorm) and return (module, qwk, seconds)."""
    L.seed_everything(42)

    in_features = X_train.shape[1]
    module = TabKAN(
        in_features=in_features,
        widths=cfg.hidden_widths,
        kan_type="chebykan",
        degree=cfg.degree,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        sparsity_lambda=cfg.sparsity_lambda,
        l1_weight=1.0,
        entropy_weight=1.0,
        use_layernorm=False,
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    t0 = time.time()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_time = time.time() - t0

    module.eval()
    with torch.no_grad():
        preds = module(X_val_t).cpu().numpy().flatten()
    preds_rounded = np.clip(np.round(preds), 1, 8).astype(int)
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, preds_rounded)

    return module, qwk, training_time


# ── Level 1: Per-Edge Exact Formulas ──────────────────────────────────────────

def level1_edge_formulas(
    module: TabKAN,
    feature_names: list[str],
    pruning_threshold: float = 0.01,
) -> pd.DataFrame:
    """Sample every active edge and fit a symbolic formula."""
    records = []
    layer_idx = 0

    for layer in module.kan_layers:
        if not isinstance(layer, ChebyKANLayer):
            continue

        l1_scores = _compute_edge_l1(layer)
        layer_degree = getattr(layer, "degree", 3)

        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                if l1_scores[out_i, in_i].item() < pruning_threshold:
                    continue

                x_vals, y_vals = sample_edge(layer, out_i, in_i, n=1000)
                formula, r2 = fit_symbolic_edge(
                    x_vals, y_vals, max_poly_degree=layer_degree,
                )

                if layer_idx == 0 and in_i < len(feature_names):
                    feat_name = feature_names[in_i]
                else:
                    feat_name = f"L{layer_idx}_h{in_i}"

                records.append({
                    "layer": layer_idx,
                    "edge_in": in_i,
                    "edge_out": out_i,
                    "input_feature": feat_name,
                    "formula": formula,
                    "r_squared": round(r2, 6),
                    "quality_tier": _quality_tier(r2),
                    "l1_norm": round(l1_scores[out_i, in_i].item(), 6),
                })

        layer_idx += 1

    return pd.DataFrame(records)


# ── Level 2: Per-Feature Conditional Response ─────────────────────────────────

def level2_feature_responses(
    module: TabKAN,
    X_train: np.ndarray,
    feature_names: list[str],
    n_points: int = 1000,
) -> list[dict]:
    """For each feature, vary it from -3 to +3 while holding others at mean.

    Forward-pass through the full model, subtract the baseline prediction,
    and fit a symbolic formula to the resulting conditional response curve.
    """
    module.eval()

    # Baseline: all features at their training mean
    means = X_train.mean(axis=0)
    baseline_input = torch.tensor(means, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        baseline_pred = module(baseline_input).item()

    # Sweep domain
    sweep = np.linspace(-3.0, 3.0, n_points)

    results = []
    for feat_idx, feat_name in enumerate(feature_names):
        # 1000 copies of mean-centered input
        tiled = np.tile(means, (n_points, 1))
        tiled[:, feat_idx] = sweep

        x_tensor = torch.tensor(tiled, dtype=torch.float32)
        with torch.no_grad():
            preds = module(x_tensor).cpu().numpy().flatten()

        # Center on baseline prediction
        response = preds - baseline_pred

        # Fit a symbolic formula to (sweep, response)
        layer_degree = 3
        for layer in module.kan_layers:
            if isinstance(layer, ChebyKANLayer):
                layer_degree = layer.degree
                break
        formula, r2 = fit_symbolic_edge(sweep, response, max_poly_degree=layer_degree)

        # Refit to get actual parameters
        params = None
        if formula in _REFIT_CANDIDATES:
            func, pnames = _REFIT_CANDIDATES[formula]
            try:
                popt, _ = curve_fit(
                    func, sweep, response,
                    p0=[1.0] * len(pnames), maxfev=3000,
                )
                params = {n: round(float(v), 6) for n, v in zip(pnames, popt)}
            except Exception:
                pass

        results.append({
            "feature": feat_name,
            "formula": formula,
            "r_squared": round(r2, 6),
            "quality_tier": _quality_tier(r2),
            "params": params,
            "response_range": round(float(response.max() - response.min()), 6),
        })

    return results


# ── Level 3: Full Model Exact Composition ─────────────────────────────────────

def level3_exact_composition(
    module: TabKAN,
    feature_names: list[str],
) -> dict[str, Any]:
    """Extract all Chebyshev coefficients, base weights, and head weights.

    Returns a JSON-serialisable dict specifying the exact model formula:
        prediction = head_weight @ kan_output + head_bias
    where each KAN layer computes:
        y[j] = sum_i( sum_d(coeffs[j,i,d] * T_d(tanh(x[i]))) + base_w[j,i] * x[i] )
    """
    composition: dict[str, Any] = {
        "feature_names": feature_names,
        "kan_layers": [],
        "head": {},
    }

    layer_idx = 0
    for layer in module.kan_layers:
        if not isinstance(layer, ChebyKANLayer):
            continue

        layer_data = {
            "layer_index": layer_idx,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "degree": layer.degree,
            "cheby_coeffs": layer.cheby_coeffs.detach().cpu().tolist(),
            "base_weight": layer.base_weight.detach().cpu().tolist(),
        }
        composition["kan_layers"].append(layer_data)
        layer_idx += 1

    composition["head"] = {
        "weight": module.head.weight.detach().cpu().tolist(),
        "bias": module.head.bias.detach().cpu().item(),
    }

    return composition


# ── Format helpers ────────────────────────────────────────────────────────────

def _format_formula(formula: str, params: Optional[dict]) -> str:
    """Render a formula with fitted parameter values for human reading."""
    if params is None:
        return formula

    mapping = {
        "a*x^3 + b*x^2 + c*x + d":
            lambda p: f"{p['a']:.4f}*x^3 + {p['b']:.4f}*x^2 + {p['c']:.4f}*x + {p['d']:.4f}",
        "a*x^2 + b*x + c":
            lambda p: f"{p['a']:.4f}*x^2 + {p['b']:.4f}*x + {p['c']:.4f}",
        "a*x + b":
            lambda p: f"{p['a']:.4f}*x + {p['b']:.4f}",
        "a*cos(x) + b":
            lambda p: f"{p['a']:.4f}*cos(x) + {p['b']:.4f}",
        "a*sin(x) + b":
            lambda p: f"{p['a']:.4f}*sin(x) + {p['b']:.4f}",
        "a*sin(2*x) + b":
            lambda p: f"{p['a']:.4f}*sin(2x) + {p['b']:.4f}",
        "a*sin(2*x) + b*cos(2*x)":
            lambda p: f"{p['a']:.4f}*sin(2x) + {p['b']:.4f}*cos(2x)",
        "a*sin(x) + b*cos(x)":
            lambda p: f"{p['a']:.4f}*sin(x) + {p['b']:.4f}*cos(x)",
        "a*exp(x) + b":
            lambda p: f"{p['a']:.4f}*exp(x) + {p['b']:.4f}",
        "a*log(|x|+1) + b":
            lambda p: f"{p['a']:.4f}*log(|x|+1) + {p['b']:.4f}",
        "a*sqrt(|x|) + b":
            lambda p: f"{p['a']:.4f}*sqrt(|x|) + {p['b']:.4f}",
        "a*|x| + b":
            lambda p: f"{p['a']:.4f}*|x| + {p['b']:.4f}",
        "a (constant)":
            lambda p: f"{p['a']:.4f}",
    }
    fmt = mapping.get(formula)
    if fmt is not None:
        try:
            return fmt(params)
        except KeyError:
            pass
    return formula


# ── Single experiment runner ──────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig, data: dict) -> ExperimentResult:
    """Run one experiment: train + 3-level interpretability extraction."""
    print(f"\n{'=' * 70}")
    print(f"Experiment: {cfg.name}")
    print(f"  features={cfg.n_features}, widths={cfg.hidden_widths}, "
          f"degree={cfg.degree}, sparsity={cfg.sparsity_lambda}")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test, feat_names = select_features(
        data, cfg.n_features,
    )
    print(f"  Selected {len(feat_names)} features")

    # ── Train ─────────────────────────────────────────────────────────────
    module, qwk, train_time = train_model(
        cfg, X_train, X_test, y_train, y_test,
    )
    print(f"  QWK = {qwk:.4f} (training time: {train_time:.1f}s)")

    # ── Level 1: per-edge symbolic fits ───────────────────────────────────
    edge_df = level1_edge_formulas(module, feat_names, cfg.pruning_threshold)

    if edge_df.empty:
        n_total_edges = sum(
            l.out_features * l.in_features
            for l in module.kan_layers if isinstance(l, ChebyKANLayer)
        )
        print("  WARNING: no active edges after pruning")
        return ExperimentResult(
            name=cfg.name, n_features=cfg.n_features,
            hidden_widths=cfg.hidden_widths, degree=cfg.degree,
            sparsity_lambda=cfg.sparsity_lambda, qwk=qwk,
            n_active_edges=0, n_total_edges=n_total_edges,
            n_clean=0, n_acceptable=0, n_flagged=0,
            mean_r2=0.0, median_r2=0.0, pct_clean=0.0,
            pct_interpretable=0.0,
            n_features_interpretable=0, n_features_total_active=0,
            pct_features_interpretable=0.0, level2_mean_r2=0.0,
            has_level3=False, training_time_s=train_time,
            edge_fits=edge_df, feature_responses=[], level3_composition=None,
        )

    n_edges = edge_df.shape[0]
    n_clean = int((edge_df["quality_tier"] == "clean").sum())
    n_accept = int((edge_df["quality_tier"] == "acceptable").sum())
    n_flag = int((edge_df["quality_tier"] == "flagged").sum())
    n_total_edges = sum(
        l.out_features * l.in_features
        for l in module.kan_layers if isinstance(l, ChebyKANLayer)
    )
    pct_clean = n_clean / n_edges * 100 if n_edges > 0 else 0
    pct_interp = (n_clean + n_accept) / n_edges * 100 if n_edges > 0 else 0
    mean_r2_l1 = float(edge_df["r_squared"].mean())
    median_r2_l1 = float(edge_df["r_squared"].median())

    print(f"  Level 1 -- Active edges: {n_edges}/{n_total_edges}")
    print(f"    Clean (R2>=0.99): {n_clean} ({pct_clean:.1f}%)")
    print(f"    Acceptable (R2>=0.90): {n_accept}")
    print(f"    Flagged (R2<0.90): {n_flag}")
    print(f"    Mean R2: {mean_r2_l1:.4f}, Median: {median_r2_l1:.4f}")

    # ── Level 2: per-feature conditional response ─────────────────────────
    feat_resp = level2_feature_responses(module, X_train, feat_names)

    n_feat_active = sum(1 for r in feat_resp if r["response_range"] > 1e-6)
    n_feat_interp = sum(
        1 for r in feat_resp
        if r["quality_tier"] in ("clean", "acceptable") and r["response_range"] > 1e-6
    )
    pct_feat_interp = n_feat_interp / max(n_feat_active, 1) * 100
    r2_vals = [r["r_squared"] for r in feat_resp if r["response_range"] > 1e-6]
    mean_r2_l2 = float(np.mean(r2_vals)) if r2_vals else 0.0

    print(f"  Level 2 -- Feature conditional responses:")
    print(f"    Active features: {n_feat_active}/{len(feat_names)}")
    print(f"    Interpretable features: {n_feat_interp} ({pct_feat_interp:.1f}%)")
    print(f"    Mean R2: {mean_r2_l2:.4f}")

    # Print per-feature detail for small models
    if len(feat_names) <= 25:
        for r in sorted(feat_resp, key=lambda x: x["response_range"], reverse=True):
            if r["response_range"] < 1e-6:
                continue
            tier_sym = {"clean": "+", "acceptable": "~", "flagged": "X"}
            sym = tier_sym.get(r["quality_tier"], "?")
            readable = _format_formula(r["formula"], r["params"])
            print(f"    [{sym}] {r['feature']:25s}  {readable:45s}  "
                  f"R2={r['r_squared']:.4f}  range={r['response_range']:.3f}")
        pruned_feats = [r["feature"] for r in feat_resp if r["response_range"] < 1e-6]
        if pruned_feats:
            print(f"    Pruned: {', '.join(pruned_feats)}")

    # ── Level 3: exact composition ────────────────────────────────────────
    composition = level3_exact_composition(module, feat_names)
    print(f"  Level 3 -- Full composition extracted "
          f"({len(composition['kan_layers'])} KAN layers + head)")

    return ExperimentResult(
        name=cfg.name,
        n_features=cfg.n_features,
        hidden_widths=cfg.hidden_widths,
        degree=cfg.degree,
        sparsity_lambda=cfg.sparsity_lambda,
        qwk=qwk,
        n_active_edges=n_edges,
        n_total_edges=n_total_edges,
        n_clean=n_clean,
        n_acceptable=n_accept,
        n_flagged=n_flag,
        mean_r2=mean_r2_l1,
        median_r2=median_r2_l1,
        pct_clean=pct_clean,
        pct_interpretable=pct_interp,
        n_features_interpretable=n_feat_interp,
        n_features_total_active=n_feat_active,
        pct_features_interpretable=pct_feat_interp,
        level2_mean_r2=mean_r2_l2,
        has_level3=True,
        training_time_s=train_time,
        edge_fits=edge_df,
        feature_responses=feat_resp,
        level3_composition=composition,
    )


# ── Experiment grid ───────────────────────────────────────────────────────────

def build_experiment_grid() -> list[ExperimentConfig]:
    """Build 63 experiments: 1-layer GAM baselines + 2-layer KANs.

    1-layer (GAM):  widths [4], [8], [16]          (3)
    2-layer (KAN):  widths [8,1], [16,1], [8,2], [16,2]  (4)
    Crossed with:   n_features in {10, 15, 20}     (3)
                    sparsity in {0.0, 0.003, 0.005} (3)
    Total: 7 * 3 * 3 = 63
    """
    configs = []

    widths_options = [
        # 1-layer GAM baselines
        [4], [8], [16],
        # 2-layer KANs
        [8, 1], [16, 1], [8, 2], [16, 2],
    ]

    for n_feat in [10, 15, 20]:
        for widths in widths_options:
            for sparsity in [0.0, 0.003, 0.005]:
                w_str = "x".join(str(w) for w in widths)
                sp_str = str(sparsity).replace(".", "")
                name = f"f{n_feat}_w{w_str}_sp{sp_str}"
                configs.append(ExperimentConfig(
                    name=name,
                    n_features=n_feat,
                    hidden_widths=widths,
                    degree=3,
                    sparsity_lambda=sparsity,
                    max_epochs=150,
                    batch_size=2048,
                ))

    return configs


# ── Output ────────────────────────────────────────────────────────────────────

def print_results_table(results: list[ExperimentResult]) -> None:
    """Print summary table sorted by composite score = QWK * %interpretable features."""
    print(f"\n{'=' * 140}")
    print("RESULTS SUMMARY (sorted by composite score = QWK * pct_features_interpretable / 100)")
    print(f"{'=' * 140}")
    header = (
        f"{'Name':30s} {'Feat':>4s} {'Widths':>8s} {'Spar':>6s} {'QWK':>6s} "
        f"{'Edges':>6s} {'L1 R2':>6s} {'L1 Int%':>7s} "
        f"{'FeatI':>5s} {'L2 R2':>6s} {'L2 Int%':>7s} "
        f"{'Score':>6s}"
    )
    print(header)
    print("-" * 140)

    scored = [
        (r, r.qwk * r.pct_features_interpretable / 100)
        for r in results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    for r, score in scored:
        w_str = "x".join(str(w) for w in r.hidden_widths)
        print(
            f"{r.name:30s} {r.n_features:>4d} {w_str:>8s} {r.sparsity_lambda:>6.3f} "
            f"{r.qwk:>6.4f} {r.n_active_edges:>6d} "
            f"{r.mean_r2:>6.4f} {r.pct_interpretable:>6.1f}% "
            f"{r.n_features_interpretable:>5d} {r.level2_mean_r2:>6.4f} "
            f"{r.pct_features_interpretable:>6.1f}% "
            f"{score:>6.3f}"
        )


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    """Save summary CSV + per-model detail JSONs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Summary CSV ───────────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "n_features": r.n_features,
            "hidden_widths": str(r.hidden_widths),
            "degree": r.degree,
            "sparsity_lambda": r.sparsity_lambda,
            "qwk": r.qwk,
            "n_active_edges": r.n_active_edges,
            "n_total_edges": r.n_total_edges,
            "n_clean": r.n_clean,
            "n_acceptable": r.n_acceptable,
            "n_flagged": r.n_flagged,
            "mean_r2_l1": r.mean_r2,
            "median_r2_l1": r.median_r2,
            "pct_clean": r.pct_clean,
            "pct_interpretable_l1": r.pct_interpretable,
            "n_features_interpretable": r.n_features_interpretable,
            "n_features_total_active": r.n_features_total_active,
            "pct_features_interpretable": r.pct_features_interpretable,
            "mean_r2_l2": r.level2_mean_r2,
            "has_level3": r.has_level3,
            "training_time_s": r.training_time_s,
            "composite_score": r.qwk * r.pct_features_interpretable / 100,
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "kan_vs_gam_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to {csv_path}")

    # ── Per-model detail JSONs ────────────────────────────────────────────
    detail_dir = output_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        detail: dict[str, Any] = {
            "name": r.name,
            "config": {
                "n_features": r.n_features,
                "hidden_widths": r.hidden_widths,
                "degree": r.degree,
                "sparsity_lambda": r.sparsity_lambda,
            },
            "qwk": r.qwk,
            "training_time_s": r.training_time_s,
        }

        # Level 1
        if r.edge_fits is not None and not r.edge_fits.empty:
            detail["level1_edge_fits"] = r.edge_fits.to_dict(orient="records")
        else:
            detail["level1_edge_fits"] = []

        detail["level1_summary"] = {
            "n_active_edges": r.n_active_edges,
            "n_total_edges": r.n_total_edges,
            "n_clean": r.n_clean,
            "n_acceptable": r.n_acceptable,
            "n_flagged": r.n_flagged,
            "mean_r2": r.mean_r2,
            "pct_interpretable": r.pct_interpretable,
        }

        # Level 2
        detail["level2_feature_responses"] = r.feature_responses or []
        detail["level2_summary"] = {
            "n_features_interpretable": r.n_features_interpretable,
            "n_features_total_active": r.n_features_total_active,
            "pct_features_interpretable": r.pct_features_interpretable,
            "mean_r2": r.level2_mean_r2,
        }

        # Level 3
        if r.level3_composition is not None:
            detail["level3_composition"] = r.level3_composition

        json_path = detail_dir / f"{r.name}.json"
        json_path.write_text(json.dumps(detail, indent=2, default=str))

    print(f"Per-model details saved to {detail_dir}/")

    # ── Per-experiment edge fit CSVs ──────────────────────────────────────
    for r in results:
        if r.edge_fits is not None and not r.edge_fits.empty:
            fits_path = output_dir / f"{r.name}_edge_fits.csv"
            r.edge_fits.to_csv(fits_path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("KAN vs GAM Grid Search")
    print("=" * 70)
    print("Loading and preprocessing data...")
    data = load_data(seed=42)
    print(f"Data loaded: {data['X_train_outer'].shape[0]} train, "
          f"{data['X_test_outer'].shape[0]} test, "
          f"{len(data['feature_names'])} features")

    configs = build_experiment_grid()
    print(f"\nRunning {len(configs)} experiments...")
    print(f"  1-layer GAM baselines:  widths in [[4], [8], [16]]")
    print(f"  2-layer KAN variants:   widths in [[8,1], [16,1], [8,2], [16,2]]")
    print(f"  Features:               {{10, 15, 20}}")
    print(f"  Sparsity:               {{0.0, 0.003, 0.005}}")
    print(f"  Fixed:                  degree=3, no LayerNorm, 150 epochs, batch=2048\n")

    output_dir = Path("outputs/kan_vs_gam")
    results: list[ExperimentResult] = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}]", end="")
        try:
            result = run_experiment(cfg, data)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────
    print_results_table(results)
    save_results(results, output_dir)

    # ── Best model ────────────────────────────────────────────────────────
    if results:
        scored = [
            (r, r.qwk * r.pct_features_interpretable / 100)
            for r in results
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        print(f"\n{'=' * 70}")
        print(f"BEST MODEL: {best.name}")
        print(f"  QWK={best.qwk:.4f}")
        print(f"  Level 1 edge interpretability: {best.pct_interpretable:.1f}%  "
              f"(mean R2={best.mean_r2:.4f})")
        print(f"  Level 2 feature interpretability: {best.pct_features_interpretable:.1f}%  "
              f"(mean R2={best.level2_mean_r2:.4f})")
        print(f"  Composite score: {best_score:.3f}")
        print(f"  Architecture: {best.n_features} features, "
              f"widths={best.hidden_widths}, sparsity={best.sparsity_lambda}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
