#!/usr/bin/env python3
"""Systematic search for maximally-interpretable KAN models.

Trains narrow ChebyKAN architectures with varying feature counts, widths,
and degrees, then runs pruning + symbolic regression to find the largest
model that still produces human-readable closed-form solutions.

Usage:
    uv run python scripts/interpretable_kan_search.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning as L

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


# ── Top features from coefficient importance (full [128,64] model) ───────────
TOP_FEATURES_RANKED = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]


@dataclass
class ExperimentConfig:
    name: str
    n_features: int
    hidden_widths: list[int]
    degree: int
    sparsity_lambda: float = 0.0
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 100
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
    n_active_edges: int
    n_total_edges: int
    n_clean: int          # R^2 >= 0.99
    n_acceptable: int     # 0.90 <= R^2 < 0.99
    n_flagged: int        # R^2 < 0.90
    mean_r2: float
    median_r2: float
    pct_clean: float
    pct_interpretable: float  # clean + acceptable
    training_time_s: float
    symbolic_fits: Optional[pd.DataFrame] = field(default=None, repr=False)


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

    # Find column indices
    keep_indices = []
    kept_names = []
    for feat in keep:
        if feat in all_names:
            keep_indices.append(all_names.index(feat))
            kept_names.append(feat)

    idx = np.array(keep_indices)
    X_train = data["X_train_outer"][:, idx]
    X_test = data["X_test_outer"][:, idx]
    y_train = data["y_train_outer"]
    y_test = data["y_test_outer"]

    return X_train, X_test, y_train, y_test, kept_names


def train_model(
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[TabKAN, float, float]:
    """Train a TabKAN model and return (module, qwk, training_time_s)."""
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
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
    )

    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

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

    # Evaluate QWK
    module.eval()
    with torch.no_grad():
        preds = module(X_val_t).cpu().numpy().flatten()
    preds_rounded = np.clip(np.round(preds), 1, 8).astype(int)
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, preds_rounded)

    return module, qwk, training_time


def run_symbolic_analysis(
    module: TabKAN,
    feature_names: list[str],
    pruning_threshold: float = 0.01,
) -> pd.DataFrame:
    """Run pruning + symbolic regression on all KAN layers."""
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
                formula, r2 = fit_symbolic_edge(x_vals, y_vals,
                                                max_poly_degree=layer_degree)

                if layer_idx == 0 and in_i < len(feature_names):
                    feat_name = feature_names[in_i]
                else:
                    feat_name = f"h{in_i}"
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


def run_experiment(cfg: ExperimentConfig, data: dict) -> ExperimentResult:
    """Run a single experiment: train + prune + symbolic fit."""
    print(f"\n{'='*70}")
    print(f"Experiment: {cfg.name}")
    print(f"  features={cfg.n_features}, widths={cfg.hidden_widths}, "
          f"degree={cfg.degree}, sparsity={cfg.sparsity_lambda}")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test, feat_names = select_features(
        data, cfg.n_features,
    )
    print(f"  Selected {len(feat_names)} features: {feat_names}")

    module, qwk, train_time = train_model(cfg, X_train, X_test, y_train, y_test)
    print(f"  QWK = {qwk:.4f} (training time: {train_time:.1f}s)")

    fits_df = run_symbolic_analysis(module, feat_names, cfg.pruning_threshold)

    if fits_df.empty:
        print("  WARNING: No active edges found after pruning!")
        return ExperimentResult(
            name=cfg.name, n_features=cfg.n_features,
            hidden_widths=cfg.hidden_widths, degree=cfg.degree,
            sparsity_lambda=cfg.sparsity_lambda,
            qwk=qwk, n_active_edges=0, n_total_edges=0,
            n_clean=0, n_acceptable=0, n_flagged=0,
            mean_r2=0.0, median_r2=0.0,
            pct_clean=0.0, pct_interpretable=0.0,
            training_time_s=train_time, symbolic_fits=fits_df,
        )

    n_total = fits_df.shape[0]
    n_clean = int((fits_df["quality_tier"] == "clean").sum())
    n_acceptable = int((fits_df["quality_tier"] == "acceptable").sum())
    n_flagged = int((fits_df["quality_tier"] == "flagged").sum())

    total_edges = sum(
        l.out_features * l.in_features
        for l in module.kan_layers if isinstance(l, ChebyKANLayer)
    )

    pct_clean = n_clean / n_total * 100 if n_total > 0 else 0
    pct_interp = (n_clean + n_acceptable) / n_total * 100 if n_total > 0 else 0

    print(f"  Active edges: {n_total}/{total_edges}")
    print(f"  Clean (R²≥0.99): {n_clean} ({pct_clean:.1f}%)")
    print(f"  Acceptable (R²≥0.90): {n_acceptable}")
    print(f"  Flagged (R²<0.90): {n_flagged}")
    print(f"  Mean R²: {fits_df['r_squared'].mean():.4f}, "
          f"Median: {fits_df['r_squared'].median():.4f}")
    print(f"  Interpretable (clean+acceptable): {pct_interp:.1f}%")

    # Show per-feature breakdown
    if n_total <= 50:
        print("\n  Per-feature symbolic fits:")
        for feat in feat_names:
            feat_edges = fits_df[fits_df["input_feature"] == feat]
            if feat_edges.empty:
                print(f"    {feat:25s}: PRUNED (no active edges)")
            else:
                for _, row in feat_edges.iterrows():
                    tier_sym = {"clean": "+", "acceptable": "~", "flagged": "X"}
                    sym = tier_sym.get(row["quality_tier"], "?")
                    print(f"    {feat:25s} [{sym}] {row['formula']:30s} R²={row['r_squared']:.4f} "
                          f"(L1={row['l1_norm']:.4f})")

    return ExperimentResult(
        name=cfg.name, n_features=cfg.n_features,
        hidden_widths=cfg.hidden_widths, degree=cfg.degree,
        sparsity_lambda=cfg.sparsity_lambda,
        qwk=qwk, n_active_edges=n_total, n_total_edges=total_edges,
        n_clean=n_clean, n_acceptable=n_acceptable, n_flagged=n_flagged,
        mean_r2=fits_df["r_squared"].mean(),
        median_r2=fits_df["r_squared"].median(),
        pct_clean=pct_clean, pct_interpretable=pct_interp,
        training_time_s=train_time, symbolic_fits=fits_df,
    )


def build_experiment_grid() -> list[ExperimentConfig]:
    """Build the grid of experiments to run."""
    configs = []

    # Sweep 1: Fix degree=3, vary architecture and features
    for n_feat in [5, 8, 10, 15, 20]:
        for widths in [[4], [8], [4, 2], [8, 4]]:
            w_str = "x".join(str(w) for w in widths)
            configs.append(ExperimentConfig(
                name=f"f{n_feat}_w{w_str}_d3",
                n_features=n_feat,
                hidden_widths=widths,
                degree=3,
                sparsity_lambda=0.0,
            ))

    # Sweep 2: Best from sweep 1 with degree=4 and sparsity
    for n_feat in [8, 10, 15]:
        for widths in [[4], [8]]:
            w_str = "x".join(str(w) for w in widths)
            configs.append(ExperimentConfig(
                name=f"f{n_feat}_w{w_str}_d4",
                n_features=n_feat,
                hidden_widths=widths,
                degree=4,
                sparsity_lambda=0.0,
            ))
            # With sparsity
            configs.append(ExperimentConfig(
                name=f"f{n_feat}_w{w_str}_d3_sp",
                n_features=n_feat,
                hidden_widths=widths,
                degree=3,
                sparsity_lambda=0.001,
            ))

    return configs


def print_results_table(results: list[ExperimentResult]) -> None:
    """Print a sorted summary table."""
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY (sorted by interpretability score = pct_interpretable * qwk)")
    print("=" * 120)
    print(f"{'Name':30s} {'Feat':>4s} {'Widths':>8s} {'Deg':>3s} {'QWK':>6s} "
          f"{'Active':>6s} {'Clean%':>6s} {'Interp%':>7s} {'MeanR²':>7s} {'Score':>6s}")
    print("-" * 120)

    # Sort by composite score: interpretability * qwk
    scored = [(r, r.pct_interpretable * r.qwk / 100) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)

    for r, score in scored:
        w_str = "x".join(str(w) for w in r.hidden_widths)
        print(f"{r.name:30s} {r.n_features:>4d} {w_str:>8s} {r.degree:>3d} "
              f"{r.qwk:>6.4f} {r.n_active_edges:>6d} "
              f"{r.pct_clean:>5.1f}% {r.pct_interpretable:>6.1f}% "
              f"{r.mean_r2:>7.4f} {score:>6.3f}")


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "mean_r2": r.mean_r2,
            "median_r2": r.median_r2,
            "pct_clean": r.pct_clean,
            "pct_interpretable": r.pct_interpretable,
            "training_time_s": r.training_time_s,
            "score": r.pct_interpretable * r.qwk / 100,
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "interpretable_kan_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save per-experiment symbolic fits
    for r in results:
        if r.symbolic_fits is not None and not r.symbolic_fits.empty:
            fits_path = output_dir / f"{r.name}_symbolic_fits.csv"
            r.symbolic_fits.to_csv(fits_path, index=False)


def main():
    print("Loading and preprocessing data...")
    data = load_data(seed=42)
    print(f"Data loaded: {data['X_train_outer'].shape[0]} train, "
          f"{data['X_test_outer'].shape[0]} test, "
          f"{len(data['feature_names'])} features")

    configs = build_experiment_grid()
    print(f"\nRunning {len(configs)} experiments...")

    output_dir = Path("outputs/interpretable_kan_search")
    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]", end="")
        try:
            result = run_experiment(cfg, data)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print_results_table(results)
    save_results(results, output_dir)

    # Identify the best model
    if results:
        scored = [(r, r.pct_interpretable * r.qwk / 100) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best.name}")
        print(f"  QWK={best.qwk:.4f}, Interpretable={best.pct_interpretable:.1f}%, "
              f"Clean={best.pct_clean:.1f}%")
        print(f"  Active edges: {best.n_active_edges}, "
              f"Features: {best.n_features}, Widths: {best.hidden_widths}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
