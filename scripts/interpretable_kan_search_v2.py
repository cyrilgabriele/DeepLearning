#!/usr/bin/env python3
"""Sweep 2: Push QWK higher with wider architectures + sparsity.

Based on Sweep 1 findings:
- Degree=3 is mandatory for interpretability
- Sparsity regularization helps clean% without losing QWK
- More features helps QWK
- [4,2] always collapses

This sweep tests wider architectures ([16], [32], [16,8]) with aggressive
sparsity to see if we can push QWK above 0.50 while maintaining 90%+
interpretability. Also tests 200 epochs for best configs.

Usage:
    uv run python scripts/interpretable_kan_search_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.interpretable_kan_search import (
    ExperimentConfig,
    ExperimentResult,
    load_data,
    run_experiment,
    print_results_table,
    save_results,
)


def build_experiment_grid_v2() -> list[ExperimentConfig]:
    """Sweep 2: wider + sparsity + more epochs."""
    configs = []

    # ── Wider single-layer with sparsity ─────────────────────────────────────
    for n_feat in [10, 15, 20]:
        for width in [16, 32]:
            for sp in [0.001, 0.005, 0.01]:
                sp_str = f"sp{sp}".replace(".", "")
                configs.append(ExperimentConfig(
                    name=f"f{n_feat}_w{width}_d3_{sp_str}",
                    n_features=n_feat,
                    hidden_widths=[width],
                    degree=3,
                    sparsity_lambda=sp,
                ))

    # ── Wider two-layer with sparsity (avoid [4,2] collapse) ─────────────────
    for n_feat in [15, 20]:
        for widths in [[16, 8], [32, 16]]:
            for sp in [0.001, 0.005]:
                w_str = "x".join(str(w) for w in widths)
                sp_str = f"sp{sp}".replace(".", "")
                configs.append(ExperimentConfig(
                    name=f"f{n_feat}_w{w_str}_d3_{sp_str}",
                    n_features=n_feat,
                    hidden_widths=widths,
                    degree=3,
                    sparsity_lambda=sp,
                ))

    # ── Best configs from sweep 1 with 200 epochs ───────────────────────────
    for n_feat in [15, 20]:
        for width in [8, 16]:
            configs.append(ExperimentConfig(
                name=f"f{n_feat}_w{width}_d3_e200",
                n_features=n_feat,
                hidden_widths=[width],
                degree=3,
                sparsity_lambda=0.0,
                max_epochs=200,
            ))
            configs.append(ExperimentConfig(
                name=f"f{n_feat}_w{width}_d3_sp001_e200",
                n_features=n_feat,
                hidden_widths=[width],
                degree=3,
                sparsity_lambda=0.001,
                max_epochs=200,
            ))

    return configs


def main():
    print("Loading and preprocessing data...")
    data = load_data(seed=42)
    print(f"Data loaded: {data['X_train_outer'].shape[0]} train, "
          f"{data['X_test_outer'].shape[0]} test")

    configs = build_experiment_grid_v2()
    print(f"\nRunning {len(configs)} experiments (Sweep 2)...")

    output_dir = Path("outputs/interpretable_kan_search_v2")
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

    if results:
        scored = [(r, r.pct_interpretable * r.qwk / 100) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        print(f"\n{'='*70}")
        print(f"BEST MODEL (Sweep 2): {best.name}")
        print(f"  QWK={best.qwk:.4f}, Interpretable={best.pct_interpretable:.1f}%, "
              f"Clean={best.pct_clean:.1f}%")
        print(f"  Active edges: {best.n_active_edges}, "
              f"Features: {best.n_features}, Widths: {best.hidden_widths}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
