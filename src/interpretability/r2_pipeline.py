"""Issue 08 — Reusable R² symbolic-fit pipeline for any pruned KAN.

Usage (programmatic):
    from src.interpretability.r2_pipeline import evaluate_symbolic_fit
    report = evaluate_symbolic_fit(module, feature_names=names)

Usage (CLI):
    uv run python -m src.interpretability.r2_pipeline \
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
from typing import Any

import numpy as np
import pandas as pd

from src.configs import ExperimentConfig


def evaluate_symbolic_fit(
    module,
    feature_names: list[str],
    threshold: float = 0.01,
    candidate_library: str = "scipy",
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Run pruning + symbolic fitting and return a structured report.

    Parameters
    ----------
    module : TabKAN (torch.nn.Module)
        A trained (but not yet pruned) or already-pruned KAN module.
    feature_names : list[str]
        Ordered list of input feature names (used for layer-0 edge labelling).
    threshold : float
        Minimum activation L1 norm for an edge to be considered active.
    candidate_library : "scipy" | "pysr"
        Which symbolic regression backend to use.
    n_samples : int
        Number of sample points for edge activation.

    Returns
    -------
    dict with keys:
        pruning   : dict  — edges_before, edges_after, sparsity_ratio
        symbolic_fits : list[dict] — layer, edge_in, edge_out, input_feature, formula, r_squared, flagged
        aggregate : dict  — mean_r2, median_r2, edges_below_090, edges_below_095
    """
    import torch
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.interpretability.kan_pruning import _compute_edge_l1
    from src.interpretability.kan_symbolic import sample_edge, fit_symbolic_edge, _quality_tier

    use_pysr = candidate_library == "pysr"

    total_before = 0
    total_after = 0
    records = []
    layer_idx = 0

    for layer in module.kan_layers:
        if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            continue

        # Precompute all edge L1 scores once per layer
        l1_scores = _compute_edge_l1(layer)
        total_before += l1_scores.numel()

        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                if l1_scores[out_i, in_i].item() < threshold:
                    continue
                total_after += 1

                x_vals, y_vals = sample_edge(layer, out_i, in_i, n=n_samples)
                formula, r2 = fit_symbolic_edge(x_vals, y_vals, use_pysr=use_pysr)
                input_feat = (
                    feature_names[in_i]
                    if layer_idx == 0 and in_i < len(feature_names)
                    else f"h{in_i}"
                )
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

    sparsity = 1.0 - (total_after / total_before) if total_before > 0 else 0.0
    r2_values = [r["r_squared"] for r in records] if records else [0.0]

    return {
        "pruning": {
            "edges_before": total_before,
            "edges_after": total_after,
            "sparsity_ratio": round(sparsity, 4),
        },
        "symbolic_fits": records,
        "aggregate": {
            "mean_r2": round(float(np.mean(r2_values)), 6),
            "median_r2": round(float(np.median(r2_values)), 6),
            "edges_below_090": int(sum(1 for r in r2_values if r < 0.90)),
            "edges_below_095": int(sum(1 for r in r2_values if r < 0.95)),
            # Three-tier quality breakdown (Liu et al. 2024 arXiv:2404.19756)
            "edges_clean": int(sum(1 for r in r2_values if r >= 0.99)),
            "edges_acceptable": int(sum(1 for r in r2_values if 0.90 <= r < 0.99)),
            "edges_flagged": int(sum(1 for r in r2_values if r < 0.90)),
        },
    }


def run(
    pruned_checkpoint_path: Path,
    pruning_summary_path: Path,
    config: ExperimentConfig,
    eval_features_path: Path,
    flavor: str,
    candidate_library: str = "scipy",
    output_dir: Path = Path("outputs"),
) -> dict:
    import torch
    from src.models.tabkan import TabKAN

    pruning_summary = json.loads(pruning_summary_path.read_text())
    threshold = pruning_summary["threshold"]

    X_eval = pd.read_parquet(eval_features_path)
    feature_names = list(X_eval.columns)
    in_features = X_eval.shape[1]
    widths = [config.model.width] * config.model.depth

    module = TabKAN(
        in_features=in_features,
        widths=widths,
        kan_type=flavor,
        degree=config.model.degree or 3,
    )
    module.load_state_dict(torch.load(pruned_checkpoint_path, map_location="cpu"))
    module.eval()

    print(f"Running R² pipeline for {flavor}…")
    report = evaluate_symbolic_fit(
        module,
        feature_names=feature_names,
        threshold=threshold,
        candidate_library=candidate_library,
    )

    agg = report["aggregate"]
    print(f"  Active edges : {report['pruning']['edges_after']}")
    print(f"  Mean R²      : {agg['mean_r2']:.4f}")
    print(f"  Median R²    : {agg['median_r2']:.4f}")
    print(f"  Clean  (≥0.99): {agg['edges_clean']}")
    print(f"  Accept (0.90–0.99): {agg['edges_acceptable']}")
    print(f"  Flagged (<0.90): {agg['edges_flagged']}")

    from src.interpretability.utils.paths import reports as rep_dir
    out_path = rep_dir(output_dir) / f"{flavor}_r2_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved → {out_path}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R² symbolic fit pipeline for KAN")
    p.add_argument("--pruned-checkpoint", type=Path, required=True)
    p.add_argument("--pruning-summary", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--flavor", choices=["chebykan", "fourierkan"], required=True)
    p.add_argument("--candidate-library", choices=["scipy", "pysr"], default="scipy")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from src.configs import load_experiment_config

    run(
        args.pruned_checkpoint,
        args.pruning_summary,
        load_experiment_config(args.config),
        args.eval_features,
        args.flavor,
        args.candidate_library,
        args.output_dir,
    )
