"""Quick experiment: train narrow ChebyKAN architectures and check if
pruning can actually eliminate input features.

Usage:
    uv run python scripts/run_narrow_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import detect_device, load_experiment_config
from src.interpretability.kan_pruning import prune_kan, _compute_edge_l1
from src.models.kan_layers import ChebyKANLayer
from src.training.trainer import run_train


CONFIGS = [
    ("140->16->8->1", "configs/experiment_stages/stage_c_explanation_package/chebykan_narrow_16_8.yaml"),
    ("140->32->16->1", "configs/experiment_stages/stage_c_explanation_package/chebykan_narrow_32_16.yaml"),
    ("140->64->32->1", "configs/experiment_stages/stage_c_explanation_package/chebykan_narrow_64_32.yaml"),
]

PRUNING_THRESHOLD = 0.01


def count_surviving_features(model, threshold: float = PRUNING_THRESHOLD) -> tuple[int, list[int]]:
    first_kan_layer = None
    for layer in model.kan_layers:
        if isinstance(layer, ChebyKANLayer):
            first_kan_layer = layer
            break
    if first_kan_layer is None:
        return 0, []
    l1_scores = _compute_edge_l1(first_kan_layer)  # (out, in)
    mask = l1_scores >= threshold
    surviving = mask.any(dim=0)
    indices = torch.where(surviving)[0].tolist()
    return int(surviving.sum().item()), indices


def main() -> None:
    device = detect_device()
    results = []

    for label, config_path in CONFIGS:
        print(f"\n{'='*70}")
        print(f"TRAINING: {label}")
        print(f"{'='*70}")

        config = load_experiment_config(Path(config_path))
        artifacts = run_train(config, device=device)
        qwk = artifacts.metrics.get("qwk", 0.0)

        kan_module = artifacts.model.module
        kan_module.eval()

        # Count edges in layer 0
        first_layer = None
        for layer in kan_module.kan_layers:
            if isinstance(layer, ChebyKANLayer):
                first_layer = layer
                break

        total_edges = first_layer.in_features * first_layer.out_features if first_layer else 0

        # Prune
        pruned_model, stats, masks = prune_kan(kan_module, PRUNING_THRESHOLD)
        n_surviving, surviving_idx = count_surviving_features(kan_module, PRUNING_THRESHOLD)

        # Evaluate pruned model
        import copy
        import pandas as pd
        from sklearn.metrics import cohen_kappa_score
        from src.interpretability.utils.paths import eval_run_dir

        eval_dir = eval_run_dir(
            Path("outputs"), config.preprocessing.recipe,
            config.trainer.experiment_name, create=False,
        )
        qwk_pruned = None
        X_eval_path = eval_dir / "X_eval.parquet"
        y_eval_path = eval_dir / "y_eval.parquet"
        if X_eval_path.exists() and y_eval_path.exists():
            X_eval = pd.read_parquet(X_eval_path)
            y_eval = pd.read_parquet(y_eval_path).iloc[:, 0]
            pruned_wrapper = copy.deepcopy(artifacts.model)
            pruned_wrapper.module = pruned_model
            preds = pruned_wrapper.predict(X_eval)
            qwk_pruned = float(cohen_kappa_score(y_eval, preds, weights="quadratic"))

        results.append({
            "architecture": label,
            "layer0_edges": total_edges,
            "qwk_trained": round(qwk, 4),
            "qwk_pruned": round(qwk_pruned, 4) if qwk_pruned is not None else None,
            "sparsity_ratio": stats.sparsity_ratio,
            "edges_after": stats.edges_after,
            "surviving_features": n_surviving,
        })

        print(f"  QWK={qwk:.4f}  pruned_QWK={qwk_pruned or 0:.4f}  "
              f"sparsity={stats.sparsity_ratio:.2%}  "
              f"surviving={n_surviving}/140")

    # Print summary table
    print(f"\n{'='*90}")
    print("NARROW ARCHITECTURE EXPERIMENT — SUMMARY")
    print(f"{'='*90}")
    print(f"{'Architecture':<20} {'L0 edges':>10} {'QWK':>8} {'Pruned':>8} {'Sparsity':>10} {'Features':>10}")
    print(f"{'-'*20:<20} {'-'*10:>10} {'-'*8:>8} {'-'*8:>8} {'-'*10:>10} {'-'*10:>10}")
    for r in results:
        print(
            f"{r['architecture']:<20} "
            f"{r['layer0_edges']:>10} "
            f"{r['qwk_trained']:>8.4f} "
            f"{(r['qwk_pruned'] or 0):>8.4f} "
            f"{r['sparsity_ratio']:>9.2%} "
            f"{r['surviving_features']:>10}/140"
        )

    # Reference: wide model
    print(f"\n{'Reference (wide):':<20} {'17920':>10} {'0.6051':>8} {'0.5957':>8} {'94.47%':>10} {'132':>10}/140")

    # Save results
    out_path = Path("outputs/narrow_architecture_experiment.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
