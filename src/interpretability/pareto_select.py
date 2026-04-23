"""Select interpretable KAN variants from a Pareto sparsity sweep.

For each Pareto-optimal trial, trains the model from scratch, prunes at
threshold=0.01, counts surviving input features (features with at least
one active edge into the first hidden layer), and produces a ranked
summary table for actuary review.

Multiple feature-count cutoffs (5, 10, 15, 20) are evaluated so an
actuary can state which level is still interpretable.

Usage:
    uv run python -m src.interpretability.pareto_select \
        --pareto-json sweeps/stage-c-chebykan-pareto-sparsity_pareto.json \
        --flavor chebykan
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import ExperimentConfig, detect_device, load_experiment_config
from src.interpretability.kan_pruning import prune_kan, _compute_edge_l1
from src.interpretability.utils.paths import (
    eval_run_dir,
    interpret_run_dir,
    reports as rep_dir,
    data as data_dir,
    models as mod_dir,
)
from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
from src.training.trainer import run_train


FEATURE_CUTOFFS = [5, 10, 15, 20]
PRUNING_THRESHOLD = 0.01


def _count_surviving_input_features(
    model, threshold: float = PRUNING_THRESHOLD,
) -> tuple[int, list[int]]:
    """Count input features with at least one active edge after pruning.

    Returns (count, list_of_surviving_feature_indices).
    """
    first_kan_layer = None
    for layer in model.kan_layers:
        if isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            first_kan_layer = layer
            break

    if first_kan_layer is None:
        return 0, []

    l1_scores = _compute_edge_l1(first_kan_layer)  # (out, in)
    mask = l1_scores >= threshold
    surviving = mask.any(dim=0)  # (in_features,)
    indices = torch.where(surviving)[0].tolist()
    return int(surviving.sum().item()), indices


def _fix_trial_config(trial_config_path: Path, local_train_csv: str, local_test_csv: str) -> ExperimentConfig:
    """Load a sweep trial config, fixing any platform-specific paths."""
    config = load_experiment_config(trial_config_path)
    payload = config.model_dump(mode="python")
    payload["trainer"]["train_csv"] = local_train_csv
    payload["trainer"]["test_csv"] = local_test_csv
    # Strip the tune block — we're retraining, not re-sweeping
    payload["tune"] = None
    return ExperimentConfig.model_validate(payload)


def run_pareto_select(
    pareto_json_path: Path,
    flavor: str,
    *,
    device: str | None = None,
    local_train_csv: str = "data/prudential-life-insurance-assessment/train.csv",
    local_test_csv: str = "data/prudential-life-insurance-assessment/test.csv",
    output_root: Path = Path("outputs"),
) -> dict[str, Any]:
    """Train each Pareto trial, prune, and produce actuary-facing summaries.

    Outputs follow the canonical layout::

        outputs/interpretability/<recipe>/pareto-select-<flavor>/
            reports/<flavor>_pareto_variants.json
            data/<flavor>_pareto_trial_analyses.json
            models/<flavor>_pruned_trial<NNN>.pt   (per recommended trial)
    """

    device = device or detect_device()
    pareto = json.loads(pareto_json_path.read_text())
    pareto_trials = pareto["pareto_front"]
    sweep_dir = pareto_json_path.parent
    study_name = pareto["study_name"]
    recipe = pareto.get("preprocessing_recipe", "kan_paper")
    experiment_name = f"pareto-select-{flavor}"

    # Canonical output directory
    interpret_dir = interpret_run_dir(output_root, recipe, experiment_name)

    print(f"\n{'='*70}")
    print(f"PARETO VARIANT ANALYSIS — {flavor.upper()}")
    print(f"Trials to train: {len(pareto_trials)} | Device: {device}")
    print(f"Output: {interpret_dir}")
    print(f"{'='*70}\n")

    trial_analyses = []
    for trial in pareto_trials:
        trial_num = trial["trial_number"]
        trial_config_path = sweep_dir / f"{study_name}_pareto_trial{trial_num:03d}.yaml"

        if not trial_config_path.exists():
            print(f"  Skipping trial {trial_num}: config not found at {trial_config_path}")
            continue

        print(f"\n--- Trial {trial_num} (λ={trial['params']['sparsity_lambda']:.5f}) ---")

        # Train from scratch
        config = _fix_trial_config(trial_config_path, local_train_csv, local_test_csv)
        artifacts = run_train(config, device=device)

        qwk = artifacts.metrics.get("qwk")
        if qwk is None:
            print(f"  Trial {trial_num}: no QWK metric, skipping")
            continue

        # Get the trained KAN module
        model = artifacts.model
        if not hasattr(model, "module"):
            print(f"  Trial {trial_num}: model has no .module attribute, skipping")
            continue

        kan_module = model.module
        kan_module.eval()

        # Prune and count surviving features
        pruned_model, stats, masks = prune_kan(kan_module, PRUNING_THRESHOLD)
        n_surviving, surviving_indices = _count_surviving_input_features(
            kan_module, PRUNING_THRESHOLD,
        )

        # Get feature names from eval data
        eval_dir = eval_run_dir(
            output_root, config.preprocessing.recipe, config.trainer.experiment_name, create=False,
        )
        feature_names_path = eval_dir / "feature_names.json"
        if feature_names_path.exists():
            feature_names = json.loads(feature_names_path.read_text())
        else:
            X_eval_path = eval_dir / "X_eval.parquet"
            if X_eval_path.exists():
                feature_names = list(pd.read_parquet(X_eval_path).columns)
            else:
                feature_names = [f"feature_{i}" for i in range(kan_module.kan_layers[0].in_features)]

        surviving_names = [
            feature_names[i] for i in surviving_indices if i < len(feature_names)
        ]

        # Evaluate pruned model QWK
        X_eval_path = eval_dir / "X_eval.parquet"
        y_eval_path = eval_dir / "y_eval.parquet"
        qwk_pruned = None
        if X_eval_path.exists() and y_eval_path.exists():
            X_eval = pd.read_parquet(X_eval_path)
            y_eval = pd.read_parquet(y_eval_path).iloc[:, 0]
            pruned_wrapper = copy.deepcopy(model)
            pruned_wrapper.module = pruned_model
            preds = pruned_wrapper.predict(X_eval)
            qwk_pruned = float(cohen_kappa_score(y_eval, preds, weights="quadratic"))

        analysis = {
            "trial_number": trial_num,
            "sparsity_lambda": trial["params"]["sparsity_lambda"],
            "qwk_trained": round(float(qwk), 6),
            "qwk_pruned": round(qwk_pruned, 6) if qwk_pruned is not None else None,
            "sparsity_ratio": stats.sparsity_ratio,
            "edges_before": stats.edges_before,
            "edges_after": stats.edges_after,
            "surviving_input_features": n_surviving,
            "surviving_feature_names": surviving_names,
            "checkpoint_path": str(artifacts.checkpoint_path) if artifacts.checkpoint_path else None,
        }
        trial_analyses.append(analysis)

        print(
            f"  QWK={qwk:.4f}  pruned_QWK={qwk_pruned or 0:.4f}  "
            f"sparsity={stats.sparsity_ratio:.2%}  "
            f"surviving_features={n_surviving}"
        )

    if not trial_analyses:
        raise RuntimeError("No trials could be trained. Check config paths and data availability.")

    # Baseline = best QWK among trained trials
    baseline_qwk = max(t["qwk_trained"] for t in trial_analyses)

    # Add relative drop
    for t in trial_analyses:
        ref_qwk = t["qwk_pruned"] if t["qwk_pruned"] is not None else t["qwk_trained"]
        drop = baseline_qwk - ref_qwk
        t["qwk_drop_rel_pct"] = round((drop / baseline_qwk) * 100, 2) if baseline_qwk > 0 else 0.0

    # Build cutoff recommendations
    recommendations = {}
    for cutoff in FEATURE_CUTOFFS:
        eligible = [t for t in trial_analyses if t["surviving_input_features"] <= cutoff]
        if eligible:
            best = max(eligible, key=lambda t: t["qwk_pruned"] or t["qwk_trained"])
            recommendations[f"max_{cutoff}_features"] = {
                "cutoff": cutoff,
                "selected_trial": best["trial_number"],
                "qwk": best["qwk_pruned"] or best["qwk_trained"],
                "qwk_drop_rel_pct": best["qwk_drop_rel_pct"],
                "surviving_input_features": best["surviving_input_features"],
                "surviving_feature_names": best["surviving_feature_names"],
                "sparsity_lambda": best["sparsity_lambda"],
                "checkpoint_path": best["checkpoint_path"],
            }
        else:
            recommendations[f"max_{cutoff}_features"] = {
                "cutoff": cutoff,
                "selected_trial": None,
                "note": f"No trial has <={cutoff} surviving features at threshold={PRUNING_THRESHOLD}",
            }

    result = {
        "flavor": flavor,
        "pareto_source": str(pareto_json_path),
        "pruning_threshold": PRUNING_THRESHOLD,
        "baseline_qwk": baseline_qwk,
        "feature_cutoffs_evaluated": FEATURE_CUTOFFS,
        "recommendations": recommendations,
        "trial_analyses": sorted(trial_analyses, key=lambda t: t["surviving_input_features"]),
    }

    # Save to canonical layout
    rep_dir(interpret_dir).mkdir(parents=True, exist_ok=True)
    out_path = rep_dir(interpret_dir) / f"{flavor}_pareto_variants.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Save detailed trial analyses as data artifact
    (data_dir(interpret_dir) / f"{flavor}_pareto_trial_analyses.json").write_text(
        json.dumps(trial_analyses, indent=2)
    )

    # Save pruned checkpoints for recommended trials
    recommended_trials = {
        rec["selected_trial"]
        for rec in recommendations.values()
        if rec.get("selected_trial") is not None
    }
    for t in trial_analyses:
        if t["trial_number"] in recommended_trials and t.get("checkpoint_path"):
            src_ckpt = Path(t["checkpoint_path"])
            if src_ckpt.exists():
                dst = mod_dir(interpret_dir) / f"{flavor}_trial{t['trial_number']:03d}.pt"
                shutil.copy2(src_ckpt, dst)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"PARETO VARIANT SUMMARY — {flavor.upper()}")
    print(f"Baseline QWK: {baseline_qwk:.4f}")
    print(f"{'='*80}")
    print(f"{'Trial':>6} {'lambda':>10} {'QWK':>8} {'Pruned':>8} {'Drop%':>7} {'Sparsity':>9} {'Features':>9}")
    print(f"{'-'*6:>6} {'-'*10:>10} {'-'*8:>8} {'-'*8:>8} {'-'*7:>7} {'-'*9:>9} {'-'*9:>9}")
    for t in sorted(trial_analyses, key=lambda x: -(x["qwk_pruned"] or x["qwk_trained"])):
        print(
            f"{t['trial_number']:6d} "
            f"{t['sparsity_lambda']:10.5f} "
            f"{t['qwk_trained']:8.4f} "
            f"{(t['qwk_pruned'] or 0):8.4f} "
            f"{t['qwk_drop_rel_pct']:6.1f}% "
            f"{t['sparsity_ratio']:8.2%} "
            f"{t['surviving_input_features']:9d}"
        )

    print(f"\nRECOMMENDATIONS FOR ACTUARY REVIEW:")
    for key, rec in recommendations.items():
        if rec.get("selected_trial") is not None:
            names = rec["surviving_feature_names"]
            print(
                f"  {key}: trial {rec['selected_trial']} — "
                f"QWK {rec['qwk']:.4f} ({rec['qwk_drop_rel_pct']:+.1f}%), "
                f"{rec['surviving_input_features']} features: "
                f"{', '.join(names[:8])}"
                f"{'...' if len(names) > 8 else ''}"
            )
        else:
            print(f"  {key}: {rec['note']}")

    print(f"\nSaved -> {out_path}")
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + prune Pareto variants for actuary review")
    p.add_argument("--pareto-json", type=Path, required=True, help="Path to *_pareto.json from sweep")
    p.add_argument("--flavor", choices=["chebykan", "fourierkan", "bsplinekan"], required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--train-csv", default="data/prudential-life-insurance-assessment/train.csv")
    p.add_argument("--test-csv", default="data/prudential-life-insurance-assessment/test.csv")
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run_pareto_select(
        pareto_json_path=args.pareto_json,
        flavor=args.flavor,
        device=args.device,
        local_train_csv=args.train_csv,
        local_test_csv=args.test_csv,
        output_root=args.output_root,
    )
