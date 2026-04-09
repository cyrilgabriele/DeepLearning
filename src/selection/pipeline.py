"""Family-wise selector for retrained KAN candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd


def run_select(
    retrain_manifest_path: Path,
    *,
    qwk_tolerance: float = 0.01,
    output_root: Path = Path("outputs"),
    selection_output_root: Path = Path("artifacts/selection"),
    interpretability_root: Path | None = None,
) -> dict[str, Any]:
    """Select best-performance and best-interpretable KAN candidates per family."""

    retrain_manifest = json.loads(retrain_manifest_path.read_text())
    family = retrain_manifest.get("family") or retrain_manifest["model_family"]
    preprocessing_recipe = retrain_manifest.get("preprocessing_recipe")
    runs = [_normalize_run_entry(dict(run)) for run in retrain_manifest.get("runs", [])]
    if not runs:
        raise ValueError("Retrain manifest does not contain any runs.")

    resolved_interpretability_root = interpretability_root or output_root
    resolved_selection_output_root = (
        output_root if interpretability_root is not None else selection_output_root
    )

    for run in runs:
        interpretability = _load_interpretability_metrics(
            run,
            output_root=resolved_interpretability_root,
            family=family,
        )
        run["interpretability"] = interpretability

    candidate_summaries = _summarize_candidates(runs)
    if not candidate_summaries:
        raise ValueError("No candidate summaries could be derived from the retrain manifest.")

    best_performance = max(
        candidate_summaries,
        key=lambda summary: (
            summary["mean_qwk"],
            -(summary["mean_edges_after"] if summary["mean_edges_after"] is not None else float("inf")),
        ),
    )
    tolerance_floor = best_performance["mean_qwk"] - qwk_tolerance
    eligible = [
        summary
        for summary in candidate_summaries
        if summary["mean_qwk"] >= tolerance_floor
    ]
    best_interpretable = min(eligible, key=_interpretable_sort_key)

    payload = {
        "family": family,
        "model_family": family,
        "preprocessing_recipe": preprocessing_recipe,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_retrain_manifest": str(retrain_manifest_path),
        "qwk_tolerance": qwk_tolerance,
        "best_performance_candidate": best_performance,
        "best_interpretable_candidate": best_interpretable,
        "eligible_candidate_ids": [summary["candidate_id"] for summary in eligible],
        "candidate_summaries": candidate_summaries,
        "ranking_criteria": {
            "predictive_metric": "mean validation QWK across seeds",
            "interpretability_tie_breakers": [
                "fewer active edges after pruning",
                "lower basis complexity",
                "higher symbolic-fit R2",
                "lower seed-to-seed QWK variance",
                "higher sparsity ratio",
            ],
        },
    }

    resolved_selection_output_root.mkdir(parents=True, exist_ok=True)
    output_path = resolved_selection_output_root / f"{family}_selection.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    payload["selection_path"] = str(output_path)
    return payload


def _normalize_run_entry(run: dict[str, Any]) -> dict[str, Any]:
    """Accept both flattened and nested retrain-manifest run schemas."""

    config_payload = run.get("config")
    if isinstance(config_payload, dict):
        run.setdefault("trainer_config", config_payload.get("trainer", {}))
        run.setdefault("preprocessing_config", config_payload.get("preprocessing", {}))
        run.setdefault("model_config", config_payload.get("model", {}))
    return run


def _load_interpretability_metrics(
    run: dict[str, Any],
    *,
    output_root: Path,
    family: str,
) -> dict[str, Any]:
    config_payload = run.get("config")
    if isinstance(config_payload, dict):
        recipe = config_payload.get("preprocessing", {}).get("recipe")
        model_config = config_payload.get("model", {})
    else:
        recipe = run.get("preprocessing_config", {}).get("recipe")
        model_config = run.get("model_config", {})
    experiment_name = run["experiment_name"]
    if recipe is None:
        raise ValueError(f"Run '{experiment_name}' does not include a preprocessing recipe.")
    interpret_dir = output_root / "interpretability" / recipe / experiment_name
    reports_dir = interpret_dir / "reports"
    data_dir = interpret_dir / "data"

    pruning_summary = _safe_read_json(reports_dir / f"{family}_pruning_summary.json")
    r2_report = _safe_read_json(reports_dir / f"{family}_r2_report.json")
    symbolic_fits_path = data_dir / f"{family}_symbolic_fits.csv"
    symbolic_mean_r2 = None
    if symbolic_fits_path.exists():
        symbolic_df = pd.read_csv(symbolic_fits_path)
        if not symbolic_df.empty:
            symbolic_mean_r2 = float(symbolic_df["r_squared"].mean())

    return {
        "interpret_dir": str(interpret_dir),
        "pruning_summary_path": str(reports_dir / f"{family}_pruning_summary.json"),
        "r2_report_path": str(reports_dir / f"{family}_r2_report.json"),
        "symbolic_fits_path": str(symbolic_fits_path),
        "edges_after": pruning_summary.get("edges_after"),
        "sparsity_ratio": pruning_summary.get("sparsity_ratio"),
        "qwk_after_pruning": pruning_summary.get("qwk_after"),
        "mean_symbolic_r2": (
            r2_report.get("aggregate", {}).get("mean_r2")
            if r2_report
            else symbolic_mean_r2
        ),
        "basis_complexity": _basis_complexity(model_config),
    }


def _summarize_candidates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["candidate_id"], []).append(run)

    summaries: list[dict[str, Any]] = []
    for candidate_id, candidate_runs in grouped.items():
        qwk_values = [
            float(run["metrics"]["qwk"])
            for run in candidate_runs
            if run.get("metrics", {}).get("qwk") is not None
        ]
        edges_after = [
            int(run["interpretability"]["edges_after"])
            for run in candidate_runs
            if run["interpretability"].get("edges_after") is not None
        ]
        sparsity = [
            float(run["interpretability"]["sparsity_ratio"])
            for run in candidate_runs
            if run["interpretability"].get("sparsity_ratio") is not None
        ]
        symbolic_r2 = [
            float(run["interpretability"]["mean_symbolic_r2"])
            for run in candidate_runs
            if run["interpretability"].get("mean_symbolic_r2") is not None
        ]
        basis_complexity = [
            float(run["interpretability"]["basis_complexity"])
            for run in candidate_runs
            if run["interpretability"].get("basis_complexity") is not None
        ]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "source_trial_number": candidate_runs[0].get("source_trial_number")
                or candidate_runs[0].get("trial_number"),
                "run_ids": [
                    run.get("run_id") or f"{run['experiment_name']}@seed-{run['seed']}"
                    for run in candidate_runs
                ],
                "mean_qwk": mean(qwk_values),
                "qwk_std": pstdev(qwk_values) if len(qwk_values) > 1 else 0.0,
                "mean_edges_after": mean(edges_after) if edges_after else None,
                "mean_sparsity_ratio": mean(sparsity) if sparsity else None,
                "mean_symbolic_r2": mean(symbolic_r2) if symbolic_r2 else None,
                "mean_basis_complexity": mean(basis_complexity) if basis_complexity else None,
                "example_run": candidate_runs[0],
            }
        )

    return sorted(summaries, key=lambda summary: summary["candidate_id"])


def _interpretable_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    symbolic_r2 = summary["mean_symbolic_r2"]
    sparsity_ratio = summary["mean_sparsity_ratio"]
    edges_after = summary["mean_edges_after"]
    basis_complexity = summary["mean_basis_complexity"]
    return (
        float(edges_after) if edges_after is not None else float("inf"),
        float(basis_complexity) if basis_complexity is not None else float("inf"),
        -(float(symbolic_r2) if symbolic_r2 is not None else 0.0),
        float(summary["qwk_std"]),
        -(float(sparsity_ratio) if sparsity_ratio is not None else 0.0),
        -float(summary["mean_qwk"]),
    )


def _basis_complexity(model_config: dict[str, Any]) -> float:
    params = model_config.get("params", {})
    if model_config.get("flavor") == "chebykan":
        return float(model_config.get("degree") or params.get("degree") or 0)
    return float(params.get("grid_size") or 0)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
