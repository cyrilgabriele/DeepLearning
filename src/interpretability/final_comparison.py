"""Manifest-driven final comparison assembly for current pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _mask_features(frame, keep: list[str]):
    """Return a copy of *frame* with non-retained columns zeroed out."""

    masked = frame.copy()
    for column in masked.columns:
        if column not in keep:
            masked[column] = 0.0
    return masked


def _top5_overlap(rankings: dict[str, list[str]]) -> int:
    """Return the number of features shared by the top 5 lists of all rankings."""

    if not rankings:
        return 0
    shared = set(next(iter(rankings.values()))[:5])
    for ranked in rankings.values():
        shared &= set(ranked[:5])
    return len(shared)


def run(
    selection_manifest_paths: Iterable[Path],
    *,
    output_root: Path = Path("outputs"),
    baseline_summary_paths: Iterable[Path] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the final comparison package from modern artifact manifests."""

    kan_records = []
    for manifest_path in selection_manifest_paths:
        selection = json.loads(Path(manifest_path).read_text())
        kan_records.extend(_records_from_selection_manifest(selection, output_root=output_root))

    baseline_records = []
    for summary_path in baseline_summary_paths or []:
        baseline_records.append(_record_from_summary(Path(summary_path), output_root=output_root))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_manifests": [str(Path(path)) for path in selection_manifest_paths],
        "models": kan_records + baseline_records,
    }

    destination = report_path or output_root / "final_comparison" / "final_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    markdown_path = destination.with_suffix(".md")
    markdown_path.write_text(_render_markdown(payload))
    payload["report_path"] = str(destination)
    payload["markdown_path"] = str(markdown_path)
    return payload


def _records_from_selection_manifest(
    selection: dict[str, Any],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for label in ("best_performance_candidate", "best_interpretable_candidate"):
        candidate = selection.get(label)
        if not candidate:
            continue
        records.append(
            _record_from_selected_candidate(
                selection=selection,
                candidate=candidate,
                label=label,
                output_root=output_root,
            )
        )
    return records


def _record_from_selected_candidate(
    *,
    selection: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
    output_root: Path,
) -> dict[str, Any]:
    example_run = candidate.get("example_run", {})
    recipe = example_run.get("preprocessing_config", {}).get("recipe")
    experiment_name = example_run.get("experiment_name")
    family = selection.get("model_family")
    interpret_dir = output_root / "interpretability" / recipe / experiment_name
    eval_dir = output_root / "eval" / recipe / experiment_name
    return {
        "model_label": f"{family}:{label}",
        "family": family,
        "selection_role": label,
        "candidate_id": candidate.get("candidate_id"),
        "source_trial_number": candidate.get("source_trial_number"),
        "experiment_name": experiment_name,
        "preprocessing_recipe": recipe,
        "metrics": example_run.get("metrics", {}),
        "selection_metrics": {
            "mean_qwk": candidate.get("mean_qwk"),
            "qwk_std": candidate.get("qwk_std"),
            "mean_edges_after": candidate.get("mean_edges_after"),
            "mean_sparsity_ratio": candidate.get("mean_sparsity_ratio"),
            "mean_symbolic_r2": candidate.get("mean_symbolic_r2"),
            "mean_basis_complexity": candidate.get("mean_basis_complexity"),
        },
        "checkpoint_path": example_run.get("checkpoint_path"),
        "summary_path": example_run.get("summary_path"),
        "eval_dir": str(eval_dir),
        "interpret_dir": str(interpret_dir),
        "artifacts": {
            "pruning_summary": str(interpret_dir / "reports" / f"{family}_pruning_summary.json"),
            "r2_report": str(interpret_dir / "reports" / f"{family}_r2_report.json"),
            "symbolic_fits": str(interpret_dir / "data" / f"{family}_symbolic_fits.csv"),
        },
        "config": {
            "trainer": example_run.get("trainer_config", {}),
            "preprocessing": example_run.get("preprocessing_config", {}),
            "model": example_run.get("model_config", {}),
        },
    }


def _record_from_summary(summary_path: Path, *, output_root: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text())
    config = payload.get("config", {})
    preprocessing = config.get("preprocessing", {})
    trainer = config.get("trainer", {})
    model = config.get("model", {})
    recipe = preprocessing.get("recipe")
    experiment_name = trainer.get("experiment_name")
    return {
        "model_label": model.get("name", experiment_name),
        "family": model.get("flavor") or model.get("name"),
        "selection_role": "baseline",
        "candidate_id": None,
        "source_trial_number": None,
        "experiment_name": experiment_name,
        "preprocessing_recipe": recipe,
        "metrics": payload.get("metrics", {}),
        "selection_metrics": {},
        "checkpoint_path": payload.get("checkpoint_path"),
        "summary_path": str(summary_path),
        "eval_dir": str(output_root / "eval" / recipe / experiment_name),
        "interpret_dir": str(output_root / "interpretability" / recipe / experiment_name),
        "artifacts": {},
        "config": config,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final Comparison",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        "| Model | Role | Mean QWK | QWK Std | Edges After | Mean Symbolic R2 | Checkpoint |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in payload["models"]:
        selection_metrics = record.get("selection_metrics", {})
        lines.append(
            "| {model} | {role} | {qwk} | {std} | {edges} | {r2} | `{checkpoint}` |".format(
                model=record["model_label"],
                role=record["selection_role"],
                qwk=_fmt(selection_metrics.get("mean_qwk") or record.get("metrics", {}).get("qwk")),
                std=_fmt(selection_metrics.get("qwk_std")),
                edges=_fmt(selection_metrics.get("mean_edges_after")),
                r2=_fmt(selection_metrics.get("mean_symbolic_r2")),
                checkpoint=record.get("checkpoint_path") or "N/A",
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _resolve_latest_summary_from_config(config_path: Path) -> Path:
    from src.config import load_experiment_config

    config = load_experiment_config(config_path)
    summary_root = Path("artifacts") / config.trainer.experiment_name
    candidates = sorted(summary_root.glob("run-summary-*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No run summary found under {summary_root}. "
            f"Train `{config_path}` before using it as a baseline input."
        )
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the final comparison package.")
    parser.add_argument(
        "--selection-manifest",
        dest="selection_manifests",
        type=Path,
        action="append",
        required=True,
        help="Selection manifest to include. Repeat for multiple families.",
    )
    parser.add_argument(
        "--baseline-summary",
        dest="baseline_summaries",
        type=Path,
        action="append",
        default=None,
        help="Baseline run summary to include. Repeat for multiple baselines.",
    )
    parser.add_argument(
        "--baseline-config",
        dest="baseline_configs",
        type=Path,
        action="append",
        default=None,
        help=(
            "Baseline experiment config to resolve from `artifacts/<experiment>/run-summary-*.json`. "
            "Repeat for multiple baselines."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    baseline_summary_paths = list(args.baseline_summaries or [])
    baseline_summary_paths.extend(
        _resolve_latest_summary_from_config(config_path)
        for config_path in (args.baseline_configs or [])
    )
    result = run(
        args.selection_manifests,
        output_root=args.output_root,
        baseline_summary_paths=baseline_summary_paths,
        report_path=args.report_path,
    )
    print(f"Saved JSON report: {result['report_path']}")
    print(f"Saved Markdown report: {result['markdown_path']}")
