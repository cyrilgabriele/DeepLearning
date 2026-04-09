"""Materialize selected KAN candidates across multiple retraining seeds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config import ExperimentConfig
from src.training.trainer import run_train


DEFAULT_SPARSITY_LAMBDA = 1e-3


def run_retrain(
    candidate_manifest_path: Path,
    *,
    device: str,
    candidate_ids: Iterable[str] | None = None,
    top_k: int | None = None,
    seeds: Iterable[int] | None = None,
    selection_name: str | None = None,
    experiment_prefix: str | None = None,
    output_experiment_prefix: str | None = None,
    output_root: Path = Path("artifacts/retrain"),
) -> dict[str, Any]:
    """Retrain selected KAN candidates across multiple seeds and persist a manifest."""

    manifest = json.loads(candidate_manifest_path.read_text())
    family = manifest["model_family"]
    resolved_experiment_prefix = output_experiment_prefix or experiment_prefix
    candidates = _select_candidates(
        manifest["candidates"],
        candidate_ids=set(candidate_ids or []),
        top_k=top_k,
    )
    if not candidates:
        raise ValueError("Retrain stage did not resolve any candidates from the candidate manifest.")

    seed_list = [int(seed) for seed in (seeds or [42])]
    resolved_selection_name = selection_name or candidate_manifest_path.stem.removesuffix("_candidates")
    run_entries: list[dict[str, Any]] = []

    for candidate in candidates:
        for seed in seed_list:
            config = _build_retrain_config(
                candidate,
                seed=seed,
                experiment_prefix=resolved_experiment_prefix,
                selection_name=resolved_selection_name,
            )
            artifacts = run_train(config, device=device)
            run_entries.append(
                {
                    "run_id": f"{candidate['candidate_id']}-seed-{seed}",
                    "candidate_id": candidate["candidate_id"],
                    "candidate_rank": candidate["rank"],
                    "source_trial_number": candidate["trial_number"],
                    "trial_number": candidate["trial_number"],
                    "seed": seed,
                    "experiment_name": config.trainer.experiment_name,
                    "config": config.model_dump(mode="json"),
                    "model_config": config.model.model_dump(mode="json"),
                    "trainer_config": config.trainer.model_dump(mode="json"),
                    "preprocessing_config": config.preprocessing.model_dump(mode="json"),
                    "metrics": artifacts.metrics,
                    "summary_path": str(artifacts.summary_path) if artifacts.summary_path else None,
                    "checkpoint_path": str(artifacts.checkpoint_path) if artifacts.checkpoint_path else None,
                    "test_predictions_path": (
                        str(artifacts.test_predictions_path)
                        if artifacts.test_predictions_path
                        else None
                    ),
                    "selection_metadata": dict(candidate.get("selection_metadata", {})),
                }
            )

    manifest_payload = {
        "family": family,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidate_manifest": str(candidate_manifest_path),
        "source_study_name": manifest.get("study_name"),
        "model_family": family,
        "preprocessing_recipe": manifest.get("preprocessing_recipe"),
        "selection_name": resolved_selection_name,
        "seed_list": seed_list,
        "selected_candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "runs": run_entries,
    }

    family_dir = output_root / family / resolved_selection_name
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
    manifest_payload["manifest_path"] = str(manifest_path)
    return manifest_payload


def _select_candidates(
    candidates: list[dict[str, Any]],
    *,
    candidate_ids: set[str],
    top_k: int | None,
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda candidate: candidate.get("rank", 0))
    if candidate_ids:
        ranked = [candidate for candidate in ranked if candidate["candidate_id"] in candidate_ids]
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def _build_retrain_config(
    candidate: dict[str, Any],
    *,
    seed: int,
    experiment_prefix: str | None,
    selection_name: str,
) -> ExperimentConfig:
    if "config" in candidate:
        payload = dict(candidate["config"])
    else:
        payload = {
            "trainer": dict(candidate["trainer_config"]),
            "preprocessing": dict(candidate["preprocessing_config"]),
            "model": dict(candidate["model_config"]),
        }
    payload["trainer"]["seed"] = seed
    base_name = experiment_prefix or selection_name
    payload["trainer"]["experiment_name"] = (
        f"{base_name}-{candidate['candidate_id']}-seed-{seed}"
    )
    model_params = dict(payload["model"].get("params") or {})
    model_name = str(payload["model"].get("name", ""))
    if model_name.startswith("tabkan"):
        sparsity = float(model_params.get("sparsity_lambda", 0.0) or 0.0)
        if sparsity <= 0.0:
            model_params["sparsity_lambda"] = DEFAULT_SPARSITY_LAMBDA
        model_params.setdefault("l1_weight", 1.0)
        model_params.setdefault("entropy_weight", 1.0)
    payload["model"]["params"] = model_params
    return ExperimentConfig.model_validate(payload)
