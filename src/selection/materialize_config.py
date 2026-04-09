"""Materialize a selected candidate config from a selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.config import ExperimentConfig


def materialize_selected_config(
    selection_manifest_path: Path,
    *,
    role: str,
    output_path: Path,
) -> Path:
    """Write the selected candidate config as a standalone experiment YAML."""

    selection = json.loads(selection_manifest_path.read_text())
    candidate = selection.get(role)
    if not isinstance(candidate, dict):
        raise ValueError(
            f"Selection manifest {selection_manifest_path} does not contain '{role}'."
        )

    example_run = candidate.get("example_run")
    if not isinstance(example_run, dict):
        raise ValueError(
            f"Selection manifest {selection_manifest_path} is missing example_run for '{role}'."
        )

    raw_config = example_run.get("config")
    if not isinstance(raw_config, dict):
        raw_config = {
            "trainer": dict(example_run.get("trainer_config") or {}),
            "preprocessing": dict(example_run.get("preprocessing_config") or {}),
            "model": dict(example_run.get("model_config") or {}),
        }

    config = ExperimentConfig.model_validate(raw_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
        )
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a selected config from a selection manifest.")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=["best_performance_candidate", "best_interpretable_candidate"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = materialize_selected_config(
        args.selection_manifest,
        role=args.role,
        output_path=args.output,
    )
    print(f"Saved config: {path}")
