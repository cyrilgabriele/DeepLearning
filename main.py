"""Project entry point orchestrating Prudential experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.models import TrainingArtifacts


def run(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and dispatch the requested experiment stage."""

    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    from src.configs import detect_device, load_experiment_config

    config = load_experiment_config(args.config)
    device = detect_device()

    if args.stage == "train":
        from src.training.trainer import run_train

        artifacts = run_train(config, device=device)
        _print_training_summary(artifacts)
        return

    if args.stage == "tune":
        from src.tune.sweep import run_tune

        run_tune(
            config,
            device=device,
            n_trials_override=args.n_trials,
            timeout_override=args.timeout_tune,
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prudential experiment entry point")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["train", "tune"],
        help="Experiment stage to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trial count for stage 'tune'.",
    )
    parser.add_argument(
        "--timeout-tune",
        type=int,
        default=None,
        help="Override Optuna timeout in seconds for stage 'tune'.",
    )
    return parser


def _print_training_summary(artifacts: "TrainingArtifacts") -> None:
    print(f"\nExperiment '{artifacts.config.trainer.experiment_name}' finished.")
    print(f"Seed: {artifacts.random_seed}")
    print(f"Device: {artifacts.device}")
    for metric, value in artifacts.metrics.items():
        if value is None:
            print(f"  {metric}: N/A")
        else:
            print(f"  {metric}: {value:.4f}")
    if artifacts.summary_path is not None:
        print(f"Run summary saved to {artifacts.summary_path}")
    if artifacts.checkpoint_path is not None:
        print(f"Checkpoint saved to {artifacts.checkpoint_path}")
    if artifacts.test_predictions_path is not None:
        print(f"Test predictions saved to {artifacts.test_predictions_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
