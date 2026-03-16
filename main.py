"""Project entry point orchestrating Prudential experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
from venv import logger

from src.configs import detect_device, load_experiment_config, set_global_seed, GLOBAL_RANDOM_SEED
from src.training.trainer import Trainer


def run(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and dispatch the trainer."""

    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    seed = set_global_seed()
    logger.info(f"seed: {seed}")
    device = detect_device()
    logger.info(f"device: {device}")
    config = load_experiment_config(args.config)


    trainer = Trainer(config, random_seed=seed, device=device)
    artifacts = trainer.run()

    print(f"\nExperiment '{config.trainer.experiment_name}' finished.")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prudential experiment entry point")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration file (single source of truth).",
    )
    return parser


def main() -> None:
    run()


if __name__ == "__main__":
    main()
