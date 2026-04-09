"""Project entry point orchestrating Prudential experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.models import TrainingArtifacts


def run(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and dispatch the requested experiment stage."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.stage in {"train", "tune"} and args.config is None:
        parser.error("--config is required for stages 'train' and 'tune'.")

    if args.stage == "interpret" and args.config is None and args.checkpoint is None:
        parser.error("stage 'interpret' requires --checkpoint when --config is omitted.")

    if args.stage == "retrain" and args.candidate_manifest is None:
        parser.error("stage 'retrain' requires --candidate-manifest.")

    if args.stage == "retrain" and not args.seeds:
        parser.error("stage 'retrain' requires at least one seed via --seeds.")

    if args.stage == "select" and args.retrain_manifest is None:
        parser.error("stage 'select' requires --retrain-manifest.")

    if args.stage == "train":
        from configs import detect_device
        from configs import load_experiment_config
        from src.training.trainer import run_train

        config = load_experiment_config(args.config)
        device = detect_device()
        artifacts = run_train(config, device=device)
        _print_training_summary(artifacts)
        return

    if args.stage == "tune":
        from configs import detect_device
        from configs import load_experiment_config
        from src.tune.sweep import run_tune

        config = load_experiment_config(args.config)
        device = detect_device()
        run_tune(
            config,
            device=device,
            n_trials_override=args.n_trials,
            timeout_override=args.timeout_tune,
        )
        return

    if args.stage == "interpret":
        from src.interpretability.pipeline import resolve_interpret_config, run_interpret

        config = resolve_interpret_config(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )

        result = run_interpret(
            config,
            checkpoint_path=args.checkpoint,
            output_root=args.output_root,
            pruning_threshold=args.pruning_threshold,
            qwk_tolerance=args.qwk_tolerance,
            candidate_library=args.candidate_library,
        )
        _print_interpret_summary(result)
        return

    if args.stage == "retrain":
        from configs import detect_device
        from src.retrain import run_retrain

        device = detect_device()
        result = run_retrain(
            args.candidate_manifest,
            device=device,
            seeds=args.seeds,
            candidate_ids=args.candidate_ids,
            top_k=args.top_k,
            selection_name=args.selection_name,
            experiment_prefix=args.output_experiment_prefix,
        )
        _print_retrain_summary(result)
        return

    if args.stage == "select":
        from src.selection import run_select

        result = run_select(
            args.retrain_manifest,
            qwk_tolerance=args.qwk_tolerance,
            selection_output_root=Path("artifacts") / "selection",
            output_root=args.output_root,
        )
        _print_selection_summary(result)
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prudential experiment entry point")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["train", "tune", "interpret", "retrain", "select"],
        help="Experiment stage to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML experiment configuration file. Required for 'train' and 'tune'.",
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Override checkpoint path for stage 'interpret'. "
            "Required when omitting --config; otherwise defaults to the latest run checkpoint."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for eval and interpretability outputs.",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=0.01,
        help="Initial pruning threshold for stage 'interpret' on supported KAN models.",
    )
    parser.add_argument(
        "--qwk-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed QWK drop during KAN pruning for stage 'interpret'.",
    )
    parser.add_argument(
        "--candidate-library",
        choices=["scipy", "pysr"],
        default="scipy",
        help="Symbolic fitting backend for KAN interpretability.",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="Top-k candidate manifest emitted by the tune stage for retraining.",
    )
    parser.add_argument(
        "--retrain-manifest",
        type=Path,
        default=None,
        help="Retraining manifest emitted by the retrain stage for final selection.",
    )
    parser.add_argument(
        "--candidate-id",
        dest="candidate_ids",
        action="append",
        default=None,
        help="Candidate id to retrain. Repeat to retrain multiple explicit candidates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit the retrain stage to the first K candidates from the candidate manifest.",
    )
    parser.add_argument(
        "--selection-name",
        type=str,
        default=None,
        help="Logical label used under `artifacts/retrain/<family>/...`.",
    )
    parser.add_argument(
        "--output-experiment-prefix",
        type=str,
        default=None,
        help="Prefix used for the materialized experiment names during retraining.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed list for the retrain stage.",
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


def _print_interpret_summary(result: dict[str, object]) -> None:
    print(f"\nInterpretability run finished for '{result['experiment_name']}'.")
    print(f"Model: {result['model']}")
    print(f"Recipe: {result['recipe']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Eval artifacts: {result['eval_dir']}")
    print(f"Interpret outputs: {result['output_dir']}")
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        for label, path in artifacts.items():
            print(f"  {label}: {path}")


def _print_retrain_summary(result: dict[str, object]) -> None:
    print("\nRetrain stage finished.")
    print(f"Family: {result['model_family']}")
    print(f"Selection: {result['selection_name']}")
    print(f"Runs materialized: {len(result.get('runs', []))}")
    print(f"Manifest: {result.get('manifest_path')}")


def _print_selection_summary(result: dict[str, object]) -> None:
    print("\nSelection stage finished.")
    print(f"Family: {result['family']}")
    print(f"Best performance: {result['best_performance_candidate']['candidate_id']}")
    print(f"Best interpretable: {result['best_interpretable_candidate']['candidate_id']}")
    manifest_path = result.get("selection_path") or (
        Path("artifacts") / "selection" / f"{result['family']}_selection.json"
    )
    print(f"Manifest: {manifest_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
