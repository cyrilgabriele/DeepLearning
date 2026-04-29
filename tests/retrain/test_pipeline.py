import json
from pathlib import Path

from src.models import TrainingArtifacts
from src.retrain.pipeline import run_retrain


def _candidate_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "study_name": "cheby-study",
                "model_family": "chebykan",
                "preprocessing_recipe": "kan_paper",
                "generated_at": "2026-04-09T00:00:00+00:00",
                "top_k": 2,
                "candidates": [
                    {
                        "candidate_id": "cheby-study-trial-000",
                        "rank": 1,
                        "trial_number": 0,
                        "qwk": 0.8,
                        "metrics": {"qwk": 0.8},
                        "model_config": {
                            "name": "tabkan-base",
                            "flavor": "chebykan",
                            "hidden_widths": [64, 64],
                            "depth": 2,
                            "width": 64,
                            "degree": 4,
                            "params": {
                            "max_epochs": 10,
                            "lr": 1e-3,
                            "weight_decay": 0.0,
                            "batch_size": 256,
                            "sparsity_lambda": 0.0,
                            "l1_weight": 1.0,
                            "entropy_weight": 1.0,
                        },
                        },
                        "trainer_config": {
                            "experiment_name": "cheby-base-trial-000",
                            "train_csv": str(tmp_path / "train.csv"),
                            "test_csv": str(tmp_path / "test.csv"),
                            "seed": 11,
                        },
                        "preprocessing_config": {
                            "contract_version": 1,
                            "recipe": "kan_paper",
                        },
                        "selection_metadata": {},
                    }
                ],
            }
        )
    )
    return path


def test_run_retrain_materializes_candidate_seed_runs(tmp_path, monkeypatch):
    candidate_manifest = _candidate_manifest(tmp_path)
    recorded = []

    def fake_run_train(config, *, device):
        recorded.append((config, device))
        stem = config.trainer.experiment_name
        summary_path = tmp_path / f"{stem}.summary.json"
        checkpoint_path = tmp_path / f"{stem}.pt"
        summary_path.write_text("{}")
        checkpoint_path.write_text("checkpoint")
        return TrainingArtifacts(
            model=object(),
            metrics={"mae": 1.0, "accuracy": 0.5, "f1_macro": 0.4, "qwk": 0.81},
            device=device,
            config=config,
            random_seed=config.trainer.seed,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
        )

    monkeypatch.setattr("src.retrain.pipeline.run_train", fake_run_train)

    result = run_retrain(
        candidate_manifest,
        device="cpu",
        seeds=[101, 202],
        selection_name="shortlist",
    )

    assert len(recorded) == 2
    assert recorded[0][0].trainer.seed == 101
    assert recorded[1][0].trainer.seed == 202
    assert recorded[0][0].model.params["sparsity_lambda"] == 0.0
    assert result["model_family"] == "chebykan"
    assert result["selection_name"] == "shortlist"
    assert len(result["runs"]) == 2
    assert Path(result["manifest_path"]).exists()
