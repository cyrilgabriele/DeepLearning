import json

from src.selection.pipeline import run_select


def test_run_select_accepts_retrain_manifest_schema_and_writes_family_selection(tmp_path):
    retrain_manifest = tmp_path / "manifest.json"
    retrain_manifest.write_text(
        json.dumps(
            {
                "family": "chebykan",
                "selection_name": "shortlist",
                "generated_at": "2026-04-09T00:00:00+00:00",
                "preprocessing_recipe": "kan_paper",
                "runs": [
                    {
                        "run_id": "cand-a-seed-11",
                        "candidate_id": "cand-a",
                        "source_trial_number": 1,
                        "seed": 11,
                        "experiment_name": "cand-a-seed-11",
                        "config": {
                            "trainer": {
                                "experiment_name": "cand-a-seed-11",
                                "train_csv": str(tmp_path / "train.csv"),
                                "seed": 11,
                            },
                            "preprocessing": {
                                "contract_version": 1,
                                "recipe": "kan_paper",
                            },
                            "model": {
                                "name": "tabkan-base",
                                "flavor": "chebykan",
                                "hidden_widths": [64, 64],
                                "depth": 2,
                                "width": 64,
                                "degree": 4,
                                "params": {},
                            },
                        },
                        "metrics": {"qwk": 0.82},
                    },
                    {
                        "run_id": "cand-b-seed-11",
                        "candidate_id": "cand-b",
                        "source_trial_number": 2,
                        "seed": 11,
                        "experiment_name": "cand-b-seed-11",
                        "config": {
                            "trainer": {
                                "experiment_name": "cand-b-seed-11",
                                "train_csv": str(tmp_path / "train.csv"),
                                "seed": 11,
                            },
                            "preprocessing": {
                                "contract_version": 1,
                                "recipe": "kan_paper",
                            },
                            "model": {
                                "name": "tabkan-base",
                                "flavor": "chebykan",
                                "hidden_widths": [32, 32],
                                "depth": 2,
                                "width": 32,
                                "degree": 3,
                                "params": {},
                            },
                        },
                        "metrics": {"qwk": 0.815},
                    },
                ],
            }
        )
    )

    for experiment_name, edges_after, sparsity_ratio, mean_r2 in (
        ("cand-a-seed-11", 24, 0.35, 0.91),
        ("cand-b-seed-11", 12, 0.52, 0.96),
    ):
        interpret_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / experiment_name
        reports_dir = interpret_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "chebykan_pruning_summary.json").write_text(
            json.dumps(
                {
                    "edges_after": edges_after,
                    "sparsity_ratio": sparsity_ratio,
                    "qwk_after": 0.8,
                }
            )
        )
        (reports_dir / "chebykan_r2_report.json").write_text(
            json.dumps({"aggregate": {"mean_r2": mean_r2}})
        )

    result = run_select(
        retrain_manifest,
        output_root=tmp_path / "outputs",
        selection_output_root=tmp_path / "artifacts" / "selection",
    )

    assert result["family"] == "chebykan"
    assert result["best_performance_candidate"]["candidate_id"] == "cand-a"
    assert result["best_interpretable_candidate"]["candidate_id"] == "cand-b"
    assert (tmp_path / "artifacts" / "selection" / "chebykan_selection.json").exists()
