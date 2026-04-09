import pytest
import numpy as np
import pandas as pd
import json


def test_mask_features_zeros_out_dropped_columns():
    from src.interpretability.final_comparison import _mask_features
    X = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    result = _mask_features(X, keep=["A", "C"])
    assert list(result["A"]) == [1.0, 2.0]
    assert list(result["B"]) == [0.0, 0.0]
    assert list(result["C"]) == [5.0, 6.0]
    assert list(X["B"]) == [3.0, 4.0]  # original unchanged


def test_top5_overlap_count():
    from src.interpretability.final_comparison import _top5_overlap
    rankings = {
        "GLM": ["A", "B", "C", "D", "E"],
        "XGBoost": ["A", "B", "C", "X", "Y"],
        "ChebyKAN": ["A", "B", "Z", "D", "E"],
        "FourierKAN": ["A", "B", "C", "D", "W"],
    }
    assert _top5_overlap(rankings) == 2  # "A" and "B" in all four


def test_run_assembles_manifest_driven_comparison(tmp_path):
    from src.interpretability.final_comparison import run

    selection_manifest = tmp_path / "cheby_selection.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "model_family": "chebykan",
                "best_performance_candidate": {
                    "candidate_id": "cand-a",
                    "source_trial_number": 4,
                    "mean_qwk": 0.82,
                    "qwk_std": 0.01,
                    "mean_edges_after": 12,
                    "mean_symbolic_r2": 0.95,
                    "example_run": {
                        "experiment_name": "cand-a-seed-11",
                        "checkpoint_path": "checkpoints/cand-a.pt",
                        "summary_path": "artifacts/cand-a.json",
                        "metrics": {"qwk": 0.82},
                        "trainer_config": {"experiment_name": "cand-a-seed-11"},
                        "preprocessing_config": {"recipe": "kan_paper"},
                        "model_config": {
                            "name": "tabkan-base",
                            "flavor": "chebykan",
                            "hidden_widths": [64, 64],
                        },
                    },
                },
                "best_interpretable_candidate": {
                    "candidate_id": "cand-b",
                    "source_trial_number": 7,
                    "mean_qwk": 0.815,
                    "qwk_std": 0.005,
                    "mean_edges_after": 8,
                    "mean_symbolic_r2": 0.97,
                    "example_run": {
                        "experiment_name": "cand-b-seed-11",
                        "checkpoint_path": "checkpoints/cand-b.pt",
                        "summary_path": "artifacts/cand-b.json",
                        "metrics": {"qwk": 0.815},
                        "trainer_config": {"experiment_name": "cand-b-seed-11"},
                        "preprocessing_config": {"recipe": "kan_paper"},
                        "model_config": {
                            "name": "tabkan-base",
                            "flavor": "chebykan",
                            "hidden_widths": [32, 32],
                        },
                    },
                },
            }
        )
    )
    baseline_summary = tmp_path / "glm-summary.json"
    baseline_summary.write_text(
        json.dumps(
            {
                "metrics": {"qwk": 0.71},
                "checkpoint_path": "checkpoints/glm.joblib",
                "config": {
                    "trainer": {"experiment_name": "glm-baseline"},
                    "preprocessing": {"recipe": "kan_paper"},
                    "model": {"name": "glm", "params": {"alpha": 1.0}},
                },
            }
        )
    )

    result = run(
        [selection_manifest],
        output_root=tmp_path / "outputs",
        baseline_summary_paths=[baseline_summary],
    )

    assert len(result["models"]) == 3
    assert result["models"][0]["family"] == "chebykan"
    assert result["models"][-1]["family"] == "glm"
    assert (tmp_path / "outputs" / "final_comparison" / "final_comparison.json").exists()
    assert (tmp_path / "outputs" / "final_comparison" / "final_comparison.md").exists()
