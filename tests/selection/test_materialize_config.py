import json

import yaml


def test_materialize_selected_config_writes_valid_yaml(tmp_path):
    from src.selection.materialize_config import materialize_selected_config

    selection_manifest = tmp_path / "chebykan_selection.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "best_interpretable_candidate": {
                    "example_run": {
                        "config": {
                            "trainer": {
                                "experiment_name": "stage-b-chebykan-cand-a-seed-13",
                                "train_csv": str(tmp_path / "train.csv"),
                                "test_csv": str(tmp_path / "test.csv"),
                                "seed": 13,
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
                                "params": {
                                    "max_epochs": 150,
                                    "sparsity_lambda": 0.001,
                                    "l1_weight": 1.0,
                                    "entropy_weight": 1.0,
                                },
                            },
                        }
                    }
                }
            }
        )
    )

    output_path = tmp_path / "materialized" / "chebykan.yaml"
    saved_path = materialize_selected_config(
        selection_manifest,
        role="best_interpretable_candidate",
        output_path=output_path,
    )

    assert saved_path == output_path
    payload = yaml.safe_load(output_path.read_text())
    assert payload["trainer"]["experiment_name"] == "stage-b-chebykan-cand-a-seed-13"
    assert payload["model"]["hidden_widths"] == [64, 64]
    assert payload["model"]["params"]["sparsity_lambda"] == 0.001
