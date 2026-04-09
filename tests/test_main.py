import json
from pathlib import Path

import main as main_module


def test_main_dispatches_interpret_stage_without_explicit_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_payload = {
        "trainer": {
            "experiment_name": "glm-baseline",
            "train_csv": "train.csv",
            "seed": 11,
        },
        "preprocessing": {
            "recipe": "kan_paper",
        },
        "model": {
            "name": "glm",
            "flavor": None,
            "depth": 1,
            "width": 1,
            "degree": None,
            "params": {"alpha": 1.0},
        },
    }

    captured: dict[str, object] = {}

    def fake_run_interpret(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {
            "experiment_name": config.trainer.experiment_name,
            "model": config.model.name,
            "recipe": config.preprocessing.recipe,
            "checkpoint_path": kwargs["checkpoint_path"],
            "eval_dir": kwargs["output_root"] / "eval",
            "output_dir": kwargs["output_root"] / "interpretability",
            "artifacts": {},
        }

    import src.interpretability.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_interpret", fake_run_interpret)

    checkpoint = tmp_path / "checkpoints" / "glm-baseline" / "model-20260406-000000.joblib"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")

    summary = tmp_path / "artifacts" / "glm-baseline" / "run-summary-20260406-000000.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"config": config_payload}))

    output_root = tmp_path / "outputs"
    main_module.run(
        [
            "--stage",
            "interpret",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--candidate-library",
            "scipy",
        ]
    )

    assert captured["config"].trainer.experiment_name == "glm-baseline"
    assert captured["checkpoint_path"] == checkpoint
    assert captured["output_root"] == output_root
    assert captured["candidate_library"] == "scipy"


def test_main_dispatches_retrain_stage(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_retrain(candidate_manifest, **kwargs):
        captured["candidate_manifest"] = candidate_manifest
        captured.update(kwargs)
        return {
            "family": "chebykan",
            "model_family": "chebykan",
            "selection_name": "shortlist",
            "runs": [],
            "manifest_path": str(tmp_path / "artifacts" / "retrain" / "manifest.json"),
        }

    import src.retrain as retrain_module

    monkeypatch.setattr(retrain_module, "run_retrain", fake_run_retrain)
    monkeypatch.setattr("configs.detect_device", lambda: "cpu")

    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text("{}")

    main_module.run(
        [
            "--stage",
            "retrain",
            "--candidate-manifest",
            str(candidate_manifest),
            "--seeds",
            "13",
            "29",
            "--candidate-id",
            "cand-a",
            "--selection-name",
            "shortlist",
            "--output-experiment-prefix",
            "retrain",
        ]
    )

    assert captured["candidate_manifest"] == candidate_manifest
    assert captured["device"] == "cpu"
    assert captured["seeds"] == [13, 29]
    assert captured["candidate_ids"] == ["cand-a"]
    assert captured["selection_name"] == "shortlist"
    assert captured["experiment_prefix"] == "retrain"


def test_main_dispatches_select_stage(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_select(retrain_manifest, **kwargs):
        captured["retrain_manifest"] = retrain_manifest
        captured.update(kwargs)
        return {
            "family": "chebykan",
            "model_family": "chebykan",
            "best_performance_candidate": {"candidate_id": "cand-a"},
            "best_interpretable_candidate": {"candidate_id": "cand-b"},
        }

    import src.selection as selection_module

    monkeypatch.setattr(selection_module, "run_select", fake_run_select)

    retrain_manifest = tmp_path / "manifest.json"
    retrain_manifest.write_text("{}")
    output_root = tmp_path / "outputs"

    main_module.run(
        [
            "--stage",
            "select",
            "--retrain-manifest",
            str(retrain_manifest),
            "--output-root",
            str(output_root),
            "--qwk-tolerance",
            "0.02",
        ]
    )

    assert captured["retrain_manifest"] == retrain_manifest
    assert captured["qwk_tolerance"] == 0.02
    assert captured["output_root"] == output_root
    assert captured["selection_output_root"] == Path("artifacts") / "selection"


def test_main_dispatches_retrain_stage(tmp_path, monkeypatch):
    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text("{}")

    captured: dict[str, object] = {}

    def fake_run_retrain(manifest_path, **kwargs):
        captured["manifest_path"] = manifest_path
        captured.update(kwargs)
        return {
            "model_family": "chebykan",
            "selection_name": kwargs["selection_name"],
            "runs": [],
            "manifest_path": "artifacts/retrain/chebykan/shortlist/manifest.json",
        }

    import src.retrain as retrain_module
    import src.config.runtime as runtime_module

    monkeypatch.setattr(retrain_module, "run_retrain", fake_run_retrain)
    monkeypatch.setattr(runtime_module, "detect_device", lambda: "cpu")

    main_module.run(
        [
            "--stage",
            "retrain",
            "--candidate-manifest",
            str(candidate_manifest),
            "--seeds",
            "13",
            "29",
            "--selection-name",
            "shortlist",
            "--output-experiment-prefix",
            "retrain-cheby",
            "--candidate-id",
            "cand-1",
            "--top-k",
            "2",
        ]
    )

    assert captured["manifest_path"] == candidate_manifest
    assert captured["device"] == "cpu"
    assert captured["seeds"] == [13, 29]
    assert captured["candidate_ids"] == ["cand-1"]
    assert captured["top_k"] == 2
    assert captured["selection_name"] == "shortlist"
    assert captured["experiment_prefix"] == "retrain-cheby"


def test_main_dispatches_select_stage(tmp_path, monkeypatch):
    retrain_manifest = tmp_path / "retrain-manifest.json"
    retrain_manifest.write_text("{}")

    captured: dict[str, object] = {}

    def fake_run_select(manifest_path, **kwargs):
        captured["manifest_path"] = manifest_path
        captured.update(kwargs)
        return {
            "family": "chebykan",
            "best_performance_candidate": {"candidate_id": "cand-a"},
            "best_interpretable_candidate": {"candidate_id": "cand-b"},
        }

    import src.selection as selection_module

    monkeypatch.setattr(selection_module, "run_select", fake_run_select)

    output_root = tmp_path / "outputs"
    main_module.run(
        [
            "--stage",
            "select",
            "--retrain-manifest",
            str(retrain_manifest),
            "--output-root",
            str(output_root),
            "--qwk-tolerance",
            "0.02",
        ]
    )

    assert captured["manifest_path"] == retrain_manifest
    assert captured["qwk_tolerance"] == 0.02
    assert captured["selection_output_root"] == main_module.Path("artifacts") / "selection"
    assert captured["output_root"] == output_root
