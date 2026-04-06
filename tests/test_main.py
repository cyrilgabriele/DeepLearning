import json

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
