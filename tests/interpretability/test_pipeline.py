import json
from pathlib import Path

import pytest

from configs import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
from src.interpretability.pipeline import resolve_interpret_config, run_interpret


def test_run_interpret_glm_uses_namespaced_eval_artifacts(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    eval_dir = output_root / "eval" / "kan_paper" / "glm-baseline"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "feature_names.json").write_text(json.dumps(["feat_a", "feat_b"]))

    checkpoint_path = tmp_path / "checkpoints" / "glm-baseline" / "model-20260406-000000.joblib"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("placeholder")

    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="glm-baseline",
            train_csv=tmp_path / "train.csv",
            test_csv=None,
            seed=42,
        ),
        preprocessing=PreprocessingConfig(recipe="kan_paper"),
        model=ModelConfig(
            name="glm",
            flavor=None,
            depth=1,
            width=1,
            degree=None,
            params={"alpha": 1.0},
        ),
    )

    captured: dict[str, Path] = {}

    def fake_run(checkpoint_arg: Path, features_arg: Path, output_dir: Path = Path("outputs")) -> Path:
        captured["checkpoint"] = checkpoint_arg
        captured["features"] = features_arg
        captured["output_dir"] = output_dir
        out_path = output_dir / "data" / "glm_coefficients.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("feature,coefficient,abs_magnitude\n")
        return out_path

    import src.interpretability.glm_coefficients as glm_coefficients

    monkeypatch.setattr(glm_coefficients, "run", fake_run)

    result = run_interpret(
        config,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
    )

    assert captured["checkpoint"] == checkpoint_path
    assert captured["features"] == eval_dir / "feature_names.json"
    assert captured["output_dir"] == output_root / "interpretability" / "kan_paper" / "glm-baseline"
    assert result["eval_dir"] == eval_dir
    assert result["output_dir"] == output_root / "interpretability" / "kan_paper" / "glm-baseline"


def test_resolve_interpret_config_uses_checkpoint_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checkpoint_path = tmp_path / "checkpoints" / "glm-baseline" / "model-20260406-000000.joblib"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("placeholder")

    summary_path = tmp_path / "artifacts" / "glm-baseline" / "run-summary-20260406-000000.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "trainer": {
                        "experiment_name": "glm-baseline",
                        "train_csv": str(tmp_path / "train.csv"),
                        "seed": 42,
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
            }
        )
    )

    config = resolve_interpret_config(config_path=None, checkpoint_path=checkpoint_path)

    assert config.trainer.experiment_name == "glm-baseline"
    assert config.model.name == "glm"


def test_resolve_interpret_config_rejects_mismatched_explicit_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    checkpoint_path = tmp_path / "checkpoints" / "glm-baseline" / "model-20260406-000000.joblib"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("placeholder")

    summary_path = tmp_path / "artifacts" / "glm-baseline" / "run-summary-20260406-000000.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "trainer": {
                        "experiment_name": "glm-baseline",
                        "train_csv": str(tmp_path / "train.csv"),
                        "seed": 42,
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
            }
        )
    )

    config_path = tmp_path / "glm.yaml"
    config_path.write_text(
        """
trainer:
  experiment_name: different-experiment
  train_csv: train.csv
  seed: 11
preprocessing:
  recipe: kan_paper
model:
  name: glm
  flavor: null
  depth: 1
  width: 1
  degree: null
  params:
    alpha: 1.0
""".strip()
    )

    with pytest.raises(ValueError, match="does not match"):
        resolve_interpret_config(config_path=config_path, checkpoint_path=checkpoint_path)
