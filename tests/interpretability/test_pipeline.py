import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig, TrainerConfig
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


def test_run_interpret_surrogate_uses_full_model_input_when_features_are_restricted(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    eval_dir = output_root / "eval" / "kan_paper" / "tabkan-baseline"
    eval_dir.mkdir(parents=True, exist_ok=True)

    X_eval = pd.DataFrame(
        {
            "feat_a": [0.0, 1.0, 2.0],
            "feat_b": [1.0, 0.0, 1.0],
            "feat_c": [5.0, 5.0, 5.0],
        }
    )
    y_eval = pd.DataFrame({"Response": [1, 2, 3]})
    X_eval.to_parquet(eval_dir / "X_eval.parquet")
    y_eval.to_parquet(eval_dir / "y_eval.parquet")
    X_eval.to_parquet(eval_dir / "X_eval_raw.parquet")
    (eval_dir / "feature_names.json").write_text(json.dumps(list(X_eval.columns)))
    (eval_dir / "feature_types.json").write_text(json.dumps({name: "continuous" for name in X_eval.columns}))

    checkpoint_path = tmp_path / "checkpoints" / "tabkan-baseline" / "model-20260406-000000.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({}, checkpoint_path)

    config = ExperimentConfig(
        trainer=TrainerConfig(
            experiment_name="tabkan-baseline",
            train_csv=tmp_path / "train.csv",
            test_csv=None,
            seed=42,
        ),
        preprocessing=PreprocessingConfig(recipe="kan_paper"),
        model=ModelConfig(
            name="tabkan-base",
            flavor="chebykan",
            hidden_widths=(4, 2),
            degree=2,
            params={
                "max_epochs": 5,
                "lr": 0.001,
                "weight_decay": 0.0,
                "batch_size": 8,
                "sparsity_lambda": 0.0,
                "l1_weight": 1.0,
                "entropy_weight": 1.0,
            },
        ),
    )

    class _FakeTabKAN:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

        def state_dict(self):
            return self.loaded_state_dict or {}

        def eval(self):
            return self

    def fake_run_kan_pruning(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        flavor = args[2]
        report_dir = output_dir / "reports"
        model_dir = output_dir / "models"
        report_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{flavor}_pruning_summary.json").write_text("{}")
        torch.save({}, model_dir / f"{flavor}_pruned_module.pt")

    def fake_run_kan_symbolic(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        flavor = args[4]
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"layer": [], "r2": []}).to_csv(data_dir / f"{flavor}_symbolic_fits.csv", index=False)

    def fake_run_r2_pipeline(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        flavor = args[4]
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{flavor}_r2_report.json").write_text("{}")

    def fake_formula_run(*args, **kwargs):
        return {"exact_available": False}

    def fake_local_case_run(*args, output_dir: Path, flavor: str, **kwargs):
        report_dir = output_dir / "reports"
        data_dir = output_dir / "data"
        report_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        case_summary = report_dir / f"{flavor}_case_summary_stub.md"
        local_sensitivities = data_dir / f"{flavor}_local_sensitivities_stub.csv"
        what_if = data_dir / f"{flavor}_case_what_if_stub.csv"
        case_summary.write_text("stub\n")
        local_sensitivities.write_text("feature,delta\n")
        what_if.write_text("feature,scenario\n")
        return {
            "case_summary": case_summary,
            "local_sensitivities": local_sensitivities,
            "what_if": what_if,
        }

    class _SurrogateCalled(Exception):
        pass

    captured: dict[str, object] = {}
    fake_first_layer = type(
        "_FakeFirstLayer",
        (),
        {
            "cheby_coeffs": torch.ones((1, 3, 1), dtype=torch.float32),
            "base_weight": torch.ones((1, 3), dtype=torch.float32),
        },
    )()

    def fake_surrogate_run(module, X_eval_arg, *, feature_names, **kwargs):
        captured["columns"] = list(X_eval_arg.columns)
        captured["feature_names"] = list(feature_names)
        raise _SurrogateCalled

    import src.interpretability.closed_form_surrogate as closed_form_surrogate
    import src.interpretability.formula_composition as formula_composition
    import src.interpretability.kan_pruning as kan_pruning
    import src.interpretability.kan_symbolic as kan_symbolic
    import src.interpretability.local_case_explanations as local_case_explanations
    import src.interpretability.r2_pipeline as r2_pipeline
    import src.interpretability.utils.kan_coefficients as kan_coefficients
    import src.models.tabkan as tabkan_module

    monkeypatch.setattr(tabkan_module, "TabKAN", _FakeTabKAN)
    monkeypatch.setattr(kan_pruning, "run", fake_run_kan_pruning)
    monkeypatch.setattr(kan_symbolic, "run", fake_run_kan_symbolic)
    monkeypatch.setattr(r2_pipeline, "run", fake_run_r2_pipeline)
    monkeypatch.setattr(formula_composition, "run", fake_formula_run)
    monkeypatch.setattr(local_case_explanations, "run", fake_local_case_run)
    monkeypatch.setattr(closed_form_surrogate, "run", fake_surrogate_run)
    monkeypatch.setattr(
        kan_coefficients,
        "coefficient_importance_from_module",
        lambda module, feature_names: pd.Series(
            [3.0, 2.0, 1.0],
            index=["feat_a", "feat_b", "feat_c"],
            dtype=float,
        ),
    )
    monkeypatch.setattr(kan_coefficients, "get_first_kan_layer", lambda module: fake_first_layer)

    with pytest.raises(_SurrogateCalled):
        run_interpret(
            config,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            max_features=2,
        )

    assert captured["columns"] == ["feat_a", "feat_b", "feat_c"]
    assert captured["feature_names"] == ["feat_a", "feat_b"]
