"""Trainer dispatch for the tabpfn_paper recipe."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def tiny_train_csv(tmp_path):
    csv = tmp_path / "train.csv"
    rows = []
    for i in range(120):
        rows.append({
            "Id": i + 1,
            "Product_Info_2": "A1",
            "Product_Info_4": 0.5,
            "BMI": 0.4 + (i % 5) * 0.05,
            "Ins_Age": 0.3,
            "Medical_History_15": float(i % 3),
            "Response": (i % 8) + 1,
        })
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_trainer_prepare_data_supports_tabpfn_paper_recipe(tiny_train_csv, tmp_path):
    from src.config import ExperimentConfig, load_experiment_config
    from src.training.trainer import Trainer

    cfg_text = f"""
trainer:
  experiment_name: stage-c-tabpfn-test
  train_csv: {tiny_train_csv}
  seed: 42
preprocessing:
  contract_version: 1
  recipe: tabpfn_paper
model:
  name: tabpfn-auto
  params:
    n_estimators: 1
    max_time: 30
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text)
    config = load_experiment_config(cfg_path)

    trainer = Trainer(config=config, device="cpu", random_seed=42)
    dataset = trainer._prepare_data()

    assert dataset.recipe == "tabpfn_paper"
    assert dataset.X_train is not None
    assert dataset.y_train is not None
    assert len(dataset.X_train) > 0
