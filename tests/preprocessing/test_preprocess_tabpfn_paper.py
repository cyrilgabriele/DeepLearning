"""Unit tests for the tabpfn_paper recipe shim."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def tiny_csv(tmp_path):
    csv = tmp_path / "train.csv"
    rows = []
    for i in range(64):
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


def test_tabpfn_paper_recipe_run_pipeline_returns_xgb_paper_shape(tiny_csv):
    from src.preprocessing import preprocess_tabpfn_paper as tabpfn_prep
    from src.preprocessing import preprocess_xgboost_paper as xgb_prep

    out_tabpfn = tabpfn_prep.run_pipeline(tiny_csv, random_seed=42)
    out_xgb = xgb_prep.run_pipeline(tiny_csv, random_seed=42)

    assert set(out_tabpfn.keys()) == set(out_xgb.keys())
    assert out_tabpfn["X_train_outer"].shape == out_xgb["X_train_outer"].shape
    assert out_tabpfn["X_test_outer"].shape == out_xgb["X_test_outer"].shape
    assert list(out_tabpfn["X_train_outer"].columns) == list(out_xgb["X_train_outer"].columns)


def test_tabpfn_paper_transform_matches_xgb_paper(tiny_csv):
    from src.preprocessing import preprocess_tabpfn_paper as tabpfn_prep
    from src.preprocessing import preprocess_xgboost_paper as xgb_prep

    out_xgb = xgb_prep.run_pipeline(tiny_csv, random_seed=42)
    state = out_xgb["preprocessor_state"]

    df = pd.read_csv(tiny_csv)
    out_tabpfn_transformed, _ = tabpfn_prep.transform(df, state)
    out_xgb_transformed, _ = xgb_prep.transform(df, state)

    pd.testing.assert_frame_equal(
        out_tabpfn_transformed.reset_index(drop=True),
        out_xgb_transformed.reset_index(drop=True),
    )
