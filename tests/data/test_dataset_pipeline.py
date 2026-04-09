import math
from pathlib import Path

import numpy as np
import pandas as pd

from configs import set_global_seed
from src.preprocessing import preprocess_xgboost_paper as paper_prep
from src.preprocessing import preprocess_kan_paper as kan_prep
from src.preprocessing import preprocess_kan_sota as kan_sota_prep


def _fixed_seed(value: int = 42) -> int:
    """Reset global RNGs and return the requested seed."""

    return set_global_seed(value)


def _write_mock_csv(tmp_path: Path, n_rows: int = 80) -> Path:
    rng = np.random.default_rng(123)
    repeats = math.ceil(n_rows / 8)
    response = np.tile(np.arange(1, 9), repeats)[:n_rows]
    df = pd.DataFrame(
        {
            "Id": np.arange(n_rows),
            "Response": response,
            "BMI": np.linspace(18, 40, num=n_rows),
            "Product_Info_2": rng.choice(list("ABCDEFG"), size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
            "Product_Info_3": rng.integers(1, 4, size=n_rows),
        }
    )
    path = tmp_path / "prudential.csv"
    df.to_csv(path, index=False)
    return path


def test_paper_pipeline_returns_reproducible_outer_split(tmp_path):
    csv_path = _write_mock_csv(tmp_path)
    seed = _fixed_seed()
    outputs = paper_prep.run_pipeline(csv_path, random_seed=seed)

    X_train = outputs["X_train_outer"]
    X_test = outputs["X_test_outer"]
    y_train = outputs["y_train_outer"]
    y_test = outputs["y_test_outer"]

    assert len(X_train) + len(X_test) == 80
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert paper_prep.PRODUCT_INFO_2 in X_train.columns

    mapping = outputs["preprocessor_state"].product_info_2_mapping
    assert mapping == dict(sorted(mapping.items()))  # deterministic ordering


def test_kan_pipeline_outputs_float32_features(tmp_path):
    csv_path = _write_mock_csv(tmp_path)
    seed = _fixed_seed()
    outputs = kan_prep.run_pipeline(csv_path, random_seed=seed)

    X_train = outputs["X_train_outer"]
    X_test = outputs["X_test_outer"]
    feature_names = outputs["feature_names"]

    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert len(feature_names) == X_train.shape[1]

    artifacts = outputs["preprocessor_state"]
    assert {"baseline", "kan"}.issubset(artifacts.keys())


def test_transform_supports_inference_without_target(tmp_path):
    csv_path = _write_mock_csv(tmp_path)
    df = pd.read_csv(csv_path)
    baseline_state = paper_prep.fit_preprocessor(df)

    # Paper pipeline inference
    paper_features, paper_target = paper_prep.transform(df.drop(columns=["Response"]), baseline_state)
    assert paper_target is None
    assert paper_prep.PRODUCT_INFO_2 in paper_features.columns

    # KAN inference using fitted states from the training pipeline
    seed = _fixed_seed()
    pipeline_outputs = kan_prep.run_pipeline(csv_path, random_seed=seed)
    seed = _fixed_seed()
    artifacts = pipeline_outputs["preprocessor_state"]
    kan_features, kan_target = kan_prep.transform(
        df.drop(columns=["Response"]),
        artifacts["baseline"],
        kan_state=artifacts["kan"],
    )
    assert kan_target is None
    assert kan_features.dtype == np.float32


def test_kan_sota_pipeline_applies_advanced_encoding(tmp_path):
    csv_path = _write_mock_csv(tmp_path)
    seed = _fixed_seed()
    outputs = kan_sota_prep.run_pipeline(csv_path, random_seed=seed)

    X_train = outputs["X_train_outer"]
    X_test = outputs["X_test_outer"]
    feature_names = outputs["feature_names"]

    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert np.all(X_train <= 1.0001) and np.all(X_train >= -1.0001)
    assert len(feature_names) == X_train.shape[1]

    artifacts = outputs["preprocessor_state"]
    assert {"baseline", "sota"}.issubset(artifacts.keys())


def test_kan_sota_drops_ultra_sparse_value_channels_and_keeps_masks(tmp_path):
    rng = np.random.default_rng(99)
    n_rows = 80
    repeats = math.ceil(n_rows / 8)
    response = np.tile(np.arange(1, 9), repeats)[:n_rows]
    df = pd.DataFrame(
        {
            "Id": np.arange(n_rows),
            "Response": response,
            "BMI": rng.normal(27, 4, size=n_rows),
            "Product_Info_2": rng.choice(["A1", "B2", "C3"], size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
            "Product_Info_3": rng.integers(1, 12, size=n_rows),
            "Employment_Info_2": rng.integers(1, 40, size=n_rows),
            "Employment_Info_4": np.where(
                rng.random(n_rows) < 0.2,
                np.nan,
                rng.uniform(0.0, 1.0, size=n_rows),
            ),
            "Family_Hist_3": np.where(
                rng.random(n_rows) < 0.7,
                np.nan,
                rng.uniform(0.0, 1.0, size=n_rows),
            ),
        }
    )
    csv_path = tmp_path / "prudential_sparse.csv"
    df.to_csv(csv_path, index=False)

    seed = _fixed_seed()
    outputs = kan_sota_prep.run_pipeline(csv_path, random_seed=seed)
    feature_names = outputs["feature_names"]
    sota_state = outputs["preprocessor_state"]["sota"]

    assert "Family_Hist_3" in sota_state.dropped_value_columns
    assert "missing_Family_Hist_3" in feature_names
    assert "qt_Family_Hist_3" not in feature_names

    assert "qt_Employment_Info_4" in feature_names
    assert "missing_Employment_Info_4" in feature_names

    assert "mm_Product_Info_3" in feature_names
    assert "mm_Employment_Info_2" in feature_names

    X_train = outputs["X_train_outer"]
    assert X_train.dtype == np.float32
    assert np.isfinite(X_train).all()
    assert np.all(X_train <= 1.0001) and np.all(X_train >= -1.0001)
