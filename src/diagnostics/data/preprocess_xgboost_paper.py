"""Paper-faithful preprocessing for the Prudential Life Insurance dataset.

Derived strictly from *Analysis Accuracy of XGBoost Model for Multiclass
Classification - A Case Study of Applicant Level Risk Prediction for Life
Insurance* (ICSITech 2019). The pipeline mirrors the paper exactly and adds
an initial log to make it explicit when the XGBoost-specific preprocessing
is invoked."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data.preprocess_paper_base import PaperPreprocessingBase, PreprocessorState

TARGET_COLUMN = PaperPreprocessingBase.TARGET_COLUMN
ID_COLUMN = PaperPreprocessingBase.ID_COLUMN
PRODUCT_INFO_2 = PaperPreprocessingBase.PRODUCT_INFO_2
OUTER_TEST_SIZE = PaperPreprocessingBase.OUTER_TEST_SIZE
INNER_TEST_SIZE = PaperPreprocessingBase.INNER_TEST_SIZE
INNER_SPLITS = PaperPreprocessingBase.INNER_SPLITS


class XGBoostPaperPreprocessor(PaperPreprocessingBase):
    """Thin wrapper around the shared paper-baseline preprocessing."""


def _build(logger=None) -> XGBoostPaperPreprocessor:
    return XGBoostPaperPreprocessor(logger=logger)


def load_data(csv_path: str | Path, *, logger=None) -> pd.DataFrame:
    return _build(logger).load_data(csv_path)


def fit_preprocessor(df: pd.DataFrame, *, logger=None) -> PreprocessorState:
    return _build(logger).fit_preprocessor(df)


def transform(
    df: pd.DataFrame,
    state: PreprocessorState,
    *,
    logger=None,
):
    return _build(logger).transform(df, state)


def make_outer_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = OUTER_TEST_SIZE,
    random_state: int,
    logger=None,
):
    return _build(logger).make_outer_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def make_inner_splits(
    X_train_outer: pd.DataFrame,
    y_train_outer: pd.Series,
    *,
    n_splits: int = INNER_SPLITS,
    test_size: float = INNER_TEST_SIZE,
    base_random_seed: int,
    logger=None,
):
    return _build(logger).make_inner_splits(
        X_train_outer,
        y_train_outer,
        n_splits=n_splits,
        test_size=test_size,
        base_random_seed=base_random_seed,
    )


def run_pipeline(
    csv_path: str | Path,
    *,
    random_seed: int,
    logger=None,
) -> Dict[str, object]:
    preprocessor = _build(logger)
    preprocessor.logger.info("Running XGBoost paper-faithful preprocessing pipeline.")
    return preprocessor.run_pipeline(csv_path, random_seed=random_seed)


if __name__ == "__main__":
    default_path = Path("data/prudential-life-insurance-assessment/train.csv")
    if default_path.exists():
        from src.configs import set_global_seed

        seed = set_global_seed(42)
        results = run_pipeline(default_path, random_seed=seed)
        print("Paper-preprocessing complete.")
        print(
            "Outer train/test shapes:",
            results["X_train_outer"].shape,
            results["X_test_outer"].shape,
        )
    else:
        print(
            "Set the Kaggle training CSV at data/prudential-life-insurance-assessment/train.csv",
        )
