"""TabPFN preprocessing recipe.

Identical preprocessing to ``preprocess_xgboost_paper`` (numeric encoding +
NaN imputation + outer/inner splits, no scaling). TabPFN does its own
internal normalization, so the same numeric-only pipeline is reused as a
thin shim. The separate module exists so artifacts and manifests are tagged
with the ``tabpfn_paper`` recipe rather than ``xgboost_paper``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.preprocessing import preprocess_xgboost_paper as _xgb


TARGET_COLUMN = _xgb.TARGET_COLUMN
ID_COLUMN = _xgb.ID_COLUMN


def load_data(csv_path: str | Path, *, logger=None) -> pd.DataFrame:
    return _xgb.load_data(csv_path, logger=logger)


def fit_preprocessor(df: pd.DataFrame, *, logger=None):
    return _xgb.fit_preprocessor(df, logger=logger)


def transform(df: pd.DataFrame, state, *, logger=None):
    return _xgb.transform(df, state, logger=logger)


def make_outer_split(X, y, *, test_size=None, random_state, logger=None):
    if test_size is None:
        return _xgb.make_outer_split(X, y, random_state=random_state, logger=logger)
    return _xgb.make_outer_split(
        X, y, test_size=test_size, random_state=random_state, logger=logger,
    )


def make_inner_splits(X_train_outer, y_train_outer, *, n_splits=None, test_size=None, base_random_seed, logger=None):
    kwargs = {"base_random_seed": base_random_seed, "logger": logger}
    if n_splits is not None:
        kwargs["n_splits"] = n_splits
    if test_size is not None:
        kwargs["test_size"] = test_size
    return _xgb.make_inner_splits(X_train_outer, y_train_outer, **kwargs)


def run_pipeline(csv_path: str | Path, *, random_seed: int, logger=None) -> Dict[str, object]:
    return _xgb.run_pipeline(csv_path, random_seed=random_seed, logger=logger)
