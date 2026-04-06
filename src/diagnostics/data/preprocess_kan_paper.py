"""KAN-ready preprocessing built on top of the paper-faithful baseline.

The pipeline subclasses :class:`PaperPreprocessingBase` so both the
XGBoost and KAN recipes share an identical loading/encoding/splitting
routine. KAN introduces only the minimal adjustments required to make the
features tensor-friendly: median imputation with missing indicators, an
explicit ``Product_Info_2`` missing token, and conversion to ``float32``
arrays. All other behaviour (column handling, split seeds, logging) is
identical to the paper baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.preprocess_paper_base import PaperPreprocessingBase, PreprocessorState

TARGET_COLUMN = PaperPreprocessingBase.TARGET_COLUMN
ID_COLUMN = PaperPreprocessingBase.ID_COLUMN
PRODUCT_INFO_2 = PaperPreprocessingBase.PRODUCT_INFO_2
OUTER_TEST_SIZE = PaperPreprocessingBase.OUTER_TEST_SIZE
INNER_TEST_SIZE = PaperPreprocessingBase.INNER_TEST_SIZE
INNER_SPLITS = PaperPreprocessingBase.INNER_SPLITS


@dataclass
class KANPreprocessorState:
    """Holds the imputer state needed to emit NaN-free float32 tensors."""

    numeric_imputer: SimpleImputer
    numeric_columns: List[str]
    numeric_indicator_names: List[str]
    product_missing_code: float
    feature_names: List[str]


class KANPreprocessor(PaperPreprocessingBase):
    """Paper-faithful pipeline with minimal tensor-safety tweaks for TabKAN."""

    def transform(
        self,
        df: pd.DataFrame,
        state: PreprocessorState,
        *,
        kan_state: KANPreprocessorState | None = None,
        **_,
    ):
        X_base, y = super().transform(df, state)
        if kan_state is None:
            return X_base, y

        X_processed = self._apply_kan_preprocessing(X_base, kan_state)
        y_array = None if y is None else y.to_numpy(dtype=np.float32, copy=True)
        if y_array is None:
            self.logger.info(
                "KAN tensor-friendly features shape: %s | no target column present.",
                X_processed.shape,
            )
        else:
            self.logger.info(
                "KAN tensor-friendly features shape: %s | target tensor shape: %s",
                X_processed.shape,
                y_array.shape,
            )
        return X_processed, y_array

    def fit_kan_value_pipeline(
        self,
        X_train: pd.DataFrame,
        state: PreprocessorState,
    ) -> KANPreprocessorState:
        numeric_columns = [col for col in X_train.columns if col != self.PRODUCT_INFO_2]
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        imputer.fit(X_train[numeric_columns])

        indicator_indices = getattr(imputer, "indicator_", None)
        indicator_names: List[str] = []
        if indicator_indices is not None:
            indicator_names = [f"missing_{numeric_columns[idx]}" for idx in indicator_indices.features_]

        product_missing_code = float(len(state.product_info_2_mapping))
        feature_names = numeric_columns + indicator_names + [self.PRODUCT_INFO_2, "missing_Product_Info_2"]

        self.logger.info(
            "KAN post-processor fitted -> %d numeric columns, %d missing indicators, missing Product_Info_2 code %s",
            len(numeric_columns),
            len(indicator_names) + 1,
            product_missing_code,
        )
        return KANPreprocessorState(
            numeric_imputer=imputer,
            numeric_columns=numeric_columns,
            numeric_indicator_names=indicator_names,
            product_missing_code=product_missing_code,
            feature_names=feature_names,
        )

    def run_pipeline(
        self,
        csv_path: str | Path,
        *,
        random_seed: int,
    ) -> Dict[str, object]:
        self.logger.info(
            "Running KAN preprocessing (baseline + minimal tensor-safety tweaks).",
        )
        df = self.load_data(csv_path)
        base_state = self.fit_preprocessor(df)
        X_base, y = super().transform(df, base_state)
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = self.make_outer_split(
            X_base,
            y,
            random_state=random_seed,
        )
        inner_splits = self.make_inner_splits(
            X_train_outer,
            y_train_outer,
            base_random_seed=random_seed,
        )
        inner_split_indices = [
            {
                "train": X_tr.index.copy(),
                "val": X_val.index.copy(),
            }
            for (X_tr, X_val, _, _) in inner_splits
        ]

        kan_state_outer = self.fit_kan_value_pipeline(X_train_outer, base_state)
        X_train_arr = self._apply_kan_preprocessing(X_train_outer, kan_state_outer)
        X_test_arr = self._apply_kan_preprocessing(X_test_outer, kan_state_outer)
        y_train_arr = y_train_outer.to_numpy(dtype=np.float32, copy=True)
        y_test_arr = y_test_outer.to_numpy(dtype=np.float32, copy=True)
        self.logger.info(
            "Outer split (KAN) tensors -> train %s / %s, test %s / %s",
            X_train_arr.shape,
            y_train_arr.shape,
            X_test_arr.shape,
            y_test_arr.shape,
        )

        inner_processed = self._process_inner_splits(inner_splits, base_state, random_seed=random_seed)
        self.logger.info(
            "KAN feature names (%d): %s",
            len(kan_state_outer.feature_names),
            kan_state_outer.feature_names,
        )

        return {
            "X_train_outer": X_train_arr,
            "X_test_outer": X_test_arr,
            "y_train_outer": y_train_arr,
            "y_test_outer": y_test_arr,
            "inner_splits": inner_processed,
            "feature_names": kan_state_outer.feature_names,
            "preprocessor_state": {
                "baseline": base_state,
                "kan": kan_state_outer,
            },
            "row_indices": {
                "outer_train": X_train_outer.index.copy(),
                "outer_test": X_test_outer.index.copy(),
            },
            "inner_split_indices": inner_split_indices,
        }

    # ---- Internal helpers -------------------------------------------------

    def _apply_kan_preprocessing(
        self,
        X: pd.DataFrame,
        kan_state: KANPreprocessorState,
    ) -> np.ndarray:
        numeric_array = kan_state.numeric_imputer.transform(X[kan_state.numeric_columns])
        n_numeric = len(kan_state.numeric_columns)
        numeric_values = numeric_array[:, :n_numeric]
        indicator_values = numeric_array[:, n_numeric:]

        frames = [
            pd.DataFrame(numeric_values, columns=kan_state.numeric_columns, index=X.index),
        ]
        if kan_state.numeric_indicator_names:
            frames.append(
                pd.DataFrame(
                    indicator_values,
                    columns=kan_state.numeric_indicator_names,
                    index=X.index,
                )
            )

        product_series = X[self.PRODUCT_INFO_2].copy()
        product_missing = product_series.isna().astype(np.float32)
        product_filled = product_series.fillna(kan_state.product_missing_code).astype(np.float32)
        frames.append(
            pd.DataFrame(
                {
                    self.PRODUCT_INFO_2: product_filled,
                    "missing_Product_Info_2": product_missing,
                },
                index=X.index,
            )
        )

        final_df = pd.concat(frames, axis=1)
        final_df = final_df[kan_state.feature_names]
        final_array = final_df.to_numpy(dtype=np.float32, copy=False)
        self.logger.debug("KAN preprocessing produced finite array with shape %s", final_array.shape)
        return final_array

    def _process_inner_splits(
        self,
        inner_splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        base_state: PreprocessorState,
        *,
        random_seed: int,
    ):
        processed = []
        for idx, (X_tr, X_val, y_tr, y_val) in enumerate(inner_splits, start=1):
            kan_state = self.fit_kan_value_pipeline(X_tr, base_state)
            X_tr_arr = self._apply_kan_preprocessing(X_tr, kan_state)
            X_val_arr = self._apply_kan_preprocessing(X_val, kan_state)
            y_tr_arr = y_tr.to_numpy(dtype=np.float32, copy=True)
            y_val_arr = y_val.to_numpy(dtype=np.float32, copy=True)
            self.logger.info(
                "Inner split %d (KAN) -> train %s / %s, val %s / %s",
                idx,
                X_tr_arr.shape,
                y_tr_arr.shape,
                X_val_arr.shape,
                y_val_arr.shape,
            )
            processed.append((X_tr_arr, X_val_arr, y_tr_arr, y_val_arr, kan_state))
        return processed


def _build(logger=None) -> KANPreprocessor:
    return KANPreprocessor(logger=logger)


def load_data(csv_path: str | Path, *, logger=None) -> pd.DataFrame:
    return _build(logger).load_data(csv_path)


def fit_preprocessor(df: pd.DataFrame, *, logger=None) -> PreprocessorState:
    return _build(logger).fit_preprocessor(df)


def transform(
    df: pd.DataFrame,
    state: PreprocessorState,
    *,
    kan_state: KANPreprocessorState | None = None,
    logger=None,
):
    return _build(logger).transform(df, state, kan_state=kan_state)


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


def fit_kan_value_pipeline(
    X_train: pd.DataFrame,
    state: PreprocessorState,
    *,
    logger=None,
) -> KANPreprocessorState:
    return _build(logger).fit_kan_value_pipeline(X_train, state)


def run_pipeline(
    csv_path: str | Path,
    *,
    random_seed: int,
    logger=None,
) -> Dict[str, object]:
    return _build(logger).run_pipeline(csv_path, random_seed=random_seed)


if __name__ == "__main__":
    default_path = Path("data/prudential-life-insurance-assessment/train.csv")
    if default_path.exists():
        from configs import set_global_seed

        seed = set_global_seed(42)
        results = run_pipeline(default_path, random_seed=seed)
        print("KAN preprocessing complete -> outer tensors:", results["X_train_outer"].shape)
    else:
        print(
            "Set the Kaggle training CSV at data/prudential-life-insurance-assessment/train.csv",
        )
