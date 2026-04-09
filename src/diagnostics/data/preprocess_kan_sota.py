"""SOTA preprocessing pipeline for TabKAN models.

This module extends the paper-faithful baseline with state-of-the-art
imputation and encoding strategies inspired by:
- The Prudential Kaggle winning solutions (heavy use of target-aware
  encodings and quantile-based calibration)
- The *Analysis Accuracy of XGBoost Model...* paper for its deterministic
  splits and Product_Info_2 handling
- Exploratory insights captured in ``src/data/playground/data_insights.ipynb``

Compared to ``preprocess_kan_paper`` we apply the following upgrades:
1. **Iterative (MICE) imputation** for continuous/ordinal fields, providing
   smoother estimates than median fills while respecting feature covariance.
2. **CatBoost target encoding** for categorical/binary features (including
   ``Product_Info_2``) with built-in smoothing for unseen categories.
3. **Quantile transformation to [-1, 1]** on numerical columns to align with
   KAN spline grids and keep magnitudes bounded.
4. **Explicit missingness indicators** for columns with historical NaNs so the
   model can leverage absence patterns without leaking raw NaNs.

The core loading/encoding/splitting logic is inherited from
``PaperPreprocessingBase`` so this recipe remains fully aligned with the
published data handling, while surfacing tensor-ready features suitable for
advanced TabKAN experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from src.preprocessing.prudential_features import get_feature_lists
from src.preprocessing.preprocess_paper_base import PaperPreprocessingBase, PreprocessorState

TARGET_COLUMN = PaperPreprocessingBase.TARGET_COLUMN
ID_COLUMN = PaperPreprocessingBase.ID_COLUMN
PRODUCT_INFO_2 = PaperPreprocessingBase.PRODUCT_INFO_2
OUTER_TEST_SIZE = PaperPreprocessingBase.OUTER_TEST_SIZE
INNER_TEST_SIZE = PaperPreprocessingBase.INNER_TEST_SIZE
INNER_SPLITS = PaperPreprocessingBase.INNER_SPLITS


@dataclass
class KANSOTAPreprocessorState:
    """Holds encoders/imputers so transforms stay deterministic."""

    categorical_encoder: CatBoostEncoder | None
    categorical_scaler: MinMaxScaler | None
    numeric_imputer: IterativeImputer | None
    quantile_transformer: QuantileTransformer | None
    categorical_columns: List[str]
    numeric_columns: List[str]
    missing_indicator_columns: List[str]
    feature_names: List[str]


class KANSOTAPreprocessor(PaperPreprocessingBase):
    """Paper-faithful pipeline enriched with SOTA feature engineering."""

    def __init__(self, *, logger=None) -> None:
        super().__init__(logger=logger)
        self._feature_lists: Dict[str, List[str]] | None = None

    def run_pipeline(
        self,
        csv_path: str | Path,
        *,
        random_seed: int,
    ) -> Dict[str, object]:
        self.logger.info("Running SOTA KAN preprocessing (CatBoost + MICE + Quantile).")
        df = self.load_data(csv_path)
        self._feature_lists = get_feature_lists(df)
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

        sota_state = self.fit_sota_pipeline(X_train_outer, y_train_outer, random_seed=random_seed)
        X_train_arr = self._apply_sota_preprocessing(X_train_outer, sota_state)
        X_test_arr = self._apply_sota_preprocessing(X_test_outer, sota_state)
        y_train_arr = y_train_outer.to_numpy(dtype=np.float32, copy=True)
        y_test_arr = y_test_outer.to_numpy(dtype=np.float32, copy=True)
        self.logger.info(
            "Outer split (KAN-SOTA) tensors -> train %s / %s, test %s / %s",
            X_train_arr.shape,
            y_train_arr.shape,
            X_test_arr.shape,
            y_test_arr.shape,
        )

        inner_processed = self._process_inner_splits(
            inner_splits,
            random_seed=random_seed,
        )
        self.logger.info("KAN-SOTA feature names (%d): %s", len(sota_state.feature_names), sota_state.feature_names)

        return {
            "X_train_outer": X_train_arr,
            "X_test_outer": X_test_arr,
            "y_train_outer": y_train_arr,
            "y_test_outer": y_test_arr,
            "inner_splits": inner_processed,
            "feature_names": sota_state.feature_names,
            "preprocessor_state": {
                "baseline": base_state,
                "sota": sota_state,
            },
            "row_indices": {
                "outer_train": X_train_outer.index.copy(),
                "outer_test": X_test_outer.index.copy(),
            },
            "inner_split_indices": inner_split_indices,
        }

    def transform(
        self,
        df: pd.DataFrame,
        state: PreprocessorState,
        *,
        sota_state: KANSOTAPreprocessorState | None = None,
    ):
        X_base, y = super().transform(df, state)
        if sota_state is None:
            return X_base, y
        features = self._apply_sota_preprocessing(X_base, sota_state)
        target = None if y is None else y.to_numpy(dtype=np.float32, copy=True)
        if target is None:
            self.logger.info("KAN-SOTA transform produced features %s with no target present.", features.shape)
        else:
            self.logger.info("KAN-SOTA transform produced features %s and target %s.", features.shape, target.shape)
        return features, target

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    def fit_sota_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        random_seed: int,
    ) -> KANSOTAPreprocessorState:
        feature_lists = self._feature_lists or get_feature_lists(pd.concat([X_train, y_train], axis=1))
        cat_cols = self._resolve_categorical_columns(X_train, feature_lists)
        numeric_cols = self._resolve_numeric_columns(X_train, feature_lists)
        missing_cols = [col for col in numeric_cols if X_train[col].isna().any()]

        categorical_encoder = None
        categorical_scaler = None
        if cat_cols:
            categorical_encoder = CatBoostEncoder(
                cols=cat_cols,
                handle_unknown="impute",
                handle_missing="value",
                random_state=random_seed,
            )
            categorical_encoder.fit(X_train[cat_cols], y_train)
            encoded_view = categorical_encoder.transform(X_train[cat_cols]).to_numpy(dtype=np.float64)
            categorical_scaler = MinMaxScaler(feature_range=(-1, 1))
            categorical_scaler.fit(encoded_view)

        numeric_imputer = None
        quantile_transformer = None
        if numeric_cols:
            numeric_imputer = IterativeImputer(
                random_state=random_seed,
                sample_posterior=False,
                initial_strategy="median",
                max_iter=15,
            )
            numeric_imputer.fit(X_train[numeric_cols])
            imputed = numeric_imputer.transform(X_train[numeric_cols])
            quantile_transformer = QuantileTransformer(
                n_quantiles=min(1000, len(X_train)),
                output_distribution="uniform",
                random_state=random_seed,
                subsample=int(1e5),
            )
            quantile_transformer.fit(imputed)

        feature_names = self._build_feature_names(cat_cols, numeric_cols, missing_cols)
        return KANSOTAPreprocessorState(
            categorical_encoder=categorical_encoder,
            categorical_scaler=categorical_scaler,
            numeric_imputer=numeric_imputer,
            quantile_transformer=quantile_transformer,
            categorical_columns=cat_cols,
            numeric_columns=numeric_cols,
            missing_indicator_columns=missing_cols,
            feature_names=feature_names,
        )

    def _resolve_categorical_columns(
        self,
        X: pd.DataFrame,
        feature_lists: Dict[str, List[str]],
    ) -> List[str]:
        candidates = set(feature_lists["categorical"]) | set(feature_lists["binary"])
        candidates.add(self.PRODUCT_INFO_2)
        return [col for col in sorted(candidates) if col in X.columns]

    def _resolve_numeric_columns(
        self,
        X: pd.DataFrame,
        feature_lists: Dict[str, List[str]],
    ) -> List[str]:
        candidates = feature_lists["continuous"] + feature_lists["ordinal"]
        return [col for col in candidates if col in X.columns]

    def _build_feature_names(
        self,
        cat_cols: List[str],
        numeric_cols: List[str],
        missing_cols: List[str],
    ) -> List[str]:
        names: List[str] = []
        names.extend([f"cb_{col}" for col in cat_cols])
        names.extend([f"qt_{col}" for col in numeric_cols])
        names.extend([f"missing_{col}" for col in missing_cols])
        return names

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _apply_sota_preprocessing(
        self,
        X: pd.DataFrame,
        state: KANSOTAPreprocessorState,
    ) -> np.ndarray:
        frames: List[pd.DataFrame] = []
        if state.categorical_encoder and state.categorical_columns:
            cat_df = state.categorical_encoder.transform(X[state.categorical_columns])
            if state.categorical_scaler:
                scaled = state.categorical_scaler.transform(cat_df.values)
            else:
                scaled = cat_df.values
            cat_df = pd.DataFrame(
                scaled,
                columns=[f"cb_{col}" for col in state.categorical_columns],
                index=X.index,
            ).astype(np.float32)
            frames.append(cat_df)
        if state.numeric_imputer and state.quantile_transformer and state.numeric_columns:
            num_array = state.numeric_imputer.transform(X[state.numeric_columns])
            num_scaled = self._scale_to_unit_interval(num_array, state.quantile_transformer)
            num_df = pd.DataFrame(
                num_scaled,
                columns=[f"qt_{col}" for col in state.numeric_columns],
                index=X.index,
            )
            frames.append(num_df)
        if state.missing_indicator_columns:
            missing_payload = {
                f"missing_{col}": X[col].isna().astype(np.float32)
                for col in state.missing_indicator_columns
            }
            frames.append(pd.DataFrame(missing_payload, index=X.index))
        if not frames:
            raise RuntimeError("No features produced during SOTA preprocessing.")
        final_df = pd.concat(frames, axis=1)
        final_df = final_df[state.feature_names]
        return final_df.to_numpy(dtype=np.float32, copy=False)

    def _scale_to_unit_interval(
        self,
        numeric_array: np.ndarray,
        transformer: QuantileTransformer,
    ) -> np.ndarray:
        transformed = transformer.transform(numeric_array)
        return (transformed * 2.0) - 1.0

    def _process_inner_splits(
        self,
        inner_splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        *,
        random_seed: int,
    ):
        processed = []
        for idx, (X_tr, X_val, y_tr, y_val) in enumerate(inner_splits, start=1):
            local_state = self.fit_sota_pipeline(
                X_tr,
                y_tr,
                random_seed=random_seed + idx,
            )
            X_tr_arr = self._apply_sota_preprocessing(X_tr, local_state)
            X_val_arr = self._apply_sota_preprocessing(X_val, local_state)
            y_tr_arr = y_tr.to_numpy(dtype=np.float32, copy=True)
            y_val_arr = y_val.to_numpy(dtype=np.float32, copy=True)
            self.logger.info(
                "Inner split %d (KAN-SOTA) -> train %s / %s, val %s / %s",
                idx,
                X_tr_arr.shape,
                y_tr_arr.shape,
                X_val_arr.shape,
                y_val_arr.shape,
            )
            processed.append((X_tr_arr, X_val_arr, y_tr_arr, y_val_arr, local_state))
        return processed


def _build(logger=None) -> KANSOTAPreprocessor:
    return KANSOTAPreprocessor(logger=logger)


def load_data(csv_path: str | Path, *, logger=None) -> pd.DataFrame:
    return _build(logger).load_data(csv_path)


def fit_preprocessor(df: pd.DataFrame, *, logger=None) -> PreprocessorState:
    return _build(logger).fit_preprocessor(df)


def transform(
    df: pd.DataFrame,
    state: PreprocessorState,
    *,
    sota_state: KANSOTAPreprocessorState | None = None,
    logger=None,
):
    return _build(logger).transform(df, state, sota_state=sota_state)


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


def fit_sota_value_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int,
    logger=None,
) -> KANSOTAPreprocessorState:
    return _build(logger).fit_sota_pipeline(X_train, y_train, random_seed=random_seed)


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
        from src.config import set_global_seed

        seed = set_global_seed(42)
        results = run_pipeline(default_path, random_seed=seed)
        print("KAN-SOTA preprocessing complete -> outer tensors:", results["X_train_outer"].shape)
    else:
        print(
            "Set the Kaggle training CSV at data/prudential-life-insurance-assessment/train.csv",
        )
