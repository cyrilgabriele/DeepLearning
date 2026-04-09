"""SOTA preprocessing pipeline for TabKAN models.

This module extends the paper-faithful baseline with state-of-the-art
imputation and encoding strategies inspired by:
- The Prudential Kaggle winning solutions (heavy use of target-aware
  encodings and quantile-based calibration)
- The *Analysis Accuracy of XGBoost Model...* paper for its deterministic
  splits and Product_Info_2 handling
- Exploratory insights captured in ``src/data/playground/data_insights.ipynb``

Compared to ``preprocess_kan_paper`` we apply the following upgrades:
1. **Leakage-safe CatBoost target encoding** for categorical/binary features
   (including ``Product_Info_2``) using out-of-fold encodings on training rows.
2. **Missingness-aware value dropping** for ultra-sparse numeric columns
   (>50% missing by default): drop their value channels but keep their masks.
3. **Iterative (MICE) imputation + quantile scaling** for retained continuous
   features to map them to a bounded ``[-1, 1]`` interval.
4. **Median imputation + min-max scaling** for ordinal features, because the
   project notes flagged quantile scaling on low-cardinality ordinals as too
   staircase-like for KAN spline layers.
5. **Explicit missingness indicators** for every numeric column that ever has
   NaNs so the model can learn informative absence patterns directly.

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
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from src.preprocessing.prudential_features import get_feature_lists
from src.preprocessing.preprocess_paper_base import PaperPreprocessingBase, PreprocessorState

TARGET_COLUMN = PaperPreprocessingBase.TARGET_COLUMN
ID_COLUMN = PaperPreprocessingBase.ID_COLUMN
PRODUCT_INFO_2 = PaperPreprocessingBase.PRODUCT_INFO_2
OUTER_TEST_SIZE = PaperPreprocessingBase.OUTER_TEST_SIZE
INNER_TEST_SIZE = PaperPreprocessingBase.INNER_TEST_SIZE
INNER_SPLITS = PaperPreprocessingBase.INNER_SPLITS
VALUE_DROP_MISSING_THRESHOLD = 0.5
TARGET_ENCODING_FOLDS = 5


@dataclass
class KANSOTAPreprocessorState:
    """Holds encoders/imputers so transforms stay deterministic."""

    categorical_encoder: CatBoostEncoder | None
    categorical_scaler: MinMaxScaler | None
    categorical_columns: List[str]
    categorical_oof_frame: pd.DataFrame | None
    continuous_imputer: IterativeImputer | None
    continuous_transformer: QuantileTransformer | None
    continuous_columns: List[str]
    ordinal_imputer: SimpleImputer | None
    ordinal_scaler: MinMaxScaler | None
    ordinal_columns: List[str]
    dropped_value_columns: List[str]
    missing_indicator_columns: List[str]
    feature_names: List[str]


class KANSOTAPreprocessor(PaperPreprocessingBase):
    """Paper-faithful pipeline enriched with SOTA feature engineering."""

    def __init__(
        self,
        *,
        logger=None,
        missing_threshold: float = VALUE_DROP_MISSING_THRESHOLD,
        target_encoding_folds: int = TARGET_ENCODING_FOLDS,
    ) -> None:
        super().__init__(logger=logger)
        if not 0.0 <= missing_threshold <= 1.0:
            raise ValueError("missing_threshold must be between 0.0 and 1.0.")
        if target_encoding_folds < 2:
            raise ValueError("target_encoding_folds must be at least 2.")
        self._feature_lists: Dict[str, List[str]] | None = None
        self.missing_threshold = missing_threshold
        self.target_encoding_folds = target_encoding_folds

    def run_pipeline(
        self,
        csv_path: str | Path,
        *,
        random_seed: int,
    ) -> Dict[str, object]:
        self.logger.info(
            "Running SOTA KAN preprocessing "
            "(OOF CatBoost + sparse-column dropping + MICE/Quantile + ordinal MinMax)."
        )
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
        all_numeric_cols = self._resolve_numeric_columns(X_train, feature_lists)
        dropped_value_cols = self._resolve_dropped_value_columns(X_train, all_numeric_cols)
        continuous_cols = self._resolve_continuous_columns(X_train, feature_lists, dropped_value_cols)
        ordinal_cols = self._resolve_ordinal_columns(X_train, feature_lists, dropped_value_cols)
        missing_cols = [col for col in all_numeric_cols if X_train[col].isna().any()]

        categorical_encoder = None
        categorical_scaler = None
        categorical_oof_frame = None
        if cat_cols:
            categorical_encoder, categorical_oof_frame = self._fit_categorical_encoder(
                X_train[cat_cols],
                y_train,
                random_seed=random_seed,
            )
            categorical_scaler = MinMaxScaler(feature_range=(-1, 1))
            categorical_scaler.fit(categorical_oof_frame.to_numpy(dtype=np.float64))

        continuous_imputer = None
        continuous_transformer = None
        if continuous_cols:
            continuous_imputer = IterativeImputer(
                random_state=random_seed,
                sample_posterior=False,
                initial_strategy="median",
                max_iter=15,
            )
            continuous_imputer.fit(X_train[continuous_cols])
            imputed = continuous_imputer.transform(X_train[continuous_cols])
            continuous_transformer = QuantileTransformer(
                n_quantiles=min(1000, len(X_train)),
                output_distribution="uniform",
                random_state=random_seed,
                subsample=int(1e5),
            )
            continuous_transformer.fit(imputed)

        ordinal_imputer = None
        ordinal_scaler = None
        if ordinal_cols:
            ordinal_imputer = SimpleImputer(strategy="median")
            ordinal_imputer.fit(X_train[ordinal_cols])
            ordinal_values = ordinal_imputer.transform(X_train[ordinal_cols])
            ordinal_scaler = MinMaxScaler(feature_range=(-1, 1))
            ordinal_scaler.fit(ordinal_values)

        feature_names = self._build_feature_names(
            cat_cols,
            continuous_cols,
            ordinal_cols,
            missing_cols,
        )
        self.logger.info(
            "KAN-SOTA fitted -> %d encoded categorical/binary, %d continuous, %d ordinal, "
            "%d missing indicators, dropped value columns (>%.0f%% missing): %s",
            len(cat_cols),
            len(continuous_cols),
            len(ordinal_cols),
            len(missing_cols),
            self.missing_threshold * 100.0,
            dropped_value_cols,
        )
        return KANSOTAPreprocessorState(
            categorical_encoder=categorical_encoder,
            categorical_scaler=categorical_scaler,
            categorical_columns=cat_cols,
            categorical_oof_frame=categorical_oof_frame,
            continuous_imputer=continuous_imputer,
            continuous_transformer=continuous_transformer,
            continuous_columns=continuous_cols,
            ordinal_imputer=ordinal_imputer,
            ordinal_scaler=ordinal_scaler,
            ordinal_columns=ordinal_cols,
            dropped_value_columns=dropped_value_cols,
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

    def _resolve_continuous_columns(
        self,
        X: pd.DataFrame,
        feature_lists: Dict[str, List[str]],
        dropped_value_columns: List[str],
    ) -> List[str]:
        return [
            col
            for col in feature_lists["continuous"]
            if col in X.columns and col not in dropped_value_columns
        ]

    def _resolve_ordinal_columns(
        self,
        X: pd.DataFrame,
        feature_lists: Dict[str, List[str]],
        dropped_value_columns: List[str],
    ) -> List[str]:
        return [
            col
            for col in feature_lists["ordinal"]
            if col in X.columns and col not in dropped_value_columns
        ]

    def _resolve_dropped_value_columns(
        self,
        X: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[str]:
        return [
            col
            for col in numeric_cols
            if float(X[col].isna().mean()) > self.missing_threshold
        ]

    def _build_feature_names(
        self,
        cat_cols: List[str],
        continuous_cols: List[str],
        ordinal_cols: List[str],
        missing_cols: List[str],
    ) -> List[str]:
        names: List[str] = []
        names.extend([f"cb_{col}" for col in cat_cols])
        names.extend([f"qt_{col}" for col in continuous_cols])
        names.extend([f"mm_{col}" for col in ordinal_cols])
        names.extend([f"missing_{col}" for col in missing_cols])
        return names

    def _fit_categorical_encoder(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        random_seed: int,
    ) -> Tuple[CatBoostEncoder, pd.DataFrame]:
        categorical_encoder = self._make_catboost_encoder(X_train.columns.tolist(), random_seed)
        categorical_encoder.fit(X_train, y_train)
        oof_frame = self._build_categorical_oof_frame(
            X_train,
            y_train,
            random_seed=random_seed,
        )
        return categorical_encoder, oof_frame

    def _build_categorical_oof_frame(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        random_seed: int,
    ) -> pd.DataFrame:
        splitter = self._make_target_encoding_splitter(y_train, random_seed=random_seed)
        if splitter is None:
            fallback_encoder = self._make_catboost_encoder(X_train.columns.tolist(), random_seed)
            return fallback_encoder.fit_transform(X_train, y_train).astype(np.float64)

        oof_frame = pd.DataFrame(index=X_train.index, columns=X_train.columns, dtype=np.float64)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            fold_encoder = self._make_catboost_encoder(
                X_train.columns.tolist(),
                random_seed + fold_idx,
            )
            fold_encoder.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            transformed = fold_encoder.transform(X_train.iloc[val_idx]).astype(np.float64)
            oof_frame.iloc[val_idx] = transformed.to_numpy(dtype=np.float64)
        return oof_frame

    def _make_catboost_encoder(
        self,
        cols: List[str],
        random_seed: int,
    ) -> CatBoostEncoder:
        return CatBoostEncoder(
            cols=cols,
            handle_unknown="value",
            handle_missing="value",
            random_state=random_seed,
        )

    def _make_target_encoding_splitter(
        self,
        y_train: pd.Series,
        *,
        random_seed: int,
    ) -> StratifiedKFold | None:
        class_counts = y_train.value_counts()
        if class_counts.empty:
            return None

        n_splits = min(self.target_encoding_folds, int(class_counts.min()))
        if n_splits < 2:
            return None
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

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
            cat_df = self._transform_categorical_frame(X[state.categorical_columns], state)
            if state.categorical_scaler:
                scaled = state.categorical_scaler.transform(cat_df.values)
            else:
                scaled = cat_df.values
            cat_df = pd.DataFrame(
                np.clip(scaled, -1.0, 1.0),
                columns=[f"cb_{col}" for col in state.categorical_columns],
                index=X.index,
            ).astype(np.float32)
            frames.append(cat_df)
        if state.continuous_imputer and state.continuous_transformer and state.continuous_columns:
            num_array = state.continuous_imputer.transform(X[state.continuous_columns])
            num_scaled = np.clip(
                self._scale_to_unit_interval(num_array, state.continuous_transformer),
                -1.0,
                1.0,
            )
            num_df = pd.DataFrame(
                num_scaled,
                columns=[f"qt_{col}" for col in state.continuous_columns],
                index=X.index,
            )
            frames.append(num_df)
        if state.ordinal_imputer and state.ordinal_scaler and state.ordinal_columns:
            ordinal_array = state.ordinal_imputer.transform(X[state.ordinal_columns])
            ordinal_scaled = np.clip(state.ordinal_scaler.transform(ordinal_array), -1.0, 1.0)
            ordinal_df = pd.DataFrame(
                ordinal_scaled,
                columns=[f"mm_{col}" for col in state.ordinal_columns],
                index=X.index,
            )
            frames.append(ordinal_df)
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

    def _transform_categorical_frame(
        self,
        X: pd.DataFrame,
        state: KANSOTAPreprocessorState,
    ) -> pd.DataFrame:
        if state.categorical_oof_frame is not None:
            aligned_oof = state.categorical_oof_frame.reindex(X.index)
            if not aligned_oof.isna().any().any():
                return aligned_oof.astype(np.float64)
        transformed = state.categorical_encoder.transform(X)
        return transformed.astype(np.float64)

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
        from configs import set_global_seed

        seed = set_global_seed(42)
        results = run_pipeline(default_path, random_seed=seed)
        print("KAN-SOTA preprocessing complete -> outer tensors:", results["X_train_outer"].shape)
    else:
        print(
            "Set the Kaggle training CSV at data/prudential-life-insurance-assessment/train.csv",
        )
