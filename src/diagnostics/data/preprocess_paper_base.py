"""Shared Prudential preprocessing logic used by paper-faithful pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger as get_project_logger


@dataclass(frozen=True)
class PreprocessorState:
    """Holds the deterministic Product_Info_2 mapping used across splits."""

    product_info_2_mapping: Dict[str, int]


class PaperPreprocessingBase:
    """Baseline Prudential preprocessing shared by all paper-aligned variants."""

    TARGET_COLUMN = "Response"
    ID_COLUMN = "Id"
    PRODUCT_INFO_2 = "Product_Info_2"
    OUTER_TEST_SIZE = 0.2
    INNER_TEST_SIZE = 0.2
    INNER_SPLITS = 5

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or get_project_logger()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        """Load the Prudential CSV and log dataset/missing stats."""

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV not found at {path}.")

        df = pd.read_csv(path)
        self.logger.info("Loaded dataset %s with shape %s", path, df.shape)

        missing_counts = df.isna().sum()
        self.logger.info("Per-column missing counts (original data):\n%s", missing_counts.to_string())
        return df

    def fit_preprocessor(self, df: pd.DataFrame) -> PreprocessorState:
        """Fit the Product_Info_2 label encoder using lexicographic ordering."""

        if self.PRODUCT_INFO_2 not in df.columns:
            raise KeyError(f"Column '{self.PRODUCT_INFO_2}' not found in dataframe.")

        categories = sorted(df[self.PRODUCT_INFO_2].dropna().unique())
        mapping = {category: idx for idx, category in enumerate(categories)}
        self.logger.info(
            "Product_Info_2 deterministic mapping (%d categories): %s",
            len(mapping),
            mapping,
        )
        return PreprocessorState(product_info_2_mapping=mapping)

    def transform(
        self,
        df: pd.DataFrame,
        state: PreprocessorState,
        **_,
    ) -> Tuple[pd.DataFrame, pd.Series | None]:
        """Apply paper-faithful preprocessing and optionally extract targets."""

        return self._baseline_transform(df, state)

    def make_outer_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        test_size: float | None = None,
        random_state: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform the single 80/20 random train/test split from the paper."""

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size if test_size is not None else self.OUTER_TEST_SIZE,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        self.logger.info(
            "Outer split shapes -> train: %s / %s, test: %s / %s",
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
        )
        return X_train, X_test, y_train, y_test

    def make_inner_splits(
        self,
        X_train_outer: pd.DataFrame,
        y_train_outer: pd.Series,
        *,
        n_splits: int | None = None,
        test_size: float | None = None,
        base_random_seed: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Create repeated 80/20 splits from the outer training fold."""

        total_splits = n_splits if n_splits is not None else self.INNER_SPLITS
        split_test_size = test_size if test_size is not None else self.INNER_TEST_SIZE
        splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = []
        for idx in range(total_splits):
            split_seed = base_random_seed + idx + 1
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_outer,
                y_train_outer,
                test_size=split_test_size,
                random_state=split_seed,
                shuffle=True,
                stratify=None,
            )
            self.logger.info(
                "Inner split %d shapes -> train: %s / %s, val: %s / %s",
                idx + 1,
                X_train.shape,
                y_train.shape,
                X_val.shape,
                y_val.shape,
            )
            splits.append((X_train, X_val, y_train, y_val))
        return splits

    def run_pipeline(self, csv_path: str | Path, *, random_seed: int) -> Dict[str, object]:
        """Execute the full paper-baseline preprocessing workflow."""

        df = self.load_data(csv_path)
        state = self.fit_preprocessor(df)
        X, y = self.transform(df, state)
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = self.make_outer_split(
            X,
            y,
            random_state=random_seed,
        )
        inner_splits = self.make_inner_splits(
            X_train_outer,
            y_train_outer,
            base_random_seed=random_seed,
        )

        return {
            "X_train_outer": X_train_outer,
            "X_test_outer": X_test_outer,
            "y_train_outer": y_train_outer,
            "y_test_outer": y_test_outer,
            "inner_splits": inner_splits,
            "preprocessor_state": state,
        }

    # ---------------------------------------------------------------------
    # ---- Shared helpers -------------------------------------------------
    def _baseline_transform(
        self,
        df: pd.DataFrame,
        state: PreprocessorState,
    ) -> Tuple[pd.DataFrame, pd.Series | None]:
        has_target = self.TARGET_COLUMN in df.columns
        if has_target:
            y = df[self.TARGET_COLUMN].copy()
            X = df.drop(columns=[self.TARGET_COLUMN]).copy()
        else:
            y = None
            X = df.copy()

        if self.PRODUCT_INFO_2 in X.columns:
            X[self.PRODUCT_INFO_2] = self._encode_product_info_2(
                X[self.PRODUCT_INFO_2],
                state.product_info_2_mapping,
            )

        if self.ID_COLUMN in X.columns:
            X = X.drop(columns=[self.ID_COLUMN])

        self.logger.info("Feature matrix shape after preprocessing (paper-faithful): %s", X.shape)
        if y is not None:
            self.logger.info("Target shape: %s", y.shape)
        else:
            self.logger.info("No target column present during transform; returning features only.")
        return X, y

    def _encode_product_info_2(
        self,
        series: pd.Series,
        mapping: Dict[str, int],
    ) -> pd.Series:
        encoded = series.map(mapping)
        mask_unknown = series.notna() & ~series.isin(mapping.keys())
        if mask_unknown.any():
            unknown_values = sorted(series[mask_unknown].unique().tolist())
            self.logger.warning(
                "Encountered %d unseen Product_Info_2 codes %s; encoding them as NaN.",
                len(unknown_values),
                unknown_values,
            )
        return encoded.astype("float64")
