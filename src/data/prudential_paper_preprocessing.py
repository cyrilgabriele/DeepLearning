"""Preprocessing pipeline mirroring the Prudential paper baseline.

Derived from *Classification of Insurance Claim Risk Using the Multilayer
Perceptron Method* (ResearchGate DOI 393359937). The authors applied simple
target encodings and median imputations before feeding the features into an
MLP/XGBoost stack. We mirror those steps verbatim, with the only adjustment
being that all scaled outputs are forced into [-1, 1] so KAN training stays
stable on our codebase.
"""

import pandas as pd
from pathlib import Path
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from .prudential_features import get_feature_lists


class PrudentialPaperPreprocessor:
    """Faithful reproduction of the paper's preprocessing recipe."""

    def __init__(self, missing_threshold: float = 0.5):
        self.missing_threshold = missing_threshold
        self.feature_lists = None
        self.dropped_features = []
        self.binary_params = {}

        self.encoder = TargetEncoder()
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler_cont = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_ord = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_cat = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()

        missing_rate = X.isnull().mean()
        self.dropped_features = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        X = X.drop(columns=self.dropped_features)

        self.feature_lists = get_feature_lists(X)

        cols_to_encode = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cols_to_encode:
            X[cols_to_encode] = self.encoder.fit_transform(X[cols_to_encode], y)

        X_no_id = X.drop(columns=["Id"], errors="ignore")
        X_imputed = self.imputer.fit_transform(X_no_id)

        orig_cols = list(X_no_id.columns)
        indicator_names = [f"missing_{orig_cols[i]}" for i in self.imputer.indicator_.features_]
        new_cols = orig_cols + indicator_names

        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)

        self.feature_lists["binary"] = list(set(self.feature_lists["binary"]) | set(indicator_names))

        return self._scale_all(X, fit=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=self.dropped_features, errors="ignore")

        cols_to_encode = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cols_to_encode:
            X[cols_to_encode] = self.encoder.transform(X[cols_to_encode])

        X_no_id = X.drop(columns=["Id"], errors="ignore")
        X_imputed = self.imputer.transform(X_no_id)

        orig_cols = list(X_no_id.columns)
        indicator_names = [f"missing_{orig_cols[i]}" for i in self.imputer.indicator_.features_]
        new_cols = orig_cols + indicator_names

        X = pd.DataFrame(X_imputed, columns=new_cols, index=X.index)

        return self._scale_all(X, fit=False)


    def _scale_all(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        X_final = X.copy()

        binary_cols = [c for c in self.feature_lists["binary"] if c in X.columns]
        for col in binary_cols:
            if fit:
                c_min, c_max = X[col].min(), X[col].max()
                self.binary_params[col] = (c_min, c_max)
            else:
                c_min, c_max = self.binary_params.get(col, (0.0, 1.0))

            if c_max > c_min:
                X_final[col] = 2 * (X[col] - c_min) / (c_max - c_min) - 1
            else:
                X_final[col] = 0.0

        cont_cols = [c for c in self.feature_lists["continuous"] if c in X.columns]
        if cont_cols:
            if fit:
                X_final[cont_cols] = self.scaler_cont.fit_transform(X[cont_cols])
            else:
                X_final[cont_cols] = self.scaler_cont.transform(X[cont_cols])

        ord_cols = [c for c in self.feature_lists["ordinal"] if c in X.columns]
        if ord_cols:
            if fit:
                X_final[ord_cols] = self.scaler_ord.fit_transform(X[ord_cols])
            else:
                X_final[ord_cols] = self.scaler_ord.transform(X[ord_cols])

        cat_cols = [c for c in self.feature_lists["categorical"] if c in X.columns]
        if cat_cols:
            if fit:
                X_final[cat_cols] = self.scaler_cat.fit_transform(X[cat_cols])
            else:
                X_final[cat_cols] = self.scaler_cat.transform(X[cat_cols])

        return X_final.clip(-1, 1)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "prudential-life-insurance-assessment"
    TRAIN_PATH = DATA_DIR / "train.csv"

    if TRAIN_PATH.exists():
        train = pd.read_csv(TRAIN_PATH)
        y = train["Response"]
        X = train.drop(columns=["Response"])

        preprocessor = PrudentialPaperPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(f"Range: [{X_processed.min().min():.4f}, {X_processed.max().max():.4f}]")
    else:
        print(f"Data file not found at {TRAIN_PATH}.")
