"""Enhanced Prudential preprocessing tailored for Kolmogorov-Arnold Networks.

The baseline follows the workflow from the Prudential MLP paper, but we add
modern best practices to squeeze more signal for KANs:

1. IterativeImputer + missing indicators to preserve dependency structure.
2. CatBoost target encoding for high-cardinality categoricals.
3. QuantileTransformer on continuous features for uniform spline coverage.
4. Explicit [-1, 1] clipping for spline stability.
"""

from pathlib import Path
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from .prudential_features import get_feature_lists


class PrudentialKANPreprocessor:
    """State-of-the-art preprocessing optimized for KAN inputs.

    The class now supports stratified k-fold target encoding so categorical
    leak-through is reduced relative to a naive single-fit encoding.
    """

    def __init__(
        self,
        missing_threshold: float = 0.5,
        *,
        random_state: int = 42,
        use_stratified_kfold: bool = True,
        n_splits: int = 5,
    ):
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        self.use_stratified_kfold = use_stratified_kfold
        self.n_splits = n_splits
        self.feature_lists = None
        self.dropped_features = []
        self.binary_params = {}
        self._cat_encoding_mode = "standard"

        self.encoder = CatBoostEncoder(return_df=True)
        self.imputer = IterativeImputer(
            random_state=random_state,
            max_iter=10,
            add_indicator=True,
        )
        self.scaler_cont = QuantileTransformer(
            output_distribution="uniform",
            random_state=random_state,
        )
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
            cat_data = X[cols_to_encode].copy()
            encoded, mode = self._encode_categoricals(cat_data, y)
            X[cols_to_encode] = encoded
            self._cat_encoding_mode = mode

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

    def _encode_categoricals(self, X_cat: pd.DataFrame, y: pd.Series):
        """Encode categoricals using stratified k-fold when requested."""

        if not self.use_stratified_kfold:
            encoded = self.encoder.fit_transform(X_cat, y)
            return encoded, "standard"

        encoded = self._stratified_kfold_encode(X_cat, y)
        # Fit on full categorical data for inference-time transforms
        self.encoder.fit(X_cat, y)
        return encoded, "kfold"

    def _stratified_kfold_encode(self, X_cat: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if y is None:
            raise ValueError("Stratified k-fold encoding requires target labels.")

        y_series = pd.Series(y)
        class_counts = y_series.value_counts()
        min_per_class = class_counts.min()
        n_splits = min(self.n_splits, int(min_per_class))

        if n_splits < 2:
            raise ValueError(
                "Not enough samples per class to run stratified k-fold encoding. "
                "Reduce n_splits or disable stratified encoding."
            )

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        encoded = pd.DataFrame(index=X_cat.index, columns=X_cat.columns, dtype=float)
        X_values = X_cat.reset_index(drop=True)
        y_values = y_series.reset_index(drop=True)

        for train_idx, val_idx in skf.split(X_values, y_values):
            fold_encoder = CatBoostEncoder(return_df=True)
            fold_encoder.fit(X_values.iloc[train_idx], y_values.iloc[train_idx])
            transformed = fold_encoder.transform(X_values.iloc[val_idx])
            encoded.iloc[val_idx] = transformed.values

        # Re-align to the original index order
        encoded.index = X_cat.index
        return encoded

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
                cont_scaled = self.scaler_cont.fit_transform(X[cont_cols])
            else:
                cont_scaled = self.scaler_cont.transform(X[cont_cols])

            X_final[cont_cols] = cont_scaled * 2 - 1

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

        preprocessor = PrudentialKANPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(f"Range: [{X_processed.min().min():.4f}, {X_processed.max().max():.4f}]")
    else:
        print(f"Data file not found at {TRAIN_PATH}.")
