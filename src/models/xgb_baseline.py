import numpy as np
import pandas as pd
import xgboost as xgb
from src.metrics.qwk import optimize_thresholds, quadratic_weighted_kappa, _apply_thresholds
from src.models.base import PrudentialModel


class XGBBaseline(PrudentialModel):
    """XGBoost regression baseline with QWK threshold optimization.

    Thresholds are optimised on the training set inside fit() so that
    predict() returns ordinal class labels (1-8) directly.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            random_state=random_state,
        )
        self.thresholds: np.ndarray | None = None

    def fit(self, X, y, eval_set=None) -> None:
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        y_cont = self.model.predict(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        self.thresholds, _ = optimize_thresholds(y_arr, y_cont)

    def predict(self, X) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Call fit() before predict().")
        y_cont = self.model.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8).astype(int)

    def evaluate(self, X, y_true) -> float:
        """Re-optimise thresholds on the given split and return QWK."""
        y_cont = self.model.predict(X)
        y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
        self.thresholds, kappa = optimize_thresholds(y_arr, y_cont)
        return kappa


def build_xgb_model(
    *,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    **_kwargs,
) -> XGBBaseline:
    """Factory for the model registry. Ignores depth/width/device kwargs."""
    return XGBBaseline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
