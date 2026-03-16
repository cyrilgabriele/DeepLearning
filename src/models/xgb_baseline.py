import numpy as np
import xgboost as xgb
from src.metrics.qwk import optimize_thresholds, quadratic_weighted_kappa, _apply_thresholds


class XGBBaseline:
    """XGBoost regression baseline with QWK threshold optimization."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            random_state=42,
            **kwargs,
        )
        self.thresholds = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_cont = self.predict(X)
        self.thresholds, kappa = optimize_thresholds(y_true, y_cont)
        return kappa

    def predict_ordinal(self, X: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Call evaluate() first to optimize thresholds.")
        y_cont = self.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8)
