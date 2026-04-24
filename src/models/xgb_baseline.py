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
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            random_state=random_state,
        )
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            tree_method="hist",
            random_state=random_state,
        )
        self.thresholds: np.ndarray | None = None
        self.threshold_source_split: str | None = None
        self.threshold_optimization_qwk: float | None = None

    def fit(self, X, y, eval_set=None, validation_data=None, validation_splits=None, **_kwargs) -> None:
        # Accept trainer-style kwargs (validation_data, validation_splits)
        # so this class drops into the current Trainer pipeline. Thresholds
        # are fit on the training predictions — same semantics the old sweep
        # used — regardless of whether eval_set / validation_data is passed.
        _ = validation_data, validation_splits  # unused
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        y_cont = self.model.predict(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        self.thresholds, self.threshold_optimization_qwk = optimize_thresholds(y_arr, y_cont)
        self.threshold_source_split = "training"

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
        self.threshold_source_split = "evaluation"
        self.threshold_optimization_qwk = float(kappa)
        return kappa

    def get_ordinal_calibration(self) -> dict[str, object] | None:
        if self.thresholds is None:
            return None
        payload: dict[str, object] = {
            "method": "optimized_thresholds",
            "num_classes": 8,
            "thresholds": [float(value) for value in self.thresholds],
        }
        if self.threshold_source_split is not None:
            payload["source_split"] = self.threshold_source_split
        if self.threshold_optimization_qwk is not None:
            payload["optimized_qwk_on_source_split"] = float(self.threshold_optimization_qwk)
        return payload


def build_xgb_model(
    *,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    **_kwargs,
) -> XGBBaseline:
    """Factory for the model registry. Forwards all tuned hyperparameters so
    the Optuna-best config in sweeps/xgb_xgb_paper_best.json is reproducible
    end-to-end through the current Trainer pipeline."""
    return XGBBaseline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        random_state=random_state,
    )
