"""GLM (Ridge regression) baseline with QWK threshold optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.metrics.qwk import optimize_thresholds, _apply_thresholds
from src.models.base import PrudentialModel


class GLMBaseline(PrudentialModel):
    """Ridge regression baseline with QWK-optimised ordinal thresholds.

    Mirrors the XGBBaseline interface so it slots into the same Trainer
    pipeline. Thresholds are fitted on the training set inside fit().
    """

    def __init__(self, *, alpha: float, random_state: int = 42, **kwargs) -> None:
        super().__init__(alpha=alpha, random_state=random_state)
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.thresholds: np.ndarray | None = None
        self.threshold_source_split: str | None = None
        self.threshold_optimization_qwk: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        validation_data=None,
        **_fit_kwargs,
    ) -> None:
        self.model.fit(X, y)
        y_cont = self.model.predict(X)
        self.thresholds, self.threshold_optimization_qwk = optimize_thresholds(y.to_numpy(), y_cont)
        self.threshold_source_split = "training"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Call fit() before predict().")
        y_cont = self.model.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8).astype(int)

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


def build_glm_model(*, random_state: int = 42, alpha: float, **_kwargs) -> GLMBaseline:
    """Factory for the model registry. Ignores depth/width/device kwargs."""
    return GLMBaseline(alpha=alpha, random_state=random_state)
