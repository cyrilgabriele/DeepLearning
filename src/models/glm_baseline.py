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

    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs) -> None:
        super().__init__(alpha=alpha, random_state=random_state)
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.thresholds: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        y_cont = self.model.predict(X)
        self.thresholds, _ = optimize_thresholds(y.to_numpy(), y_cont)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Call fit() before predict().")
        y_cont = self.model.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8).astype(int)


def build_glm_model(*, random_state: int = 42, alpha: float = 1.0, **_kwargs) -> GLMBaseline:
    """Factory for the model registry. Ignores depth/width/device kwargs."""
    return GLMBaseline(alpha=alpha, random_state=random_state)
