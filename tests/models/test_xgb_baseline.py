import numpy as np
import pytest
from src.models.xgb_baseline import XGBBaseline


class TestXGBBaseline:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.randint(1, 9, size=100).astype(float)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_evaluate_returns_qwk(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.randint(1, 9, size=100).astype(float)
        model = XGBBaseline(n_estimators=50, max_depth=3)
        model.fit(X, y)
        qwk = model.evaluate(X, y)
        assert -1.0 <= qwk <= 1.0
