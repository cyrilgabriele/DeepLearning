"""Unit tests for TabPFNAutoRegressor wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def fake_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(((np.arange(40) % 8) + 1).astype(int), name="Response")
    return X, y


@pytest.fixture()
def fake_estimator():
    """A stand-in TabPFNRegressor that returns deterministic regression outputs."""
    est = MagicMock()
    def _predict(X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return (np.arange(n) % 8 + 1).astype(float) + 0.1
    est.fit.return_value = est
    est.predict.side_effect = _predict
    return est


def test_tabpfn_wrapper_predict_returns_ordinal_classes(fake_xy, fake_estimator):
    X, y = fake_xy
    with patch(
        "src.models.tabpfn._build_auto_tabpfn",
        return_value=fake_estimator,
    ):
        from src.models.tabpfn import TabPFNAutoRegressor

        model = TabPFNAutoRegressor(n_estimators=1, max_time=30, device="cpu", random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

    assert preds.dtype.kind in ("i", "u")
    assert preds.min() >= 1
    assert preds.max() <= 8


def test_tabpfn_wrapper_get_ordinal_calibration_returns_payload(fake_xy, fake_estimator):
    X, y = fake_xy
    with patch(
        "src.models.tabpfn._build_auto_tabpfn",
        return_value=fake_estimator,
    ):
        from src.models.tabpfn import TabPFNAutoRegressor

        model = TabPFNAutoRegressor(n_estimators=1, max_time=30, device="cpu", random_state=42)
        model.fit(X, y)
        cal = model.get_ordinal_calibration()

    assert cal is not None
    assert cal["method"] == "optimized_thresholds"
    assert cal["num_classes"] == 8
    assert len(cal["thresholds"]) == 7
    assert cal["source_split"] == "training"


def test_tabpfn_wrapper_predict_before_fit_raises(fake_xy):
    X, _ = fake_xy
    from src.models.tabpfn import TabPFNAutoRegressor

    model = TabPFNAutoRegressor(n_estimators=1, max_time=30, device="cpu", random_state=42)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_tabpfn_wrapper_evaluate_recalibrates_thresholds(fake_xy, fake_estimator):
    X, y = fake_xy
    with patch(
        "src.models.tabpfn._build_auto_tabpfn",
        return_value=fake_estimator,
    ):
        from src.models.tabpfn import TabPFNAutoRegressor

        model = TabPFNAutoRegressor(n_estimators=1, max_time=30, device="cpu", random_state=42)
        model.fit(X, y)
        kappa = model.evaluate(X, y)

    assert isinstance(kappa, float)
    cal = model.get_ordinal_calibration()
    assert cal["source_split"] == "evaluation"
