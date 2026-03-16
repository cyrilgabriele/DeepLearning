import numpy as np
import pytest
from src.metrics.qwk import quadratic_weighted_kappa, optimize_thresholds


class TestQWK:
    def test_perfect_agreement(self):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)

    def test_no_agreement(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([8, 7, 6, 5])
        kappa = quadratic_weighted_kappa(y_true, y_pred)
        assert kappa < 0.0

    def test_partial_agreement(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 4])
        kappa = quadratic_weighted_kappa(y_true, y_pred)
        assert 0.5 < kappa < 1.0

    def test_symmetric(self):
        y_true = np.array([1, 3, 5, 7])
        y_pred = np.array([2, 3, 4, 8])
        assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(
            quadratic_weighted_kappa(y_pred, y_true)
        )


class TestThresholdOptimizer:
    def test_perfect_continuous(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 10)
        y_cont = y_true.astype(float)
        thresholds, kappa = optimize_thresholds(y_true, y_cont)
        assert len(thresholds) == 7
        assert kappa > 0.95

    def test_noisy_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(1, 9, size=200)
        y_cont = y_true + rng.normal(0, 0.5, size=200)
        thresholds, kappa = optimize_thresholds(y_true, y_cont)
        assert kappa > 0.5

    def test_thresholds_are_sorted(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(1, 9, size=200)
        y_cont = y_true + rng.normal(0, 1.0, size=200)
        thresholds, _ = optimize_thresholds(y_true, y_cont)
        assert all(thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1))
