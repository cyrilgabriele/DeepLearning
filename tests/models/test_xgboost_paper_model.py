import numpy as np
import pandas as pd
import pytest
from collections import OrderedDict

from src.models.xgboost_paper import XGBoostPaperModel


def _toy_dataset(n: int = 80, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(1, 9, size=n), name="Response")
    return X, y


class TestXGBoostPaperModel:
    def test_fit_predict_without_tuning(self):
        X, y = _toy_dataset(48)
        model = XGBoostPaperModel(
            random_state=0,
            n_estimators=5,
            auto_tune=False,
            max_depth=3,
            learning_rate=0.2,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (48,)
        assert preds.dtype == int
        assert set(np.unique(preds)).issubset(set(range(1, 9)))

    def test_auto_tune_respects_custom_grid(self, monkeypatch):
        X, y = _toy_dataset(40)
        X_train = X.iloc[:28]
        y_train = y.iloc[:28]
        X_val = X.iloc[28:]
        y_val = y.iloc[28:]

        grid = OrderedDict([("max_depth", (2, 4))])
        model = XGBoostPaperModel(
            random_state=1,
            n_estimators=3,
            auto_tune=True,
            tuning_grid=grid,
        )

        tried = []

        def fake_eval(self, params, splits, seed):
            value = params["max_depth"]
            tried.append(value)
            return float(value)

        monkeypatch.setattr(XGBoostPaperModel, "_evaluate_candidate", fake_eval)

        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        assert model.best_params_["max_depth"] == 4
        assert tried == [2, 4]

    def test_validation_splits_used_during_tuning(self, monkeypatch):
        X, y = _toy_dataset(60)
        X_train = X.iloc[:30]
        y_train = y.iloc[:30]
        X_val = X.iloc[30:40]
        y_val = y.iloc[30:40]

        split_a = (X.iloc[0:20], X.iloc[20:30], y.iloc[0:20], y.iloc[20:30])
        split_b = (X.iloc[5:25], X.iloc[25:35], y.iloc[5:25], y.iloc[25:35])

        grid = OrderedDict([("max_depth", (2, 4))])
        model = XGBoostPaperModel(
            random_state=2,
            n_estimators=3,
            auto_tune=True,
            tuning_grid=grid,
        )

        seen_split_lengths = []

        def fake_eval(self, params, splits, seed):
            seen_split_lengths.append(len(splits))
            return float(params["max_depth"])

        monkeypatch.setattr(XGBoostPaperModel, "_evaluate_candidate", fake_eval)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            validation_splits=[split_a, split_b],
        )

        assert model.best_params_["max_depth"] == 4
        assert seen_split_lengths == [2, 2]

    def test_seed_trials_selects_best_seed(self, monkeypatch):
        X, y = _toy_dataset(24)
        scores = {11: 0.2, 13: 0.6, 17: 0.3}

        class DummyEstimator:
            def __init__(self, seed):
                self.seed = seed

            def fit(self, *args, **kwargs):
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        def fake_single_seed(self, *, X_train, y_train, X_val, y_val, tuning_splits, seed):
            estimator = DummyEstimator(seed)
            params = {"max_depth": seed}
            return estimator, params, scores[seed]

        monkeypatch.setattr(XGBoostPaperModel, "_fit_single_seed", fake_single_seed)

        model = XGBoostPaperModel(
            random_state=5,
            n_estimators=2,
            auto_tune=False,
            seed_trials=[11, 13, 17],
        )
        model.fit(X, y)
        assert isinstance(model._estimator, DummyEstimator)
        assert model._estimator.seed == 13
        assert model.best_params_["max_depth"] == 13
        assert model.best_kappa_ == pytest.approx(0.6)
