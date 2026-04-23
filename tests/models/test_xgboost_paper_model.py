import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_paper import XGBoostPaperModel


_XGB_REQUIRED = dict(
    min_child_weight=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    num_classes=8,
    tree_method="hist",
    eval_metric="mlogloss",
    refit_full_training=False,
)


def _toy_dataset(n: int = 80, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(1, 9, size=n), name="Response")
    return X, y


class DummyEstimator:
    def __init__(self) -> None:
        self.fit_rows: int | None = None

    def fit(self, X, y, *args, **kwargs):
        self.fit_rows = len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestXGBoostPaperModel:
    def test_fit_predict_with_fixed_params(self):
        X, y = _toy_dataset(48)
        model = XGBoostPaperModel(
            random_state=0,
            n_estimators=5,
            max_depth=3,
            learning_rate=0.2,
            **_XGB_REQUIRED,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (48,)
        assert preds.dtype == int
        assert set(np.unique(preds)).issubset(set(range(1, 9)))

    def test_validation_splits_are_used_for_scoring(self, monkeypatch):
        X, y = _toy_dataset(60)
        X_train = X.iloc[:30]
        y_train = y.iloc[:30]
        X_val = X.iloc[30:40]
        y_val = y.iloc[30:40]

        split_a = (X.iloc[0:20], X.iloc[20:30], y.iloc[0:20], y.iloc[20:30])
        split_b = (X.iloc[5:25], X.iloc[25:35], y.iloc[5:25], y.iloc[25:35])

        seen_split_lengths: list[int] = []

        def fake_evaluate(self, params, splits, seed):
            seen_split_lengths.append(len(splits))
            return 0.42

        monkeypatch.setattr(XGBoostPaperModel, "_evaluate_candidate", fake_evaluate)
        monkeypatch.setattr(XGBoostPaperModel, "_build_estimator", lambda *args, **kwargs: DummyEstimator())

        model = XGBoostPaperModel(
            random_state=2,
            n_estimators=3,
            max_depth=4,
            learning_rate=0.1,
            **_XGB_REQUIRED,
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            validation_splits=[split_a, split_b],
        )

        assert model.best_params_["max_depth"] == 4
        assert model.best_kappa_ == pytest.approx(0.42)
        assert seen_split_lengths == [2]

    def test_refit_full_training_combines_train_and_validation(self, monkeypatch):
        X, y = _toy_dataset(40)
        X_train = X.iloc[:24]
        y_train = y.iloc[:24]
        X_val = X.iloc[24:40]
        y_val = y.iloc[24:40]
        estimator = DummyEstimator()

        monkeypatch.setattr(XGBoostPaperModel, "_build_estimator", lambda *args, **kwargs: estimator)

        model = XGBoostPaperModel(
            random_state=3,
            n_estimators=3,
            max_depth=3,
            learning_rate=0.1,
            **{**_XGB_REQUIRED, "refit_full_training": True},
        )
        model.fit(X_train, y_train, validation_data=(X_val, y_val))

        assert estimator.fit_rows == len(X_train) + len(X_val)
