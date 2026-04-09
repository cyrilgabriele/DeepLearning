"""Tests for feature_validation.py — feature subset validation curves."""

import numpy as np
import pandas as pd
import pytest

from src.interpretability.feature_validation import (
    compute_feature_validation_curves,
)


@pytest.fixture
def sample_data():
    """Create minimal evaluation data with 5 features and 50 samples."""
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    feature_names = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
    )
    y = pd.Series(np.random.randint(1, 9, n_samples), name="Response")
    return X, y, feature_names


@pytest.fixture
def dummy_rankings(sample_data):
    _, _, feature_names = sample_data
    return {
        "ModelA": feature_names,
        "ModelB": list(reversed(feature_names)),
    }


@pytest.fixture
def dummy_predictors(sample_data):
    """Predictors that return constant predictions (for testing structure)."""
    X, _, _ = sample_data
    return {
        "ModelA": lambda X_df: np.full(len(X_df), 4),
        "ModelB": lambda X_df: np.full(len(X_df), 5),
    }


class TestComputeFeatureValidationCurves:
    def test_returns_dict_per_model(self, dummy_rankings, dummy_predictors, sample_data):
        X, y, _ = sample_data
        curves = compute_feature_validation_curves(
            dummy_rankings, dummy_predictors, X, y,
            retention_steps=[2, 5],
        )
        assert "ModelA" in curves
        assert "ModelB" in curves

    def test_curve_points_have_expected_keys(self, dummy_rankings, dummy_predictors, sample_data):
        X, y, _ = sample_data
        curves = compute_feature_validation_curves(
            dummy_rankings, dummy_predictors, X, y,
            retention_steps=[3],
        )
        point = curves["ModelA"][0]
        assert "n_features" in point
        assert "pct_features" in point
        assert "qwk" in point
        assert point["n_features"] == 3

    def test_retention_steps_order(self, dummy_rankings, dummy_predictors, sample_data):
        X, y, _ = sample_data
        steps = [1, 3, 5]
        curves = compute_feature_validation_curves(
            dummy_rankings, dummy_predictors, X, y,
            retention_steps=steps,
        )
        for model_points in curves.values():
            n_feats = [p["n_features"] for p in model_points]
            assert n_feats == steps

    def test_missing_model_in_predict_skipped(self, sample_data):
        X, y, feature_names = sample_data
        rankings = {"ModelA": feature_names}
        predictors = {}  # empty — ModelA not in predictors
        curves = compute_feature_validation_curves(
            rankings, predictors, X, y,
            retention_steps=[3],
        )
        assert curves == {}

    def test_full_features_uses_all_columns(self, dummy_rankings, dummy_predictors, sample_data):
        X, y, _ = sample_data
        curves = compute_feature_validation_curves(
            dummy_rankings, dummy_predictors, X, y,
            retention_steps=[X.shape[1]],
        )
        for model_points in curves.values():
            assert model_points[0]["pct_features"] == 1.0
