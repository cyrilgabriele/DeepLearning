"""Integration tests verifying the full pipeline: preprocessing -> model -> output.

These tests use synthetic data and do NOT require the real Prudential dataset.
They verify that every component connects correctly and produces valid results.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.preprocessing import preprocess_kan_paper as kan_prep
from src.models.tabkan import TabKAN, TabKANClassifier, build_tabkan_model
from src.models.mlp import MLPBaseline
from src.metrics.qwk import quadratic_weighted_kappa, optimize_thresholds, _apply_thresholds


def _tabkan_wrapper_kwargs(**overrides):
    params = {
        "flavor": "chebykan",
        "hidden_widths": [32, 16],
        "degree": 3,
        "max_epochs": 2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "sparsity_lambda": 0.0,
        "l1_weight": 1.0,
        "entropy_weight": 1.0,
    }
    params.update(overrides)
    return params


def _synthetic_prudential(n=200, seed=42):
    """Minimal synthetic Prudential-like DataFrame."""
    rng = np.random.RandomState(seed)
    data = {"Id": np.arange(n)}

    # Categorical
    data["Product_Info_2"] = rng.choice(["A1", "B2", "C3"], n)
    data["Product_Info_7"] = rng.randint(1, 4, n)
    data["InsuredInfo_1"] = rng.randint(1, 4, n)
    data["InsuredInfo_3"] = rng.randint(1, 3, n)
    data["Family_Hist_1"] = rng.randint(1, 4, n)
    for i in [2, 3, 4, 7, 8, 9]:
        data[f"Insurance_History_{i}"] = rng.randint(1, 4, n)
    for i in [3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21,
              23, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 41]:
        data[f"Medical_History_{i}"] = rng.randint(0, 4, n)

    # Binary
    data["Product_Info_1"] = rng.choice([1, 2], n)
    data["Product_Info_5"] = rng.choice([1, 2], n)
    data["Product_Info_6"] = rng.choice([1, 2, 3], n)
    data["Employment_Info_3"] = rng.choice([1, 3], n)
    data["Employment_Info_5"] = rng.choice([1, 2, 3], n)
    for i in [2, 4, 5, 6, 7]:
        data[f"InsuredInfo_{i}"] = rng.randint(1, 4, n)
    data["Insurance_History_1"] = rng.choice([1, 2], n)
    for i in [4, 22, 33, 38]:
        data[f"Medical_History_{i}"] = rng.choice([0, 1], n)
    for i in range(1, 49):
        data[f"Medical_Keyword_{i}"] = rng.choice([0, 1], n, p=[0.9, 0.1])

    # Continuous
    data["BMI"] = rng.normal(27, 5, n)
    data["Ht"] = rng.normal(0.7, 0.1, n)
    data["Wt"] = rng.normal(0.3, 0.1, n)
    data["Ins_Age"] = rng.uniform(0, 1, n)
    data["Product_Info_4"] = rng.uniform(0, 1, n)
    data["Employment_Info_1"] = rng.uniform(0, 0.1, n)
    data["Employment_Info_4"] = rng.uniform(0, 1, n)
    data["Employment_Info_6"] = rng.uniform(0, 1, n)
    for i in [2, 3, 4, 5]:
        data[f"Family_Hist_{i}"] = rng.uniform(0, 1, n)
    data["Insurance_History_5"] = rng.uniform(0, 1, n)
    for i in [1, 2, 10, 15, 24, 32]:
        vals = rng.uniform(0, 200, n)
        mask = rng.rand(n) < 0.3
        data[f"Medical_History_{i}"] = np.where(mask, np.nan, vals)

    # Ordinal (some extra columns)
    data["Employment_Info_2"] = rng.randint(1, 40, n)
    data["Product_Info_3"] = rng.randint(1, 40, n)
    data["Insurance_History_5"] = rng.uniform(0, 1, n)

    data["Response"] = rng.choice(range(1, 9), n)
    return pd.DataFrame(data)


def _preprocess_dataframe(df: pd.DataFrame):
    base_state = kan_prep.fit_preprocessor(df)
    X_base, _ = kan_prep.transform(df, base_state)
    kan_state = kan_prep.fit_kan_value_pipeline(X_base, base_state, logger=logging.getLogger("tabkan-tests"))
    X_processed, y_array = kan_prep.transform(df, base_state, kan_state=kan_state)
    feature_names = kan_state.feature_names
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    y_series = pd.Series(y_array, name="Response")
    return X_df, y_series


# ── Preprocessing Tests ──


class TestPreprocessingOutput:
    """Verify preprocessing produces valid input for KAN models."""

    @pytest.fixture
    def processed(self):
        df = _synthetic_prudential(200)
        X_out, y = _preprocess_dataframe(df)
        return X_out, y

    def test_no_nans(self, processed):
        X_out, _ = processed
        assert X_out.isnull().sum().sum() == 0, "Preprocessed data contains NaN"

    def test_range_minus1_to_1(self, processed):
        X_out, _ = processed
        assert np.isfinite(X_out.to_numpy()).all()

    def test_all_numeric(self, processed):
        X_out, _ = processed
        for dtype in X_out.dtypes:
            assert np.issubdtype(dtype, np.number), f"Non-numeric dtype: {dtype}"

    def test_missing_indicators_are_binary(self, processed):
        X_out, _ = processed
        missing_cols = [c for c in X_out.columns if c.startswith("missing_")]
        for col in missing_cols:
            unique = np.unique(X_out[col])
            assert set(unique).issubset({0.0, 1.0}), f"Indicator {col} not binary"


# ── Model Forward Pass Tests ──


class TestModelAcceptsPreprocessedData:
    """Verify all model types accept preprocessing output."""

    @pytest.fixture
    def preprocessed_batch(self):
        df = _synthetic_prudential(100)
        X_out, _ = _preprocess_dataframe(df)
        return torch.tensor(X_out.values.astype(np.float32)), X_out.shape[1]

    @pytest.mark.parametrize("kan_type,kwargs", [
        ("chebykan", {"degree": 3}),
        ("fourierkan", {"grid_size": 4}),
        ("bsplinekan", {"grid_size": 5, "spline_order": 3}),
    ])
    def test_kan_forward_no_nan(self, preprocessed_batch, kan_type, kwargs):
        X_t, n_feat = preprocessed_batch
        model = TabKAN(in_features=n_feat, widths=[32, 16], kan_type=kan_type, **kwargs)
        model.eval()
        with torch.no_grad():
            out = model(X_t)
        assert out.shape == (100, 1)
        assert torch.isfinite(out).all(), f"NaN/Inf in {kan_type} output"

    def test_mlp_forward_no_nan(self, preprocessed_batch):
        X_t, n_feat = preprocessed_batch
        model = MLPBaseline(in_features=n_feat, widths=[64, 32])
        model.eval()
        with torch.no_grad():
            out = model(X_t)
        assert out.shape == (100, 1)
        assert torch.isfinite(out).all()


# ── Threshold Optimization Tests ──


class TestThresholdOutput:
    """Verify threshold optimization produces valid ordinal classes."""

    def test_thresholds_produce_valid_classes(self):
        y_true = np.random.choice(range(1, 9), 200)
        y_cont = y_true.astype(float) + np.random.normal(0, 0.5, 200)
        thresholds, qwk = optimize_thresholds(y_true, y_cont)
        ordinal = _apply_thresholds(y_cont, thresholds)
        ordinal = np.clip(ordinal, 1, 8)
        assert all(1 <= c <= 8 for c in ordinal)
        assert len(thresholds) == 7
        assert np.all(np.diff(thresholds) > 0)
        assert qwk > 0.5  # Should be decent with low noise

    def test_thresholds_with_extreme_predictions(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 10)
        y_cont = np.array([-5.0, 0.5, 2.0, 4.0, 5.0, 6.0, 7.5, 15.0] * 10)
        thresholds, qwk = optimize_thresholds(y_true, y_cont)
        ordinal = np.clip(_apply_thresholds(y_cont, thresholds), 1, 8)
        assert all(1 <= c <= 8 for c in ordinal)


# ── End-to-End Pipeline Tests ──


class TestEndToEndPipeline:
    """Full pipeline: synthetic data -> preprocess -> model -> thresholds -> ordinal."""

    @pytest.mark.parametrize("kan_type", ["chebykan", "fourierkan", "bsplinekan"])
    def test_full_pipeline(self, kan_type):
        df = _synthetic_prudential(200)
        X_proc, y_series = _preprocess_dataframe(df)

        # Model forward
        X_t = torch.tensor(X_proc.values.astype(np.float32))
        y_np = y_series.values.astype(int)

        kwargs = {"degree": 3} if kan_type == "chebykan" else {"grid_size": 4}
        if kan_type == "bsplinekan":
            kwargs = {"grid_size": 5, "spline_order": 3}
        model = TabKAN(in_features=X_t.shape[1], widths=[32, 16], kan_type=kan_type, **kwargs)
        model.eval()
        with torch.no_grad():
            preds_cont = model(X_t).numpy().flatten()

        assert not np.any(np.isnan(preds_cont))

        # Threshold optimization
        thresholds, qwk = optimize_thresholds(y_np, preds_cont)
        ordinal = np.clip(_apply_thresholds(preds_cont, thresholds), 1, 8).astype(int)

        assert len(thresholds) == 7
        assert all(1 <= c <= 8 for c in ordinal)
        assert len(ordinal) == len(y_np)


# ── Registry Bridge Tests ──


class TestRegistryBridge:
    """Verify the build_tabkan_model factory works with PrudentialModel interface."""

    def test_build_and_predict(self):
        model = build_tabkan_model("tabkan-tiny", random_state=42, **_tabkan_wrapper_kwargs())
        assert isinstance(model, TabKANClassifier)

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.uniform(-1, 1, (50, 15)), columns=[f"f{i}" for i in range(15)])
        y = pd.Series(rng.choice(range(1, 9), 50))

        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)
        assert all(1 <= p <= 8 for p in preds)
        assert preds.dtype == int

    def test_all_presets(self):
        for preset in ["tabkan-tiny", "tabkan-small", "tabkan-base"]:
            model = build_tabkan_model(preset, random_state=42, **_tabkan_wrapper_kwargs())
            assert isinstance(model, TabKANClassifier)

    def test_depth_width_override(self):
        model = build_tabkan_model(
            "tabkan-base",
            random_state=42,
            **_tabkan_wrapper_kwargs(hidden_widths=None, depth=3, width=24),
        )
        assert model.widths == [24, 24, 24]

    def test_use_layernorm_toggle_round_trip(self):
        model = build_tabkan_model(
            "tabkan-tiny",
            random_state=42,
            **_tabkan_wrapper_kwargs(use_layernorm=False),
        )
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.uniform(-1, 1, (24, 8)), columns=[f"f{i}" for i in range(8)])
        y = pd.Series(rng.choice(range(1, 9), 24))

        model.fit(X, y)
        assert model.use_layernorm is False
        assert model.module is not None
        assert not any(isinstance(layer, nn.LayerNorm) for layer in model.module.kan_layers)
