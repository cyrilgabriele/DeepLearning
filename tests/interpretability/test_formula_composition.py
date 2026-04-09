"""Tests for formula_composition.py — SymPy composition of per-edge symbolic fits."""

import numpy as np
import pandas as pd
import pytest
import sympy as sp
import torch

from src.interpretability.formula_composition import (
    _build_templates,
    _fit_params,
    compose_symbolic_model,
    formula_to_sympy,
)
from src.models.tabkan import TabKAN


@pytest.fixture
def tiny_kan():
    """A tiny ChebyKAN with 3 inputs, 2 hidden, 1 output layer."""
    module = TabKAN(in_features=3, widths=[2, 1], kan_type="chebykan", degree=3)
    module.eval()
    return module


@pytest.fixture
def feature_names():
    return ["feat_a", "feat_b", "feat_c"]


@pytest.fixture
def sample_fits_df():
    """Minimal symbolic fits DataFrame for a 2-layer KAN."""
    return pd.DataFrame([
        {"layer": 0, "edge_in": 0, "edge_out": 0, "input_feature": "feat_a",
         "formula": "a*x + b", "r_squared": 0.995, "quality_tier": "clean"},
        {"layer": 0, "edge_in": 1, "edge_out": 0, "input_feature": "feat_b",
         "formula": "a*x^2 + b*x + c", "r_squared": 0.92, "quality_tier": "acceptable"},
        {"layer": 0, "edge_in": 0, "edge_out": 1, "input_feature": "feat_a",
         "formula": "a*sin(x) + b", "r_squared": 0.85, "quality_tier": "flagged"},
        {"layer": 1, "edge_in": 0, "edge_out": 0, "input_feature": "h0",
         "formula": "a*x + b", "r_squared": 0.99, "quality_tier": "clean"},
    ])


class TestBuildTemplates:
    def test_returns_dict(self):
        templates = _build_templates()
        assert isinstance(templates, dict)
        assert len(templates) >= 10

    def test_all_are_sympy_exprs(self):
        for name, expr in _build_templates().items():
            assert isinstance(expr, sp.Basic), f"{name} is not a SymPy expression"

    def test_linear_template(self):
        templates = _build_templates()
        x = sp.Symbol("x")
        a, b = sp.symbols("a b")
        assert templates["a*x + b"] == a * x + b


class TestFitParams:
    def test_linear_fit(self):
        x = np.linspace(-1, 1, 100)
        y = 2.5 * x + 0.3
        params = _fit_params("a*x + b", x, y)
        assert params is not None
        assert abs(params["a"] - 2.5) < 0.01
        assert abs(params["b"] - 0.3) < 0.01

    def test_unknown_formula_returns_none(self):
        x = np.linspace(-1, 1, 100)
        y = x
        assert _fit_params("unknown_formula", x, y) is None

    def test_constant_fit(self):
        x = np.linspace(-1, 1, 100)
        y = np.full_like(x, 3.14)
        params = _fit_params("a (constant)", x, y)
        assert params is not None
        assert abs(params["a"] - 3.14) < 0.01


class TestFormulaToSympy:
    def test_linear(self):
        x = sp.Symbol("feat_a")
        expr = formula_to_sympy("a*x + b", {"a": 2.0, "b": 1.0}, x)
        assert expr is not None
        # Evaluate at x=0 should give b=1.0
        val = float(expr.subs(x, 0))
        assert abs(val - 1.0) < 0.01

    def test_unknown_returns_none(self):
        x = sp.Symbol("x")
        assert formula_to_sympy("nope", {"a": 1}, x) is None

    def test_substitution_uses_input_symbol(self):
        feat = sp.Symbol("BMI")
        expr = formula_to_sympy("a*x + b", {"a": 1.0, "b": 0.0}, feat)
        assert feat in expr.free_symbols


class TestComposeSymbolicModel:
    def test_returns_expected_keys(self, tiny_kan, sample_fits_df, feature_names):
        result = compose_symbolic_model(
            sample_fits_df, tiny_kan, feature_names, min_r2=0.90,
        )
        assert "formulas" in result
        assert "sympy_exprs" in result
        assert "coverage" in result
        assert "feature_symbols" in result

    def test_coverage_counts(self, tiny_kan, sample_fits_df, feature_names):
        result = compose_symbolic_model(
            sample_fits_df, tiny_kan, feature_names, min_r2=0.90,
        )
        cov = result["coverage"]
        assert cov["total_edges"] == 4
        # min_r2=0.90 keeps edges with r2 >= 0.90 (3 of 4)
        assert cov["included_edges"] == 3

    def test_higher_threshold_fewer_edges(self, tiny_kan, sample_fits_df, feature_names):
        result_low = compose_symbolic_model(
            sample_fits_df, tiny_kan, feature_names, min_r2=0.50,
        )
        result_high = compose_symbolic_model(
            sample_fits_df, tiny_kan, feature_names, min_r2=0.99,
        )
        assert result_high["coverage"]["included_edges"] <= result_low["coverage"]["included_edges"]

    def test_empty_fits_returns_empty(self, tiny_kan, feature_names):
        empty_df = pd.DataFrame(columns=[
            "layer", "edge_in", "edge_out", "input_feature",
            "formula", "r_squared", "quality_tier",
        ])
        result = compose_symbolic_model(empty_df, tiny_kan, feature_names)
        assert result["formulas"] == {}

    def test_feature_symbols_created(self, tiny_kan, sample_fits_df, feature_names):
        result = compose_symbolic_model(
            sample_fits_df, tiny_kan, feature_names, min_r2=0.90,
        )
        for name in feature_names:
            assert name in result["feature_symbols"]
