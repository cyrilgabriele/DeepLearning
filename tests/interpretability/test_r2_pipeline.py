"""Unit tests for the R² symbolic-fit pipeline (Issue 08).

Uses a small synthetic 2-layer ChebyKAN with known activations to verify
that the pipeline correctly identifies the symbolic form of simple functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_chebykan(in_features: int = 3, hidden: int = 4, degree: int = 3):
    """Build a tiny 2-layer ChebyKAN module."""
    from src.models.tabkan import TabKAN
    return TabKAN(
        in_features=in_features,
        widths=[hidden, 2],
        kan_type="chebykan",
        degree=degree,
    )


def _set_edge_to_identity(layer, out_i: int, in_i: int) -> None:
    """Force edge (out_i, in_i) to approximate the identity function.

    For a degree-3 Chebyshev layer:
      T_0=1, T_1=x, so setting coeff[1]=1, rest=0 gives f(x)=tanh(x)≈x near 0.
    We set it to T_1 only (linear in the normalised input).
    """
    with torch.no_grad():
        layer.cheby_coeffs[out_i, in_i, :] = 0.0
        layer.cheby_coeffs[out_i, in_i, 1] = 1.0   # T_1(tanh(x)) = tanh(x)
        layer.base_weight[out_i, in_i] = 0.0


def _set_edge_to_quadratic(layer, out_i: int, in_i: int) -> None:
    """Force edge to approximate a quadratic: a*x^2 (using T_2 = 2x^2-1)."""
    with torch.no_grad():
        layer.cheby_coeffs[out_i, in_i, :] = 0.0
        layer.cheby_coeffs[out_i, in_i, 0] = -0.5   # constant offset from T_2
        layer.cheby_coeffs[out_i, in_i, 2] = 0.5    # T_2(tanh(x)) = 2*tanh(x)^2 - 1
        layer.base_weight[out_i, in_i] = 0.0


def _set_edge_to_zero(layer, out_i: int, in_i: int) -> None:
    """Force an edge to produce near-zero output (should be pruned)."""
    with torch.no_grad():
        layer.cheby_coeffs[out_i, in_i, :] = 1e-8
        layer.base_weight[out_i, in_i] = 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEdgeSampling:
    def test_sample_chebykan_edge_shape(self):
        from src.models.kan_layers import ChebyKANLayer
        from src.interpretability.kan_symbolic import sample_edge

        layer = ChebyKANLayer(in_features=4, out_features=3, degree=3)
        x_vals, y_vals = sample_edge(layer, out_idx=0, in_idx=0, n=200)
        assert x_vals.shape == (200,)
        assert y_vals.shape == (200,)
        assert not np.isnan(y_vals).any()

    def test_sample_chebykan_edge_range(self):
        from src.models.kan_layers import ChebyKANLayer
        from src.interpretability.kan_symbolic import sample_edge

        layer = ChebyKANLayer(in_features=4, out_features=3, degree=3)
        x_vals, _ = sample_edge(layer, out_idx=0, in_idx=0, n=100)
        assert x_vals.min() >= -1.0 - 1e-6
        assert x_vals.max() <= 1.0 + 1e-6

    def test_sample_fourierkan_edge_shape(self):
        from src.models.kan_layers import FourierKANLayer
        from src.interpretability.kan_symbolic import sample_edge

        layer = FourierKANLayer(in_features=4, out_features=3, grid_size=4)
        x_vals, y_vals = sample_edge(layer, out_idx=1, in_idx=2, n=300)
        assert x_vals.shape == (300,)
        assert not np.isnan(y_vals).any()


class TestEdgeVariance:
    def test_zero_edge_has_low_variance(self):
        from src.models.kan_layers import ChebyKANLayer
        from src.interpretability.kan_pruning import _edge_variance_chebykan

        layer = ChebyKANLayer(in_features=3, out_features=2, degree=3)
        _set_edge_to_zero(layer, 0, 0)
        variances = _edge_variance_chebykan(layer)
        assert variances[0, 0].item() < 1e-6

    def test_active_edge_has_nonzero_variance(self):
        from src.models.kan_layers import ChebyKANLayer
        from src.interpretability.kan_pruning import _edge_variance_chebykan

        layer = ChebyKANLayer(in_features=3, out_features=2, degree=3)
        _set_edge_to_identity(layer, 0, 0)
        variances = _edge_variance_chebykan(layer)
        assert variances[0, 0].item() > 1e-4


class TestSymbolicFitting:
    def test_linear_fit_high_r2(self):
        """An identity function should be well-approximated by 'a*x + b'."""
        from src.interpretability.kan_symbolic import _fit_scipy_candidates

        x = np.linspace(-1, 1, 500)
        y = 0.8 * x + 0.1  # simple linear
        formula, r2 = _fit_scipy_candidates(x, y)
        assert r2 > 0.99, f"Expected R²>0.99 for linear, got {r2:.4f} ({formula})"

    def test_quadratic_fit_high_r2(self):
        """A quadratic should be captured by the polynomial candidates."""
        from src.interpretability.kan_symbolic import _fit_scipy_candidates

        x = np.linspace(-1, 1, 500)
        y = 1.5 * x**2 - 0.3 * x + 0.1
        formula, r2 = _fit_scipy_candidates(x, y)
        assert r2 > 0.99, f"Expected R²>0.99 for quadratic, got {r2:.4f} ({formula})"

    def test_constant_fit_r2_is_one(self):
        """A perfectly flat output should give R²=1.0 for the constant candidate."""
        from src.interpretability.kan_symbolic import _fit_scipy_candidates

        x = np.linspace(-1, 1, 200)
        y = np.full_like(x, 3.14)
        formula, r2 = _fit_scipy_candidates(x, y)
        assert r2 >= 0.999, f"Constant function should have R²≈1.0, got {r2:.4f}"


class TestR2Pipeline:
    def test_pipeline_returns_expected_keys(self):
        from src.interpretability.r2_pipeline import evaluate_symbolic_fit

        module = _make_synthetic_chebykan(in_features=3, hidden=4)
        feature_names = ["feat_a", "feat_b", "feat_c"]
        report = evaluate_symbolic_fit(module, feature_names=feature_names, threshold=0.001)

        assert "pruning" in report
        assert "symbolic_fits" in report
        assert "aggregate" in report
        assert "mean_r2" in report["aggregate"]
        assert "median_r2" in report["aggregate"]
        assert "edges_below_090" in report["aggregate"]
        assert "edges_below_095" in report["aggregate"]

    def test_pipeline_detects_zero_edges_as_pruned(self):
        """Edges set to near-zero should not appear in symbolic fits."""
        from src.interpretability.r2_pipeline import evaluate_symbolic_fit
        from src.models.kan_layers import ChebyKANLayer

        module = _make_synthetic_chebykan(in_features=2, hidden=3)
        # Zero out ALL edges in the first KAN layer
        for layer in module.kan_layers:
            if isinstance(layer, ChebyKANLayer):
                with torch.no_grad():
                    layer.cheby_coeffs.fill_(1e-9)
                    layer.base_weight.fill_(0.0)
                break

        report = evaluate_symbolic_fit(module, feature_names=["x0", "x1"], threshold=0.001)
        first_layer_fits = [r for r in report["symbolic_fits"] if r["layer"] == 0]
        assert len(first_layer_fits) == 0, "Zeroed layer-0 edges should all be pruned"

    def test_pipeline_known_quadratic_r2(self):
        """A known-quadratic edge should yield R² close to 1.0."""
        from src.interpretability.r2_pipeline import evaluate_symbolic_fit
        from src.models.kan_layers import ChebyKANLayer

        module = _make_synthetic_chebykan(in_features=2, hidden=2, degree=3)
        kan_layers = [l for l in module.kan_layers if isinstance(l, ChebyKANLayer)]
        assert kan_layers, "No ChebyKANLayer found in synthetic model"

        first_layer = kan_layers[0]
        # Zero all edges, then set one to quadratic
        with torch.no_grad():
            first_layer.cheby_coeffs.fill_(1e-9)
            first_layer.base_weight.fill_(0.0)
        _set_edge_to_quadratic(first_layer, out_i=0, in_i=0)

        report = evaluate_symbolic_fit(
            module,
            feature_names=["x0", "x1"],
            threshold=1e-6,
        )
        fits_layer0 = [r for r in report["symbolic_fits"] if r["layer"] == 0]
        assert fits_layer0, "Expected at least one active edge in layer 0"

        best_r2 = max(r["r_squared"] for r in fits_layer0)
        assert best_r2 > 0.95, (
            f"Known quadratic edge should have R²>0.95, got {best_r2:.4f}"
        )

    def test_pipeline_feature_names_in_layer0(self):
        """Layer-0 symbolic fits should carry the correct input feature names."""
        from src.interpretability.r2_pipeline import evaluate_symbolic_fit

        module = _make_synthetic_chebykan(in_features=3, hidden=4)
        feature_names = ["age", "bmi", "income"]
        report = evaluate_symbolic_fit(module, feature_names=feature_names, threshold=0.0)

        layer0_features = {r["input_feature"] for r in report["symbolic_fits"] if r["layer"] == 0}
        assert layer0_features.issubset(set(feature_names)), (
            f"Layer-0 features {layer0_features} not in {feature_names}"
        )

    def test_sparsity_ratio_in_0_to_1(self):
        from src.interpretability.r2_pipeline import evaluate_symbolic_fit

        module = _make_synthetic_chebykan(in_features=4, hidden=4)
        report = evaluate_symbolic_fit(module, feature_names=list("abcd"), threshold=0.001)
        sparsity = report["pruning"]["sparsity_ratio"]
        assert 0.0 <= sparsity <= 1.0, f"Sparsity must be in [0,1], got {sparsity}"
