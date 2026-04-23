"""Tests for the ChebyKAN-native per-edge formula extractor.

The native path reads ``cheby_coeffs`` and ``base_weight`` directly from the
layer and returns the exact symbolic form for one edge. R² against the layer's
actual forward pass must be ≈ 1 because the formula IS the edge.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import torch

from src.models.kan_layers import ChebyKANLayer


def _isolate_edge(layer: ChebyKANLayer, out_idx: int, in_idx: int) -> None:
    """Zero every weight feeding ``out_idx`` except the (out_idx, in_idx) edge.

    This makes the layer output at ``out_idx`` depend only on the target edge,
    so we can compare the extracted edge formula against the layer forward pass.
    """
    with torch.no_grad():
        for j in range(layer.in_features):
            if j == in_idx:
                continue
            layer.cheby_coeffs[out_idx, j, :] = 0.0
            layer.base_weight[out_idx, j] = 0.0


def test_native_fit_matches_layer_forward_on_isolated_edge():
    torch.manual_seed(0)
    layer = ChebyKANLayer(in_features=3, out_features=2, degree=4)
    out_idx, in_idx = 1, 0
    with torch.no_grad():
        layer.cheby_coeffs[out_idx, in_idx, :] = torch.tensor([0.5, 0.25, 0.1, -0.05, 0.02])
        layer.base_weight[out_idx, in_idx] = 0.3
    _isolate_edge(layer, out_idx, in_idx)

    from src.interpretability.kan_symbolic import fit_symbolic_edge_chebykan_native

    formula_str, r2 = fit_symbolic_edge_chebykan_native(layer, out_idx=out_idx, in_idx=in_idx)

    assert r2 >= 0.9999, f"Expected exact fit, got r2={r2}"

    x_sym = sp.Symbol("x")
    expr = sp.sympify(formula_str, locals={"x": x_sym})
    f = sp.lambdify(x_sym, expr, modules=["numpy"])

    x_test = np.array([-2.0, -0.75, -0.1, 0.0, 0.1, 0.75, 2.0], dtype=np.float64)
    x_tensor = torch.zeros(len(x_test), layer.in_features)
    x_tensor[:, in_idx] = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        y_layer = layer(x_tensor)[:, out_idx].cpu().numpy().astype(np.float64)

    y_formula = np.asarray(f(x_test), dtype=np.float64)

    np.testing.assert_allclose(y_formula, y_layer, atol=1e-4)


def test_build_edge_records_is_native_for_chebykan_module():
    """All records from a ChebyKAN module must be native with R² ≈ 1 and quality_tier 'clean'."""
    from src.interpretability.kan_symbolic import _build_edge_records
    from src.models.tabkan import TabKAN

    torch.manual_seed(42)
    module = TabKAN(
        in_features=4,
        widths=[3, 2],
        kan_type="chebykan",
        degree=3,
        use_layernorm=False,
    )
    module.eval()

    feature_names = [f"f{i}" for i in range(4)]
    records = _build_edge_records(module, threshold=0.0, feature_names=feature_names)

    assert len(records) > 0
    assert all(r["fit_mode"] == "chebykan_native" for r in records)
    assert all(r["r_squared"] >= 0.9999 for r in records), (
        f"Min r²={min(r['r_squared'] for r in records)}"
    )
    assert all(r["quality_tier"] == "clean" for r in records)
    # Layer 0 edges record the input feature name
    layer0 = [r for r in records if r["layer"] == 0]
    assert all(r["input_feature"] in feature_names for r in layer0)


def test_r2_pipeline_uses_native_path_for_chebykan():
    """evaluate_symbolic_fit must report R² ≈ 1 for all ChebyKAN edges via the native path."""
    from src.interpretability.r2_pipeline import evaluate_symbolic_fit
    from src.models.tabkan import TabKAN

    torch.manual_seed(7)
    module = TabKAN(
        in_features=4,
        widths=[3, 2],
        kan_type="chebykan",
        degree=3,
        use_layernorm=False,
    )
    module.eval()

    feature_names = [f"f{i}" for i in range(4)]
    report = evaluate_symbolic_fit(module, feature_names=feature_names, threshold=0.0)

    assert report["pruning"]["edges_after"] > 0
    assert report["aggregate"]["mean_r2"] >= 0.9999
    assert report["aggregate"]["edges_flagged"] == 0
    assert all(r["fit_mode"] == "chebykan_native" for r in report["symbolic_fits"])


def test_native_fit_zero_edge_returns_constant_zero():
    """An edge with all-zero coefficients and zero base_weight must fit to 0."""
    layer = ChebyKANLayer(in_features=2, out_features=1, degree=3)
    with torch.no_grad():
        layer.cheby_coeffs.zero_()
        layer.base_weight.zero_()

    from src.interpretability.kan_symbolic import fit_symbolic_edge_chebykan_native

    formula_str, r2 = fit_symbolic_edge_chebykan_native(layer, out_idx=0, in_idx=0)

    # r² is undefined for a constant target, but the fitter should return a
    # meaningful (1.0 or 1.0-by-convention) value rather than NaN/inf.
    assert np.isfinite(r2)
    expr = sp.sympify(formula_str, locals={"x": sp.Symbol("x")})
    assert sp.simplify(expr) == 0
