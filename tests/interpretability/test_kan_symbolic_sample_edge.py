import torch
import numpy as np


def _make_chebykan_layer(in_f=3, out_f=2, degree=3):
    from src.models.kan_layers import ChebyKANLayer
    layer = ChebyKANLayer(in_features=in_f, out_features=out_f, degree=degree)
    return layer


def test_sample_edge_returns_xnorm_in_tanh_range():
    """After patch: sample_edge must return x_norm = tanh(linspace(-3,3,n))."""
    from src.interpretability.kan_symbolic import sample_edge
    layer = _make_chebykan_layer()
    x_vals, _ = sample_edge(layer, out_idx=0, in_idx=0, n=100)
    # x_norm = tanh(linspace(-3,3)) ≈ [-0.995, +0.995]
    assert x_vals.min() < -0.99, f"Expected x_min < -0.99, got {x_vals.min()}"
    assert x_vals.max() > 0.99, f"Expected x_max > 0.99, got {x_vals.max()}"
    # Must not exceed tanh range
    assert x_vals.min() > -1.0 and x_vals.max() < 1.0
