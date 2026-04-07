"""Verify base_weight is applied to raw x (not tanh(x)) in edge sampling functions."""
import numpy as np
import torch
import torch.nn as nn


class _LinearChebyLayer(nn.Module):
    """ChebyKAN layer where cheby_coeffs=0 so output = base_weight * x (raw)."""
    def __init__(self):
        super().__init__()
        self.in_features = 1
        self.out_features = 1
        self.degree = 3
        self.cheby_coeffs = nn.Parameter(torch.zeros(1, 1, 4))
        self.base_weight = nn.Parameter(torch.ones(1, 1))  # slope = 1.0

    def forward(self, x):
        raise NotImplementedError


class _LinearFourierLayer(nn.Module):
    """FourierKAN layer where fourier_a=fourier_b=0 so output = base_weight * x (raw)."""
    def __init__(self):
        super().__init__()
        self.in_features = 1
        self.out_features = 1
        self.grid_size = 4
        self.fourier_a = nn.Parameter(torch.zeros(1, 1, 4))
        self.fourier_b = nn.Parameter(torch.zeros(1, 1, 4))
        self.base_weight = nn.Parameter(torch.ones(1, 1))  # slope = 1.0

    def forward(self, x):
        raise NotImplementedError


def test_sample_chebykan_edge_uses_raw_x_for_base_weight():
    """With cheby_coeffs=0 and base_weight=1, y must equal raw x (not tanh(x))."""
    from src.interpretability.kan_symbolic import _sample_chebykan_edge

    layer = _LinearChebyLayer()
    x_norm, y = _sample_chebykan_edge(layer, out_idx=0, in_idx=0, n=200)

    # raw x = arctanh(x_norm); at tails |x_norm|>0.9, raw x >> x_norm
    x_raw = np.arctanh(np.clip(x_norm, -0.9999, 0.9999))
    tail = np.abs(x_norm) > 0.9
    assert tail.sum() > 5

    # y should equal x_raw at the tails (not x_norm)
    np.testing.assert_allclose(
        y[tail], x_raw[tail], rtol=0.02,
        err_msg="base_weight must multiply raw x, not tanh(x), in _sample_chebykan_edge"
    )


def test_sample_fourierkan_edge_uses_raw_x_for_base_weight():
    """With fourier_a=fourier_b=0 and base_weight=1, y must equal raw x."""
    from src.interpretability.kan_symbolic import _sample_fourierkan_edge

    layer = _LinearFourierLayer()
    x_norm, y = _sample_fourierkan_edge(layer, out_idx=0, in_idx=0, n=200)

    x_raw = np.arctanh(np.clip(x_norm, -0.9999, 0.9999))
    tail = np.abs(x_norm) > 0.9
    assert tail.sum() > 5

    np.testing.assert_allclose(
        y[tail], x_raw[tail], rtol=0.02,
        err_msg="base_weight must multiply raw x, not tanh(x), in _sample_fourierkan_edge"
    )


def test_sample_feature_function_uses_raw_x_for_base_weight(monkeypatch):
    """With cheby_coeffs=0 and base_weight=1, sample_feature_function y must equal raw x."""
    import src.interpretability.utils.kan_coefficients as kc

    layer = _LinearChebyLayer()
    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_LinearChebyLayer,))

    x_norm, y_feature, _ = kc.sample_feature_function(layer, feature_idx=0, n=200)

    x_raw = np.arctanh(np.clip(x_norm, -0.9999, 0.9999))
    tail = np.abs(x_norm) > 0.9
    assert tail.sum() > 5

    # y_feature is the mean over out_features (just 1 here), should equal raw x
    np.testing.assert_allclose(
        y_feature[tail], x_raw[tail], rtol=0.02,
        err_msg="base_weight must multiply raw x, not tanh(x), in sample_feature_function"
    )
