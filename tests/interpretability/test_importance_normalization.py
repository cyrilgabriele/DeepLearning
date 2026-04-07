"""Normalized KAN importance scores must be comparable across architectures."""
import pytest
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class _FakeChebyLayer4(nn.Module):
    """degree=3 → 4 basis terms. All coefficients = 1.0."""
    def __init__(self):
        super().__init__()
        self.in_features = 2
        self.out_features = 4
        self.degree = 3
        self.cheby_coeffs = nn.Parameter(torch.ones(4, 2, 4))
        self.base_weight = nn.Parameter(torch.zeros(4, 2))

    def forward(self, x):
        raise NotImplementedError


class _FakeFourierLayer8(nn.Module):
    """grid_size=4 → 8 basis terms. All coefficients = 1.0."""
    def __init__(self):
        super().__init__()
        self.in_features = 2
        self.out_features = 4
        self.grid_size = 4
        self.fourier_a = nn.Parameter(torch.ones(4, 2, 4))
        self.fourier_b = nn.Parameter(torch.ones(4, 2, 4))
        self.base_weight = nn.Parameter(torch.zeros(4, 2))

    def forward(self, x):
        raise NotImplementedError


def test_normalized_scores_are_equal_when_coefficients_identical(monkeypatch, tmp_path):
    """With identical coefficient values, normalized scores must be equal across architectures."""
    import src.interpretability.utils.kan_coefficients as kc
    from src.interpretability.comparison_per_risk import _kan_importance_global

    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer4, _FakeFourierLayer8))

    cheby_layer = _FakeChebyLayer4()
    fourier_layer = _FakeFourierLayer8()
    feature_names = ["f0", "f1"]

    sym_path = tmp_path / "dummy.csv"
    sym_path.write_text("layer,edge_in,edge_out,input_feature\n")

    cheby_scores = _kan_importance_global(sym_path, feature_names, cheby_layer)
    fourier_scores = _kan_importance_global(sym_path, feature_names, fourier_layer)

    # After normalization, scores should be equal (both have all-ones coefficients)
    np.testing.assert_allclose(
        cheby_scores.values,
        fourier_scores.values,
        rtol=1e-5,
        err_msg="Normalized scores must be equal when all coefficients are 1.0",
    )


def test_raw_scores_differ_before_normalization(monkeypatch, tmp_path):
    """Without normalization, FourierKAN raw score would be ~2x ChebyKAN (8 vs 4 basis terms)."""
    import src.interpretability.utils.kan_coefficients as kc

    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer4, _FakeFourierLayer8))

    cheby_layer = _FakeChebyLayer4()
    fourier_layer = _FakeFourierLayer8()
    feature_names = ["f0", "f1"]

    cheby_frame = kc.coefficient_importance_from_layer(cheby_layer, feature_names)
    fourier_frame = kc.coefficient_importance_from_layer(fourier_layer, feature_names)

    cheby_raw = cheby_frame["importance"].values
    fourier_raw = fourier_frame["importance"].values

    # FourierKAN has 8 basis terms vs ChebyKAN's 4, so raw score is ~2x
    ratio = fourier_raw[0] / cheby_raw[0]
    assert abs(ratio - 2.0) < 0.01, f"Expected ~2x ratio, got {ratio:.3f}"
