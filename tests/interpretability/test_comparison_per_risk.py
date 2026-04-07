"""Tests for comparison_per_risk helpers."""
import pytest
import torch
import torch.nn as nn
import pandas as pd


class _FakeChebyLayer(nn.Module):
    """Fake ChebyKAN layer with controllable importance values."""
    def __init__(self):
        super().__init__()
        self.in_features = 3
        self.out_features = 4
        self.degree = 2
        # Feature 0 has large coefficients, feature 2 has small
        data = torch.zeros(4, 3, 3)
        data[:, 0, :] = 2.0   # feature 0 — high importance
        data[:, 1, :] = 0.5   # feature 1 — medium
        data[:, 2, :] = 0.1   # feature 2 — low
        self.cheby_coeffs = nn.Parameter(data)
        self.base_weight = nn.Parameter(torch.zeros(4, 3))

    def forward(self, x):
        raise NotImplementedError


def test_kan_importance_global_ranks_correctly(monkeypatch, tmp_path):
    """_kan_importance_global must rank features by coefficient magnitude."""
    import src.interpretability.utils.kan_coefficients as kc
    from src.interpretability.comparison_per_risk import _kan_importance_global

    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer,))

    layer = _FakeChebyLayer()
    feature_names = ["high", "medium", "low"]
    sym_path = tmp_path / "dummy_symbolic.csv"
    sym_path.write_text("layer,edge_in,edge_out,input_feature\n0,0,0,high\n")

    result = _kan_importance_global(sym_path, feature_names, layer)

    assert isinstance(result, pd.Series)
    assert result.index[0] == "high",  f"Expected 'high' first, got {result.index[0]}"
    assert result.index[-1] == "low",  f"Expected 'low' last, got {result.index[-1]}"


def test_kan_importance_global_falls_back_to_edge_count_when_no_layer(tmp_path):
    """Without a model layer, _kan_importance_global falls back to edge count."""
    from src.interpretability.comparison_per_risk import _kan_importance_global

    sym_path = tmp_path / "sym.csv"
    sym_path.write_text(
        "layer,edge_in,edge_out,input_feature\n"
        "0,0,0,alpha\n0,1,0,alpha\n0,0,0,beta\n"
    )
    result = _kan_importance_global(sym_path, ["alpha", "beta"], kan_layer=None)

    assert result["alpha"] > result["beta"], "alpha has 2 edges vs beta's 1"
