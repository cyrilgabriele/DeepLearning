import pytest
import torch


def test_kan_importance_uses_edge_variance():
    """KAN feature importance should be sum of edge variances per input feature."""
    from src.interpretability.comparison_per_risk import _kan_importance_from_variance

    # variances (out=2, in=2): feature 0 has high variance, feature 1 low
    variances = torch.tensor([[0.5, 0.01], [0.4, 0.02]])
    feature_names = ["high_impact", "low_impact"]
    result = _kan_importance_from_variance(variances, feature_names)
    assert result["high_impact"] > result["low_impact"]
    assert result["high_impact"] == pytest.approx(0.9, abs=1e-5)
    assert result["low_impact"] == pytest.approx(0.03, abs=1e-5)
