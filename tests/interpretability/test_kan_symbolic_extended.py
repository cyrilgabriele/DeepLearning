import pytest
import torch


def _make_chebykan_layer(in_f=5, out_f=3, degree=3):
    from src.models.kan_layers import ChebyKANLayer
    return ChebyKANLayer(in_features=in_f, out_features=out_f, degree=degree)


def test_top_features_by_variance():
    """Top features should be ranked by sum of first-layer edge variances."""
    from src.interpretability.kan_symbolic import _top_features_by_variance

    variances = torch.tensor([[0.1, 0.5, 0.2], [0.1, 0.4, 0.1]])
    names = ["low", "high", "medium"]
    result = _top_features_by_variance(variances, names, top_n=2)
    assert result[0] == "high"
    assert result[1] == "medium"
