import pytest
import torch
import numpy as np
import torch.nn as nn


def _make_chebykan_layer(in_f=5, out_f=3, degree=3):
    from src.models.kan_layers import ChebyKANLayer
    return ChebyKANLayer(in_features=in_f, out_features=out_f, degree=degree)


def test_top_features_by_l1():
    """Top features should be ranked by sum of first-layer edge L1 norms."""
    from src.interpretability.kan_symbolic import _top_features_by_l1

    l1_scores = torch.tensor([[0.1, 0.5, 0.2], [0.1, 0.4, 0.1]])
    names = ["low", "high", "medium"]
    result = _top_features_by_l1(l1_scores, names, top_n=2)
    assert result[0] == "high"
    assert result[1] == "medium"


# ── lock_in_symbolic_edges ────────────────────────────────────────────────

import copy
import pandas as pd


def test_lock_in_replaces_coefficients_for_clean_edges(monkeypatch):
    """lock_in_symbolic_edges must overwrite cheby_coeffs for clean edges."""
    import src.interpretability.kan_symbolic as ks
    from src.models.kan_layers import ChebyKANLayer

    # Use a real ChebyKANLayer (small, no model weights needed)
    layer = ChebyKANLayer(in_features=2, out_features=2, degree=3)

    class _Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.kan_layers = nn.ModuleList([layer])

    module = _Mod()

    n = 200
    x_norm = np.tanh(np.linspace(-3, 3, n)).astype(np.float32)
    monkeypatch.setattr(ks, "sample_edge", lambda l, o, i, n=1000: (x_norm, x_norm))

    fits_df = pd.DataFrame([{
        "layer": 0, "edge_out": 0, "edge_in": 0,
        "formula": "a*x + b", "r_squared": 0.999, "quality_tier": "clean",
    }])

    orig_coeffs = layer.cheby_coeffs[0, 0, :].detach().clone()
    symbolified, log = ks.lock_in_symbolic_edges(module, fits_df, n_samples=n)

    new_coeffs = list(symbolified.kan_layers)[0].cheby_coeffs[0, 0, :].detach()
    assert len(log) == 1, "Exactly one edge should be locked in"
    assert not torch.allclose(orig_coeffs, new_coeffs, atol=1e-4), \
        "Coefficients should change after lock-in"
    assert float(list(symbolified.kan_layers)[0].base_weight[0, 0]) == pytest.approx(0.0), \
        "base_weight must be zeroed for the locked edge"


def test_lock_in_skips_non_clean_edges(monkeypatch):
    """Edges with quality_tier != 'clean' must not be modified."""
    import src.interpretability.kan_symbolic as ks
    from src.models.kan_layers import ChebyKANLayer

    layer = ChebyKANLayer(in_features=2, out_features=2, degree=3)

    class _Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.kan_layers = nn.ModuleList([layer])

    module = _Mod()

    n = 200
    x_norm = np.tanh(np.linspace(-3, 3, n)).astype(np.float32)
    monkeypatch.setattr(ks, "sample_edge", lambda l, o, i, n=1000: (x_norm, x_norm))

    fits_df = pd.DataFrame([{
        "layer": 0, "edge_out": 0, "edge_in": 0,
        "formula": "a*x + b", "r_squared": 0.85, "quality_tier": "flagged",
    }])

    _, log = ks.lock_in_symbolic_edges(module, fits_df, n_samples=n)
    assert len(log) == 0, "No edges should be locked for non-clean quality"
