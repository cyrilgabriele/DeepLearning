"""Tests for multi-layer coefficient utilities in kan_coefficients.py."""
import pytest
import torch
import torch.nn as nn


class _FakeChebyLayer(nn.Module):
    """Minimal stand-in for ChebyKANLayer."""
    def __init__(self, in_features, out_features, degree):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.ones(out_features, in_features, degree + 1))
        self.base_weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        raise NotImplementedError


class _FakeFourierLayer(nn.Module):
    """Minimal stand-in for FourierKANLayer."""
    def __init__(self, in_features, out_features, grid_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.fourier_a = nn.Parameter(torch.ones(out_features, in_features, grid_size))
        self.fourier_b = nn.Parameter(torch.ones(out_features, in_features, grid_size))
        self.base_weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        raise NotImplementedError


class _FakeLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x)


class _FakeModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.kan_layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return x


@pytest.fixture
def two_layer_cheby_module():
    """Depth-2 ChebyKAN: 4 inputs -> 8 hidden -> 3 outputs."""
    layer0 = _FakeChebyLayer(in_features=4, out_features=8, degree=3)
    layer1 = _FakeChebyLayer(in_features=8, out_features=3, degree=3)
    return _FakeModule([layer0, layer1])


@pytest.fixture
def mixed_module():
    """Module with non-KAN layer sandwiched between KAN layers."""
    layer0 = _FakeChebyLayer(in_features=4, out_features=8, degree=3)
    linear = _FakeLinear(8, 8)
    layer1 = _FakeFourierLayer(in_features=8, out_features=3, grid_size=4)
    return _FakeModule([layer0, linear, layer1])


# -- get_all_kan_layers -------------------------------------------------------

def test_get_all_kan_layers_returns_both_layers(two_layer_cheby_module, monkeypatch):
    import src.interpretability.utils.kan_coefficients as kc
    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer, _FakeFourierLayer))
    layers = kc.get_all_kan_layers(two_layer_cheby_module)
    assert len(layers) == 2


def test_get_all_kan_layers_skips_non_kan(mixed_module, monkeypatch):
    import src.interpretability.utils.kan_coefficients as kc
    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer, _FakeFourierLayer))
    layers = kc.get_all_kan_layers(mixed_module)
    assert len(layers) == 2


# -- coefficient_importance_all_layers ----------------------------------------

def test_all_layers_returns_two_layer_frames(two_layer_cheby_module, monkeypatch):
    import src.interpretability.utils.kan_coefficients as kc
    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer, _FakeFourierLayer))
    feature_names = ["f0", "f1", "f2", "f3"]
    df = kc.coefficient_importance_all_layers(two_layer_cheby_module, feature_names)
    assert set(df["layer"].unique()) == {0, 1}


def test_layer1_features_labelled_as_hidden(two_layer_cheby_module, monkeypatch):
    import src.interpretability.utils.kan_coefficients as kc
    monkeypatch.setattr(kc, "_KAN_LAYER_TYPES", (_FakeChebyLayer, _FakeFourierLayer))
    feature_names = ["f0", "f1", "f2", "f3"]
    df = kc.coefficient_importance_all_layers(two_layer_cheby_module, feature_names)
    layer1_feats = df[df["layer"] == 1]["feature"].tolist()
    assert all(f.startswith("h") for f in layer1_feats)
    assert "h0" in layer1_feats
    assert "h7" in layer1_feats


def test_draw_pruned_network_graph_handles_fewer_features_than_top_n(tmp_path, monkeypatch):
    """draw_pruned_network_graph must not crash when top_n_inputs > actual feature count."""
    import torch
    import torch.nn as nn
    from src.interpretability.final_comparison import draw_pruned_network_graph
    from src.models.kan_layers import ChebyKANLayer

    layer0 = ChebyKANLayer(in_features=3, out_features=4, degree=2)
    layer1 = ChebyKANLayer(in_features=4, out_features=2, degree=2)

    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.kan_layers = nn.ModuleList([layer0, layer1])

    module = _Module()
    feature_names = ["feat_a", "feat_b", "feat_c"]

    import src.interpretability.kan_pruning as kp
    def fake_edge_l1(layer):
        if layer is layer0:
            return torch.ones(4, 3) * 0.5
        return torch.ones(2, 4) * 0.3
    monkeypatch.setattr(kp, "_compute_edge_l1", fake_edge_l1)

    import src.interpretability.utils.style as style_mod
    monkeypatch.setattr(style_mod, "savefig_pdf", lambda fig, path: None)

    draw_pruned_network_graph(module, feature_names, "ChebyKAN", tmp_path, top_n_inputs=15)
