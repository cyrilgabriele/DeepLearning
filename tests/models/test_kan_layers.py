import torch
import pytest
from src.models.kan_layers import ChebyKANLayer, FourierKANLayer, BSplineKANLayer


class TestChebyKANLayer:
    def test_output_shape(self):
        layer = ChebyKANLayer(in_features=10, out_features=5, degree=3)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)

    def test_degree_one(self):
        layer = ChebyKANLayer(in_features=4, out_features=3, degree=1)
        x = torch.randn(8, 4)
        out = layer(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        layer = ChebyKANLayer(in_features=6, out_features=2, degree=4)
        x = torch.randn(4, 6, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters())

    def test_input_clamping(self):
        layer = ChebyKANLayer(in_features=3, out_features=2, degree=3)
        x = torch.tensor([[10.0, -10.0, 5.0]])
        out = layer(x)
        assert torch.isfinite(out).all()


class TestFourierKANLayer:
    def test_output_shape(self):
        layer = FourierKANLayer(in_features=10, out_features=5, grid_size=4)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)

    def test_grid_size_one(self):
        layer = FourierKANLayer(in_features=4, out_features=3, grid_size=1)
        x = torch.randn(8, 4)
        out = layer(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        layer = FourierKANLayer(in_features=6, out_features=2, grid_size=5)
        x = torch.randn(4, 6, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters())

    def test_handles_large_input(self):
        layer = FourierKANLayer(in_features=3, out_features=2, grid_size=4)
        x = torch.tensor([[100.0, -100.0, 50.0]])
        out = layer(x)
        assert torch.isfinite(out).all()


class TestBSplineKANLayer:
    def test_output_shape(self):
        layer = BSplineKANLayer(in_features=10, out_features=5, grid_size=5, spline_order=3)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)

    def test_small_grid(self):
        layer = BSplineKANLayer(in_features=4, out_features=3, grid_size=3, spline_order=2)
        x = torch.randn(8, 4)
        out = layer(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        layer = BSplineKANLayer(in_features=6, out_features=2, grid_size=5, spline_order=3)
        x = torch.randn(4, 6, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in layer.parameters() if p.requires_grad)

    def test_handles_large_input(self):
        layer = BSplineKANLayer(in_features=3, out_features=2, grid_size=5, spline_order=3)
        x = torch.tensor([[100.0, -100.0, 50.0]])
        out = layer(x)
        assert torch.isfinite(out).all()
