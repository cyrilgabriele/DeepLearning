import torch
import pytest
from src.models.mlp import MLPBaseline


class TestMLPBaseline:
    def test_forward_shape(self):
        model = MLPBaseline(in_features=20, widths=[64, 32])
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_single_layer(self):
        model = MLPBaseline(in_features=10, widths=[32])
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 1)

    def test_with_dropout(self):
        model = MLPBaseline(in_features=10, widths=[32, 16], dropout=0.5)
        model.eval()
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 1)

    def test_training_step(self):
        model = MLPBaseline(in_features=10, widths=[16])
        batch = (torch.randn(4, 10), torch.randn(4, 1))
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
