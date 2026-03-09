import torch
import pytest
from src.models.tabkan import TabKAN


class TestTabKAN:
    def test_chebykan_forward(self):
        model = TabKAN(
            in_features=20, widths=[32, 16], kan_type="chebykan", degree=3
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_fourierkan_forward(self):
        model = TabKAN(
            in_features=20, widths=[32, 16], kan_type="fourierkan", grid_size=4
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 1)

    def test_single_layer(self):
        model = TabKAN(in_features=10, widths=[64], kan_type="chebykan", degree=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_three_layers(self):
        model = TabKAN(
            in_features=50, widths=[128, 64, 32], kan_type="fourierkan", grid_size=5
        )
        x = torch.randn(4, 50)
        out = model(x)
        assert out.shape == (4, 1)

    def test_training_step(self):
        model = TabKAN(in_features=10, widths=[16], kan_type="chebykan", degree=3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        batch = (x, y)
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_validation_step(self):
        model = TabKAN(in_features=10, widths=[16], kan_type="chebykan", degree=3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        batch = (x, y)
        loss = model.validation_step(batch, 0)
        assert loss.ndim == 0

    def test_invalid_kan_type_raises(self):
        with pytest.raises(ValueError, match="Unknown kan_type"):
            TabKAN(in_features=10, widths=[16], kan_type="invalidkan")
