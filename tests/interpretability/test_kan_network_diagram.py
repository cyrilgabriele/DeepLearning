"""Tests for kan_network_diagram.py — KAN network visualization."""

import pandas as pd
import pytest
import torch

from src.models.tabkan import TabKAN


@pytest.fixture
def tiny_kan():
    module = TabKAN(in_features=5, widths=[3, 1], kan_type="chebykan", degree=3)
    module.eval()
    return module


@pytest.fixture
def feature_names():
    return [f"feat_{i}" for i in range(5)]


@pytest.fixture
def sample_fits():
    return pd.DataFrame([
        {"layer": 0, "edge_in": 0, "edge_out": 0, "input_feature": "feat_0",
         "formula": "a*x + b", "r_squared": 0.99, "quality_tier": "clean"},
        {"layer": 0, "edge_in": 1, "edge_out": 1, "input_feature": "feat_1",
         "formula": "a*x^2 + b*x + c", "r_squared": 0.95, "quality_tier": "acceptable"},
    ])


class TestDrawKanDiagram:
    def test_produces_pdf(self, tiny_kan, feature_names, sample_fits, tmp_path):
        from src.interpretability.kan_network_diagram import draw_kan_diagram
        result = draw_kan_diagram(
            tiny_kan, feature_names, "chebykan", tmp_path,
            symbolic_fits=sample_fits,
            top_n_inputs=5,
            top_n_hidden=3,
        )
        assert result is not None
        assert result.exists()
        assert result.suffix == ".pdf"

    def test_works_without_symbolic_fits(self, tiny_kan, feature_names, tmp_path):
        from src.interpretability.kan_network_diagram import draw_kan_diagram
        result = draw_kan_diagram(
            tiny_kan, feature_names, "chebykan", tmp_path,
            symbolic_fits=None,
        )
        assert result is not None
        assert result.exists()


class TestBeforeAfterPruning:
    def test_produces_pdf(self, tiny_kan, feature_names, tmp_path):
        import copy
        from src.interpretability.kan_network_diagram import draw_before_after_pruning
        module_before = copy.deepcopy(tiny_kan)
        module_after = copy.deepcopy(tiny_kan)
        # Zero out some edges in the "pruned" version
        with torch.no_grad():
            layer = [l for l in module_after.kan_layers
                     if hasattr(l, "cheby_coeffs")][0]
            layer.cheby_coeffs[0, 0, :] = 0.0
            layer.base_weight[0, 0] = 0.0

        result = draw_before_after_pruning(
            module_before, module_after, feature_names, "chebykan", tmp_path,
        )
        assert result is not None
        assert result.exists()
