"""Tests for quality_figures.py — R² distribution and pruning Pareto."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.tabkan import TabKAN


@pytest.fixture
def sample_fits():
    """Symbolic fits with a mix of quality tiers."""
    return pd.DataFrame({
        "layer": [0] * 10,
        "edge_in": list(range(10)),
        "edge_out": [0] * 10,
        "input_feature": [f"f{i}" for i in range(10)],
        "formula": ["a*x + b"] * 10,
        "r_squared": [0.99, 0.995, 0.85, 0.92, 0.88, 0.996, 0.75, 0.93, 0.991, 0.50],
        "quality_tier": ["clean", "clean", "flagged", "acceptable", "flagged",
                         "clean", "flagged", "acceptable", "clean", "flagged"],
    })


class TestPlotR2Distribution:
    def test_produces_pdf(self, sample_fits, tmp_path):
        from src.interpretability.quality_figures import plot_r2_distribution
        result = plot_r2_distribution(sample_fits, "chebykan", tmp_path)
        assert result.exists()
        assert result.suffix == ".pdf"

    def test_handles_all_clean(self, tmp_path):
        from src.interpretability.quality_figures import plot_r2_distribution
        df = pd.DataFrame({
            "r_squared": [0.995, 0.999, 0.991],
            "quality_tier": ["clean"] * 3,
        })
        result = plot_r2_distribution(df, "fourierkan", tmp_path)
        assert result.exists()

    def test_handles_all_flagged(self, tmp_path):
        from src.interpretability.quality_figures import plot_r2_distribution
        df = pd.DataFrame({
            "r_squared": [0.5, 0.3, 0.7],
            "quality_tier": ["flagged"] * 3,
        })
        result = plot_r2_distribution(df, "chebykan", tmp_path)
        assert result.exists()


class TestComputePruningPareto:
    def test_returns_list_of_dicts(self, tmp_path):
        from src.interpretability.quality_figures import compute_pruning_pareto

        module = TabKAN(in_features=5, widths=[3, 1], kan_type="chebykan", degree=3)
        module.eval()

        # Create minimal eval data
        X = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({"Response": np.random.randint(1, 9, 20)})
        x_path = tmp_path / "X_eval.parquet"
        y_path = tmp_path / "y_eval.parquet"
        X.to_parquet(x_path)
        y.to_parquet(y_path)

        results = compute_pruning_pareto(
            module, None, "chebykan", x_path, y_path,
            thresholds=[0.01, 0.1],
        )
        assert len(results) == 2
        for r in results:
            assert "threshold" in r
            assert "sparsity" in r
            assert "qwk" in r
            assert 0 <= r["sparsity"] <= 1


class TestPlotPruningPareto:
    def test_produces_pdf(self, tmp_path):
        from src.interpretability.quality_figures import plot_pruning_pareto
        data = {
            "chebykan": [
                {"threshold": 0.01, "sparsity": 0.2, "qwk": 0.58},
                {"threshold": 0.1, "sparsity": 0.7, "qwk": 0.45},
            ],
            "fourierkan": [
                {"threshold": 0.01, "sparsity": 0.15, "qwk": 0.57},
                {"threshold": 0.1, "sparsity": 0.65, "qwk": 0.44},
            ],
        }
        result = plot_pruning_pareto(data, tmp_path)
        assert result.exists()
        assert result.suffix == ".pdf"
