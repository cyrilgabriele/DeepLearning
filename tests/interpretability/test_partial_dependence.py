import numpy as np
import pandas as pd
import torch


class _SingleFeatureModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


def test_build_feature_grid_uses_observed_states_for_discrete_inputs():
    from src.interpretability.utils.style import build_feature_grid, resolve_feature_display_spec

    X_eval = pd.DataFrame({"Medical_Keyword_3": [0.0, 1.0, 0.0, 1.0]})
    spec = resolve_feature_display_spec(
        "Medical_Keyword_3",
        feat_types={"Medical_Keyword_3": "binary"},
        preprocessing_recipe="kan_paper",
    )

    grid = build_feature_grid(spec, X_eval, grid_resolution=100, percentile_range=(1.0, 99.0))

    np.testing.assert_allclose(grid, np.array([0.0, 1.0]))


def test_compute_partial_dependence_respects_explicit_discrete_grid():
    from src.interpretability.partial_dependence import compute_partial_dependence

    X_eval = pd.DataFrame({"Medical_History_23": [1.0, 3.0, 1.0, 3.0]})
    module = _SingleFeatureModel()
    grid = np.array([1.0, 3.0])

    returned_grid, avg_preds = compute_partial_dependence(
        module,
        X_eval,
        "Medical_History_23",
        grid_values=grid,
    )

    np.testing.assert_allclose(returned_grid, grid)
    np.testing.assert_allclose(avg_preds, grid)
