from __future__ import annotations

import pandas as pd
import torch

from src.interpretability.closed_form_surrogate import run


class _ThreeInputLinear(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0] + 2.0 * x[:, 1]).unsqueeze(-1)


def test_surrogate_supports_feature_subset_smaller_than_model_input(tmp_path):
    module = _ThreeInputLinear()
    X_eval = pd.DataFrame(
        {
            "feat_a": [0.0, 1.0, 2.0, 3.0],
            "feat_b": [1.0, 0.0, 1.0, 0.0],
            "feat_c": [5.0, 5.0, 5.0, 5.0],
        }
    )

    artifacts = run(
        module,
        X_eval,
        output_dir=tmp_path,
        feature_names=["feat_a", "feat_b"],
        flavor="chebykan",
        y_eval=pd.Series([2, 2, 4, 4], name="Response"),
        ordinal_calibration={
            "method": "optimized_thresholds",
            "thresholds": [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75],
        },
    )

    report = artifacts["report"]
    assert report["feature_names"] == ["feat_a", "feat_b"]
    assert report["fidelity_r2"] == 1.0
    assert report["qwk_metric"] == "optimized_thresholds"
    assert artifacts["json_path"].exists()
    assert artifacts["md_path"].exists()
