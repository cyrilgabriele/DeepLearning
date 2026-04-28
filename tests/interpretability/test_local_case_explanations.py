import pandas as pd
import torch

from src.interpretability.local_case_explanations import run
from src.models.tabkan import TabKAN


def test_local_case_explanations_smoke(tmp_path):
    module = TabKAN(in_features=3, widths=[2, 1], kan_type="chebykan", degree=2, use_layernorm=False)
    with torch.no_grad():
        for layer in module.kan_layers:
            if hasattr(layer, "cheby_coeffs"):
                layer.cheby_coeffs.zero_()
                layer.base_weight.zero_()
        first_layer = module.kan_layers[0]
        second_layer = module.kan_layers[1]
        first_layer.base_weight[0, 0] = 0.8
        first_layer.base_weight[1, 1] = -0.4
        second_layer.base_weight[0, 0] = 0.5
        second_layer.base_weight[0, 1] = 0.3
        module.head.weight.zero_()
        module.head.weight[0, 0] = 1.0
        module.head.bias[0] = 0.2

    X_eval = pd.DataFrame(
        {
            "BMI": [21.0, 25.0, 31.0, 36.0],
            "Medical_Keyword_1": [0.0, 1.0, 0.0, 1.0],
            "Product_Info_2": [0.0, 1.0, 2.0, 1.0],
        }
    )
    X_eval_raw = pd.DataFrame(
        {
            "Id": [1001, 1002, 1003, 1004],
            "BMI": [21.0, 25.0, 31.0, 36.0],
            "Medical_Keyword_1": [0, 1, 0, 1],
            "Product_Info_2": ["A", "B", "C", "B"],
        }
    )
    feature_types = {
        "BMI": "continuous",
        "Medical_Keyword_1": "binary",
        "Product_Info_2": "categorical",
    }

    artifacts = run(
        module,
        X_eval,
        output_dir=tmp_path,
        flavor="chebykan",
        feature_types=feature_types,
        X_eval_raw=X_eval_raw,
        candidate_features=["BMI", "Medical_Keyword_1", "Product_Info_2"],
        row_position=0,
        ordinal_calibration={
            "method": "optimized_thresholds",
            "thresholds": [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75],
        },
    )

    assert artifacts["case_summary"].exists()
    assert artifacts["local_sensitivities"].exists()
    assert artifacts["what_if"].exists()

    sensitivities = pd.read_csv(artifacts["local_sensitivities"])
    what_if = pd.read_csv(artifacts["what_if"])
    assert {"feature", "contribution_vs_reference"}.issubset(sensitivities.columns)
    assert {"feature", "scenario", "output_delta", "class_mapping"}.issubset(what_if.columns)
    assert set(what_if["class_mapping"]) == {"optimized_thresholds"}
    assert len(what_if) >= 3
