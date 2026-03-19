import pytest
import pandas as pd


def test_select_top_features_includes_binary_and_continuous():
    """Top-feature selection must guarantee at least 2 continuous + 2 binary."""
    from src.interpretability.comparison_side_by_side import _select_top_features

    feat_types = {
        "BMI": "continuous", "Age": "continuous", "Height": "continuous",
        "KW_1": "binary", "KW_2": "binary", "KW_3": "binary",
    }
    glm = pd.DataFrame({
        "feature": ["KW_1", "KW_2", "KW_3", "BMI", "Age", "Height"],
        "abs_magnitude": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    })
    shap = pd.DataFrame({
        "KW_1": [0.5], "KW_2": [0.4], "KW_3": [0.3],
        "BMI": [0.2], "Age": [0.15], "Height": [0.1],
    })
    sym = pd.DataFrame({
        "layer": [0, 0, 0],
        "input_feature": ["KW_1", "KW_2", "BMI"],
        "r_squared": [0.95, 0.90, 0.85],
    })
    result = _select_top_features(glm, shap, sym, feat_types, n=5)
    cont = [f for f in result if feat_types.get(f) in ("continuous", "ordinal")]
    binary = [f for f in result if feat_types.get(f) in ("binary", "missing_indicator")]
    assert len(cont) >= 2, f"Expected >=2 continuous in {result}"
    assert len(binary) >= 2, f"Expected >=2 binary in {result}"
