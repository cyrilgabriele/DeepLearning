import pytest
import numpy as np
import pandas as pd


def test_binary_dot_values_shap():
    """SHAP dot values should be mean SHAP per encoded class group."""
    from src.interpretability.feature_risk_influence import _binary_dot_values_shap

    shap_vals = pd.Series([0.1, 0.2, -0.3, -0.4, 0.05])
    enc = pd.Series([-1.0, -1.0, 1.0, 1.0, -1.0])
    neg_val, pos_val = _binary_dot_values_shap(shap_vals, enc)
    assert neg_val == pytest.approx((0.1 + 0.2 + 0.05) / 3, abs=1e-6)
    assert pos_val == pytest.approx((-0.3 + -0.4) / 2, abs=1e-6)
