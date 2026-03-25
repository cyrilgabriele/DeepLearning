import pytest
import numpy as np
import pandas as pd


def test_mask_features_zeros_out_dropped_columns():
    from src.interpretability.final_comparison import _mask_features
    X = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    result = _mask_features(X, keep=["A", "C"])
    assert list(result["A"]) == [1.0, 2.0]
    assert list(result["B"]) == [0.0, 0.0]
    assert list(result["C"]) == [5.0, 6.0]
    assert list(X["B"]) == [3.0, 4.0]  # original unchanged


def test_top5_overlap_count():
    from src.interpretability.final_comparison import _top5_overlap
    rankings = {
        "GLM": ["A", "B", "C", "D", "E"],
        "XGBoost": ["A", "B", "C", "X", "Y"],
        "ChebyKAN": ["A", "B", "Z", "D", "E"],
        "FourierKAN": ["A", "B", "C", "D", "W"],
    }
    assert _top5_overlap(rankings) == 2  # "A" and "B" in all four
