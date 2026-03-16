import math
import numpy as np
import pandas as pd

from src.data.prudential_kan_preprocessing import PrudentialKANPreprocessor


def _toy_df(n_rows: int = 64) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    repeats = math.ceil(n_rows / 8)
    response = np.tile(np.arange(1, 9), repeats)[:n_rows]
    df = pd.DataFrame(
        {
            "Id": np.arange(n_rows),
            "Response": response,
            "Product_Info_2": rng.choice(list("ABCDE"), size=n_rows),
            "BMI": rng.normal(30, 5, size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
        }
    )
    y = df["Response"]
    X = df.drop(columns=["Response"])
    return X, y


def test_kan_preprocessor_uses_kfold_encoding():
    X, y = _toy_df()
    preprocessor = PrudentialKANPreprocessor(
        missing_threshold=0.95,
        random_state=0,
        n_splits=4,
        use_stratified_kfold=True,
    )

    X_proc = preprocessor.fit_transform(X, y)
    assert preprocessor._cat_encoding_mode == "kfold"

    # Ensure transform path reuses fitted encoder
    X_proc_again = preprocessor.transform(X)
    assert X_proc_again.shape == X_proc.shape
    assert np.isfinite(X_proc_again.values).all()


def test_kan_preprocessor_raises_when_not_enough_per_class():
    X, y = _toy_df(n_rows=8)
    preprocessor = PrudentialKANPreprocessor(
        missing_threshold=0.95,
        random_state=0,
        n_splits=10,
        use_stratified_kfold=True,
    )

    try:
        preprocessor.fit_transform(X, y)
    except ValueError as exc:  # not enough per class to split 10 ways
        assert "Stratified" in str(exc) or "Reduce" in str(exc)
    else:
        raise AssertionError("Expected ValueError due to insufficient per-class samples.")
