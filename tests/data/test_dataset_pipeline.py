import math
import numpy as np
import pandas as pd

from src.data.prudential_dataset import split_prudential_training_df
from src.data.prudential_paper_preprocessing import PrudentialPaperPreprocessor


def _make_mock_prudential_df(n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    repeats = math.ceil(n_rows / 8)
    response = np.tile(np.arange(1, 9), repeats)[:n_rows]
    return pd.DataFrame(
        {
            "Id": np.arange(n_rows),
            "Response": response,
            "BMI": np.linspace(18, 40, num=n_rows),
            "Product_Info_2": rng.choice(list("ABCDEFG"), size=n_rows),
            "Medical_Keyword_1": rng.integers(0, 2, size=n_rows),
            "Product_Info_3": rng.integers(1, 4, size=n_rows),
        }
    )


def test_split_pipeline_handles_eval_fraction_of_train():
    df = _make_mock_prudential_df()
    preprocessor = PrudentialPaperPreprocessor(missing_threshold=0.95)

    splits = split_prudential_training_df(
        df,
        preprocessor=preprocessor,
        target_column="Response",
        eval_size=0.2,
        random_state=0,
        stratify=False,
    )

    assert len(splits.X_train_raw) + len(splits.X_eval_raw) == len(df)
    assert splits.X_train.shape[1] == splits.X_eval.shape[1]

    bmi_train_min = splits.X_train_raw["BMI"].min()
    assert preprocessor.scaler_cont.data_min_[0] == bmi_train_min

    manual_eval = preprocessor.transform(splits.X_eval_raw.copy())
    pd.testing.assert_frame_equal(manual_eval, splits.X_eval)


def test_split_pipeline_without_eval_keeps_all_training_rows():
    df = _make_mock_prudential_df(n_rows=32)
    preprocessor = PrudentialPaperPreprocessor(missing_threshold=0.95)

    splits = split_prudential_training_df(
        df,
        preprocessor=preprocessor,
        target_column="Response",
        eval_size=0.0,
        random_state=0,
        stratify=False,
    )

    assert splits.X_eval_raw is None
    assert splits.X_eval is None
    assert len(splits.X_train_raw) == len(df)


def test_invalid_eval_size_raises():
    df = _make_mock_prudential_df()
    preprocessor = PrudentialPaperPreprocessor()

    try:
        split_prudential_training_df(
            df,
            preprocessor=preprocessor,
            target_column="Response",
            eval_size=1.2,
            random_state=0,
            stratify=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when eval_size is not (0, 1).")
