"""Dataset splitting utilities aligned with the Kaggle Prudential setup.

Kaggle already supplies distinct `train.csv` and `test.csv` files. The helpers
here therefore only carve an *optional* evaluation slice out of `train.csv`
before fitting the preprocessing pipeline. Every learned statistic remains
training-only so the resulting metrics stay publishable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PrudentialDataSplits:
    """Container storing both raw and processed train/eval data."""

    X_train_raw: pd.DataFrame
    y_train: pd.Series
    X_eval_raw: Optional[pd.DataFrame]
    y_eval: Optional[pd.Series]
    X_train: pd.DataFrame
    X_eval: Optional[pd.DataFrame]
    preprocessor: object

    def summary(self) -> dict:
        summary = {
            "train_size": len(self.X_train_raw),
            "n_features_raw": self.X_train_raw.shape[1],
            "n_features_processed": self.X_train.shape[1],
        }

        if self.X_eval_raw is not None:
            summary.update({
                "eval_size": len(self.X_eval_raw),
            })
        else:
            summary["eval_size"] = 0

        return summary


def split_prudential_training_df(
    df: pd.DataFrame,
    *,
    preprocessor,
    target_column: str,
    eval_size: float,
    random_state: int,
    stratify: bool,
) -> PrudentialDataSplits:
    """Split the Kaggle training dataframe into train/eval and preprocess.

    Parameters
    ----------
    df : pd.DataFrame
        The Kaggle `train.csv` contents (contains both features and labels).
    preprocessor : object
        Instance of either `PrudentialPaperPreprocessor` or
        `PrudentialKANPreprocessor`.
    eval_size : float
        Fraction of the training dataframe reserved for evaluation. Because the
        Kaggle training file is already a subset of the entire competition data,
        this fraction is relative to the training file only. Set to 0.0 to skip
        creating an evaluation split altogether.
    """

    if preprocessor is None:
        raise ValueError("A preprocessor instance must be provided.")

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    if not 0.0 <= eval_size < 1.0:
        raise ValueError("eval_size must be within [0, 1) to keep holdout meaningful.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    strat_series = y if stratify else None

    has_eval = eval_size > 0.0

    if not has_eval:
        X_train_raw = X
        y_train = y
        X_eval_raw = None
        y_eval = None
    else:
        X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=eval_size,
            random_state=random_state,
            stratify=strat_series,
        )

    X_train = preprocessor.fit_transform(X_train_raw.copy(), y_train)

    if has_eval and X_eval_raw is not None:
        X_eval = preprocessor.transform(X_eval_raw.copy())
    else:
        X_eval = None

    return PrudentialDataSplits(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_eval_raw=X_eval_raw,
        y_eval=y_eval,
        X_train=X_train,
        X_eval=X_eval,
        preprocessor=preprocessor,
    )


def load_and_prepare_prudential_training_data(
    train_path: Path | str,
    *,
    preprocessor,
    target_column: str = "Response",
    eval_size: float,
    random_state: int = 42,
    stratify: bool,
) -> PrudentialDataSplits:
    """Read Kaggle's train file and prepare leak-free splits."""

    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}.")

    df = pd.read_csv(path)
    return split_prudential_training_df(
        df,
        preprocessor=preprocessor,
        target_column=target_column,
        eval_size=eval_size,
        random_state=random_state,
        stratify=stratify,
    )


__all__ = [
    "PrudentialDataSplits",
    "split_prudential_training_df",
    "load_and_prepare_prudential_training_data",
]
