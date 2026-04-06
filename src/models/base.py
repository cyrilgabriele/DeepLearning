"""Common abstractions for Prudential models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing help only
    from src.configs import ExperimentConfig


class PrudentialModel(ABC):
    """Abstract base class so trainers can interact with any estimator."""

    def __init__(self, **model_params: Any) -> None:
        self.model_params = model_params

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **fit_kwargs: Any,
    ) -> None:  # pragma: no cover - interface
        """Fit the estimator using processed features."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        """Return predictions for downstream metric computation."""

    def config(self) -> Dict[str, Any]:
        """Return a serializable view of the model parameters."""

        return dict(self.model_params)


@dataclass
class TrainingArtifacts:
    """Lightweight bundle returned by the trainer."""

    model: PrudentialModel
    metrics: Dict[str, Optional[float]]
    device: str
    config: "ExperimentConfig"
    random_seed: int
    summary_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    test_predictions_path: Optional[Path] = None
