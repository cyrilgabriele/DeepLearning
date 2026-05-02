"""TabPFN regressor wrapper exposing the PrudentialModel interface.

Option A: uses base ``tabpfn.TabPFNRegressor`` with
``ignore_pretraining_limits=True`` (no AutoTabPFN ensembling — the
``tabpfn-extensions`` package pins ``pandas<3`` and conflicts with this
project's ``pandas>=3.0.1``).

Training-set subsampling: when the input training set exceeds
``max_train_samples`` (default 10,000), the wrapper randomly subsamples
to that size using ``numpy.random.default_rng(self.random_state)``.
TabPFN-v2 is designed for ≤10k training samples; subsampling keeps
inference within tractable wall-clock on CPU and matches the standard
published TabPFN workflow at >10k. Each seed produces a different but
reproducible subsample, so 3-seed variance reflects subsample variance
in addition to TabPFN's internal randomness.

Threshold calibration mirrors the XGBBaseline pattern: thresholds are
fit on the (subsampled) training predictions inside ``fit()`` so
``predict()`` returns ordinal class labels 1-8 directly. ``evaluate()``
recalibrates thresholds on the supplied split and returns its QWK.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.metrics.qwk import _apply_thresholds, optimize_thresholds
from src.models.base import PrudentialModel


def _build_auto_tabpfn(
    *,
    device: str,
    random_state: int,
    ignore_pretraining_limits: bool = True,
):
    """Construct and return a TabPFNRegressor instance.

    Isolated as a free function so unit tests can patch it without
    instantiating the real (large) pretrained model. Despite the legacy
    function name, this returns the base ``TabPFNRegressor`` (Option A —
    no AutoTabPFN ensemble; ``ignore_pretraining_limits=True`` allows the
    >10k-row Prudential training set as in-context input).
    """

    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        device=device,
        random_state=random_state,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


class TabPFNAutoRegressor(PrudentialModel):
    """Pretrained TabPFN-v2 regressor with ordinal threshold calibration."""

    def __init__(
        self,
        *,
        n_estimators: int = 8,
        max_time: int = 300,
        device: str = "auto",
        random_state: int = 42,
        max_train_samples: int = 10000,
        **kwargs: Any,
    ) -> None:
        # n_estimators and max_time are accepted (and stored) for config
        # compatibility, but are unused under Option A. They remain in the
        # constructor signature so the YAML configs and registry layer do
        # not need to be reshaped.
        super().__init__(
            n_estimators=n_estimators,
            max_time=max_time,
            device=device,
            random_state=random_state,
            max_train_samples=max_train_samples,
        )
        self.n_estimators = n_estimators
        self.max_time = max_time
        self.device = device
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self._estimator = None
        self.thresholds: Optional[np.ndarray] = None
        self.threshold_source_split: Optional[str] = None
        self.threshold_optimization_qwk: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        validation_splits=None,
        **_kwargs: Any,
    ) -> None:
        _ = validation_data, validation_splits  # unused; thresholds fit on training
        if len(X) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(len(X), size=self.max_train_samples, replace=False))
            X = X.iloc[idx]
            y = y.iloc[idx]
        self._estimator = _build_auto_tabpfn(
            device=self.device,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y)
        y_cont = self._estimator.predict(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        self.thresholds, kappa = optimize_thresholds(y_arr, y_cont)
        self.threshold_optimization_qwk = float(kappa)
        self.threshold_source_split = "training"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._estimator is None or self.thresholds is None:
            raise RuntimeError("Call fit() before predict().")
        y_cont = self._estimator.predict(X)
        return np.clip(_apply_thresholds(y_cont, self.thresholds), 1, 8).astype(int)

    def predict_continuous(self, X: pd.DataFrame) -> np.ndarray:
        """Return the underlying continuous regression output (pre-threshold).

        Used by downstream scripts that need the raw score (permutation
        importance can score on either continuous or discrete outputs).
        """

        if self._estimator is None:
            raise RuntimeError("Call fit() before predict_continuous().")
        return np.asarray(self._estimator.predict(X))

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        if self._estimator is None:
            raise RuntimeError("Call fit() before evaluate().")
        y_cont = self._estimator.predict(X)
        y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
        self.thresholds, kappa = optimize_thresholds(y_arr, y_cont)
        self.threshold_optimization_qwk = float(kappa)
        self.threshold_source_split = "evaluation"
        return float(kappa)

    def get_ordinal_calibration(self) -> Optional[Dict[str, Any]]:
        if self.thresholds is None:
            return None
        payload: Dict[str, Any] = {
            "method": "optimized_thresholds",
            "num_classes": 8,
            "thresholds": [float(value) for value in self.thresholds],
        }
        if self.threshold_source_split is not None:
            payload["source_split"] = self.threshold_source_split
        if self.threshold_optimization_qwk is not None:
            payload["optimized_qwk_on_source_split"] = float(self.threshold_optimization_qwk)
        return payload


def build_tabpfn_auto_model(
    *,
    random_state: int = 42,
    n_estimators: int = 8,
    max_time: int = 300,
    device: str = "auto",
    **_kwargs: Any,
) -> TabPFNAutoRegressor:
    """Factory for the model registry."""

    # Trainer always passes a ``device`` kwarg. Other family-specific kwargs
    # (depth, width, hidden_widths, degree, flavor) are dropped.
    return TabPFNAutoRegressor(
        n_estimators=n_estimators,
        max_time=max_time,
        device=device if device != "auto" else "cpu",
        random_state=random_state,
    )
