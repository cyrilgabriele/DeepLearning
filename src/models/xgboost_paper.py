"""Prudential XGBoost model mirroring the ICSITech 2019 case study pipeline.

The paper *Analysis Accuracy of XGBoost Model for Multiclass Classification -
A Case Study of Applicant Level Risk Prediction for Life Insurance* tuned
`max_depth`, `min_child_weight`, `learning_rate`, `subsample`,
`colsample_bytree`, `alpha`, and `lambda` sequentially using Quadratic
Weighted Kappa (QWK) on a held-out validation split before refitting on the
combined train+val fold. This module reproduces that estimator inside the
PrudentialModel registry so the modern Trainer pipeline can call it directly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

from .base import PrudentialModel


class XGBoostPaperModel(PrudentialModel):
    """Paper-faithful XGBoost classifier with sequential hyperparameter tuning."""

    DEFAULT_TUNING_GRID: "OrderedDict[str, Tuple[float, ...]]" = OrderedDict(
        [
            ("max_depth", (5, 10, 15, 20, 25, 30, 35)),
            ("min_child_weight", (1, 5, 10, 15)),
            ("learning_rate", (0.001, 0.01, 0.1, 1.0, 1.5)),
            ("subsample", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            ("colsample_bytree", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            ("reg_alpha", (0.0, 0.25, 0.5, 0.75, 1.0)),
            ("reg_lambda", (0.0, 0.25, 0.5, 0.75, 1.0)),
        ]
    )

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth: int = 15,
        min_child_weight: float = 1.0,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        num_classes: int = 8,
        tree_method: str = "hist",
        eval_metric: str = "mlogloss",
        auto_tune: bool = True,
        refit_full_training: bool = True,
        tuning_grid: Mapping[str, Sequence[float]] | None = None,
        n_jobs: Optional[int] = None,
        seed_trials: Sequence[int] | None = None,
    ) -> None:
        params = {
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "num_classes": num_classes,
            "tree_method": tree_method,
            "eval_metric": eval_metric,
            "auto_tune": auto_tune,
            "refit_full_training": refit_full_training,
        }
        super().__init__(**params)

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.tree_method = tree_method
        self.eval_metric = eval_metric
        self.auto_tune = auto_tune
        self.refit_full_training = refit_full_training
        self.n_jobs = n_jobs
        self.seed_trials = [int(s) for s in seed_trials] if seed_trials else None

        self._base_params = {
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self.tuning_grid = self._build_tuning_grid(tuning_grid)

        self._estimator: Optional[xgb.XGBClassifier] = None
        self._classes_: Optional[np.ndarray] = None
        self._label_to_index: Dict[int, int] = {}
        self.best_params_: Dict[str, float] = dict(self._base_params)
        self.best_kappa_: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        validation_data: Optional[Tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]] = None,
        validation_splits: Optional[
            Sequence[Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, pd.Series | np.ndarray, pd.Series | np.ndarray]]
        ] = None,
    ) -> None:
        X_train = self._to_numpy_features(X)
        y_train_raw = self._to_numpy_labels(y)

        X_val = y_val_raw = None
        if validation_data is not None:
            X_val = self._to_numpy_features(validation_data[0])
            y_val_raw = self._to_numpy_labels(validation_data[1])

        extra_label_arrays: list[np.ndarray] = []
        raw_tuning_splits: list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if y_val_raw is not None:
            extra_label_arrays.append(y_val_raw)

        if validation_splits:
            for split in validation_splits:
                X_split_train, X_split_val, y_split_train, y_split_val = split
                split_train = self._to_numpy_features(X_split_train)
                split_val = self._to_numpy_features(X_split_val)
                split_y_train = self._to_numpy_labels(y_split_train)
                split_y_val = self._to_numpy_labels(y_split_val)
                raw_tuning_splits.append((split_train, split_y_train, split_val, split_y_val))
                extra_label_arrays.extend([split_y_train, split_y_val])

        combined_extra = None
        if extra_label_arrays:
            combined_extra = np.concatenate(extra_label_arrays, axis=0)

        self._fit_label_encoder(y_train_raw, combined_extra)
        y_train = self._encode_labels(y_train_raw)
        y_val = self._encode_labels(y_val_raw) if y_val_raw is not None else None

        tuning_splits: list[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if raw_tuning_splits:
            for train_features, train_labels_raw, val_features, val_labels_raw in raw_tuning_splits:
                tuning_splits.append(
                    (
                        train_features,
                        self._encode_labels(train_labels_raw),
                        val_features,
                        self._encode_labels(val_labels_raw),
                        val_labels_raw,
                    )
                )
        elif X_val is not None and y_val is not None and y_val_raw is not None:
            tuning_splits.append((X_train, y_train, X_val, y_val, y_val_raw))

        seeds = self.seed_trials or [self.random_state]
        best_result: Optional[Tuple[xgb.XGBClassifier, Dict[str, float], Optional[float], int]] = None
        best_score = float("-inf")

        for seed in seeds:
            estimator, tuned_params, seed_score = self._fit_single_seed(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tuning_splits=tuning_splits,
                seed=seed,
            )

            if best_result is None:
                best_result = (estimator, tuned_params, seed_score, seed)
                best_score = seed_score if seed_score is not None else best_score
                continue

            if seed_score is not None and (best_result[2] is None or seed_score > best_score):
                best_result = (estimator, tuned_params, seed_score, seed)
                best_score = seed_score

        if best_result is None:
            raise RuntimeError("XGBoostPaperModel failed to fit any seed trial.")

        self._estimator, self.best_params_, self.best_kappa_, selected_seed = best_result
        if selected_seed is not None:
            self.random_state = selected_seed

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._estimator is None or self._classes_ is None:
            raise RuntimeError("Call fit() before predict().")

        X_np = self._to_numpy_features(X)
        encoded = self._estimator.predict(X_np)
        decoded = self._decode_labels(encoded)
        return decoded.astype(int)

    # ------------------------------------------------------------------
    def _build_tuning_grid(
        self,
        override: Mapping[str, Sequence[float]] | None,
    ) -> "OrderedDict[str, Tuple[float, ...]]":
        if override is None:
            return OrderedDict((k, tuple(v)) for k, v in self.DEFAULT_TUNING_GRID.items())
        return OrderedDict((key, tuple(values)) for key, values in override.items())

    def _fit_label_encoder(
        self,
        y_train: np.ndarray,
        y_val: Optional[np.ndarray],
    ) -> None:
        all_labels = y_train if y_val is None else np.concatenate([y_train, y_val])
        classes = np.arange(1, self.num_classes + 1)
        if not np.all(np.isin(all_labels, classes)):
            unknown = sorted(set(int(v) for v in all_labels) - set(classes.tolist()))
            raise ValueError(f"Encountered labels outside 1-{self.num_classes}: {unknown}")
        self._classes_ = classes
        self._label_to_index = {int(label): idx for idx, label in enumerate(classes)}

    def _encode_labels(self, y: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if y is None:
            return None
        indices = [self._label_to_index[int(value)] for value in y]
        return np.asarray(indices, dtype=np.int32)

    def _decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        if self._classes_ is None:
            raise RuntimeError("Label encoder was not fitted.")
        encoded = np.asarray(encoded, dtype=int)
        encoded = np.clip(encoded, 0, len(self._classes_) - 1)
        return self._classes_[encoded]

    def _fit_single_seed(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        tuning_splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        seed: int,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float], Optional[float]]:
        tuned_params = dict(self._base_params)
        seed_score: Optional[float] = None
        if self.auto_tune and tuning_splits:
            tuned_params, seed_score = self._tune_hyperparameters(tuning_splits, seed)
        elif tuning_splits:
            seed_score = self._evaluate_candidate(tuned_params, tuning_splits, seed)

        final_X = X_train
        final_y = y_train
        if self.refit_full_training and X_val is not None and y_val is not None:
            final_X = np.concatenate([X_train, X_val], axis=0)
            final_y = np.concatenate([y_train, y_val], axis=0)

        estimator = self._build_estimator(tuned_params, seed)
        estimator.fit(final_X, final_y)
        return estimator, tuned_params, seed_score

    def _tune_hyperparameters(
        self,
        splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        seed: int,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        tuned = dict(self._base_params)
        best_overall = None
        for param_name, candidates in self.tuning_grid.items():
            if not candidates:
                continue
            best_value = tuned.get(param_name)
            best_score = -np.inf
            for candidate in candidates:
                trial_params = dict(tuned)
                trial_params[param_name] = candidate
                score = self._evaluate_candidate(trial_params, splits, seed)
                if score > best_score:
                    best_score = score
                    best_value = candidate
            tuned[param_name] = best_value
            best_overall = best_score
        return tuned, best_overall

    def _evaluate_candidate(
        self,
        params: Dict[str, float],
        splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        seed: int,
    ) -> float:
        scores = []
        for X_train, y_train, X_val, y_val, y_val_raw in splits:
            estimator = self._build_estimator(params, seed)
            estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds_encoded = estimator.predict(X_val)
            preds = self._decode_labels(preds_encoded)
            scores.append(cohen_kappa_score(y_val_raw, preds, weights="quadratic"))
        return float(np.mean(scores)) if scores else float("nan")

    def _build_estimator(self, params: Dict[str, float], seed: int) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="multi:softprob",
            num_class=self.num_classes,
            random_state=seed,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            eval_metric=self.eval_metric,
        )

    @staticmethod
    def _to_numpy_features(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _to_numpy_labels(y: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=int, copy=False)
        return np.asarray(y, dtype=int)


def build_xgboost_paper_model(*, random_state: int, **model_params: Any) -> XGBoostPaperModel:
    """Factory forwarding registry kwargs to the paper-faithful estimator."""

    # Trainer always passes a `device` kwarg so TabKAN can place tensors. XGB
    # ignores that, so drop it (and other TabKAN-specific knobs) here.
    for unused in ("depth", "width", "degree", "flavor", "device"):
        model_params.pop(unused, None)
    return XGBoostPaperModel(random_state=random_state, **model_params)
