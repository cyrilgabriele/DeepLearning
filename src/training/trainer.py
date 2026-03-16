"""Experiment trainer coordinating preprocessing, models, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.configs import ExperimentConfig, GLOBAL_RANDOM_SEED
from src.data.prudential_dataset import (
    PrudentialDataSplits,
    load_and_prepare_prudential_training_data,
)
from src.data.prudential_kan_preprocessing import PrudentialKANPreprocessor
from src.data.prudential_paper_preprocessing import PrudentialPaperPreprocessor
from src.models import PrudentialModel, TrainingArtifacts, create_model


@dataclass(frozen=True)
class Trainer:
    config: ExperimentConfig
    device: str
    random_seed: int = GLOBAL_RANDOM_SEED

    def run(self) -> TrainingArtifacts:
        trainer_cfg = self.config.trainer
        preprocessor = self._build_preprocessor()
        splits = load_and_prepare_prudential_training_data(
            trainer_cfg.train_csv,
            preprocessor=preprocessor,
            eval_size=trainer_cfg.eval_size,
            random_state=self.random_seed,
            stratify=self.config.preprocessing.stratify,
        )

        model_kwargs = self.config.model.registry_kwargs()
        model_kwargs.setdefault("device", self.device)

        model = create_model(
            self.config.model.name,
            random_state=self.random_seed,
            **model_kwargs,
        )
        model.fit(splits.X_train, splits.y_train)

        metrics = self._evaluate(model, splits)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        summary_path = self._persist_run_summary(metrics, timestamp)
        checkpoint_path = self._persist_checkpoint(model, timestamp)
        test_predictions_path = self._generate_test_predictions(
            model=model,
            preprocessor=splits.preprocessor,
            timestamp=timestamp,
        )
        return TrainingArtifacts(
            model=model,
            metrics=metrics,
            device=self.device,
            config=self.config,
            random_seed=self.random_seed,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
            test_predictions_path=test_predictions_path,
        )


    def _build_preprocessor(self):
        prep_cfg = self.config.preprocessing

        if prep_cfg.recipe == "paper":
            return PrudentialPaperPreprocessor(missing_threshold=prep_cfg.missing_threshold)

        return PrudentialKANPreprocessor(
            missing_threshold=prep_cfg.missing_threshold,
            random_state=self.random_seed,
            use_stratified_kfold=prep_cfg.use_stratified_kfold,
            n_splits=prep_cfg.kan_n_splits,
        )

    def _evaluate(self, model, splits: PrudentialDataSplits) -> Dict[str, Optional[float]]:
        if splits.X_eval is None:
            return self._nan_metrics()

        preds = model.predict(splits.X_eval)
        y_true = splits.y_eval
        return self._build_metrics(y_true, preds)

    @staticmethod
    def _nan_metrics() -> Dict[str, Optional[float]]:
        return {"mae": None, "accuracy": None, "f1_macro": None, "qwk": None}

    @staticmethod
    def _build_metrics(y_true, preds) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, preds)),
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1_macro": float(f1_score(y_true, preds, average="macro")),
            "qwk": float(cohen_kappa_score(y_true, preds, weights="quadratic")),
        }

    def _persist_run_summary(self, metrics: Dict[str, float], timestamp: str) -> Optional[Path]:
        """Write a JSON summary capturing config, seed, device, and metrics."""

        output_root = Path("artifacts") / self.config.trainer.experiment_name
        summary_path = output_root / f"run-summary-{timestamp}.json"

        payload = {
            "experiment_name": self.config.trainer.experiment_name,
            "random_seed": self.random_seed,
            "device": self.device,
            "metrics": metrics,
            "config": self.config.model_dump(mode="json"),
        }

        try:
            output_root.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return summary_path
        except OSError as exc: 
            print(f"Warning: failed to persist run summary at {summary_path}: {exc}")
            return None

    def _persist_checkpoint(self, model: PrudentialModel, timestamp: str) -> Optional[Path]:
        """Save a Torch checkpoint if the estimator exposes a module."""
        try:  # Optional torch dependency
            import torch
        except ImportError:  # pragma: no cover - torch-less envs
            return None

        module = self._resolve_torch_module(model, torch)
        if module is None:
            return None

        checkpoint_root = Path("checkpoints") / self.config.trainer.experiment_name
        checkpoint_path = checkpoint_root / f"model-{timestamp}.pt"

        try:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            torch.save(module.state_dict(), checkpoint_path)
            return checkpoint_path
        except OSError as exc:
            print(f"Warning: failed to persist checkpoint at {checkpoint_path}: {exc}")
        except Exception as exc:  # pragma: no cover - torch save failures
            print(f"Warning: torch.save failed for checkpoint {checkpoint_path}: {exc}")
        return None

    @staticmethod
    def _resolve_torch_module(model: PrudentialModel, torch_module) -> Optional["torch.nn.Module"]:
        """Return the underlying torch.nn.Module if present."""

        ModuleType = torch_module.nn.Module
        if isinstance(model, ModuleType):
            return model

        for attr in ("module", "model", "estimator"):
            candidate = getattr(model, attr, None)
            if isinstance(candidate, ModuleType):
                return candidate
        return None

    def _generate_test_predictions(
        self,
        *,
        model: PrudentialModel,
        preprocessor,
        timestamp: str,
    ) -> Optional[Path]:
        """Transform test.csv and emit predictions for Kaggle submission."""

        test_path = self.config.trainer.test_csv
        if test_path is None:
            return None

        if not test_path.exists():
            print(f"Warning: test CSV not found at {test_path}, skipping test predictions.")
            return None

        try:
            test_df = pd.read_csv(test_path)
        except OSError as exc:  # pragma: no cover - defensive IO guard
            print(f"Warning: failed to read test CSV at {test_path}: {exc}")
            return None

        processed = preprocessor.transform(test_df.copy())
        preds = model.predict(processed)

        ids = test_df["Id"] if "Id" in test_df.columns else pd.Series(range(len(preds)))
        predictions_df = pd.DataFrame({"Id": ids, "Response": preds})

        output_root = Path("artifacts") / self.config.trainer.experiment_name
        predictions_path = output_root / f"test-predictions-{timestamp}.csv"

        try:
            output_root.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(predictions_path, index=False)
            return predictions_path
        except OSError as exc:  # pragma: no cover - filesystem guard
            print(f"Warning: failed to write test predictions at {predictions_path}: {exc}")
            return None
