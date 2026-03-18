"""Experiment trainer coordinating preprocessing, models, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.configs import ExperimentConfig
from src.data import preprocess_xgboost_paper as paper_prep
from src.data import preprocess_kan_paper as kan_prep
from src.data import preprocess_kan_sota as kan_sota_prep
from src.models import PrudentialModel, TrainingArtifacts, create_model


@dataclass(frozen=True)
class PreparedDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_eval: Optional[pd.DataFrame]
    y_eval: Optional[pd.Series]
    recipe: str
    preprocess_artifacts: Dict[str, object]
    feature_names: Optional[list[str]] = None
    X_train_inner: Optional[pd.DataFrame] = None
    y_train_inner: Optional[pd.Series] = None
    X_val_inner: Optional[pd.DataFrame] = None
    y_val_inner: Optional[pd.Series] = None


@dataclass(frozen=True)
class Trainer:
    config: ExperimentConfig
    device: str
    random_seed: int

    def run(self) -> TrainingArtifacts:
        dataset = self._prepare_data()

        model_kwargs = self.config.model.registry_kwargs()
        model_kwargs.setdefault("device", self.device)

        model = create_model(
            self.config.model.name,
            random_state=self.random_seed,
            **model_kwargs,
        )

        X_train = dataset.X_train_inner if dataset.X_train_inner is not None else dataset.X_train
        y_train = dataset.y_train_inner if dataset.y_train_inner is not None else dataset.y_train
        validation_data = None
        if dataset.X_val_inner is not None and dataset.y_val_inner is not None:
            validation_data = (dataset.X_val_inner, dataset.y_val_inner)

        fit_kwargs = {}
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data

        inner_splits = dataset.preprocess_artifacts.get("inner_splits") if dataset.preprocess_artifacts else None
        if inner_splits:
            fit_kwargs["validation_splits"] = inner_splits

        model.fit(X_train, y_train, **fit_kwargs)

        metrics = self._evaluate(model, dataset)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        summary_path = self._persist_run_summary(metrics, timestamp)
        checkpoint_path = self._persist_checkpoint(model, timestamp)
        self._export_eval_data(splits)
        test_predictions_path = self._generate_test_predictions(
            model=model,
            dataset=dataset,
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


    def _prepare_data(self) -> "PreparedDataset":
        recipe = self.config.preprocessing.recipe
        train_csv = self.config.trainer.train_csv

        if recipe == "xgboost_paper":
            outputs = paper_prep.run_pipeline(train_csv, random_seed=self.random_seed)
            artifacts = {
                "state": outputs["preprocessor_state"],
                "inner_splits": outputs["inner_splits"],
            }
            inner_train = None
            inner_val = None
            inner_train_y = None
            inner_val_y = None
            if outputs["inner_splits"]:
                inner_train, inner_val, inner_train_y, inner_val_y = outputs["inner_splits"][0]
            return PreparedDataset(
                X_train=outputs["X_train_outer"],
                y_train=outputs["y_train_outer"],
                X_eval=outputs["X_test_outer"],
                y_eval=outputs["y_test_outer"],
                recipe=recipe,
                preprocess_artifacts=artifacts,
                feature_names=list(outputs["X_train_outer"].columns),
                X_train_inner=inner_train,
                y_train_inner=inner_train_y,
                X_val_inner=inner_val,
                y_val_inner=inner_val_y,
            )

        if recipe == "kan_paper":
            outputs = kan_prep.run_pipeline(train_csv, random_seed=self.random_seed)
            return self._build_kan_dataset(outputs, recipe, kan_prep.TARGET_COLUMN)
        if recipe == "kan_sota":
            outputs = kan_sota_prep.run_pipeline(train_csv, random_seed=self.random_seed)
            return self._build_kan_dataset(outputs, recipe, kan_sota_prep.TARGET_COLUMN)
        raise ValueError(f"Unknown preprocessing recipe: {recipe}")

    def _evaluate(self, model, dataset: "PreparedDataset") -> Dict[str, Optional[float]]:
        if dataset.X_eval is None:
            return self._nan_metrics()

        preds = model.predict(dataset.X_eval)
        y_true = dataset.y_eval
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
        """Save a Torch checkpoint or a joblib pickle depending on model type."""
        checkpoint_root = Path("checkpoints") / self.config.trainer.experiment_name

        try:
            import torch
            module = self._resolve_torch_module(model, torch)
        except ImportError:
            module = None

        if module is not None:
            checkpoint_path = checkpoint_root / f"model-{timestamp}.pt"
            try:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                torch.save(module.state_dict(), checkpoint_path)
                return checkpoint_path
            except Exception as exc:  # pragma: no cover
                print(f"Warning: torch.save failed at {checkpoint_path}: {exc}")
            return None

        # Sklearn / non-torch models: persist with joblib
        try:
            import joblib
        except ImportError:  # pragma: no cover
            return None

        checkpoint_path = checkpoint_root / f"model-{timestamp}.joblib"
        try:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, checkpoint_path)
            return checkpoint_path
        except Exception as exc:  # pragma: no cover
            print(f"Warning: joblib.dump failed at {checkpoint_path}: {exc}")
        return None

    def _export_eval_data(self, splits) -> None:
        """Persist the preprocessed eval split for downstream interpretability scripts."""
        if splits.X_eval is None or splits.y_eval is None:
            return
        try:
            import json
            data_dir = Path("outputs") / "data"
            reports_dir = Path("outputs") / "reports"
            data_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)
            splits.X_eval.to_parquet(data_dir / "X_eval.parquet", index=False)
            splits.y_eval.to_frame(name="Response").to_parquet(
                data_dir / "y_eval.parquet", index=False
            )
            feature_names = list(splits.X_eval.columns)
            (reports_dir / "feature_names.json").write_text(
                json.dumps(feature_names, indent=2)
            )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to export eval data: {exc}")

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
        dataset: "PreparedDataset",
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

        processed = self._transform_test_dataframe(test_df.copy(), dataset)
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

    def _transform_test_dataframe(self, df: pd.DataFrame, dataset: "PreparedDataset") -> pd.DataFrame:
        if dataset.recipe == "xgboost_paper":
            processed, _ = paper_prep.transform(
                df,
                dataset.preprocess_artifacts["state"],
            )
            return processed

        if dataset.recipe == "kan_paper":
            base_state = dataset.preprocess_artifacts["baseline"]
            kan_state = dataset.preprocess_artifacts["kan"]
            processed_array, _ = kan_prep.transform(
                df,
                base_state,
                kan_state=kan_state,
            )
            return pd.DataFrame(processed_array, columns=dataset.feature_names)

        if dataset.recipe == "kan_sota":
            base_state = dataset.preprocess_artifacts["baseline"]
            sota_state = dataset.preprocess_artifacts["sota"]
            processed_array, _ = kan_sota_prep.transform(
                df,
                base_state,
                sota_state=sota_state,
            )
            return pd.DataFrame(processed_array, columns=dataset.feature_names)

        raise ValueError(f"Unknown preprocessing recipe for test transform: {dataset.recipe}")

    def _build_kan_dataset(
        self,
        outputs: Dict[str, object],
        recipe: str,
        target_name: str,
    ) -> "PreparedDataset":
        feature_names = outputs["feature_names"]
        artifacts = outputs["preprocessor_state"]
        row_indices = outputs.get("row_indices", {})
        train_index = row_indices.get("outer_train")
        test_index = row_indices.get("outer_test")
        X_train = pd.DataFrame(outputs["X_train_outer"], columns=feature_names, index=train_index)
        y_train = pd.Series(
            outputs["y_train_outer"],
            name=target_name,
            index=train_index,
        )
        X_eval = pd.DataFrame(outputs["X_test_outer"], columns=feature_names, index=test_index)
        y_eval = pd.Series(
            outputs["y_test_outer"],
            name=target_name,
            index=test_index,
        )
        inner_indices = outputs.get("inner_split_indices") or []
        inner_train = None
        inner_val = None
        inner_train_y = None
        inner_val_y = None
        if inner_indices and train_index is not None:
            first = inner_indices[0]
            train_idx = first["train"]
            val_idx = first["val"]
            inner_train = X_train.loc[train_idx]
            inner_val = X_train.loc[val_idx]
            inner_train_y = y_train.loc[train_idx]
            inner_val_y = y_train.loc[val_idx]
        return PreparedDataset(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            recipe=recipe,
            preprocess_artifacts=artifacts,
            feature_names=feature_names,
            X_train_inner=inner_train,
            y_train_inner=inner_train_y,
            X_val_inner=inner_val,
            y_val_inner=inner_val_y,
        )
