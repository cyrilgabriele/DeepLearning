"""Experiment trainer coordinating preprocessing, models, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.config import ExperimentConfig, set_global_seed
from src.preprocessing import preprocess_xgboost_paper as paper_prep
from src.preprocessing import preprocess_kan_paper as kan_prep
from src.preprocessing import preprocess_kan_sota as kan_sota_prep
from src.models import PrudentialModel, TrainingArtifacts, create_model


@dataclass(frozen=True)
class PreparedDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_eval: Optional[pd.DataFrame]
    y_eval: Optional[pd.Series]
    recipe: str
    preprocess_artifacts: Dict[str, object]
    X_eval_raw: Optional[pd.DataFrame] = None
    feature_names: Optional[list[str]] = None
    X_train_inner: Optional[pd.DataFrame] = None
    y_train_inner: Optional[pd.Series] = None
    X_val_inner: Optional[pd.DataFrame] = None
    y_val_inner: Optional[pd.Series] = None
    all_feature_names: Optional[list[str]] = None


@dataclass(frozen=True)
class Trainer:
    config: ExperimentConfig
    device: str
    random_seed: int

    def run(self) -> TrainingArtifacts:
        dataset = self._prepare_data()
        self.config.model.assert_training_ready()

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
        ordinal_calibration = self._resolve_ordinal_calibration(model)

        metrics = self._evaluate(model, dataset)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        checkpoint_path = self._persist_checkpoint(model, timestamp)
        summary_path = self._persist_run_summary(
            metrics,
            dataset,
            timestamp,
            checkpoint_path,
            ordinal_calibration=ordinal_calibration,
        )
        self._persist_checkpoint_manifest(
            metrics=metrics,
            dataset=dataset,
            timestamp=timestamp,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            ordinal_calibration=ordinal_calibration,
        )
        self._export_eval_data(dataset, ordinal_calibration=ordinal_calibration)
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
            ordinal_calibration=ordinal_calibration,
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
            dataset = PreparedDataset(
                X_train=outputs["X_train_outer"],
                y_train=outputs["y_train_outer"],
                X_eval=outputs["X_test_outer"],
                y_eval=outputs["y_test_outer"],
                X_eval_raw=self._load_raw_eval_features(outputs["X_test_outer"].index),
                recipe=recipe,
                preprocess_artifacts=artifacts,
                feature_names=list(outputs["X_train_outer"].columns),
                X_train_inner=inner_train,
                y_train_inner=inner_train_y,
                X_val_inner=inner_val,
                y_val_inner=inner_val_y,
                all_feature_names=list(outputs["X_train_outer"].columns),
            )
            return self._apply_selected_features(dataset)

        if recipe == "kan_paper":
            outputs = kan_prep.run_pipeline(train_csv, random_seed=self.random_seed)
            return self._apply_selected_features(
                self._build_kan_dataset(outputs, recipe, kan_prep.TARGET_COLUMN)
            )
        if recipe == "kan_sota":
            outputs = kan_sota_prep.run_pipeline(train_csv, random_seed=self.random_seed)
            return self._apply_selected_features(
                self._build_kan_dataset(outputs, recipe, kan_sota_prep.TARGET_COLUMN)
            )
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

    def _persist_run_summary(
        self,
        metrics: Dict[str, float],
        dataset: "PreparedDataset",
        timestamp: str,
        checkpoint_path: Optional[Path],
        *,
        ordinal_calibration: Dict[str, object] | None,
    ) -> Optional[Path]:
        """Write a JSON summary capturing config, seed, device, and metrics."""

        output_root = Path("artifacts") / self.config.trainer.experiment_name
        summary_path = output_root / f"run-summary-{timestamp}.json"
        preprocessing_contract = self._build_preprocessing_contract(dataset)

        payload = {
            "experiment_name": self.config.trainer.experiment_name,
            "random_seed": self.random_seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": self.device,
            "metrics": metrics,
            "model_architecture": self.config.model.architecture_payload(),
            "ordinal_calibration": ordinal_calibration,
            "preprocessing": preprocessing_contract,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "config": self.config.model_dump(mode="json"),
        }

        try:
            output_root.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return summary_path
        except OSError as exc: 
            print(f"Warning: failed to persist run summary at {summary_path}: {exc}")
            return None

    def _persist_checkpoint_manifest(
        self,
        *,
        metrics: Dict[str, float],
        dataset: "PreparedDataset",
        timestamp: str,
        checkpoint_path: Optional[Path],
        summary_path: Optional[Path],
        ordinal_calibration: Dict[str, object] | None,
    ) -> Optional[Path]:
        """Write a checkpoint-adjacent manifest mirroring the effective run contract."""

        if checkpoint_path is None:
            return None

        manifest_path = checkpoint_path.with_suffix(".manifest.json")
        payload = {
            "experiment_name": self.config.trainer.experiment_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": self.random_seed,
            "device": self.device,
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path) if summary_path is not None else None,
            "metrics": metrics,
            "model_architecture": self.config.model.architecture_payload(),
            "ordinal_calibration": ordinal_calibration,
            "preprocessing": self._build_preprocessing_contract(dataset),
            "config": self.config.model_dump(mode="json"),
        }

        try:
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return manifest_path
        except OSError as exc:  # pragma: no cover - filesystem guard
            print(f"Warning: failed to persist checkpoint manifest at {manifest_path}: {exc}")
        return None

    def _build_preprocessing_contract(self, dataset: "PreparedDataset") -> Dict[str, object]:
        """Return the effective preprocessing payload stored with the run artifacts."""

        feature_names = list(dataset.feature_names or dataset.X_train.columns)
        contract_payload = self.config.preprocessing.contract_payload()
        return {
            **contract_payload,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        }

    def _apply_selected_features(self, dataset: "PreparedDataset") -> "PreparedDataset":
        selected_path = self.config.preprocessing.selected_features_path
        if selected_path is None:
            return dataset

        full_feature_names = list(dataset.all_feature_names or dataset.feature_names or dataset.X_train.columns)
        selected_features = self._load_selected_features(selected_path)
        missing = [feature for feature in selected_features if feature not in full_feature_names]
        if missing:
            raise ValueError(
                "Selected features were not found in the preprocessed feature space: "
                f"{', '.join(missing[:10])}"
            )

        def _subset_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if frame is None:
                return None
            return frame.loc[:, selected_features].copy()

        preprocess_artifacts = dict(dataset.preprocess_artifacts)
        inner_splits = preprocess_artifacts.get("inner_splits")
        if inner_splits:
            preprocess_artifacts["inner_splits"] = self._subset_inner_splits(
                inner_splits,
                selected_features,
            )

        X_eval_raw = dataset.X_eval_raw
        if X_eval_raw is not None:
            raw_columns = self._resolve_raw_columns_for_selected_features(selected_features, X_eval_raw)
            X_eval_raw = X_eval_raw.loc[:, raw_columns].copy()

        return PreparedDataset(
            X_train=_subset_frame(dataset.X_train),
            y_train=dataset.y_train,
            X_eval=_subset_frame(dataset.X_eval),
            y_eval=dataset.y_eval,
            recipe=dataset.recipe,
            preprocess_artifacts=preprocess_artifacts,
            X_eval_raw=X_eval_raw,
            feature_names=list(selected_features),
            X_train_inner=_subset_frame(dataset.X_train_inner),
            y_train_inner=dataset.y_train_inner,
            X_val_inner=_subset_frame(dataset.X_val_inner),
            y_val_inner=dataset.y_val_inner,
            all_feature_names=full_feature_names,
        )

    @staticmethod
    def _load_selected_features(selected_path: Path) -> list[str]:
        try:
            raw_text = selected_path.read_text()
        except OSError as exc:
            raise OSError(f"Failed to read selected feature list at {selected_path}: {exc}") from exc

        features: list[str]
        suffix = selected_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(raw_text)
            if isinstance(payload, list):
                features = [str(item) for item in payload]
            elif isinstance(payload, dict):
                for key in ("features", "selected_features", "feature_names"):
                    if key in payload:
                        candidate = payload[key]
                        if not isinstance(candidate, list):
                            raise TypeError(
                                f"Selected feature payload key '{key}' must contain a list."
                            )
                        features = [str(item) for item in candidate]
                        break
                else:
                    raise ValueError(
                        "Selected feature JSON must be a list or include one of: "
                        "'features', 'selected_features', 'feature_names'."
                    )
            else:
                raise TypeError("Selected feature JSON must be a list or object.")
        else:
            features = [line.strip() for line in raw_text.splitlines() if line.strip()]

        deduped = list(dict.fromkeys(features))
        if not deduped:
            raise ValueError(f"Selected feature list at {selected_path} is empty.")
        return deduped

    @staticmethod
    def _resolve_raw_columns_for_selected_features(
        selected_features: list[str],
        raw_frame: pd.DataFrame,
    ) -> list[str]:
        raw_columns: list[str] = []
        if "Id" in raw_frame.columns:
            raw_columns.append("Id")

        for feature in selected_features:
            base_feature = feature
            for prefix in ("cb_", "qt_", "mm_"):
                if base_feature.startswith(prefix):
                    base_feature = base_feature[len(prefix):]
                    break
            if base_feature.startswith("missing_"):
                base_feature = base_feature[len("missing_"):]
            if base_feature in raw_frame.columns:
                raw_columns.append(base_feature)

        deduped = list(dict.fromkeys(raw_columns))
        return deduped or list(raw_frame.columns)

    @staticmethod
    def _subset_inner_splits(inner_splits, selected_features: list[str]):
        subset_splits = []
        for split in inner_splits:
            if len(split) < 4:
                subset_splits.append(split)
                continue
            X_split_train, X_split_val, y_split_train, y_split_val, *rest = split
            if isinstance(X_split_train, pd.DataFrame):
                X_split_train = X_split_train.loc[:, selected_features].copy()
            if isinstance(X_split_val, pd.DataFrame):
                X_split_val = X_split_val.loc[:, selected_features].copy()
            subset_splits.append(
                (X_split_train, X_split_val, y_split_train, y_split_val, *rest)
            )
        return subset_splits

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

    def _export_eval_data(
        self,
        dataset: "PreparedDataset",
        *,
        ordinal_calibration: Dict[str, object] | None,
    ) -> None:
        """Persist the preprocessed eval split for downstream interpretability scripts."""
        if dataset.X_eval is None or dataset.y_eval is None:
            return
        try:
            from src.preprocessing.prudential_features import get_feature_lists
            from src.interpretability.utils.paths import eval_run_dir

            eval_dir = eval_run_dir(
                Path("outputs"),
                self.config.preprocessing.recipe,
                self.config.trainer.experiment_name,
            )
            dataset.X_eval.to_parquet(eval_dir / "X_eval.parquet", index=False)
            dataset.y_eval.to_frame(name="Response").to_parquet(
                eval_dir / "y_eval.parquet", index=False
            )
            feature_names = list(dataset.X_eval.columns)
            (eval_dir / "feature_names.json").write_text(
                json.dumps(feature_names, indent=2)
            )
            # Export raw (pre-preprocessing) eval features for interpretable x-axes
            if dataset.X_eval_raw is not None:
                dataset.X_eval_raw.reset_index(drop=True).to_parquet(
                    eval_dir / "X_eval_raw.parquet", index=False
                )
            # Export feature type taxonomy so plots can annotate axes correctly
            taxonomy_source = dataset.X_eval_raw if dataset.X_eval_raw is not None else dataset.X_eval
            feat_lists = get_feature_lists(taxonomy_source)
            feature_type_map: dict[str, str] = {}
            for feat in feature_names:
                base_feat = feat
                for prefix in ("cb_", "qt_", "mm_"):
                    if feat.startswith(prefix):
                        base_feat = feat[len(prefix):]
                        break
                if feat.startswith("missing_"):
                    feature_type_map[feat] = "missing_indicator"
                elif base_feat in feat_lists["categorical"]:
                    feature_type_map[feat] = "categorical"
                elif base_feat in feat_lists["binary"]:
                    feature_type_map[feat] = "binary"
                elif base_feat in feat_lists["continuous"]:
                    feature_type_map[feat] = "continuous"
                elif base_feat in feat_lists["ordinal"]:
                    feature_type_map[feat] = "ordinal"
                else:
                    feature_type_map[feat] = "unknown"
            (eval_dir / "feature_types.json").write_text(
                json.dumps(feature_type_map, indent=2, sort_keys=True)
            )
            if ordinal_calibration is not None:
                (eval_dir / "ordinal_thresholds.json").write_text(
                    json.dumps(ordinal_calibration, indent=2, sort_keys=True)
                )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to export eval data: {exc}")

    @staticmethod
    def _resolve_ordinal_calibration(model: PrudentialModel) -> Dict[str, object] | None:
        payload = model.get_ordinal_calibration()
        if payload is None:
            return None

        thresholds = payload.get("thresholds")
        if thresholds is None:
            return None

        serializable_thresholds = [float(value) for value in thresholds]
        result: Dict[str, object] = {
            "method": str(payload.get("method", "optimized_thresholds")),
            "num_classes": int(payload.get("num_classes", 8)),
            "thresholds": serializable_thresholds,
        }
        if payload.get("source_split") is not None:
            result["source_split"] = str(payload["source_split"])
        if payload.get("optimized_qwk_on_source_split") is not None:
            result["optimized_qwk_on_source_split"] = float(
                payload["optimized_qwk_on_source_split"]
            )
        return result

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
        full_feature_names = list(dataset.all_feature_names or dataset.feature_names or [])

        if dataset.recipe == "xgboost_paper":
            processed, _ = paper_prep.transform(
                df,
                dataset.preprocess_artifacts["state"],
            )
            processed_df = processed.copy()
            return processed_df.loc[:, list(dataset.feature_names or processed_df.columns)].copy()

        if dataset.recipe == "kan_paper":
            base_state = dataset.preprocess_artifacts["baseline"]
            kan_state = dataset.preprocess_artifacts["kan"]
            processed_array, _ = kan_prep.transform(
                df,
                base_state,
                kan_state=kan_state,
            )
            processed_df = pd.DataFrame(processed_array, columns=full_feature_names)
            return processed_df.loc[:, list(dataset.feature_names or processed_df.columns)].copy()

        if dataset.recipe == "kan_sota":
            base_state = dataset.preprocess_artifacts["baseline"]
            sota_state = dataset.preprocess_artifacts["sota"]
            processed_array, _ = kan_sota_prep.transform(
                df,
                base_state,
                sota_state=sota_state,
            )
            processed_df = pd.DataFrame(processed_array, columns=full_feature_names)
            return processed_df.loc[:, list(dataset.feature_names or processed_df.columns)].copy()

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
            X_eval_raw=self._load_raw_eval_features(test_index),
            recipe=recipe,
            preprocess_artifacts=artifacts,
            feature_names=feature_names,
            X_train_inner=inner_train,
            y_train_inner=inner_train_y,
            X_val_inner=inner_val,
            y_val_inner=inner_val_y,
            all_feature_names=list(feature_names),
        )

    def _load_raw_eval_features(self, row_index) -> Optional[pd.DataFrame]:
        """Reconstruct raw eval rows using the original training CSV indices."""

        if row_index is None:
            return None

        try:
            raw_df = pd.read_csv(self.config.trainer.train_csv)
        except OSError as exc:
            print(f"Warning: failed to load raw eval rows from {self.config.trainer.train_csv}: {exc}")
            return None

        try:
            raw_eval = raw_df.loc[row_index].copy()
        except KeyError as exc:
            print(f"Warning: eval row indices were not found in the raw training CSV: {exc}")
            return None

        if "Response" in raw_eval.columns:
            raw_eval = raw_eval.drop(columns=["Response"])
        return raw_eval


def run_train(
    config: ExperimentConfig,
    *,
    device: str,
    random_seed: int | None = None,
) -> TrainingArtifacts:
    """Resolve the random seed, run the trainer, and return the artifacts."""

    resolved_seed = random_seed if random_seed is not None else set_global_seed(config.trainer.seed)
    trainer = Trainer(config=config, device=device, random_seed=resolved_seed)
    return trainer.run()
