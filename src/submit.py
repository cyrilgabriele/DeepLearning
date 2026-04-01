"""Generate Kaggle submission CSVs.

Usage:
    .venv/Scripts/python.exe -m src.submit --model xgb_paper --preprocessing kan_paper
    .venv/Scripts/python.exe -m src.submit --model xgb --preprocessing xgb_paper
    .venv/Scripts/python.exe -m src.submit --model bsplinekan --preprocessing kan_paper
    .venv/Scripts/python.exe -m src.submit --model chebykan --preprocessing kan_paper
    .venv/Scripts/python.exe -m src.submit --model fourierkan --preprocessing kan_paper
"""
import sys
import time
import json
import argparse
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import lightning as L
from sklearn.metrics import cohen_kappa_score
from lightning.pytorch.callbacks import EarlyStopping

from src.data.dataset import PrudentialDataModule, PREPROCESSING_PIPELINES
from src.models.xgb_baseline import XGBBaseline
from src.models.xgboost_paper import XGBoostPaperModel
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.metrics.qwk import optimize_thresholds

DATA_PATH = Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment"
SUBMIT_DIR = Path(_PROJECT_ROOT) / "submissions"
SWEEP_DIR = Path(_PROJECT_ROOT) / "sweeps"

ALL_MODELS = ["xgb_paper", "xgb", "bsplinekan", "chebykan", "fourierkan", "mlp"]


def make_submission(model_name: str, preprocessing: str):
    SUBMIT_DIR.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data (preprocessing: {preprocessing})...")
    dm = PrudentialDataModule(
        data_path=str(DATA_PATH / "train.csv"),
        batch_size=256,
        num_workers=0,
        seed=42,
        preprocessing=preprocessing,
    )
    dm.setup()
    print(f"Features: {dm.num_features} | Train: {dm.X_train.shape} | Val: {dm.X_val.shape}")

    # Train model
    if model_name == "xgb_paper":
        print("\nTraining XGBoostPaperModel (auto_tune=True, refit_full=True)...")
        t0 = time.time()
        model = XGBoostPaperModel(auto_tune=True, refit_full_training=True, n_estimators=500)
        model.fit(
            dm.X_train,
            dm.y_train.astype(int),
            validation_data=(dm.X_val, dm.y_val.astype(int)),
        )
        duration = time.time() - t0

        preds_val = model.predict(dm.X_val)
        val_qwk = cohen_kappa_score(dm.y_val.astype(int), preds_val, weights="quadratic")
        print(f"Val QWK: {val_qwk:.4f} | Time: {duration:.1f}s")
        print(f"Best params: {model.best_params_}")

    elif model_name in ("bsplinekan", "chebykan", "fourierkan", "mlp"):
        # Load best params from sweep
        suffix = f"_{preprocessing}" if preprocessing != "kan_paper" else ""
        sweep_path = SWEEP_DIR / f"{model_name}{suffix}_best.json"
        if not sweep_path.exists():
            sweep_path = SWEEP_DIR / f"{model_name}_best.json"
        print(f"Loading best params from {sweep_path}...")
        bp = json.loads(sweep_path.read_text())["best_params"]
        n_layers = bp["n_layers"]
        widths = [bp[f"width_{i}"] for i in range(n_layers)]
        print(f"Best params: widths={widths}, {bp}")

        if model_name == "mlp":
            model = MLPBaseline(
                in_features=dm.num_features,
                widths=widths,
                dropout=bp.get("dropout", 0.0),
                lr=bp["lr"],
                weight_decay=bp["weight_decay"],
            )
        else:
            model = TabKAN(
                in_features=dm.num_features,
                widths=widths,
                kan_type=model_name,
                degree=bp.get("degree", 3),
                grid_size=bp.get("grid_size", 4),
                lr=bp["lr"],
                weight_decay=bp["weight_decay"],
            )

        batch_size = bp.get("batch_size", 256)
        dm_train = PrudentialDataModule(
            data_path=str(DATA_PATH / "train.csv"),
            batch_size=batch_size,
            num_workers=0,
            seed=42,
            preprocessing=preprocessing,
        )
        dm_train._num_features = dm._num_features
        dm_train.preprocessor = dm.preprocessor
        dm_train.train_dataset = dm.train_dataset
        dm_train.val_dataset = dm.val_dataset
        dm_train.X_train = dm.X_train
        dm_train.y_train = dm.y_train
        dm_train.X_val = dm.X_val
        dm_train.y_val = dm.y_val
        dm_train.feature_names = dm.feature_names

        print(f"\nTraining {model_name} (widths={widths}, batch_size={batch_size})...")
        trainer = L.Trainer(
            max_epochs=100,
            accelerator="auto",
            callbacks=[EarlyStopping(monitor="val/qwk", patience=10, mode="max")],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=True,
        )
        t0 = time.time()
        trainer.fit(model, dm_train)
        duration = time.time() - t0

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in dm_train.val_dataloader():
                x, y = batch
                if next(model.parameters()).is_cuda:
                    x = x.cuda()
                val_preds.append(model(x).cpu().numpy())
                val_targets.append(y.cpu().numpy())

        preds = np.concatenate(val_preds).flatten()
        targets = np.concatenate(val_targets).flatten()
        _, val_qwk = optimize_thresholds(targets, preds)
        print(f"Val QWK: {val_qwk:.4f} | Epochs: {trainer.current_epoch + 1} | Time: {duration:.1f}s")

    elif model_name == "xgb":
        # Load best params from sweep
        suffix = f"_{preprocessing}" if preprocessing != "kan_paper" else ""
        sweep_path = Path(_PROJECT_ROOT) / "sweeps" / f"xgb{suffix}_best.json"
        if not sweep_path.exists():
            # Try alternative naming
            for p in Path(_PROJECT_ROOT).glob("sweeps/xgb*best.json"):
                if preprocessing in str(p) or (preprocessing == "kan_paper" and "kan" not in str(p) and "xgb_paper" not in str(p)):
                    sweep_path = p
                    break

        print(f"Loading best params from {sweep_path}...")
        bp = json.loads(sweep_path.read_text())["best_params"]
        print(f"Best params: {bp}")

        print("\nTraining XGBBaseline with Optuna-tuned params...")
        t0 = time.time()
        model = XGBBaseline(
            n_estimators=bp["n_estimators"],
            max_depth=bp["max_depth"],
            learning_rate=bp["learning_rate"],
            subsample=bp.get("subsample", 1.0),
            colsample_bytree=bp.get("colsample_bytree", 1.0),
            min_child_weight=bp.get("min_child_weight", 1),
            reg_alpha=bp.get("reg_alpha", 0),
            reg_lambda=bp.get("reg_lambda", 1),
            gamma=bp.get("gamma", 0),
        )
        # Train on full train+val for best submission
        X_full = np.concatenate([dm.X_train, dm.X_val], axis=0)
        y_full = np.concatenate([dm.y_train, dm.y_val], axis=0)
        model.fit(X_full, y_full)
        duration = time.time() - t0

        # Evaluate on val for reporting
        y_cont_val = model.predict(dm.X_val)
        _, val_qwk = optimize_thresholds(dm.y_val, y_cont_val)
        print(f"Val QWK: {val_qwk:.4f} | Time: {duration:.1f}s")

    # Predict on test.csv using fitted preprocessor states from training
    print("\nPredicting on test.csv...")
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    ids = test_df["Id"].values

    # Run pipeline on train to get fitted states
    prep_module = PREPROCESSING_PIPELINES[preprocessing]
    pipeline_out = prep_module.run_pipeline(str(DATA_PATH / "train.csv"), random_seed=42)

    # Get the fitted preprocessor class and states
    raw_states = pipeline_out.get("preprocessor_state", {})

    if hasattr(prep_module, 'KANPreprocessor'):
        preprocessor = prep_module.KANPreprocessor()
        X_test_processed, _ = preprocessor.transform(
            test_df, raw_states["baseline"], kan_state=raw_states.get("kan"),
        )
    elif hasattr(prep_module, 'KANSOTAPreprocessor'):
        preprocessor = prep_module.KANSOTAPreprocessor()
        X_test_processed, _ = preprocessor.transform(
            test_df, raw_states["baseline"], sota_state=raw_states.get("sota"),
        )
    elif hasattr(prep_module, 'XGBoostPaperPreprocessor'):
        preprocessor = prep_module.XGBoostPaperPreprocessor()
        # xgb_paper returns state directly, not as a dict
        base_state = raw_states if not isinstance(raw_states, dict) else raw_states.get("baseline", raw_states)
        X_test_processed, _ = preprocessor.transform(test_df, base_state)
    else:
        raise ValueError(f"Cannot find preprocessor class in {preprocessing}")

    X_test_np = np.asarray(X_test_processed, dtype=np.float32)
    print(f"Test shape: {X_test_np.shape}")

    if model_name == "xgb_paper":
        preds_test = model.predict(X_test_np)
    elif model_name in ("bsplinekan", "chebykan", "fourierkan", "mlp"):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test_np, dtype=torch.float32)
            if next(model.parameters()).is_cuda:
                X_tensor = X_tensor.cuda()
            y_cont_test = model(X_tensor).cpu().numpy().flatten()
        thresholds, _ = optimize_thresholds(dm.y_val, preds)  # preds from val
        from src.metrics.qwk import _apply_thresholds
        preds_test = np.clip(_apply_thresholds(y_cont_test, thresholds), 1, 8).astype(int)
    else:
        y_cont_test = model.predict(X_test_np)
        thresholds, _ = optimize_thresholds(dm.y_val, model.predict(dm.X_val))
        from src.metrics.qwk import _apply_thresholds
        preds_test = np.clip(_apply_thresholds(y_cont_test, thresholds), 1, 8).astype(int)

    preds_test = np.clip(preds_test, 1, 8).astype(int)
    print(f"Predictions: classes {np.unique(preds_test)}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({"Id": ids, "Response": preds_test})
    sub_path = SUBMIT_DIR / f"submission_{model_name}_{preprocessing}_{timestamp}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")
    print(f"Rows: {len(sub)} | Val QWK: {val_qwk:.4f}")
    print(sub["Response"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--preprocessing", type=str, required=True, choices=list(PREPROCESSING_PIPELINES.keys()))
    args = parser.parse_args()
    make_submission(args.model, args.preprocessing)


if __name__ == "__main__":
    main()
