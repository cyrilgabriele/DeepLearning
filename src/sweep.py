"""Optuna hyperparameter sweep for all models on Prudential dataset.

Usage:
    .venv/Scripts/python.exe -m src.sweep --model chebykan --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --model xgb --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --model mlp --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --model bsplinekan --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --model fourierkan --n-trials 50
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
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.trial import Trial

from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.models.xgb_baseline import XGBBaseline
from src.metrics.qwk import optimize_thresholds


_SWEEP_DIR = Path(_PROJECT_ROOT) / "sweeps"


def create_datamodule() -> PrudentialDataModule:
    data_path = Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment" / "train.csv"
    dm = PrudentialDataModule(
        data_path=str(data_path),
        val_split=0.2,
        batch_size=256,
        num_workers=0,
        missing_threshold=0.5,
        seed=42,
    )
    dm.setup()
    return dm


def _get_dm_with_batch_size(dm: PrudentialDataModule, batch_size: int) -> PrudentialDataModule:
    if batch_size == dm.batch_size:
        return dm
    dm_trial = PrudentialDataModule(
        data_path=str(Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment" / "train.csv"),
        val_split=0.2,
        batch_size=batch_size,
        num_workers=0,
        missing_threshold=0.5,
        seed=42,
    )
    dm_trial.setup()
    dm_trial.preprocessor = dm.preprocessor
    dm_trial.train_dataset = dm.train_dataset
    dm_trial.val_dataset = dm.val_dataset
    dm_trial.X_train = dm.X_train
    dm_trial.y_train = dm.y_train
    dm_trial.X_val = dm.X_val
    dm_trial.y_val = dm.y_val
    dm_trial._num_features = dm._num_features
    return dm_trial


# ── XGBoost objective ──

def objective_xgb(trial: Trial, dm: PrudentialDataModule) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 2000)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)

    print(f"\n  Trial {trial.number}: depth={max_depth} n_est={n_estimators} lr={learning_rate:.4f} sub={subsample:.2f} col={colsample_bytree:.2f}")

    t0 = time.time()
    model = XGBBaseline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
    )
    model.fit(dm.X_train, dm.y_train, eval_set=[(dm.X_val, dm.y_val)])
    duration = time.time() - t0

    y_cont = model.predict(dm.X_val)
    _, val_qwk = optimize_thresholds(dm.y_val, y_cont)

    print(f"  -> QWK={val_qwk:.4f} | time={duration:.1f}s")

    trial.set_user_attr("duration", round(duration, 1))
    trial.set_user_attr("n_estimators", n_estimators)
    return val_qwk


# ── MLP objective ──

def objective_mlp(trial: Trial, dm: PrudentialDataModule) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    widths = []
    for i in range(n_layers):
        w = trial.suggest_categorical(f"width_{i}", [64, 128, 256, 512])
        widths.append(w)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])

    dm_trial = _get_dm_with_batch_size(dm, batch_size)

    model = MLPBaseline(
        in_features=dm.num_features,
        widths=widths,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Trial {trial.number}: widths={widths} dropout={dropout:.2f} lr={lr:.5f} bs={batch_size} params={num_params:,}")

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[EarlyStopping(monitor="val/loss", patience=10, mode="min")],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    t0 = time.time()
    trainer.fit(model, dm_trial)
    duration = time.time() - t0

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in dm_trial.val_dataloader():
            x, y = batch
            if next(model.parameters()).is_cuda:
                x = x.cuda()
            val_preds.append(model(x).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()
    _, val_qwk = optimize_thresholds(targets, preds)

    epochs = trainer.current_epoch + 1
    print(f"  -> QWK={val_qwk:.4f} | epochs={epochs} | time={duration:.1f}s | params={num_params:,}")

    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("duration", round(duration, 1))
    trial.set_user_attr("num_params", num_params)
    return val_qwk


# ── KAN objective (chebykan / bsplinekan / fourierkan) ──

def objective_kan(trial: Trial, dm: PrudentialDataModule, kan_type: str) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    widths = []
    for i in range(n_layers):
        w = trial.suggest_categorical(f"width_{i}", [32, 64, 128, 256])
        widths.append(w)

    if kan_type == "chebykan":
        degree = trial.suggest_int("degree", 2, 6)
        grid_size = 4
    elif kan_type == "bsplinekan":
        degree = 3
        grid_size = trial.suggest_int("grid_size", 3, 10)
    else:  # fourierkan
        degree = 3
        grid_size = trial.suggest_int("grid_size", 3, 10)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])

    dm_trial = _get_dm_with_batch_size(dm, batch_size)

    model = TabKAN(
        in_features=dm.num_features,
        widths=widths,
        kan_type=kan_type,
        degree=degree,
        grid_size=grid_size,
        lr=lr,
        weight_decay=weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters())
    extra = f"degree={degree}" if kan_type == "chebykan" else f"grid={grid_size}"
    print(f"\n  Trial {trial.number}: widths={widths} {extra} lr={lr:.5f} bs={batch_size} params={num_params:,}")

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[EarlyStopping(monitor="val/loss", patience=10, mode="min")],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    t0 = time.time()
    trainer.fit(model, dm_trial)
    duration = time.time() - t0

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in dm_trial.val_dataloader():
            x, y = batch
            if next(model.parameters()).is_cuda:
                x = x.cuda()
            val_preds.append(model(x).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()
    _, val_qwk = optimize_thresholds(targets, preds)

    epochs = trainer.current_epoch + 1
    print(f"  -> QWK={val_qwk:.4f} | epochs={epochs} | time={duration:.1f}s | params={num_params:,}")

    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("duration", round(duration, 1))
    trial.set_user_attr("num_params", num_params)
    return val_qwk


# ── Main ──

MODELS = ["chebykan", "bsplinekan", "fourierkan", "mlp", "xgb"]


def run_sweep(model_name: str, n_trials: int, timeout: int | None, dm: PrudentialDataModule):
    _SWEEP_DIR.mkdir(exist_ok=True)
    db_path = _SWEEP_DIR / f"{model_name}_sweep.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{model_name}_sweep",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    if model_name == "xgb":
        obj_fn = lambda trial: objective_xgb(trial, dm)
    elif model_name == "mlp":
        obj_fn = lambda trial: objective_mlp(trial, dm)
    else:
        obj_fn = lambda trial: objective_kan(trial, dm, model_name)

    print(f"\nStarting {model_name} sweep: {n_trials} trials")
    print(f"Storage: {db_path}")
    print("=" * 70)

    study.optimize(obj_fn, n_trials=n_trials, timeout=timeout)

    # Print results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'=' * 70}")
    print(f"{model_name.upper()} SWEEP COMPLETE — {len(completed)} trials")
    print(f"{'=' * 70}")
    print(f"Best QWK: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save results
    results = {
        "model": model_name,
        "best_qwk": round(study.best_value, 4),
        "best_params": study.best_params,
        "total_trials": len(completed),
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": dict(study.best_trial.user_attrs),
    }
    results_path = _SWEEP_DIR / f"{model_name}_best.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Top 10
    top = sorted(completed, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:10]
    print(f"\nTop 10:")
    for rank, t in enumerate(top, 1):
        if t.value is None:
            continue
        dur = t.user_attrs.get("duration", "?")
        print(f"  {rank}. QWK={t.value:.4f} | {t.params} | time={dur}s")

    print(f"\nSaved: {results_path}")
    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for Prudential models")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="Model to sweep")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds")
    args = parser.parse_args()

    L.seed_everything(42)

    print(f"Loading data...")
    dm = create_datamodule()
    print(f"Features: {dm.num_features} | Train: {len(dm.train_dataset)} | Val: {len(dm.val_dataset)}")

    run_sweep(args.model, args.n_trials, args.timeout, dm)


if __name__ == "__main__":
    main()
