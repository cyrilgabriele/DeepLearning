"""Optuna hyperparameter sweep for all models on Prudential dataset.

Usage:
    .venv/Scripts/python.exe -m src.sweep --model chebykan --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --model xgb --n-trials 50 --preprocessing paper_base
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

import yaml
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.trial import Trial

from src.data.dataset import PrudentialDataModule, PREPROCESSING_PIPELINES
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.models.xgb_baseline import XGBBaseline
from src.models.xgboost_paper import XGBoostPaperModel
from src.metrics.qwk import optimize_thresholds


_SWEEP_DIR = Path(_PROJECT_ROOT) / "sweeps"
_CONFIG_DIR = Path(_PROJECT_ROOT) / "configs" / "model"

MODELS = ["chebykan", "bsplinekan", "fourierkan", "mlp", "xgb", "xgb_paper"]
PREPROCESSINGS = list(PREPROCESSING_PIPELINES.keys())


def create_datamodule(preprocessing: str = "kan_paper") -> PrudentialDataModule:
    data_path = Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment" / "train.csv"
    dm = PrudentialDataModule(
        data_path=str(data_path),
        batch_size=256,
        num_workers=0,
        seed=42,
        preprocessing=preprocessing,
    )
    dm.setup()
    return dm


def _get_dm_with_batch_size(dm: PrudentialDataModule, batch_size: int) -> PrudentialDataModule:
    if batch_size == dm.batch_size:
        return dm
    dm_trial = PrudentialDataModule(
        data_path=dm.data_path,
        batch_size=batch_size,
        num_workers=0,
        seed=dm.seed,
        preprocessing=dm.preprocessing,
    )
    # Reuse already-computed data to avoid reprocessing
    dm_trial._num_features = dm._num_features
    dm_trial.preprocessor = dm.preprocessor
    dm_trial.train_dataset = dm.train_dataset
    dm_trial.val_dataset = dm.val_dataset
    dm_trial.X_train = dm.X_train
    dm_trial.y_train = dm.y_train
    dm_trial.X_val = dm.X_val
    dm_trial.y_val = dm.y_val
    dm_trial.feature_names = dm.feature_names
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


# ── XGBoost Paper objective (auto-tuning, no Optuna needed) ──

def run_xgb_paper(dm: PrudentialDataModule, preprocessing: str = "kan_paper") -> float:
    """Run the paper-faithful XGBoostPaperModel with built-in sequential tuning."""
    from sklearn.metrics import cohen_kappa_score

    print("\n  XGBoostPaperModel: auto_tune=True, sequential grid search (paper methodology)")
    print(f"  Grid: max_depth, min_child_weight, lr, subsample, colsample, alpha, lambda")

    t0 = time.time()
    model = XGBoostPaperModel(
        auto_tune=True,
        refit_full_training=True,
        n_estimators=500,
    )
    model.fit(
        dm.X_train,
        dm.y_train.astype(int),
        validation_data=(dm.X_val, dm.y_val.astype(int)),
    )
    duration = time.time() - t0

    # Use tuning kappa (evaluated BEFORE refit on train+val) as the
    # comparable validation score.  Evaluating model.predict(X_val) after
    # refit_full_training=True would test on data the model was trained on.
    val_qwk = model.best_kappa_ if model.best_kappa_ is not None else 0.0

    print(f"\n  XGBoostPaperModel COMPLETE")
    print(f"  QWK={val_qwk:.4f} (tuning, before refit) | time={duration:.1f}s")
    print(f"  Best params: {model.best_params_}")

    # Save results
    _SWEEP_DIR.mkdir(exist_ok=True)
    suffix = f"_{preprocessing}" if preprocessing != "kan_paper" else ""
    results = {
        "model": "xgb_paper",
        "preprocessing": preprocessing,
        "best_qwk": round(val_qwk, 4),
        "best_params": {k: v for k, v in model.best_params_.items()},
        "duration": round(duration, 1),
    }
    results_path = _SWEEP_DIR / f"xgb_paper{suffix}_best.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {results_path}")

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
        callbacks=[EarlyStopping(monitor="val/qwk", patience=10, mode="max")],
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
        callbacks=[EarlyStopping(monitor="val/qwk", patience=10, mode="max")],
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


# ── Config generation ──

def _save_tuned_config(model_name: str, best_params: dict, suffix: str = "") -> Path:
    """Generate a Hydra YAML config from the best sweep params."""
    bp = best_params
    n_layers = bp.get("n_layers", 1)
    widths = [bp[f"width_{i}"] for i in range(n_layers)] if "n_layers" in bp else []

    if model_name == "xgb":
        config = {
            "name": "xgb",
            "n_estimators": bp["n_estimators"],
            "max_depth": bp["max_depth"],
            "learning_rate": bp["learning_rate"],
            "subsample": round(bp["subsample"], 4),
            "colsample_bytree": round(bp["colsample_bytree"], 4),
            "min_child_weight": bp["min_child_weight"],
            "reg_alpha": float(f"{bp['reg_alpha']:.6g}"),
            "reg_lambda": float(f"{bp['reg_lambda']:.6g}"),
            "gamma": round(bp["gamma"], 4),
        }
    elif model_name == "mlp":
        config = {
            "name": "mlp",
            "widths": widths,
            "dropout": round(bp["dropout"], 4),
            "lr": float(f"{bp['lr']:.6g}"),
            "weight_decay": float(f"{bp['weight_decay']:.6g}"),
        }
    elif model_name == "chebykan":
        config = {
            "name": "chebykan",
            "kan_type": "chebykan",
            "widths": widths,
            "degree": bp["degree"],
            "grid_size": None,
            "lr": float(f"{bp['lr']:.6g}"),
            "weight_decay": float(f"{bp['weight_decay']:.6g}"),
        }
    elif model_name == "bsplinekan":
        config = {
            "name": "bsplinekan",
            "kan_type": "bsplinekan",
            "widths": widths,
            "degree": None,
            "grid_size": bp["grid_size"],
            "spline_order": 3,
            "lr": float(f"{bp['lr']:.6g}"),
            "weight_decay": float(f"{bp['weight_decay']:.6g}"),
        }
    elif model_name == "fourierkan":
        config = {
            "name": "fourierkan",
            "kan_type": "fourierkan",
            "widths": widths,
            "degree": None,
            "grid_size": bp["grid_size"],
            "lr": float(f"{bp['lr']:.6g}"),
            "weight_decay": float(f"{bp['weight_decay']:.6g}"),
        }
    else:
        return Path("/dev/null")

    config_name = f"{model_name}{suffix}_tuned"
    config_path = _CONFIG_DIR / f"{config_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ── Sweep runner ──

def run_sweep(model_name: str, n_trials: int, timeout: int | None, dm: PrudentialDataModule, preprocessing: str = "kan_paper"):
    _SWEEP_DIR.mkdir(exist_ok=True)
    suffix = f"_{preprocessing}" if preprocessing != "kan_paper" else ""
    db_path = _SWEEP_DIR / f"{model_name}{suffix}_sweep.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{model_name}{suffix}_sweep",
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

    print(f"\nStarting {model_name} sweep ({preprocessing}): {n_trials} trials")
    print(f"Storage: {db_path}")
    print("=" * 70)

    study.optimize(obj_fn, n_trials=n_trials, timeout=timeout)

    # Print results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'=' * 70}")
    print(f"{model_name.upper()} SWEEP COMPLETE ({preprocessing}) — {len(completed)} trials")
    print(f"{'=' * 70}")
    print(f"Best QWK: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save results
    results = {
        "model": model_name,
        "preprocessing": preprocessing,
        "best_qwk": round(study.best_value, 4),
        "best_params": study.best_params,
        "total_trials": len(completed),
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": dict(study.best_trial.user_attrs),
    }
    results_path = _SWEEP_DIR / f"{model_name}{suffix}_best.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Generate tuned config
    config_path = _save_tuned_config(model_name, study.best_params, suffix)

    # Top 10
    top = sorted(completed, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:10]
    print(f"\nTop 10:")
    for rank, t in enumerate(top, 1):
        if t.value is None:
            continue
        dur = t.user_attrs.get("duration", "?")
        print(f"  {rank}. QWK={t.value:.4f} | {t.params} | time={dur}s")

    print(f"\nSaved: {results_path}")
    print(f"Config: {config_path}")
    print(f"  Run with: python -m src.train model={config_path.stem}")
    return study


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for Prudential models")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="Model to sweep")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds")
    parser.add_argument("--preprocessing", type=str, default="kan_paper", choices=PREPROCESSINGS, help="Preprocessing pipeline")
    args = parser.parse_args()

    L.seed_everything(42)

    print(f"Loading data (preprocessing: {args.preprocessing})...")
    dm = create_datamodule(preprocessing=args.preprocessing)
    print(f"Features: {dm.num_features} | Train: {len(dm.train_dataset)} | Val: {len(dm.val_dataset)}")

    if args.model == "xgb_paper":
        run_xgb_paper(dm, preprocessing=args.preprocessing)
    else:
        run_sweep(args.model, args.n_trials, args.timeout, dm, preprocessing=args.preprocessing)


if __name__ == "__main__":
    main()
