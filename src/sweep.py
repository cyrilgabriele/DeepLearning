"""Optuna hyperparameter sweep for ChebyKAN on Prudential dataset.

Usage:
    .venv/Scripts/python.exe -m src.sweep --n-trials 50
    .venv/Scripts/python.exe -m src.sweep --n-trials 100 --timeout 7200
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
from src.metrics.qwk import optimize_thresholds, _apply_thresholds


_SWEEP_DIR = Path(_PROJECT_ROOT) / "sweeps"


def create_datamodule(batch_size: int) -> PrudentialDataModule:
    data_path = Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment" / "train.csv"
    dm = PrudentialDataModule(
        data_path=str(data_path),
        val_split=0.2,
        batch_size=batch_size,
        num_workers=0,
        missing_threshold=0.5,
        seed=42,
    )
    dm.setup()
    return dm


def objective(trial: Trial, dm: PrudentialDataModule) -> float:
    # --- Sample hyperparameters ---
    n_layers = trial.suggest_int("n_layers", 1, 3)
    widths = []
    for i in range(n_layers):
        w = trial.suggest_categorical(f"width_{i}", [32, 64, 128, 256])
        widths.append(w)

    degree = trial.suggest_int("degree", 2, 6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])

    # Rebuild datamodule if batch size changed
    if batch_size != dm.batch_size:
        dm_trial = PrudentialDataModule(
            data_path=str(Path(_PROJECT_ROOT) / "data" / "prudential-life-insurance-assessment" / "train.csv"),
            val_split=0.2,
            batch_size=batch_size,
            num_workers=0,
            missing_threshold=0.5,
            seed=42,
        )
        dm_trial.setup()
        # Reuse the same preprocessor fit
        dm_trial.preprocessor = dm.preprocessor
        dm_trial.train_dataset = dm.train_dataset
        dm_trial.val_dataset = dm.val_dataset
        dm_trial.X_train = dm.X_train
        dm_trial.y_train = dm.y_train
        dm_trial.X_val = dm.X_val
        dm_trial.y_val = dm.y_val
        dm_trial._num_features = dm._num_features
    else:
        dm_trial = dm

    # --- Build model ---
    model = TabKAN(
        in_features=dm.num_features,
        widths=widths,
        kan_type="chebykan",
        degree=degree,
        lr=lr,
        weight_decay=weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Trial {trial.number}: widths={widths} degree={degree} lr={lr:.5f} wd={weight_decay:.6f} bs={batch_size} params={num_params:,}")

    # --- Train ---
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    t0 = time.time()
    trainer.fit(model, dm_trial)
    duration = time.time() - t0

    # --- Evaluate with threshold optimization ---
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

    # Store extra info
    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("duration", round(duration, 1))
    trial.set_user_attr("num_params", num_params)

    return val_qwk


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for ChebyKAN")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds for entire sweep")
    args = parser.parse_args()

    L.seed_everything(42)

    print("Loading data...")
    dm = create_datamodule(batch_size=256)
    print(f"Features: {dm.num_features} | Train: {len(dm.train_dataset)} | Val: {len(dm.val_dataset)}")

    # Create study with SQLite storage for persistence
    _SWEEP_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{_SWEEP_DIR / 'chebykan_sweep.db'}"
    study = optuna.create_study(
        study_name="chebykan_sweep",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"\nStarting sweep: {args.n_trials} trials")
    print(f"Results saved to: {_SWEEP_DIR}")
    print("=" * 70)

    study.optimize(
        lambda trial: objective(trial, dm),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # --- Print results ---
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Best QWK: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best config as a Hydra yaml
    bp = study.best_params
    n_layers = bp["n_layers"]
    widths = [bp[f"width_{i}"] for i in range(n_layers)]
    best_config = {
        "name": "chebykan",
        "kan_type": "chebykan",
        "widths": widths,
        "degree": bp["degree"],
        "grid_size": None,
        "lr": bp["lr"],
        "weight_decay": bp["weight_decay"],
    }
    best_train = {
        "batch_size": bp["batch_size"],
    }

    # Save results summary
    results_path = _SWEEP_DIR / "best_config.json"
    results_path.write_text(json.dumps({
        "best_qwk": round(study.best_value, 4),
        "best_model_config": best_config,
        "best_train_config": best_train,
        "total_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_trial_epochs": study.best_trial.user_attrs.get("epochs"),
        "best_trial_duration": study.best_trial.user_attrs.get("duration"),
        "best_trial_params": study.best_trial.user_attrs.get("num_params"),
    }, indent=2))

    # Save top 10 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:10]
    print(f"\n{'Rank':<6}{'QWK':>8}{'Widths':>20}{'Deg':>5}{'LR':>10}{'BS':>6}{'Epochs':>8}{'Time':>8}")
    print("-" * 75)
    for rank, t in enumerate(top_trials, 1):
        if t.value is None:
            continue
        n = t.params["n_layers"]
        w = [t.params[f"width_{i}"] for i in range(n)]
        print(f"{rank:<6}{t.value:>8.4f}{str(w):>20}{t.params['degree']:>5}{t.params['lr']:>10.5f}{t.params['batch_size']:>6}{t.user_attrs.get('epochs', '?'):>8}{t.user_attrs.get('duration', '?'):>8}")

    print(f"\nBest config saved to: {results_path}")
    print(f"To train with best config:")
    print(f"  python -m src.train model=chebykan model.widths='{widths}' model.degree={bp['degree']} model.lr={bp['lr']} model.weight_decay={bp['weight_decay']} train.batch_size={bp['batch_size']} train.max_epochs=100")


if __name__ == "__main__":
    main()
