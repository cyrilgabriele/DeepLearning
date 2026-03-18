import sys
import json
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import numpy as np
import torch

from src.configs import set_global_seed
from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.models.xgb_baseline import XGBBaseline
from src.metrics.qwk import optimize_thresholds
from lightning.pytorch.callbacks import Callback

from src.utils import (
    make_run_dir, setup_logger, JSONLLogger, EpochMetricsCSV,
    log_preprocessing, log_model, log_forward_pass, log_epoch, log_output, log_training_complete,
)


class EpochLoggerCallback(Callback):
    """Lightning callback that logs epoch metrics to our own loggers."""

    def __init__(self, log, jl, epoch_csv):
        super().__init__()
        self._log = log
        self._jl = jl
        self._epoch_csv = epoch_csv
        self._train_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self._train_losses.append(outputs["loss"].item())
        elif isinstance(outputs, torch.Tensor):
            self._train_losses.append(outputs.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        # Average train loss for this epoch
        train_loss = np.mean(self._train_losses) if self._train_losses else 0.0
        self._train_losses.clear()

        # Get val metrics from Lightning's logged values
        val_loss = trainer.callback_metrics.get("val/loss", torch.tensor(0.0)).item()
        val_qwk = trainer.callback_metrics.get("val/qwk", torch.tensor(0.0)).item()

        log_epoch(self._log, self._jl, self._epoch_csv, epoch, train_loss, val_loss, val_qwk)

_RESULTS_DIR = Path(_PROJECT_ROOT) / "outputs" / "results"


def build_model(cfg: DictConfig, num_features: int):
    if cfg.model.name == "xgb":
        return XGBBaseline(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
        )
    elif cfg.model.name == "mlp":
        return MLPBaseline(
            in_features=num_features,
            widths=list(cfg.model.widths),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )
    else:
        return TabKAN(
            in_features=num_features,
            widths=list(cfg.model.widths),
            kan_type=cfg.model.kan_type,
            degree=cfg.model.get("degree", 3),
            grid_size=cfg.model.get("grid_size", 4),
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )


def _save_result(name: str, val_qwk: float, params: int, duration: float, epochs: int, cfg: DictConfig):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": name,
        "val_qwk": round(val_qwk, 4),
        "params": params,
        "duration_s": round(duration, 1),
        "epochs": epochs,
        "config": OmegaConf.to_container(cfg.model, resolve=True),
    }
    path = _RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))


def print_summary_table():
    if not _RESULTS_DIR.exists():
        return
    results = [json.loads(f.read_text()) for f in sorted(_RESULTS_DIR.glob("*.json"))]
    if not results:
        return
    results.sort(key=lambda r: r["val_qwk"], reverse=True)
    print("\n" + "=" * 65)
    print(f"{'Model':<16} {'Val QWK':>10} {'Params':>10} {'Epochs':>8} {'Time':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<16} {r['val_qwk']:>10.4f} {r['params']:>10,} {r['epochs']:>8} {r['duration_s']:.1f}s")
    print("=" * 65)
    print(f"Best: {results[0]['model']} (QWK = {results[0]['val_qwk']:.4f})\n")


def train_xgb(cfg: DictConfig, dm: PrudentialDataModule):
    t0 = time.time()
    model = XGBBaseline(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
    )
    model.fit(dm.X_train, dm.y_train, eval_set=[(dm.X_val, dm.y_val)])
    val_qwk = model.evaluate(dm.X_val, dm.y_val)
    duration = time.time() - t0
    print(f"XGBoost — Val QWK: {val_qwk:.4f}")
    _save_result("xgb", val_qwk, params=0, duration=duration, epochs=cfg.model.n_estimators, cfg=cfg)
    return model


def train_neural(cfg: DictConfig, dm: PrudentialDataModule):
    # ── Set up run directory with both loggers ──
    run_dir = make_run_dir(tag=cfg.model.name)
    log = setup_logger(run_dir)
    jl = JSONLLogger(run_dir)

    # Save config for reproducibility
    config_path = Path(run_dir) / "config.json"
    config_path.write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    jl.log("config", "experiment", **OmegaConf.to_container(cfg, resolve=True))

    # ── Preprocessing ──
    log_preprocessing(log, jl, dm, run_dir)

    t0 = time.time()
    model = build_model(cfg, dm.num_features)

    # ── Model ──
    log_model(log, jl, model, cfg, run_dir)

    # ── Forward pass (before training) ──
    log_forward_pass(log, jl, model, dm)

    # ── Training with per-epoch logging via callback ──
    epoch_csv = EpochMetricsCSV(run_dir)
    epoch_logger_cb = EpochLoggerCallback(log, jl, epoch_csv)

    orig_cwd = hydra.utils.get_original_cwd()
    log_dir = str(Path(orig_cwd) / "logs")

    loggers = []
    try:
        loggers.append(TensorBoardLogger(log_dir, name=cfg.model.name))
    except Exception:
        pass
    loggers.append(CSVLogger(log_dir, name=cfg.model.name))

    callbacks = [
        epoch_logger_cb,
        EarlyStopping(monitor="val/loss", patience=cfg.train.early_stopping_patience, mode="min"),
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1,
                        dirpath=str(Path(orig_cwd) / "checkpoints"),
                        filename=f"{cfg.model.name}-{{epoch}}-{{val/loss:.4f}}"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )

    log.info("")
    log.info("=" * 70)
    log.info("TRAINING")
    log.info("=" * 70)
    log.info(f"Max epochs: {cfg.train.max_epochs} | Batch size: {cfg.train.batch_size} | "
             f"Early stopping patience: {cfg.train.early_stopping_patience}")
    log.info(f"{'Epoch':>6s} {'Train Loss':>12s} {'Val Loss':>12s} {'Val QWK':>10s}")
    log.info("-" * 44)

    trainer.fit(model, dm)

    epoch_csv.close()
    log.info("-" * 44)
    log.info(f"Training complete. Epochs: {trainer.current_epoch + 1}")
    log.info("Saved: epoch_metrics.csv")
    log.info("=" * 70)

    # ── Post-training threshold optimization ──
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y = batch
            val_preds.append(model(x).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()
    thresholds, final_qwk = optimize_thresholds(targets, preds)
    duration = time.time() - t0
    num_params = sum(p.numel() for p in model.parameters())

    # ── Output ──
    log_output(log, jl, preds, targets, thresholds, final_qwk, run_dir)
    log_training_complete(log, jl, cfg.model.name, trainer.current_epoch + 1,
                          duration, final_qwk, num_params, run_dir)

    jl.close()

    _save_result(cfg.model.name, final_qwk, params=num_params,
                 duration=duration, epochs=trainer.current_epoch + 1, cfg=cfg)

    return model, thresholds


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed = set_global_seed(cfg.seed)

    data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.path
    dm = PrudentialDataModule(
        data_path=str(data_path),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        seed=seed,
    )
    dm.setup()

    if cfg.model.name == "xgb":
        train_xgb(cfg, dm)
    else:
        train_neural(cfg, dm)

    print_summary_table()


if __name__ == "__main__":
    main()
