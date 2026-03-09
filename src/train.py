import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
# regardless of whether we run `python src/train.py` or `python main.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import numpy as np

from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.models.xgb_baseline import XGBBaseline
from src.metrics.qwk import optimize_thresholds

# Directory for storing per-run results (for summary table)
_RESULTS_DIR = Path(_PROJECT_ROOT) / "outputs" / "results"


def build_model(cfg: DictConfig, num_features: int):
    """Instantiate the correct model from config."""
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
    """Save a single run result to JSON for the summary table."""
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
    """Print a comparison table from all saved results."""
    if not _RESULTS_DIR.exists():
        print("No results found. Run training first.")
        return

    results = []
    for f in sorted(_RESULTS_DIR.glob("*.json")):
        results.append(json.loads(f.read_text()))

    if not results:
        print("No results found.")
        return

    results.sort(key=lambda r: r["val_qwk"], reverse=True)

    print("\n" + "=" * 65)
    print(f"{'Model':<16} {'Val QWK':>10} {'Params':>10} {'Epochs':>8} {'Time':>10}")
    print("-" * 65)
    for r in results:
        time_str = f"{r['duration_s']:.1f}s"
        print(f"{r['model']:<16} {r['val_qwk']:>10.4f} {r['params']:>10,} {r['epochs']:>8} {time_str:>10}")
    print("=" * 65)
    best = results[0]
    print(f"Best: {best['model']} (QWK = {best['val_qwk']:.4f})")
    print()


def train_xgb(cfg: DictConfig, dm: PrudentialDataModule):
    """Train and evaluate XGBoost baseline."""
    t0 = time.time()
    model = XGBBaseline(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
    )
    model.fit(dm.X_train, dm.y_train,
              eval_set=[(dm.X_val, dm.y_val)])

    train_qwk = model.evaluate(dm.X_train, dm.y_train)
    val_qwk = model.evaluate(dm.X_val, dm.y_val)
    duration = time.time() - t0
    print(f"XGBoost — Train QWK: {train_qwk:.4f}, Val QWK: {val_qwk:.4f}")

    _save_result("xgb", val_qwk, params=0, duration=duration,
                 epochs=cfg.model.n_estimators, cfg=cfg)
    return model


def train_neural(cfg: DictConfig, dm: PrudentialDataModule):
    """Train a neural model (TabKAN or MLP) with Lightning."""
    t0 = time.time()
    model = build_model(cfg, dm.num_features)

    loggers = []
    try:
        loggers.append(TensorBoardLogger("logs", name=cfg.model.name))
    except Exception:
        pass
    loggers.append(CSVLogger("logs", name=cfg.model.name))

    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stopping_patience,
            mode="min",
        ),
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename=f"{cfg.model.name}-{{epoch}}-{{val/loss:.4f}}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )

    trainer.fit(model, dm)

    # Post-training threshold optimization
    model.eval()
    val_preds = []
    val_targets = []
    import torch
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y = batch
            y_hat = model(x)
            val_preds.append(y_hat.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()
    thresholds, final_qwk = optimize_thresholds(targets, preds)
    duration = time.time() - t0
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.model.name} — Final Val QWK (optimized thresholds): {final_qwk:.4f}")
    print(f"Thresholds: {thresholds}")

    _save_result(cfg.model.name, final_qwk, params=num_params,
                 duration=duration, epochs=trainer.current_epoch + 1, cfg=cfg)

    return model, thresholds


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed)

    data_path = Path(hydra.utils.get_original_cwd()) / cfg.data.path

    dm = PrudentialDataModule(
        data_path=str(data_path),
        val_split=cfg.data.val_split,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        use_sota=cfg.data.use_sota,
        missing_threshold=cfg.data.missing_threshold,
        seed=cfg.seed,
    )
    dm.setup()

    if cfg.model.name == "xgb":
        train_xgb(cfg, dm)
    else:
        train_neural(cfg, dm)

    # Print summary table after each run (shows all results so far)
    print_summary_table()


if __name__ == "__main__":
    main()
