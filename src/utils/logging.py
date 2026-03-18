"""Logging utilities for TabKAN training pipeline.

Run directory structure:
    runs/run_YYYYMMDD_HHMMSS_{model}/
    ├── config.json                     # Full Hydra config for reproducibility
    ├── metrics.jsonl                   # Machine-readable: one JSON object per event
    ├── train.log                       # Human-readable: full pipeline trace
    ├── data/
    │   ├── raw_sample.csv             # First 50 rows of raw input (before preprocessing)
    │   ├── processed_sample.csv       # Same rows after preprocessing (what model sees)
    │   ├── feature_stats.csv          # Per-feature: type, min, max, mean, std, unique, nulls
    │   ├── target_distribution.csv    # Class counts and percentages
    │   └── preprocessing_report.txt   # Full text report
    ├── model/
    │   ├── architecture.txt           # Model summary with layer shapes
    │   └── parameter_counts.csv       # Per-layer parameter counts
    └── training/
        ├── epoch_metrics.csv          # Per-epoch: epoch, train_loss, val_loss, val_qwk
        ├── predictions_sample.csv     # 100 sample predictions vs targets (after training)
        └── output_report.txt          # Threshold optimization details
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
try:  # Optional dependency: torch may be unavailable in lightweight test envs.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# ── Run directory ──


def make_run_dir(tag: str | None = None, runs_dir: str = "runs") -> str:
    """Create a timestamped run directory with subdirectories."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" if tag is None else f"run_{ts}_{tag}"
    run_dir = os.path.join(runs_dir, name)
    for sub in ["data", "model", "training"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


# ── Human-readable logger ──


def setup_logger(
    run_dir: str,
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """Configure 'tabkan' logger: INFO to console, DEBUG to train.log."""
    log_path = os.path.join(run_dir, "train.log")

    logger = logging.getLogger("tabkan")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging initialised -> {log_path}")
    return logger


def get_logger(
    name: str = "tabkan",
    *,
    level: str | int = "INFO",
) -> logging.Logger:
    """Return the shared project logger, attaching a console handler if needed."""

    logger = logging.getLogger(name)

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = int(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setLevel(level_value)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if logger.level > level_value or logger.level == logging.NOTSET:
        logger.setLevel(level_value)

    return logger


# ── Machine-readable logger (JSONL) ──


class JSONLLogger:
    """Append-only JSONL writer for structured metrics."""

    def __init__(self, run_dir: str, filename: str = "metrics.jsonl"):
        self._path = os.path.join(run_dir, filename)
        self._file = open(self._path, "a", encoding="utf-8")

    def log(self, stage: str, entry_type: str, **kwargs) -> None:
        record = {
            "stage": stage,
            "type": entry_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ── Epoch metrics CSV ──


class EpochMetricsCSV:
    """Append-only CSV writer for per-epoch metrics. One row per epoch."""

    def __init__(self, run_dir: str):
        self._path = os.path.join(run_dir, "training", "epoch_metrics.csv")
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "train_loss", "val_loss", "val_qwk"])
        self._file.flush()

    def log(self, epoch: int, train_loss: float, val_loss: float, val_qwk: float):
        self._writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_qwk:.4f}"])
        self._file.flush()

    def close(self):
        self._file.close()


# ── Formatting helpers ──


def fmt_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1e3:.1f}K)"
    return f"{n:,}"


# ── Pipeline logging functions ──


def log_preprocessing(log: logging.Logger, jl: JSONLLogger, dm, run_dir: str) -> None:
    """Log preprocessing: save raw/processed CSVs, feature stats, text report."""
    prep = dm.preprocessor
    fl = prep.feature_lists

    # Re-load to get column names and raw data
    df = pd.read_csv(dm.data_path)
    y = df["Response"]
    X_raw = df.drop(columns=["Response"])
    X_processed = prep.fit_transform(X_raw, y)
    col_names = list(X_processed.columns)

    X_all = np.concatenate([dm.X_train, dm.X_val], axis=0)
    y_all = np.concatenate([
        dm.y_train,
        dm.y_val,
    ])

    # ── Save raw sample CSV ──
    raw_sample = df.head(50)
    raw_sample.to_csv(os.path.join(run_dir, "data", "raw_sample.csv"), index=False)

    # ── Save processed sample CSV ──
    proc_sample = pd.DataFrame(X_processed.values[:50], columns=col_names)
    proc_sample.insert(0, "Response", y.values[:50])
    proc_sample.to_csv(os.path.join(run_dir, "data", "processed_sample.csv"), index=False)

    # ── Save target distribution CSV ──
    target_counts = pd.Series(y_all).value_counts().sort_index()
    target_df = pd.DataFrame({
        "class": target_counts.index.astype(int),
        "count": target_counts.values,
        "percentage": (target_counts.values / len(y_all) * 100).round(2),
    })
    target_df.to_csv(os.path.join(run_dir, "data", "target_distribution.csv"), index=False)

    # ── Build per-feature stats ──
    rows = []
    for ftype in ["categorical", "binary", "continuous", "ordinal"]:
        cols = [c for c in fl[ftype] if c in col_names]
        for col in sorted(cols):
            idx = col_names.index(col)
            vals = X_all[:, idx]
            rows.append({
                "feature": col,
                "type": ftype,
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6),
                "mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std()), 6),
                "unique": int(len(np.unique(vals))),
                "pct_zero": round(float((vals == 0).mean() * 100), 2),
            })
    miss_cols = [c for c in col_names if c.startswith("missing_")]
    for col in sorted(miss_cols):
        idx = col_names.index(col)
        vals = X_all[:, idx]
        rows.append({
            "feature": col,
            "type": "missing_indicator",
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "unique": int(len(np.unique(vals))),
            "pct_zero": round(float((vals == 0).mean() * 100), 2),
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(run_dir, "data", "feature_stats.csv"), index=False)

    # ── Preprocessing text report ──
    report_lines = [
        "PREPROCESSING REPORT",
        "=" * 80,
        "",
        f"Dataset:     {dm.data_path}",
        f"Raw shape:   {X_raw.shape}",
        f"Processed:   {X_processed.shape}",
        f"Train:       {len(dm.X_train)} samples",
        f"Val:         {len(dm.X_val)} samples",
        f"",
        f"Dropped features (>{prep.missing_threshold*100:.0f}% missing):",
    ]
    for feat in prep.dropped_features:
        report_lines.append(f"  - {feat}")
    report_lines += [
        "",
        f"Feature type breakdown:",
        f"  Categorical:       {len([c for c in fl['categorical'] if c in col_names]):>4d}",
        f"  Binary:            {len([c for c in fl['binary'] if c in col_names]):>4d}",
        f"  Continuous:        {len([c for c in fl['continuous'] if c in col_names]):>4d}",
        f"  Ordinal:           {len([c for c in fl['ordinal'] if c in col_names]):>4d}",
        f"  Missing indicators:{len(miss_cols):>4d}",
        f"  TOTAL:             {dm.num_features:>4d}",
        "",
        f"Value ranges:",
        f"  Global min:  {X_all.min():.6f}",
        f"  Global max:  {X_all.max():.6f}",
        f"  NaN count:   {int(np.isnan(X_all).sum())}",
        f"  All in [-1, 1]: {X_all.min() >= -1.0 - 1e-6 and X_all.max() <= 1.0 + 1e-6}",
        "",
        "Target distribution:",
    ]
    for cls, count, pct in zip(target_df["class"], target_df["count"], target_df["percentage"]):
        bar = "#" * int(pct)
        report_lines.append(f"  Class {cls}: {count:>6d} ({pct:>5.1f}%) {bar}")

    report_lines += [
        "",
        "Per-feature statistics (see feature_stats.csv for full data):",
        f"{'Feature':<35s} {'Type':<16s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s} {'Unique':>8s}",
        "-" * 100,
    ]
    for r in rows:
        report_lines.append(
            f"{r['feature']:<35s} {r['type']:<16s} {r['min']:>8.4f} {r['max']:>8.4f} "
            f"{r['mean']:>8.4f} {r['std']:>8.4f} {r['unique']:>8d}"
        )

    Path(run_dir, "data", "preprocessing_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # ── Console + train.log summary ──
    log.info("=" * 70)
    log.info("PREPROCESSING REPORT")
    log.info("=" * 70)
    log.info(f"Raw: {X_raw.shape} -> Processed: {X_processed.shape}")
    log.info(f"Train: {len(dm.X_train)} | Val: {len(dm.X_val)} | Features: {dm.num_features}")
    log.info(f"Dropped: {prep.dropped_features}")
    log.info("")
    log.info(f"{'Type':<18s} {'Count':>6s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
    log.info("-" * 58)

    type_stats = {}
    for ftype in ["categorical", "binary", "continuous", "ordinal"]:
        cols = [c for c in fl[ftype] if c in col_names]
        idxs = [col_names.index(c) for c in cols]
        if idxs:
            subset = X_all[:, idxs]
            s = {"count": len(cols), "min": float(subset.min()), "max": float(subset.max()),
                 "mean": float(subset.mean()), "std": float(subset.std())}
            type_stats[ftype] = s
            log.info(f"{ftype:<18s} {s['count']:>6d} {s['min']:>8.4f} {s['max']:>8.4f} "
                     f"{s['mean']:>8.4f} {s['std']:>8.4f}")

    if miss_cols:
        subset = X_all[:, [col_names.index(c) for c in miss_cols]]
        log.info(f"{'missing_indicator':<18s} {len(miss_cols):>6d} {subset.min():>8.4f} {subset.max():>8.4f} "
                 f"{subset.mean():>8.4f} {subset.std():>8.4f}")

    log.info(f"{'TOTAL':<18s} {dm.num_features:>6d} {X_all.min():>8.4f} {X_all.max():>8.4f} "
             f"{X_all.mean():>8.4f} {X_all.std():>8.4f}")
    log.info(f"NaN: {int(np.isnan(X_all).sum())} | All in [-1, 1]: True")
    log.info(f"Saved: raw_sample.csv, processed_sample.csv, feature_stats.csv, preprocessing_report.txt")
    log.info("=" * 70)

    # JSONL
    jl.log("preprocessing", "summary",
           raw_shape=list(X_raw.shape), processed_shape=list(X_processed.shape),
           n_train=len(dm.X_train), n_val=len(dm.X_val), n_features=dm.num_features,
           dropped=prep.dropped_features, nan_count=int(np.isnan(X_all).sum()),
           type_stats=type_stats)


def log_model(log: logging.Logger, jl: JSONLLogger, model, cfg, run_dir: str) -> None:
    """Log model architecture: save architecture.txt and parameter_counts.csv."""
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    kan_type = model_cfg.get("kan_type", "mlp")
    widths = list(cfg.model.get("widths", []))
    degree = model_cfg.get("degree", None)
    grid_size = model_cfg.get("grid_size", None)
    num_params = sum(p.numel() for p in model.parameters())

    # ── Parameter counts CSV ──
    layer_rows = []
    for name, param in model.named_parameters():
        layer_rows.append({
            "layer": name,
            "shape": str(list(param.shape)),
            "parameters": param.numel(),
            "trainable": param.requires_grad,
        })
    pd.DataFrame(layer_rows).to_csv(os.path.join(run_dir, "model", "parameter_counts.csv"), index=False)

    # ── Architecture text ──
    arch_lines = [
        "MODEL ARCHITECTURE",
        "=" * 80,
        "",
        f"Model:      {cfg.model.name}",
        f"KAN type:   {kan_type}",
        f"Widths:     {widths}",
        f"Degree:     {degree}",
        f"Grid size:  {grid_size}",
        f"LR:         {model_cfg.get('lr', '-')}",
        f"Weight decay: {model_cfg.get('weight_decay', '-')}",
        "",
        f"Total parameters: {fmt_param_count(num_params)}",
        "",
        "Layer breakdown:",
        f"{'Layer':<50s} {'Shape':<25s} {'Params':>10s}",
        "-" * 90,
    ]
    for r in layer_rows:
        arch_lines.append(f"{r['layer']:<50s} {r['shape']:<25s} {r['parameters']:>10,}")
    arch_lines.append("-" * 90)
    arch_lines.append(f"{'TOTAL':<50s} {'':25s} {num_params:>10,}")

    Path(run_dir, "model", "architecture.txt").write_text("\n".join(arch_lines), encoding="utf-8")

    # Console
    log.info("")
    log.info("=" * 70)
    log.info("MODEL ARCHITECTURE")
    log.info("=" * 70)
    log.info(f"{cfg.model.name} | {kan_type} | widths={widths} | degree={degree} | grid={grid_size}")
    log.info(f"Parameters: {fmt_param_count(num_params)}")
    log.info(f"Saved: architecture.txt, parameter_counts.csv")
    log.info("=" * 70)

    jl.log("model", "architecture", name=cfg.model.name, kan_type=kan_type,
           widths=widths, degree=degree, grid_size=grid_size, total_params=num_params)


def log_forward_pass(log: logging.Logger, jl: JSONLLogger, model, dm) -> None:
    """Log one forward pass before training."""
    if torch is None:
        raise RuntimeError("Torch is required to log forward passes.")
    log.info("")
    log.info("=" * 70)
    log.info("FORWARD PASS (before training, first val batch)")
    log.info("=" * 70)

    batch_x, batch_y = next(iter(dm.val_dataloader()))

    model.eval()
    with torch.no_grad():
        y_hat = model(batch_x)

    n_nan = torch.isnan(y_hat).sum().item()
    n_inf = torch.isinf(y_hat).sum().item()
    log.info(f"Input:  {list(batch_x.shape)}, range=[{batch_x.min():.4f}, {batch_x.max():.4f}]")
    log.info(f"Output: {list(y_hat.shape)}, range=[{y_hat.min():.4f}, {y_hat.max():.4f}], "
             f"mean={y_hat.mean():.4f} | NaN={n_nan} Inf={n_inf}")

    log.info(f"{'':8s} {'Pred':>10s} {'Round':>6s} {'True':>6s} {'Error':>8s}")
    for i in range(min(8, len(batch_x))):
        p, t = y_hat[i].item(), batch_y[i].item()
        log.info(f"  [{i}]    {p:>10.4f} {int(np.clip(np.round(p), 1, 8)):>6d} {int(t):>6d} {abs(p-t):>8.4f}")

    jl.log("forward_pass", "pre_training",
           input_shape=list(batch_x.shape),
           output_range=[float(y_hat.min()), float(y_hat.max())],
           n_nan=n_nan, n_inf=n_inf)
    log.info("=" * 70)


def log_epoch(log: logging.Logger, jl: JSONLLogger, epoch_csv: EpochMetricsCSV,
              epoch: int, train_loss: float, val_loss: float, val_qwk: float) -> None:
    """Log one epoch (human + machine + CSV)."""
    log.info(f"  Epoch {epoch:>3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_qwk={val_qwk:.4f}")
    epoch_csv.log(epoch, train_loss, val_loss, val_qwk)
    jl.log("training", "epoch", epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_qwk=val_qwk)


def log_output(log: logging.Logger, jl: JSONLLogger, preds: np.ndarray, targets: np.ndarray,
               thresholds: np.ndarray, final_qwk: float, run_dir: str) -> None:
    """Log output: thresholds, save predictions CSV, output report."""
    from src.metrics.qwk import _apply_thresholds, quadratic_weighted_kappa

    ordinal = np.clip(_apply_thresholds(preds, thresholds), 1, 8).astype(int)
    naive = np.clip(np.round(preds), 1, 8).astype(int)
    naive_qwk = quadratic_weighted_kappa(targets.astype(int), naive)

    # ── Predictions sample CSV ──
    n_sample = min(200, len(preds))
    pred_df = pd.DataFrame({
        "index": range(n_sample),
        "true_class": targets[:n_sample].astype(int),
        "continuous_pred": np.round(preds[:n_sample], 4),
        "naive_rounded": naive[:n_sample],
        "optimized_class": ordinal[:n_sample],
        "error": np.round(np.abs(preds[:n_sample] - targets[:n_sample]), 4),
    })
    pred_df.to_csv(os.path.join(run_dir, "training", "predictions_sample.csv"), index=False)

    # ── Output report ──
    opt_dist = {int(k): int(v) for k, v in zip(*np.unique(ordinal, return_counts=True))}
    naive_dist = {int(k): int(v) for k, v in zip(*np.unique(naive, return_counts=True))}
    target_dist = {int(k): int(v) for k, v in zip(*np.unique(targets.astype(int), return_counts=True))}

    report = [
        "OUTPUT & THRESHOLD OPTIMIZATION REPORT",
        "=" * 80,
        "",
        f"Total predictions: {len(preds)}",
        f"Continuous output range: [{preds.min():.4f}, {preds.max():.4f}]",
        f"Continuous output mean:  {preds.mean():.4f}",
        f"Continuous output std:   {preds.std():.4f}",
        "",
        "NAIVE ROUNDING (round + clip to [1, 8]):",
        f"  QWK: {naive_qwk:.4f}",
        f"  Distribution: {naive_dist}",
        "",
        "OPTIMIZED THRESHOLDS (Nelder-Mead):",
        f"  QWK: {final_qwk:.4f} (improvement: {final_qwk - naive_qwk:+.4f})",
        f"  Thresholds: {np.round(thresholds, 4).tolist()}",
        f"  Distribution: {opt_dist}",
        "",
        "Threshold boundaries:",
    ]
    for i, t in enumerate(thresholds):
        report.append(f"  pred < {t:.4f}  ->  class {i + 1}")
    report.append(f"  pred >= {thresholds[-1]:.4f} ->  class 8")

    report += [
        "",
        "TRUE TARGET DISTRIBUTION:",
    ]
    for cls in sorted(target_dist.keys()):
        count = target_dist[cls]
        pct = count / len(targets) * 100
        bar = "#" * int(pct)
        report.append(f"  Class {cls}: {count:>6d} ({pct:>5.1f}%) {bar}")

    report += [
        "",
        "COMPARISON (true vs predicted):",
        f"  {'Class':<8s} {'True':>8s} {'Naive':>8s} {'Optimized':>10s}",
        "  " + "-" * 40,
    ]
    for cls in range(1, 9):
        report.append(f"  {cls:<8d} {target_dist.get(cls, 0):>8d} {naive_dist.get(cls, 0):>8d} "
                      f"{opt_dist.get(cls, 0):>10d}")

    Path(run_dir, "training", "output_report.txt").write_text("\n".join(report), encoding="utf-8")

    # Console
    log.info("")
    log.info("=" * 70)
    log.info("OUTPUT — THRESHOLD OPTIMIZATION")
    log.info("=" * 70)
    log.info(f"Predictions: n={len(preds)}, range=[{preds.min():.4f}, {preds.max():.4f}]")
    log.info(f"Naive QWK:     {naive_qwk:.4f}")
    log.info(f"Optimized QWK: {final_qwk:.4f} ({final_qwk - naive_qwk:+.4f})")
    log.info(f"Thresholds: {np.round(thresholds, 4)}")
    log.info(f"Saved: predictions_sample.csv, output_report.txt")
    log.info("=" * 70)

    jl.log("output", "threshold_optimization",
           n=len(preds), pred_range=[float(preds.min()), float(preds.max())],
           naive_qwk=float(naive_qwk), optimized_qwk=float(final_qwk),
           thresholds=[float(t) for t in thresholds],
           naive_dist=naive_dist, optimized_dist=opt_dist, target_dist=target_dist)


def log_training_complete(log: logging.Logger, jl: JSONLLogger,
                          model_name: str, epochs: int, duration: float,
                          final_qwk: float, num_params: int, run_dir: str) -> None:
    """Log final summary."""
    log.info("")
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"Model:  {model_name}")
    log.info(f"Epochs: {epochs} | Time: {duration:.1f}s | Params: {fmt_param_count(num_params)}")
    log.info(f"Final QWK: {final_qwk:.4f}")
    log.info(f"Run dir: {run_dir}")
    log.info("=" * 70)

    jl.log("training", "complete",
           model=model_name, epochs=epochs, duration_s=round(duration, 1),
           total_params=num_params, final_qwk=float(final_qwk), run_dir=run_dir)
