"""Canonical output layout for eval artifacts and interpretability results.

    outputs/
    ├── eval/
    │   └── <recipe>/<experiment>/
    │       ├── X_eval.parquet
    │       ├── y_eval.parquet
    │       ├── X_eval_raw.parquet
    │       ├── feature_names.json
    │       └── feature_types.json
    └── interpretability/
        └── <recipe>/<experiment>/
            ├── figures/   PNG + PDF visualisations
            ├── data/      CSV + Parquet tables
            ├── reports/   JSON summaries + Markdown reports
            └── models/    Pruned .pt weight files
"""

from __future__ import annotations

from pathlib import Path


def eval_run_dir(
    output_root: Path,
    recipe: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> Path:
    """Return the recipe-scoped evaluation artifact directory."""

    p = output_root / "eval" / recipe / experiment_name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def interpret_run_dir(
    output_root: Path,
    recipe: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> Path:
    """Return the recipe-scoped interpretability output directory."""

    p = output_root / "interpretability" / recipe / experiment_name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def figures(output_dir: Path) -> Path:
    p = output_dir / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data(output_dir: Path) -> Path:
    p = output_dir / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports(output_dir: Path) -> Path:
    p = output_dir / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models(output_dir: Path) -> Path:
    p = output_dir / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
