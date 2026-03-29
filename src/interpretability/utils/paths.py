"""Canonical output sub-directory layout for the interpretability pipeline.

    outputs/
    ├── figures/   PNG + PDF visualisations
    ├── data/      CSV + Parquet tables
    ├── reports/   JSON summaries + Markdown reports
    └── models/    Pruned .pt weight files
"""

from __future__ import annotations

from pathlib import Path


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
