"""Shared style configuration for all interpretability figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COLORS: dict[str, str] = {
    "GLM": "#4C72B0",
    "XGBoost": "#DD8452",
    "ChebyKAN": "#55A868",
    "FourierKAN": "#C44E52",
}

FEATURE_TYPE_COLORS: dict[str, str] = {
    "continuous": "#2196F3",
    "binary": "#FF9800",
    "categorical": "#9C27B0",
    "ordinal": "#4CAF50",
    "missing_indicator": "#9E9E9E",
}

FEATURE_TYPE_MARKERS: dict[str, str] = {
    "continuous": "[C]",
    "binary": "[B]",
    "categorical": "[K]",
    "ordinal": "[O]",
    "missing_indicator": "[M]",
}

RISK_CMAP: str = "RdYlGn_r"


def apply_paper_style() -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def savefig_pdf(fig, path: Path) -> None:
    """Save figure as PDF at 300 dpi with tight bounding box."""
    fig.savefig(path, dpi=300, bbox_inches="tight", format="pdf")


def feature_type_label(feat: str, feat_types: dict) -> str:
    """Return feature name with type marker, e.g. 'BMI [C]'."""
    ftype = feat_types.get(feat)
    marker = FEATURE_TYPE_MARKERS.get(ftype, "")
    return f"{feat} {marker}".strip() if marker else feat


def encode_to_raw_lookup(
    feat: str,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame,
    x_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Map encoded [-1,1] values back to original scale via sorted lookup.

    Co-sorts (encoded, raw) pairs by encoded value, then uses np.interp
    (monotone interpolation). Out-of-range inputs are clamped to boundary
    raw values by np.interp's default behaviour — no extrapolation.

    Args:
        feat: Feature name present in both X_eval and X_raw.
        X_eval: DataFrame with encoded feature values.
        X_raw: DataFrame with original-scale feature values.
        x_norm: Encoded values to map. If None, maps X_eval[feat] itself.

    Returns:
        Array of original-scale values, same length as x_norm.
    """
    enc = np.asarray(X_eval[feat], dtype=float)
    raw = np.asarray(X_raw[feat], dtype=float)
    valid = ~(np.isnan(enc) | np.isnan(raw))
    enc, raw = enc[valid], raw[valid]
    order = np.argsort(enc)
    enc_sorted, raw_sorted = enc[order], raw[order]
    if x_norm is None:
        x_norm = np.asarray(X_eval[feat], dtype=float)
    return np.interp(x_norm, enc_sorted, raw_sorted)
