"""Shared style and feature-display helpers for interpretability figures."""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FeatureDisplaySpec:
    """Resolved display semantics for one model-input feature."""

    model_feature: str
    raw_feature: str | None
    feature_type: str
    transform: str
    model_input_kind: str
    preprocessing_recipe: str | None = None


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


def resolve_feature_display_spec(
    feat: str,
    *,
    feat_types: dict | None = None,
    preprocessing_recipe: str | None = None,
) -> FeatureDisplaySpec:
    """Resolve raw-feature mapping and model-input semantics for one feature."""
    feature_type = (feat_types or {}).get(feat, "unknown")

    if feat.startswith("missing_"):
        raw_feature = feat.removeprefix("missing_")
        transform = "missing_indicator"
    elif feat.startswith("cb_"):
        raw_feature = feat.removeprefix("cb_")
        transform = "catboost_encoded"
    elif feat.startswith("qt_"):
        raw_feature = feat.removeprefix("qt_")
        transform = "quantile_scaled"
    elif feat.startswith("mm_"):
        raw_feature = feat.removeprefix("mm_")
        transform = "minmax_scaled"
    else:
        raw_feature = feat
        transform = "identity"

    if transform == "missing_indicator":
        model_input_kind = "discrete"
    elif transform in {"catboost_encoded", "quantile_scaled", "minmax_scaled"}:
        model_input_kind = "continuous"
    elif feature_type in {"binary", "categorical", "missing_indicator"}:
        model_input_kind = "discrete"
    else:
        model_input_kind = "continuous"

    return FeatureDisplaySpec(
        model_feature=feat,
        raw_feature=raw_feature,
        feature_type=feature_type,
        transform=transform,
        model_input_kind=model_input_kind,
        preprocessing_recipe=preprocessing_recipe,
    )


def feature_plot_kind(spec: FeatureDisplaySpec) -> str:
    """Return the most faithful plot family for the model-input semantics."""
    if spec.model_input_kind == "continuous":
        return "continuous"
    if spec.feature_type in {"binary", "missing_indicator"}:
        return "binary"
    return "categorical"


def get_feature_raw_series(
    spec: FeatureDisplaySpec,
    X_raw: pd.DataFrame | None,
) -> pd.Series | None:
    """Return the raw feature column when available."""
    if X_raw is None or spec.raw_feature is None or spec.raw_feature not in X_raw.columns:
        return None
    return X_raw[spec.raw_feature]


def can_display_on_raw_numeric_axis(
    spec: FeatureDisplaySpec,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame | None,
) -> bool:
    """Return True when model-input values can be shown on a numeric raw axis."""
    raw_series = get_feature_raw_series(spec, X_raw)
    if raw_series is None or spec.model_feature not in X_eval.columns:
        return False
    if spec.model_input_kind != "continuous":
        return False
    if spec.transform not in {"identity", "quantile_scaled", "minmax_scaled"}:
        return False
    if not pd.api.types.is_numeric_dtype(raw_series):
        return False

    enc = pd.to_numeric(X_eval[spec.model_feature], errors="coerce")
    raw = pd.to_numeric(raw_series, errors="coerce")
    valid = ~(enc.isna() | raw.isna())
    if valid.sum() < 2:
        return False
    return enc[valid].nunique() > 1


def feature_axis_label(
    spec: FeatureDisplaySpec,
    *,
    use_raw_axis: bool = False,
) -> str:
    """Return an honest x-axis label for the chosen display domain."""
    if spec.model_input_kind == "discrete":
        if spec.transform == "missing_indicator":
            return "Observed state"
        if spec.raw_feature is not None:
            return "Observed category/value"
        return "Observed model-input value"

    if use_raw_axis:
        return "Original scale"

    labels = {
        "identity": (
            "Raw/model-input scale"
            if spec.preprocessing_recipe == "kan_paper"
            else "Model-input scale"
        ),
        "quantile_scaled": "Quantile-scaled model input",
        "minmax_scaled": "Min-max-scaled model input",
        "catboost_encoded": "CatBoost-encoded model input",
        "missing_indicator": "Missing indicator",
    }
    return labels.get(spec.transform, "Model-input scale")


def build_feature_grid(
    spec: FeatureDisplaySpec,
    X_eval: pd.DataFrame,
    *,
    grid_resolution: int = 100,
    percentile_range: tuple[float, float] | None = (1.0, 99.0),
) -> np.ndarray:
    """Build a feature grid on the model-input domain."""
    col = pd.to_numeric(X_eval[spec.model_feature], errors="coerce")
    values = col[np.isfinite(col.to_numpy(dtype=float, copy=False))].to_numpy(dtype=float, copy=False)
    if values.size == 0:
        return np.array([], dtype=float)

    if spec.model_input_kind == "discrete":
        return np.sort(np.unique(values))

    if percentile_range is None:
        lo = float(np.min(values))
        hi = float(np.max(values))
    else:
        lo, hi = np.percentile(values, percentile_range)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([], dtype=float)
    if np.isclose(lo, hi):
        return np.array([lo], dtype=float)
    return np.linspace(lo, hi, grid_resolution, dtype=float)


def display_feature_values(
    spec: FeatureDisplaySpec,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame | None,
    values: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Map model-input values to plotted values when a raw numeric axis is safe."""
    if can_display_on_raw_numeric_axis(spec, X_eval, X_raw):
        return encode_to_raw_lookup(
            spec.model_feature,
            X_eval,
            X_raw,
            x_norm=values,
            raw_feature=spec.raw_feature,
        ), True
    return np.asarray(values, dtype=float), False


def _format_tick_value(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3g}"


def discrete_feature_ticks(
    spec: FeatureDisplaySpec,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame | None,
) -> tuple[np.ndarray, list[str]]:
    """Return tick positions and human-readable labels for discrete features."""
    positions = build_feature_grid(spec, X_eval, percentile_range=None)
    labels: list[str] = []

    if spec.transform == "missing_indicator":
        for value in positions:
            if np.isclose(value, 0.0):
                labels.append("observed")
            elif np.isclose(value, 1.0):
                labels.append("missing")
            else:
                labels.append(_format_tick_value(float(value)))
        return positions, labels

    raw_series = get_feature_raw_series(spec, X_raw)
    enc = pd.to_numeric(X_eval[spec.model_feature], errors="coerce")
    if raw_series is None:
        return positions, [_format_tick_value(float(value)) for value in positions]

    for value in positions:
        mask = np.isclose(enc.to_numpy(dtype=float, copy=False), value, atol=1e-8, rtol=0.0)
        raw_subset = raw_series.loc[mask]
        if raw_subset.empty:
            labels.append(_format_tick_value(float(value)))
            continue
        mode = raw_subset.mode(dropna=True)
        if not mode.empty:
            labels.append(str(mode.iloc[0]))
        else:
            labels.append(str(raw_subset.iloc[0]))
    return positions, labels


def encode_to_raw_lookup(
    feat: str,
    X_eval: pd.DataFrame,
    X_raw: pd.DataFrame,
    x_norm: np.ndarray | None = None,
    *,
    raw_feature: str | None = None,
) -> np.ndarray:
    """Map model-input values back to original scale via sorted lookup.

    Co-sorts (encoded, raw) pairs by encoded value, then uses np.interp
    (monotone interpolation). Out-of-range inputs are clamped to boundary
    raw values by np.interp's default behaviour — no extrapolation.

    Args:
        feat: Feature name present in X_eval.
        X_eval: DataFrame with model-input feature values.
        X_raw: DataFrame with original-scale feature values.
        x_norm: Model-input values to map. If None, maps X_eval[feat] itself.
        raw_feature: Raw column name when it differs from feat.

    Returns:
        Array of original-scale values, same length as x_norm.
    """
    enc = np.asarray(X_eval[feat], dtype=float)
    raw_col = raw_feature or feat
    raw = np.asarray(X_raw[raw_col], dtype=float)
    valid = ~(np.isnan(enc) | np.isnan(raw))
    enc, raw = enc[valid], raw[valid]
    order = np.argsort(enc)
    enc_sorted, raw_sorted = enc[order], raw[order]
    if x_norm is None:
        x_norm = np.asarray(X_eval[feat], dtype=float)
    return np.interp(x_norm, enc_sorted, raw_sorted)
