"""Paper-faithful coefficient utilities for ChebyKAN and FourierKAN.

TabKAN (arXiv:2504.06559v3) derives feature importance from the magnitude of
the learned first-layer basis coefficients. This module centralizes that logic
so every interpretability script uses the same scoring and feature-function
reconstruction.
"""

from __future__ import annotations

import math

import pandas as pd


_KAN_LAYER_TYPES: tuple | None = None


def _get_kan_layer_types() -> tuple:
    global _KAN_LAYER_TYPES
    if _KAN_LAYER_TYPES is None:
        from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
        _KAN_LAYER_TYPES = (ChebyKANLayer, FourierKANLayer)
    return _KAN_LAYER_TYPES


def _is_kan_layer(layer) -> bool:
    return isinstance(layer, _get_kan_layer_types())


def get_first_kan_layer(module):
    """Return the first learnable KAN layer or ``None`` when unavailable."""
    return next((l for l in module.kan_layers if _is_kan_layer(l)), None)


def get_all_kan_layers(module) -> list:
    """Return all learnable KAN layers in order."""
    return [l for l in module.kan_layers if _is_kan_layer(l)]


def coefficient_importance_from_layer(
    layer,
    feature_names: list[str],
    *,
    include_linear_term: bool = False,
) -> pd.DataFrame:
    """Return layer-0 paper-native coefficient importance per input feature.

    The paper text describes feature ranking via the absolute magnitude of the
    learned Chebyshev/Fourier coefficients (Section 5.7, equations 19/20).
    We therefore aggregate ``sum(abs(coefficients))`` over all hidden outputs
    and basis terms for each input feature.

    ``base_weight`` is exported separately. It is excluded from the default
    paper-native score because the TabKAN equations only describe the basis
    coefficients, not the residual linear path used in this codebase.
    """
    import torch

    if layer is None:
        return pd.DataFrame(columns=[
            "feature",
            "importance",
            "basis_abs_sum",
            "basis_signed_sum_abs",
            "linear_abs_sum",
            "importance_with_linear",
            "n_hidden_outputs",
            "n_basis_terms",
            "layer",
            "metric",
        ])

    n_features = min(len(feature_names), int(layer.in_features))
    linear_abs = layer.base_weight.detach().abs().sum(dim=0)

    if hasattr(layer, "cheby_coeffs"):
        coeffs = layer.cheby_coeffs.detach()
        basis_abs = coeffs.abs().sum(dim=(0, 2))
        basis_signed_abs = coeffs.sum(dim=(0, 2)).abs()
        n_basis_terms = int(coeffs.shape[-1])
        metric = "chebyshev_abs_sum"
    elif hasattr(layer, "fourier_a") and hasattr(layer, "fourier_b"):
        coeff_a = layer.fourier_a.detach()
        coeff_b = layer.fourier_b.detach()
        basis_abs = coeff_a.abs().sum(dim=(0, 2)) + coeff_b.abs().sum(dim=(0, 2))
        basis_signed_abs = (coeff_a.sum(dim=(0, 2)) + coeff_b.sum(dim=(0, 2))).abs()
        n_basis_terms = int(coeff_a.shape[-1] + coeff_b.shape[-1])
        metric = "fourier_abs_sum"
    else:  # pragma: no cover - defensive branch
        raise TypeError(f"Unsupported layer type: {type(layer)}")

    score = basis_abs + (linear_abs if include_linear_term else torch.zeros_like(linear_abs))
    frame = pd.DataFrame(
        {
            "feature": feature_names[:n_features],
            "importance": [float(score[i]) for i in range(n_features)],
            "basis_abs_sum": [float(basis_abs[i]) for i in range(n_features)],
            "basis_signed_sum_abs": [float(basis_signed_abs[i]) for i in range(n_features)],
            "linear_abs_sum": [float(linear_abs[i]) for i in range(n_features)],
            "importance_with_linear": [float((basis_abs + linear_abs)[i]) for i in range(n_features)],
            "n_hidden_outputs": int(layer.out_features),
            "n_basis_terms": n_basis_terms,
            "layer": 0,
            "metric": metric,
        }
    )
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def coefficient_importance_all_layers(
    module,
    feature_names: list[str],
    *,
    include_linear_term: bool = False,
) -> pd.DataFrame:
    """Return paper-native coefficient importance for every KAN layer.

    Layer 0 uses the supplied feature_names as input labels.
    Subsequent layers label their inputs as h0, h1, ... (hidden nodes
    from the previous KAN layer).
    """
    layers = get_all_kan_layers(module)
    frames = []
    input_labels = list(feature_names)
    for layer_idx, layer in enumerate(layers):
        frame = coefficient_importance_from_layer(
            layer, input_labels, include_linear_term=include_linear_term
        )
        frame = frame.copy()
        frame["layer"] = layer_idx
        frames.append(frame)
        # Next layer's inputs are hidden nodes of this layer
        input_labels = [f"h{i}" for i in range(layer.out_features)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def coefficient_importance_from_module(
    module,
    feature_names: list[str],
    *,
    include_linear_term: bool = False,
) -> pd.Series:
    """Return the paper-native layer-0 coefficient importance as a Series."""
    layer = get_first_kan_layer(module)
    frame = coefficient_importance_from_layer(
        layer,
        feature_names,
        include_linear_term=include_linear_term,
    )
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.set_index("feature")["importance"].sort_values(ascending=False)


def top_features_by_coefficients(
    module,
    feature_names: list[str],
    *,
    top_n: int = 10,
    include_linear_term: bool = False,
) -> list[str]:
    """Return the top features ranked by paper-native coefficient magnitude."""
    ranking = coefficient_importance_from_module(
        module,
        feature_names,
        include_linear_term=include_linear_term,
    )
    return ranking.head(top_n).index.tolist()


def sample_feature_function(
    layer,
    feature_idx: int,
    *,
    n: int = 1000,
    reduction: str = "mean",
):
    """Sample the aggregated first-layer feature function across hidden outputs.

    The TabKAN notebooks reconstruct feature-level explanations directly from
    first-layer coefficients. We follow that pattern by aggregating all outgoing
    hidden-edge functions for one input feature. ``reduction="mean"`` mirrors
    the reference Fourier notebook's averaging across outputs.
    """
    import torch

    x = torch.linspace(-3.0, 3.0, n, dtype=torch.float32)
    x_norm = torch.tanh(x)

    if hasattr(layer, "cheby_coeffs"):
        coeffs = layer.cheby_coeffs[:, feature_idx, :].detach()
        basis_terms = [torch.ones_like(x_norm), x_norm]
        for _ in range(2, layer.degree + 1):
            basis_terms.append(2 * x_norm * basis_terms[-1] - basis_terms[-2])
        basis = torch.stack(basis_terms, dim=-1)
        y_all = torch.einsum("sd,od->so", basis, coeffs)
    elif hasattr(layer, "fourier_a") and hasattr(layer, "fourier_b"):
        k = torch.arange(1, layer.grid_size + 1, dtype=torch.float32)
        x_scaled = (x_norm + 1.0) * math.pi
        x_k = x_scaled.unsqueeze(-1) * k
        cos_basis = torch.cos(x_k)
        sin_basis = torch.sin(x_k)
        y_all = torch.einsum("sg,og->so", cos_basis, layer.fourier_a[:, feature_idx, :].detach())
        y_all = y_all + torch.einsum("sg,og->so", sin_basis, layer.fourier_b[:, feature_idx, :].detach())
    else:  # pragma: no cover - defensive branch
        raise TypeError(f"Unsupported layer type: {type(layer)}")

    # Keep the actual learned linear residual path in the plotted attribution.
    base = layer.base_weight[:, feature_idx].detach().unsqueeze(0) * x_norm.unsqueeze(-1)
    y_all = y_all + base

    if reduction == "sum":
        y_feature = y_all.sum(dim=-1)
    elif reduction == "mean":
        y_feature = y_all.mean(dim=-1)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported reduction: {reduction}")

    return x_norm.numpy(), y_feature.numpy(), y_all.numpy()
