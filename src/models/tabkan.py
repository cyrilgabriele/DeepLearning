"""Placeholder TabKAN-style models with configurable architecture knobs.

These wrappers stand in for the real TabKAN implementations while we wire up the
training pipeline. They currently delegate to scikit-learn estimators so the
trainer can run end-to-end without the heavy dependencies a full KAN would
require. The builder already understands the TabKAN flavors from the
`feature/tabkan-models` branch (ChebyKAN, FourierKAN, BSplineKAN) so we can pass
depth/width/degree overrides from the CLI today and plug the actual
implementations in later without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from .base import PrudentialModel


class TabKANFlavor(str, Enum):
    """Supported TabKAN variants aligned with the feature/tabkan-models branch."""

    CHEBYKAN = "chebykan"
    FOURIERKAN = "fourierkan"
    BSPLINEKAN = "bsplinekan"


@dataclass(frozen=True)
class TabKANConfig:
    """Lightweight container describing the placeholder architecture."""

    hidden_layers: Tuple[int, ...]
    flavor: TabKANFlavor = TabKANFlavor.CHEBYKAN
    degree: int | None = 3
    learning_rate_init: float = 1e-3
    max_iter: int = 300


class TabKANClassifier(PrudentialModel):
    """Simple wrapper mimicking a TabKAN classifier via MLPClassifier."""

    def __init__(
        self,
        config: TabKANConfig,
        *,
        random_state: int = 42,
        **extra_params,
    ) -> None:
        params = {
            "hidden_layers": config.hidden_layers,
            "flavor": config.flavor.value,
            "degree": config.degree,
            "learning_rate_init": config.learning_rate_init,
            "max_iter": config.max_iter,
            "random_state": random_state,
        }
        params.update(extra_params)
        super().__init__(**params)

        self.flavor = config.flavor
        self.degree = config.degree
        self.estimator = MLPClassifier(
            hidden_layer_sizes=config.hidden_layers,
            learning_rate_init=config.learning_rate_init,
            max_iter=config.max_iter,
            random_state=random_state,
            verbose=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)


def build_tabkan_model(
    preset: str,
    *,
    random_state: int,
    flavor: str | TabKANFlavor | None = None,
    depth: int | None = None,
    width: int | None = None,
    degree: int | None = None,
    **extra_params,
) -> TabKANClassifier:
    """Instantiate the placeholder TabKAN with optional architecture overrides."""

    presets = {
        "tabkan-tiny": TabKANConfig(hidden_layers=(32, 16), learning_rate_init=5e-3, max_iter=200),
        "tabkan-small": TabKANConfig(hidden_layers=(64, 32), learning_rate_init=3e-3, max_iter=250),
        "tabkan-base": TabKANConfig(hidden_layers=(128, 64), learning_rate_init=1e-3, max_iter=300),
    }

    try:
        base_config = presets[preset]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown TabKAN preset '{preset}'.") from exc

    resolved_flavor = TabKANFlavor(flavor) if flavor else base_config.flavor
    resolved_degree = _resolve_degree(resolved_flavor, base_config.degree, degree)
    hidden_layers = _resolve_hidden_layers(base_config.hidden_layers, depth, width)

    config = TabKANConfig(
        hidden_layers=hidden_layers,
        flavor=resolved_flavor,
        degree=resolved_degree,
        learning_rate_init=base_config.learning_rate_init,
        max_iter=base_config.max_iter,
    )
    return TabKANClassifier(
        config=config,
        random_state=random_state,
        **extra_params,
    )


def _resolve_hidden_layers(
    base_layers: Tuple[int, ...],
    depth: int | None,
    width: int | None,
) -> Tuple[int, ...]:
    if depth is None and width is None:
        return base_layers

    if depth is None:
        depth = len(base_layers) or 1
    if width is None:
        width = base_layers[0] if base_layers else 32

    if depth <= 0:
        raise ValueError("depth must be a positive integer")
    if width <= 0:
        raise ValueError("width must be a positive integer")

    return tuple([width] * depth)


def _resolve_degree(
    flavor: TabKANFlavor,
    base_degree: int | None,
    requested: int | None,
) -> int | None:
    if flavor != TabKANFlavor.CHEBYKAN:
        return None
    if requested is not None:
        if requested <= 0:
            raise ValueError("degree must be positive")
        return requested
    return base_degree
