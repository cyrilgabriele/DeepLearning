"""Central registry for experiment-ready models."""

from __future__ import annotations

from typing import Callable, Dict

from .base import PrudentialModel
from .tabkan import build_tabkan_model
from .xgboost_paper import build_xgboost_paper_model


ModelFactory = Callable[..., PrudentialModel]


MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "tabkan-tiny": build_tabkan_model,
    "tabkan-small": build_tabkan_model,
    "tabkan-base": build_tabkan_model,
    "xgboost-paper": build_xgboost_paper_model,
}


def create_model(model_name: str, *, random_state: int, **model_params) -> PrudentialModel:
    """Instantiate a model from the registry."""

    try:
        factory = MODEL_REGISTRY[model_name]
    except KeyError as exc:  # pragma: no cover - defensive
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}") from exc

    if model_name.startswith("tabkan"):
        return factory(model_name, random_state=random_state, **model_params)

    return factory(random_state=random_state, **model_params)


def available_models() -> Dict[str, ModelFactory]:
    return dict(MODEL_REGISTRY)
