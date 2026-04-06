"""Pydantic model that mirrors experiment-time model choices."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class ModelConfig(BaseModel):
    """Encodes TabKAN presets and architecture overrides."""

    name: str = Field(..., description="Key registered inside src.models.registry.")
    flavor: Optional[Literal["chebykan", "fourierkan", "bsplinekan"]] = Field(
        None, description="TabKAN variant sourced from feature/tabkan-models."
    )
    depth: PositiveInt = Field(..., description="Number of hidden layers if the model supports it.")
    width: PositiveInt = Field(..., description="Width of each hidden layer if relevant.")
    degree: Optional[PositiveInt] = Field(None, description="Polynomial degree for ChebyKAN flavors.")
    params: Dict[str, Any] = Field(..., description="Free-form JSON payload forwarded to the registry.")

    @field_validator("params")
    @classmethod
    def _ensure_dict(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return dict(value or {})

    def registry_kwargs(self) -> Dict[str, Any]:
        """Return a copy of params plus the architecture overrides."""

        kwargs = dict(self.params)
        kwargs.setdefault("depth", self.depth)
        kwargs.setdefault("width", self.width)
        if self.flavor is not None:
            kwargs.setdefault("flavor", self.flavor)
        if self.degree is not None:
            kwargs.setdefault("degree", self.degree)
        return kwargs

    model_config = ConfigDict(frozen=True)
