"""Pydantic model that mirrors experiment-time model choices."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class ModelConfig(BaseModel):
    """Encodes TabKAN presets and architecture overrides."""

    name: str = Field(..., description="Key registered inside src.models.registry.")
    flavor: Optional[Literal["chebykan", "fourierkan", "bsplinekan"]] = Field(
        None, description="TabKAN variant sourced from feature/tabkan-models."
    )
    depth: PositiveInt | None = Field(
        default=None,
        description="Number of hidden layers when using the legacy uniform-width shorthand.",
    )
    width: PositiveInt | None = Field(
        default=None,
        description="Uniform hidden width when using the legacy `depth` + `width` shorthand.",
    )
    hidden_widths: tuple[PositiveInt, ...] | None = Field(
        default=None,
        description="Canonical hidden-layer widths for KAN architectures.",
    )
    degree: Optional[PositiveInt] = Field(None, description="Polynomial degree for ChebyKAN flavors.")
    params: Dict[str, Any] = Field(..., description="Free-form JSON payload forwarded to the registry.")

    @field_validator("params")
    @classmethod
    def _ensure_dict(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return dict(value or {})

    @model_validator(mode="before")
    @classmethod
    def _normalize_architecture(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        raw_hidden_widths = payload.get("hidden_widths")
        depth = payload.get("depth")
        width = payload.get("width")

        if raw_hidden_widths is not None:
            hidden_widths = [int(item) for item in raw_hidden_widths]
            if not hidden_widths:
                raise ValueError("`hidden_widths` must contain at least one hidden layer width.")
            if depth is not None and int(depth) != len(hidden_widths):
                raise ValueError("`depth` must match the number of entries in `hidden_widths`.")
            if width is not None and any(int(width) != item for item in hidden_widths):
                raise ValueError(
                    "`width` can only be provided alongside `hidden_widths` when the architecture "
                    "is uniform."
                )
            payload["hidden_widths"] = hidden_widths
            payload["depth"] = len(hidden_widths)
            if all(item == hidden_widths[0] for item in hidden_widths):
                payload["width"] = hidden_widths[0]
            else:
                payload["width"] = None
            return payload

        if depth is not None and width is not None:
            payload["hidden_widths"] = [int(width)] * int(depth)
        return payload

    @model_validator(mode="after")
    def _validate_model_architecture(self) -> "ModelConfig":
        if self.name.startswith("tabkan") and not self.resolved_hidden_widths():
            raise ValueError(
                "TabKAN models require either `hidden_widths` or both `depth` and `width`."
            )
        return self

    def resolved_hidden_widths(self) -> list[int]:
        """Return the canonical hidden-width list for architecture-aware consumers."""

        if self.hidden_widths is not None:
            return [int(width) for width in self.hidden_widths]
        if self.depth is not None and self.width is not None:
            return [int(self.width)] * int(self.depth)
        return []

    def architecture_payload(self) -> Dict[str, Any]:
        """Return a serializable view of the effective hidden-layer layout."""

        widths = self.resolved_hidden_widths()
        return {
            "depth": len(widths),
            "width": widths[0] if widths and all(width == widths[0] for width in widths) else None,
            "hidden_widths": widths,
            "is_uniform": bool(widths) and all(width == widths[0] for width in widths),
        }

    def registry_kwargs(self) -> Dict[str, Any]:
        """Return a copy of params plus the architecture overrides."""

        kwargs = dict(self.params)
        widths = self.resolved_hidden_widths()
        if widths:
            kwargs.setdefault("hidden_widths", widths)
        if self.depth is not None:
            kwargs.setdefault("depth", self.depth)
        if self.width is not None:
            kwargs.setdefault("width", self.width)
        if self.flavor is not None:
            kwargs.setdefault("flavor", self.flavor)
        if self.degree is not None:
            kwargs.setdefault("degree", self.degree)
        return kwargs

    model_config = ConfigDict(extra="forbid", frozen=True)
