"""Pydantic model that mirrors experiment-time model choices."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


_TABKAN_REQUIRED_TRAINING_PARAMS = {
    "max_epochs",
    "lr",
    "weight_decay",
    "batch_size",
    "sparsity_lambda",
    "l1_weight",
    "entropy_weight",
}
_XGBOOST_REQUIRED_PARAMS = {
    "n_estimators",
    "max_depth",
    "min_child_weight",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "num_classes",
    "tree_method",
    "eval_metric",
    "refit_full_training",
}
_GLM_REQUIRED_PARAMS = {"alpha"}


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
    use_layernorm: bool = Field(
        default=True,
        description="Whether TabKAN inserts LayerNorm after each hidden KAN layer.",
    )
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
        if self.name.startswith("tabkan") and self.flavor is None:
            raise ValueError("TabKAN models require `flavor`.")
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
            "use_layernorm": self.use_layernorm,
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
        if self.name.startswith("tabkan"):
            kwargs.setdefault("use_layernorm", self.use_layernorm)
        return kwargs

    def allowed_param_keys(self) -> set[str]:
        """Return the supported `model.params` keys for the configured model."""

        if self.name.startswith("tabkan"):
            allowed = set(_TABKAN_REQUIRED_TRAINING_PARAMS)
            if self.flavor in {"fourierkan", "bsplinekan"}:
                allowed.add("grid_size")
            if self.flavor == "bsplinekan":
                allowed.add("spline_order")
            return allowed
        if self.name == "xgboost-paper":
            return set(_XGBOOST_REQUIRED_PARAMS)
        if self.name == "glm":
            return set(_GLM_REQUIRED_PARAMS)
        return set()

    def allowed_tune_keys(self) -> set[str]:
        """Return supported `tune.search_space` keys for this model."""

        if self.name.startswith("tabkan"):
            allowed = self.allowed_param_keys() | {"depth", "width"}
            if self.flavor == "chebykan":
                allowed.add("degree")
            return allowed
        return self.allowed_param_keys()

    def validate_registry_contract(self, *, tune_param_names: set[str] | None = None) -> None:
        """Fail fast on unsupported or missing config keys before model creation."""

        tune_param_names = set(tune_param_names or set())
        allowed_params = self.allowed_param_keys()
        unknown_params = sorted(set(self.params) - allowed_params)
        if unknown_params:
            raise ValueError(
                f"Model '{self.name}' does not support `model.params` keys: {', '.join(unknown_params)}."
            )

        unknown_tune_keys = sorted(tune_param_names - self.allowed_tune_keys())
        if unknown_tune_keys:
            raise ValueError(
                f"Model '{self.name}' does not support `tune.search_space` keys: "
                f"{', '.join(unknown_tune_keys)}."
            )

        missing_params = sorted(allowed_params & _TABKAN_REQUIRED_TRAINING_PARAMS - set(self.params) - tune_param_names)
        if self.name.startswith("tabkan"):
            if missing_params:
                source_hint = "Provide them in `model.params` or `tune.search_space`."
                if not tune_param_names:
                    source_hint = "Provide them in `model.params`."
                raise ValueError(
                    f"TabKAN config is missing required training parameter(s): {', '.join(missing_params)}. "
                    f"{source_hint}"
                )
            if self.flavor == "chebykan" and self.degree is None and "degree" not in tune_param_names:
                raise ValueError(
                    "ChebyKAN configs require `model.degree` or a `tune.search_space.degree` entry."
                )
            if self.flavor in {"fourierkan", "bsplinekan"} and (
                "grid_size" not in self.params and "grid_size" not in tune_param_names
            ):
                raise ValueError(
                    f"{self.flavor} configs require `model.params.grid_size` or a "
                    "`tune.search_space.grid_size` entry."
                )
            if self.flavor == "bsplinekan" and (
                "spline_order" not in self.params and "spline_order" not in tune_param_names
            ):
                raise ValueError(
                    "bsplinekan configs require `model.params.spline_order` or a "
                    "`tune.search_space.spline_order` entry."
                )
            return

        required_params = set()
        if self.name == "xgboost-paper":
            required_params = set(_XGBOOST_REQUIRED_PARAMS)
        elif self.name == "glm":
            required_params = set(_GLM_REQUIRED_PARAMS)

        missing_required = sorted(required_params - set(self.params) - tune_param_names)
        if missing_required:
            source_hint = "Provide them in `model.params` or `tune.search_space`."
            if not tune_param_names:
                source_hint = "Provide them in `model.params`."
            raise ValueError(
                f"Model '{self.name}' is missing required parameter(s): {', '.join(missing_required)}. "
                f"{source_hint}"
            )

    def assert_training_ready(self) -> None:
        """Require concrete model hyperparameters before `--stage train` or retraining."""

        self.validate_registry_contract()

    model_config = ConfigDict(extra="forbid", frozen=True)
