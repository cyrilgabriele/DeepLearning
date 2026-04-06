"""Pydantic models for Optuna study and search-space configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchParamConfig(BaseModel):
    """Single hyperparameter search-space entry."""

    model_config = ConfigDict(extra="allow")

    type: Literal["log_uniform", "uniform", "int", "categorical"] = Field(...)
    low: float | None = None
    high: float | None = None
    step: int | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "SearchParamConfig":
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("Categorical search parameters require a non-empty `choices` list.")
            return self

        if self.low is None or self.high is None:
            raise ValueError(
                f"Search parameter type '{self.type}' requires both `low` and `high`."
            )
        if self.high < self.low:
            raise ValueError("Search parameter `high` must be greater than or equal to `low`.")
        if self.type == "log_uniform" and self.low <= 0:
            raise ValueError("Log-uniform search parameters require `low > 0`.")
        if self.step is not None and self.step <= 0:
            raise ValueError("Integer search parameter `step` must be positive.")
        return self


class TuneConfig(BaseModel):
    """Validated configuration for Optuna tuning."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str | None = Field(
        default=None,
        description="Optional Optuna study name. Defaults to '<experiment>-<family>-tune'.",
    )
    storage: Path | None = Field(
        default=None,
        description="Optional SQLite database path for study persistence.",
    )
    n_trials: int = Field(default=50, ge=1, description="Number of Optuna trials to run.")
    timeout: int | None = Field(default=None, ge=1, description="Study timeout in seconds.")
    sampler: Literal["tpe", "random"] = Field(
        default="tpe",
        description="Optuna sampler to use for the study.",
    )
    search_space: dict[str, SearchParamConfig] = Field(
        ...,
        description="Hyperparameter search space definitions.",
    )

    @field_validator("storage")
    @classmethod
    def _expand_storage_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser().resolve()
