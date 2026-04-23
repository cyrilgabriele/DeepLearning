"""Pydantic model covering preprocessing-specific settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PreprocessingConfig(BaseModel):
    """Encapsulates the frozen preprocessing contract for a run."""

    contract_version: int = Field(
        default=1,
        ge=1,
        description="Version of the preprocessing contract recorded in run artifacts.",
    )

    recipe: Literal[
        "xgboost_paper",
        "kan_paper",
        "kan_sota",
    ] = Field(
        ...,
        description="Choose between the paper baseline, the minimal KAN tweaks, or the SOTA KAN recipe.",
    )
    selected_features_path: Path | None = Field(
        default=None,
        description=(
            "Optional JSON or text file listing the exact preprocessed feature names to keep "
            "during train/eval/test transforms."
        ),
    )

    @field_validator("selected_features_path")
    @classmethod
    def _expand_selected_features_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser().resolve()

    def contract_payload(self) -> dict[str, object]:
        """Return the serializable preprocessing payload persisted with run artifacts."""

        return self.model_dump(mode="json")

    model_config = ConfigDict(extra="forbid", frozen=True)
