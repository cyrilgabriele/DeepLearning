"""Pydantic model covering preprocessing-specific settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PreprocessingConfig(BaseModel):
    """Encapsulates which paper-aligned pipeline to execute."""

    recipe: Literal[
        "xgboost_paper",
        "kan_paper",
        "kan_sota",
    ] = Field(
        ...,
        description="Choose between the paper baseline, the minimal KAN tweaks, or the SOTA KAN recipe.",
    )

    model_config = ConfigDict(frozen=True)
