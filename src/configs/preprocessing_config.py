"""Pydantic model covering preprocessing-specific settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class PreprocessingConfig(BaseModel):
    """Encapsulates how raw Prudential data is cleaned and split."""

    recipe: Literal["paper", "kan"] = Field(..., description="Choose between the paper baseline and KAN pipeline.")
    missing_threshold: float = Field(..., description="Columns missing over this ratio are dropped before processing.")
    stratify: bool = Field(..., description="Whether to use stratified splits when carving eval data.")
    use_stratified_kfold: bool = Field(
        ..., description="When true, categorical encodings rely on stratified k-fold logic."
    )
    kan_n_splits: PositiveInt = Field(
        ..., description="Number of folds used for the stratified categorical encoder.")

    @field_validator("missing_threshold")
    @classmethod
    def _validate_missing_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("missing_threshold must be within [0, 1].")
        return value

    @field_validator("kan_n_splits")
    @classmethod
    def _minimum_kfolds(cls, value: PositiveInt) -> PositiveInt:
        if value < 2:
            raise ValueError("kan_n_splits must be at least 2 to run k-fold encoding.")
        return value

    model_config = ConfigDict(frozen=True)
